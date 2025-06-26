### Helpers for sampling from models.
import jax.experimental
import jax.experimental.multihost_utils
from models.qwen3 import Qwen3Model, KVCache, count_left_padding
import jax.numpy as jnp
import numpy as np
import math
import jax
from functools import partial

from utils.sharding import create_sharding

def pad_and_collate(token_batch, pad_id=0, multi_host=False, force_length=None):
    max_len = max([len(x) for x in token_batch])
    if multi_host:
        max_len = max(jax.experimental.multihost_utils.process_allgather(max_len))
    if force_length is not None:
        if max_len > force_length:
            token_batch = [x[:force_length] for x in token_batch]
        max_len = force_length
    return np.array([(max_len - len(x)) * [pad_id] + x for x in token_batch])

# @partial(jax.jit, out_shardings=(None), static_argnames=('model'))
def get_logprobs(model: Qwen3Model, params, token_batch, cache):
    print("JIT compiling update function for token_batch of shape", token_batch.shape)
    text_input, text_target = token_batch[:, :-1], token_batch[:, 1:]
    token_mask = jnp.where(text_input != 0, 1, 0).astype(jnp.int32)
    logits, _ = model.apply({'params': params}, text_input, token_mask, cache=cache)
    logprobs = jax.nn.log_softmax(logits, axis=-1) # [batch, time, vocab_size]
    logprobs = jnp.sum(logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
    return logprobs

model_apply = None # Global variable to cache the JIT-compiled model application function.
def autoregressive_sample(model: Qwen3Model, params, prompt_tokens, num_generation_tokens, rng, pad_id=0, data_shard=None, no_shard=None, force_answer=False):
    """
    Samples tokens autoregressively, and can batch for performance.
    Args:
        prompt_tokens: An array of tokens, padded by `pad_id` on the LEFT. [batch, time].
        max_seq_len: The maximum sequence length to sample.
    """
    global model_apply
    batch_size = prompt_tokens.shape[0]
    # pad_to = 2 ** math.ceil(math.log2((prompt_tokens.shape[-1])))
    # pad_to = 256
    pad_to = 64
    prompt_tokens = jnp.pad(prompt_tokens, [(0, 0), (0, pad_to - prompt_tokens.shape[-1])])
    prompt_tokens = prompt_tokens[:, :4]
    token_mask = jnp.where(prompt_tokens != pad_id, 1, 0).astype(jnp.int32)
    max_seq_len = pad_to + num_generation_tokens

    cache = KVCache.create(model.num_layers, batch_size, max_seq_len, model.head_dim, model.kv_heads)
    cache = cache.replace(starts=count_left_padding(prompt_tokens, pad_id=0))
    cache_sharding = KVCache.get_sharding(data_shard, no_shard)
    cache = jax.jit(lambda x: x, out_shardings=cache_sharding)(cache)

    if model_apply is None:
        # @partial(jax.jit, out_shardings=(data_shard, cache_sharding))
        # @partial(jax.jit, out_shardings=(data_shard, None))
        def model_apply(params, tokens, token_mask, cache):
            print("JIT compiling model_apply for tokens of shape", tokens.shape)
            return model.apply({'params': params}, tokens, token_mask, cache=cache)
        

    # [p1, p2, p3, p4, o1, o2, o3, o4]
    # [p1, p2, p3] prefilled.
    # next input -> from p4, predict

    # Fill cache with the prompt tokens.
    logits, cache = model_apply(params, prompt_tokens[:, :-1], token_mask[:, :-1], cache=cache)
    # logits, cache = model_apply(params, prompt_tokens[:, :-1], token_mask[:, :-1], cache=None)

    sampled_token = prompt_tokens[:, -1]  # Start with the last token of the prompt.

    tokens_list = []
    logprobs_list = []

    # Prompt in the tokens list for debugging.
    logprobs = jax.nn.log_softmax(logits, axis=-1)

    batch_indices = jnp.arange(prompt_tokens.shape[0])[:, None]  # Shape: (8, 1)
    position_indices = jnp.arange(prompt_tokens.shape[1]-1)[None, :]  # Shape: (1, 63)
    token_indices = prompt_tokens[:, 1:]  # Shape: (8, 63)
    prompt_logprobs = logprobs[batch_indices, position_indices, token_indices]
    # tokens_list.append(prompt_tokens[:, 1:])  # Exclude the first token (padding).
    # logprobs_list.append(prompt_logprobs)

    max_samples = max_seq_len - prompt_tokens.shape[-1]
    max_samples = 1
    for i in range(max_samples):
        next_token_mask = jnp.ones(sampled_token.shape, dtype=jnp.int32)
        print(" =============== Doing an autogressive step with cache.")
        print('cache k is', cache.k[27][0, 2])
        logits, cache = model_apply(params, sampled_token[:, None], next_token_mask[:, None], cache=cache)
        print("============================")
        logits = logits[:, 0, :]
        key, rng = jax.random.split(rng)
        # sampled_token = jax.random.categorical(key, logits, axis=-1) # [batch]
        sampled_token = jnp.argmax(logits, axis=-1)  # [batch]

        # Very ugly to support forcing a <answer> tag.
        if force_answer:
            if i == max_samples - 51 or i == max_samples - 52:
                sampled_token = jnp.ones_like(sampled_token) * 198
            if i == max_samples - 50:
                sampled_token = jnp.ones_like(sampled_token) * 27
            if i == max_samples - 49:
                sampled_token = jnp.ones_like(sampled_token) * 9217
            if i == max_samples - 48:
                sampled_token = jnp.ones_like(sampled_token) * 29

        logprobs = jax.nn.log_softmax(logits, axis=-1) # [batch, vocab_size]
        sampled_logprob = logprobs[np.arange(batch_size), sampled_token] # [batch]
        tokens_list.append(sampled_token)
        logprobs_list.append(sampled_logprob)

        del logprobs, logits

    cache_blank = KVCache.create(model.num_layers, batch_size, max_seq_len, model.head_dim, model.kv_heads)
    cache_blank = cache_blank.replace(starts=count_left_padding(prompt_tokens, pad_id=0))
    cache_sharding = KVCache.get_sharding(data_shard, no_shard)
    cache_blank = jax.jit(lambda x: x, out_shardings=cache_sharding)(cache_blank)
    
    tokens = jnp.stack(tokens_list, axis=-1) # [batch, time]
    logprobs = jnp.stack(logprobs_list, axis=-1)
    logprobs = jnp.concatenate([prompt_logprobs, logprobs], axis=-1)  # [batch, time]
    print("=============== Recalculating with a blank cache. ")
    logprobs_recalc = get_logprobs(model, params, jnp.concatenate([prompt_tokens, tokens], axis=-1), cache_blank)
    print("============================")


    # logprobs = prompt_logprobs
    # logprobs_recalc = get_logprobs(model, params, prompt_tokens, cache_blank)
    logratio = logprobs - logprobs_recalc
    ratio = jnp.exp(logratio)
    breakpoint()

    return tokens, logprobs