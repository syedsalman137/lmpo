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

def pad_and_collate(token_batch, pad_id=0, multi_host=False):
    max_len = max([len(x) for x in token_batch])
    if multi_host:
        max_len = max(jax.experimental.multihost_utils.process_allgather(max_len))
    return np.array([(max_len - len(x)) * [pad_id] + x for x in token_batch])

def autoregressive_sample(model: Qwen3Model, params, prompt_tokens, num_generation_tokens, pad_id=0, data_shard=None, no_shard=None):
    """
    Samples tokens autoregressively, and can batch for performance.
    Args:
        prompt_tokens: An array of tokens, padded by `pad_id` on the LEFT. [batch, time].
        max_seq_len: The maximum sequence length to sample.
    """
    batch_size = prompt_tokens.shape[0]
    pad_to = 2 ** math.ceil(math.log2((prompt_tokens.shape[-1])))
    prompt_tokens = jnp.pad(prompt_tokens, [(0, 0), (0, pad_to - prompt_tokens.shape[-1])])
    token_mask = jnp.where(prompt_tokens != pad_id, 1, 0).astype(jnp.int32)
    max_seq_len = pad_to + num_generation_tokens

    cache = KVCache.create(model.num_layers, batch_size, max_seq_len, model.head_dim, model.kv_heads)
    cache = cache.replace(starts=count_left_padding(prompt_tokens, pad_id=0))
    cache_sharding = KVCache.get_sharding(data_shard, no_shard)
    cache = jax.jit(lambda x: x, out_shardings=cache_sharding)(cache)

    @partial(jax.jit, out_shardings=(data_shard, cache_sharding))
    def model_apply(params, tokens, token_mask, cache):
        print("JIT compiling model_apply for tokens of shape", tokens.shape)
        return model.apply({'params': params}, tokens, token_mask, cache=cache)

    # Fill cache with the prompt tokens, and sample one token.
    logits, cache = model_apply(params, prompt_tokens, token_mask, cache=cache)
    sampled_token = jnp.argmax(logits, axis=-1)
    sampled_token = sampled_token[:, cache.length - 1 : cache.length]

    tokens_list = []
    max_samples = max_seq_len - prompt_tokens.shape[-1]
    for i in range(max_samples):
        tokens_list.append(sampled_token)
        next_token_mask = jnp.ones(sampled_token.shape, dtype=jnp.int32)
        logits, cache = model_apply(params, sampled_token, next_token_mask, cache=cache)
        sampled_token = jnp.argmax(logits, axis=-1)
        # TODO: Early stopping if EOS token is sampled in all batches.
    tokens = jnp.array(jnp.concatenate(tokens_list, axis=-1))

    return tokens