from lmpo.models.qwen3 import create_model_from_hf, create_model_from_ckpt
from lmpo.inference.sampling import pad_and_collate, autoregressive_sample
import json
import jax.numpy as jnp
import jax
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizerFast, AddedToken
from utils.sharding import create_sharding
from models.tokenizer import create_tokenizer

from localutils.debugger import enable_debug
enable_debug()


is_multi_host = len(jax.local_devices()) != len(jax.devices())

# hf_dir = '/nfs/hf/Qwen--Qwen3-0.6B/'
# model, params = create_model_from_hf(hf_dir)

ckpt_dir = '/nfs/gcs/jaxconverted/Qwen3-0.6B/'
# ckpt_dir = '/nfs/gcs/jaxconverted/Qwen3-8B/'
model, params = create_model_from_ckpt(ckpt_dir)
param_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape=params)
# param_shard, no_shard, data_shard, shard_data_fn = create_sharding('dp', train_state_shape=params)
params = jax.jit(lambda x: x, out_shardings=param_shard)(params)
jax.debug.visualize_array_sharding(params['Block_0']['Dense_0']['kernel'])
tokenizer = create_tokenizer(ckpt_dir)

imagenet_labels = open('inference/imagenet_labels.txt').read().splitlines()
# poem_prompts = [f'Write a haiku about of {imagenet_labels[np.random.randint(len(imagenet_labels))]}' for _ in range(len(jax.local_devices()))]
poem_prompts = [f'Write a poem about cats.' for _ in range(len(jax.local_devices()))]

pad_id = 0
# token_list = [
#     tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
#     for text in poem_prompts
# ]


SYSTEM_PROMPT = "You are a helpful assistant. You first think about the reasoning process in the mind, and then provide the user with the answer."
USER_PROMPT = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. Think for only ten sentences, then return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
target = np.random.randint(100)
numbers = [np.random.randint(100) for _ in range(4)]
output_tokens = tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(target=target, numbers=numbers)},
    ],
    add_generation_prompt=True,
    enable_thinking=True
)
token_list = [output_tokens] * len(jax.local_devices())



token_batch = pad_and_collate(token_list, pad_id=pad_id, multi_host=is_multi_host, force_length=256)
print("Input tokens local:", token_batch.shape)
token_batch = shard_data_fn(token_batch)
print("Input tokens global:", token_batch.shape)
num_generation_tokens = 32
rng = jax.random.PRNGKey(0)
tokens_out = autoregressive_sample(
    model, params, token_batch, rng=rng, num_generation_tokens=num_generation_tokens, pad_id=pad_id, data_shard=data_shard, no_shard=no_shard, force_answer=False
)
if is_multi_host:
    tokens_out = jax.experimental.multihost_utils.process_allgather(tokens_out)

responses = [tokenizer.decode(row) for row in tokens_out]
if jax.process_index() == 0:
    for i, text in enumerate(poem_prompts):
        print(f" ======= {text} =======")
        print(responses[i].split('<|im_end|>')[0])

    print(tokenizer.decode(token_list[0] + tokens_out[0].tolist()))
print(tokens_out[0])
print('Total tokens shape', tokens_out.shape)
token_batch = jax.experimental.multihost_utils.process_allgather(token_batch)
breakpoint()