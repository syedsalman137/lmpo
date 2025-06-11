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

is_multi_host = len(jax.local_devices()) != len(jax.devices())

# hf_dir = '/nfs/hf/Qwen--Qwen3-0.6B/'
# model, params = create_model_from_hf(hf_dir)

ckpt_dir = '/nfs/gcs/jaxconverted/Qwen3-0.6B/'
model, params = create_model_from_ckpt(ckpt_dir)
param_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape=params)
params = jax.jit(lambda x: x, out_shardings=param_shard)(params)
jax.debug.visualize_array_sharding(params['Block_0']['Dense_0']['kernel'])
tokenizer = create_tokenizer(ckpt_dir)

imagenet_labels = open('inference/imagenet_labels.txt').read().splitlines()
poem_prompts = [f'Write a haiku about of {imagenet_labels[np.random.randint(len(imagenet_labels))]}' for _ in range(len(jax.local_devices()))]

pad_id = 0
token_list = [
    tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    for text in poem_prompts
]
token_batch = pad_and_collate(token_list, pad_id=pad_id, multi_host=is_multi_host)
print("Input tokens local:", token_batch.shape)
token_batch = shard_data_fn(token_batch)
print("Input tokens global:", token_batch.shape)
num_generation_tokens = 32
tokens_out = autoregressive_sample(
    model, params, token_batch, num_generation_tokens=num_generation_tokens, pad_id=pad_id, data_shard=data_shard, no_shard=no_shard
)
if is_multi_host:
    tokens_out = jax.experimental.multihost_utils.process_allgather(tokens_out)
responses = [tokenizer.decode(row) for row in tokens_out]
if jax.process_index() == 0:
    for i, text in enumerate(poem_prompts):
        print(f" ======= {text} =======")
        print(responses[i].split('<|im_end|>')[0])
print(tokens_out[0])
print('Total tokens shape', tokens_out.shape)