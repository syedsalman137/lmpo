import jax
import numpy as np
import argparse

from lmpo.models.qwen3 import create_model_from_ckpt
from lmpo.inference.sampling import pad_and_collate, autoregressive_sample
from lmpo.utils.sharding import create_sharding, host_gather
from lmpo.models.tokenizer import create_tokenizer

try: # If you like to use these helpers, you can.
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache('/nfs/jax-cache')
    from localutils.debugger import enable_debug
    enable_debug()
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='/nfs/gcs/jaxconverted/Qwen3-0.6B/')
args = parser.parse_args()
ckpt_dir = args.ckpt_dir

model, params = create_model_from_ckpt(ckpt_dir)
param_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape=params)
params = jax.jit(lambda x: x, out_shardings=param_shard)(params)
tokenizer = create_tokenizer(ckpt_dir)

imagenet_labels = open('inference/imagenet_labels.txt').read().splitlines()
poem_prompts = [f'Write a haiku about of {imagenet_labels[np.random.randint(len(imagenet_labels))]}' for _ in range(len(jax.local_devices()))]

pad_id = 0
token_list = [
    tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    for text in poem_prompts
]

token_batch = pad_and_collate(token_list, pad_id=pad_id, force_length=256)
print("Input tokens local:", token_batch.shape)
token_batch = shard_data_fn(token_batch)
print("Input tokens global:", token_batch.shape)
num_generation_tokens = 32
rng = jax.random.PRNGKey(0)
tokens_out = autoregressive_sample(
    model, params, token_batch, rng=rng, num_generation_tokens=num_generation_tokens, pad_id=pad_id, data_shard=data_shard, no_shard=no_shard)
tokens_out = host_gather(tokens_out)

responses = [tokenizer.decode(row) for row in tokens_out]
if jax.process_index() == 0:
    for i, text in enumerate(poem_prompts):
        print(f" ======= {text} =======")
        print(responses[i].split('<|im_end|>')[0])

    print("========= Full raw decoded tokens =========")
    print(tokenizer.decode(token_list[0] + tokens_out[0].tolist()))
    print('Total tokens shape', tokens_out.shape)
    print("=============")