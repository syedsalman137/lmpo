from lmpo.models.qwen3 import create_model_from_hf
from lmpo.inference.sampling import pad_and_collate, autoregressive_sample
import json
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import math
from transformers import PreTrainedTokenizerFast, AddedToken
from utils.checkpoint import Checkpoint
import shutil

hf_dir = '/nfs/hf/Qwen--Qwen3-0.6B/'
ckpt_dir = '/nfs/gcs/jaxconverted/Qwen3-0.6B/'

model, params = create_model_from_hf(hf_dir)
ckpt = Checkpoint(ckpt_dir+'params.pkl', parallel=False)
ckpt.params = params
ckpt.save()

# copy config.json to new dir.
shutil.copy(hf_dir + 'config.json', ckpt_dir + 'config.json')
shutil.copy(hf_dir + 'tokenizer_config.json', ckpt_dir + 'tokenizer_config.json')
shutil.copy(hf_dir + 'tokenizer.json', ckpt_dir + 'tokenizer.json')