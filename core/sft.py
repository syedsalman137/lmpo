import re, os, sys, json, time, shutil
from typing import Iterable
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import optax
import tqdm
import wandb
import ml_collections
from absl import flags

def batched(iterable: Iterable, n: int):
    item_list = []
    for i, item in enumerate(iterable):
        item_list.append(item)
        if (i + 1) % n == 0 and i != 0:
            yield item_list
            del item_list
            item_list = []
    yield item_list
    del item_list

try:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir('/nfs/jax-cache')
    from localutils.debugger import enable_debug
    enable_debug()
except:
    pass

from lmpo.models.qwen3 import create_model_from_ckpt
from lmpo.models.tokenizer import create_tokenizer
from lmpo.core.sampling import pad_and_collate, autoregressive_sample
from lmpo.utils.sharding import create_sharding, host_gather
from lmpo.utils.train_state import TrainState
from lmpo.utils.checkpoint import Checkpoint
from lmpo.utils.configs import define_flag_dict
from lmpo.utils.wandb import setup_wandb

# -----------------------
# Config
# -----------------------
config = ml_collections.ConfigDict({
    'enable_wandb': 0,
    'wandb_project': "lmpo",
    'wandb_name': 'sft-run',
    'wandb_group': 'SFT',

    'model_dir': '/nfs/gcs/jaxconverted/Qwen3-1.7B/',
    'save_dir': "",
    'save_interval': 2000,

    # Data
    'dataset_path': '/path/to/train.jsonl',
    'eval_dataset_path': '',       # optional
    'test_dataset_path': '',       # optional
    'max_seq_len': 38912,
    'loss_on_prompt': 0,           # 0 = only compute loss on response; 1 = also include prompt
    'truncate_direction': 'right', # 'right' or 'left' truncation

    # Training
    'train_steps': 50_000,
    'log_interval': 20,
    'eval_interval': 1000,
    'test_interval': 4000,
    'eval_batches': 50,
    'train_batch_per_device': 2,   # increase until OOM
    'eval_batch_per_device': 4,   # increase until OOM
    'test_batch_per_device': 1,   # increase until OOM
    'seed': 42,

    # Optimizer
    'lr': 5e-5,
    'adam_b1': 0.9,
    'adam_b2': 0.95,
    'adam_eps': 1e-8,
    'weight_decay': 1e-2,
    'grad_clip_norm': 1.0,
})
define_flag_dict(config)
FLAGS = flags.FLAGS
FLAGS(sys.argv)

host_id = jax.process_index()
num_hosts = jax.process_count()
local_device_count = jax.local_device_count()

if host_id == 0 and FLAGS.enable_wandb:
    setup_wandb(FLAGS.flag_values_dict(),
                project=FLAGS.wandb_project,
                name=FLAGS.wandb_name,
                group=FLAGS.wandb_group)

# -----------------------
# Helpers: data
# -----------------------
def _tok_encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)

def _build_example(tokenizer, ex, eos_id):
    if 'prompt' in ex and 'response' in ex:
        prompt, response = ex['prompt'], ex['response']
    elif 'input' in ex and 'output' in ex:
        prompt, response = ex['input'], ex['output']
    elif 'text' in ex:
        prompt, response = "", ex['text']
    else:
        raise ValueError(f"Unsupported example keys: {list(ex.keys())}")

    chat_ex = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ],
        tokenize=False,
        enable_thinking=False,
    )

    prompt_match = re.search(r'<\|im_start\|>user.*<\|im_end\|>\n', chat_ex, re.DOTALL)
    start_prompt, end_prompt = prompt_match.span()
    prompt = chat_ex[start_prompt:end_prompt]

    response_match = re.search(r'<\|im_start\|>assistant.*', chat_ex, re.DOTALL)
    start_response, end_response = response_match.span()
    response = chat_ex[start_response:end_response]

    prompt_ids = _tok_encode(tokenizer, prompt) if prompt else []
    response_ids = _tok_encode(tokenizer, response)
    if len(response_ids) == 0 or response_ids[-1] != eos_id:
        response_ids = response_ids + [eos_id]
    return prompt_ids, response_ids

def _concat_truncate_and_pad(prompt_ids, response_ids, max_len, pad_id, truncate_direction="right", loss_on_prompt=False):
    full = prompt_ids + response_ids
    if len(full) > max_len:
        if truncate_direction == 'right':
            full = full[:max_len]
        else:  # left
            full = full[-max_len:]

    mask = np.zeros(len(full), dtype=np.int32)
    if loss_on_prompt:
        mask[:] = 1
    else:
        # start idx of response in full depends on truncation
        if truncate_direction == 'right':
            resp_start = min(len(prompt_ids), max_len)
        else:  # left truncation
            # if we cut from left, prompt may be partially/fully cut
            cut = max(0, len(prompt_ids) + len(response_ids) - max_len)
            # portion of prompt removed:
            prompt_kept = max(0, len(prompt_ids) - cut)
            resp_start = prompt_kept
        mask[resp_start:] = 1

    # pad
    pad_len = max_len - len(full)
    if pad_len > 0:
        full = full + [pad_id] * pad_len
        pad_mask = np.zeros(pad_len, dtype=np.int32)
        mask = np.concatenate([mask, pad_mask], axis=0)

    return np.array(full, dtype=np.int32), mask

class JsonlBatcher:
    def __init__(self, path, tokenizer, batch_size_per_host, max_len, loss_on_prompt=False, truncate_direction='right',
                 shard_by_host=True, seed=0):
        self.path = path
        self.tokenizer = tokenizer
        self.batch_size = batch_size_per_host
        self.max_len = max_len
        self.loss_on_prompt = loss_on_prompt
        self.truncate_direction = truncate_direction
        self.pad_id = tokenizer.get_pad_token_id()
        self.eos_id = tokenizer.get_eos_token_id()
        self.rng = np.random.RandomState(seed)

        # Load into memory (simple and robust). For giant datasets, switch to streaming.
        with open(path, 'r') as f:
            self.raw = [json.loads(line) for line in f if line.strip()]

        # Evenly shard across hosts to avoid duplication
        if shard_by_host and num_hosts > 1:
            self.raw = self.raw[host_id::num_hosts]

        if len(self.raw) == 0:
            raise ValueError(f"No examples found in {path} for host {host_id}.")
        self.indices = jnp.arange(len(self.raw))
        # self._reshuffle()
        self.ptr = 0

    def _reshuffle(self):
        self.indices = self.rng.permutation(len(self.raw)).tolist()
        self.ptr = 0

    def _next_example(self):
        # if self.ptr >= len(self.indices):
        #     self._reshuffle()
        self.ptr = 0
        ex = self.raw[self.indices[self.ptr]]
        self.ptr += 1
        return ex

    def next_batch(self):
        prompt_ids_list = []
        response_ids_list = []
        max_length_allowed = self.max_len
        max_length_used = -1
        for _ in range(self.batch_size):
            ex = self._next_example()
            prompt_ids, response_ids = _build_example(
                tokenizer=self.tokenizer,
                ex=ex,
                eos_id=self.eos_id,
            )
            prompt_ids_list.append(prompt_ids)
            response_ids_list.append(response_ids)
            max_length_used = max(max_length_used, len(prompt_ids) + len(response_ids))
        max_length_used = min(max_length_used, max_length_allowed)

        xs = []
        ms = []
        for prompt_ids, token_ids in zip(prompt_ids_list, response_ids_list):
            tokens, mask = _concat_truncate_and_pad(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                max_len=max_length_used,
                pad_id=self.pad_id,
                truncate_direction=self.truncate_direction,
                loss_on_prompt=self.loss_on_prompt,
            )
            xs.append(tokens)
            ms.append(mask)
        return np.stack(xs, axis=0), np.stack(ms, axis=0)

# -----------------------
# Model/opt init
# -----------------------
ckpt_dir = FLAGS.model_dir
model, params = create_model_from_ckpt(ckpt_dir)

rng = jax.random.PRNGKey(FLAGS.seed)
tx = optax.chain(
    optax.clip_by_global_norm(FLAGS.grad_clip_norm),
    optax.sgd(
        learning_rate=FLAGS.lr,
    ),
)
init_fn = partial(TrainState.create_with_params, model_def=model, tx=tx, use_ema=False)
train_state_shape = jax.eval_shape(init_fn, rng=rng, params=params)
train_state_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape)
train_state = jax.jit(lambda r, p: init_fn(rng=r, params=p), out_shardings=train_state_shard)(rng, params)

jax.debug.visualize_array_sharding(train_state.params['Block_0']['Dense_0']['kernel'])

tokenizer = create_tokenizer(ckpt_dir)
pad_id = tokenizer.get_pad_token_id()
eos_id = tokenizer.get_eos_token_id()

# -----------------------
# Dataloaders
# -----------------------
per_host_batch = local_device_count * FLAGS.train_batch_per_device
train_iter = JsonlBatcher(
    path=FLAGS.dataset_path,
    tokenizer=tokenizer,
    batch_size_per_host=per_host_batch,
    max_len=FLAGS.max_seq_len,
    loss_on_prompt=FLAGS.loss_on_prompt,
    truncate_direction=FLAGS.truncate_direction,
    shard_by_host=True,
    seed=FLAGS.seed + host_id
)

eval_iter = None
if FLAGS.eval_dataset_path:
    eval_per_host_batch = local_device_count * FLAGS.eval_batch_per_device
    eval_iter = JsonlBatcher(
        path=FLAGS.eval_dataset_path,
        tokenizer=tokenizer,
        batch_size_per_host=eval_per_host_batch,
        max_len=FLAGS.max_seq_len,
        loss_on_prompt=FLAGS.loss_on_prompt,
        truncate_direction=FLAGS.truncate_direction,
        shard_by_host=True,
        seed=FLAGS.seed + 1234 + host_id
    )

test_ds = None
if FLAGS.test_dataset_path:
    with open(FLAGS.test_dataset_path, 'r') as f:
        test_ds = [json.loads(line) for line in f if line.strip()]
    

# -----------------------
# Jitted steps
# -----------------------
@partial(jax.jit, out_shardings=(train_state_shard, None))
def sft_update_step(train_state: TrainState, token_batch, loss_mask):
    # token_batch: [B, L], loss_mask: [B, L] with 1s where we apply the loss
    text_in = token_batch[:, :-1]
    text_tgt = token_batch[:, 1:]
    mask = loss_mask[:, 1:]  # align mask with targets
    attn_mask = (text_in != pad_id).astype(jnp.int32)

    def loss_fn(grad_params):
        logits, _ = train_state.call_model(text_in, attn_mask, cache=None, params=grad_params)  # [B, T, V]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.sum(log_probs * jax.nn.one_hot(text_tgt, logits.shape[-1]), axis=-1)  # [B, T]
        denom = jnp.maximum(jnp.sum(mask), 1)
        loss = jnp.sum(nll * mask) / denom

        # metrics
        ce = loss
        preds = jnp.argmax(logits, axis=-1)
        correct = jnp.sum((preds == text_tgt) * mask)
        acc = correct / denom
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1)
        ent_avg = jnp.sum(entropy * mask) / denom

        return loss, {
            'loss': loss,
            'cross_entropy': ce,
            'ppl': jnp.exp(ce),
            'accuracy': acc,
            'entropy_per_token': ent_avg,
            'trained_tokens_per_seq': jnp.mean(jnp.sum(mask, axis=-1)),
        }

    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    updates, opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)

    new_state = train_state.replace(
        params=new_params,
        opt_state=opt_state,
        step=train_state.step + 1,
    )
    return new_state, info

@jax.jit
def sft_eval_step(train_state: TrainState, token_batch, loss_mask):
    text_in = token_batch[:, :-1]
    text_tgt = token_batch[:, 1:]
    mask = loss_mask[:, 1:]
    attn_mask = (text_in != pad_id).astype(jnp.int32)

    logits, _ = train_state.call_model(text_in, attn_mask, cache=None)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.sum(log_probs * jax.nn.one_hot(text_tgt, logits.shape[-1]), axis=-1)
    denom = jnp.maximum(jnp.sum(mask), 1)

    ce = jnp.sum(nll * mask) / denom
    ppl = jnp.exp(ce)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.sum((preds == text_tgt) * mask) / denom
    return {'cross_entropy': ce, 'ppl': ppl, 'accuracy': acc}

def sft_test_step(train_state: TrainState, batch_ex):
    def get_integer_answer(text):
        try:
            boxed_integers = re.findall(r'\\boxed{(\d[\d,_\s]*)}', text)
            if not boxed_integers:
                normal_integers = re.findall(r'(\d[\d,_\s]*)', text)
                if not normal_integers:
                    return 0
                return int(re.sub(r'[_,\s]+', '', normal_integers[-1]))
            return int(re.sub(r'[_,\s]+', '', boxed_integers[-1]))
        except ValueError:
            return 0

    def get_prompt_n_response(ex):
        if 'prompt' in ex and 'response' in ex:
            prompt, response = ex['prompt'], ex['response']
        elif 'input' in ex and 'output' in ex:
            prompt, response = ex['input'], ex['output']
        elif 'text' in ex:
            prompt, response = "", ex['text']
        else:
            raise ValueError(f"Unsupported example keys: {list(ex.keys())}")
        return prompt, response

    prompt_list = [get_prompt_n_response(ex)[0] for ex in batch_ex]
    response_list = [get_prompt_n_response(ex)[1] for ex in batch_ex]
    prompt_token_list = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
        for prompt in prompt_list
    ]
    response_token_list = [
        tokenizer.apply_chat_template([{"role": "user", "content": response}])
        for response in response_list
    ]
    max_response_tokens = len(max(response_token_list, key=len))
    num_generation_tokens = min(FLAGS.max_seq_len, -(-max_response_tokens // 4096) * 4096)
    prompt_token_batch = pad_and_collate(prompt_token_list, pad_id=0, force_length=128)
    prompt_token_batch = shard_data_fn(prompt_token_batch)

    rng = jax.random.PRNGKey(FLAGS.seed)
    tokens_out = autoregressive_sample(
        train_state.model_def,
        train_state.params,
        prompt_token_batch,
        num_generation_tokens,
        rng,
        temp=0.6,
        pad_id=0,
        data_shard=data_shard,
        no_shard=no_shard,
        force_answer_at=-1
    )
    tokens_out = host_gather(tokens_out)
    output_list = [tokenizer.decode(row) for row in tokens_out]

    target_answer_list = [get_integer_answer(response)
                    for response in response_list]
    output_answer_list = [get_integer_answer(output)
                    for output in output_list]

    return output_answer_list, target_answer_list

# -----------------------
# Train loop
# -----------------------
rng = jax.random.PRNGKey(FLAGS.seed + host_id)
step_start_time = time.time()

for step in tqdm.tqdm(range(FLAGS.train_steps), disable=(host_id != 0)):
    tokens_np, mask_np = train_iter.next_batch()
    tokens = shard_data_fn(jnp.array(tokens_np))
    mask = shard_data_fn(jnp.array(mask_np))

    train_state, info = sft_update_step(train_state, tokens, mask)
    info = jax.device_get(info)
    info = jax.tree.map(lambda x: np.array(x), info)
    info = jax.tree.map(lambda x: x.mean(), info)  # average across devices

    if host_id == 0 and (step % FLAGS.log_interval == 0):
        info['global_step'] = step
        info['time_per_step'] = time.time() - step_start_time
        step_start_time = time.time()
        print(info)
        if FLAGS.enable_wandb:
            wandb.log(info)

    # Eval
    if eval_iter is not None and step > 0 and step % FLAGS.eval_interval == 0:
        eval_metrics = {'cross_entropy': [], 'ppl': [], 'accuracy': []}
        for _ in range(FLAGS.eval_batches):
            etokens_np, emask_np = eval_iter.next_batch()
            etokens = shard_data_fn(jnp.array(etokens_np))
            emask = shard_data_fn(jnp.array(emask_np))
            m = sft_eval_step(train_state, etokens, emask)
            m = jax.device_get(m)
            m = jax.tree.map(lambda x: np.array(x), m)
            m = jax.tree.map(lambda x: x.mean(), m)
            for k in eval_metrics:
                eval_metrics[k].append(m[k])
        eval_info = {f'eval/{k}': float(np.mean(v)) for k, v in eval_metrics.items()}
        print(eval_info)
        if host_id == 0 and FLAGS.enable_wandb:
            wandb.log(eval_info, commit=False)
    
    if test_ds is not None and step % FLAGS.test_interval == 0:
        test_metrics = {'answer accuracy': []}
        test_per_host_batch = local_device_count * FLAGS.test_batch_per_device 
        for test_ex_batch in batched(test_ds, n=test_per_host_batch):
            output_list, target_list = sft_test_step(train_state, test_ex_batch)
            for k in test_metrics:
                test_metrics[k].extend(
                    zip(output_list, target_list)
                )
        test_info = {f'test/{k}': float(np.mean(v)) for k, v in test_metrics.items()}
        print(test_info)
        if host_id == 0 and FLAGS.enable_wandb:
            wandb.log(test_info, commit=False)

    # Save
    if FLAGS.save_dir != "" and step > 0 and step % FLAGS.save_interval == 0:
        params_gather = host_gather(train_state.params)
        if host_id == 0:
            step_dir = os.path.join(FLAGS.save_dir, f'step{step}')
            os.makedirs(step_dir, exist_ok=True)
            cp = Checkpoint(os.path.join(step_dir, 'params.pkl'), parallel=False)
            cp.params = params_gather
            cp.save()
            del cp
            shutil.copy(os.path.join(FLAGS.model_dir, 'config.json'), os.path.join(step_dir, 'config.json'))
            shutil.copy(os.path.join(FLAGS.model_dir, 'tokenizer_config.json'), os.path.join(step_dir, 'tokenizer_config.json'))
            shutil.copy(os.path.join(FLAGS.model_dir, 'tokenizer.json'), os.path.join(step_dir, 'tokenizer.json'))
        del params_gather