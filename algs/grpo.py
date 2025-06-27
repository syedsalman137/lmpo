
import jax.numpy as jnp
import jax
import numpy as np
from utils.sharding import create_sharding
from utils.train_state import TrainState
from models.tokenizer import create_tokenizer
import tqdm
import optax
from functools import partial
import wandb
import ml_collections
import sys
from absl import app, flags

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache('/nfs/jax-cache')

from localutils.debugger import enable_debug
enable_debug()

from lmpo.models.qwen3 import create_model_from_hf, create_model_from_ckpt
from lmpo.inference.sampling import pad_and_collate, autoregressive_sample
from lmpo.utils.configs import define_flag_dict
from lmpo.utils.wandb import setup_wandb
from lmpo.envs.poem_length import PoemLengthEnv
from lmpo.envs.gsm8k import GSM8KEnv
from lmpo.envs.countdown import CountdownEnv

config = ml_collections.ConfigDict({
    'wandb_project': "lmpo",
    'wandb_name': 'lmpo-run',
    'model_dir': '/nfs/gcs/jaxconverted/Qwen3-1.7B/',
    # env settings.
    'env_name': 'poem',
    'num_generation_tokens': -1, # Use default from env.
    # training settings.
    'groups_per_batch': 128, # for global batch, multiply by group_size.
    'ppo_minibatch': 64,
    'group_size': 8,
    'do_group_normalization': 1,
    'do_global_normalization': 0,
    'do_group_filter': 1,
    'lr': 1e-6,
    'clip_epsilon': 0.2,
})
define_flag_dict(config)
FLAGS = flags.FLAGS
FLAGS(sys.argv)
if jax.process_index() == 0:
    setup_wandb(FLAGS.flag_values_dict(), project=FLAGS.wandb_project, name=FLAGS.env_name+'-'+FLAGS.wandb_name)
    # rollouts_table = wandb.Table(columns=["step", "text", "reward"])
    rollouts_list = []

is_multi_host = len(jax.local_devices()) != len(jax.devices())
host_id = jax.process_index()
def host_gather(x):
    return jax.experimental.multihost_utils.process_allgather(x) if is_multi_host else x
                                          
ckpt_dir = FLAGS.model_dir
model, params = create_model_from_ckpt(ckpt_dir)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(FLAGS.lr, b1=0.9, b2=0.95, weight_decay=1e-2)
)
rng = jax.random.PRNGKey(0)
init_fn = partial(TrainState.create_with_params, model_def=model, tx=tx, use_ema=False)
train_state_shape = jax.eval_shape(init_fn, rng=rng, params=params)
train_state_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape)
train_state = jax.jit(lambda r, p: init_fn(rng=r, params=p), out_shardings=train_state_shard)(rng, params)

jax.debug.visualize_array_sharding(train_state.params['Block_0']['Dense_0']['kernel'])
tokenizer = create_tokenizer(ckpt_dir)
pad_id = tokenizer.get_pad_token_id()

if FLAGS.env_name.lower() == 'poem':
    env = PoemLengthEnv(tokenizer)
elif FLAGS.env_name.lower() == 'gsm8k':
    env = GSM8KEnv(tokenizer)
elif FLAGS.env_name.lower() == 'countdown':
    env = CountdownEnv(tokenizer)
if FLAGS.num_generation_tokens == -1:
    FLAGS.num_generation_tokens = env.tokens_per_action
np.random.seed(jax.process_index())

@partial(jax.jit, out_shardings=(None))
def get_logprobs(train_state: TrainState, token_batch, mask):
    print("JIT compiling logprob function for token_batch of shape", token_batch.shape)
    text_input, text_target = token_batch[:, :-1], token_batch[:, 1:]
    mask = mask[:, 1:]
    token_mask = jnp.where(text_input != pad_id, 1, 0).astype(jnp.int32)
    logits, _ = train_state.call_model(text_input, token_mask, cache=None)
    logprobs = jax.nn.log_softmax(logits, axis=-1) # [batch, time, vocab_size]
    logprobs = jnp.sum(logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
    return logprobs

@partial(jax.jit, out_shardings=(train_state_shard, None))
def update(train_state: TrainState, token_batch, mask, advantages, old_logprobs):
    print("JIT compiling update function for token_batch of shape", token_batch.shape)
    text_input, text_target = token_batch[:, :-1], token_batch[:, 1:]
    mask = mask[:, 1:]
    token_mask = jnp.where(text_input != pad_id, 1, 0).astype(jnp.int32)
    def loss_fn(grad_params):
        logits, _ = train_state.call_model(text_input, token_mask, cache=None, params=grad_params)
        logprobs = jax.nn.log_softmax(logits) # [batch, time, vocab_size]
        token_logprobs = jnp.sum(logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)

        # PPO loss.
        logratio = token_logprobs - old_logprobs
        ratio = jnp.exp(logratio)
        pg_loss1 = -advantages[:, None] * ratio
        pg_loss2 = -advantages[:, None] * jnp.clip(ratio, 1 - FLAGS.clip_epsilon, 1 + FLAGS.clip_epsilon)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2)

        # Metrics
        avg_over_mask = lambda x : jnp.sum(x * mask) / jnp.sum(mask)
        importance_ratio = avg_over_mask(ratio)
        importance_ratio_mag = avg_over_mask(jnp.abs(1 - ratio))
        approx_kl = avg_over_mask((ratio - 1) - logratio)
        entropy = avg_over_mask(-jnp.sum(jax.nn.softmax(logits) * logprobs, axis=-1))
        clip_fracs = avg_over_mask(jnp.abs(ratio - 1.0) > FLAGS.clip_epsilon)
        cross_entropy = avg_over_mask(-token_logprobs)

        # jax.debug.breakpoint()

        loss = jnp.mean(pg_loss * mask)  # Average over the batch and time steps.
        return loss, {
            'loss': loss,
            'advantages': jnp.mean(advantages),
            'advantages_magnitude': jnp.mean(jnp.abs(advantages)),
            'nonzero_advantages': jnp.mean(advantages != 0),
            'entropy_per_token': entropy,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fracs,
            'cross_entropy': cross_entropy,
            'importance_ratio': importance_ratio,
            'importance_ratio_magnitude': importance_ratio_mag,
            'importrance_ratio_max': jnp.max(ratio * mask),
            'importrance_ratio_min': jnp.min(ratio * mask),
            'trained_tokens_per_seq': jnp.mean(jnp.sum(mask, axis=-1)),
            'is_max_tokens': jnp.mean(mask[:, -1] == True),
        }
    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    updates, opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)
    train_state = train_state.replace(
        params=new_params,
        opt_state=opt_state,
        step=train_state.step + 1,
    )
    return train_state, info

rollout_batch_size = jax.local_device_count()
assert rollout_batch_size % FLAGS.group_size == 0
rng = jax.random.PRNGKey(jax.process_index())

for i in tqdm.tqdm(range(10000)):

    # Fill this global on-policy buffer with groups that have A != 0.
    buffer_tokens = []
    buffer_logprobs = []
    buffer_advantages = []
    env_infos_history = {}
    env_infos_history['return'] = []
    num_rollout_iters = 0
    while len(buffer_tokens) < FLAGS.groups_per_batch:
        num_rollout_iters += 1
        env_states, env_tokens = [], []
        for _ in range(rollout_batch_size // FLAGS.group_size):
            env_state, output_tokens = env.reset()
            for _ in range(FLAGS.group_size):
                env_states.append(env_state)
                env_tokens.append(output_tokens)

        prompt_tokens = pad_and_collate(env_tokens, pad_id=pad_id, multi_host=is_multi_host, force_length=256)
        prompt_tokens = shard_data_fn(prompt_tokens)
        num_generation_tokens = FLAGS.num_generation_tokens
        rng, key = jax.random.split(rng)
        action_tokens = autoregressive_sample(
            train_state.model_def, train_state.params, prompt_tokens, rng=key, num_generation_tokens=num_generation_tokens, 
            pad_id=pad_id, data_shard=data_shard, no_shard=no_shard, force_answer=FLAGS.env_name.lower() == 'countdown'
        )
        prompt_tokens = host_gather(prompt_tokens)
        action_tokens = host_gather(action_tokens)
        all_tokens = jnp.concatenate([prompt_tokens, action_tokens], axis=-1)

        action_tokens_local = action_tokens[host_id * rollout_batch_size : (host_id+1) * rollout_batch_size]
        new_states, _, returns_local, dones, env_infos = env.step_list(env_states, [t.tolist() for t in action_tokens_local])
        assert dones[0] # Only supports bandit envs for now.
        returns_local = np.array(returns_local)
        returns = host_gather(shard_data_fn(returns_local))
        for k, v in env_infos.items():
            if k not in env_infos_history:
                env_infos_history[k] = []
            v_global = host_gather(shard_data_fn(np.array(v)))
            env_infos_history[k] += v_global.tolist()
        env_infos_history['return'] += returns.tolist()


        mask_size = prompt_tokens.shape[-1]

        # Advantage calculation.
        returns = jnp.reshape(returns, (-1, FLAGS.group_size))
        advantages = returns
        if FLAGS.do_group_normalization:
            group_mean = np.mean(advantages, axis=-1)
            group_std = np.std(advantages, axis=-1) + 1e-8
            advantages = (advantages - group_mean[:, None]) / group_std[:, None]
        if FLAGS.do_global_normalization:
            global_mean = np.mean(advantages)
            global_std = np.std(advantages) + 1e-8
            advantages = (advantages - global_mean) / global_std
        advantages_grouped = advantages # [batch_size // group_size, group_size]
        all_tokens_grouped = all_tokens.reshape(-1, FLAGS.group_size, all_tokens.shape[-1])

        for group_idx in range(advantages_grouped.shape[0]):
            if np.all(advantages_grouped[group_idx, :] == 0) and FLAGS.do_group_filter:
                continue
            else:
                buffer_tokens.append(all_tokens_grouped[group_idx, :])
                buffer_advantages.append(advantages_grouped[group_idx, :])
        print(f"Buffer size: {len(buffer_tokens) * FLAGS.group_size}. Return avg: {np.mean(returns)}")
        if jax.process_index() == 0:
            print(env.render(new_states[0]))
            print("Rollout returns:", returns)
            print("Rollout advantages:", advantages_grouped)

    def ppo_shard(x):
        """Helper function that takes a local buffer, shards across devices, then splits into PPO minibatches."""
        host_id = jax.process_index()
        host_slice = FLAGS.ppo_minibatch // jax.process_count()
        x = jnp.reshape(x, (FLAGS.ppo_minibatch, -1, *x.shape[1:]))
        x = x[host_id * host_slice : (host_id + 1) * host_slice, :]
        x = shard_data_fn(x)
        return x # [ppo_minibatch, num_minibatches (j), ...] where first dim is sharded.

    # The buffer is syncronized among hosts.
    tokens_all = jnp.concatenate(buffer_tokens, axis=0)
    advantages = jnp.concatenate(buffer_advantages, axis=0)
    global_batch_size = FLAGS.groups_per_batch * FLAGS.group_size
    tokens_all = tokens_all[:global_batch_size]
    advantages = advantages[:global_batch_size]

    # Mask = False for all prompt tokens, and tokens after <|im_end|> token.
    mask = (jnp.arange(tokens_all.shape[-1]) >= mask_size - 1)[None, :]
    eos_idx = jnp.argmax(tokens_all[:, mask_size:] == tokenizer.get_eos_token_id(), axis=-1)
    eos_idx = jnp.where(eos_idx == 0, tokens_all.shape[-1], eos_idx)
    mask = mask & (jnp.arange(tokens_all.shape[-1])[None, :] <= eos_idx[:, None] + mask_size)

    tokens_all_minibatch = ppo_shard(tokens_all)
    advantages_minibatch = ppo_shard(advantages)
    mask_minibatch = ppo_shard(mask)

    # First, we do a forward pass to get prior logprobs for each token.
    logprobs_list = []
    for j in range(global_batch_size // FLAGS.ppo_minibatch):
        logprobs_minibatch = get_logprobs(train_state, tokens_all_minibatch[:, j], mask_minibatch[:, j])
        logprobs_list.append(logprobs_minibatch)
    logprobs_all_minibatch = jnp.stack(logprobs_list, axis=1)

    # Then, the training loop.
    for j in range(global_batch_size // FLAGS.ppo_minibatch):
        train_state, info = update(train_state, tokens_all_minibatch[:, j], mask_minibatch[:, j], advantages_minibatch[:, j], logprobs_all_minibatch[:, j])
        info = jax.device_get(info)
        info['output_tokens'] = eos_idx
        info = jax.tree.map(lambda x: np.array(x), info)
        info = jax.tree.map(lambda x: x.mean(), info)
        info['rollout_iters_per_update'] = num_rollout_iters
        info['global_step'] = i
        info['minibatches_per_global_step'] = global_batch_size // FLAGS.ppo_minibatch
        # info.update(jax.tree.map(lambda x: np.mean(x), env_infos_history))
        for k, v in env_infos_history.items():
            info[k] = np.mean(v)
        if jax.process_index() == 0:
            # rollouts_table.add_data(i, env.render(new_states[0]), returns_local[0])
            rollouts_list.append([i, env.render(new_states[0]), returns_local[0]])
            rollouts_table = wandb.Table(data=rollouts_list, columns=["step", "text", "reward"])
            info['rollouts_table'] = rollouts_table
            print(info)
            wandb.log(info)