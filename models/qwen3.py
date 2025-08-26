"""
Model structure for Qwen3. Taken from https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py.
"""

import re
import json
import glob
import functools
from safetensors import safe_open

import jax
import jax.numpy as jnp
import flax

import flax.linen as nn
import jax.numpy as jnp

def rms_norm(x, gamma, eps):
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
    return jnp.astype(gamma * x / rms, jnp.bfloat16)

def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def count_left_padding(ids, pad_id=0):
    return jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1)

def length_minus_padding(token_mask):
    return jnp.sum(jnp.cumsum(jnp.flip(token_mask != 0, -1), axis=-1) > 0, -1)

def get_positions(token_mask):
    """Counts positions for segment ids."""
    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])
    vals = (jnp.zeros_like(token_mask), token_mask)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")

def generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    rope_theta: float,
) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

# @functools.partial(jax.jit, static_argnames=['causal', 'query_block_size', 'key_block_size'])
def tiled_multihead_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_indices: jax.Array,
    k_indices: jax.Array,
    q_pos: jax.Array,
    k_pos: jax.Array,
    causal: bool = True,
    query_block_size: int = 64,
    key_block_size: int = 64
) -> jax.Array:
    MASK_VALUE = -1e30

    # Shapes:
    # q: [B, q_heads, q_len, D]
    # k,v: [B, kv_heads, kv_len, D]
    batch_size, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    original_dtype = q.dtype

    if query_block_size > q_len:
        query_block_size = q_len
    
    if key_block_size > kv_len:
        key_block_size = kv_len

    # Ensure boolean masks
    q_indices = q_indices.astype(bool)
    k_indices = k_indices.astype(bool)

    # Figure out block counts and any padding independently for Q and KV
    num_query_blocks = (q_len + query_block_size - 1) // query_block_size
    padded_q_len = num_query_blocks * query_block_size
    q_padding = padded_q_len - q_len

    num_key_blocks = (kv_len + key_block_size - 1) // key_block_size
    padded_kv_len = num_key_blocks * key_block_size
    kv_padding = padded_kv_len - kv_len

    # Pad Q and metadata if needed
    if q_padding != 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, q_padding), (0, 0)))
        q_indices = jnp.pad(q_indices, ((0, 0), (0, q_padding)), constant_values=False)
        q_pos = jnp.pad(q_pos, ((0, 0), (0, q_padding)), constant_values=-1)

    # Pad K/V and metadata if needed
    if kv_padding != 0:
        k = jnp.pad(k, ((0, 0), (0, 0), (0, kv_padding), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_padding), (0, 0)))
        k_indices = jnp.pad(k_indices, ((0, 0), (0, kv_padding)), constant_values=False)
        k_pos = jnp.pad(k_pos, ((0, 0), (0, kv_padding)), constant_values=-1)

    # Refresh lengths after padding
    q_len = q.shape[2]
    kv_len = k.shape[2]

    # Upcast for compute
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)

    # Expand KV heads to match Q heads
    if num_kv_heads < num_heads:
        assert (num_heads % num_kv_heads) == 0, "q_heads must be a multiple of kv_heads"
        num_repeats = num_heads // num_kv_heads
        k = jnp.repeat(k, repeats=num_repeats, axis=1)
        v = jnp.repeat(v, repeats=num_repeats, axis=1)

    scale = 1.0 / jnp.sqrt(jnp.array(head_size, dtype=jnp.float32))

    # Output and running stats
    o = jnp.zeros_like(q, dtype=jnp.float32)
    m = jnp.full((batch_size, num_heads, q_len), MASK_VALUE, dtype=jnp.float32)
    l = jnp.zeros((batch_size, num_heads, q_len), dtype=jnp.float32)

    def query_loop_body(i_block, state):
        o, m, l = state
        i_start = i_block * query_block_size

        q_i = jax.lax.dynamic_slice_in_dim(q, i_start, query_block_size, axis=2)
        q_indices_i = jax.lax.dynamic_slice_in_dim(q_indices, i_start, query_block_size, axis=1)
        q_pos_i = jax.lax.dynamic_slice_in_dim(q_pos, i_start, query_block_size, axis=1)

        o_i = jnp.zeros_like(q_i)
        m_i = jnp.full((batch_size, num_heads, query_block_size), MASK_VALUE, dtype=jnp.float32)
        l_i = jnp.zeros((batch_size, num_heads, query_block_size), dtype=jnp.float32)

        upper_j = num_key_blocks

        def key_loop_body(j_block, inner_state):
            o_i, m_i, l_i = inner_state
            j_start = j_block * key_block_size

            k_j = jax.lax.dynamic_slice_in_dim(k, j_start, key_block_size, axis=2)
            v_j = jax.lax.dynamic_slice_in_dim(v, j_start, key_block_size, axis=2)
            k_indices_j = jax.lax.dynamic_slice_in_dim(k_indices, j_start, key_block_size, axis=1)
            k_pos_j = jax.lax.dynamic_slice_in_dim(k_pos, j_start, key_block_size, axis=1)

            # [B, H, q_bs, k_bs]
            s_ij = (q_i @ k_j.swapaxes(-2, -1)) * scale

            mask = q_indices_i[:, None, :, None] & k_indices_j[:, None, None, :]
            if causal:
                causal_mask = q_pos_i[:, None, :, None] >= k_pos_j[:, None, None, :]
                mask &= causal_mask

            s_ij = jnp.where(mask, s_ij, MASK_VALUE)

            m_ij = jnp.max(s_ij, axis=-1)                             # [B, H, q_bs]
            m_i_new = jnp.maximum(m_i, m_ij)                          # [B, H, q_bs]

            p_ij = jnp.exp(s_ij - m_i_new[..., None])                 # [B, H, q_bs, k_bs]
            scale_factor = jnp.exp(m_i - m_i_new)                     # [B, H, q_bs]
            l_i_new = scale_factor * l_i + jnp.sum(p_ij, axis=-1)     # [B, H, q_bs]
            l_i_new_safe = jnp.where(l_i_new == 0, 1e-6, l_i_new)

            o_i_rescaled = o_i * (scale_factor * l_i / l_i_new_safe)[..., None]
            v_contribution = (p_ij @ v_j) / l_i_new_safe[..., None]
            o_i_new = o_i_rescaled + v_contribution

            return o_i_new, m_i_new, l_i_new

        o_i_final, m_i_final, l_i_final = jax.lax.fori_loop(
            0, upper_j, key_loop_body, (o_i, m_i, l_i)
        )

        o = jax.lax.dynamic_update_slice_in_dim(o, o_i_final, i_start, axis=2)
        m = jax.lax.dynamic_update_slice_in_dim(m, m_i_final, i_start, axis=2)
        l = jax.lax.dynamic_update_slice_in_dim(l, l_i_final, i_start, axis=2)

        return o, m, l

    final_o, _, _ = jax.lax.fori_loop(0, num_query_blocks, query_loop_body, (o, m, l))

    final_o = final_o[:, :, :((q.shape[2] - q_padding) if q_padding != 0 else q.shape[2]), :]
    initial_q_len = padded_q_len - q_padding
    final_o = final_o[:, :, :initial_q_len, :]

    return final_o.astype(original_dtype)

class KVCache(flax.struct.PyTreeNode):
    k: list[jax.Array]
    v: list[jax.Array]
    length: int
    starts: jax.Array

    @classmethod
    def create(cls, num_layers, batch_size, max_seq_len, head_dim, kv_heads):
        k = [jnp.zeros((batch_size, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16) for _ in range(num_layers)]
        v = [jnp.zeros((batch_size, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16) for _ in range(num_layers)]
        length = 0
        starts = jnp.zeros((batch_size,), dtype=jnp.int32)
        return cls(k=k, v=v, length=length, starts=starts)

    @classmethod
    def get_sharding(cls, cache_shard, none_shard):
        # Yes, the type annotations are wrong, but this works nicely for jax sharding...
        return KVCache(k=cache_shard, v=cache_shard, length=none_shard, starts=cache_shard)

class Block(nn.Module):
    """ A standard transformer block. Has residual connection, self-attention, and a two-layer MLP. """
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    mlp_ffw_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, sin, cos, token_mask, layer_id, cache=None):

        # =========================
        # === Self-Attention Block. 
        # =========================

        pre_gamma = self.param('pre_gamma', nn.initializers.constant(1.0), (self.hidden_size,))
        x_norm = rms_norm(x, pre_gamma, self.eps)
        
        # Calculate Q,K,V.
        q = nn.Dense(self.q_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        q = jnp.reshape(q, (q.shape[0], q.shape[1], self.q_heads, self.head_dim))
        k = nn.Dense(self.kv_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        k = jnp.reshape(k, (k.shape[0], k.shape[1], self.kv_heads, self.head_dim))
        v = nn.Dense(self.kv_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        v = jnp.reshape(v, (v.shape[0], v.shape[1], self.kv_heads, self.head_dim))

        q_gamma = self.param('q_gamma', nn.initializers.constant(1.0), (self.head_dim,))
        q = rms_norm(q, q_gamma, self.eps)
        q = apply_rotary_embedding(q, sin, cos)
        k_gamma = self.param('k_gamma', nn.initializers.constant(1.0), (self.head_dim,))
        k = rms_norm(k, k_gamma, self.eps)
        k = apply_rotary_embedding(k, sin, cos)

        if cache is not None:
            k = jax.lax.dynamic_update_slice_in_dim(cache.k[layer_id], k, cache.length, axis=1)
            v = jax.lax.dynamic_update_slice_in_dim(cache.v[layer_id], v, cache.length, axis=1)
            time_idx = jnp.arange(0, v.shape[1], dtype=jnp.int32)[None, :] # [1, seqlen]
            q_idx = jnp.where(token_mask != 0, 1, 0) # [B, seqlen] where tokens exist.
            incremental_pos = jnp.max(length_minus_padding(token_mask))
            k_idx = (time_idx >= cache.starts[:, None]) & (time_idx < (cache.length + incremental_pos).astype(jnp.int32))
            q_offset = cache.length
        else:
            q_idx, k_idx = token_mask, token_mask
            q_offset = 0

        # # Causal Attention Mask.
        # b, t, qh, d = q.shape # qh = 16
        # _, T, kh, _ = k.shape # kh = 8
        # mask = q_idx[:, :, None] & k_idx[:, None, :]
        # mask = mask[:, None, :, :] # [B, 1, t, T]
        # qk_size = (1, 1, t, T)
        # q_iota = jax.lax.broadcasted_iota(jnp.int32, qk_size, 2)
        # k_iota = jax.lax.broadcasted_iota(jnp.int32, qk_size, 3)
        # q_positions = q_iota + q_offset
        # causal_mask = q_positions >= k_iota
        # mask = jnp.logical_and(mask, causal_mask)
        # mask = jnp.transpose(mask, (0, 2, 3, 1)) # [B, t, T, 1]

        # # Attention.
        # q_ = jnp.reshape(q, (b, t, kh, qh // kh, d))
        # qk = jnp.einsum("bthgd,bThd->btThg", q_, k) * (d ** -0.5)
        # qk = jnp.reshape(qk, (b, t, T, qh)) 
        # qk = jnp.where(mask, qk, -1e30) # good
        # attn = jax.nn.softmax(qk.astype(jnp.float32), axis=2) # on T dimension.
        # attn = jnp.reshape(attn, (b, t, T, kh, qh // kh))
        # qkv = jnp.einsum("btThg,bThd->bthgd", attn, v).astype(x.dtype)
        # qkv = jnp.reshape(qkv, (b, t, qh*d))
        # attn_x = nn.Dense(self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(qkv)
        # x = x + attn_x

        # Tiled attention
        b, t, qh, d = q.shape # qh = 16
        _, T, kh, _ = k.shape # kh = 8
        q_pos = jax.lax.broadcasted_iota(jnp.int32, (b, t), 1)
        q_pos = q_pos + q_offset
        k_pos = jax.lax.broadcasted_iota(jnp.int32, (b, T), 1)

        attn_x = tiled_multihead_attention(
          q.swapaxes(1, 2),
          k.swapaxes(1, 2),
          v.swapaxes(1, 2),
          q_idx,
          k_idx,
          q_pos,
          k_pos,
        )
        attn_x = attn_x.swapaxes(1, 2)
        attn_x = jnp.reshape(attn_x, (attn_x.shape[0], attn_x.shape[1], -1))
        attn_x = nn.Dense(self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(attn_x)
        x = x + attn_x
        
        # =========================
        # === MLP Block. 
        # =========================
        post_gamma = self.param('post_gamma', nn.initializers.constant(1.0), (self.hidden_size,))
        x_norm = rms_norm(x, post_gamma, self.eps)
        g = nn.Dense(features=self.mlp_ffw_size, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        g = nn.silu(g)
        y = nn.Dense(features=self.mlp_ffw_size, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        y = g * y
        mlp_x = nn.Dense(features=self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(y)
        x = x + mlp_x
        return x, k, v

class Qwen3Model(nn.Module):
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    mlp_ffw_size: int
    num_layers: int
    rope_theta: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, token_mask, cache = None):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)(x)
        x = x.astype(jnp.bfloat16)
        positions = get_positions(token_mask)
        if cache is not None:
            start_indices = jnp.where(cache.length != 0, cache.length - cache.starts, 0)
        else:
            start_indices = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        positions = start_indices[:, None] + positions
        sin, cos = generate_pos_embeddings(positions, self.head_dim, self.rope_theta)
        sin, cos = sin.astype(jnp.bfloat16), cos.astype(jnp.bfloat16)
        for layer_id in range(self.num_layers):
            x, k, v = Block(hidden_size=self.hidden_size, q_heads=self.q_heads, kv_heads=self.kv_heads, head_dim=self.head_dim, mlp_ffw_size=self.mlp_ffw_size, eps=self.eps)(x, sin, cos, token_mask, layer_id, cache)
            if cache is not None:
                cache.k[layer_id] = k
                cache.v[layer_id] = v

        gamma_final = self.param('gamma_final', nn.initializers.constant(1.0), (self.hidden_size,))
        x = rms_norm(x, gamma_final, self.eps)
        logits = nn.Dense(self.vocab_size, use_bias=False)(x)

        if cache is not None:
            cache = cache.replace(length=cache.length + jnp.max(length_minus_padding(token_mask)))

        return logits, cache
    
###############################
##### Utils for loading models.
###############################

def create_model_from_hf(hf_dir: str):
    with open(hf_dir + "config.json") as f:
        cfg = json.load(f)
    model = Qwen3Model(
        hidden_size=cfg['hidden_size'],
        q_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],
        num_layers=cfg['num_hidden_layers'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        mlp_ffw_size=cfg['intermediate_size'],
        eps=cfg['rms_norm_eps'],
        rope_theta=cfg['rope_theta']
    )
    tokens = jnp.ones((1,1), dtype=jnp.int32)
    idx = jnp.ones((1,1), dtype=jnp.int32)
    # params = model.init(jax.random.PRNGKey(0), tokens, idx)['params']
    params = jax.eval_shape(model.init, jax.random.PRNGKey(0), tokens, idx)['params']

    _HF_KEY_MAPPING = {
        r"model\.embed_tokens\.weight": "Embed_0.embedding",
        # attention projection weights
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"Block_\1.Dense_0.kernel",
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"Block_\1.Dense_1.kernel",
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"Block_\1.Dense_2.kernel",
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"Block_\1.Dense_3.kernel",
        # norms
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"Block_\1.q_gamma",
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"Block_\1.k_gamma",
        # layer norms (pre/post attention)
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"Block_\1.pre_gamma",
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1.post_gamma",
        # mlp
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"Block_\1.Dense_4.kernel",
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"Block_\1.Dense_5.kernel",
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"Block_\1.Dense_6.kernel",
        r"model\.norm\.weight": "gamma_final",
        r"lm_head\.weight": "Dense_0.kernel",
    }

    def _torch_key_to_jax_key(source_key, custom_key_map: dict[str, str] | None = None):
        key_maps = dict(_HF_KEY_MAPPING, **(dict() if custom_key_map is None else custom_key_map))
        subs = [re.sub(pat, repl, source_key) for pat, repl in key_maps.items() if re.match(pat, source_key)]
        if len(subs) > 1:
            raise ValueError(f"More than 1 key matched: {subs}")
        else:
            return None if len(subs) == 0 else subs[0]

    torch_params = {}
    files = list(glob.glob(hf_dir+"*safetensors"))
    for file in files:
        with safe_open(file, framework="torch") as f:
            for key in f.keys():
                torch_params[key] = f.get_tensor(key)
                jax_key = _torch_key_to_jax_key(key)
                jax_key_list = jax_key.split('.')
                jax_param = params
                while len(jax_key_list) > 0:
                    jax_key = jax_key_list.pop(0)
                    if len(jax_key_list) == 0:
                        if 'kernel' in jax_key:
                            new_param = torch_params[key].float().T.numpy()
                            # new_param = jnp.array(torch_params[key].float()).T
                            # new_param = jax.device_put(torch_params[key].float(), device=jax.devices("cpu")[0]).T
                        else:
                            new_param = torch_params[key].float().numpy()
                            # new_param = jnp.array(torch_params[key].float())
                            # new_param = jax.device_put(torch_params[key].float(), device=jax.devices("cpu")[0]).T
                        assert new_param.shape == jax_param[jax_key].shape
                        jax_param[jax_key] = new_param
                    jax_param = jax_param[jax_key]
    
    return model, params

def create_model_from_ckpt(ckpt_dir: str):
    from lmpo.utils.checkpoint import Checkpoint
    with open(ckpt_dir + "config.json") as f:
        cfg = json.load(f)
    model = Qwen3Model(
        hidden_size=cfg['hidden_size'],
        q_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],
        num_layers=cfg['num_hidden_layers'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        mlp_ffw_size=cfg['intermediate_size'],
        eps=cfg['rms_norm_eps'],
        rope_theta=cfg['rope_theta']
    )
    ckpt = Checkpoint(ckpt_dir + "params.pkl", parallel=False)
    params = ckpt.load_as_dict()['params']
        
    return model, params