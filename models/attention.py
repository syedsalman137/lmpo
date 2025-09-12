import jax
import jax.numpy as jnp

from lmpo.kernels.flash_attention import flash_attention, BlockSizes

MASK_VALUE = 1e-30

def naive_multihead_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_offsets: jax.Array = None,
    q_idx: jax.Array = None,
    k_idx: jax.Array = None,
    causal: bool = True,
) -> jax.Array:
  batch_size, num_heads, num_tokens, head_size = q.shape
  _, num_kv_heads, num_kv_tokens, _ = k.shape

  if num_kv_heads < num_heads:
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    num_repeats = num_heads // num_kv_heads
    k = jnp.repeat(k, repeats=num_repeats, axis=1)
    v = jnp.repeat(v, repeats=num_repeats, axis=1)

  scores = q @ k.swapaxes(-2, -1)
  scaled_scores = scores / jnp.sqrt(head_size)
  mask = None
  if q_idx is not None and k_idx is not None:
    mask = q_idx[:, :, None] & k_idx[:, None, :]
    mask = mask.astype(bool)

  if q_offsets is None:
    q_offsets = jnp.zeros((batch_size, ), dtype=jnp.int32)

  if causal:
    mask_shape  = (batch_size, 1, num_tokens, num_kv_tokens)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 2) + q_offsets[:, None, None, None]
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 3)
    causal_mask = col_ids <= row_ids
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask).astype(bool)

  row_has_any = jnp.ones((batch_size, 1, num_tokens), dtype=bool)
  if mask is not None:
    row_has_any = jnp.any(mask, axis=-1)
    scaled_scores = jnp.where(mask, scaled_scores, MASK_VALUE)

  weights = jnp.where(row_has_any[..., None], jax.nn.softmax(scaled_scores, axis=-1), 0.0)
  attention_output = weights @ v

  return attention_output

def pallas_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_offsets: jax.Array=None,
    q_idx: jax.Array=None,
    k_idx: jax.Array=None,
    sm_scale: float=1.0/128.0 ** 0.5,
    causal: bool=True,
    debug: bool=False
) -> jax.Array:
    b, num_heads, t, d = q.shape
    _, num_kv_heads, T, _ = k.shape

    MIN_BLOCK_Q = 8
    MIN_BLOCK_K = 128

    MAX_BLOCK_Q = MAX_BLOCK_K = 512

    q_padding = 0
    kv_padding = 0

    if (t % MIN_BLOCK_Q != 0 and t < MAX_BLOCK_Q) or (t % MAX_BLOCK_Q != 0 and t > MAX_BLOCK_Q):
        q_padding = -(-t // MIN_BLOCK_Q) * MIN_BLOCK_Q - t if t < MAX_BLOCK_Q else \
        -(-t // MAX_BLOCK_Q) * MAX_BLOCK_Q - t
        q = jnp.pad(q, ((0, 0), (0, 0), (0, q_padding), (0, 0)))
        if q_idx is not None:
            q_idx = jnp.pad(q_idx, ((0, 0), (0, q_padding)), constant_values=False)

    if (T % MIN_BLOCK_K != 0 and T < MAX_BLOCK_K) or (T % MAX_BLOCK_K != 0 and T > MAX_BLOCK_K):
        kv_padding = -(-T // MIN_BLOCK_K) * MIN_BLOCK_K - T if T <= MAX_BLOCK_K else \
        -(-T // MAX_BLOCK_K) * MAX_BLOCK_K - T
        k = jnp.pad(k, ((0, 0), (0, 0), (0, kv_padding), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_padding), (0, 0)))
        if k_idx is not None:
            k_idx = jnp.pad(k_idx, ((0, 0), (0, kv_padding)), constant_values=False)

    block_q = min(t + q_padding, MAX_BLOCK_Q)
    block_k = min(T + kv_padding, MAX_BLOCK_K)
    block_b = 1

    block_sizes = BlockSizes(
        block_q=block_q,
        block_k_major=block_k,
        block_k=block_k,
        block_b=block_b,
        block_q_major_dkv=block_q,
        block_k_major_dkv=block_k,
        block_k_dkv=block_k,
        block_q_dkv=block_q,
        block_k_major_dq=block_k,
        block_k_dq=block_k,
        block_q_dq=block_q,
    )
    result = flash_attention(
        q, k, v,
        q_offsets=q_offsets, q_idx=q_idx, k_idx=k_idx,
        sm_scale=sm_scale, causal=causal, block_sizes=block_sizes,
        debug=debug,
    )

    if q_padding != 0:
        result = result[:, :, :t, :]

    return result

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