"""Multi-head and multi-query attention in JAX.

Implements MHA (Vaswani et al., 2017) and MQA (Shazeer, 2019).
GQA variant coming next.

All functions are functional (no side effects) and JIT-compilable.
Shapes follow the convention [batch, seq_len, num_heads, head_dim].
"""

import logging
import math
from typing import Optional

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

NEG_INF: float = -1e9


def scaled_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    dropout_rate: float = 0.0,
    dropout_rng: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Scaled dot-product attention (Vaswani et al., 2017).

    Args:
        query: Shape [batch, seq_len, num_heads, head_dim].
        key: Shape [batch, kv_seq_len, num_heads, head_dim].
        value: Shape [batch, kv_seq_len, num_heads, head_dim].
        mask: Optional boolean mask. True = attend.
        dropout_rate: Attention dropout probability.
        dropout_rng: PRNG key for dropout.

    Returns:
        Context tensor of shape [batch, seq_len, num_heads, head_dim].

    Raises:
        ValueError: If dropout_rate > 0 and dropout_rng is None.
    """
    if dropout_rate > 0.0 and dropout_rng is None:
        raise ValueError("dropout_rng must be provided when dropout_rate > 0")

    head_dim: int = query.shape[-1]
    scale: float = 1.0 / math.sqrt(head_dim)

    query_t = jnp.einsum("bshd->bhsd", query)
    key_t = jnp.einsum("bshd->bhsd", key)
    value_t = jnp.einsum("bshd->bhsd", value)

    attn_logits = jnp.einsum("bhqd,bhkd->bhqk", query_t, key_t) * scale

    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, NEG_INF)

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    if dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        keep_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        attn_weights = jnp.where(keep_mask, attn_weights / keep_prob, 0.0)

    context = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value_t)
    return jnp.einsum("bhsd->bshd", context)


def multi_head_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    dropout_rate: float = 0.0,
    dropout_rng: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Standard multi-head attention (MHA).

    Args:
        query: Shape [batch, seq_len, num_heads, head_dim].
        key: Shape [batch, kv_seq_len, num_heads, head_dim].
        value: Shape [batch, kv_seq_len, num_heads, head_dim].
        mask: Optional attention mask, True = attend.
        dropout_rate: Attention dropout probability.
        dropout_rng: PRNG key for dropout.

    Returns:
        Context tensor of shape [batch, seq_len, num_heads, head_dim].
    """
    logger.debug("MHA: q=%s k=%s v=%s", query.shape, key.shape, value.shape)
    return scaled_dot_product_attention(
        query, key, value, mask=mask,
        dropout_rate=dropout_rate, dropout_rng=dropout_rng,
    )


def multi_query_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Multi-query attention (MQA) — Shazeer (2019).

    A single KV head is shared across all query heads. MQA is the
    extreme case of GQA where num_kv_heads == 1.

    Args:
        query: Shape [batch, seq_len, num_heads, head_dim].
        key: Shape [batch, kv_seq_len, 1, head_dim].
        value: Shape [batch, kv_seq_len, 1, head_dim].
        mask: Optional attention mask.

    Returns:
        Context tensor of shape [batch, seq_len, num_heads, head_dim].
    """
    num_heads: int = query.shape[2]
    logger.debug("MQA: q=%s k=%s, broadcasting 1 KV head to %d Q heads",
                 query.shape, key.shape, num_heads)
    key_expanded = jnp.repeat(key, num_heads, axis=2)
    value_expanded = jnp.repeat(value, num_heads, axis=2)
    return scaled_dot_product_attention(query, key_expanded, value_expanded, mask=mask)
