"""Multi-head attention in JAX.

Implements standard multi-head attention (Vaswani et al., 2017).
MQA and GQA variants are planned for next iteration.

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

    Computes softmax((Q @ K^T) / sqrt(d_k)) @ V. The scaling by 1/sqrt(d_k)
    keeps dot-product magnitudes stable regardless of head dimension.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim].
        key: Key tensor of shape [batch, kv_seq_len, num_heads, head_dim].
        value: Value tensor of shape [batch, kv_seq_len, num_heads, head_dim].
        mask: Optional boolean mask. True means attend, False means mask.
        dropout_rate: Attention weight dropout probability.
        dropout_rng: JAX PRNG key required when dropout_rate > 0.

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

    Each query head attends over its own dedicated key/value head.
    This is the most memory-intensive variant.

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
