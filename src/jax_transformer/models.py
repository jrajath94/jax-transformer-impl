"""Transformer block with Grouped Query Attention, RMSNorm, and SwiGLU FFN.

Follows the architecture pattern in Llama 2 / Gemma:
  - Pre-norm (RMSNorm before both attention and FFN)
  - GQA with configurable KV heads
  - SwiGLU gated FFN
  - Residual connections

The module is implemented as a pure functional JAX module with explicit
parameter trees, making it straightforward to use with optax optimizers
and compatible with pmap for multi-device training.

Note on JIT compilation: GQAConfig is a frozen dataclass, which means
it is hashable and can be passed as a static argument to jax.jit. If you
need to jit transformer_block_forward with config as a captured closure,
use functools.partial or wrap in a lambda. Passing config as a traced
(non-static) argument will cause recompilation on every distinct config.
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jax_transformer.attention import grouped_query_attention

logger = logging.getLogger(__name__)

# Feed-forward expansion multiplier relative to model dimension.
# 4/3 * expansion_factor ≈ 2.67 used in SwiGLU to keep param count equal to
# the standard 4x FFN (Touvron et al., 2023).
FFN_EXPANSION_FACTOR: float = 8.0 / 3.0

# RMSNorm stability epsilon — small enough to not affect outputs at fp32.
RMSN_EPS: float = 1e-6


@dataclass(frozen=True)
class GQAConfig:
    """Configuration for a GQA-based Transformer block.

    Attributes:
        model_dim: Embedding dimension d_model.
        num_heads: Number of query heads H.
        num_kv_heads: Number of key/value heads G (must divide num_heads evenly).
        head_dim: Dimension of each attention head d_k.
        ffn_dim: Hidden dimension of the feed-forward network. If None, computed
            as floor(model_dim * FFN_EXPANSION_FACTOR) rounded to multiple of 256.
        dropout_rate: Dropout applied to attention weights.
        max_seq_len: Maximum sequence length for mask pre-allocation.
    """

    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    ffn_dim: int | None = None
    dropout_rate: float = 0.0
    max_seq_len: int = 2048

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        expected_dim = self.num_heads * self.head_dim
        if expected_dim != self.model_dim:
            logger.warning(
                "num_heads * head_dim = %d != model_dim = %d; "
                "projections will handle the mismatch",
                expected_dim, self.model_dim,
            )

    @property
    def resolved_ffn_dim(self) -> int:
        """FFN hidden dim, rounded up to nearest 256 for XLA efficiency."""
        if self.ffn_dim is not None:
            return self.ffn_dim
        raw = int(self.model_dim * FFN_EXPANSION_FACTOR)
        # Round up to multiple of 256 — XLA matrix ops are most efficient on
        # dimensions that are multiples of the tensor core tile size (128 or 256).
        return ((raw + 255) // 256) * 256


# ---------------------------------------------------------------------------
# Parameter tree helpers
# ---------------------------------------------------------------------------

def _init_linear(
    rng: jax.Array, in_dim: int, out_dim: int
) -> jnp.ndarray:
    """Kaiming uniform initialization for a linear weight matrix.

    Args:
        rng: PRNG key.
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.

    Returns:
        Weight matrix of shape [in_dim, out_dim].
    """
    std = (2.0 / in_dim) ** 0.5
    return jax.random.normal(rng, (in_dim, out_dim)) * std


def init_transformer_block(
    rng: jax.Array, config: GQAConfig
) -> dict:
    """Initialize all learnable parameters for a single TransformerBlock.

    Returns a nested dict that acts as the parameter tree for the block.
    Organized by component so optax can apply per-group LR schedules.

    Args:
        rng: PRNG key for initialization.
        config: Block configuration.

    Returns:
        Dict with keys: attn, ffn, norm_attn, norm_ffn.
    """
    keys = jax.random.split(rng, 10)
    d = config.model_dim
    h = config.num_heads
    g = config.num_kv_heads
    dh = config.head_dim
    df = config.resolved_ffn_dim

    return {
        "attn": {
            # Q projects to full num_heads * head_dim
            "w_q": _init_linear(keys[0], d, h * dh),
            # K, V project to num_kv_heads * head_dim (GQA reduction)
            "w_k": _init_linear(keys[1], d, g * dh),
            "w_v": _init_linear(keys[2], d, g * dh),
            "w_o": _init_linear(keys[3], h * dh, d),
        },
        "ffn": {
            # SwiGLU has two gate projections (w1, w3) and one down projection
            "w1": _init_linear(keys[4], d, df),
            "w3": _init_linear(keys[5], d, df),
            "w2": _init_linear(keys[6], df, d),
        },
        "norm_attn": {
            # RMSNorm has a single learnable scale vector (no bias)
            "scale": jnp.ones((d,)),
        },
        "norm_ffn": {
            "scale": jnp.ones((d,)),
        },
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def rms_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    eps: float = RMSN_EPS,
) -> jnp.ndarray:
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    RMSNorm omits the mean-centering step of LayerNorm, which reduces
    computation and is empirically sufficient for transformers.

    Args:
        x: Input tensor of shape [..., d_model].
        scale: Learnable scale of shape [d_model].
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of the same shape as x.
    """
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

def swiglu_ffn(
    x: jnp.ndarray,
    w1: jnp.ndarray,
    w2: jnp.ndarray,
    w3: jnp.ndarray,
) -> jnp.ndarray:
    """SwiGLU feed-forward network (Shazeer, 2020).

    SwiGLU(x) = (x @ W1) * sigmoid(x @ W1 * beta) @ W2, where the gate
    is computed jointly. It consistently outperforms vanilla ReLU/GELU FFNs
    on language modeling at matched parameter counts.

    Args:
        x: Input of shape [batch, seq_len, d_model].
        w1: Gate+value weight [d_model, ffn_dim].
        w2: Down-projection weight [ffn_dim, d_model].
        w3: Gate weight [d_model, ffn_dim].

    Returns:
        Output of shape [batch, seq_len, d_model].
    """
    gate = jax.nn.silu(x @ w1)
    value = x @ w3
    return (gate * value) @ w2


# ---------------------------------------------------------------------------
# Attention with linear projections
# ---------------------------------------------------------------------------

def _project_qkv(
    x: jnp.ndarray,
    params: dict,
    config: GQAConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project input to query, key, value tensors.

    Args:
        x: Input of shape [batch, seq_len, d_model].
        params: Attention parameter dict with w_q, w_k, w_v.
        config: Block configuration.

    Returns:
        Tuple of (query, key, value), each [batch, seq_len, heads, head_dim].
    """
    batch, seq_len, _ = x.shape

    query = (x @ params["w_q"]).reshape(batch, seq_len, config.num_heads, config.head_dim)
    key = (x @ params["w_k"]).reshape(batch, seq_len, config.num_kv_heads, config.head_dim)
    value = (x @ params["w_v"]).reshape(batch, seq_len, config.num_kv_heads, config.head_dim)
    return query, key, value


def attention_forward(
    x: jnp.ndarray,
    params: dict,
    config: GQAConfig,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """GQA attention sublayer with input/output projections.

    Args:
        x: Input of shape [batch, seq_len, d_model].
        params: Attention parameters (w_q, w_k, w_v, w_o).
        config: Block configuration.
        mask: Optional attention mask.

    Returns:
        Output of shape [batch, seq_len, d_model].
    """
    batch, seq_len, d_model = x.shape
    query, key, value = _project_qkv(x, params, config)
    context = grouped_query_attention(query, key, value, mask=mask)
    context_flat = context.reshape(batch, seq_len, config.num_heads * config.head_dim)
    return context_flat @ params["w_o"]


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

def transformer_block_forward(
    x: jnp.ndarray,
    params: dict,
    config: GQAConfig,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Single pre-norm TransformerBlock forward pass.

    Architecture: Pre-RMSNorm → GQA → Residual → Pre-RMSNorm → SwiGLU FFN → Residual.

    Pre-norm placement (vs. post-norm) stabilizes training at large scale
    because gradients flow cleanly through the residual stream without
    passing through a normalization bottleneck.

    Args:
        x: Input of shape [batch, seq_len, d_model].
        params: Parameter tree from init_transformer_block.
        config: Block configuration.
        mask: Optional causal or padding mask.

    Returns:
        Output of shape [batch, seq_len, d_model].
    """
    # --- Attention sublayer ---
    normed_x = rms_norm(x, params["norm_attn"]["scale"])
    attn_out = attention_forward(normed_x, params["attn"], config, mask=mask)
    x = x + attn_out

    # --- FFN sublayer ---
    normed_x = rms_norm(x, params["norm_ffn"]["scale"])
    ffn_out = swiglu_ffn(
        normed_x,
        params["ffn"]["w1"],
        params["ffn"]["w2"],
        params["ffn"]["w3"],
    )
    return x + ffn_out


# Convenience alias that makes the functional API feel more like a class.
class TransformerBlock:
    """Thin wrapper providing an OOP interface over the functional transformer block.

    Stores config and params together so callers can treat it like a module
    while the underlying implementation remains purely functional (and thus
    fully JIT/vmap/pmap compatible).

    Example:
        config = GQAConfig(model_dim=512, num_heads=8, num_kv_heads=2, head_dim=64)
        block = TransformerBlock.init(jax.random.PRNGKey(0), config)
        out = block(x)
    """

    def __init__(self, params: dict, config: GQAConfig) -> None:
        self.params = params
        self.config = config

    @classmethod
    def init(cls, rng: jax.Array, config: GQAConfig) -> "TransformerBlock":
        """Initialize a TransformerBlock with random parameters.

        Args:
            rng: PRNG key.
            config: Block configuration.

        Returns:
            Initialized TransformerBlock.
        """
        params = init_transformer_block(rng, config)
        return cls(params, config)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input of shape [batch, seq_len, d_model].
            mask: Optional attention mask.

        Returns:
            Output of shape [batch, seq_len, d_model].
        """
        return transformer_block_forward(x, self.params, self.config, mask=mask)
