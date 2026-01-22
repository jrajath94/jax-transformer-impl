"""Quickstart: Run GQA forward pass and verify it matches MHA when G=H.

This script demonstrates:
  1. Creating GQA tensors with 8 Q heads and 2 KV heads
  2. Running grouped_query_attention with JIT compilation
  3. Verifying GQA == MHA when num_kv_heads == num_heads
  4. Running a TransformerBlock forward + backward pass
  5. Printing KV cache memory comparison

Run: python examples/quickstart.py
"""

import logging

import jax
import jax.numpy as jnp

from jax_transformer.attention import grouped_query_attention, multi_head_attention
from jax_transformer.models import GQAConfig, TransformerBlock
from jax_transformer.utils import kv_cache_memory_bytes, make_causal_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def demo_gqa_forward() -> None:
    """Run GQA forward pass and print output stats."""
    logger.info("─── GQA Forward Pass ───")
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 3)

    batch, seq_len = 2, 32
    num_heads, num_kv_heads, head_dim = 8, 2, 64

    query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
    key = jax.random.normal(keys[1], (batch, seq_len, num_kv_heads, head_dim))
    value = jax.random.normal(keys[2], (batch, seq_len, num_kv_heads, head_dim))

    # JIT compile — XLA traces the computation graph on first call.
    gqa_jit = jax.jit(grouped_query_attention)
    output = gqa_jit(query, key, value)

    logger.info("  Input  Q: %s  K: %s  V: %s", query.shape, key.shape, value.shape)
    logger.info("  Output  : %s  (same as Q shape — correct)", output.shape)
    logger.info("  Output mean=%.4f  std=%.4f", float(jnp.mean(output)), float(jnp.std(output)))


def demo_gqa_equals_mha() -> None:
    """Verify GQA(H=G) == MHA — the fundamental correctness invariant."""
    logger.info("─── GQA == MHA When G == H ───")
    rng = jax.random.PRNGKey(1)
    keys = jax.random.split(rng, 3)

    batch, seq_len, num_heads, head_dim = 2, 16, 8, 32
    query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
    key = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim))
    value = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim))

    gqa_out = grouped_query_attention(query, key, value)  # G == H == 8
    mha_out = multi_head_attention(query, key, value)

    max_diff = float(jnp.max(jnp.abs(gqa_out - mha_out)))
    logger.info("  Max diff GQA(G=H) vs MHA: %.2e  (should be < 1e-5)", max_diff)
    assert max_diff < 1e-5, f"GQA and MHA diverge: max_diff={max_diff}"
    logger.info("  PASSED: GQA is a strict generalization of MHA")


def demo_transformer_block() -> None:
    """Run a TransformerBlock forward pass and compute gradients."""
    logger.info("─── TransformerBlock Forward + Backward ───")
    rng = jax.random.PRNGKey(2)

    config = GQAConfig(
        model_dim=256,
        num_heads=8,
        num_kv_heads=2,
        head_dim=32,
    )
    batch, seq_len = 2, 64
    x = jax.random.normal(rng, (batch, seq_len, config.model_dim)) * 0.1
    mask = make_causal_mask(seq_len)

    block = TransformerBlock.init(rng, config)
    out = block(x, mask=mask)
    logger.info("  Input : %s  Output: %s", x.shape, out.shape)

    # Compute gradients through the block.
    from jax_transformer.models import transformer_block_forward

    def loss(inp: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(transformer_block_forward(inp, block.params, config, mask=mask) ** 2)

    grads = jax.grad(loss)(x)
    grad_norm = float(jnp.sqrt(jnp.sum(grads ** 2)))
    logger.info("  Gradient norm: %.4f  (finite: %s)", grad_norm, jnp.all(jnp.isfinite(grads)))


def demo_kv_cache_memory() -> None:
    """Show KV cache savings for realistic model configurations."""
    logger.info("─── KV Cache Memory Comparison (32 layers, bs=8, seq=2048) ───")
    batch, seq_len, head_dim, layers = 8, 2048, 128, 32

    configs = [
        ("MHA (H=32, G=32)", 32, 32),
        ("GQA (H=32, G=8)",  32, 8),
        ("GQA (H=32, G=4)",  32, 4),
        ("MQA (H=32, G=1)",  32, 1),
    ]
    mha_bytes = kv_cache_memory_bytes(batch, seq_len, 32, head_dim, layers)

    for label, _num_heads, kv_heads in configs:
        mem = kv_cache_memory_bytes(batch, seq_len, kv_heads, head_dim, layers)
        reduction = mha_bytes / mem
        logger.info("  %-25s  %6.1f MB  (%4.1fx reduction vs MHA)",
                    label, mem / 1e6, reduction)


def main() -> None:
    """Run all quickstart demos."""
    logger.info("JAX version: %s  |  Backend: %s", jax.__version__, jax.default_backend())
    print()
    demo_gqa_forward()
    print()
    demo_gqa_equals_mha()
    print()
    demo_transformer_block()
    print()
    demo_kv_cache_memory()
    print()
    logger.info("All demos passed. See benchmarks/bench_gqa.py for full perf numbers.")


if __name__ == "__main__":
    main()
