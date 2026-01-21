"""GQA vs MHA memory and latency benchmarks.

Measures:
  - Forward-pass latency: JAX GQA vs JAX MHA vs PyTorch MHA (if available)
  - KV cache memory footprint at Llama-2-70B-like configurations
  - JIT compilation overhead (first call vs subsequent calls)

Run: python benchmarks/bench_gqa.py
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_transformer.attention import grouped_query_attention, multi_head_attention
from jax_transformer.utils import kv_cache_memory_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Number of calls to discard for JIT warmup.
WARMUP_ITERS: int = 5

# Number of timed iterations.
BENCH_ITERS: int = 20

# Llama-2 70B production configuration (for memory comparison).
LLAMA2_70B_HEADS: int = 64
LLAMA2_70B_KV_HEADS: int = 8
LLAMA2_70B_HEAD_DIM: int = 128
LLAMA2_70B_LAYERS: int = 80


@dataclass
class BenchResult:
    """Single benchmark measurement.

    Attributes:
        config: Human-readable config label.
        latency_ms: Mean forward-pass latency in milliseconds.
        kv_cache_mb: KV cache size in megabytes.
        kv_reduction_vs_mha: Ratio of MHA KV cache to this config's KV cache.
    """

    config: str
    latency_ms: float
    kv_cache_mb: float
    kv_reduction_vs_mha: float


def _warmup_and_time(fn, *args) -> Tuple[float, float]:
    """Warmup then measure mean latency.

    Args:
        fn: JIT-compiled callable.
        *args: Arguments forwarded to fn on every call.

    Returns:
        Tuple of (first_call_ms, steady_state_ms).
    """
    # First call triggers JIT compilation.
    t0 = time.perf_counter()
    result = fn(*args)
    result.block_until_ready()
    first_call_ms = (time.perf_counter() - t0) * 1000

    # Warmup to fill any remaining compilation caches.
    for _ in range(WARMUP_ITERS - 1):
        fn(*args).block_until_ready()

    # Timed iterations.
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        fn(*args).block_until_ready()
    steady_ms = (time.perf_counter() - t0) * 1000 / BENCH_ITERS

    return first_call_ms, steady_ms


def run_latency_benchmarks(
    batch_size: int = 4,
    seq_len: int = 512,
    head_dim: int = 64,
) -> List[BenchResult]:
    """Compare GQA and MHA latency across head configurations.

    Args:
        batch_size: Batch size for forward pass.
        seq_len: Sequence length.
        head_dim: Dimension of each attention head.

    Returns:
        List of BenchResult entries, one per configuration.
    """
    num_heads = 8
    configs: List[Tuple[str, int]] = [
        ("MHA (H=8, G=8)", 8),
        ("GQA (H=8, G=4)", 4),
        ("GQA (H=8, G=2)", 2),
        ("MQA (H=8, G=1)", 1),
    ]

    rng = jax.random.PRNGKey(0)
    results: List[BenchResult] = []

    # Pre-compile MHA to get its KV baseline for reduction ratio.
    mha_kv_mb = kv_cache_memory_bytes(batch_size, seq_len, num_heads, head_dim, 32) / 1e6

    for label, num_kv_heads in configs:
        keys = jax.random.split(rng, 3)
        query = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        key = jax.random.normal(keys[1], (batch_size, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(keys[2], (batch_size, seq_len, num_kv_heads, head_dim))

        if num_kv_heads == num_heads:
            fn = jax.jit(multi_head_attention)
        else:
            fn = jax.jit(grouped_query_attention)

        first_ms, steady_ms = _warmup_and_time(fn, query, key, value)
        logger.info("%s: JIT compile=%.1fms, steady=%.2fms", label, first_ms, steady_ms)

        kv_mb = kv_cache_memory_bytes(batch_size, seq_len, num_kv_heads, head_dim, 32) / 1e6
        reduction = mha_kv_mb / kv_mb if kv_mb > 0 else 1.0

        results.append(BenchResult(
            config=label,
            latency_ms=steady_ms,
            kv_cache_mb=kv_mb,
            kv_reduction_vs_mha=reduction,
        ))

    return results


def run_production_memory_benchmark() -> None:
    """Show KV cache savings at Llama-2-70B scale.

    The production workload that motivated GQA: at 70B parameters,
    MHA requires enormous KV cache for long-context inference.
    """
    logger.info("=" * 60)
    logger.info("Llama-2 70B scale KV cache comparison")
    logger.info("batch=32, seq=4096, layers=%d", LLAMA2_70B_LAYERS)
    logger.info("=" * 60)

    batch, seq_len = 32, 4096
    num_layers = LLAMA2_70B_LAYERS

    configs = [
        ("MHA (H=64)", LLAMA2_70B_HEADS),
        ("GQA like Llama-2-70B (G=8)", LLAMA2_70B_KV_HEADS),
        ("MQA (G=1)", 1),
    ]

    for label, kv_heads in configs:
        mem_bytes = kv_cache_memory_bytes(
            batch, seq_len, kv_heads, LLAMA2_70B_HEAD_DIM, num_layers
        )
        logger.info("  %-35s  %8.2f GB", label, mem_bytes / 1e9)


def print_results_table(results: List[BenchResult]) -> None:
    """Print formatted benchmark results.

    Args:
        results: List of BenchResult to display.
    """
    print("\n" + "=" * 75)
    print(f"  {'Config':<28} {'Latency (ms)':>14} {'KV Cache (MB)':>14} {'KV Reduction':>12}")
    print(f"  {'-' * 71}")
    for r in results:
        print(
            f"  {r.config:<28} {r.latency_ms:>14.3f} "
            f"{r.kv_cache_mb:>14.2f} {r.kv_reduction_vs_mha:>11.1f}x"
        )
    print("=" * 75 + "\n")


def main() -> None:
    """Entry point for the benchmark script."""
    logger.info("JAX backend: %s", jax.default_backend())
    logger.info("JAX devices: %s", jax.devices())

    logger.info("Running latency benchmarks (batch=4, seq=512, head_dim=64) ...")
    results = run_latency_benchmarks(batch_size=4, seq_len=512, head_dim=64)
    print_results_table(results)

    run_production_memory_benchmark()


if __name__ == "__main__":
    main()
