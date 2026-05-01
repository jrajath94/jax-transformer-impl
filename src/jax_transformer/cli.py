"""Command-line interface for benchmarking GQA configurations.

Usage:
    python -m jax_transformer.cli benchmark --num-heads 8 --num-kv-heads 2
    python -m jax_transformer.cli profile --model-dim 512 --seq-len 1024
"""

import argparse
import logging
import sys
import time

import jax
import jax.numpy as jnp

from jax_transformer.attention import grouped_query_attention, multi_head_attention
from jax_transformer.models import GQAConfig, init_transformer_block
from jax_transformer.utils import count_parameters, kv_cache_memory_bytes, xla_compilation_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Number of warmup iterations discarded before timing.
WARMUP_ITERS: int = 3

# Number of timed iterations for latency measurement.
BENCH_ITERS: int = 10


def _timed_run(fn, *args) -> float:
    """Run fn(*args) BENCH_ITERS times and return mean latency in ms.

    Warms up first to ensure JIT compilation is not included in the timing.

    Args:
        fn: Callable to benchmark.
        *args: Arguments passed to fn on every call.

    Returns:
        Mean wall-clock latency in milliseconds.
    """
    for _ in range(WARMUP_ITERS):
        result = fn(*args)
        result.block_until_ready()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        result = fn(*args)
        result.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1000 / BENCH_ITERS
    return elapsed_ms


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run GQA vs MHA latency and memory comparison.

    Args:
        args: Parsed CLI arguments.
    """
    batch_size: int = args.batch_size
    seq_len: int = args.seq_len
    num_heads: int = args.num_heads
    num_kv_heads: int = args.num_kv_heads
    head_dim: int = args.head_dim
    num_layers: int = args.num_layers

    rng = jax.random.PRNGKey(0)

    query = jax.random.normal(rng, (batch_size, seq_len, num_heads, head_dim))
    key_gqa = jax.random.normal(rng, (batch_size, seq_len, num_kv_heads, head_dim))
    value_gqa = jax.random.normal(rng, (batch_size, seq_len, num_kv_heads, head_dim))
    key_mha = jax.random.normal(rng, (batch_size, seq_len, num_heads, head_dim))
    value_mha = jax.random.normal(rng, (batch_size, seq_len, num_heads, head_dim))

    gqa_jit = jax.jit(grouped_query_attention)
    mha_jit = jax.jit(multi_head_attention)

    logger.info("Benchmarking GQA vs MHA ...")
    gqa_ms = _timed_run(gqa_jit, query, key_gqa, value_gqa)
    mha_ms = _timed_run(mha_jit, query, key_mha, value_mha)

    gqa_kv_bytes = kv_cache_memory_bytes(batch_size, seq_len, num_kv_heads, head_dim, num_layers)
    mha_kv_bytes = kv_cache_memory_bytes(batch_size, seq_len, num_heads, head_dim, num_layers)
    reduction = mha_kv_bytes / gqa_kv_bytes if gqa_kv_bytes else 0

    print("\n" + "=" * 60)
    print(f"  Benchmark: GQA (H={num_heads}, G={num_kv_heads}) vs MHA (H={num_heads})")
    print(f"  Config: batch={batch_size} seq={seq_len} head_dim={head_dim} layers={num_layers}")
    print("=" * 60)
    print(f"  {'Metric':<30} {'GQA':>10} {'MHA':>10} {'Speedup':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Latency (ms)':<30} {gqa_ms:>10.2f} {mha_ms:>10.2f} {mha_ms/gqa_ms:>9.2f}x")
    print(f"  {'KV cache (MB)':<30} {gqa_kv_bytes/1e6:>10.2f} {mha_kv_bytes/1e6:>10.2f} {reduction:>9.2f}x")
    print("=" * 60 + "\n")


def cmd_profile(args: argparse.Namespace) -> None:
    """Print XLA compilation profile and parameter count for a TransformerBlock.

    Args:
        args: Parsed CLI arguments.
    """
    config = GQAConfig(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    rng = jax.random.PRNGKey(0)
    params = init_transformer_block(rng, config)

    param_count = count_parameters(params)
    logger.info("Parameter count: %d (%.2fM)", param_count, param_count / 1e6)

    from jax_transformer.models import transformer_block_forward
    sample_x = jnp.zeros((1, args.seq_len, config.model_dim))
    xla_compilation_profile(
        lambda x: transformer_block_forward(x, params, config),
        sample_x,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="jax-transformer",
        description="JAX GQA Transformer — benchmark and profile tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- benchmark subcommand ---
    bench = subparsers.add_parser("benchmark", help="Run GQA vs MHA latency/memory benchmark")
    bench.add_argument("--batch-size", type=int, default=4)
    bench.add_argument("--seq-len", type=int, default=512)
    bench.add_argument("--num-heads", type=int, default=8)
    bench.add_argument("--num-kv-heads", type=int, default=2)
    bench.add_argument("--head-dim", type=int, default=64)
    bench.add_argument("--num-layers", type=int, default=32)

    # --- profile subcommand ---
    prof = subparsers.add_parser("profile", help="Print XLA compilation profile")
    prof.add_argument("--model-dim", type=int, default=512)
    prof.add_argument("--num-heads", type=int, default=8)
    prof.add_argument("--num-kv-heads", type=int, default=2)
    prof.add_argument("--head-dim", type=int, default=64)
    prof.add_argument("--seq-len", type=int, default=256)

    return parser


def main() -> None:
    """Entry point for the jax-transformer CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "profile":
        cmd_profile(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
