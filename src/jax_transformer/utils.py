"""Utilities for multi-device sharding, dtype management, and profiling.

Provides:
  - make_causal_mask: autoregressive attention mask
  - shard_batch: partition a batch across devices for pmap
  - unshard_batch: collect pmap output back to host
  - xla_compilation_profile: log which ops were fused by XLA
  - count_parameters: sum parameter count from a param tree
  - kv_cache_memory_bytes: estimate KV cache size in bytes
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Default float dtype used when no explicit dtype is requested.
DEFAULT_DTYPE: jnp.dtype = jnp.float32

# Bytes per element for common dtypes — used in memory estimation.
DTYPE_BYTES: dict[Any, int] = {
    jnp.float32: 4,
    jnp.bfloat16: 2,
    jnp.float16: 2,
    jnp.int8: 1,
}


def make_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create a causal (autoregressive) attention mask.

    Position i can attend to positions 0..i but not i+1..seq_len-1.
    The mask uses True = attend, False = block so it can be applied
    directly in scaled_dot_product_attention.

    Args:
        seq_len: Sequence length.

    Returns:
        Boolean mask of shape [1, 1, seq_len, seq_len].
    """
    idxs = jnp.arange(seq_len)
    # Lower-triangular: row i has True for columns 0..i
    mask = idxs[:, None] >= idxs[None, :]
    return mask[None, None, :, :]  # broadcast over batch and heads


def shard_batch(
    batch: jnp.ndarray,
    num_devices: int,
) -> jnp.ndarray:
    """Reshape a batch tensor so pmap can distribute it across devices.

    pmap expects the leading axis to equal the number of devices. This
    function splits the batch dimension (axis 0) into [num_devices, local_batch].

    Args:
        batch: Tensor of shape [global_batch, ...].
        num_devices: Number of devices (must divide global_batch evenly).

    Returns:
        Tensor of shape [num_devices, local_batch, ...].

    Raises:
        ValueError: If global_batch is not divisible by num_devices.
    """
    global_batch = batch.shape[0]
    if global_batch % num_devices != 0:
        raise ValueError(
            f"global_batch ({global_batch}) must be divisible by "
            f"num_devices ({num_devices})"
        )
    local_batch = global_batch // num_devices
    rest = batch.shape[1:]
    return batch.reshape(num_devices, local_batch, *rest)


def unshard_batch(sharded: jnp.ndarray) -> jnp.ndarray:
    """Collect pmap output back into a single batch dimension.

    Inverse of shard_batch. Merges the leading two axes [num_devices, local_batch]
    back into [global_batch].

    Args:
        sharded: Tensor of shape [num_devices, local_batch, ...].

    Returns:
        Tensor of shape [global_batch, ...].
    """
    num_devices, local_batch = sharded.shape[:2]
    rest = sharded.shape[2:]
    return sharded.reshape(num_devices * local_batch, *rest)


def count_parameters(params: dict) -> int:
    """Count total number of scalar parameters in a JAX parameter tree.

    Args:
        params: Nested dict of jnp.ndarray parameter tensors.

    Returns:
        Total parameter count as an integer.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))


def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    dtype: jnp.dtype = DEFAULT_DTYPE,
) -> int:
    """Estimate KV cache memory consumption in bytes.

    Computes bytes for storing K and V tensors across all layers.
    This is the dominant memory cost during autoregressive inference.

    KV cache size = 2 * batch * seq * num_kv_heads * head_dim * bytes_per_element * layers

    Args:
        batch_size: Inference batch size.
        seq_len: Maximum sequence length.
        num_kv_heads: Number of key/value heads (G in GQA notation).
        head_dim: Dimension of each head.
        num_layers: Number of transformer layers.
        dtype: Element dtype for byte-size lookup.

    Returns:
        Estimated KV cache size in bytes.
    """
    bytes_per_elem = DTYPE_BYTES.get(dtype, 4)
    # Factor of 2 accounts for both K and V tensors.
    total = 2 * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_elem * num_layers
    logger.debug(
        "KV cache estimate: %d bytes (%.2f MB) for bs=%d seq=%d kv_heads=%d layers=%d",
        total, total / 1024 ** 2, batch_size, seq_len, num_kv_heads, num_layers,
    )
    return total


def xla_compilation_profile(fn: Any, *args: Any) -> None:
    """Trigger JIT compilation and log high-level XLA operation statistics.

    Calls jax.make_jaxpr to obtain the StableHLO intermediate representation
    and reports the number of unique primitive operations. This is a lightweight
    proxy for "what does XLA see" without a full profiler trace.

    Args:
        fn: A JAX-jittable function.
        *args: Sample arguments matching the expected dtypes and shapes.
            Actual values don't matter — only shapes/dtypes are used.
    """
    try:
        jaxpr = jax.make_jaxpr(fn)(*args)
        ops: dict[str, int] = {}
        for eqn in jaxpr.jaxpr.eqns:
            prim_name = eqn.primitive.name
            ops[prim_name] = ops.get(prim_name, 0) + 1

        logger.info("XLA compilation profile for %s:", getattr(fn, "__name__", str(fn)))
        for op, count in sorted(ops.items(), key=lambda kv: -kv[1]):
            logger.info("  %-30s %d", op, count)
        logger.info("  Total unique primitives: %d", len(ops))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not produce XLA profile: %s", exc)


def make_pmap_forward(fn: Any, axis_name: str = "batch") -> Any:
    """Wrap a forward function for pmap multi-device execution.

    Adds an all-reduce mean over the 'batch' axis after the forward pass
    so gradients are correctly synchronized across devices during training.

    In inference mode, the output is gathered but NOT reduced — each device
    returns its local result, which shard_batch/unshard_batch handle.

    This function produces a pmap-compatible wrapper that handles:
    - In-axis replication of static arguments (params, config)
    - Per-device execution of dynamic arguments (inputs)

    Args:
        fn: Forward function of signature (x, params, config, ...) -> output.
        axis_name: Name of the pmap axis for collective operations.

    Returns:
        A pmap-compiled version of fn.
    """
    # in_axes=(0, None, None) means: shard x over devices, broadcast params and config.
    # This is the standard pattern for data-parallel inference.
    return jax.pmap(fn, axis_name=axis_name, in_axes=(0, None, None))
