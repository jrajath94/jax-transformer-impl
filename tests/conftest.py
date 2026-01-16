"""Shared pytest fixtures for jax-transformer-impl test suite."""

import pytest
import jax
import jax.numpy as jnp

from jax_transformer.models import GQAConfig


# ---------------------------------------------------------------------------
# PRNG fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> jax.Array:
    """Session-scoped PRNG key — JAX is deterministic given the same key."""
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> GQAConfig:
    """Minimal GQAConfig suitable for fast unit tests."""
    return GQAConfig(
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=32,
    )


@pytest.fixture
def standard_config() -> GQAConfig:
    """Standard GQAConfig resembling a small production model."""
    return GQAConfig(
        model_dim=512,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        max_seq_len=256,
    )


@pytest.fixture
def mha_inputs(rng: jax.Array):
    """Standard MHA input tensors (Q=K=V heads = 8)."""
    batch, seq_len, num_heads, head_dim = 2, 16, 8, 32
    keys = jax.random.split(rng, 3)
    query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
    key = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim))
    value = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim))
    return query, key, value


@pytest.fixture
def gqa_inputs(rng: jax.Array):
    """GQA input tensors with 8 Q heads and 2 KV heads."""
    batch, seq_len = 2, 16
    num_heads, num_kv_heads, head_dim = 8, 2, 32
    keys = jax.random.split(rng, 3)
    query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
    key = jax.random.normal(keys[1], (batch, seq_len, num_kv_heads, head_dim))
    value = jax.random.normal(keys[2], (batch, seq_len, num_kv_heads, head_dim))
    return query, key, value
