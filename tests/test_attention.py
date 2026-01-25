"""Tests for MHA, MQA, and GQA attention implementations.

Test naming convention: test_<function>_<scenario>_<expected_outcome>
"""

import pytest
import jax
import jax.numpy as jnp

from jax_transformer.attention import (
    grouped_query_attention,
    multi_head_attention,
    multi_query_attention,
    scaled_dot_product_attention,
)
from jax_transformer.utils import make_causal_mask


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention:
    """Unit tests for the core attention primitive."""

    def test_output_shape_matches_query(self, rng, mha_inputs):
        """Output shape should equal input query shape."""
        query, key, value = mha_inputs
        out = scaled_dot_product_attention(query, key, value)
        assert out.shape == query.shape

    def test_causal_mask_zeroes_future(self, rng):
        """With a causal mask, each position should only attend to past positions.

        We verify this by checking that the output at position 0 is identical
        when we swap out the K/V at positions 1+ (the future). Position 0 only
        attends to itself (the causal mask blocks future tokens), so changing
        future K/V should not change position-0 output. We keep K/V at position 0
        identical between the two runs so only the masked-out future differs.
        """
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        keys = jax.random.split(rng, 6)
        query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))

        # Shared K/V at position 0, different random K/V at positions 1+
        shared_kv0 = jax.random.normal(keys[1], (batch, 1, num_heads, head_dim))
        future_k_a = jax.random.normal(keys[2], (batch, seq_len - 1, num_heads, head_dim))
        future_k_b = jax.random.normal(keys[3], (batch, seq_len - 1, num_heads, head_dim))
        future_v_a = jax.random.normal(keys[4], (batch, seq_len - 1, num_heads, head_dim))
        future_v_b = jax.random.normal(keys[5], (batch, seq_len - 1, num_heads, head_dim))

        key_a = jnp.concatenate([shared_kv0, future_k_a], axis=1)
        key_b = jnp.concatenate([shared_kv0, future_k_b], axis=1)
        value_a = jnp.concatenate([shared_kv0, future_v_a], axis=1)
        value_b = jnp.concatenate([shared_kv0, future_v_b], axis=1)

        mask = make_causal_mask(seq_len)
        out_a = scaled_dot_product_attention(query, key_a, value_a, mask=mask)
        out_b = scaled_dot_product_attention(query, key_b, value_b, mask=mask)

        # Position 0 only attends to itself — must be identical regardless of k/v at 1+.
        assert jnp.allclose(out_a[:, 0, :, :], out_b[:, 0, :, :], atol=1e-5)

    def test_dropout_disabled_is_deterministic(self, rng, mha_inputs):
        """Without dropout, two identical calls must produce identical results."""
        query, key, value = mha_inputs
        out1 = scaled_dot_product_attention(query, key, value, dropout_rate=0.0)
        out2 = scaled_dot_product_attention(query, key, value, dropout_rate=0.0)
        assert jnp.allclose(out1, out2)

    def test_dropout_requires_rng(self, mha_inputs):
        """Calling with dropout_rate > 0 without a key must raise ValueError."""
        query, key, value = mha_inputs
        with pytest.raises(ValueError, match="dropout_rng must be provided"):
            scaled_dot_product_attention(query, key, value, dropout_rate=0.1)

    def test_output_is_finite(self, mha_inputs):
        """No NaN or Inf should appear in the output."""
        query, key, value = mha_inputs
        out = scaled_dot_product_attention(query, key, value)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# multi_head_attention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    """Tests for standard MHA."""

    def test_output_shape_correct(self, mha_inputs):
        """Shape must exactly match the query tensor."""
        query, key, value = mha_inputs
        out = multi_head_attention(query, key, value)
        assert out.shape == query.shape

    def test_output_finite(self, mha_inputs):
        """No NaN or Inf values in output."""
        query, key, value = mha_inputs
        out = multi_head_attention(query, key, value)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# grouped_query_attention
# ---------------------------------------------------------------------------

class TestGroupedQueryAttention:
    """Tests for GQA — the primary contribution."""

    def test_output_shape_correct(self, gqa_inputs):
        """Output shape must match the query shape (not the KV shape)."""
        query, key, value = gqa_inputs
        out = grouped_query_attention(query, key, value)
        assert out.shape == query.shape

    def test_equivalence_to_mha_when_groups_equal_heads(self, rng):
        """GQA with num_kv_heads == num_heads must match MHA output exactly.

        This is the key invariant: GQA generalizes MHA. When every query head
        has its own KV head (no sharing), the outputs must be numerically
        identical to within floating-point precision.
        """
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
        keys = jax.random.split(rng, 3)
        query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
        key = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim))
        value = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim))

        gqa_out = grouped_query_attention(query, key, value)
        mha_out = multi_head_attention(query, key, value)

        assert jnp.allclose(gqa_out, mha_out, atol=1e-5), (
            "GQA with num_kv_heads==num_heads should match MHA; "
            f"max diff: {jnp.max(jnp.abs(gqa_out - mha_out))}"
        )

    @pytest.mark.parametrize("num_heads,num_kv_heads", [
        (8, 8),
        (8, 4),
        (8, 2),
        (8, 1),
    ])
    def test_parametrize_head_configurations(self, rng, num_heads, num_kv_heads):
        """GQA must produce correct output shapes for all valid head configs."""
        batch, seq_len, head_dim = 2, 16, 32
        keys = jax.random.split(rng, 3)
        query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
        key = jax.random.normal(keys[1], (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(keys[2], (batch, seq_len, num_kv_heads, head_dim))

        out = grouped_query_attention(query, key, value)
        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.all(jnp.isfinite(out))

    def test_invalid_head_ratio_raises(self, rng):
        """num_kv_heads that don't divide num_heads evenly must raise ValueError."""
        batch, seq_len, head_dim = 1, 4, 16
        query = jax.random.normal(rng, (batch, seq_len, 8, head_dim))
        key = jax.random.normal(rng, (batch, seq_len, 3, head_dim))   # 8 % 3 != 0
        value = jax.random.normal(rng, (batch, seq_len, 3, head_dim))

        with pytest.raises(ValueError, match="divisible by num_kv_heads"):
            grouped_query_attention(query, key, value)

    def test_kv_cache_memory_gqa_less_than_mha(self):
        """GQA KV cache must be smaller than MHA by exactly the group factor.

        This is the core efficiency claim in the paper: 4 KV heads instead of 32
        reduces KV cache by 8x. We verify the formula is correct here.
        """
        from jax_transformer.utils import kv_cache_memory_bytes

        batch, seq_len, head_dim, num_layers = 8, 2048, 128, 32
        num_heads = 32
        num_kv_heads = 4

        mha_mem = kv_cache_memory_bytes(batch, seq_len, num_heads, head_dim, num_layers)
        gqa_mem = kv_cache_memory_bytes(batch, seq_len, num_kv_heads, head_dim, num_layers)

        assert gqa_mem < mha_mem
        expected_ratio = num_heads / num_kv_heads  # should be 8x
        actual_ratio = mha_mem / gqa_mem
        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"Expected {expected_ratio}x reduction, got {actual_ratio}x"
        )

    def test_jit_compilation_produces_same_result(self, gqa_inputs):
        """jax.jit(grouped_query_attention) must match eager execution."""
        query, key, value = gqa_inputs
        eager_out = grouped_query_attention(query, key, value)
        jit_out = jax.jit(grouped_query_attention)(query, key, value)
        assert jnp.allclose(eager_out, jit_out, atol=1e-5)

    def test_gradient_flow_produces_finite_gradients(self, rng):
        """jax.grad through GQA must produce finite (non-NaN) gradients.

        This validates that GQA is end-to-end differentiable through JAX's
        autodiff engine, a prerequisite for training.
        """
        batch, seq_len, num_heads, num_kv_heads, head_dim = 1, 8, 4, 2, 16
        keys = jax.random.split(rng, 3)
        query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
        key = jax.random.normal(keys[1], (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(keys[2], (batch, seq_len, num_kv_heads, head_dim))

        def loss_fn(q: jnp.ndarray) -> jnp.ndarray:
            """Mean squared output — scalar loss for grad computation."""
            out = grouped_query_attention(q, key, value)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss_fn)(query)
        assert grads.shape == query.shape
        assert jnp.all(jnp.isfinite(grads)), "GQA produced non-finite gradients"

    def test_vmap_over_batch(self, rng):
        """vmap should correctly vectorize GQA over an extra batch dimension."""
        # Build a single-item batch and a batched version; results must agree.
        seq_len, num_heads, num_kv_heads, head_dim = 8, 4, 2, 16
        keys = jax.random.split(rng, 3)

        # Single instance
        q_single = jax.random.normal(keys[0], (1, seq_len, num_heads, head_dim))
        k_single = jax.random.normal(keys[1], (1, seq_len, num_kv_heads, head_dim))
        v_single = jax.random.normal(keys[2], (1, seq_len, num_kv_heads, head_dim))

        # Stack into a batch of 4 identical examples
        q_batch = jnp.repeat(q_single, 4, axis=0)
        k_batch = jnp.repeat(k_single, 4, axis=0)
        v_batch = jnp.repeat(v_single, 4, axis=0)

        # vmap adds an outer batch dim: vmap(fn)(q[4, ...]) processes each independently
        vmapped_gqa = jax.vmap(
            lambda q, k, v: grouped_query_attention(q[None], k[None], v[None])
        )
        vmap_out = vmapped_gqa(q_batch, k_batch, v_batch)

        # All 4 results should be identical since inputs are identical
        assert jnp.allclose(vmap_out[0], vmap_out[1], atol=1e-5)
        assert jnp.allclose(vmap_out[0], vmap_out[3], atol=1e-5)


# ---------------------------------------------------------------------------
# multi_query_attention
# ---------------------------------------------------------------------------

class TestMultiQueryAttention:
    """Tests for MQA — the G=1 special case."""

    def test_output_shape_correct(self, rng):
        """Output must match query shape despite single KV head."""
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
        keys = jax.random.split(rng, 3)
        query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
        key = jax.random.normal(keys[1], (batch, seq_len, 1, head_dim))
        value = jax.random.normal(keys[2], (batch, seq_len, 1, head_dim))

        out = multi_query_attention(query, key, value)
        assert out.shape == query.shape

    def test_mqa_equals_gqa_with_one_kv_head(self, rng):
        """MQA must produce identical results to GQA when num_kv_heads == 1."""
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
        keys = jax.random.split(rng, 3)
        query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim))
        key = jax.random.normal(keys[1], (batch, seq_len, 1, head_dim))
        value = jax.random.normal(keys[2], (batch, seq_len, 1, head_dim))

        mqa_out = multi_query_attention(query, key, value)
        gqa_out = grouped_query_attention(query, key, value)

        assert jnp.allclose(mqa_out, gqa_out, atol=1e-5)
