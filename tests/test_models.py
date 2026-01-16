"""Tests for TransformerBlock and supporting model components."""

import pytest
import jax
import jax.numpy as jnp

from jax_transformer.models import (
    GQAConfig,
    TransformerBlock,
    init_transformer_block,
    rms_norm,
    swiglu_ffn,
    transformer_block_forward,
)
from jax_transformer.utils import count_parameters, make_causal_mask


# ---------------------------------------------------------------------------
# GQAConfig
# ---------------------------------------------------------------------------

class TestGQAConfig:
    """Validation tests for the configuration dataclass."""

    def test_valid_config_creates_without_error(self):
        config = GQAConfig(model_dim=512, num_heads=8, num_kv_heads=2, head_dim=64)
        assert config.model_dim == 512

    def test_invalid_head_ratio_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            GQAConfig(model_dim=512, num_heads=8, num_kv_heads=3, head_dim=64)

    def test_resolved_ffn_dim_is_multiple_of_256(self):
        config = GQAConfig(model_dim=512)
        assert config.resolved_ffn_dim % 256 == 0

    @pytest.mark.parametrize("model_dim", [64, 128, 256, 512, 1024])
    def test_ffn_dim_scales_with_model_dim(self, model_dim):
        config = GQAConfig(model_dim=model_dim, num_heads=4, head_dim=model_dim // 4)
        assert config.resolved_ffn_dim >= model_dim


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    """Unit tests for RMS normalization."""

    def test_output_shape_preserved(self):
        x = jnp.ones((2, 8, 64))
        scale = jnp.ones(64)
        out = rms_norm(x, scale)
        assert out.shape == x.shape

    def test_unit_input_produces_scale_output(self):
        """When all elements are equal, output = scale value."""
        x = jnp.ones((1, 1, 4))
        scale = jnp.array([2.0, 2.0, 2.0, 2.0])
        out = rms_norm(x, scale)
        # RMS of ones is 1; 1/1 * 2 = 2
        assert jnp.allclose(out, 2.0 * jnp.ones_like(out), atol=1e-6)

    def test_zero_input_is_stable(self):
        """Zero input should not produce NaN due to eps."""
        x = jnp.zeros((1, 4, 16))
        scale = jnp.ones(16)
        out = rms_norm(x, scale)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class TestSwiGLUFFN:
    """Tests for the gated feed-forward network."""

    def test_output_shape_correct(self, rng):
        d, df = 64, 128
        keys = jax.random.split(rng, 3)
        x = jax.random.normal(keys[0], (2, 8, d))
        w1 = jax.random.normal(keys[1], (d, df)) * 0.02
        w2 = jax.random.normal(keys[1], (df, d)) * 0.02
        w3 = jax.random.normal(keys[2], (d, df)) * 0.02
        out = swiglu_ffn(x, w1, w2, w3)
        assert out.shape == x.shape

    def test_output_is_finite(self, rng):
        d, df = 32, 64
        keys = jax.random.split(rng, 4)
        x = jax.random.normal(keys[0], (2, 4, d)) * 0.1
        w1 = jax.random.normal(keys[1], (d, df)) * 0.02
        w2 = jax.random.normal(keys[2], (df, d)) * 0.02
        w3 = jax.random.normal(keys[3], (d, df)) * 0.02
        out = swiglu_ffn(x, w1, w2, w3)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    """Integration tests for the full transformer block."""

    def test_output_shape_matches_input(self, rng, small_config):
        """TransformerBlock is shape-preserving."""
        batch, seq_len = 2, 8
        x = jax.random.normal(rng, (batch, seq_len, small_config.model_dim))
        block = TransformerBlock.init(rng, small_config)
        out = block(x)
        assert out.shape == x.shape

    def test_output_is_finite(self, rng, small_config):
        """No NaN or Inf in block output with random inputs."""
        x = jax.random.normal(rng, (2, 8, small_config.model_dim))
        block = TransformerBlock.init(rng, small_config)
        out = block(x)
        assert jnp.all(jnp.isfinite(out))

    def test_block_with_causal_mask(self, rng, small_config):
        """Block should accept and respect causal mask without errors."""
        batch, seq_len = 1, 16
        x = jax.random.normal(rng, (batch, seq_len, small_config.model_dim))
        mask = make_causal_mask(seq_len)
        block = TransformerBlock.init(rng, small_config)
        out = block(x, mask=mask)
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))

    def test_jit_compiled_block_matches_eager(self, rng, small_config):
        """JIT-compiled block must produce same result as eager execution."""
        x = jax.random.normal(rng, (2, 8, small_config.model_dim))
        block = TransformerBlock.init(rng, small_config)

        eager_out = block(x)
        jit_fn = jax.jit(lambda inp: transformer_block_forward(inp, block.params, block.config))
        jit_out = jit_fn(x)

        assert jnp.allclose(eager_out, jit_out, atol=1e-5)

    def test_gradient_flows_through_block(self, rng, small_config):
        """jax.grad through the full block should produce finite gradients."""
        x = jax.random.normal(rng, (1, 4, small_config.model_dim)) * 0.1
        block = TransformerBlock.init(rng, small_config)

        def loss_fn(inp: jnp.ndarray) -> jnp.ndarray:
            out = transformer_block_forward(inp, block.params, small_config)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss_fn)(x)
        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads)), "Non-finite gradients in block"

    def test_parameter_count_scales_with_config(self):
        """Larger configs should have proportionally more parameters."""
        rng = jax.random.PRNGKey(0)
        small = GQAConfig(model_dim=64, num_heads=4, num_kv_heads=2, head_dim=16)
        large = GQAConfig(model_dim=256, num_heads=8, num_kv_heads=2, head_dim=32)

        small_params = init_transformer_block(rng, small)
        large_params = init_transformer_block(rng, large)

        small_count = count_parameters(small_params)
        large_count = count_parameters(large_params)
        assert large_count > small_count
