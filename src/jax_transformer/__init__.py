"""jax-transformer-impl: Production-grade JAX implementation of Grouped Query Attention.

Implements MHA, MQA, and GQA variants with JIT compilation, vmap/pmap support,
and benchmark tooling against PyTorch baseline.

Based on: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
Checkpoints" (Ainslie et al., 2023) — arXiv:2305.13245
"""

from jax_transformer.attention import (
    grouped_query_attention,
    multi_head_attention,
    multi_query_attention,
    scaled_dot_product_attention,
)
from jax_transformer.models import GQAConfig, TransformerBlock

__version__ = "0.1.0"
__author__ = "Rajath John"
__email__ = "jrajath94@gmail.com"

__all__ = [
    "grouped_query_attention",
    "multi_head_attention",
    "multi_query_attention",
    "scaled_dot_product_attention",
    "GQAConfig",
    "TransformerBlock",
]
