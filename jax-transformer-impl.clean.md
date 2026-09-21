# jax-transformer-impl

A clean JAX implementation of Multi-Head Attention (MHA), Multi-Query
Attention (MQA), and Grouped Query Attention (GQA, Ainslie et al. 2023), with
JIT compilation, vmap batch parallelism, and pmap multi-device training.
Functional style, no Flax linen: every step is inspectable.

**Verified 2026-09-21:** 39 tests pass on CPU. GQA matches MHA output when
G=H (tested; see "The GQA math" below).

## Why this exists

PyTorch has thousands of Transformer implementations. JAX has few that are
clean, tested, and implement recent attention variants. This repo is the
attention stack from first principles: functional attention with einsum,
RMSNorm, SwiGLU, pre-norm residual blocks, JIT/vmap/pmap support, and
KV-cache memory tooling.

The pieces map to production reality. GQA is the attention variant behind
Llama 2 70B: fewer KV heads than query heads means a smaller KV cache, which
is usually what caps inference batch size and context length.

## The GQA math

GQA generalizes MHA and MQA. When `num_kv_heads == num_heads`, grouped query
attention produces output identical to multi-head attention (tested). When
`num_kv_heads == 1`, it is MQA. In between, K and V are shared across groups
of query heads:

```
MHA (G = H)  →  GQA (1 < G < H)  →  MQA (G = 1)
```

The expansion is `jnp.repeat(kv, num_heads // num_kv_heads, axis=2)`, turning
(B, T, G, D) into (B, T, H, D): each KV head is repeated across its group of
query heads, which realizes the paper's grouped sharing. Known limitation:
`jnp.repeat` materializes the expanded tensor, so this code does not realize
the cache savings the tables promise; full savings need a grouped einsum
that never materializes the repeat. The memory tables below describe the
efficient form.

## KV-cache memory

**Dtype note.** Every number below assumes float32, the code's default dtype.
Real inference deployments run fp16 or bf16, which halves the absolute bytes.
The ratios (2x/4x/8x) are unchanged and are the scientifically load-bearing
part.

Small config (batch=4, seq=512, head_dim=64, 32 layers), fp32:

| Config | KV Cache (MB) | Reduction vs MHA |
| --- | --- | --- |
| MHA (H=8, G=8) | 268.44 | 1.0x |
| GQA (H=8, G=4) | 134.22 | 2.0x |
| GQA (H=8, G=2) | 67.11 | 4.0x |
| MQA (H=8, G=1) | 33.55 | 8.0x |

Llama-2-70B config (bs=32, seq=4096, 80 layers), fp32 - halve for fp16:

| Config | KV Cache |
| --- | --- |
| MHA (H=64) | 687.19 GB |
| GQA like Llama-2-70B (G=8) | 85.90 GB |
| MQA (G=1) | 10.74 GB |

In fp16, the GQA cache is ~43 GB. The 8x A100 80GB budget (640 GB) also
holds the ~140 GB fp16 weights plus activations, so 4K-context inference
fits. That is why Llama-2-70B uses GQA with G=8.

The memory story above is inference-only. In training, the O(T^2) causal
mask is the bigger memory term.

To measure latency on your hardware: `make bench`.

## Quick start

```
git clone https://github.com/jrajath94/jax-transformer-impl.git
cd jax-transformer-impl
pip install "jax[cpu]" && pip install -e ".[dev]"
python examples/quickstart.py
```

Multi-device data parallel:

```
from jax_transformer.models import transformer_block_forward, GQAConfig, init_transformer_block
from jax_transformer.utils import shard_batch, unshard_batch, make_pmap_forward
import jax

config = GQAConfig(model_dim=512, num_heads=8, num_kv_heads=2, head_dim=64)
params = init_transformer_block(jax.random.PRNGKey(0), config)

pmap_forward = make_pmap_forward(transformer_block_forward)  # jax.pmap wrapper:
    # axis_name="batch", in_axes=(0, None, None) - x is sharded across devices,
    # params and config are broadcast to each device

x = jax.random.normal(jax.random.PRNGKey(1), (8, 512, 512))
x_sharded = shard_batch(x, num_devices=jax.device_count())
out = unshard_batch(pmap_forward(x_sharded, params, config))
```

## Key design decisions

- **Functional style, no Flax linen.** Plain dict parameter trees work
  directly with optax and pmap; no framework overhead; every step is
  inspectable. Tradeoff: less tooling than Flax modules.
- **Pre-norm RMSNorm before each sublayer.** Training stability at scale;
  what Llama, Gemma, and Mistral use. Tradeoff: post-norm (original
  Transformer) is harder to train deep with no upside at scale.
- **SwiGLU over GELU FFN.** Outperforms ReLU/GELU FFNs at matched param
  counts in the literature. Tradeoff: slightly more complex than GELU.
- **`jnp.repeat` for KV expansion.** Functional and XLA-friendly. Tradeoff:
  materializes the full tensor (see the limitation note above).
- **NEG_INF = -1e9 for masked positions.** Standard with fp32. In fp16 it
  saturates to -inf, which makes softmax NaN on fully-masked rows; flagged
  for anyone changing the dtype.

## Testing

```
make test    # 39 tests, all pass on CPU (verified 2026-09-21)
make bench   # GQA vs MHA memory + latency benchmarks
make lint    # ruff + mypy
```

Tests cover: GQA/MHA equivalence, shape and output validity, gradient flow
for all attention variants, JIT compilation and vmap compatibility, and a
full transformer block forward/backward pass.

## Implements

- Scaled dot-product attention (Vaswani et al., 2017)
- Multi-query attention (Shazeer, 2019)
- Grouped query attention - arXiv:2305.13245 (Ainslie et al., 2023)
- RMSNorm (Zhang & Sennrich, 2019)
- SwiGLU FFN (Shazeer, 2020)

## License

MIT
