# Patch 7 — non-Marlin int4 Linear for compressed-tensors on ROCm ⭐

**File:** `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
**Apply with:** [`patch_wna16_rocm.py`](patch_wna16_rocm.py)

**What it unlocks:** quantized *dense* layers on RDNA. Before this patch, SGLang can run a MoE's
experts in int4 on gfx1151 (Triton path) but **not** its attention / projection layers — those go
through `wNa16`, which is Marlin-only, and Marlin is CUDA-only. That is why every public
quantization of Qwen3.5-35B-A3B quantizes `mlp.experts.*` and nothing else: a checkpoint with
quantized dense layers is unloadable on anything but NVIDIA.

Measured effect on `Qwen3.5-35B-A3B` (35B MoE, warm TunableOp, `--cuda-graph-max-bs 4`):

| Concurrent streams | experts-only int4 (before) | + dense int4 (patch 7) | Δ |
|---:|---:|---:|---:|
| 1 | 23.4 | **29.7** | **+27%** |
| 4 | 72.4 | **97.8** | +35% |
| 8 | 127.0 | **137.6** | +8% |

Bytes streamed per decode token drop from **3.70 GB → 1.67 GB**, below Ollama's ~1.8 GB.
Single-stream goes from 0.62× to **0.79×** of Ollama; the remaining gap is the MoE kernel
(~6× off its bandwidth roof) plus fixed per-step overhead, not dense-layer traffic.

## How it works

`compressed_tensors_wNa16.py` imports the Marlin utilities unconditionally and
`gptq_marlin_repack` under `if _is_cuda`, so on ROCm it dies with
`NameError: name 'gptq_marlin_repack' is not defined`. The patch adds a ROCm branch to
`process_weights_after_loading` / `apply_weights` that uses vLLM's `gptq_gemm` instead.

The conversion is nearly free, which is the nice part:

- compressed-tensors `pack-quantized` stores `weight_packed` as `[N, K//8]`, packing 8 int4
  values per int32 **along the input dim**, with nibbles written as `q_signed + 8`
  (`pack_to_int32` shifts by `2**(bits-1)` to make them unsigned).
- GPTQ's `qweight` is `[K//8, N]` with the same sequential nibble order, and dequantizes as
  `(nibble - zero) * scale`.

So a **transpose** turns one into the other, and symmetric compressed-tensors is exactly GPTQ
with a constant zero point of 8 — stored as 7, since GPTQ v1 keeps `zero - 1`. No bit-level
repacking is required. Scales transpose from `[N, K//G]` to `[K//G, N]`.

Two hard requirements found by testing, both non-obvious:

1. **`use_exllama=True`** (after `gptq_shuffle`). The non-exllama branch returns cos ≈ 0 on
   gfx1151 and page-faults on some shapes.
2. **fp16 activations.** `gptq_gemm` returns cos ≈ 0 with bf16 input, so `apply_weights` casts
   to fp16 and back. Verified no overflow in practice (activation absmax 1.3–12.6).

Verified against a plain PyTorch reference on a real tensor from the checkpoint:

```
Stage A (quantizer round-trip):  cos=+0.99481  rel_err=0.10242
Stage B (CT->GPTQ + gptq_gemm):  cos=+1.00000
```

(The 0.102 in stage A is RTN quantization error, not a transform error.)

## Limits

- **Symmetric 4-bit only.** Asymmetric checkpoints raise `NotImplementedError`; they need the
  real zero points unpacked and repacked rather than a constant 7.
- **`lm_head` must stay bf16.** `CompressedTensorsConfig.get_quant_method` handles only
  `LinearBase` and `FusedMoE` and returns `None` for `ParallelLMHead`, so a quantized head falls
  back to `UnquantizedEmbeddingMethod`, whose `weight` parameter the checkpoint never fills —
  giving uninitialized logits and `!!!!` output. Unlike the GPTQ/AWQ configs, compressed-tensors
  has no `lm_head_quantized` path. Quantizing `lm_head` is worth another ~1.02 GB/token, so this
  is the obvious next fix.
- The original `weight_packed` parameter is not freed after conversion (~1.4 GB retained).

## Producing a compatible checkpoint

[`tools/quantize_nonexpert.py`](../tools/quantize_nonexpert.py) requantizes the bf16 non-expert
weights of an existing experts-only AWQ checkpoint in place — no base model download needed,
since those tensors are already bf16 in the checkpoint:

```bash
python3 tools/quantize_nonexpert.py \
    --src <snapshot dir> --dst <out dir> --no-lm-head --include-in-proj
```

`--no-lm-head` is required until the `ParallelLMHead` limitation above is fixed.
