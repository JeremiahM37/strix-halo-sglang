# Known issues

## AWQ MoE inference page-faults — RESOLVED

**Resolved by [patch 4](../patches/04-warp-size-wave32.md)** (host/device `WARP_SIZE` mismatch in the sgl-kernel topk gating kernels), which is baked into the default image build. AWQ MoE inference now works end-to-end — see [`RUNNING_AWQ_MOE.md`](RUNNING_AWQ_MOE.md).

The historical detail below is kept as a debugging record.

**Symptom (historical):** Model loads (correct packed-int4 weight size on GPU), server starts, `/v1/models` serves. First inference request triggers:

```
Memory access fault by GPU node-1 on address 0x7ff63d77c000.
Reason: Page not present or supervisor privilege.
```

The crash appeared to be inside `fused_moe_kernel_gptq_awq` (`sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py:91`) when called with `use_int4_w4a16=True` — but that attribution turned out to be wrong: the AWQ kernel never actually ran. The real fault was a failed launch of the topk gating softmax kernel (`__launch_bounds__` compiled for 128 threads, launched with 256) that poisoned the GPU command queue; the *next* kernel then page-faulted. Root-cause analysis in [patch 4](../patches/04-warp-size-wave32.md); the full debugging record is in [`AWQ_MOE_DEBUG.md`](AWQ_MOE_DEBUG.md).

## aiter Flash Attention won't build on RDNA 3.5

**Symptom:** Setting `--attention-backend aiter` triggers a JIT build that fails:

```
block_gemm_areg_bsmem_creg_v2.hpp:79: error: no viable overloaded '='
```

**Cause:** aiter's MHA kernels rely on Composable Kernel tile templates that assume wave64 thread tile shapes (CDNA). gfx1151 is wave32 (RDNA 3.5). The `=` operator between thread buffers doesn't have an overload for the wave32 type.

Mitigation: use `--attention-backend triton` (default). Slower than aiter would be, but functionally correct.

Fix (upstream): wave32-aware CK template specializations in `ROCm/aiter`. Bigger lift than the inline-asm kernels.

## aiter RMSNorm uses CDNA-only inline asm

**Symptom:** Setting `SGLANG_USE_AITER=1` triggers a JIT build that fails:

```
rmsnorm_quant_kernels.cu:173: error: instruction not supported on this GPU
  v_pk_mul_f32 %0, %1, %2
```

**Cause:** `v_pk_mul_f32` (packed FP32 multiply) is a CDNA instruction. RDNA 3.5 doesn't have it.

Mitigation: default to `SGLANG_FORCE_NATIVE_LAYERNORM=1` (already in Dockerfile). Falls through to native PyTorch RMSNorm.

Fix (upstream): replace inline asm with portable HIP intrinsics, or add an RDNA branch.

## Wave attention backend incompatible with Qwen3.5

**Symptom:** `--attention-backend wave` errors out with:

```
ValueError: layer_id=0 not in full attention layers: dict_keys([3, 7, 11, 15, ...])
```

**Cause:** Wave backend assumes layer 0 is full attention. Qwen3.5 is a hybrid architecture — most layers are linear attention (GDN/Mamba), only every 4th layer is full attention.

Mitigation: use `--attention-backend triton`.

## Single-stream decode is slower than Ollama

**Symptom:** Sequential (single-stream) decode on Qwen3.5-4B trails Ollama. Current measurements live in [`bench/results.md`](../bench/results.md) — the single source of truth for numbers.

**Cause:** All the AMD fast paths (aiter Flash Attention, aiter RMSNorm, AWQ Marlin) are unavailable on gfx1151 today. SGLang falls back to Triton attention + native PyTorch RMSNorm, which are correct but slower than Ollama's hand-tuned llama.cpp HIP kernels.

**Where SGLang still wins:** continuous batching — at 8 concurrent streams SGLang's aggregate throughput is many times Ollama's serialized rate (see [`bench/results.md`](../bench/results.md)). For multi-user / multi-agent workloads SGLang wins decisively; for solo single-stream use Ollama is faster today.

Fix (upstream, in measured order of impact): aiter wave32 CK templates for MHA, then the int4 MoE GEMM, then an RDNA RMSNorm path. That ordering is empirical, not intuitive — a fused RMSNorm is worth only ~2% at decode (it is worth 8× at prefill shapes), CUDA graphs are worth nothing, and shrinking the MoE `BLOCK_SIZE_M` to fit decode makes things slightly *worse*. See the "three hypotheses, all negative" table in [`bench/results.md`](../bench/results.md) before spending time here.
