# Patch 2 — RMSNorm native fallback on gfx1151

**File:** `python/sglang/srt/layers/layernorm.py`

**Why:** SGLang's `_is_hip` path imports `fused_add_rms_norm` from `vllm._custom_ops`. The vLLM build in our base image ships the older 4-arg signature, but SGLang main calls it with 6 args. The aiter alternative isn't usable either — `rmsnorm_quant_kernels.cu` uses `v_pk_mul_f32` inline asm which is CDNA-only.

There are three RMSNorm classes in `layernorm.py`, each with its own `forward_hip` method. Patching them individually is fragile; the cleaner fix is a single point — disable `_has_vllm_rms_norm` globally and let all three fall through to `forward_native`.

## Diff

```diff
 elif _is_hip:
     try:
         from vllm._custom_ops import fused_add_rms_norm, rms_norm

         _has_vllm_rms_norm = True
+        # gfx1151 (RDNA 3.5): vllm's fused_add_rms_norm has older 4-arg signature.
+        # Force-disable so every forward_hip method falls through to forward_native.
+        import os as _os
+        if _os.environ.get('SGLANG_FORCE_NATIVE_LAYERNORM', '0') == '1':
+            _has_vllm_rms_norm = False
     except ImportError:
-        # Fallback: vllm not available, will use forward_native
         _has_vllm_rms_norm = False
```

## Activation

`SGLANG_FORCE_NATIVE_LAYERNORM=1` is set by default in the Dockerfile.

## Trade-off

Native PyTorch RMSNorm is slower than a fused HIP kernel — but **measurably less than this doc used to claim.** An earlier revision asserted it "contributes most of the ~40% single-stream gap vs Ollama." Measured on gfx1151 with [`bench/rmsnorm_micro.py`](../bench/rmsnorm_micro.py):

| Shape | eager (`forward_native`) | fused Triton | speedup |
|---|---:|---:|---:|
| M=1, hidden 2560 (4B decode) | 32.5 µs | 20.4 µs | 1.59× |
| M=1, hidden 4096 (35B decode) | 31.5 µs | 19.8 µs | 1.59× |
| M=512, hidden 4096 (prefill) | 264.6 µs | 33.1 µs | 8.00× |

At decode, 2 norms × 36 layers × 32.5 µs = **2.3 ms per token out of a ~43 ms budget — 5.4%.** Replacing eager with a fused Triton kernel recovers ~0.9 ms/token, about **2%**. Worth having, not the gap. Note both numbers are far above the ~1 µs the memory traffic implies, so at these shapes the eager path is dominated by dispatch overhead, not arithmetic — which is also why CUDA graphs, which remove that overhead, produce no end-to-end gain (the dispatch is already hidden behind GPU work). The real single-stream gap is in the attention and MoE kernels; see [`bench/results.md`](../bench/results.md).

Prefill is a different story — 8× there is real, and a fused kernel is clearly worth upstreaming for that alone.

Recovering the fused path requires patches to:

- `ROCm/aiter` — add RDNA 3.5 path for `rmsnorm_quant_kernels.cu` (replace `v_pk_mul_f32` etc. with portable HIP intrinsics)
- or upstream a Triton RMSNorm fallback to `sglang/srt/layers/layernorm.py`

Either way, this env-var gate is the minimum patch to unblock everything else.
