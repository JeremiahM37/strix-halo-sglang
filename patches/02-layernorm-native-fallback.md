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

Native PyTorch RMSNorm is slower than a fused HIP kernel. On Qwen3.5-4B this contributes most of the ~40% single-stream gap vs Ollama (which has hand-tuned llama.cpp RMSNorm). Recovering this requires patches to:

- `ROCm/aiter` — add RDNA 3.5 path for `rmsnorm_quant_kernels.cu` (replace `v_pk_mul_f32` etc. with portable HIP intrinsics)
- or upstream a Triton RMSNorm fallback to `sglang/srt/layers/layernorm.py`

Either way, this env-var gate is the minimum patch to unblock everything else.
