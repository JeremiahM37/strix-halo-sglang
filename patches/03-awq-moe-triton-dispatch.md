# Patch 3 — AWQ MoE Triton dispatcher on ROCm

**File:** `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`

**Why:** `AWQMoEScheme` hardcodes `MoeRunnerBackend.MARLIN`. Marlin is NVIDIA-only (Cutlass-based). On ROCm there's no MoE path, so loading an AWQ MoE model either fails outright or dequantizes the int4 weights to FP16 — which OOMs a 35B-A3B model on 61.7 GB GTT (23 GB on disk → ~60 GB on GPU).

SGLang already ships `fused_moe_kernel_gptq_awq` (`sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py:91`), a fused Triton MoE kernel with a `use_int4_w4a16` flag. `TritonMoeQuantInfo` already has `w13_zp`/`w2_zp` fields for AWQ zero points. The entire Triton AWQ-MoE path exists; the dispatcher just isn't wiring it up on ROCm.

This patch routes AWQ MoE through the existing Triton path when `SGLANG_AWQ_MOE_TRITON_ROCM=1` is set on a ROCm host. Default is **off**, because the kernel currently page-faults on gfx1151 (see [`docs/AWQ_MOE_DEBUG.md`](../docs/AWQ_MOE_DEBUG.md)).

## What changes

Four hooks in `AWQMoEScheme`:

1. `__init__` — detect ROCm + env var, set `self._rocm_triton`.
2. `process_weights_after_loading` — skip Marlin-specific repacking when `_rocm_triton`.
3. `create_moe_runner` — pick `MoeRunnerBackend.TRITON` instead of `MARLIN` when `_rocm_triton`.
4. `apply_weights` — build a `TritonMoeQuantInfo` from the raw AWQ-packed weights and run through the Triton runner.

## Diff (logical)

```diff
+import os
+from sglang.srt.utils import is_hip

 class AWQMoEScheme(AWQMoESchemeBase):
     def __init__(self, quant_config: "AWQMarlinConfig"):
         self.quant_config = quant_config
+        self._rocm_triton = is_hip() and os.environ.get("SGLANG_AWQ_MOE_TRITON_ROCM", "0") == "1"
         if self.quant_config.weight_bits != 4:
             raise ValueError(...)

     def process_weights_after_loading(self, layer):
+        if self._rocm_triton:
+            return
         self.kernel.process_weights_after_loading(layer)

     def create_moe_runner(self, layer, moe_runner_config):
         self.moe_runner_config = moe_runner_config
-        self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)
+        backend = MoeRunnerBackend.TRITON if self._rocm_triton else MoeRunnerBackend.MARLIN
+        self.kernel.runner = MoeRunner(backend, moe_runner_config)

     def apply_weights(self, layer, dispatch_output):
+        if self._rocm_triton:
+            from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
+            quant_info = TritonMoeQuantInfo(
+                w13_weight=layer.w13_qweight,
+                w2_weight=layer.w2_qweight,
+                use_int4_w4a16=True,
+                w13_scale=layer.w13_scales, w2_scale=layer.w2_scales,
+                w13_zp=layer.w13_qzeros, w2_zp=layer.w2_qzeros,
+                block_shape=[self.quant_config.group_size, self.quant_config.group_size],
+            )
+            return self.kernel.runner.run(dispatch_output, quant_info)
         return self.kernel.apply(layer, dispatch_output)
```

## Validation

With the patch active, Qwen3.5-35B-A3B-AWQ-4bit loads correctly:

```
Load weight end. type=Qwen3_5MoeForConditionalGeneration, quant=compressed-tensors,
  avail mem=91.18 GB, mem usage=23.36 GB.
```

That's the correct packed-int4 size on GPU — vs ~60 GB when Marlin dispatch fails and dequantization to FP16 kicks in.

## Known issue

Inference triggers a GPU page fault inside `fused_moe_kernel_gptq_awq`:

```
Memory access fault by GPU node-1 on address 0x7ff63d77c000.
Reason: Page not present or supervisor privilege.
```

The kernel reaches the GPU, launches, and faults during execution. Likely causes (ranked): tile-size LDS overflow, wave32 vs wave64 stride assumption, `tl.interleave` codegen difference. Full debug plan in [`docs/AWQ_MOE_DEBUG.md`](../docs/AWQ_MOE_DEBUG.md). Until that's fixed, this patch is opt-in via env var so it doesn't break general use.

## Upstream

When the underlying kernel is fixed, this patch is a strong candidate for `sgl-project/sglang` — it's also a no-op on NVIDIA (just adds an env-var-gated branch that the existing Marlin path skips through).
