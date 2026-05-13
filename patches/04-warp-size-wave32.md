# Patch 4 — Fix `WARP_SIZE` host/device mismatch in sgl-kernel topk gating

**File:** `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu` (and the parallel `moe_topk_sigmoid_kernels.cu`)

**Why:** These two kernels use `WARP_SIZE` for both the `__launch_bounds__` declaration and the host-side launch `dim3 block_dim(WARP_SIZE, WARPS_PER_TB)`, but never define it themselves. On gfx1151 (RDNA 3.5, wave32) the HIP runtime headers expand `WARP_SIZE` to **32 in device-compile context** (matching the actual hardware wave size) but to **64 in host-compile context** (the CDNA default). The result:

- Device-side: kernel is compiled with `__launch_bounds__(WARPS_PER_CTA * 32) = __launch_bounds__(128)` — the compiler reserves registers for at most 128 threads per block.
- Host-side: `dim3 block_dim(64, 4)` launches the kernel with **256 threads** per block.

HIP rejects this at launch:

```
:1:hip_module.cpp:311 : Launch params (64, 4, 1) are larger than launch bounds (128)
   for kernel _Z17topkGatingSoftmaxI14__hip_bfloat16Li8ELi256ELi4ELi16E...
:3:hip_module.cpp:791 : hipLaunchKernel: Returned hipErrorLaunchFailure
```

The failed launch leaves the GPU command queue in a bad state, and the next kernel (`moe_align_block_size_kernel`) page-faults:

```
Memory access fault by GPU node-1 on address 0x7fc74d8ae000. Reason: Page not present.
```

This is the page fault we'd been blaming on `fused_moe_kernel_gptq_awq` — but the AWQ MoE kernel never actually got to run. Found via `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3`.

The sibling kernel `moe_fused_gate.cu` already pins `static constexpr int WARP_SIZE = 32;` at file scope; these two were just missed.

## What changes

Rename `WARP_SIZE` to a unique identifier (`kStrixWarp`) throughout both files, then pin it to `32`. We use a unique name rather than reusing `WARP_SIZE` because a `static constexpr int WARP_SIZE = 32;` at file scope is silently shadowed by the HIP `WARP_SIZE` macro that gets pulled in transitively from `<torch/all.h>` — which is what caused our first attempted fix to compile cleanly but have no effect.

```diff
+// gfx1151 wave32 fix — own identifier, no HIP macro shadow
+static constexpr int kStrixWarp = 32;

-__launch_bounds__(WARPS_PER_CTA * WARP_SIZE) __global__ void topkGatingSoftmax(
+__launch_bounds__(WARPS_PER_CTA * kStrixWarp) __global__ void topkGatingSoftmax(
   ...
-  static_assert(WARP_SIZE % THREADS_PER_ROW == 0, ...);
+  static_assert(kStrixWarp % THREADS_PER_ROW == 0, ...);
   ...
-  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
+  dim3 block_dim(kStrixWarp, WARPS_PER_TB);
```

## Validation

With patch 4 applied, `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (256-expert Qwen 3.5 MoE) generates coherent output end-to-end:

```
prompt:     "Write one sentence about cats."
output:     "Thinking Process:\n\n1. **Analyze the Request:**\n   *   Topic: Cats. ..."
prompt_tokens=16  completion_tokens=50  finish_reason=length
```

GPU memory after load: 22.84 GB on a 61.7 GB GTT pool — matches the expected size for 4-bit 35B weights.

## Why downstream only

The bug is real for every wave32 ROCm target (gfx1100/1101/1102/1151/1200/1201) — same `WARP_SIZE` reliance, same fault. The fix is small and a no-op on CUDA / CDNA. But `sgl-kernel/setup_rocm.py` explicitly guards against anything other than `gfx942`/`gfx950` ("Unsupported GPU architecture detected"), so SGLang isn't accepting consumer-RDNA fixes — supporting wave32 would expand their support surface to consumer SKUs they don't test. An attempted upstream PR ([#25175](https://github.com/sgl-project/sglang/pull/25175)) was auto-closed by CI in minutes. The fix lives here instead.
