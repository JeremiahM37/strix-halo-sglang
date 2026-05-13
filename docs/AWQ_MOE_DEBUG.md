# Debug plan — `fused_moe_kernel_gptq_awq` page fault on gfx1151

## Current state

With the AWQ MoE Triton dispatcher patch active (`SGLANG_AWQ_MOE_TRITON_ROCM=1`), Qwen3.5-35B-A3B-AWQ-4bit loads cleanly:

```
Load weight end. type=Qwen3_5MoeForConditionalGeneration, quant=compressed-tensors,
  avail mem=91.18 GB, mem usage=23.36 GB.
```

Weights stay packed at 23 GB (correct int4 size). KV + Mamba cache allocate fine. Server reports `Application startup complete`.

First inference request crashes the scheduler subprocess:

```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7ff63d77c000.
Reason: Page not present or supervisor privilege.
Fatal Python error: Aborted
```

Stack trace points into the Triton kernel launch:

```
fused_moe_triton_kernels.py:822  fused_moe_kernel_gptq_awq[grid](...)
fused_moe.py:482                  invoke_fused_moe_kernel(...)
fused_moe.py:854                  fused_experts_impl(...)
```

SGLang also warns on startup:

```
Using default MoE kernel config. Performance might be sub-optimal!
Config file not found at .../configs/triton_3_7_0/E=256,N=256,
  device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json
```

— no tuned tile sizes for this hardware/quant combo.

## Hypotheses, ranked

### 1. Tile-size overflow of LDS / register file

Default tile sizes (BLOCK_SIZE_M=64, N=64, K=32 typical) may exceed gfx1151's LDS budget (64 KB/workgroup) once we include the unpacked int4 → bf16 staging plus scales and zeros. Memory fault would happen on out-of-bounds shared-mem store.

**Test:** drop a hand-written config at the warned path with conservative tiles (M=16, N=32, K=64, num_warps=4, num_stages=2). If the fault goes away → it's purely tile-size. Tune for all common M values and PR upstream.

### 2. Wave32 vs wave64 stride math

The Triton kernel was authored against NVIDIA SMs (32-thread warps) but the AMD codegen path may compute pointer offsets assuming wave64. gfx1151 is wave32 — same as NVIDIA in thread count per warp but different in how Triton lowers tile primitives.

**Test:** examine generated TritonGPU IR. Look for `wave_size` constants in the lowered ops. Compare strides against what the kernel intends.

### 3. `tl.interleave` codegen difference on ROCm

The kernel uses chained `tl.interleave(b, b)` for 4-bit unpacking (8 packed → 8 unpacked). Triton's AMD backend may lower this differently than NVIDIA, especially around the masked-load alignment.

**Test:** replace `tl.interleave` chains with explicit shift+mask sequence (closer to the existing per-tensor AWQ kernel in vLLM ROCm). Compare correctness on a small reproducer.

### 4. Mask off-by-one on actual M ≠ padded M

For prefill, the kernel pads M to the nearest tile multiple. If the mask logic uses the un-padded M for store but the padded M for offset, a store can land beyond the output buffer.

**Test:** force `M == padded_M` (single token, BLOCK_SIZE_M=1). If fault disappears, mask is the culprit.

## Reproducer

Standalone, no full SGLang server stack — isolate to just the kernel:

```python
# bench/repro_awq_moe_kernel.py (TODO)
import torch
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import _fused_moe_kernel_sequence

num_experts = 256
num_tokens = 8
hidden_size = 2048
intermediate = 1024
group_size = 128

# Fake AWQ-packed weights
w13 = torch.zeros(num_experts, hidden_size, 2 * intermediate // 8,
                  dtype=torch.int32, device="cuda")
w2  = torch.zeros(num_experts, intermediate, hidden_size // 8,
                  dtype=torch.int32, device="cuda")
scales_13 = torch.ones(num_experts, hidden_size // group_size,
                       2 * intermediate, dtype=torch.bfloat16, device="cuda")
# ... zeros, topk_ids, sorted_token_ids ...

# Call _fused_moe_kernel_sequence with use_int4_w4a16=True
```

This is the minimum-viable repro. Iterating on the kernel becomes a 5-second cycle instead of 1 min per server restart.

## Outcome

When fixed, the fix is one or more of:

- A `configs/triton_3_7_0/...Radeon_8060S_Graphics,dtype=int4_w4a16*.json` tuned tile-size file (cheapest fix; goes upstream as a config drop)
- A small patch to `fused_moe_kernel_gptq_awq` for the wave32 / lowering bug (medium effort; goes upstream as a kernel PR)
- An `aiter`-based AWQ-MoE wrapper that bypasses Triton entirely (bigger lift; only worth it if the upstream Triton path can't be fixed)

Either way the resulting patch is small, well-tested, and PR-ready for `sgl-project/sglang`.
