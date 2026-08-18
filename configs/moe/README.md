# Tuned MoE Triton configs for gfx1151

Mount over SGLang's config directory:

```
-v $(pwd)/configs/moe:/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0
```

**Why these values.** At batch size 1 the fused MoE GEMM is bound by how much of the GPU it
can keep busy, not by tile efficiency. `BLOCK_SIZE_M` is irrelevant (only one row tile exists
at decode). `BLOCK_SIZE_N` and `num_warps` together decide workgroup count, and the upstream
defaults launch far too few for a 40-CU part.

Measured end-to-end on Qwen3.5-35B-A3B, single stream:

| BLOCK_SIZE_N | num_warps | tps |
|---:|---:|---:|
| 128 (upstream default) | 4 | 16.3 |
| 32 | 4 | 34.5 |
| 16 | 4 | 38.0 |
| **16** | **2** | **39.7** |
| 16 | 1 | 35.7 |
| 16 | 8 | 29.7 |

`num_warps=1` is too few to cover memory latency; 8 oversubscribes. 2 is the optimum here.
