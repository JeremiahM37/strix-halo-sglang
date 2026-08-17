# Patch 5 — MoE tuner emits config files the runtime never reads (int4 only)

**File:** `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`

**Why:** Tuning a fused MoE kernel silently no-ops for int4-quantized MoE models. The tuner
writes its results to one filename and the runtime looks up another, so the server keeps
using default tiles and reports `Config file not found` even after a successful tuning run.

The runtime derives its lookup key from the down-projection weight shape:

```python
# fused_moe_triton_config.py, try_get_optimal_moe_config
E, _, N = w2_shape
```

The tuner derives the same key with an extra int4-only shift:

```python
# tuning_fused_moe_triton.py, MoEBenchmark.benchmark
N = shard_intermediate_size // 2
if use_int4_w4a16:
    N = N // 2          # <-- runtime has no equivalent
```

On `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (E=256, `moe_intermediate_size` 512, hidden 2048)
the two disagree by exactly the factor of two:

```
tuner:   Start tuning over 400 configurations to create
         E=256,N=128,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json
server:  Using default MoE kernel config. Performance might be sub-optimal!
         Config file not found at .../E=256,N=256,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json
```

This only affects `use_int4_w4a16`, which is why fp8/bf16 MoE tuning works fine upstream and
the bug has gone unnoticed. It is silent in the worst way: the tuner runs for ~20 minutes,
reports success, writes a file, and changes nothing.

## What changes

Drop the extra halving so the tuner's filename matches the runtime's lookup.

```diff
         N = shard_intermediate_size // 2
-        if use_int4_w4a16:
-            N = N // 2
```

`N` is used only for the config lookup and the output filename — the benchmarked shapes come
from `shard_intermediate_size` and `hidden_size` and were already correct. So this changes
*where the tuning result lands*, not what was tuned.

## This is a correctness fix, not a speedup

Because `N` only affects the filename, installing the tuner's output under the name the
runtime expects is equivalent to applying this patch. We measured that directly on
Qwen3.5-35B-A3B-AWQ-4bit (35B MoE, warm TunableOp, `--cuda-graph-max-bs 4`):

| MoE config | 1 stream | 4 | 8 |
|---|---:|---:|---:|
| Default tiles (`BSK=128, GSM=1`) | **23.1** | **72.4** | **127.0** |
| Tuner's pick (`BSK=32, GSM=16`) | 19.9 | 63.4 | 108.2 |

The tuned config is ~14% *slower* end-to-end. The likely reason: the tuner benchmarks one
expert set in a tight loop, so the weights stay resident in the 8060S's 32 MB MALL, and it
picks a small-`BLOCK_SIZE_K`, low-`num_stages` shape that suits cache-resident operands. In
real decode each layer's experts are touched once per token and streamed cold from GTT, where
that shape is the wrong trade. Fixing the filename makes tuning *take effect*; it does not
make it *helpful* on this hardware. Tuning the MoE for gfx1151 decode needs a harness that
reproduces the cold-streaming access pattern.

See [`bench/results.md`](../bench/results.md) for the decode-step profile that puts the MoE at
31% of the step.
