# Benchmark results

All runs on **AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151)**, ROCm 7.13 nightly, 61.7 GB GTT.

## Concurrent throughput — Qwen3.5-4B

> ⚠️ **Withdrawn 2026-08-16 — the Ollama baseline in this section is contaminated.**
> It was measured while the GPU was busy with SGLang work. Both engines share a single
> iGPU, and this harness benches them back-to-back against a live SGLang server, so any
> SGLang activity that overlaps the Ollama phase (server warm-up, TunableOp autotuning,
> CUDA-graph capture) lands directly in the Ollama numbers. See
> [Contamination postmortem](#contamination-postmortem-2026-08-16) for the reproduction.
> The SGLang column has not been re-run either and is provisional. Treat the
> [35B MoE section](#concurrent-throughput--qwen3535b-a3b-moe-a3b--33b-active) as the
> reference result until this is re-measured cleanly.

Same model on both engines (`qwen3.5:4b` on Ollama, `Qwen/Qwen3.5-4B` on SGLang). 80-token generations, identical prompts. SGLang numbers are warm-cache (TunableOp results already tuned).

| Concurrent streams | Ollama agg tps (contaminated) | Ollama agg tps (clean re-run) | strix-halo-sglang agg tps | per-stream tps (SGLang) | SGLang advantage (provisional) |
|---:|---:|---:|---:|---:|---:|
| 1 | ~~9.0~~ | 43.3 | 23.1 | 23.1 | 0.53× |
| 4 | ~~9.3~~ | 44.6 | 85.8 | 21.4 | 1.92× |
| 8 | ~~9.2~~ | 45.1 | 159.9 | 20.0 | **3.55×** |

The clean Ollama column was re-measured 2026-08-16 with the same script, same model tag, and the same Ollama build (binary dated 2026-04-10 — the version did not change between runs), on an otherwise-idle GPU. The advantage column mixes an August Ollama run with a May SGLang run and is therefore provisional; a same-session re-run of both engines is the fix.

**Reading:** Ollama serializes — 8 concurrent streams yield the same aggregate as 1 stream (43.3 → 45.1). That part held up. What changed is the level: Ollama's serialized rate is ~45 agg tps, not ~9, so SGLang's concurrency win on this model is ~3.5×, in line with the 3.35× measured on the 35B MoE — not 17.4×. SGLang's continuous batching still keeps per-stream throughput nearly flat (23.1 → 20.0 tps, 13% drop) while aggregate scales near-linearly. And Ollama's single-stream win is now unambiguous: at 1 stream it is **1.9× faster** than SGLang here, consistent with the single-stream decode table below.

### Contamination postmortem (2026-08-16)

The tell was internal inconsistency: Ollama at 9.0 tps on a 4-bit 4.7B dense model while scoring 37.8 tps on a 4-bit 35B-A3B MoE (3.3B active) in the section below. Those two cannot both be right — the MoE has fewer active parameters per token but not 4× fewer bytes to stream.

Reproduction, all on the same box with the same script and model:

| Condition | Ollama agg tps @ 1 | @ 8 |
|---|---:|---:|
| Idle GPU | 43.3 | 45.1 |
| 20 GB of GTT held by another process | 43.7 | 45.3 |
| Another process issuing continuous matmuls | **10.9** | **11.0** |
| *Originally recorded in this file* | *9.0* | *9.2* |

Memory pressure is not the mechanism — with 20 GB of GTT held elsewhere, Ollama still reports the model fully GPU-resident (`size_vram == size`) and loses nothing. **Compute contention is:** a second process actively using the iGPU reproduces the recorded numbers to within ~20%, including the flat-across-concurrency shape.

Lesson for anyone re-running this: on a single-GPU box, benchmark one engine at a time and stop the other engine's server first. `concurrent_throughput.py` requires both endpoints to be live, which makes it easy to start the SGLang server and begin benchmarking before its warm-up, autotuning, and graph capture have quiesced — the Ollama phase runs first, so it absorbs all of it.

### Tuning history (Qwen3.5-4B, 8 concurrent)

| Configuration | agg tps @ 8 | Δ vs baseline |
|---|---:|---:|
| Initial (CUDA graphs off, FP16 cast, no env tuning) | 116.7 | — |
| TunableOp + dev kernarg + AOTriton + native BF16 + CUDA graphs ≤ bs8 | **159.9** | **+37%** |

## Concurrent throughput — Qwen3.5-35B-A3B (MoE, A3B = 3.3B active)

Same model family on both engines: `qwen3.5:35b-a3b` (GGUF, ~24 GB) on Ollama, `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (compressed-tensors AWQ, ~23 GB) on SGLang. 80-token generations, identical prompts, `enable_thinking=false`. SGLang runs require the [wave32 WARP_SIZE patch](../patches/04-warp-size-wave32.md); without it the first MoE forward GPU-faults. **SGLang numbers below are warm TunableOp cache** — see "Cache-mount caveat" below.

| Concurrent streams | Ollama agg tps | SGLang agg tps | per-stream tps (SGLang) | SGLang advantage |
|---:|---:|---:|---:|---:|
| 1 | 37.8 | 23.4 | 23.4 | 0.62× |
| 4 | 37.8 | 73.2 | 18.3 | 1.94× |
| 8 | 39.1 | **131.1** | 16.4 | **3.35×** |

**Reading:** Ollama still wins single-stream (1.6×) — llama.cpp ships hand-tuned HIP MoE kernels and a near-zero dispatch path. But SGLang's continuous batching holds per-stream throughput nearly flat (23.4 → 16.4 tps from 1 → 8 streams) while Ollama collapses to per-stream serialized rate (4.9 tps at 8 concurrent), so SGLang reaches **3.35× Ollama at 8 streams**. The remaining single-stream gap is in the MoE Triton kernels themselves; tuning per-shape configs for gfx1151 is the obvious next lever (synthetic tuning hurt — see "What didn't work" below).

### Cache-mount caveat

The headline numbers depend on a persistent TunableOp cache. The image enables TunableOp and points it at `/root/.tunableop/tunableop_results.csv`, but if you don't mount that path as a volume, the cache is wiped on every restart and single-stream collapses to ~11.6 tps (cold). `start-sglang.sh` and `compose.yaml` both mount it automatically; ad-hoc `docker run` invocations need:

```
-v ~/.cache/strix-halo-sglang-tunableop:/root/.tunableop
```

Cache is small (~17 KB) and is populated by the first ~10 requests after a fresh start.

### What didn't work (combo tests on top of warm TunableOp)

Tested separately to see if they stack with the warm cache. None did — the warm cache covers the attention/projection bottleneck, and the remaining time is in MoE GEMMs where these knobs don't help.

| Knob | Single-stream agg tps | vs warm baseline |
|---|---:|---:|
| Warm TunableOp (baseline) | 23.4 | — |
| + Tuned MoE config (batch_size=1 only) | 19.8 | **-15%** (synthetic-tuned tile picks BSM=32, bad fit for real M=8 decode) |
| + `--moe-runner-backend aiter` | 9.9 (cold) / no warm rerun | within noise |
| + `--enforce-piecewise-cuda-graph` | tested cold only; 100s capture time | within noise |

The MoE-tuning regression is interesting: the upstream `tuning_fused_moe_triton.py` benchmarks the full forward with random expert routing and picks a configuration that minimizes that wall-clock — but at decode time real models hit the kernel with a much smaller effective M than the tuner's synthetic uniform distribution implies. Larger tile sizes that look fast on synthetic traces just waste compute on padding in production. Would need topk-id-driven tuning (sgl-kernel ships `tuning_fused_moe_triton_sep.py` for this) to fix.

### Single-stream gap: three hypotheses, all tested, all negative (2026-08-16)

Same server, same warm TunableOp cache, same model, one variable at a time.

| Hypothesis | Change tested | Result @ 1 stream | Verdict |
|---|---|---:|---|
| Eager RMSNorm is most of the gap | Fused Triton RMSNorm vs `forward_native`, microbenchmarked | would save ~0.9 ms of a ~43 ms token | **No** — 5.4% of decode even if made free |
| Launch overhead is the gap | CUDA graphs on (`--cuda-graph-max-bs 4`) vs `--disable-cuda-graph` | 23.1 vs 23.4 tps | **No** — within noise |
| MoE tiles are M-padded at decode | `BLOCK_SIZE_M` 64 → 16 for M ≤ 32 | 22.5 vs 23.1 tps | **No** — 2–5% *worse* |

Notes on each:

- **RMSNorm.** See [`rmsnorm_micro.py`](rmsnorm_micro.py) and [patch 2](../patches/02-layernorm-native-fallback.md). Real 1.6× win on the kernel at decode shapes, 8× at prefill shapes — but too small a slice of the token budget to move the headline.
- **CUDA graphs.** Capture takes 8 s and costs **0.60 GB**, not the "extra memory" the run guide implied. They simply don't help: per-op dispatch is already hidden behind GPU execution, which is itself evidence the GPU is the bottleneck. `--disable-cuda-graph` in the AWQ run guide is therefore about memory headroom on smaller GTT pools, not throughput.
- **MoE tile shape.** The default for `int4_w4a16` with M ≤ E is `BLOCK_SIZE_M=64`, so at decode ~63/64 of the M tile is padding — which looks like an obvious waste and isn't. Shrinking it doesn't help because the decode MoE GEMM is bound by streaming each active expert's int4 weights, and that traffic is identical at any M tile. Smaller tiles just cost occupancy. This independently explains the `-15%` synthetic-tuning regression above: the tuner was chasing the wrong axis, not merely mistuning it.

**Where the gap actually is:** at 23 tps the model spends ~43 ms per token, while streaming 3.3 B active params at 4 bits (~1.65 GB) over ~240 GB/s implies a ~7 ms floor. Both engines are far off that roof; Ollama is ~1.6× closer. The remaining suspects are the attention path (Triton fallback, since aiter's CK templates assume wave64) and the int4 MoE GEMM itself. Those are kernel rewrites, not flags — which is the honest answer to "why not just tune it."

Run with `--mem-fraction-static 0.55 --context-length 2048 --max-total-tokens 4096 --max-mamba-cache-size 32 --disable-cuda-graph` and `-v $TUNABLE_DIR:/root/.tunableop`; the SGLang server is sized to fit on a 61.7 GB GTT pool. See [docs/RUNNING_AWQ_MOE.md](../docs/RUNNING_AWQ_MOE.md) for a full guide.

## What the single-stream gap actually is (2026-08-16)

Short version: **it is bytes per token, not the engine.** The 35B comparison above pairs a
checkpoint that quantizes only the routed experts against a GGUF that quantizes everything,
so it measures checkpoints more than it measures SGLang vs llama.cpp.

### Kernel-level profile

`rocprofv3 --kernel-trace --stats` over 400 decode tokens (torch/kineto reports no GPU
activity in this build — use rocprof, not the torch profiler):

| Kernel | Share | Calls/step | µs/call | ms/token |
|---|---:|---:|---:|---:|
| `Cijk_...MT16x16x64` (rocBLAS GEMM) | 33.2% | 113 | 157.0 | 17.7 |
| `fused_moe_kernel_gptq_awq` | 24.8% | 80 | 165.2 | 13.2 |
| `Cijk_...MT32x16x128` | 6.7% | 52 | 68.3 | 3.6 |
| `rocblas_gemvt` / `gemvn` | 2.5% | 109 | ~12 | 0.9 |
| `fused_recurrent_gated_delta_rule` (GDN decode) | 1.1% | 30 | 20.1 | 0.6 |

Those GEMMs are **not** badly chosen. Resolving the big dispatches against their operand
sizes gives 165–197 GB/s of effective bandwidth — 70–82% of this APU's ~240 GB/s peak:

| Dispatch | bf16 bytes | µs | Effective BW |
|---|---:|---:|---:|
| `lm_head` (248320×2048) | 1.02 GB | 5147 | 197 GB/s |
| `linear_attn.in_proj_qkv` (12288×2048) ×30 | 50 MB | 276 | 182 GB/s |
| 2048×2048 projections ×80 | 8.4 MB | 51 | 165 GB/s |

### Bytes per token

The AWQ checkpoint's `quantization_config.ignore` list excludes `lm_head`, every
`self_attn.*`, every `linear_attn.in_proj_*` and every `mlp.shared_expert.*` — all bf16.
Only the routed experts are int4:

| Component | Precision | Per token |
|---|---|---:|
| Routed experts | int4 | 0.50 GB |
| `linear_attn.in_proj_qkv` ×30 | bf16 | 1.51 GB |
| `lm_head` | bf16 | 1.02 GB |
| self_attn / shared_expert / out_proj | bf16 | 0.67 GB |
| **Total** | | **3.70 GB** |

Ollama's Q4_K_M moves roughly 1.8 GB for the same token. Traffic ratio 2.06×; measured
speed ratio 1.62×.

### Precision-matched engine test

Same model (`Qwen3-0.6B`), same precision (bf16 on SGLang, fp16 GGUF on Ollama), one engine
resident at a time:

| Concurrent streams | Ollama fp16 | strix-halo-sglang bf16 | SGLang advantage |
|---:|---:|---:|---:|
| 1 | 106.3 | 87.1 | 0.82× |
| 8 | 185.6 | **725.7** | **3.91×** |

With the bits matched, the single-stream gap falls from 1.62× to 1.22×, and the residual is
fixed per-step overhead (~6.5 ms/token for SGLang vs ~4.4 ms for Ollama) rather than slower
kernels — on a 43 ms 35B step that overhead is ~5%, not 20%. On a 0.6B model SGLang's decode
step is ~90% GPU kernel time, so there is little dispatch overhead left to remove.

### Acting on it — quantized dense layers (patch 7)

Upstream, int4 dense Linear layers can't run on ROCm at all: `wNa16` is Marlin-only and Marlin is
CUDA-only. [Patch 7](../patches/07-wna16-rocm-linear.md) adds a `gptq_gemm` path, and
[`tools/quantize_nonexpert.py`](../tools/quantize_nonexpert.py) requantizes the bf16 non-expert
weights of the existing checkpoint (they're already bf16 on disk, so no base model download).

| Concurrent streams | experts-only int4 | + dense int4 | + `in_proj` | + `lm_head` (patch 8) | + MoE tiles | Ollama |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.4 | 24.9 | 29.7 | 33.4 | **39.6** | 37.8 |
| 4 | 72.4 | 79.7 | 97.8 | 94.6 | **124.2** | 37.8 |
| 8 | 127.0 | 141.6 | 137.6 | 180.8 | **199.3** | 39.1 |

Bytes per token: **3.70 GB → 1.69 GB** (measured from the checkpoint), just under Ollama's ~1.8 GB. Net single-stream gain **+69%** (23.4 → 39.7). See the clean head-to-head below.

### MoE tiles: it's workgroup count, not the M tile

With the dense layers quantized, the MoE kernel became **43.5% of the decode step** and was
running at **45 GB/s — 19% of the ~240 GB/s peak**, while the new int4 dense path was already at
**183 GB/s (76%)**. So all remaining headroom was in one kernel.

Shrinking `BLOCK_SIZE_M` never helped because at decode `tiles_m == 1` either way — it changes no
parallelism. The knobs that matter are `BLOCK_SIZE_N` and `num_warps`, which together set how many
workgroups launch. Upstream's defaults launch roughly 64 across a 40-CU part.

| BLOCK_SIZE_N | num_warps | single-stream tps |
|---:|---:|---:|
| 128 (upstream default) | 4 | 16.3 |
| 32 | 4 | 34.5 |
| 16 | 4 | 38.0 |
| **16** | **2** | **39.7** |
| 16 | 1 | 35.7 |
| 16 | 8 | 29.7 |

`num_warps=1` can't cover memory latency; 8 oversubscribes. Two is the optimum, reproduced three
times. `BLOCK_SIZE_K` 64 and 256 were both worse than 128; `GROUP_SIZE_M` made no difference.
Shipped in [`configs/moe/`](../configs/moe/).

**This is what took SGLang past Ollama on single-stream decode** — 34.5 → 39.6 tps, +15%.

### Optimizations tried that did NOT help (2026-08-17)

Swept at the post-patch operating point so the record is complete:

| Change | Result |
|---|---|
| `--num-continuous-decode-steps 2` / `4` | 34.5 / 34.4 vs 34.6 — no effect. The residual gap is not scheduler-loop overhead. |
| `--enable-torch-compile` | Crashes: inductor `PicklingError` |
| `HSA_NO_SCRATCH_RECLAIM=1` | 33.8 vs 33.4 — within run-to-run noise (baseline varies 33.4–34.6) |
| `GPU_MAX_HW_QUEUES=1` | 34.4 — within noise |
| ngram speculative decoding | **Blocked.** `--mamba-scheduler-strategy extra_buffer` asserts *"only supported on CUDA and MUSA and NPU devices"*; with the default `no_buffer` the server starts but every request returns HTTP 500. Note the guard is `device.startswith("cuda")`, which is **true on ROCm**, so it passes validation and fails at runtime. |

### What's left

At 39.6 tps the step is ~25 ms. The MoE kernel is still the largest single item and still well off
its bandwidth roof, so a better int4 MoE GEMM remains the main lever. Everything cheap has been
tried; what's left is kernel work, not configuration.


## Single-stream decode — Qwen3.5-4B

Same setup, sequential requests only. **Direction holds, magnitude is suspect** — Ollama's clean single-stream rate on this model is ~43 tps (see the postmortem above), so the 21.7–26.6 tps recorded here was likely measured under partial contention too. Ollama's win over SGLang is therefore probably *larger* than shown, not smaller. Pending re-run.

| Workload | Ollama (median) | strix-halo-sglang (median) | Notes |
|---|---:|---:|---|
| short (20 tok) | 922 ms / 21.7 tps | 1214 ms / 16.5 tps | Ollama 1.3× faster |
| long-prefix (30 tok) | 1587 ms / 18.9 tps | 1921 ms / 15.6 tps | Ollama 1.2× faster |
| codegen (200 tok) | 7528 ms / 26.6 tps | 11969 ms / 16.7 tps | Ollama 1.6× faster |

Ollama wins single-stream because llama.cpp ships hand-tuned HIP RMSNorm and Flash Attention for RDNA 3.5. SGLang on gfx1151 is currently kernel-bound — see [`docs/KNOWN_ISSUES.md`](../docs/KNOWN_ISSUES.md).

## Reproducing

```bash
# Server side
docker run -d --name sglang ... strix-halo-sglang:dev \
    python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-4B ...

# Bench side
python3 bench/concurrent_throughput.py \
    --ollama http://<ollama-host>:11434/v1/chat/completions \
    --sglang http://<sglang-host>:30000/v1/chat/completions
```


## Clean head-to-head, one engine at a time (2026-08-18)

Both re-measured in a single session with the other engine's server stopped — the methodology the
earlier contamination postmortem prescribes. Ollama model re-pulled (`qwen3.5:35b-a3b`).

| Concurrent streams | Ollama | strix-halo-sglang | ratio |
|---:|---:|---:|---:|
| 1 | 37.6 | **39.7** | 1.06× |
| 4 | 38.7 | **126.5** | 3.27× |
| 8 | 38.9 | **185.1** | 4.76× |

Ollama re-measured within 1% of the May baseline (37.8 / 37.8 / 39.1), which retroactively
validates that figure.

### ⚠️ Why this is parity, not a win

The single-stream margin is **6%**, and the two sides are not quantization-matched:

- **Ours:** every linear layer int4, group size 32, plain round-to-nearest. **21 GB** on disk,
  1.69 GB streamed per token.
- **Ollama:** Q4_K_M — mixed precision, mostly Q4_K with some tensors at Q6_K. **26.2 GB**
  resident.

So we move roughly 10–15% fewer bytes because we quantize harder, and we are roughly 6% faster.
That is consistent with the speed difference being explained by the quantization choice rather
than by the engine.

**No quality evaluation has been done** — correctness checking stopped at "produces coherent text
and gets 17 × 23 right." The quantizer reported up to **14% relative error** on one `v_proj`
tensor. A perplexity comparison against the bf16 model, and against Ollama's Q4_K_M, is the
missing piece before any performance claim here is meaningful.

The concurrency result (4.76×) does not depend on this caveat: it is a scheduler property, and it
holds regardless of how either side is quantized.
