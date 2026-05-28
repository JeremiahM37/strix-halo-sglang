# Benchmark results

All runs on **AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151)**, ROCm 7.13 nightly, 61.7 GB GTT.

## Concurrent throughput — Qwen3.5-4B

Same model on both engines (`qwen3.5:4b` on Ollama, `Qwen/Qwen3.5-4B` on SGLang). 80-token generations, identical prompts. SGLang numbers are warm-cache (TunableOp results already tuned).

| Concurrent streams | Ollama agg tps | strix-halo-sglang agg tps | per-stream tps (SGLang) | SGLang advantage |
|---:|---:|---:|---:|---:|
| 1 | 9.0 | 23.1 | 23.1 | 2.57× |
| 4 | 9.3 | 85.8 | 21.4 | 9.23× |
| 8 | 9.2 | 159.9 | 20.0 | **17.4×** |

**Reading:** Ollama serializes — 8 concurrent streams yield the same aggregate as 1 stream. SGLang's continuous batching keeps per-stream throughput nearly flat (23.1 → 20.0 tps, 13% drop) while aggregate scales near-linearly.

### Tuning history (Qwen3.5-4B, 8 concurrent)

| Configuration | agg tps @ 8 | Δ vs baseline |
|---|---:|---:|
| Initial (CUDA graphs off, FP16 cast, no env tuning) | 116.7 | — |
| TunableOp + dev kernarg + AOTriton + native BF16 + CUDA graphs ≤ bs8 | **159.9** | **+37%** |

## Concurrent throughput — Qwen3.5-35B-A3B (MoE, A3B = 3.3B active)

Same model family on both engines: `qwen3.5:35b-a3b` (GGUF, ~24 GB) on Ollama, `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (compressed-tensors AWQ, ~23 GB) on SGLang. 80-token generations, identical prompts, `enable_thinking=false`. SGLang runs require the [wave32 WARP_SIZE patch](../patches/04-warp-size-wave32.md); without it the first MoE forward GPU-faults.

| Concurrent streams | Ollama agg tps | SGLang agg tps | per-stream tps (SGLang) | SGLang advantage |
|---:|---:|---:|---:|---:|
| 1 | 37.7 | 11.6 | 11.6 | 0.31× |
| 4 | 38.8 | 42.5 | 10.6 | 1.10× |
| 8 | 39.0 | **80.1** | 10.0 | **2.05×** |

**Reading:** Ollama wins single-stream by 3.3× — llama.cpp ships hand-tuned HIP MoE kernels and a near-zero dispatch path. SGLang's AWQ MoE goes through generic untuned Triton kernels (no `int4_w4a16` config file for gfx1151). But continuous batching keeps per-stream throughput nearly flat (11.6 → 10.0 tps from 1 → 8 streams), so SGLang doubles Ollama at 8 concurrent streams. Tuning the MoE Triton kernel configs for gfx1151 should close the single-stream gap.

Run with `--mem-fraction-static 0.55 --context-length 2048 --max-total-tokens 4096 --max-mamba-cache-size 32 --disable-cuda-graph`; the SGLang server is sized to fit on a 61.7 GB GTT pool. See [docs/RUNNING_AWQ_MOE.md](../docs/RUNNING_AWQ_MOE.md) for a full guide.

## Single-stream decode — Qwen3.5-4B

Same setup, sequential requests only.

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
