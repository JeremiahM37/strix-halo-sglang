# Benchmark results

All runs on **AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151)**, ROCm 7.13 nightly, 61.7 GB GTT.

## Concurrent throughput — Qwen3.5-4B

Same model on both engines (`qwen3.5:4b` on Ollama, `Qwen/Qwen3.5-4B` on SGLang). 80-token generations, identical prompts.

| Concurrent streams | Ollama agg tps | strix-halo-sglang agg tps | per-stream tps (SGLang) | SGLang advantage |
|---:|---:|---:|---:|---:|
| 1 | 9.0 | 16.7 | 16.7 | 1.85× |
| 2 | 9.2 | 31.8 | 15.9 | 3.45× |
| 4 | 9.3 | 62.4 | 15.6 | 6.71× |
| 8 | 9.2 | 116.7 | 14.6 | **12.7×** |

**Reading:** Ollama serializes — 8 concurrent streams yield the same aggregate as 1 stream. SGLang's continuous batching keeps per-stream throughput nearly flat (16.7 → 14.6 tps, 12% drop) while aggregate scales near-linearly.

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
