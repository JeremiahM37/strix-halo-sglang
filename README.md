# strix-halo-sglang

[SGLang](https://github.com/sgl-project/sglang) Docker image for **AMD Ryzen AI Max+ 395 / Radeon 8060S** (gfx1151, "Strix Halo").

AMD's official `rocm/sgl-dev` images only target MI300/MI350 data-center GPUs. This image fills the gap for consumer Strix Halo hardware.

## Status

| Capability | Status |
|---|---|
| Server starts, loads model | ✅ |
| Chat completion, tool calling | ✅ |
| RadixAttention prefix caching | ✅ |
| Continuous batching | ✅ — **17.4× Ollama at 8 concurrent streams** |
| Single-stream decode | ⚠️ ~60% of Ollama (no aiter Flash Attention on RDNA 3.5 yet) |
| AWQ-MoE inference | ✅ — Qwen3.5-35B-A3B-AWQ-4bit works end-to-end after [patch 4](patches/04-warp-size-wave32.md) |

Tested on Fedora 43 host, ROCm 7.13 nightly, PyTorch 2.13.

## Tested models

| Model | Loads | Inference | GPU mem | Notes |
|---|:-:|:-:|---:|---|
| `Qwen/Qwen3-0.6B` | ✅ | ✅ | ~2 GB | Smoke test. Loads in seconds. |
| `Qwen/Qwen3.5-4B` | ✅ | ✅ | ~10 GB | Reference benchmark. 16.5 tps single-stream, 116 tps at 8 concurrent. |
| `Qwen/Qwen3.6-27B` | ✅ | ✅ | ~52 GB | Dense Mamba+attention hybrid, BF16. Comfortable on 96 GB+ GTT; on a 64 GB box squeeze it in with `--mem-fraction-static 0.96 --max-mamba-cache-size 16 --max-total-tokens 8192 --disable-cuda-graph`. ~1.7 tps single-stream — BF16 27B is GTT-bound on the iGPU. |
| `cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4` | ❌ | — | — | 4-bit (compressed-tensors wNa16) calls `gptq_marlin_repack`, which is NVIDIA-only — same wall as the GPTQ row below. Run the BF16 model above instead. |
| `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | ✅ | ✅ | ~23 GB | Mamba+MoE hybrid; needs `--max-total-tokens N --max-mamba-cache-size M` to fit. See [docs/RUNNING_AWQ_MOE.md](docs/RUNNING_AWQ_MOE.md). |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | ❌ | — | — | GPTQ-on-MoE needs the `gptq_marlin` backend, which is NVIDIA-only today. Use the AWQ variant above instead. |

Hardware tested: AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 61.7 GB GTT. If you've run this on a different Strix Halo SKU please open an issue with results.

## Quickstart

Build the image:

```bash
docker build -t strix-halo-sglang:dev .
```

Run it with the helper script:

```bash
./start-sglang.sh                          # defaults to Qwen3.5-4B
./start-sglang.sh Qwen/Qwen3.5-4B
```

Or with Docker Compose:

```bash
docker compose up -d
```

Or the long form if you prefer:

```bash
docker run -d --name sglang \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --network=host \
    --security-opt seccomp=unconfined \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/strix-halo-sglang-tunableop:/root/.tunableop \
    -e HF_TOKEN=$HF_TOKEN \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
    strix-halo-sglang:dev \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-4B \
        --host 0.0.0.0 --port 30000 \
        --mem-fraction-static 0.5 \
        --context-length 8192 \
        --attention-backend triton \
        --disable-cuda-graph
```

**Mount `~/.cache/strix-halo-sglang-tunableop:/root/.tunableop` — without it, the TunableOp cache is wiped on every restart and single-stream throughput collapses by ~50%.** `start-sglang.sh` and `compose.yaml` do this automatically; the long-form command above is the only invocation that needs the explicit volume.

Then:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"hi"}]}'
```

Build takes ~10 min on first run; details in [docs/BUILDING.md](docs/BUILDING.md).

## What's in the image

- Base: [`kyuz0/vllm-therock-gfx1151:stable`](https://hub.docker.com/r/kyuz0/vllm-therock-gfx1151)
- PyTorch 2.13.0a0 + ROCm 7.13 (TheRock nightly)
- AITER pre-built for gfx1151
- SGLang pinned to a verified commit (see `SGL_BRANCH` in the [Dockerfile](Dockerfile)), built from source
- `sgl-kernel` compiled with `--amdgpu-target=gfx1151`

## Patches applied

Four small patches let SGLang run on gfx1151:

1. **`sgl-kernel/setup_rocm.py`** — allow `gfx1151` in the arch list (upstream guards against anything but `gfx942`/`gfx950`).
2. **`sglang/srt/layers/layernorm.py`** — `SGLANG_FORCE_NATIVE_LAYERNORM=1` skips the aiter (CDNA-only inline asm) and vLLM (older 4-arg signature) RMSNorm paths.
3. **`sglang/srt/layers/quantization/awq/schemes/awq_moe.py`** — `SGLANG_AWQ_MOE_TRITON_ROCM=1` routes AWQ MoE through SGLang's existing Triton path on ROCm.
4. **`sgl-kernel/csrc/moe/moe_topk_{softmax,sigmoid}_kernels.cu`** — fix host/device `WARP_SIZE` mismatch that caused `hipErrorLaunchFailure` → GPU page fault on the first MoE forward pass for any wave32 (RDNA 3.5) target. **This is the patch that unlocks AWQ-MoE inference on consumer Strix Halo.** Lives downstream only — SGLang's sgl-kernel arch guard explicitly targets `gfx942`/`gfx950` (CDNA) and rejects everything else, so the project isn't accepting consumer-RDNA fixes.

All four are baked into the Dockerfile by default. See [`patches/`](patches/) for the diffs.

## Benchmarks

Concurrent throughput on Qwen3.5-4B, same model on both engines:

| Concurrent streams | Ollama (llama.cpp) tps | strix-halo-sglang tps | SGLang advantage |
|---:|---:|---:|---:|
| 1 | 9.0 | 23.1 | 2.57× |
| 4 | 9.3 | 85.8 | 9.23× |
| 8 | 9.2 | 159.9 | **17.4×** |

Same family on the 35B MoE (`qwen3.5:35b-a3b` GGUF vs `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit`):

| Concurrent streams | Ollama tps | strix-halo-sglang tps | SGLang advantage |
|---:|---:|---:|---:|
| 1 | 37.8 | 23.4 | 0.62× |
| 4 | 37.8 | 73.2 | 1.94× |
| 8 | 39.1 | **131.1** | **3.35×** |

Ollama serializes; SGLang's continuous batching keeps per-stream throughput nearly flat (4B: 23.1 → 20.0 tps, 35B-A3B: 23.4 → 16.4 tps from 1 → 8 streams) while aggregate scales near-linearly. Full numbers + what-didn't-work table: [`bench/results.md`](bench/results.md). Script: [`bench/concurrent_throughput.py`](bench/concurrent_throughput.py).

The image enables TunableOp (`PYTORCH_TUNABLEOP_ENABLED=1`) by default — first request to each unique GEMM shape autotunes; results persist to a mounted volume at `/root/.tunableop`. Subsequent runs use the cached tunings (warm cache = the numbers above). **You must mount this path** — `start-sglang.sh` and `compose.yaml` do it automatically.

## Known limitations

- **AWQ MoE page fault — fixed.** Early builds crashed on the first MoE forward pass; the cause was a host/device `WARP_SIZE` mismatch in the topk gating kernels, fixed by [patch 4](patches/04-warp-size-wave32.md) (baked into the default build). Qwen3.5-35B-A3B-AWQ-4bit now runs end-to-end — see [docs/RUNNING_AWQ_MOE.md](docs/RUNNING_AWQ_MOE.md). The debugging record lives in [docs/AWQ_MOE_DEBUG.md](docs/AWQ_MOE_DEBUG.md).
- **No aiter Flash Attention on gfx1151.** aiter's MHA kernels use Composable Kernel templates that assume wave64; gfx1151 is wave32. Falls back to Triton attention (slower).
- **No aiter RMSNorm.** `rmsnorm_quant_kernels.cu` uses CDNA-only `v_pk_mul_f32` inline asm.
- **CUDA graphs engage but the bottleneck is in-kernel.** No measurable speedup on this stack today.

## Why this exists

The `strix-halo-toolboxes` ecosystem (notably [kyuz0's vLLM toolbox](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes)) covers vLLM, PyTorch, ComfyUI, fine-tuning, and voice. SGLang has been the missing piece. SGLang's RadixAttention prefix caching and continuous batching are exactly what consumer homelab AI servers need — multiple agents and users hitting the same model concurrently.

## Acknowledgments

- **[kyuz0](https://github.com/kyuz0)** for `vllm-therock-gfx1151` — the base image, AITER build pattern, and proof that this is possible.
- **[paudley/ai-notes](https://github.com/paudley/ai-notes)** for the original Strix Halo build research.
- **[sgl-project/sglang](https://github.com/sgl-project/sglang)** maintainers.
- **AMD's TheRock** team for the ROCm 7.13 nightlies that made gfx1151 PyTorch viable.

## License

Apache 2.0. See [LICENSE](LICENSE).
