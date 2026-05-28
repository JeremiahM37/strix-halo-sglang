# strix-halo-sglang

[SGLang](https://github.com/sgl-project/sglang) Docker image for **AMD Ryzen AI Max+ 395 / Radeon 8060S** (gfx1151, "Strix Halo").

AMD's official `rocm/sgl-dev` images only target MI300/MI350 data-center GPUs. This image fills the gap for consumer Strix Halo hardware.

## Status

| Capability | Status |
|---|---|
| Server starts, loads model | ✅ |
| Chat completion, tool calling | ✅ |
| RadixAttention prefix caching | ✅ |
| Continuous batching | ✅ — **12.7× Ollama at 8 concurrent streams** |
| Single-stream decode | ⚠️ ~60% of Ollama (no aiter Flash Attention on RDNA 3.5 yet) |
| AWQ-MoE inference | ❌ GPU page fault in `fused_moe_kernel_gptq_awq` — see [docs/AWQ_MOE_DEBUG.md](docs/AWQ_MOE_DEBUG.md) |

Tested on Fedora 43 host, ROCm 7.13 nightly, PyTorch 2.13.

## Quickstart

Build the image:

```bash
docker build -t strix-halo-sglang:dev .
```

Run it:

```bash
docker run -d --name sglang \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --network=host \
    --security-opt seccomp=unconfined \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
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
- SGLang `main`, built from source
- `sgl-kernel` compiled with `--amdgpu-target=gfx1151`

## Patches applied

Three small patches let SGLang run on gfx1151:

1. **`sgl-kernel/setup_rocm.py`** — allow `gfx1151` in the arch list (upstream guards against anything but `gfx942`/`gfx950`).
2. **`sglang/srt/layers/layernorm.py`** — `SGLANG_FORCE_NATIVE_LAYERNORM=1` skips the aiter (CDNA-only inline asm) and vLLM (older 4-arg signature) RMSNorm paths.
3. **`sglang/srt/layers/quantization/awq/schemes/awq_moe.py`** — `SGLANG_AWQ_MOE_TRITON_ROCM=1` routes AWQ MoE through SGLang's existing Triton `fused_moe_kernel_gptq_awq` instead of NVIDIA Marlin.

Patches #1 and #2 are baked into the Dockerfile by default. Patch #3 is included but loads only when the env var is set; the kernel page-faults on gfx1151 (see [AWQ_MOE_DEBUG.md](docs/AWQ_MOE_DEBUG.md)).

See [`patches/`](patches/) for the diffs.

## Benchmarks

Concurrent throughput on Qwen3.5-4B, same model on both engines:

| Concurrent streams | Ollama (llama.cpp) tps | strix-halo-sglang tps | SGLang advantage |
|---:|---:|---:|---:|
| 1 | 9.0 | 16.7 | 1.85× |
| 2 | 9.2 | 31.8 | 3.45× |
| 4 | 9.3 | 62.4 | 6.71× |
| 8 | 9.2 | 116.7 | **12.7×** |

Ollama serializes; SGLang's continuous batching keeps per-stream throughput nearly flat as concurrency rises. Full script: [`bench/concurrent_throughput.py`](bench/concurrent_throughput.py).

## Known limitations

- **AWQ MoE inference page-faults.** Model loads (23 GB for Qwen3.5-35B-A3B-AWQ), forward pass crashes. Debug plan in [docs/AWQ_MOE_DEBUG.md](docs/AWQ_MOE_DEBUG.md).
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

MIT. See [LICENSE](LICENSE).
