# Running AWQ-MoE models on Strix Halo

AWQ MoE inference works on gfx1151 once the [wave32 `WARP_SIZE` patch](../patches/04-warp-size-wave32.md) is applied — that fix is already baked into the default image build. This page documents the runtime knobs needed to actually fit a 30B-class MoE on a 61.7 GB GTT pool.

## Tested model

`cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` — Qwen 3.5, 35B total / 3.3B active, 256 experts, top-k 8, hybrid attention + mamba.

## Launch command

```bash
docker run -d --name sglang-awq \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --network=host \
    --security-opt seccomp=unconfined \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/strix-halo-sglang-tunableop:/root/.tunableop \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
    strix-halo-sglang:dev \
    python3 -m sglang.launch_server \
        --model-path cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit \
        --host 0.0.0.0 --port 30000 \
        --mem-fraction-static 0.55 \
        --context-length 2048 \
        --max-total-tokens 4096 \
        --max-mamba-cache-size 32 \
        --attention-backend triton \
        --disable-cuda-graph
```

**The `-v ~/.cache/strix-halo-sglang-tunableop:/root/.tunableop` mount is not optional.** Without it, TunableOp tunes every GEMM shape from scratch on each restart, and single-stream throughput drops from ~23 tps to ~12 tps. The cache file is tiny (~17 KB after warmup).

## Why these flags

| Flag | Reason |
|---|---|
| `--max-total-tokens 4096` | SGLang sizes its KV pool from `torch.cuda.mem_get_info` on AMD, which reports the GTT-inclusive total (~115 GB on Strix Halo) rather than the actual PyTorch-allocatable cap (~61.7 GB). Without an explicit cap, it requests an oversized pool and OOMs during pool init. Cap at a value comfortably under the cap. |
| `--max-mamba-cache-size 32` | Qwen 3.5 A3B is a hybrid attention + mamba model. The default `max_mamba_cache_size=591` allocates a 35 GB SSM state pool that, combined with model weights (23 GB), leaves no room for KV cache. 32 entries is plenty for typical workloads. |
| `--mem-fraction-static 0.55` | Together with the token cap, this gives the model + mamba + KV pool enough headroom on a 61.7 GB GPU. |
| `--disable-cuda-graph` | Optional. CUDA graphs don't help on gfx1151 — measured 23.1 tps with graphs vs 23.4 without, i.e. noise — because the bottleneck is in-kernel and per-op dispatch is already hidden behind GPU execution. They are cheaper than this table used to claim (capture takes 8 s and 0.60 GB, not "extra memory" worth worrying about), so keep them if you like; drop them only if you are tight on a small GTT pool. |
| `--attention-backend triton` | aiter's flash-attention path uses CDNA-only Composable Kernel templates (wave64 assumptions). Triton fallback works correctly. |

`SGLANG_FORCE_NATIVE_LAYERNORM=1` is set in the image; it skips aiter's CDNA-only inline-asm RMSNorm.

## Expected memory layout

```
Load weight end. ... mem usage=22.84 GB.
Mamba Cache is allocated. max_mamba_cache_size: 32, conv_state size: 0.05GB, ssm_state size: 1.93GB
KV Cache is allocated. #tokens: 4096, K size: 0.04 GB, V size: 0.04 GB
```

Total GPU usage ≈ 25 GB. The remainder of the GTT pool is available for activations.

## Throughput

See [bench/results.md](../bench/results.md#concurrent-throughput--qwen35-35b-a3b-moe-a3b--33b-active) for numbers.

## Other AWQ MoE checkpoints

`compressed-tensors` AWQ checkpoints (e.g. anything quantized with `llmcompressor`) route through SGLang's `CompressedTensorsWNA16TritonMoE` scheme and work the same way once the WARP_SIZE patch is in place — no additional flag needed.

Raw AWQ format checkpoints (no `compressed-tensors`) go through `AWQMoEScheme`, which still defaults to NVIDIA's Marlin backend. Set `SGLANG_AWQ_MOE_TRITON_ROCM=1` to opt into the ROCm Triton path; this loads weights cleanly but the dispatcher needs the [included repack helper](../patches/awq_moe_rocm_repack.py) — currently included in the image but exercised only via that env var.
