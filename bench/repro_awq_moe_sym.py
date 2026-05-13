"""Page-fault repro for fused_moe_kernel_gptq_awq on gfx1151 — symmetric (no zp) path.

This mirrors how CompressedTensorsWNA16TritonMoE actually calls the kernel for
the Qwen3.5-35B-A3B-AWQ-4bit checkpoint: symmetric int4 (no zero-points), exact
shapes from the failing server.

Usage:
    python3 repro_awq_moe_sym.py                  # default shapes
    python3 repro_awq_moe_sym.py --has-zp          # add zero-points
    python3 repro_awq_moe_sym.py --n 512           # try a different N
"""

import argparse
import sys
import torch

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 2048      # K
GROUP_SIZE = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256,
                    help="moe intermediate size (kernel-warning value, default 256)")
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--has-zp", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no GPU")
        return 1

    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    K = HIDDEN
    N = args.n
    KERNEL_PACK = 2

    print(f"shapes: E={NUM_EXPERTS}, K={K}, N={N}, top_k={TOP_K}, has_zp={args.has_zp}")

    a = torch.randn(args.num_tokens, K, device=dev, dtype=dtype)

    # Kernel layout: [E, 2*N, K // 2] uint8 (K packed by 2)
    w13 = torch.randint(0, 256, (NUM_EXPERTS, 2 * N, K // KERNEL_PACK),
                        dtype=torch.uint8, device=dev)
    w2  = torch.randint(0, 256, (NUM_EXPERTS, K, N // KERNEL_PACK),
                        dtype=torch.uint8, device=dev)

    s13 = torch.randn(NUM_EXPERTS, 2 * N, K // GROUP_SIZE, device=dev, dtype=dtype) * 0.01
    s2  = torch.randn(NUM_EXPERTS, K,    N // GROUP_SIZE, device=dev, dtype=dtype) * 0.01

    if args.has_zp:
        z13 = torch.randint(0, 256, (NUM_EXPERTS, 2 * N // KERNEL_PACK, K // GROUP_SIZE),
                            dtype=torch.uint8, device=dev)
        z2  = torch.randint(0, 256, (NUM_EXPERTS, K // KERNEL_PACK, N // GROUP_SIZE),
                            dtype=torch.uint8, device=dev)
    else:
        z13 = None
        z2  = None

    topk_ids = torch.randint(0, NUM_EXPERTS, (args.num_tokens, TOP_K),
                             dtype=torch.int32, device=dev)
    topk_weights = torch.full((args.num_tokens, TOP_K), 1.0 / TOP_K,
                              dtype=dtype, device=dev)

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    print("calling fused_experts_impl ...")
    torch.cuda.synchronize()
    try:
        out = fused_experts_impl(
            hidden_states=a, w1=w13, w2=w2,
            topk_weights=topk_weights, topk_ids=topk_ids,
            inplace=False, activation="silu", is_gated=True,
            use_int4_w4a16=True,
            w1_scale=s13, w2_scale=s2,
            w1_zp=z13, w2_zp=z2,
            block_shape=[0, GROUP_SIZE],
        )
        torch.cuda.synchronize()
        print(f"OK: shape={tuple(out.shape)} sum={out.float().sum().item():.4f}")
        return 0
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
