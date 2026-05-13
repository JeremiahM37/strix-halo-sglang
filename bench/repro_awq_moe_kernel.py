"""Isolated reproducer for the fused_moe_kernel_gptq_awq GPU page fault on gfx1151.

Skips the full SGLang server stack so the kernel can be iterated on in ~5s
instead of ~1 min per server restart.

Builds synthetic AWQ-packed weights at realistic shapes and calls the same
Triton kernel SGLang dispatches for AWQ MoE inference. If the kernel
page-faults on gfx1151, it'll fault here too.

Usage:
    python3 repro_awq_moe_kernel.py
    python3 repro_awq_moe_kernel.py --tiles 16,32,64  # try smaller tiles
"""

import argparse
import sys

import torch

# Match the most common AWQ MoE shape — Qwen3.5-35B-A3B-class.
# Adjust if reproducing on a different model.
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 1024
GROUP_SIZE = 128
PACK_FACTOR = 8  # 8 int4s packed into one int32


def make_fake_weights(num_tokens, device="cuda", dtype=torch.bfloat16):
    """Synthetic packed-int4 weights + scales + zeros + routing."""
    # Activations (input)
    a = torch.randn(num_tokens, HIDDEN, device=device, dtype=dtype)

    # w13: gate+up projection weights, packed
    w13 = torch.randint(
        0, 2**31 - 1,
        (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE // PACK_FACTOR),
        dtype=torch.int32, device=device,
    )
    # w2: down projection weights, packed
    w2 = torch.randint(
        0, 2**31 - 1,
        (NUM_EXPERTS, INTERMEDIATE, HIDDEN // PACK_FACTOR),
        dtype=torch.int32, device=device,
    )

    # Per-group scales (one per group along K)
    s13 = torch.randn(
        NUM_EXPERTS, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE,
        device=device, dtype=dtype,
    ) * 0.01
    s2 = torch.randn(
        NUM_EXPERTS, INTERMEDIATE // GROUP_SIZE, HIDDEN,
        device=device, dtype=dtype,
    ) * 0.01

    # Per-group zero points, packed
    z13 = torch.randint(
        0, 2**31 - 1,
        (NUM_EXPERTS, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE // PACK_FACTOR),
        dtype=torch.int32, device=device,
    )
    z2 = torch.randint(
        0, 2**31 - 1,
        (NUM_EXPERTS, INTERMEDIATE // GROUP_SIZE, HIDDEN // PACK_FACTOR),
        dtype=torch.int32, device=device,
    )

    # Routing: random expert assignment, all-ones weights
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOP_K), dtype=torch.int32, device=device,
    )
    topk_weights = torch.full(
        (num_tokens, TOP_K), 1.0 / TOP_K, dtype=dtype, device=device,
    )

    return {
        "a": a, "w13": w13, "w2": w2,
        "s13": s13, "s2": s2, "z13": z13, "z2": z2,
        "topk_ids": topk_ids, "topk_weights": topk_weights,
    }


def run(num_tokens):
    """Call the actual SGLang fused MoE path with use_int4_w4a16=True."""
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
    except ImportError as e:
        print(f"FAIL: sglang import — is this running inside the strix-halo-sglang container? ({e})")
        return 1

    w = make_fake_weights(num_tokens)

    print(f"shapes:")
    print(f"  a:        {tuple(w['a'].shape)} {w['a'].dtype}")
    print(f"  w13:      {tuple(w['w13'].shape)} {w['w13'].dtype}")
    print(f"  s13:      {tuple(w['s13'].shape)} {w['s13'].dtype}")
    print(f"  topk_ids: {tuple(w['topk_ids'].shape)} {w['topk_ids'].dtype}")
    print()
    print(f"invoking fused_experts with use_int4_w4a16=True ...")
    torch.cuda.synchronize()

    out = fused_experts(
        hidden_states=w["a"],
        w1=w["w13"],
        w2=w["w2"],
        topk_weights=w["topk_weights"],
        topk_ids=w["topk_ids"],
        inplace=False,
        activation="silu",
        use_int4_w4a16=True,
        w1_scale=w["s13"],
        w2_scale=w["s2"],
        w1_zp=w["z13"],
        w2_zp=w["z2"],
        block_shape=[GROUP_SIZE, GROUP_SIZE],
    )
    torch.cuda.synchronize()

    print(f"OK: out shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"    sum={out.float().sum().item():.4f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=8,
                    help="batch size for prefill (default: 8)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no GPU detected. Run inside the container with /dev/kfd bound.")
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")
    print(f"num_tokens: {args.num_tokens}")
    print()

    return run(args.num_tokens)


if __name__ == "__main__":
    sys.exit(main())
