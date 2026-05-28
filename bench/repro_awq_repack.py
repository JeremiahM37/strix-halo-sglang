"""End-to-end check: AWQ-stored layout → repack → fused_moe_kernel_gptq_awq.

If the kernel output matches a dequant-then-bf16-matmul reference within tolerance,
the repack math is correct and we can wire it into AWQMoEScheme.process_weights_after_loading.

Usage:
    python3 repro_awq_repack.py
"""

import sys
import torch

NUM_EXPERTS = 16   # smaller than Qwen3.5-35B (256) for fast iteration
TOP_K = 2
HIDDEN = 256       # K
INTERMEDIATE = 128 # N
GROUP_SIZE = 64
AWQ_PACK = 8
KERNEL_PACK = 2

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def quantize_awq_layout(w_logical, group_size):
    """Quantize `[OUT, IN]` bf16 → AWQ-stored `[IN, OUT // 8]` int32 + scales + zeros.

    Per-group symmetric-with-zero-point quantization, mirroring AWQ's storage convention:
      qweight: [IN, OUT // 8] int32, 8 nibbles per int32 along OUT using reverse_awq_order
      scales:  [IN // G, OUT] params_dtype
      qzeros:  [IN // G, OUT // 8] int32
    """
    OUT, IN = w_logical.shape
    assert IN % group_size == 0
    NUM_GROUPS = IN // group_size

    # group along IN: reshape to [OUT, NUM_GROUPS, group_size]
    w_g = w_logical.transpose(0, 1).reshape(NUM_GROUPS, group_size, OUT)  # [G, gs, OUT]

    w_max = w_g.amax(dim=1)  # [G, OUT]
    w_min = w_g.amin(dim=1)
    scales = (w_max - w_min).clamp(min=1e-5) / 15.0   # [G, OUT]
    zeros = torch.round(-w_min / scales).clamp(0, 15).to(torch.int32)  # [G, OUT]

    # Quantize: q = round(w / s) + zp, clamp to [0, 15]
    q = torch.round(w_g / scales.unsqueeze(1)).to(torch.int32) + zeros.unsqueeze(1)
    q = q.clamp(0, 15)  # [G, gs, OUT]

    # Reshape back to [IN, OUT]
    q_full = q.reshape(IN, OUT)  # int4 values 0..15

    # Pack 8 nibbles into int32 along OUT axis, using AWQ's reverse_awq_order
    # logical[i] should end up at storage nibble position reverse_awq_order[i]
    OUT_PACKED = OUT // AWQ_PACK
    q_grouped = q_full.reshape(IN, OUT_PACKED, AWQ_PACK)  # [IN, OUT//8, 8] (logical)
    packed = torch.zeros((IN, OUT_PACKED), dtype=torch.int32, device=w_logical.device)
    for i, storage_pos in enumerate(AWQ_REVERSE_ORDER):
        packed |= q_grouped[..., i].to(torch.int32) << (storage_pos * 4)

    # Pack zeros the same way
    zeros_packed = torch.zeros((NUM_GROUPS, OUT_PACKED), dtype=torch.int32, device=w_logical.device)
    z_grouped = zeros.reshape(NUM_GROUPS, OUT_PACKED, AWQ_PACK)
    for i, storage_pos in enumerate(AWQ_REVERSE_ORDER):
        zeros_packed |= z_grouped[..., i].to(torch.int32) << (storage_pos * 4)

    return packed, scales.to(w_logical.dtype), zeros_packed


def repack_awq_to_triton_moe(qweight_awq, qzeros_awq, scales_awq):
    """AWQ-stored → Triton kernel layout.

    Inputs:
      qweight_awq: [E, IN, OUT // 8] int32
      qzeros_awq:  [E, IN // G, OUT // 8] int32
      scales_awq:  [E, IN // G, OUT]

    Outputs:
      qweight: [E, OUT, IN // 2] uint8
      qzeros:  [E, OUT // 2, IN // G] uint8
      scales:  [E, OUT, IN // G]
    """
    E, IN, OUT_PACK8 = qweight_awq.shape
    OUT = OUT_PACK8 * AWQ_PACK
    NUM_GROUPS = scales_awq.shape[1]

    rev = torch.tensor(AWQ_REVERSE_ORDER, device=qweight_awq.device)
    shifts = rev * 4  # [8]

    # ---- weights: unpack -> transpose -> repack ----
    # [E, IN, OUT//8] -> [E, IN, OUT//8, 8] -> [E, IN, OUT]
    unpacked = ((qweight_awq.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    unpacked = unpacked.reshape(E, IN, OUT)             # [E, IN, OUT]
    unpacked = unpacked.transpose(1, 2).contiguous()    # [E, OUT, IN]
    # pack 2 nibbles per byte along IN, low-nibble = even index
    qweight_k = (unpacked[..., 1::2] << 4) | unpacked[..., ::2]  # [E, OUT, IN//2]

    # ---- zeros: unpack -> transpose -> repack along OUT ----
    unp_z = ((qzeros_awq.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    unp_z = unp_z.reshape(E, NUM_GROUPS, OUT)           # [E, NUM_GROUPS, OUT]
    unp_z = unp_z.transpose(1, 2).contiguous()          # [E, OUT, NUM_GROUPS]
    qzeros_k = (unp_z[:, 1::2, :] << 4) | unp_z[:, ::2, :]  # [E, OUT//2, NUM_GROUPS]

    # ---- scales: just transpose ----
    scales_k = scales_awq.transpose(1, 2).contiguous()  # [E, OUT, NUM_GROUPS]

    return qweight_k, qzeros_k, scales_k


def dequantize_awq(qweight_awq, qzeros_awq, scales_awq, group_size):
    """Dequantize AWQ-stored back to logical [E, OUT, IN] bf16."""
    E, IN, OUT_PACK8 = qweight_awq.shape
    OUT = OUT_PACK8 * AWQ_PACK
    rev = torch.tensor(AWQ_REVERSE_ORDER, device=qweight_awq.device)
    shifts = rev * 4
    q = ((qweight_awq.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)
    q = q.reshape(E, IN, OUT)
    z = ((qzeros_awq.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)
    z = z.reshape(E, IN // group_size, OUT)
    # broadcast scales/zeros over group_size
    z_full = z.repeat_interleave(group_size, dim=1)         # [E, IN, OUT]
    s_full = scales_awq.repeat_interleave(group_size, dim=1) # [E, IN, OUT]
    w = (q - z_full).to(s_full.dtype) * s_full
    return w.transpose(1, 2).contiguous()                    # [E, OUT, IN]


def torch_moe_reference(a, w13, w2, topk_ids, topk_weights):
    """Reference MoE using full-precision dequantized weights.

    w13: [E, 2*N, K] bf16 (gate+up stacked along output)
    w2:  [E, K, N]   bf16
    """
    M, K = a.shape
    M_TK = M * topk_ids.shape[1]
    E, twoN, _ = w13.shape
    N = twoN // 2
    a_exp = a.repeat_interleave(topk_ids.shape[1], dim=0)  # [M*TK, K]
    ids = topk_ids.reshape(-1)
    weights = topk_weights.reshape(-1)
    out = torch.zeros(M_TK, K, device=a.device, dtype=a.dtype)
    for e in range(E):
        mask = ids == e
        if not mask.any():
            continue
        x = a_exp[mask]                # [tok, K]
        gu = x @ w13[e].T              # [tok, 2*N]
        gate, up = gu[:, :N], gu[:, N:]
        h = torch.nn.functional.silu(gate) * up   # [tok, N]
        out[mask] = h @ w2[e].T        # [tok, K]
    out = out * weights.unsqueeze(-1)
    out = out.reshape(M, topk_ids.shape[1], K).sum(dim=1)
    return out


def main():
    if not torch.cuda.is_available():
        print("FAIL: no GPU")
        return 1
    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    K = HIDDEN
    N = INTERMEDIATE
    TWO_N = 2 * N
    M = 8

    # 1) Generate logical full-precision weights
    w13 = (torch.randn(NUM_EXPERTS, TWO_N, K, device=dev, dtype=dtype) * 0.05)
    w2  = (torch.randn(NUM_EXPERTS, K,    N, device=dev, dtype=dtype) * 0.05)

    # 2) AWQ-quantize them
    q13, s13, z13 = [], [], []
    for e in range(NUM_EXPERTS):
        q, s, z = quantize_awq_layout(w13[e], GROUP_SIZE)
        q13.append(q); s13.append(s); z13.append(z)
    q2, s2, z2 = [], [], []
    for e in range(NUM_EXPERTS):
        q, s, z = quantize_awq_layout(w2[e], GROUP_SIZE)
        q2.append(q); s2.append(s); z2.append(z)
    qw13_awq = torch.stack(q13);   sc13_awq = torch.stack(s13);   zr13_awq = torch.stack(z13)
    qw2_awq  = torch.stack(q2);    sc2_awq  = torch.stack(s2);    zr2_awq  = torch.stack(z2)

    print(f"AWQ-stored shapes:")
    print(f"  qw13:  {tuple(qw13_awq.shape)} {qw13_awq.dtype}")
    print(f"  sc13:  {tuple(sc13_awq.shape)} {sc13_awq.dtype}")
    print(f"  zr13:  {tuple(zr13_awq.shape)} {zr13_awq.dtype}")

    # 3) Dequantize back (sanity: should be ~= w13/w2 within quant noise)
    w13_dq = dequantize_awq(qw13_awq, zr13_awq, sc13_awq, GROUP_SIZE)
    w2_dq  = dequantize_awq(qw2_awq,  zr2_awq,  sc2_awq,  GROUP_SIZE)
    err13 = (w13_dq - w13).abs().mean().item()
    err2  = (w2_dq  - w2).abs().mean().item()
    print(f"AWQ round-trip mean abs error: w13={err13:.4f}  w2={err2:.4f} (should be ~scales/16)")

    # 4) Repack to kernel layout
    qw13_k, zr13_k, sc13_k = repack_awq_to_triton_moe(qw13_awq, zr13_awq, sc13_awq)
    qw2_k,  zr2_k,  sc2_k  = repack_awq_to_triton_moe(qw2_awq,  zr2_awq,  sc2_awq)
    print(f"Kernel-layout shapes:")
    print(f"  qw13:  {tuple(qw13_k.shape)} {qw13_k.dtype}")
    print(f"  sc13:  {tuple(sc13_k.shape)} {sc13_k.dtype}")
    print(f"  zr13:  {tuple(zr13_k.shape)} {zr13_k.dtype}")

    # 5) Run the kernel
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    a = torch.randn(M, K, device=dev, dtype=dtype)
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=dev)
    topk_weights = torch.full((M, TOP_K), 1.0 / TOP_K, dtype=dtype, device=dev)

    print(f"\ninvoking fused_experts_impl ...")
    torch.cuda.synchronize()
    out_kernel = fused_experts_impl(
        hidden_states=a,
        w1=qw13_k, w2=qw2_k,
        topk_weights=topk_weights, topk_ids=topk_ids,
        inplace=False, activation="silu", is_gated=True,
        use_int4_w4a16=True,
        w1_scale=sc13_k, w2_scale=sc2_k,
        w1_zp=zr13_k, w2_zp=zr2_k,
        block_shape=[0, GROUP_SIZE],
    )
    torch.cuda.synchronize()
    print(f"kernel out:    shape={tuple(out_kernel.shape)} sum={out_kernel.float().sum().item():.4f}")

    # 6) Reference using DEQUANTIZED weights (so quant error is shared)
    out_ref = torch_moe_reference(a, w13_dq, w2_dq, topk_ids, topk_weights)
    print(f"reference out: shape={tuple(out_ref.shape)} sum={out_ref.float().sum().item():.4f}")

    err = (out_kernel.float() - out_ref.float()).abs()
    print(f"\nabs error: mean={err.mean().item():.4f}  max={err.max().item():.4f}")
    print(f"ref     : abs mean={out_ref.float().abs().mean().item():.4f}  max={out_ref.float().abs().max().item():.4f}")

    rel = err.mean().item() / max(out_ref.float().abs().mean().item(), 1e-6)
    print(f"relative mean error: {rel:.3f}")

    if rel < 0.05:
        print("\nPASS — repack math is correct.")
        return 0
    print("\nFAIL — kernel output disagrees with reference (>5% relative).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
