"""ROCm AWQ-MoE weight repack: AWQ-stored layout -> Triton fused_moe kernel layout.

AWQ stores quantized MoE weights as [E, IN, OUT // 8] int32, with 8 int4 nibbles
packed along OUT using `reverse_awq_order=[0, 4, 1, 5, 2, 6, 3, 7]`. SGLang's
existing Triton `fused_moe_kernel_gptq_awq` expects [E, OUT, IN // 2] uint8 with
2 nibbles per byte along IN (natural order: byte = nibble[1] << 4 | nibble[0]).

On NVIDIA, the AWQ-MoE dispatcher repacks to Marlin's tiled layout via
`awq_marlin_moe_repack`. Marlin is Cutlass-based and not available on ROCm, so
we repack to the kernel-native layout instead.

Verified bit-exact against a dequant -> bf16-matmul reference (see
`bench/repro_awq_repack.py`).
"""

from typing import Tuple

import torch

# AWQ packs 8 int4 values per int32. The logical i-th value lives at storage
# nibble `_AWQ_REVERSE_ORDER[i]` (i.e. bit position `_AWQ_REVERSE_ORDER[i] * 4`).
_AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
_AWQ_PACK = 8


def _unpack_awq_int32(packed: torch.Tensor) -> torch.Tensor:
    """[..., L_packed] int32 -> [..., L_packed * 8] uint8 (logical order)."""
    shifts = torch.tensor(_AWQ_REVERSE_ORDER, device=packed.device) * 4  # [8]
    # Broadcast-shift across 8 nibble positions.
    expanded = ((packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    out_shape = packed.shape[:-1] + (packed.shape[-1] * _AWQ_PACK,)
    return expanded.reshape(out_shape)


def repack_awq_moe_to_triton(
    qweight_awq: torch.Tensor,   # [E, IN, OUT // 8] int32
    qzeros_awq: torch.Tensor,    # [E, IN // G, OUT // 8] int32
    scales_awq: torch.Tensor,    # [E, IN // G, OUT]   bf16/fp16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert AWQ-stored MoE weights to fused_moe_kernel_gptq_awq layout.

    Returns:
        qweight: [E, OUT, IN // 2] uint8
        qzeros:  [E, OUT // 2, IN // G] uint8
        scales:  [E, OUT, IN // G] (params dtype)
    """
    E, IN, OUT_PACK8 = qweight_awq.shape
    OUT = OUT_PACK8 * _AWQ_PACK
    NUM_GROUPS = scales_awq.shape[1]
    assert scales_awq.shape == (E, NUM_GROUPS, OUT), (
        f"scales shape mismatch: got {tuple(scales_awq.shape)}, "
        f"expected {(E, NUM_GROUPS, OUT)}"
    )
    assert qzeros_awq.shape == (E, NUM_GROUPS, OUT_PACK8), (
        f"qzeros shape mismatch: got {tuple(qzeros_awq.shape)}, "
        f"expected {(E, NUM_GROUPS, OUT_PACK8)}"
    )

    # Weights: unpack along OUT, transpose so OUT is the leading axis, repack
    # along IN with 2 nibbles per byte.
    w_unpacked = _unpack_awq_int32(qweight_awq)             # [E, IN, OUT]
    w_unpacked = w_unpacked.transpose(1, 2).contiguous()    # [E, OUT, IN]
    qweight = (w_unpacked[..., 1::2] << 4) | w_unpacked[..., ::2]  # [E, OUT, IN // 2]

    # Zeros: unpack along OUT, transpose, repack along OUT with 2 nibbles per byte.
    z_unpacked = _unpack_awq_int32(qzeros_awq)              # [E, NUM_GROUPS, OUT]
    z_unpacked = z_unpacked.transpose(1, 2).contiguous()    # [E, OUT, NUM_GROUPS]
    qzeros = (z_unpacked[:, 1::2, :] << 4) | z_unpacked[:, ::2, :]  # [E, OUT // 2, NUM_GROUPS]

    # Scales: transpose only.
    scales = scales_awq.transpose(1, 2).contiguous()        # [E, OUT, NUM_GROUPS]

    return qweight, qzeros, scales
