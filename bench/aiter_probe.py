"""Which aiter kernels work on gfx1151 (RDNA 3.5 / wave32)?

Ordered cheapest-and-most-informative first: symbol inventory (free), then
activations, then MoE helpers, then normalization LAST (module_rmsnorm is the
known-bad one and takes ~10 min of hipcc before it fails).
"""
import torch

import aiter

names = sorted(n for n in dir(aiter) if not n.startswith("_"))
print(f"aiter exposes {len(names)} top-level symbols", flush=True)

H, N = 2048, 8
dev = "cuda"


def probe(label, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        t = out[0] if isinstance(out, (tuple, list)) else out
        ok = t is None or torch.isfinite(t).all().item()
        print(f"  {'OK  ' if ok else 'NaN '} {label}", flush=True)
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: "
              f"{str(e).replace(chr(10), ' ')[:120]}", flush=True)


print("\n--- inventory (no build) ---", flush=True)
groups = {
    "moe": ("moe_sorting", "topk_softmax", "biased_grouped_topk", "fmoe", "ck_moe",
            "fmoe_int8_g1u0", "moe_stage1_g1u1"),
    "attention": ("flash_attn_func", "paged_attention_rocm", "mha_fwd", "pa_fwd_asm"),
    "gemm": ("gemm_a8w8", "gemm_a8w8_bpreshuffle", "gemm_a4w4", "gemm_tune"),
    "norm": ("rms_norm", "rmsnorm2d_fwd", "layer_norm", "rmsnorm2d_fwd_with_add"),
    "act": ("silu_and_mul", "gelu_and_mul", "scaled_silu_and_mul"),
}
for g, syms in groups.items():
    present = [s for s in syms if hasattr(aiter, s)]
    print(f"  {g:10} present: {present}", flush=True)

print("\n--- activations (builds module_activation) ---", flush=True)
if hasattr(aiter, "silu_and_mul"):
    xx = torch.randn(N, 2 * H, device=dev, dtype=torch.bfloat16)
    out = torch.empty(N, H, device=dev, dtype=torch.bfloat16)
    probe("silu_and_mul", lambda: (aiter.silu_and_mul(out, xx), out)[1])

print("\n--- moe helpers (builds module_moe*) ---", flush=True)
if hasattr(aiter, "topk_softmax"):
    E, TK = 256, 8
    gate = torch.randn(N, E, device=dev, dtype=torch.float32)
    tw = torch.empty(N, TK, device=dev, dtype=torch.float32)
    ti = torch.empty(N, TK, device=dev, dtype=torch.int32)
    tok = torch.empty(N, TK, device=dev, dtype=torch.int32)
    probe("topk_softmax", lambda: (aiter.topk_softmax(tw, ti, tok, gate, False), tw)[1])

print("\n--- normalization (module_rmsnorm; known-bad, slow) ---", flush=True)
x = torch.randn(N, H, device=dev, dtype=torch.bfloat16)
w = torch.randn(H, device=dev, dtype=torch.bfloat16)
if hasattr(aiter, "rms_norm"):
    probe("rms_norm", lambda: aiter.rms_norm(x, w, 1e-6))

print("\nDONE", flush=True)
