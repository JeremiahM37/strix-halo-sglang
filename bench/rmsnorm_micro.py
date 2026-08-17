"""Microbenchmark: eager (forward_native) RMSNorm vs a fused Triton RMSNorm on gfx1151.

Patch 2 forces SGLang's RMSNorm onto `forward_native` (eager PyTorch) because the aiter
kernel is CDNA-only and the vLLM one has a mismatched signature. This measures what that
actually costs at decode shapes, and what a fused Triton kernel would recover.

Run inside the strix-halo-sglang image:
    python3 rmsnorm_micro.py
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- eager path
def eager_rmsnorm(x, weight, eps, residual=None):
    """Mirrors sglang RMSNorm.forward_native: fp32 upcast, optional residual add."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return (x, residual) if residual is not None else x


# --------------------------------------------------------------- triton path
@triton.jit
def _rmsnorm_kernel(
    X, W, RES, OUT, RESOUT,
    stride, N, eps,
    HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    OUT += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(RES + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(RESOUT + row * stride + cols, x.to(OUT.dtype.element_ty), mask=mask)

    var = tl.sum(x * x, axis=0) / N
    x = x * tl.math.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + cols, (x * w).to(OUT.dtype.element_ty), mask=mask)


def triton_rmsnorm(x, weight, eps, residual=None):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    res2 = residual.reshape(-1, N) if residual is not None else x2
    resout = torch.empty_like(x2) if residual is not None else x2
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_kernel[(M,)](
        x2, weight, res2, out, resout,
        x2.stride(0), N, eps,
        HAS_RES=residual is not None, BLOCK=BLOCK,
        num_warps=8,
    )
    out = out.reshape(orig_shape)
    if residual is not None:
        return out, resout.reshape(orig_shape)
    return out


# ------------------------------------------------------------------ harness
def timeit(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # microseconds


def main():
    torch.manual_seed(0)
    dev = "cuda"
    eps = 1e-6

    # (label, hidden_size, n_layers) — 2 RMSNorms per transformer layer
    models = [
        ("Qwen3.5-4B", 2560, 36),
        ("Qwen3.5-35B-A3B", 4096, 48),
    ]
    batches = [1, 8, 512]

    print(f"{'model':18} {'M':>5} {'hidden':>7} "
          f"{'eager us':>10} {'triton us':>10} {'speedup':>8}  {'per-token saved':>16}")
    print("-" * 88)

    for label, hidden, n_layers in models:
        for M in batches:
            x = torch.randn(M, hidden, device=dev, dtype=torch.bfloat16)
            res = torch.randn(M, hidden, device=dev, dtype=torch.bfloat16)
            w = torch.randn(hidden, device=dev, dtype=torch.bfloat16)

            # numerics check (residual variant)
            e_out, e_res = eager_rmsnorm(x, w, eps, res.clone())
            t_out, t_res = triton_rmsnorm(x, w, eps, res.clone())
            ok = torch.allclose(e_out.float(), t_out.float(), atol=2e-2, rtol=2e-2) and \
                 torch.allclose(e_res.float(), t_res.float(), atol=2e-2, rtol=2e-2)
            if not ok:
                print(f"  !! numerics mismatch at {label} M={M}")

            te = timeit(lambda: eager_rmsnorm(x, w, eps, res.clone()))
            tt = timeit(lambda: triton_rmsnorm(x, w, eps, res.clone()))
            # 2 norms per layer
            saved_ms = (te - tt) * 2 * n_layers / 1000.0
            print(f"{label:18} {M:>5} {hidden:>7} {te:>10.1f} {tt:>10.1f} "
                  f"{te/tt:>7.2f}x  {saved_ms:>13.2f} ms")
        print()


if __name__ == "__main__":
    main()
