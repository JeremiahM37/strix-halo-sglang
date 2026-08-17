"""Patch 7 -- give compressed-tensors wNa16 a non-Marlin int4 Linear path on ROCm.

SGLang's wNa16 scheme is Marlin-or-nothing, and Marlin is CUDA-only, so quantized dense
Linear layers cannot load on RDNA. This adds a ROCm branch that uses vLLM's gptq_gemm,
which is numerically correct on gfx1151 (verified cos=1.00000 with exllama + v1 zero points).

Why this is cheap: after `permute_param_layout_(input_dim=0, output_dim=1, packed_dim=0)`
the compressed-tensors weight is already `[K//8, N]` -- GPTQ's qweight layout -- and CT
packs nibbles as `q_signed + 8`, which is exactly GPTQ's unsigned encoding with zero=8.
So no bit-level repack is needed, only the layout normalisation Marlin already does.

Applies to a copy of the file; mount the result over the image path.
"""

import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/compressed_tensors_wNa16.py"
t = open(path).read()

# ---------------------------------------------------------------- imports
anchor_imp = """if _is_cuda:
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
"""
add_imp = '''if _is_cuda:
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
else:
    # gfx1151 patch 7: Marlin is CUDA-only. vLLM's ROCm build ships gptq_gemm /
    # gptq_shuffle and both run correctly on RDNA 3.5 via the exllama path.
    from vllm._custom_ops import gptq_gemm as _rocm_gptq_gemm
    from vllm._custom_ops import gptq_shuffle as _rocm_gptq_shuffle

    def _rocm_pack_cols_int4(t_):
        """[R, C] int32 values 0..15 -> [R, C//8] int32, 8 nibbles per word."""
        r, c = t_.shape
        out = torch.zeros(r, c // 8, dtype=torch.int32, device=t_.device)
        for j in range(8):
            out |= (t_[:, j::8] & 0xF) << (4 * j)
        return out
'''
assert t.count(anchor_imp) == 1, "import anchor not found"
t = t.replace(anchor_imp, add_imp)

# ------------------------------------------------- process_weights_after_loading
anchor_proc = """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Default names since marlin requires empty parameters for these,"""
add_proc = '''    def _process_weights_rocm(self, layer: torch.nn.Module) -> None:
        """gfx1151 patch 7: prepare GPTQ-layout tensors for gptq_gemm."""
        c = self.kernel_config
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"

        if c.zero_points:
            raise NotImplementedError(
                "patch 7 supports symmetric wNa16 only; this checkpoint has zero points"
            )
        if c.weight_type.size_bits != 4:
            raise NotImplementedError("patch 7 supports 4-bit wNa16 only")

        k, n = c.partition_weight_shape[0], c.partition_weight_shape[1]

        # Orient explicitly rather than trusting a helper: GPTQ wants qweight [K//8, N]
        # and scales [K//G, N]; compressed-tensors stores [N, K//8] / [N, K//G].
        qw = getattr(layer, self.w_q_name).data
        if tuple(qw.shape) == (n, k // 8):
            qw = qw.t()
        assert tuple(qw.shape) == (k // 8, n), f"unexpected w_q shape {tuple(qw.shape)}"
        qweight = qw.contiguous()

        sc = getattr(layer, self.w_s_name).data
        group_tmp = c.group_size if c.group_size != -1 else k
        if tuple(sc.shape) == (n, max(k // group_tmp, 1)):
            sc = sc.t()
        scales = sc.contiguous().to(torch.float16)   # gptq_gemm is fp16-only

        device = qweight.device
        group = c.group_size if c.group_size != -1 else k
        grouped_k = max(k // group, 1)

        # Symmetric CT stores q_signed + 8, i.e. GPTQ zero point 8; v1 format stores z - 1.
        zp = torch.full((grouped_k, n), 7, dtype=torch.int32, device=device)
        qzeros = _rocm_pack_cols_int4(zp)

        g_idx = torch.empty(0, dtype=torch.int, device=device)
        _rocm_gptq_shuffle(qweight, g_idx, 4)    # exllama path; the non-exllama path is wrong on ROCm

        layer.rocm_qweight = qweight
        layer.rocm_qzeros = qzeros
        layer.rocm_scales = scales
        layer.rocm_g_idx = g_idx

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _is_cuda:
            return self._process_weights_rocm(layer)
        # Default names since marlin requires empty parameters for these,'''
assert t.count(anchor_proc) == 1, "process anchor not found"
t = t.replace(anchor_proc, add_proc)

# ----------------------------------------------------------------- apply_weights
anchor_apply = """    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.kernel_config"""
add_apply = '''    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        if not _is_cuda:
            orig_dtype = x.dtype
            x2 = x.reshape(-1, x.shape[-1]).to(torch.float16)
            out = _rocm_gptq_gemm(
                x2,
                layer.rocm_qweight,
                layer.rocm_qzeros,
                layer.rocm_scales,
                layer.rocm_g_idx,
                True,   # use_exllama
                False,  # use_v2_format
                4,
            )
            out = out.to(orig_dtype).reshape(*x.shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out

        c = self.kernel_config'''
assert t.count(anchor_apply) == 1, "apply anchor not found"
t = t.replace(anchor_apply, add_apply)

open(path, "w").write(t)
print("patched", path)
