# strix-halo-sglang — SGLang for AMD Strix Halo (gfx1151)
#
# Build:   docker build -t strix-halo-sglang:dev .
# Run:     see README.md

FROM kyuz0/vllm-therock-gfx1151:stable

ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_FORCE_NATIVE_LAYERNORM=1
ENV HF_HOME=/root/.cache/huggingface
ENV PYTORCH_ROCM_ARCH=gfx1151

WORKDIR /sgl-workspace

ARG SGL_REPO=https://github.com/sgl-project/sglang.git
ARG SGL_BRANCH=main
RUN git clone --depth 1 -b ${SGL_BRANCH} ${SGL_REPO}

WORKDIR /sgl-workspace/sglang

# Patch 1 — allow gfx1151 in sgl-kernel's arch guard (see patches/01-allow-gfx1151.md).
RUN sed -i \
    -e 's|\["gfx942", "gfx950"\]|["gfx942", "gfx950", "gfx1151"]|' \
    -e "s|'gfx942' or 'gfx950'|'gfx942', 'gfx950', or 'gfx1151'|" \
    sgl-kernel/setup_rocm.py

# Patch 2 — RMSNorm native fallback on gfx1151 (see patches/02-layernorm-native-fallback.md).
RUN python3 - <<'PYEOF'
p = '/sgl-workspace/sglang/python/sglang/srt/layers/layernorm.py'
old = '''elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
    except ImportError:
        # Fallback: vllm not available, will use forward_native
        _has_vllm_rms_norm = False'''
new = '''elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
        import os as _os
        if _os.environ.get('SGLANG_FORCE_NATIVE_LAYERNORM', '0') == '1':
            _has_vllm_rms_norm = False
    except ImportError:
        _has_vllm_rms_norm = False'''
t = open(p).read()
assert old in t, 'layernorm.py: elif _is_hip block not found, upstream layout changed'
open(p, 'w').write(t.replace(old, new))
PYEOF

# Patch 3 — AWQ MoE Triton dispatcher on ROCm (see patches/03-awq-moe-triton-dispatch.md).
# Loaded only when SGLANG_AWQ_MOE_TRITON_ROCM=1; defaults off because the underlying
# fused_moe_kernel_gptq_awq currently page-faults on gfx1151 (see docs/AWQ_MOE_DEBUG.md).
RUN python3 - <<'PYEOF'
p = '/sgl-workspace/sglang/python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py'
t = open(p).read()
t = t.replace(
    'from sglang.srt.layers.moe import (',
    'import os\nfrom sglang.srt.utils import is_hip\nfrom sglang.srt.layers.moe import (',
    1,
)
t = t.replace(
    '''    def __init__(self, quant_config: "AWQMarlinConfig"):
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:''',
    '''    def __init__(self, quant_config: "AWQMarlinConfig"):
        self.quant_config = quant_config
        self._rocm_triton = is_hip() and os.environ.get("SGLANG_AWQ_MOE_TRITON_ROCM", "0") == "1"
        if self.quant_config.weight_bits != 4:''',
)
t = t.replace(
    '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)''',
    '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._rocm_triton:
            return
        self.kernel.process_weights_after_loading(layer)''',
)
t = t.replace(
    'self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)',
    '''backend = MoeRunnerBackend.TRITON if self._rocm_triton else MoeRunnerBackend.MARLIN
        self.kernel.runner = MoeRunner(backend, moe_runner_config)''',
)
t = t.replace(
    '''    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        return self.kernel.apply(layer, dispatch_output)''',
    '''    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        if self._rocm_triton:
            from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_qweight,
                w2_weight=layer.w2_qweight,
                use_int4_w4a16=True,
                w13_scale=layer.w13_scales,
                w2_scale=layer.w2_scales,
                w13_zp=layer.w13_qzeros,
                w2_zp=layer.w2_qzeros,
                block_shape=[self.quant_config.group_size, self.quant_config.group_size],
            )
            return self.kernel.runner.run(dispatch_output, quant_info)
        return self.kernel.apply(layer, dispatch_output)''',
)
open(p, 'w').write(t)
PYEOF

# Compile sgl-kernel for gfx1151
WORKDIR /sgl-workspace/sglang/sgl-kernel
RUN AMDGPU_TARGET=gfx1151 python3 setup_rocm.py develop

# Install SGLang via pyproject_other.toml (ROCm-safe deps, no NVIDIA wheels)
WORKDIR /sgl-workspace/sglang
RUN cp python/pyproject_other.toml python/pyproject.toml \
    && pip install -e 'python[srt_hip]' --no-build-isolation \
    && (pip cache purge 2>/dev/null || true)

# File-level verification (build host has no GPU; runtime check on container start).
RUN test -f /sgl-workspace/sglang/sgl-kernel/python/sgl_kernel/common_ops*.so

EXPOSE 30000

CMD ["python3", "-m", "sglang.launch_server", "--help"]
