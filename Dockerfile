# strix-halo-sglang — SGLang for AMD Strix Halo (gfx1151)
#
# Build:   docker build -t strix-halo-sglang:dev .
# Run:     see README.md

# Override BASE_IMAGE to use a registry mirror when Docker Hub is unreachable,
# e.g. --build-arg BASE_IMAGE=dockerproxy.com/kyuz0/vllm-therock-gfx1151:stable
# See docs/BUILDING.md for details.
ARG BASE_IMAGE=kyuz0/vllm-therock-gfx1151:stable
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_FORCE_NATIVE_LAYERNORM=1
ENV HF_HOME=/root/.cache/huggingface
ENV PYTORCH_ROCM_ARCH=gfx1151

# Perf flags — measured ~38% throughput uplift on gfx1151 vs disabled defaults.
# TunableOp autotunes GEMM kernels per-shape; results cached at $PYTORCH_TUNABLEOP_FILENAME.
# Mount /root/.tunableop as a volume to persist tunings across container restarts.
ENV PYTORCH_TUNABLEOP_ENABLED=1
ENV PYTORCH_TUNABLEOP_FILENAME=/root/.tunableop/tunableop_results.csv
ENV HIP_FORCE_DEV_KERNARG=1
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

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

# Patch 1b — fix host/device WARP_SIZE mismatch in topk softmax/sigmoid sgl-kernels.
# Without an explicit definition, HIP's WARP_SIZE evaluates to different values
# on host vs device when targeting gfx1151 (wave32). This causes
# __launch_bounds__(WARPS*WARP_SIZE) to be compiled for 128 threads but launched
# with 256, raising hipErrorLaunchFailure and a downstream GPU page fault on the
# first MoE forward. The sibling moe_fused_gate.cu already hardcodes WARP_SIZE=32;
# replicate the same fix in the topk kernels. See patches/04-warp-size-wave32.md.
RUN for f in sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu; do \
      python3 -c "import sys, re; p=sys.argv[1]; t=open(p).read(); marker='// added: gfx1151 wave32 kStrixWarp'; \
        assert marker not in t, f'already patched: {p}'; \
        anchor='#include <torch/all.h>'; \
        assert anchor in t, f'anchor not found: {p}'; \
        # Rename WARP_SIZE -> kStrixWarp throughout the file so a HIP macro cannot shadow it. \
        t=re.sub(r'\bWARP_SIZE\b', 'kStrixWarp', t); \
        t=t.replace(anchor, anchor+'\n\n'+marker+'\nstatic constexpr int kStrixWarp = 32;', 1); \
        open(p,'w').write(t); print('patched', p)" "$f"; \
    done

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
# Loaded only when SGLANG_AWQ_MOE_TRITON_ROCM=1; defaults off until the repack
# helper below is validated end-to-end on hardware.
COPY patches/awq_moe_rocm_repack.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/awq/schemes/awq_moe_rocm_repack.py
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
            from .awq_moe_rocm_repack import repack_awq_moe_to_triton
            qw13, qz13, sc13 = repack_awq_moe_to_triton(
                layer.w13_qweight, layer.w13_qzeros, layer.w13_scales,
            )
            qw2, qz2, sc2 = repack_awq_moe_to_triton(
                layer.w2_qweight, layer.w2_qzeros, layer.w2_scales,
            )
            layer.w13_qweight = torch.nn.Parameter(qw13, requires_grad=False)
            layer.w13_qzeros  = torch.nn.Parameter(qz13, requires_grad=False)
            layer.w13_scales  = torch.nn.Parameter(sc13, requires_grad=False)
            layer.w2_qweight  = torch.nn.Parameter(qw2,  requires_grad=False)
            layer.w2_qzeros   = torch.nn.Parameter(qz2,  requires_grad=False)
            layer.w2_scales   = torch.nn.Parameter(sc2,  requires_grad=False)
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
                block_shape=[0, self.quant_config.group_size],
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
