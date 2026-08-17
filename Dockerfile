# strix-halo-sglang — SGLang for AMD Strix Halo (gfx1151)
#
# Build:   docker build -t strix-halo-sglang:dev .
# Run:     see README.md

# Base is pinned by digest so `:stable` can't drift under us (same rationale as
# the SGL_BRANCH pin below). Override BASE_IMAGE to bump the base or to use a
# registry mirror when Docker Hub is unreachable,
# e.g. --build-arg BASE_IMAGE=mirror.gcr.io/kyuz0/vllm-therock-gfx1151:stable
# See docs/BUILDING.md for details.
ARG BASE_IMAGE=kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2
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
# Pinned to a commit verified against this base image. Unpinned `main` drifts,
# which is what broke fresh builds in issue #5. SGL_BRANCH accepts any ref —
# a branch, tag, or commit SHA — because we fetch+checkout rather than clone -b.
ARG SGL_BRANCH=b0b8436f1c031caba61c4cadb10d22ba097cd960
RUN git init sglang \
    && cd sglang \
    && git remote add origin ${SGL_REPO} \
    && git fetch --depth 1 origin ${SGL_BRANCH} \
    && git checkout FETCH_HEAD

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


def rep(old: str, new: str, what: str, count: int = -1) -> None:
    # Assert the anchor is present so upstream drift fails the build loudly
    # instead of silently skipping the patch (same style as patches 1b/2).
    global t
    assert old in t, f'awq_moe.py: {what} anchor not found, upstream layout changed'
    t = t.replace(old, new, count)


rep(
    'from sglang.srt.layers.moe import (',
    'import os\nfrom sglang.srt.utils import is_hip\nfrom sglang.srt.layers.moe import (',
    'moe import',
    1,
)
rep(
    '''    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:''',
    '''    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config
        self._rocm_triton = is_hip() and os.environ.get("SGLANG_AWQ_MOE_TRITON_ROCM", "0") == "1"
        if self.quant_config.weight_bits != 4:''',
    '__init__',
)
rep(
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
    'process_weights_after_loading',
)
rep(
    'self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)',
    '''backend = MoeRunnerBackend.TRITON if self._rocm_triton else MoeRunnerBackend.MARLIN
        self.kernel.runner = MoeRunner(backend, moe_runner_config)''',
    'MoeRunner MARLIN backend',
)
rep(
    '''    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ):
        return self.kernel.apply(layer, dispatch_output)''',
    '''    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
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
    'apply_weights',
)
open(p, 'w').write(t)
PYEOF

# Patch 5 — MoE tuner writes config files the runtime never reads for int4 MoE.
# The runtime keys its lookup on w2.shape[2] (`E, _, N = w2_shape`), but the tuner
# halves N a second time when use_int4_w4a16 is set. For Qwen3.5-35B-A3B-AWQ-4bit
# the tuner emits E=256,N=128,...int4_w4a16.json while the server looks up
# E=256,N=256 — so tuning an AWQ/GPTQ MoE silently no-ops. See
# patches/05-moe-tuner-n-mismatch.md.
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
t = open(p).read()
old = """        N = shard_intermediate_size // 2
        if use_int4_w4a16:
            N = N // 2
"""
new = """        N = shard_intermediate_size // 2
        # gfx1151 patch 5: do NOT halve again for int4. The runtime keys its
        # config lookup on w2.shape[2] without this extra shift, so halving here
        # writes a filename the runtime will never open.
"""
assert old in t, "patch 5: upstream tuner N computation changed, re-check"
open(p, "w").write(t.replace(old, new))
print("patched", p)
PYEOF

# Compile sgl-kernel for gfx1151
WORKDIR /sgl-workspace/sglang/sgl-kernel
RUN AMDGPU_TARGET=gfx1151 python3 setup_rocm.py develop

# Install SGLang via pyproject_other.toml (ROCm-safe deps, no NVIDIA wheels).
#
# Two guards stop the install from clobbering the base image's gfx1151-compiled
# torch/torchvision — the root cause of issue #5, where a fresh build pulled
# generic PyPI wheels and failed at runtime with
# `libc10_hip.so: cannot open shared object file` and
# `operator torchvision::nms does not exist`:
#
#   1. compressed-tensors' old `==0.15.0` pin requires torch<2.11. The base image
#      ships torch 2.13, so pip would resolve that conflict by *downgrading* torch
#      to a non-ROCm wheel. 0.16.0+ relaxes the bound to torch>=2.10, which the
#      base satisfies.
#   2. A pip constraints file freezes torch + torchvision to the exact ROCm builds
#      already present, so no dependency can replace them. A future incompatibility
#      then fails the build loudly instead of silently breaking at runtime.
WORKDIR /sgl-workspace/sglang
RUN cp python/pyproject_other.toml python/pyproject.toml \
    && sed -i 's/compressed-tensors==0.15.0/compressed-tensors>=0.16.0/' python/pyproject.toml \
    && python3 -c "import torch, torchvision; open('/tmp/rocm-constraints.txt', 'w').write(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n')" \
    && PIP_CONSTRAINT=/tmp/rocm-constraints.txt pip install -e 'python[srt_hip]' --no-build-isolation \
    && (pip cache purge 2>/dev/null || true)

# File-level verification (build host has no GPU; runtime check on container start).
RUN test -f /sgl-workspace/sglang/sgl-kernel/python/sgl_kernel/common_ops*.so

EXPOSE 30000

CMD ["python3", "-m", "sglang.launch_server", "--help"]
