# Building from source

## Host prerequisites

- Linux host with a Strix Halo APU (Ryzen AI Max+ 395 / Radeon 8060S, `gfx1151`)
- Kernel 6.18+ recommended (older kernels have VGPR mismatch issues that cause hangs under load)
- Docker or Podman with GPU passthrough working (`--device=/dev/kfd --device=/dev/dri`)

A quick host sanity check:

```bash
rocminfo | grep -E 'Name:|gfx'
# Expect:
#   Name: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
#   Name: gfx1151
```

## Build

```bash
git clone https://github.com/<your-account>/strix-halo-sglang.git
cd strix-halo-sglang
docker build -t strix-halo-sglang:dev .
```

The build takes ~10 minutes on first run, mostly spent compiling `sgl-kernel` for gfx1151. Subsequent builds reuse layers.

## What the build does

1. Pulls `kyuz0/vllm-therock-gfx1151:stable` (PyTorch 2.13 + ROCm 7.13 + AITER, gfx1151-compiled).
2. Clones SGLang `main`.
3. Applies the three patches in [`patches/`](../patches/).
4. Compiles `sgl-kernel` with `AMDGPU_TARGET=gfx1151 python3 setup_rocm.py develop`.
5. Installs SGLang via `pyproject_other.toml` + `srt_hip` extra (avoids the NVIDIA-only PyPI wheels).

## Customizing

- **SGLang version:** `--build-arg SGL_BRANCH=v0.5.11` (default: `main`)
- **SGLang fork:** `--build-arg SGL_REPO=https://github.com/your/fork.git`

## Verifying the build

The build does a file-level check that `sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so` exists. A functional check (GPU op registration) needs the container running with `/dev/kfd` bound — do this on first launch:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --ipc=host \
    strix-halo-sglang:dev \
    python3 -c "
import sgl_kernel, torch
ops = [n for n in torch._C._dispatch_get_all_op_names() if n.startswith('sgl_kernel')]
print(f'{len(ops)} sgl_kernel ops registered')
print('cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
"
```

Expected output: 46+ ops, `Radeon 8060S Graphics`.

## LXC / unprivileged container hosts

For Proxmox LXC: the container needs `nesting=1` and `keyctl=1` features, plus `/dev/kfd` and `/dev/dri/{card0,renderD128}` bind-mounted. The `lxc.cgroup2.devices.allow` entries for char devices `226:0`, `226:128`, `234:0` are required for unprivileged passthrough. Then Docker inside the LXC works normally.
