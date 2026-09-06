#!/usr/bin/env python3
"""Fix aiter so Quark/MXFP4 checkpoints work on gfx1151 (Radeon 8060S).

Patches the base image's aiter installation in place:
  1. arch_info.is_fp4_avail() only whitelists gfx950 -> allow gfx1151.
  2. No gfx1151 GEMM configs are shipped; the gfx950 configs need 100KB of
     shared memory while RDNA 3.5 (wave32) only exposes 64KB. Copy every
     gfx950-*.json to gfx1151-*.json and clamp BLOCK_SIZE_N/K<=128,
     num_stages<=2 so the Triton kernel fits in 64KB.

Idempotent: existing gfx1151 configs are left alone, and the arch_info edit
is guarded by an assert, so this is safe to run on rebuilds.

See patches/09-aiter-gfx1151-mxfp4.md for details.
"""

import glob
import json
import os
import shutil

CFG = "/opt/venv/lib64/python3.12/site-packages/aiter/ops/triton/configs/gemm"
ARCH_INFO = "/opt/venv/lib64/python3.12/site-packages/aiter/ops/triton/utils/_triton/arch_info.py"


def patch_arch_info():
    with open(ARCH_INFO, encoding="utf-8") as f:
        t = f.read()
    old = 'def is_fp4_avail():\n    return get_arch() in ("gfx950")'
    new = 'def is_fp4_avail():\n    return get_arch() in ("gfx950", "gfx1151")'
    if new in t:
        print("is_fp4_avail already patched")
        return
    assert old in t, f"arch_info anchor not found in {ARCH_INFO}"
    with open(ARCH_INFO, "w", encoding="utf-8") as f:
        f.write(t.replace(old, new))
    print("patched is_fp4_avail")


def create_and_clamp_configs():
    os.makedirs(CFG, exist_ok=True)
    created = adjusted = 0
    for src in glob.glob(f"{CFG}/gfx950-*.json"):
        dst = os.path.join(
            CFG, os.path.basename(src).replace("gfx950-", "gfx1151-", 1)
        )
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            created += 1
    for f in glob.glob(f"{CFG}/gfx1151-*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        mod = False
        for val in d.values():
            if not isinstance(val, dict):
                continue
            if val.get("BLOCK_SIZE_N", 128) > 128:
                val["BLOCK_SIZE_N"] = 128
                mod = True
            if val.get("BLOCK_SIZE_K", 128) > 128:
                val["BLOCK_SIZE_K"] = 128
                mod = True
            if val.get("num_stages", 2) > 2:
                val["num_stages"] = 2
                mod = True
        if mod:
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=2)
            adjusted += 1
    print(f"created {created} gfx1151 configs, adjusted {adjusted}")


if __name__ == "__main__":
    patch_arch_info()
    create_and_clamp_configs()
