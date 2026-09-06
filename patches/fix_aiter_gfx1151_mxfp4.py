#!/usr/bin/env python3
"""Fix aiter so Quark/MXFP4 checkpoints work on gfx1151 (Radeon 8060S).

Patches the base image's aiter installation in place:
  1. arch_info.is_fp4_avail() only whitelists gfx950 -> allow gfx1151.
  2. No gfx1151 GEMM configs are shipped; the gfx950 configs need 100KB of
     shared memory while RDNA 3.5 (wave32) only exposes 64KB. Copy the FP4
     gfx950-*.json configs to gfx1151-*.json with block sizes clamped so the
     Triton kernel fits in 64KB.

Only the FP4 configs are cloned. The other gfx950 families (A16W16, A8W8,
A8W8_BLOCKSCALE, ...) are not what this patch is about, and on gfx1151 they
currently fail the config lookup with a clear assertion. Cloning them would
replace that with a clamped CDNA config on a path nobody has tested, and the
clamp does not even rescue them: 128x128x128 at fp16 needs 128KB of LDS at
two stages and cannot fit in 64KB at any M. Measured over aiter's own
configs, cloning everything ships 82 entries that still exceed 64KB, while
the FP4 subset ships none.

Idempotent: a gfx1151 config aiter already ships is left exactly as it is,
and every anchor is asserted, so this is safe to run on rebuilds.

See patches/09-aiter-gfx1151-mxfp4.md for details.
"""

import glob
import json
import os
import re

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


# RDNA 3.5 (wave32) exposes 64KB of LDS per workgroup, against the 160KB the
# gfx950 configs were tuned for.
LDS_LIMIT = 65536
TILE_LIMIT = 128
_OPERAND_BYTES = {"FP4": 0.5, "8": 1.0, "16": 2.0}


def operand_bytes(basename):
    """Bytes per element of the A and B operands, read from the config's name."""
    m = re.search(r"-A(FP4|8|16)W(FP4|8|16)", basename)
    assert m, f"cannot read operand dtypes from config name: {basename}"
    return _OPERAND_BYTES[m.group(1)], _OPERAND_BYTES[m.group(2)]


def lds_bytes(cfg, a_bytes, b_bytes):
    """LDS a tile needs: both operand tiles per stage, plus FP4 scale bytes.

    Calibrated against the failure this patch exists for -- 32x32x1024 at three
    stages, both operands FP4, reports Required: 100352, which this reproduces.
    """
    m, n, k = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    total = cfg["num_stages"] * (m * k * a_bytes + k * n * b_bytes)
    # One e8m0 scale byte per 32 elements, for whichever operand is FP4.
    if a_bytes == 0.5:
        total += m * k / 32
    if b_bytes == 0.5:
        total += k * n / 32
    return int(total)


def clamp(cfg, a_bytes, b_bytes, where):
    """Shrink one tile config until it fits in 64KB of LDS."""
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        cfg[key] = min(cfg[key], TILE_LIMIT)
    cfg["num_stages"] = min(cfg["num_stages"], 2)
    # The dimension clamps above cover every config aiter ships today. Halving
    # is the fallback for a config family added later: shipping one that does
    # not fit trades a build failure for an OutOfResources at the first
    # request, which is a far worse place to find out.
    while lds_bytes(cfg, a_bytes, b_bytes) > LDS_LIMIT:
        biggest = max(("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"), key=lambda k: cfg[k])
        assert cfg[biggest] > 16, f"{where}: cannot fit {cfg} in {LDS_LIMIT} bytes of LDS"
        cfg[biggest] //= 2


def create_and_clamp_configs():
    # No makedirs: a missing directory means aiter's layout moved, and a run
    # that creates it, matches nothing and reports success is how a build ships
    # an image whose first FP4 request dies on a missing config file.
    assert os.path.isdir(CFG), f"aiter gemm config dir not found: {CFG}"
    srcs = sorted(glob.glob(f"{CFG}/gfx950-*FP4*.json"))
    assert srcs, f"no gfx950-*FP4*.json configs in {CFG}: aiter's config layout changed"

    created = present = 0
    for src in srcs:
        base = os.path.basename(src)
        dst = os.path.join(CFG, base.replace("gfx950-", "gfx1151-", 1))
        if os.path.exists(dst):
            # aiter ships a real gfx1151 config. It is tuned for this device and
            # the clamp is not; leave it alone. (It already ships gfx1250, so
            # this is a plausible thing for a base image bump to bring.)
            present += 1
            continue
        a_bytes, b_bytes = operand_bytes(base)
        with open(src, encoding="utf-8") as fh:
            d = json.load(fh)
        tiles = [
            v for v in d.values()
            if isinstance(v, dict) and all(
                key in v for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages")
            )
        ]
        assert tiles, f"{base}: no BLOCK_SIZE_* entries, config schema changed"
        for cfg in tiles:
            clamp(cfg, a_bytes, b_bytes, base)
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
        created += 1

    assert created or present, "no gfx1151 FP4 configs written"
    print(f"wrote {created} clamped gfx1151 FP4 configs ({present} already shipped by aiter)")


if __name__ == "__main__":
    patch_arch_info()
    create_and_clamp_configs()
