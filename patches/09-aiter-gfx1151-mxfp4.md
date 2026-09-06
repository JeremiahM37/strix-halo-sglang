# Patch 9 — enable Quark/MXFP4 (aiter FP4 GEMM) on gfx1151

**Files:**
- Dockerfile: adds a `COPY` + `RUN` step that patches the **base image's** aiter
  installation at build time.
- **Apply with:** [`fix_aiter_gfx1151_mxfp4.py`](fix_aiter_gfx1151_mxfp4.py) — baked into the
  Dockerfile before the `common_ops` file check.

## What it unlocks

Quark/MXFP4 checkpoints (`quant_method=quark`, e.g. `Qwen3.5-27B-Quark-AWQ-MXFP4`) on gfx1151.
Without this patch they fail at the first `aiter` FP4 GEMM with:

```
AssertionError: MXFP4 is not available on your device
```

and, after forcing the check, with:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 100352, Hardware limit: 65536
```

## How it works

Two independent blockers in the **base image's** aiter (not in sglang itself):

1. `aiter/ops/triton/utils/_triton/arch_info.py::is_fp4_avail()` only whitelists `gfx950`, so
   MXFP4 is reported unavailable on every other device:

   ```python
   def is_fp4_avail():
       return get_arch() in ("gfx950")
   ```

   The patch adds `gfx1151`.

2. aiter ships pre-tuned GEMM configs only for the data-center arches it was built for
   (`gfx942` / `gfx950`). On a fresh gfx1151 host the lookup
   (`aiter/ops/triton/utils/gemm_config_utils.py`) demands
   `.../configs/gemm/gfx1151-GEMM-AFP4WFP4.json`, which does not exist. Naively copying the
   `gfx950` configs works, but their block sizes need **100 KB** of shared memory while RDNA 3.5
   (wave32) only exposes **64 KB**, producing `OutOfResources`.

   The script copies the **FP4** `gfx950-*.json` configs to `gfx1151-*.json` and clamps
   `BLOCK_SIZE_M` / `BLOCK_SIZE_N` / `BLOCK_SIZE_K` to ≤128 and `num_stages` to ≤2 so the
   kernel fits in 64 KB. Measured over aiter's own configs, that leaves **0 of 547** FP4 tile
   entries above the 64 KB limit (worst case 49,664 bytes).

   Only the FP4 families are cloned. The other `gfx950` configs (`A16W16`, `A8W8`,
   `A8W8_BLOCKSCALE`, …) are not what this patch is for, and on gfx1151 they currently fail
   the config lookup with a clear assertion. Cloning them would replace that with a clamped
   CDNA config on an untested path, and the clamp does not rescue them anyway: 128×128×128 at
   fp16 needs 128 KB of LDS at two stages and cannot fit in 64 KB at any `M`. Cloning
   everything ships **82** entries that still exceed the limit — as an `OutOfResources` at the
   first request, not a build failure.

## Notes

- This is a runtime/layout heuristic fix: the Triton kernel is still JIT-compiled for the actual
  device, only the tuning knobs come from the config. The resulting kernels are correct but not
  benchmark-tuned for gfx1151.
- The script is idempotent and safe to keep in the image on rebuilds: a `gfx1151-*.json` that
  is already present is left byte-identical, so if a base-image bump ever brings real
  gfx1151 configs (aiter already ships `gfx1250`, a wave32 RDNA arch) this patch defers to
  them rather than clamping tuned values down.
- Every assumption is asserted, because aiter lives in the **base image** and
  `tests/check_patches.py` shallow-clones only the pinned *sglang* repo, so CI cannot cover
  this anchor. A missing config directory, a renamed arch prefix, a moved config tree, a
  schema change and an unparseable config each fail the build with the path in the message.
  The alternative is an image that reports MXFP4 available while shipping no configs, whose
  `/health` is green and whose first inference request 500s.
- Verified with `Qwen3.5-27B-Quark-AWQ-MXFP4` on an AMD Ryzen AI Max+ 395 (gfx1151) — loads and
  serves (`200 OK`) end to end.
