# Patch 1 — Allow `gfx1151` in sgl-kernel's arch guard

**File:** `sgl-kernel/setup_rocm.py`

**Why:** Upstream rejects anything other than `gfx942` (MI300) and `gfx950` (MI350). RDNA 3.5 / Strix Halo isn't in their support matrix. But the kernels themselves compile cleanly with `--amdgpu-target=gfx1151` once the guard is relaxed.

## Diff

```diff
-if amdgpu_target not in ["gfx942", "gfx950"]:
+if amdgpu_target not in ["gfx942", "gfx950", "gfx1151"]:
     print(
-        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. Expected 'gfx942' or 'gfx950'."
+        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. Expected 'gfx942', 'gfx950', or 'gfx1151'."
     )
     sys.exit(1)
```

## Verification

After build, 46+ ops register cleanly:

```bash
python3 -c "
import sgl_kernel, torch
ops = [n for n in torch._C._dispatch_get_all_op_names() if n.startswith('sgl_kernel')]
print(len(ops), 'ops registered')
"
```

Numerical correctness check (silu_and_mul vs PyTorch reference) shows max abs error of 0.0.

## Upstream PR

Candidate for `sgl-project/sglang`. Two-line change. Could be paired with patch 2 in a single "gfx1151 support" PR.
