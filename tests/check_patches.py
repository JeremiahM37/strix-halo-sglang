#!/usr/bin/env python3
"""Guard against the failure modes behind issue #5.

A fresh build broke because the Dockerfile tracked an unpinned upstream `main`,
and because a transitive dependency (`compressed-tensors`) silently downgraded
the base image's gfx1151 ROCm torch. This script makes both failure modes loud
and deterministic in CI — no GPU and no 50 GB image build required.

It checks three things:

  1. The Dockerfile pins SGLang to an immutable ref (not a moving branch).
  2. Every source-patch anchor in the Dockerfile still exists in that pinned
     SGLang revision, so the patches will apply.
  3. The dependency fix is intact: the pinned `pyproject_other.toml` still has
     the `compressed-tensors==0.15.0` pin our build rewrites, and the Dockerfile
     still relaxes it and freezes torch/torchvision via a pip constraint.

Run locally with: python3 tests/check_patches.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def arg_default(name: str) -> str | None:
    m = re.search(rf"^ARG {re.escape(name)}=(.+)$", DOCKERFILE, re.MULTILINE)
    return m.group(1).strip() if m else None


def clone_pinned(repo: str, ref: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="sglang-src-"))
    run = lambda *a: subprocess.run(a, check=True, capture_output=True)
    run("git", "init", "-q", str(d))
    run("git", "-C", str(d), "remote", "add", "origin", repo)
    run("git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", ref)
    run("git", "-C", str(d), "checkout", "-q", "FETCH_HEAD")
    return d


# Anchors each Dockerfile patch depends on. If upstream moves these, the build's
# `sed`/`assert` patch steps fail — catch it here instead.
SOURCE_ANCHORS: dict[str, list[str]] = {
    # Patch 1 — arch guard sed target
    "sgl-kernel/setup_rocm.py": ['["gfx942", "gfx950"]'],
    # Patch 1b — wave32 WARP_SIZE fix (insert anchor + token it rewrites)
    "sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu": ["#include <torch/all.h>", "WARP_SIZE"],
    "sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu": ["#include <torch/all.h>", "WARP_SIZE"],
    # Patch 2 — RMSNorm native fallback
    "python/sglang/srt/layers/layernorm.py": [
        "elif _is_hip:",
        "from vllm._custom_ops import fused_add_rms_norm, rms_norm",
    ],
    # Patch 3 — AWQ MoE Triton dispatcher
    "python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py": [
        "from sglang.srt.layers.moe import (",
        "self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)",
        "return self.kernel.apply(layer, dispatch_output)",
    ],
    # Dependency fix — the pin our build rewrites must still be the one present
    "python/pyproject_other.toml": ["compressed-tensors==0.15.0"],
}


def main() -> int:
    print("Dockerfile invariants:")
    repo = arg_default("SGL_REPO") or ""
    ref = arg_default("SGL_BRANCH") or ""
    # A pinned ref is an immutable tag or 40-char SHA — never a moving branch.
    pinned = bool(re.fullmatch(r"[0-9a-f]{40}", ref)) or ref.startswith("v")
    check("SGLang pinned to an immutable ref (issue #5)", pinned, f"SGL_BRANCH={ref!r}")
    check("build relaxes compressed-tensors to >=0.16.0",
          "compressed-tensors>=0.16.0" in DOCKERFILE)
    check("build freezes torch/torchvision via PIP_CONSTRAINT",
          "PIP_CONSTRAINT" in DOCKERFILE and "rocm-constraints.txt" in DOCKERFILE)

    if not repo or not ref:
        print("\nCould not read SGL_REPO/SGL_BRANCH from Dockerfile — aborting.")
        return 1

    print(f"\nCloning pinned SGLang {ref[:12]} to verify patch anchors…")
    try:
        src = clone_pinned(repo, ref)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ fetch failed: {e.stderr.decode()[:300]}")
        return 1

    print("Upstream patch anchors:")
    for rel, anchors in SOURCE_ANCHORS.items():
        path = src / rel
        text = path.read_text() if path.is_file() else None
        for anchor in anchors:
            check(f"{rel}: {anchor[:48]}", text is not None and anchor in text,
                  "file missing" if text is None else "anchor not found")

    print(f"\n{checks - len(failures)}/{checks} checks passed.")
    if failures:
        print("FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("All patch anchors and fix invariants intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
