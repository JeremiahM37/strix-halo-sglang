# Contributing

Thanks for your interest. This is a small project but contributions are welcome — bug reports, hardware reports, kernel fixes, build improvements.

## Reporting issues

Useful info to include:

- Hardware (CPU + GPU model)
- `rocminfo | grep -E 'Name:|gfx'` output
- Host kernel (`uname -r`) and ROCm version
- Docker version (`docker version`)
- The exact `docker run` command and full server log

## Patches

Small, focused diffs land fastest. If you're adding a new patch for a new SGLang/ROCm version, please:

1. Add a `patches/0N-short-name.md` describing what it does and why
2. Bake it into the `Dockerfile` so the build stays one-step
3. Update the README "Patches applied" section
4. Add a line to `docs/KNOWN_ISSUES.md` if it papers over a known bug

## Testing changes

After modifying the Dockerfile or a patch:

```bash
docker build -t strix-halo-sglang:dev .
./start-sglang.sh Qwen/Qwen3-0.6B
curl http://localhost:30000/v1/models
```

Qwen3-0.6B is the cheapest smoke test (~1 GB, loads in seconds).

For perf changes, please run `bench/concurrent_throughput.py` before and after and include numbers in the PR.

## Upstream first

If a fix belongs in SGLang, aiter, or transformers itself, please PR it there too — this repo is a staging area, not a permanent home. Link the upstream PR in your PR here.
