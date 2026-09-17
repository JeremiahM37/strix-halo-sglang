# Patch 10 — idle scheduler sleeps instead of spinning a CPU core

**File:** `python/sglang/srt/server_args.py`
**Apply with:** [`patch_sleep_on_idle.py`](patch_sleep_on_idle.py), baked into the Dockerfile.

## Symptom

With the server up and **no requests at all**, one CPU core sits at 100% forever and the
GPU is idle. `top -H` shows the thread as `sglang::schedul`. On a Strix Halo the CPU and
GPU share one cooler, so this is not just wasted power: on an otherwise idle Ryzen AI Max+
395 the spinning core alone took **Tctl from 38 °C to 71 °C**. On a mini PC with weak
cooling, or with a second core busy, that is enough headroom gone to reach a thermal
shutdown.

## Cause

Upstream's scheduler loop receives requests with `recv_pyobj(zmq.NOBLOCK)`. When nothing
is waiting it immediately loops again. SGLang already ships the fix — `--sleep-on-idle`
makes an idle scheduler block in `zmq.Poller.poll(1000)`, which returns as soon as a
request arrives — but it is **off by default**, both at the pinned commit and on current
`main`.

## Fix

The patch flips the default in the image:

- `ServerArgs.sleep_on_idle` defaults to `True`.
- The CLI flag becomes `argparse.BooleanOptionalAction`, so `--sleep-on-idle` still parses
  and **`--no-sleep-on-idle` restores upstream busy-polling**.

Both anchors are asserted, and `tests/check_patches.py` checks them against the pinned
SGLang in CI.

## Measured (Qwen3-0.6B, gfx1151, same image, only the flag differs)

| | default (patched) | `--no-sleep-on-idle` (upstream) |
|---|---|---|
| Idle CPU, whole container | **1%** of one core | **102%** of one core |
| Idle Tctl | 39.5 °C | 71 °C |
| 1-token request after 2.5 s idle, median | 19.8–21.5 ms | 19.9–20.6 ms |
| 32 requests × 128 tokens, 16 concurrent | 1291–1324 tok/s | 1289–1324 tok/s |

The poller wakes on the incoming request, so there is no measurable latency cost for the
first request after idle, and the loop never sleeps while a batch is running. (One
`--no-sleep-on-idle` throughput run dropped to 204 tok/s. That happened with the upstream
busy-poll, not with the patch.)

## Not fixed by this patch

If a **second** core stays at 100% after this patch, look at the thread name. A thread
spinning inside `libhsa-runtime64` (`rocr::core::Runtime::AsyncEventsLoop`) is a ROCm
runtime bug, not SGLang — see [KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md#an-idle-server-pins-a-cpu-core).
