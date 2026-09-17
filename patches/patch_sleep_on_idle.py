#!/usr/bin/env python3
"""Make the idle SGLang scheduler sleep instead of busy-polling (patch 10).

Upstream's scheduler polls its ZMQ sockets with NOBLOCK in a tight loop, so an
idle server pins one CPU core at 100% forever. `--sleep-on-idle` replaces that
with a blocking zmq.Poller wait (1 s timeout, returns as soon as a request
arrives), but it is off by default. On a Strix Halo APU the CPU and GPU share
one cooler, and the spinning core alone took Tctl from 38 to 71 C at idle.

This flips the default in the pinned SGLang:
  1. ServerArgs.sleep_on_idle defaults to True (Engine / Python API).
  2. The CLI flag becomes a BooleanOptionalAction defaulting to True, so
     `--sleep-on-idle` still parses and `--no-sleep-on-idle` restores the
     upstream busy-poll.

Every anchor is asserted, so an upstream move fails the build instead of
silently shipping the spin again. See patches/10-sleep-on-idle-default.md.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
text = open(path).read()

REPLACEMENTS = [
    (
        "    sleep_on_idle: bool = False\n",
        "    sleep_on_idle: bool = True  # gfx1151 patch 10: don't spin a core when idle\n",
    ),
    (
        '''            "--sleep-on-idle",
            action="store_true",
            help="Reduce CPU usage when sglang is idle.",
''',
        '''            "--sleep-on-idle",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.sleep_on_idle,
            help="Reduce CPU usage when sglang is idle (default on in strix-halo-sglang; "
            "--no-sleep-on-idle restores upstream busy-polling).",
''',
    ),
]

for old, new in REPLACEMENTS:
    count = text.count(old)
    assert count == 1, f"patch 10 anchor found {count} times, expected 1: {old.splitlines()[0].strip()!r}"
    text = text.replace(old, new)

open(path, "w").write(text)
print("patched sleep_on_idle default -> True")
