"""Concurrent throughput benchmark — Ollama vs SGLang on the same model.

Sends N identical requests in parallel and measures aggregate tokens/sec.
This is where SGLang's continuous batching beats Ollama's serialization.

Usage:
    python3 concurrent_throughput.py \
        --ollama http://127.0.0.1:11434/v1/chat/completions \
        --ollama-model qwen3.5:4b \
        --sglang http://127.0.0.1:30000/v1/chat/completions \
        --sglang-model Qwen/Qwen3.5-4B
"""

import argparse
import asyncio
import json
import time
import urllib.request


async def one_call(url, model, prompt, max_tokens):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    def do_req():
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        # Raises urllib.error.HTTPError on non-200; the caller counts failures.
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        d = json.loads(body)
        try:
            return d["usage"]["completion_tokens"]
        except (KeyError, TypeError):
            # 200 with an error body (e.g. {"error": ...}) or missing usage.
            raise RuntimeError(
                f"no usage.completion_tokens in response: {body[:200]!r}"
            ) from None

    t0 = time.time()
    tokens = await asyncio.get_running_loop().run_in_executor(None, do_req)
    return time.time() - t0, tokens


async def bench(url, model, label, concurrency_levels):
    print(f"\n[{label}]")
    print(f"  {'concurrency':>12}  {'wall (s)':>10}  {'total tok':>10}  "
          f"{'agg tps':>10}  {'per-stream tps':>14}")
    for c in concurrency_levels:
        t0 = time.time()
        results = await asyncio.gather(
            *[one_call(url, model, "Write a Python function that counts vowels.", 80)
              for _ in range(c)],
            return_exceptions=True,
        )
        elapsed = time.time() - t0
        ok = [r for r in results if not isinstance(r, BaseException)]
        failed = len(results) - len(ok)
        if failed:
            first_err = next(r for r in results if isinstance(r, BaseException))
            print(f"  {c:>12}  {failed}/{c} requests failed "
                  f"(first error: {first_err})")
            if not ok:
                continue
        total_tokens = sum(r[1] for r in ok)
        agg = total_tokens / elapsed
        per = (total_tokens / len(ok)) / elapsed
        print(f"  {c:>12}  {elapsed:>10.2f}  {total_tokens:>10}  "
              f"{agg:>10.1f}  {per:>14.1f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--ollama-model", default="qwen3.5:4b")
    ap.add_argument("--sglang", default="http://127.0.0.1:30000/v1/chat/completions")
    ap.add_argument("--sglang-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--concurrency", default="1,2,4,8",
                    help="comma-separated concurrency levels")
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--skip-sglang", action="store_true")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]

    for label, url, model, skip in [
        ("Ollama", args.ollama, args.ollama_model, args.skip_ollama),
        ("SGLang", args.sglang, args.sglang_model, args.skip_sglang),
    ]:
        if skip:
            continue
        try:
            t, _ = await one_call(url, model, "hi", 5)
            print(f"[{label}] warmup ok ({t*1000:.0f} ms)")
        except Exception as e:
            print(f"[{label}] warmup FAILED: {e}")
            continue
        await bench(url, model, label, levels)


if __name__ == "__main__":
    asyncio.run(main())
