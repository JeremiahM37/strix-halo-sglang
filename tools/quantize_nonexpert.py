"""Quantize the bf16 non-expert weights of a Qwen3.5-MoE AWQ checkpoint to int4.

Public AWQ/GPTQ releases of Qwen3.5-35B-A3B quantize only `mlp.experts.*`; lm_head,
self_attn, linear_attn and the shared experts stay bf16. On a bandwidth-bound APU that is
~3.2 GB of the ~3.7 GB streamed per decode token, which is why SGLang trails llama.cpp
single-stream (llama.cpp's GGUF quantizes everything). See bench/results.md.

This rewrites those tensors in the same compressed-tensors `pack-quantized` layout the
checkpoint already uses, so no loader changes are needed:

    weight_packed : [N, K // 8]        int32   (4-bit, packed along input dim)
    weight_scale  : [N, K // group]    bf16
    weight_shape  : [2]                int64   ([N, K])

Stage 1 (default) skips `linear_attn.in_proj_{qkv,z,a,b}` because SGLang fuses those into
in_proj_qkvz / in_proj_ba at load time and its packed-weight loader only binds
("weight", "weight_scale_inv", "weight_scale", "input_scale") -- not "weight_packed".
Quantizing them needs that loader patched first; pass --include-in-proj once it is.

Usage:
    python3 quantize_nonexpert.py --src <snapshot dir> --dst <out dir> [--include-in-proj]
"""

import argparse, glob, json, os, re, shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

NUM_BITS = 4
Q_MAX = 2 ** (NUM_BITS - 1) - 1          # 7, symmetric int4
Q_MIN = -(2 ** (NUM_BITS - 1))           # -8

# Modules whose 2-D .weight we quantize. Deliberately excluded:
#   mlp.gate / shared_expert_gate  -- routers; tiny and accuracy-critical
#   *_layernorm, norm, conv1d, A_log, dt_bias, biases -- not 2-D Linear weights
#   embed_tokens -- a lookup, not streamed per token
#   mtp.*        -- speculative-decoding module, unused here
BASE_TARGETS = (
    r"\.self_attn\.(q|k|v|o)_proj\.weight$",
    r"\.mlp\.shared_expert\.(gate|up|down)_proj\.weight$",
    r"\.linear_attn\.out_proj\.weight$",
    r"^lm_head\.weight$",
)
IN_PROJ_TARGETS = (r"\.linear_attn\.in_proj_(qkv|z|a|b)\.weight$",)


def is_target(name, patterns):
    if ".mtp." in name or name.startswith("mtp."):
        return False
    return any(re.search(p, name) for p in patterns)


def quantize_tensor(w, group_size):
    """bf16 [N, K] -> (packed int32 [N, K//8], scale bf16 [N, K//group], shape int64 [2])."""
    n, k = w.shape
    assert k % group_size == 0, f"K={k} not divisible by group_size={group_size}"
    wf = w.to(torch.float32)
    grouped = wf.reshape(n, k // group_size, group_size)

    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / Q_MAX).clamp(min=1e-8)

    q = torch.round(grouped / scale).clamp(Q_MIN, Q_MAX).to(torch.int8)
    q = q.reshape(n, k)

    packed = pack_to_int32(q, NUM_BITS, packed_dim=1)
    scale_out = scale.squeeze(-1).to(torch.bfloat16)

    # dequant error, for reporting
    deq = (q.reshape(n, k // group_size, group_size).to(torch.float32) * scale).reshape(n, k)
    denom = wf.norm().item() or 1.0
    rel = (deq - wf).norm().item() / denom
    return packed, scale_out, torch.tensor([n, k], dtype=torch.int64), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--include-in-proj", action="store_true")
    ap.add_argument("--no-lm-head", action="store_true",
                    help="leave lm_head in bf16 (isolates logit-path issues)")
    ap.add_argument("--only", default=None,
                    help="regex; quantize only tensors matching it (for bisecting)")
    ap.add_argument("--shard-bytes", type=int, default=4_000_000_000)
    args = ap.parse_args()

    patterns = BASE_TARGETS + (IN_PROJ_TARGETS if args.include_in_proj else ())
    if args.no_lm_head:
        patterns = tuple(p for p in patterns if "lm_head" not in p)
    if args.only:
        patterns = (args.only,)
    os.makedirs(args.dst, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    print(f"source shards: {len(shards)}")

    out_tensors, out_idx, shard_no, cur_bytes = {}, {}, 1, 0
    quantized, saved_bytes, worst = [], 0, (0.0, "")

    def flush():
        nonlocal out_tensors, shard_no, cur_bytes
        if not out_tensors:
            return
        fn = f"model-{shard_no:05d}.safetensors"
        save_file(out_tensors, os.path.join(args.dst, fn), metadata={"format": "pt"})
        for k in out_tensors:
            out_idx[k] = fn
        print(f"  wrote {fn} ({cur_bytes/1e9:.2f} GB, {len(out_tensors)} tensors)")
        out_tensors, cur_bytes = {}, 0
        shard_no += 1

    for s in shards:
        with safe_open(s, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if t.ndim == 2 and is_target(name, patterns):
                    before = t.numel() * 2
                    packed, scale, shape, rel = quantize_tensor(t, args.group_size)
                    pre = name[: -len("weight")]
                    out_tensors[pre + "weight_packed"] = packed
                    out_tensors[pre + "weight_scale"] = scale
                    out_tensors[pre + "weight_shape"] = shape
                    after = packed.numel() * 4 + scale.numel() * 2 + 16
                    cur_bytes += after
                    saved_bytes += before - after
                    quantized.append(name)
                    if rel > worst[0]:
                        worst = (rel, name)
                else:
                    out_tensors[name] = t
                    cur_bytes += t.numel() * t.element_size()
                if cur_bytes >= args.shard_bytes:
                    flush()
    flush()

    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": out_idx}, f, indent=2)

    # config: drop the newly quantized modules from the ignore list
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    qc = cfg["quantization_config"]
    qmods = {n[: -len(".weight")] for n in quantized}

    def still_ignored(entry):
        e = entry[3:] if entry.startswith("re:") else entry
        return not any(m.endswith(e) or e in m for m in qmods)

    before_n = len(qc.get("ignore", []))
    qc["ignore"] = [e for e in qc.get("ignore", []) if still_ignored(e)]
    json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)

    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt",
                  "preprocessor_config.json", "chat_template.jinja"):
        p = os.path.join(args.src, extra)
        if os.path.exists(p):
            shutil.copy2(p, args.dst)

    print(f"\nquantized {len(quantized)} tensors")
    print(f"ignore list: {before_n} -> {len(qc['ignore'])} entries")
    print(f"weight bytes saved: {saved_bytes/1e9:.2f} GB")
    print(f"worst relative dequant error: {worst[0]:.4f} ({worst[1]})")


if __name__ == "__main__":
    main()
