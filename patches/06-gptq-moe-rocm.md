# Patch 6 — let GPTQ MoE models load on gfx1151 (experimental, numerics unresolved)

**Status: NOT enabled in the default image.** These two changes are enough to get a GPTQ
MoE checkpoint to load and serve on RDNA 3.5, which SGLang refuses to do today — but the
model emits garbage tokens, so wiring it in by default would trade a loud failure for a
silent one. Documented here so the two blockers are recorded and the remaining bug is
findable.

## Blocker 1 — `moe_wna16` is denied on ROCm by a blanket list

**File:** `python/sglang/srt/configs/model_config.py`

`moe_wna16` is SGLang's Triton path for int4 MoE without Marlin — exactly what a
consumer RDNA card needs, since Marlin is CUDA-only. It is rejected before it can be tried:

```
ValueError: moe_wna16 quantization is currently not supported in ROCm.
```

That comes from a membership test, not a capability test:

```python
if is_hip() and self.quantization not in rocm_supported_quantization:
```

and `moe_wna16` simply isn't in the list. Notably `MoeWNA16Config.is_moe_wna16_compatible`
gates its **awq** branch on device capability but its **gptq** branch not at all —
`quant_method == "gptq" and not desc_act and num_bits in [4, 8]` — so nothing about the
GPTQ path is CUDA-specific by design.

```diff
             "auto-round",
             "quark_int4fp8_moe",
+            "moe_wna16",
         ]
```

## Blocker 2 — GPTQ ops are imported only under `if _is_cuda`

**File:** `python/sglang/srt/layers/quantization/gptq.py`

Past blocker 1, loading dies with `NameError: name 'gptq_shuffle' is not defined`, because:

```python
if _is_cuda:
    from sgl_kernel import gptq_gemm, gptq_shuffle
```

There is no ROCm branch, even though the ROCm vLLM build in the base image ships both ops
(`vllm._custom_ops.gptq_gemm`, `.gptq_shuffle`) and they run on gfx1151. vLLM's `gptq_gemm`
takes an extra `use_v2_format` argument that SGLang's call site doesn't pass, so it needs a
thin shim; `gptq_shuffle` matches 1:1.

```diff
 if _is_cuda:
     from sgl_kernel import gptq_gemm, gptq_shuffle
+else:
+    from vllm._custom_ops import gptq_gemm as _vllm_gptq_gemm
+    from vllm._custom_ops import gptq_shuffle as _vllm_gptq_shuffle
+
+    def gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
+                  use_exllama, bit):
+        return _vllm_gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
+                               b_g_idx, use_exllama, False, bit)
+
+    def gptq_shuffle(q_weight, q_perm, bit):
+        return _vllm_gptq_shuffle(q_weight, q_perm, bit)
```

Also relevant: `GPTQMarlinConfig.is_gptq_marlin_compatible` begins `if not _is_cuda: return
False`, so on ROCm `MoeWNA16Config.use_marlin` is already False and linear layers route to
plain `GPTQLinearMethod`. Nothing else needs forcing.

## Result

With both applied, `btbtyler09/Qwen3.5-35B-A3B-GPTQ-4bit` loads and serves:

```
Load weight end. elapsed=19.55 s, type=Qwen3_5MoeForConditionalGeneration,
quant=gptq, bits=4, avail mem=88.21 GB, mem usage=22.47 GB.
Capture cuda graph end. Time elapsed: 8.38 s.
```

**But generation is garbage** — every request returns `!!!!!!...` (token 0 repeated),
the signature of broken dequantization. Flipping `use_v2_format` to `True` changes nothing,
so it is not the v1/v2 zero-point convention. The remaining suspects are the `moe_wna16`
MoE weight path on ROCm (untested there) and the packing assumptions in `gptq_gemm`'s
exllama branch. Isolating it means comparing one quantized layer against a manual
dequantization — start there.

## Why this ended up not mattering for performance

The reason for trying a GPTQ checkpoint was bytes-per-token: the AWQ checkpoint we ship with
leaves `linear_attn`, `lm_head` and the shared experts in bf16. `btbtyler09` advertises
`ignore: []`, which reads as "everything quantized" — it isn't. Reading the tensor names
instead of the config:

```
QUANTIZED (int4):  mlp.experts.*.{gate,up,down}_proj   (10240 each)
UNQUANTIZED (bf16): linear_attn.in_proj_qkv (30), mlp.shared_expert.* (40 each),
                    lm_head, self_attn.*, mlp.gate
```

Identical coverage to `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit`. Same bytes per token, so no
speedup was available even with working numerics. **Check the tensor list, not the
`ignore` field** — every public quantization of this model covers only the routed experts.
