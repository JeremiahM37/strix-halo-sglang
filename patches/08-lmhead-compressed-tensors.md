# Patch 8 — quantized `lm_head` for compressed-tensors

**File:** `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
**Apply with:** [`patch_lmhead_rocm.py`](patch_lmhead_rocm.py)

On Qwen3.5-35B-A3B, `lm_head` is a 248320 × 2048 bf16 matrix — **1.02 GB streamed on every
decode token**, the single largest tensor in the model. compressed-tensors cannot quantize it,
and the failure is silent.

## The bug

`CompressedTensorsConfig.get_quant_method` dispatches on two types and returns `None` for
everything else:

```python
if isinstance(layer, LinearBase):   ...
if isinstance(layer, FusedMoE):     ...
return None
```

`ParallelLMHead` is a `VocabParallelEmbedding`, not a `LinearBase`, so it gets `None` and falls
back to `UnquantizedEmbeddingMethod`. That creates a plain `weight` parameter which a quantized
checkpoint never fills — the weights sit in `weight_packed` / `weight_scale`, which nothing
reads. The server loads cleanly, reports no error, and then emits **uninitialized logits**:
every request returns `!!!!!!...` (argmax pinned to token 0).

Unlike the GPTQ and AWQ configs, which carry an explicit `lm_head_quantized` flag,
compressed-tensors has no path for this at all.

## The fix

`ParallelLMHead` already calls `quant_method.create_weights` with exactly the
`LinearMethodBase` signature — `input_size_per_partition=embedding_dim`,
`output_partition_sizes=[num_embeddings_per_partition]` — so the linear method works unmodified:

```diff
             return CompressedTensorsFusedMoEMethod(self)
+
+        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
+
+        if isinstance(layer, ParallelLMHead):
+            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
+            if scheme is None:
+                return None  # ignored in the checkpoint -> unquantized path
+            layer.scheme = scheme
+            return CompressedTensorsLinearMethod(self)
         return None
```

With the linear method attached the layer has `weight_packed` rather than `weight`, so
`LogitsProcessor._compute_lm_head` skips its `hasattr(lm_head, "weight")` fast path and takes
the `quant_method.apply(...)` branch — the one its comment labels "GGUF models".

## Checkpoint side: `lm_head` must be a target

The code change alone is not enough. Scheme matching is by module class name or layer name, and
a `ParallelLMHead` matches neither `"Linear"` nor anything else in a default config:

```
ValueError: Unable to find matching target for lm_head in the compressed-tensors config.
```

So `config_groups.*.targets` must include `lm_head` (this is what llm-compressor emits when it
quantizes the head), and `lm_head` must not be in `ignore`:

```json
"targets": ["Linear", "lm_head"]
```

[`tools/quantize_nonexpert.py`](../tools/quantize_nonexpert.py) does this automatically unless
`--no-lm-head` is passed.

## Result

Qwen3.5-35B-A3B, on top of [patch 7](07-wna16-rocm-linear.md), warm TunableOp:

| Concurrent streams | experts-only (before) | patch 7 | patch 7 + 8 |
|---:|---:|---:|---:|
| 1 | 23.4 | 29.7 | **33.4** |
| 4 | 72.4 | 97.8 | 94.6 |
| 8 | 127.0 | 137.6 | **180.8** |

Bytes streamed per decode token: **3.70 GB → 0.94 GB**, about half of Ollama's ~1.8 GB.
Single-stream goes from 0.62× to **0.88×** of Ollama, and 8-stream from 3.35× to **4.62×**.
Output verified coherent with correct arithmetic.
