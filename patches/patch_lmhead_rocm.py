"""Patch 8 -- let compressed-tensors quantize lm_head.

`CompressedTensorsConfig.get_quant_method` handles only LinearBase and FusedMoE and returns
None for everything else. ParallelLMHead therefore falls back to UnquantizedEmbeddingMethod,
which creates a plain `weight` parameter that a quantized checkpoint never fills -- the model
loads, then emits uninitialized logits (repeated token 0, i.e. "!!!!").

ParallelLMHead calls quant_method.create_weights with exactly the LinearMethodBase signature
(input_size_per_partition=embedding_dim, output_partition_sizes=[num_embeddings_per_partition]),
so the linear method works unmodified. With it in place the layer has `weight_packed` rather
than `weight`, so LogitsProcessor._compute_lm_head takes its `quant_method.apply(...)` branch.

Worth ~1.02 GB of streamed weights per decode token on Qwen3.5-35B-A3B.
"""

import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/compressed_tensors.py"
t = open(path).read()

anchor = """            return CompressedTensorsFusedMoEMethod(self)
        return None"""

new = """            return CompressedTensorsFusedMoEMethod(self)

        # gfx1151 patch 8: quantized lm_head. ParallelLMHead is not a LinearBase, so
        # upstream returns None here and the head silently falls back to an unquantized
        # parameter the checkpoint never fills. Its create_weights call already matches
        # the LinearMethodBase signature, so the linear method works as-is.
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, ParallelLMHead):
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return None  # ignored in the checkpoint -> keep the unquantized path
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        return None"""

assert t.count(anchor) == 1, "anchor not found or not unique"
open(path, "w").write(t.replace(anchor, new))
print("patched", path)
