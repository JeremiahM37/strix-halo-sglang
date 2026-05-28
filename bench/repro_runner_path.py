"""Replicate the exact production path: MoeRunner + TritonMoeQuantInfo + StandardDispatchOutput.

The repros that call fused_experts_impl directly all pass. So the page fault
must be triggered by something the higher-level runner path does differently.
"""

import sys
import torch


def main():
    if not torch.cuda.is_available():
        print("FAIL: no GPU")
        return 1

    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    NUM_EXPERTS = 256
    TOP_K = 8
    K = 2048
    N = 256
    GS = 128
    M = 8

    # Build weights in kernel layout
    w13 = torch.randint(0, 256, (NUM_EXPERTS, 2*N, K//2), dtype=torch.uint8, device=dev)
    w2  = torch.randint(0, 256, (NUM_EXPERTS, K,   N//2), dtype=torch.uint8, device=dev)
    s13 = torch.randn(NUM_EXPERTS, 2*N, K//GS, device=dev, dtype=dtype) * 0.01
    s2  = torch.randn(NUM_EXPERTS, K,    N//GS, device=dev, dtype=dtype) * 0.01

    a = torch.randn(M, K, device=dev, dtype=dtype)
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=dev)
    topk_weights = torch.full((M, TOP_K), 1.0/TOP_K, dtype=dtype, device=dev)

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path='dummy'))

    # Build the runner + quant info exactly like CompressedTensorsWNA16TritonMoE.apply_weights does
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
        StandardTopKOutput,
    )

    moe_config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=K,
        intermediate_size_per_partition=N,
        params_dtype=dtype,
        top_k=TOP_K,
        activation="silu",
        is_gated=True,
        inplace=True,
    )
    runner = MoeRunner(MoeRunnerBackend.TRITON, moe_config)

    quant_info = TritonMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_int4_w4a16=True,
        w13_scale=s13,
        w2_scale=s2,
        block_shape=[0, GS],
    )

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.zeros(M, NUM_EXPERTS, device=dev, dtype=dtype),
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=a, hidden_states_scale=None, topk_output=topk_output,
    )

    print(f"shapes: M={M}, K={K}, N={N}, E={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"hidden_states: {a.shape} stride={a.stride()} contig={a.is_contiguous()}")
    print(f"w13:  {w13.shape} stride={w13.stride()}")
    print(f"w2:   {w2.shape} stride={w2.stride()}")
    print(f"s13:  {s13.shape} stride={s13.stride()}")

    print("\ninvoking runner.run ...")
    torch.cuda.synchronize()
    out = runner.run(dispatch_output, quant_info)
    torch.cuda.synchronize()
    print(f"OK: {type(out).__name__}, hidden_states sum={out.hidden_states.float().sum().item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
