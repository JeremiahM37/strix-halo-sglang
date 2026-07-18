#!/bin/bash
# Wrapper around `docker run` for strix-halo-sglang. Sensible defaults for gfx1151.
#
# Usage:
#   ./start-sglang.sh [MODEL] [extra sglang args...]
#
# Examples:
#   ./start-sglang.sh                           # defaults to Qwen3.5-4B
#   ./start-sglang.sh Qwen/Qwen3.5-4B
#   ./start-sglang.sh Qwen/Qwen3.5-4B --context-length 16384

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-4B}"
shift || true

IMAGE="${SGLANG_IMAGE:-strix-halo-sglang:dev}"
PORT="${SGLANG_PORT:-30000}"
NAME="${SGLANG_CONTAINER:-sglang}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TUNABLE_DIR="${TUNABLE_DIR:-$HOME/.cache/strix-halo-sglang-tunableop}"
MEM_FRAC="${SGLANG_MEM_FRAC:-0.5}"
CONTEXT="${SGLANG_CONTEXT:-8192}"
CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}"

mkdir -p "$HF_CACHE" "$TUNABLE_DIR"

# If a container with this name already exists, remove it (with a notice, so
# an unnoticed name collision doesn't silently kill someone's running server).
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Note: removing existing container '$NAME' (set SGLANG_CONTAINER to run alongside it)." >&2
    docker rm -f "$NAME" >/dev/null
fi

exec docker run -d --name "$NAME" \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --network=host \
    --security-opt seccomp=unconfined \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$TUNABLE_DIR:/root/.tunableop" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
    "$IMAGE" \
    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --mem-fraction-static "$MEM_FRAC" \
        --context-length "$CONTEXT" \
        --attention-backend triton \
        --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
        "$@"
