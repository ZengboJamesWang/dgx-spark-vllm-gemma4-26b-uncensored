#!/bin/bash
set -e

# Alternative startup using AEON-7's pre-built DGX Spark image
# This image ships with vLLM 0.19.1rc1 + transformers 5.5.0 + CUDA 13.0
# and is compiled specifically for SM 12.1 (DGX Spark / GB10).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER_NAME="vllm-gemma4-26b"
IMAGE="ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest"
MODEL_PATH="/root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4"
PATCH_FILE="$REPO_DIR/patches/gemma4_patched.py"
GEMMA4_PY="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py"

echo "=================================="
echo "DGX Spark vLLM Starter (AEON-7)"
echo "=================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check NVIDIA runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "⚠️  Warning: NVIDIA Container Runtime may not be configured."
    echo "   Make sure 'docker run --gpus all' works on your system."
fi

# Check patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "❌ Patch file not found: $PATCH_FILE"
    echo "   This patch is required for the AEON-7 model to load correctly."
    exit 1
fi

# Stop existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
fi

# Ensure HuggingFace cache directory exists
mkdir -p ~/.cache/huggingface

echo ""
echo "Pulling image: $IMAGE"
docker pull "$IMAGE"

echo ""
echo "Starting vLLM container with AEON-7 pre-built image..."
echo "Model: $MODEL_PATH"
echo "This may take 5-10 minutes on first run for model download + CUDA graph compilation."
echo ""

docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PATCH_FILE:$GEMMA4_PY" \
  "$IMAGE" \
  vllm serve "$MODEL_PATH" \
    --served-model-name gemma-4-26b-uncensored-vllm \
    --tensor-parallel-size 1 \
    --max-model-len 262000 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 131072 \
    --load-format safetensors \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4

echo ""
echo "Container started: $CONTAINER_NAME"
echo ""
echo "⏳ Waiting for server to be ready (this can take 5-10 min on first run)..."

for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is ready!"
        echo ""
        echo "Test it:"
        echo "  curl http://localhost:8000/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"gemma-4-26b-uncensored-vllm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":100}'"
        echo ""
        echo "View logs: docker logs -f $CONTAINER_NAME"
        echo "Stop: docker stop $CONTAINER_NAME"
        exit 0
    fi
    echo -n "."
    if [ $((i % 10)) -eq 0 ]; then
        echo " ($((i*10))s)"
    fi
    sleep 10
done

echo ""
echo "⚠️  Server did not become ready within 10 minutes."
echo "Check logs: docker logs $CONTAINER_NAME"
exit 1
