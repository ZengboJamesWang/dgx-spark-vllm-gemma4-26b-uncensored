#!/bin/bash
set -e

# DGX Spark vLLM Startup Script
# Starts the AEON-7 Gemma-4-26B Uncensored NVFP4 model with optimal settings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER_NAME="vllm-gemma4-26b"
# Use AEON-7 pre-built image for DGX Spark (SM 12.1) with transformers 5.5.0
IMAGE="ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest"
MODEL_PATH="/root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4"
STARTUP_SCRIPT="$SCRIPT_DIR/startup.sh"
PATCH_FILE="$REPO_DIR/patches/gemma4_patched.py"

echo "=================================="
echo "DGX Spark vLLM Starter"
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

# Check startup script exists
if [ ! -f "$STARTUP_SCRIPT" ]; then
    echo "❌ Startup script not found: $STARTUP_SCRIPT"
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

# Only pull image if not present locally (avoids DNS issues during boot)
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo ""
    echo "Pulling image: $IMAGE"
    docker pull "$IMAGE"
else
    echo ""
    echo "Using local image: $IMAGE"
fi

echo ""
echo "Starting vLLM container..."
echo "Model: $MODEL_PATH"
echo "This may take 5-10 minutes on first run for model download + CUDA graph compilation."
echo ""

docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$STARTUP_SCRIPT:/startup.sh" \
  -v "$PATCH_FILE:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py" \
  "$IMAGE" \
  bash /startup.sh

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
