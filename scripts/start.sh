#!/bin/bash
set -e

# DGX Spark vLLM Startup Script
# Starts the AEON-7 Gemma-4-26B Uncensored NVFP4 model with optimal settings

CONTAINER_NAME="vllm-gemma4-26b"
IMAGE="vllm/vllm-openai:cu130-nightly"
MODEL_PATH="/root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4"

echo "=================================="
echo "DGX Spark vLLM Starter"
echo "=================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check NVIDIA runtime
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  Warning: NVIDIA Container Runtime may not be configured."
    echo "   Make sure 'docker run --gpus all' works on your system."
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
echo "Starting vLLM container..."
echo "Model: $MODEL_PATH"
echo "This may take 5-10 minutes on first run for model download + CUDA graph compilation."
echo ""

docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "$IMAGE" \
  --model "$MODEL_PATH" \
  --served-model-name gemma-4-26b-uncensored-vllm \
  --quantization compressed-tensors \
  --load-format safetensors \
  --max-model-len 262000 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.60 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000

echo ""
echo "Container started: $CONTAINER_NAME"
echo ""
echo "⏳ Waiting for server to be ready..."
sleep 30

for i in {1..12}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
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
    echo "  ...still waiting ($((i * 10))s elapsed)"
    sleep 10
done

echo ""
echo "⚠️  Server did not become ready within 2 minutes."
echo "Check logs: docker logs $CONTAINER_NAME"
