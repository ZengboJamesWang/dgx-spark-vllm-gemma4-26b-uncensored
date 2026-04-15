#!/bin/bash
set -e

# Container startup script — runs INSIDE the Docker container
# Upgrades transformers to support Gemma-4, then starts vLLM

echo "Updating transformers to support gemma4..."
pip install --upgrade transformers -q

echo "Starting vLLM server with AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4..."
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4 \
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
