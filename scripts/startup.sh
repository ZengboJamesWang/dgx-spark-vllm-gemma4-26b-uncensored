#!/bin/bash
set -e

# Container startup script — runs INSIDE the Docker container
# AEON-7 pre-built image already has transformers 5.5.0, no need to upgrade

echo "Starting vLLM server with AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4..."
exec vllm serve /root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4 \
  --served-model-name gemma4-26b-uncensored \
  --tensor-parallel-size 1 \
  --quantization compressed-tensors \
  --load-format safetensors \
  --max-model-len 262000 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.45 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 131072 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8001 \
  --dtype auto
