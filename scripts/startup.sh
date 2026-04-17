#!/bin/bash
set -e

# Container startup script — runs INSIDE the Docker container
# AEON-7 pre-built image already has transformers 5.5.0, no need to upgrade

echo "Starting vLLM server with AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4..."
exec vllm serve /root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4 \
  --served-model-name gemma-4-26b-uncensored-vllm \
  --tensor-parallel-size 1 \
  --max-model-len 262000 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 131072 \
  --load-format safetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
