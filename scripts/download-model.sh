#!/bin/bash
set -e

# Pre-download the model using huggingface-cli
# This is optional but recommended for faster first startup

MODEL_ID="AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"

echo "=================================="
echo "Model Pre-downloader"
echo "Model: $MODEL_ID"
echo "=================================="

if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install -U huggingface-hub
fi

echo ""
echo "Downloading model (~15GB)..."
huggingface-cli download "$MODEL_ID" \
  --local-dir ~/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4 \
  --local-dir-use-symlinks False

echo ""
echo "✅ Model downloaded to: ~/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4"
