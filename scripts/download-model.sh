#!/bin/bash
set -e

# Pre-download the model using huggingface-cli
# This is optional but recommended for faster first startup

MODEL_ID="AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"
LOCAL_DIR="${HOME}/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4"

echo "=================================="
echo "Model Pre-downloader"
echo "Model: $MODEL_ID"
echo "=================================="

# Check disk space (need at least 20GB free)
AVAILABLE_GB=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 20 ]; then
    echo "⚠️  Warning: Only ${AVAILABLE_GB}GB disk space available."
    echo "   Model needs ~15GB. Free up space first."
    exit 1
fi

# Skip if already downloaded
if [ -f "$LOCAL_DIR/model.safetensors" ]; then
    echo ""
    echo "✅ Model already exists at: $LOCAL_DIR"
    echo "   Skipping download. Delete the folder to re-download."
    exit 0
fi

if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install -U huggingface-hub
fi

# Check if HF_TOKEN is set (optional but recommended for private/gated models)
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "ℹ️  Tip: Set HF_TOKEN environment variable if you encounter rate limits"
    echo "   or need to download gated models:"
    echo "   export HF_TOKEN='your_token_here'"
fi

echo ""
echo "Downloading model (~15GB)..."
huggingface-cli download "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --local-dir-use-symlinks False

echo ""
echo "✅ Model downloaded to: $LOCAL_DIR"
