# DGX Spark vLLM Gemma-4-26B Uncensored Guide

> High-performance LLM inference on NVIDIA DGX Spark using vLLM with uncensored Gemma-4 models.

[![Docker](https://img.shields.io/badge/Docker-ghcr.io/aeon--7/vllm--spark--gemma4--nvfp4-blue)](https://github.com/AEON-7/vllm-spark-gemma4-nvfp4)
[![Hardware](https://img.shields.io/badge/Hardware-DGX%20Spark%20(GB10)-green)](https://www.nvidia.com/en-us/data-center/dgx-spark/)
[![Throughput](https://img.shields.io/badge/Throughput-45%2B%20tok/s-orange)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository documents how to run **fast, uncensored large language models** on the **NVIDIA DGX Spark** (GB10 Blackwell GPU) using vLLM. We achieved **45+ tokens/second** with the [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) model — a significant performance win over both Ollama and slower 31B quantized variants. We use `--gpu-memory-utilization 0.60` and `--max-model-len 262000`, which differ from the HuggingFace card defaults (`0.85` and `65536`) — see [Recommended Settings](#recommended-settings) for why.

### What Makes This Setup Special

- **CUDA 13.0 + Blackwell optimization**: Uses the AEON-7 pre-built vLLM image compiled for SM12.1
- **NVFP4 quantization**: Leverages Blackwell's native FP4 tensor cores for 2-5× speedup
- **FP8 KV cache**: Halves memory usage without accuracy loss
- **CUDA graphs + chunked prefill**: Additional 20-40% throughput gains
- **Memory-optimized settings**: Tuned `--gpu-memory-utilization 0.60` with full `--max-model-len 262000` — achieves same 45+ tok/s while using ~47GB GPU memory instead of ~100GB (see [Recommended Settings](#recommended-settings))
- **Auto-start on boot**: Includes systemd user service for persistence after reboot
- **OpenClaw ready**: Pre-configured integration with [OpenClaw](https://openclaw.ai) agents

## Performance

| Setup | Model | Avg Speed | Model Weights | Notes |
|-------|-------|-----------|---------------|-------|
| **This Setup** ✅ | Gemma-4-26B Uncensored NVFP4 | **45.26 tok/s** | ~16.3 GB | Fastest, uncensored |
| vLLM (LilaRest) | Gemma-4-31B NVFP4 | 9.16 tok/s | ~18.5 GB | Too slow on DGX Spark |
| Ollama | gemma4:31b | 8.05 tok/s | ~19 GB | Baseline |

**The 26B uncensored model is 5× faster than the 31B alternatives.**

### Detailed Benchmark Results

Tested on DGX Spark (GB10) with 128GB unified memory, `max_tokens=200`, 5 diverse prompts, warmup excluded:

```
Test 1:  200 tokens in 4.42s → 45.21 tok/s
Test 2:  200 tokens in 4.41s → 45.38 tok/s
Test 3:  200 tokens in 4.43s → 45.17 tok/s
Test 4:  200 tokens in 4.41s → 45.32 tok/s
Test 5:  200 tokens in 4.42s → 45.23 tok/s
─────────────────────────────────────
Average: 45.26 tok/s (σ = 0.07)
```

## Quick Start

### 1. Prerequisites

- NVIDIA DGX Spark or any **Blackwell SM12.1+** GPU (GB10, RTX 5090, etc.)
- Docker with NVIDIA Container Toolkit
- At least 20GB free disk space for the model
- `python3` and `curl` available on the host

### 2. Clone this repo

```bash
git clone https://github.com/ZengboJamesWang/dgx-spark-vllm-gemma4-26b-uncensored.git
cd dgx-spark-vllm-gemma4-26b-uncensored
```

### 3. Download the model (recommended)

```bash
bash scripts/download-model.sh
```

This downloads the ~15GB model to `~/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4`.

### 4. One-Command Start

```bash
bash scripts/start.sh
```

**Note**: First startup takes ~5-10 minutes while the container:
1. Downloads the model if not pre-downloaded (~15GB)
2. Loads weights (~100s)
3. Compiles CUDA graphs (~55s with caching)

To stop:

```bash
bash scripts/stop.sh
```

### 5. Test It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-uncensored-vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

## Connect to OpenClaw (Recommended)

This repo includes ready-to-use configuration for [OpenClaw](https://openclaw.ai) — connect your local vLLM endpoint to OpenClaw agents in minutes.

### One-Command Setup (Easiest)

```bash
bash scripts/configure-openclaw.sh
```

This automatically patches your `~/.openclaw/openclaw.json` with:
- ✅ vLLM provider configuration
- ✅ vLLM plugin enabled
- ✅ Model alias added
- 📦 Automatic backup created

To also set vLLM as your **primary** model:

```bash
bash scripts/configure-openclaw.sh --primary
```

Then restart OpenClaw:

```bash
systemctl --user restart openclaw-gateway
```

### Manual Setup

If you prefer to configure manually, copy the configuration from [`openclaw/openclaw-config-snippet.json`](openclaw/openclaw-config-snippet.json) into your `~/.openclaw/openclaw.json`:

**1. Add the vLLM provider** (inside `models.providers`):

```json
"vllm": {
  "baseUrl": "http://localhost:8000/v1",
  "apiKey": "vllm-local",
  "api": "openai-completions",
  "models": [
    {
      "id": "gemma-4-26b-uncensored-vllm",
      "name": "gemma-4-26b-uncensored-vllm",
      "reasoning": false,
      "input": ["text"],
      "cost": {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0
      },
      "contextWindow": 262000,
      "maxTokens": 16384
    }
  ]
}
```

**2. Enable the vLLM plugin** (inside `plugins.entries`):

```json
"vllm": {
  "enabled": true
}
```

**3. Set as your primary model** (inside `agents.defaults.model`):

```json
"model": {
  "primary": "vllm/gemma-4-26b-uncensored-vllm",
  "fallbacks": [
    "minimax/MiniMax-M2.5",
    "ollama/gemma4:26b"
  ]
}
```

**4. Add model alias** (inside `agents.defaults.models`):

```json
"vllm/gemma-4-26b-uncensored-vllm": {
  "alias": "gemma4-26b-vllm"
}
```

That's it! OpenClaw will now use your local uncensored Gemma-4 model with **262K context**, **tool calling**, and **reasoning support**.

### Features Available

- **262K context window** — handle long documents and conversations
- **Uncensored outputs** — no safety filters, full model capabilities
- **Tool calling** — native function calling via `--enable-auto-tool-choice`
- **Reasoning parser** — handles Gemma 4's thinking tokens via `--reasoning-parser gemma4`

## Manual Docker Run

If you prefer to run Docker manually instead of `scripts/start.sh`:

```bash
mkdir -p ~/.cache/huggingface

docker run -d --name vllm-gemma4-26b \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/patches/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py" \
  ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest \
  vllm serve /root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4 \
    --served-model-name gemma-4-26b-uncensored-vllm \
    --tensor-parallel-size 1 \
    --max-model-len 262000 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.60 \
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
```

The `gemma4_patched.py` mount is **required** for the AEON-7 model to load correctly with `compressed-tensors` NVFP4.

> **Note on settings**: The command above uses `--gpu-memory-utilization 0.60` and `--max-model-len 262000`, which differ from the [HuggingFace model card](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) defaults (`0.85` and `65536`). After extensive testing on DGX Spark, we found `0.60` with full `262000` context provides the optimal balance — using ~47GB GPU memory (vs ~100GB with the card's defaults) while maintaining the full context window and same 45+ tok/s performance.

## Auto-Start on Boot (Systemd)

To make vLLM automatically start after reboots:

```bash
bash scripts/install-service.sh
systemctl --user start vllm-gemma4-26b.service
```

To also start at **boot time** (before anyone logs in), enable lingering:

```bash
sudo loginctl enable-linger $USER
```

To check status:

```bash
systemctl --user status vllm-gemma4-26b.service
```

## Open WebUI Integration

You can connect [Open WebUI](https://github.com/open-webui/open-webui) to this local vLLM endpoint for a chat-GPT-like interface.

### Install Open WebUI

```bash
# Using pip (recommended for local single-user setups)
pip install open-webui
```

Or via pipx:
```bash
pipx install open-webui
```

### Start Open WebUI (Host install)

If you installed Open WebUI with **pip/pipx on the host**, it shares the same network as vLLM:

```bash
# Set the OpenAI-compatible API base to your local vLLM
export OPENAI_API_BASE_URL="http://localhost:8000/v1"

# Start Open WebUI
open-webui serve
```

Then open `http://localhost:8080` in your browser.

In the Open WebUI settings:
1. Go to **Admin Panel → Settings → Connections**
2. Under **OpenAI API**, set:
   - **API URL**: `http://localhost:8000/v1`
   - **API Key**: `sk-1234567890` (any dummy key works; vLLM doesn't validate it)
3. Click **Save**
4. Go to **Admin Panel → Settings → Models**
5. Verify `gemma-4-26b-uncensored-vllm` appears in the model list
6. Select it from the model dropdown in the chat page

### Open WebUI in Docker

If Open WebUI runs inside a Docker container, **`localhost` / `127.0.0.1` will not work** because inside a container those refer to the container itself, not the host machine where vLLM is running.

Use the host-guest DNS name instead:

```
http://host.docker.internal:8000/v1
```

> **Note**: `host.docker.internal` works on Docker Desktop and Docker Engine 20.10+ on Linux. If it doesn't resolve on your system, start the Open WebUI container with `--network=host` (Linux only) or use the host's LAN IP (e.g., `http://192.168.1.42:8000/v1`).

## Why This Works on DGX Spark

The DGX Spark's **GB10 GPU** (Blackwell architecture, SM12.1) has several unique characteristics that make model selection critical:

### What Works

| Technique | Why It Helps | Gain |
|-----------|--------------|------|
| **CUDA 13.0** | Blackwell FP4 requires CUDA 13+ | Baseline |
| **NVFP4 quantization** | Native FP4 tensor core support | ~2-5× |
| **compressed-tensors** | Model's native format, no conversion overhead | ~10-20% |
| **FP8 KV cache** | Halves KV cache memory, more batching | ~15% |
| **CUDA graphs** | Avoids Python overhead per token | ~20-40% |
| **Chunked prefill** | Better interleaving of prefill/decode | ~10% |

### What Doesn't Work Well

| Setup | Problem |
|-------|---------|
| **LilaRest/gemma-4-31B-it-NVFP4-turbo** | Only 9 tok/s — modelopt quantization path underperforms on DGX Spark compared to compressed-tensors |
| **Ollama gemma4:31b** | 8 tok/s — no CUDA graph optimization, no FP4 tensor core path |
| **Standard vLLM cu124** | Missing Blackwell SM12.1 support entirely |

## Systematic Setup Journey

Here is exactly how we arrived at this 45 tok/s configuration:

### Phase 1: Baseline (Ollama)
- Started with `gemma4:31b` on Ollama
- **Result**: 8.05 tok/s
- **Issue**: Ollama's runtime lacks CUDA graph capture and FP4-optimized kernels

### Phase 2: vLLM with LilaRest 31B
- Tried `LilaRest/gemma-4-31B-it-NVFP4-turbo` on generic vLLM images
- Required `--quantization modelopt` and `transformers>=5.5.0` for gemma4 support
- **Result**: 9.16 tok/s
- **Issue**: Surprisingly slow. The `modelopt` quantization backend has higher kernel latency on DGX Spark compared to `compressed-tensors` format models.

### Phase 3: The Winner (AEON-7 26B Uncensored)
- Switched to `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- Uses `--quantization compressed-tensors` — the model's native format
- Added `--enable-chunked-prefill` and `--enable-prefix-caching`
- Set `--gpu-memory-utilization 0.60` (262K context fits comfortably)
- **Result**: **45.26 tok/s** — a **5× improvement** over both alternatives

### Key Breakthrough

The **compressed-tensors NVFP4 format** directly maps to vLLM's `FlashInferCutlassNvFp4LinearKernel`, which is specifically optimized for Blackwell. The `modelopt` backend used by LilaRest triggers a slower fallback path on SM12.1.

> **Lesson**: On DGX Spark, prefer models natively quantized with `compressed-tensors` + NVFP4 over `modelopt` wrapped models.

## Repository Structure

```
dgx-spark-vllm-gemma4-26b-uncensored/
├── README.md                          # This file
├── patches/
│   └── gemma4_patched.py              # Required patch for AEON-7 NVFP4 loading
├── scripts/
│   ├── start.sh                       # One-command container startup (AEON-7)
│   ├── stop.sh                        # One-command container stop
│   ├── startup.sh                     # In-container startup script
│   ├── benchmark.sh                   # Reproduce our 45 tok/s benchmark
│   ├── download-model.sh              # Pre-download the model
│   ├── install-service.sh             # Install systemd auto-start service
│   ├── configure-openclaw.sh          # Configure OpenClaw integration
│   └── configure-openclaw.py          # OpenClaw config helper (Python)
├── systemd/
│   └── vllm-gemma4-26b.service        # Systemd user service file
├── benchmarks/
│   └── results-gemma4-26b.csv         # Raw benchmark data
├── docs/
│   ├── ARCHITECTURE.md                # Deep dive into DGX Spark + vLLM
│   ├── TROUBLESHOOTING.md             # Common issues and fixes
│   └── MODEL_COMPARISON.md            # Full comparison matrix
├── openclaw/
│   └── openclaw-config-snippet.json   # OpenClaw integration config
└── LICENSE
```

## Benchmarking

Run the official benchmark script to reproduce our results:

```bash
bash scripts/benchmark.sh
```

This will:
1. Send a warmup request (excluded from results)
2. Run 5 diverse prompts with `max_tokens=200`
3. Print average throughput and consistency metrics

## Environment Variables

These are baked into the AEON-7 pre-built image and should not need changing:

```bash
TORCH_CUDA_ARCH_LIST="8.7 8.9 9.0 10.0+PTX 12.0 12.1"
CUDA_VERSION=13.0.1
```

## Recommended Settings

After extensive testing, we found the **optimal balance of performance and memory usage**:

| Setting | Value | Why |
|---------|-------|-----|
| `--gpu-memory-utilization` | **0.60** | Sweet spot: ~47GB usage with full 262K context. Higher values (0.65-0.85) use 30-55GB more memory without benefit. |
| `--max-model-len` | **262000** | Full context window. No need to reduce — 0.60 utilization handles it efficiently. |
| `--max-num-batched-tokens` | **131072** | Optimal batch size for throughput. |

**Do NOT set** `VLLM_NVFP4_GEMM_BACKEND=marlin` — let vLLM auto-detect native FP4 kernels for best performance.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for:
- "model type gemma4 not recognized" fixes
- Transformer version conflicts
- Memory errors and how to tune `--gpu-memory-utilization`

## Credits

- **Model**: [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)
- **Base Model**: Google Gemma 4
- **vLLM**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Hardware**: NVIDIA DGX Spark

## License

MIT License — see [LICENSE](LICENSE).
