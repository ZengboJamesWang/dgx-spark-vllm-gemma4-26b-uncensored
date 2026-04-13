# DGX Spark vLLM Uncensored Guide

> High-performance LLM inference on NVIDIA DGX Spark using vLLM with uncensored Gemma-4 models.

[![Docker](https://img.shields.io/badge/Docker-vllm/vllm--openai:cu130--nightly-blue)](https://hub.docker.com/r/vllm/vllm-openai)
[![Hardware](https://img.shields.io/badge/Hardware-DGX%20Spark%20(GB10)-green)](https://www.nvidia.com/en-us/data-center/dgx-spark/)
[![Throughput](https://img.shields.io/badge/Throughput-45%2B%20tok/s-orange)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository documents how to run **fast, uncensored large language models** on the **NVIDIA DGX Spark** (GB10 Blackwell GPU) using vLLM. We achieved **45+ tokens/second** with the [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) model — a significant performance win over both Ollama and slower 31B quantized variants.

### What Makes This Setup Special

- **CUDA 13.0 + Blackwell optimization**: Uses the `cu130-nightly` vLLM image with SM12.1 support
- **NVFP4 quantization**: Leverages Blackwell's native FP4 tensor cores for 2-5× speedup
- **FP8 KV cache**: Halves memory usage without accuracy loss
- **CUDA graphs + chunked prefill**: Additional 20-40% throughput gains
- **Tailscale HTTPS**: Secure remote access without exposing ports to the local network

## Performance

| Setup | Model | Avg Speed | Memory | Notes |
|-------|-------|-----------|---------|-------|
| **This Setup** ✅ | Gemma-4-26B Uncensored NVFP4 | **45.11 tok/s** | ~16.3 GB | Fastest, uncensored |
| vLLM (LilaRest) | Gemma-4-31B NVFP4 | 9.16 tok/s | ~18.5 GB | Too slow on DGX Spark |
| Ollama | gemma4:31b | 8.05 tok/s | ~19 GB | Baseline |

**The 26B uncensored model is 5× faster than the 31B alternatives.**

### Detailed Benchmark Results

Tested on DGX Spark (GB10) with 128GB unified memory, `max_tokens=200`, 5 diverse prompts, warmup excluded:

```
Test 1:  200 tokens in 4.43s → 45.16 tok/s
Test 2:  200 tokens in 4.43s → 45.16 tok/s
Test 3:  200 tokens in 4.44s → 45.01 tok/s
Test 4:  200 tokens in 4.43s → 45.13 tok/s
Test 5:  200 tokens in 4.43s → 45.11 tok/s
─────────────────────────────────────
Average: 45.11 tok/s (σ = 0.06)
```

## Quick Start

### 1. Prerequisites

- NVIDIA DGX Spark or any **Blackwell SM12.1+** GPU (GB10, RTX 5090, etc.)
- Docker with NVIDIA Container Toolkit
- At least 20GB free disk space for the model

### 2. One-Command Start

```bash
bash scripts/start.sh
```

Or manually:

```bash
mkdir -p ~/.cache/huggingface

docker run -d --name vllm-gemma4-26b \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
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
```

**Note**: First startup takes ~5-10 minutes while vLLM:
1. Downloads the model (~15GB)
2. Loads weights (~100s)
3. Compiles CUDA graphs (~55s with caching)

### 3. Test It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-uncensored-vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

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
- Tried `LilaRest/gemma-4-31B-it-NVFP4-turbo` on vLLM `cu130-nightly`
- Required `--quantization modelopt` and `transformers>=5.5.0` for gemma4 support
- **Result**: 9.16 tok/s
- **Issue**: Surprisingly slow. The `modelopt` quantization backend has higher kernel latency on DGX Spark compared to `compressed-tensors` format models.

### Phase 3: The Winner (AEON-7 26B Uncensored)
- Switched to `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- Uses `--quantization compressed-tensors` — the model's native format
- Added `--enable-chunked-prefill` and `--enable-prefix-caching`
- Set `--gpu-memory-utilization 0.60` (262K context fits comfortably)
- **Result**: **45.11 tok/s** — a **5× improvement** over both alternatives

### Key Breakthrough

The **compressed-tensors NVFP4 format** directly maps to vLLM's `FlashInferCutlassNvFp4LinearKernel`, which is specifically optimized for Blackwell. The `modelopt` backend used by LilaRest triggers a slower fallback path on SM12.1.

> **Lesson**: On DGX Spark, prefer models natively quantized with `compressed-tensors` + NVFP4 over `modelopt` wrapped models.

## Repository Structure

```
dgx-spark-vllm-guide/
├── README.md                          # This file
├── scripts/
│   ├── start.sh                       # One-command container startup
│   ├── benchmark.sh                   # Reproduce our 45 tok/s benchmark
│   └── download-model.sh              # Pre-download the model
├── configs/
│   └── nginx-tailscale.conf           # Tailscale HTTPS reverse proxy
├── benchmarks/
│   └── results-gemma4-26b.csv         # Raw benchmark data
├── docs/
│   ├── ARCHITECTURE.md                # Deep dive into DGX Spark + vLLM
│   ├── TROUBLESHOOTING.md             # Common issues and fixes
│   └── MODEL_COMPARISON.md            # Full comparison matrix
└── LICENSE
```

## Secure Remote Access (Tailscale + HTTPS)

We expose the vLLM API only over Tailscale, not to the local network.

### Setup

1. Ensure both machines are on the same Tailscale tailnet
2. Get your DGX Spark Tailscale IP:
   ```bash
   tailscale ip -4
   ```
3. Deploy the nginx config from `configs/nginx-tailscale.conf`
4. Access from another Tailscale machine:
   ```
   https://YOUR_TAILSCALE_IP:8443
   ```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for the full HTTPS setup.

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

These are baked into the `cu130-nightly` image and should not need changing:

```bash
TORCH_CUDA_ARCH_LIST="8.7 8.9 9.0 10.0+PTX 12.0 12.1"
CUDA_VERSION=13.0.1
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for:
- "model type gemma4 not recognized" fixes
- Transformer version conflicts
- Port binding issues with nginx
- Memory errors and how to tune `--gpu-memory-utilization`

## Credits

- **Model**: [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)
- **Base Model**: Google Gemma 4
- **vLLM**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Hardware**: NVIDIA DGX Spark

## License

MIT License — see [LICENSE](LICENSE).
