Title: [Guide] Uncensored Gemma-4-26B at 45 tok/s on DGX Spark — Actually Feels Great to Use!

Hey DGX Spark community! 👋

I've been experimenting with LLM inference on my DGX Spark and found a setup that not only gets **45+ tokens/second** but actually feels **great** to use day-to-day.

**GitHub Repo**: https://github.com/ZengboJamesWang/dgx-spark-vllm-gemma4-26b-uncensored

### 🚀 What Makes This Special: UNCENSORED + FAST

#### **UNCENSORED — No Filtered Responses**

This is the **AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4** model. It's completely uncensored — no alignment filtering, no refusals, no "I cannot help with that" walls. It responds directly and honestly without the typical guardrails. This is genuinely refreshing if you're tired of models that over-refuse or give sanitized answers.

#### **BLAZING FAST with OpenClaw**

When paired with **OpenClaw**, this setup feels incredibly responsive:
- **45.26 tok/s** average speed
- Responses stream in smoothly without lag
- Long outputs finish quickly
- The typing experience is fluid and satisfying

It doesn't feel like you're waiting for a model — it feels like a tool that keeps up with you. **Very good feeling overall!**

### Performance Comparison

Tested on DGX Spark with `max_tokens=200`, warmup excluded:

| Setup | Model | Speed | Memory |
|-------|-------|-------|--------|
| **This Setup** ✅ | Gemma-4-26B Uncensored NVFP4 (MoE) | **45.26 tok/s** | ~16.3 GB |
| vLLM LilaRest 31B | Gemma-4-31B NVFP4 (Dense) | 9.16 tok/s | ~18.5 GB |
| Ollama | gemma4:31b (Dense) | 8.05 tok/s | ~19 GB |

### Perfect for OpenClaw

If you're using **OpenClaw** as your UI, this setup shines:
- Fast streaming responses
- No censorship getting in the way
- Smooth, responsive feel
- 262K context support for long conversations

The combination of **uncensored model + 45 tok/s speed + OpenClaw UI** = genuinely pleasant experience.

### Quick Start

```bash
git clone https://github.com/ZengboJamesWang/dgx-spark-vllm-gemma4-26b-uncensored.git
cd dgx-spark-vllm-gemma4-26b-uncensored
bash scripts/start.sh
bash scripts/benchmark.sh
```

### Try It Out

Looking for feedback from other DGX Spark users, especially on OpenClaw integration and other MoE models!

Happy (uncensored) inferencing! 🚀
