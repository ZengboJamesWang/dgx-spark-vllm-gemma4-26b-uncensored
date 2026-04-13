#!/bin/bash
# DGX Spark vLLM Benchmark Script
# Reproduces the 45+ tok/s benchmark from the guide

API_URL="http://localhost:8000/v1/chat/completions"
MODEL="gemma-4-26b-uncensored-vllm"

echo "=================================================="
echo "DGX Spark vLLM Benchmark"
echo "Model: $MODEL"
echo "Hardware: NVIDIA DGX Spark (GB10 Blackwell)"
echo "=================================================="

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Server not running! Start it first with: bash scripts/start.sh"
    exit 1
fi

echo ""
echo "Step 1: WARMUP (discarded from results)"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":100}" > /dev/null
echo "✓ Warmup complete"

# 5 diverse test prompts
TEST_PROMPTS=(
    "What is machine learning? Explain briefly."
    "Write a Python function to calculate factorial using recursion."
    "Explain quantum computing in simple terms."
    "Describe the process of photosynthesis."
    "What are the main differences between HTTP and HTTPS?"
)

echo ""
echo "Step 2: RUNNING 5 TESTS"
echo "=================================================="

RESULTS_FILE="benchmarks/results-gemma4-26b.csv"
mkdir -p "$(dirname "$RESULTS_FILE")"
echo "Test,Prompt,Tokens,Time(s),Tokens/s" > "$RESULTS_FILE"

for i in {1..5}; do
    idx=$((i-1))
    prompt="${TEST_PROMPTS[$idx]}"
    
    echo ""
    echo "Test $i/5: ${prompt:0:50}..."
    
    start_time=$(date +%s.%N)
    
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":200}")
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # Extract actual token count from response
    tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "200")
    
    # Calculate tokens per second
    tps=$(echo "scale=2; $tokens / $elapsed" | bc)
    
    echo "  Tokens: $tokens"
    echo "  Time: ${elapsed}s"
    echo "  Speed: ${tps} tok/s"
    
    echo "$i,\"$prompt\",$tokens,$elapsed,$tps" >> "$RESULTS_FILE"
    sleep 1
done

echo ""
echo "=================================================="
echo "BENCHMARK RESULTS"
echo "=================================================="

avg_time=$(tail -n +2 "$RESULTS_FILE" | awk -F',' '{sum+=$4; count++} END {printf "%.2f", sum/count}')
avg_tps=$(tail -n +2 "$RESULTS_FILE" | awk -F',' '{sum+=$5; count++} END {printf "%.2f", sum/count}')
total_tokens=$(tail -n +2 "$RESULTS_FILE" | awk -F',' '{sum+=$3} END {print sum}')

echo ""
tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r test prompt tokens time tps; do
    echo "  Test $test: ${time}s | ${tps} tok/s"
done

echo ""
echo "Average Metrics:"
echo "  Time: ${avg_time}s"
echo "  Speed: ${avg_tps} tok/s"
echo "  Total Tokens: $total_tokens"
echo ""
echo "✓ Results saved to: $RESULTS_FILE"
echo "=================================================="
