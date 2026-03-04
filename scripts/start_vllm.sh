#!/usr/bin/env bash
# GreenLight CI — Start vLLM synthesis servers
# Launches Qwen2.5-72B on 4 GPUs per instance for failure scenario synthesis.
# Adjust CUDA_VISIBLE_DEVICES based on your GPU count.

set -euo pipefail

MODEL="Qwen/Qwen2.5-72B-Instruct"
TENSOR_PARALLEL=4

echo "Starting vLLM synthesis servers..."

# Instance 1 (GPUs 0-3, port 8001)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" \
	--tensor-parallel-size $TENSOR_PARALLEL \
	--port 8001 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.90 \
	--trust-remote-code \
	&

# Instance 2 (GPUs 4-7, port 8002)
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" \
	--tensor-parallel-size $TENSOR_PARALLEL \
	--port 8002 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.90 \
	--trust-remote-code \
	&

# Instance 3 (GPUs 8-11, port 8003)
CUDA_VISIBLE_DEVICES=8,9,10,11 python -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" \
	--tensor-parallel-size $TENSOR_PARALLEL \
	--port 8003 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.90 \
	--trust-remote-code \
	&

# Instance 4 (GPUs 12-15, port 8004)
CUDA_VISIBLE_DEVICES=12,13,14,15 python -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" \
	--tensor-parallel-size $TENSOR_PARALLEL \
	--port 8004 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.90 \
	--trust-remote-code \
	&

echo "vLLM servers starting on ports 8001-8004. Waiting 30s for warmup..."
sleep 30

# Health check
for port in 8001 8002 8003 8004; do
	if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
		echo "  [OK] vLLM server on port $port"
	else
		echo "  [WARN] vLLM server on port $port not responding (may still be loading)"
	fi
done

echo ""
echo "Export VLLM_URLS for synthesis scripts:"
echo "  export VLLM_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004"
