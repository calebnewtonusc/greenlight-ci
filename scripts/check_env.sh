#!/usr/bin/env bash
# GreenLight CI — Environment Validation
# Checks all prerequisites before running the pipeline.

set -euo pipefail

PASS=0
FAIL=0

check() {
	local name="$1"
	local result="$2"
	if [ "$result" = "ok" ]; then
		echo "  [PASS] $name"
		((PASS++)) || true
	else
		echo "  [FAIL] $name — $result"
		((FAIL++)) || true
	fi
}

echo ""
echo "GreenLight CI — Environment Check"
echo "=================================="
echo ""

# Python version
PY_VER=$(python3 --version 2>&1 | grep -oP '3\.\d+' || echo "not found")
if [[ "$PY_VER" == "3.11" || "$PY_VER" == "3.12" ]]; then
	check "Python version ($PY_VER)" "ok"
else
	check "Python version" "need 3.11 or 3.12, got $PY_VER"
fi

# Required env vars
for var in GITHUB_TOKEN ANTHROPIC_API_KEY VLLM_API_KEY; do
	if [ -n "${!var:-}" ]; then
		check "Env: $var" "ok"
	else
		check "Env: $var" "not set (required)"
	fi
done

# Optional env vars
for var in WANDB_API_KEY HF_TOKEN; do
	if [ -n "${!var:-}" ]; then
		check "Env: $var (optional)" "ok"
	else
		echo "  [WARN] Env: $var — not set (optional, skipping W&B/HF)"
	fi
done

# GPU check
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -ge 1 ]; then
	check "GPUs ($GPU_COUNT detected)" "ok"
	if [ "$GPU_COUNT" -ge 18 ]; then
		check "18× A6000 configuration" "ok"
	else
		echo "  [WARN] Only $GPU_COUNT GPUs (18 recommended for full training)"
	fi
else
	check "GPUs" "no GPUs detected — synthesis only (no GPU training)"
fi

# Python packages
for pkg in torch transformers peft trl deepspeed datasets loguru; do
	if python3 -c "import $pkg" 2>/dev/null; then
		check "Package: $pkg" "ok"
	else
		check "Package: $pkg" "not installed — run pip install -r requirements.txt"
	fi
done

# Docker (for sandbox execution)
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
	check "Docker daemon" "ok"
else
	echo "  [WARN] Docker not available — GRPO sandbox execution won't work"
fi

# Disk space (need ~300GB for data + checkpoints)
DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "${DISK_GB:-0}" -ge 300 ]; then
	check "Disk space (${DISK_GB}GB available)" "ok"
else
	check "Disk space" "need ~300GB, have ${DISK_GB:-unknown}GB"
fi

# RAM (need ~128GB for ZeRO-3)
RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
if [ "${RAM_GB:-0}" -ge 128 ]; then
	check "RAM (${RAM_GB}GB)" "ok"
elif [ "${RAM_GB:-0}" -ge 32 ]; then
	echo "  [WARN] RAM ${RAM_GB}GB — 128GB+ recommended for ZeRO-3 optimizer offload"
else
	check "RAM" "need 128GB+, have ${RAM_GB:-unknown}GB"
fi

echo ""
echo "=================================="
echo "  Passed: $PASS | Failed: $FAIL"
echo "=================================="
echo ""

if [ "$FAIL" -gt 0 ]; then
	echo "Fix the failed checks before running the pipeline."
	exit 1
else
	echo "Environment OK. Run: bash scripts/run_all.sh"
fi
