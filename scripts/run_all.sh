#!/usr/bin/env bash
# GreenLight CI — Full Pipeline Script
# Runs all stages end-to-end: discovery → synthesis → train → eval
# Estimated time: ~40 hours on 18× A6000

set -euo pipefail

echo "========================================"
echo " GreenLight CI — Full Training Pipeline"
echo "========================================"
echo ""

# Validate environment
bash scripts/check_env.sh
echo ""

# Stage 1: Discovery
echo "[1/4] Starting data discovery..."
python discovery/github_ci_logs.py --repos 10000 --workers 30 --output data/raw/github_actions
python discovery/fetch_failure_patterns.py --workers 20 --output data/raw/ci_history
python discovery/github_ci_logs.py --dep-drift-mode --workers 20 --output data/raw/dep_drift
echo "[1/4] Discovery complete."
echo ""

# Stage 2: Synthesis
echo "[2/4] Starting synthesis..."
bash scripts/start_vllm.sh
sleep 30  # Wait for vLLM servers to warm up
python synthesis/failure_classifier.py \
    --input data/raw/ \
    --output data/classified/ \
    --backend vllm
python synthesis/synthesize_bulk.py \
    --concurrency 32 \
    --backend vllm \
    --pairs-per-subclass 2500
python synthesis/patch_generator.py \
    --dpo-mode \
    --n-pairs 50000 \
    --backend vllm
echo "[2/4] Synthesis complete."
echo ""

# Stage 3: Training
echo "[3/4] Starting training..."

# SFT
echo "  [3a] Stage 1: SFT (~8h)..."
deepspeed --num_gpus=18 training/train.py \
    --deepspeed training/configs/deepspeed_zero3.json
echo "  [3a] SFT complete."

# GRPO
echo "  [3b] Stage 2: GRPO (~4h)..."
python agents/patch_validator.py --sandbox-pool 8 --port 8080 &
SANDBOX_PID=$!
sleep 10  # Wait for sandbox pool
deepspeed --num_gpus=18 training/train_rl.py \
    --deepspeed training/configs/deepspeed_zero3.json
kill $SANDBOX_PID 2>/dev/null || true
echo "  [3b] GRPO complete."

# DPO
echo "  [3c] Stage 3: DPO (~2h)..."
deepspeed --num_gpus=18 training/train_dpo.py \
    --deepspeed training/configs/deepspeed_zero3.json
echo "  [3c] DPO complete."

echo "[3/4] Training complete."
echo ""

# Stage 4: Evaluation
echo "[4/4] Running CIBench evaluation..."
python evaluation/ci_bench.py \
    --model checkpoints/greenlight-final \
    --output-json results/ci_bench_results.json
echo "[4/4] Evaluation complete."
echo ""

echo "========================================"
echo " Pipeline complete!"
echo " Results: results/ci_bench_results.json"
echo " Model: checkpoints/greenlight-final/"
echo "========================================"
