# GreenLight CI — 18× A6000 GPU Setup

## Hardware Configuration

| Resource | Spec |
|----------|------|
| GPUs | 18× NVIDIA A6000 (48GB VRAM each) |
| Total VRAM | 864GB |
| RAM | 512GB+ (required for ZeRO-3 optimizer offload) |
| Storage | 2TB NVMe SSD (CI logs + model checkpoints) |
| Network | 100Gbps InfiniBand (GPU-to-GPU communication) |
| Strategy | DeepSpeed ZeRO-3 + CPU offload |

---

## Environment Setup

```bash
# 1. NVIDIA drivers (525.x or newer for A6000)
nvidia-smi  # Verify all 18 GPUs visible

# 2. CUDA 12.1+
nvcc --version

# 3. Python 3.11 (required)
python3 --version

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify DeepSpeed installation
ds_report  # Should show ZeRO-3 available

# 6. Configure environment
cp .env.example .env
# Fill in GITHUB_TOKEN, ANTHROPIC_API_KEY, WANDB_API_KEY

# 7. Validate full environment
bash scripts/check_env.sh
```

---

## Multi-GPU Training Commands

### Stage 1: SFT (8 hours)

```bash
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/configs/deepspeed_zero3.json \
  --config training/configs/sft_config.yaml
```

### Stage 2: GRPO (4 hours)

```bash
# Start sandbox executor pool first (separate terminal)
python agents/patch_validator.py --sandbox-pool 8

# Then launch GRPO training
deepspeed --num_gpus=18 training/train_rl.py \
  --deepspeed training/configs/deepspeed_zero3.json \
  --config training/configs/rl_config.yaml
```

### Stage 3: DPO (2 hours)

```bash
deepspeed --num_gpus=18 training/train_dpo.py \
  --deepspeed training/configs/deepspeed_zero3.json \
  --config training/configs/dpo_config.yaml
```

---

## Synthesis Server Setup (Qwen2.5-72B for data augmentation)

Use 4× A6000 cards for each synthesis server. With 18 GPUs, you can run:
- 1 training server (18 GPUs)
- OR 4 synthesis servers (4 GPUs each) + 2 spare GPUs

For synthesis phase, temporarily repurpose all 18 GPUs across 4 synthesis instances:

```bash
# Instance 1 (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/start_vllm.sh --port 8001

# Instance 2 (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/start_vllm.sh --port 8002

# Instance 3 (GPUs 8-11)
CUDA_VISIBLE_DEVICES=8,9,10,11 bash scripts/start_vllm.sh --port 8003

# Instance 4 (GPUs 12-15)
CUDA_VISIBLE_DEVICES=12,13,14,15 bash scripts/start_vllm.sh --port 8004
```

---

## Monitoring

```bash
# GPU utilization
watch -n 2 nvidia-smi

# Training loss (W&B)
# Auto-reported if WANDB_API_KEY is set

# DeepSpeed throughput
# Printed to stdout during training: "throughput: X samples/sec"

# Sandbox executor pool health
curl http://localhost:8080/health  # sandbox pool status
```

---

## Common Issues

**Issue**: `CUDA out of memory` on ZeRO-3
- Increase CPU offload: set `offload_optimizer.device: "cpu"` and `offload_param.device: "cpu"` in `deepspeed_zero3.json`
- Reduce `per_device_train_batch_size` to 1

**Issue**: InfiniBand errors during gradient sync
- Verify all GPUs are on same NVLink/InfiniBand fabric
- Check: `ibstat` and `nvidia-smi topo -m`

**Issue**: Sandbox executor OOM during GRPO
- Reduce `--sandbox-pool` workers (default 8)
- Each sandbox needs ~4GB RAM for CI execution

**Issue**: GRPO slow (reward function bottleneck)
- Sandbox execution is the bottleneck, not GPU compute
- Increase sandbox pool: `--sandbox-pool 16` (needs more RAM)
- Use faster Docker images with pre-installed deps
