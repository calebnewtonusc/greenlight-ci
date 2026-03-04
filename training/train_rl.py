"""
Stage 2: CI-Verified Reinforcement Learning (GRPO)
The core technical innovation of GreenLight CI.

Uses CI green/red as the reward signal — the same "free verifiable reward"
insight as DeepSeek-R1, applied to CI repair. Each generated patch is
applied to a sandbox environment, CI is rerun, and the reward is based on
whether CI turns green AND stays green on a second rerun.

Run:
  deepspeed --num_gpus=18 training/train_rl.py --deepspeed training/configs/deepspeed_zero3.json
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RLTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    sft_adapter: str = "./checkpoints/sft"
    output_dir: str = "./checkpoints/rl"

    # GRPO
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8  # 8 candidate patches per prompt

    # Reward
    sandbox_pool_size: int = 8  # Parallel sandbox executors
    sandbox_api_url: str = "http://localhost:8080"

    # LoRA (continue from SFT)
    lora_r: int = 64
    lora_alpha: int = 128

    # Data
    rl_tasks_path: str = "./data/rl/ci_sandbox_tasks.jsonl"

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    wandb_project: str = "greenlight-ci-rl"


def compute_patch_reward(
    patch_text: str,
    task_meta: dict,
    sandbox_api_url: str,
) -> float:
    """
    Execute a generated patch in a CI sandbox and compute the reward.

    Reward scheme:
      0.0 — patch fails to apply (invalid diff)
      0.0 — CI still red after patch
      0.5 — CI green on first run
      1.0 — CI green on first AND second run (stable fix)
     +0.1 — bonus if diff < 10 lines (minimality)
     -0.1 — penalty if diff > 50 lines (verbosity)
    """
    import requests
    import re

    # Extract the <fix> block from the model's response
    fix_match = re.search(r"<fix>(.*?)</fix>", patch_text, re.DOTALL)
    if not fix_match:
        logger.debug("No <fix> block in generated response")
        return 0.0

    diff_text = fix_match.group(1).strip()
    if not diff_text or "---" not in diff_text:
        logger.debug("Empty or invalid diff in <fix> block")
        return 0.0

    # Count diff lines (for minimality bonus/penalty)
    diff_lines = len([
        l for l in diff_text.split("\n")
        if l.startswith(("+", "-")) and not l.startswith(("---", "+++"))
    ])

    # Submit to sandbox executor
    try:
        resp = requests.post(
            f"{sandbox_api_url}/execute",
            json={
                "repo": task_meta.get("repo"),
                "failing_sha": task_meta.get("failing_sha"),
                "diff": diff_text,
                "test_command": task_meta.get("test_command", "pytest"),
                "language": task_meta.get("language", "python"),
                "stability_reruns": 2,  # Must pass twice
            },
            timeout=300,  # 5 minutes per sandbox execution
        )
        result = resp.json()
    except Exception as e:
        logger.debug(f"Sandbox API error: {e}")
        return 0.0

    # Base reward
    reward = 0.0
    if result.get("first_run_passed"):
        reward = 0.5
        if result.get("second_run_passed"):
            reward = 1.0

    # Minimality bonus/penalty
    if reward > 0:
        if diff_lines < 10:
            reward += 0.1
        elif diff_lines > 50:
            reward -= 0.1

    return float(min(max(reward, 0.0), 1.1))


def build_reward_function(config: RLTrainingConfig):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    TRL calls: reward_fn(prompts=prompts, completions=completions, **dataset_cols)
    completions is a flat list[str] of length num_prompts * num_generations.
    The function must return one reward per completion (same flat length).
    """
    def reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        metadata_list = kwargs.get("metadata", [])
        num_generations = config.num_generations
        rewards = []

        for i, completion in enumerate(completions):
            # Map flat completion index back to its source prompt
            prompt_idx = i // num_generations
            meta = metadata_list[prompt_idx % len(metadata_list)] if metadata_list else {}
            reward = compute_patch_reward(
                completion,
                meta,
                config.sandbox_api_url,
            )
            rewards.append(reward)
            logger.debug(f"  Completion {i} (prompt {prompt_idx}): reward={reward:.2f}")

        if rewards:
            logger.info(
                f"Batch rewards: mean={sum(rewards)/len(rewards):.3f}, "
                f"max={max(rewards):.3f}, min={min(rewards):.3f}, "
                f"nonzero={sum(1 for r in rewards if r > 0)}/{len(rewards)}"
            )
        return rewards

    return reward_fn


def load_rl_dataset(data_path: str) -> Dataset:
    """
    Load sandbox-executable CI tasks for RL training.
    Each example has: prompt, repo, failing_sha, test_command, language.
    """
    from synthesis.prompts import GREENLIGHT_SYSTEM_PROMPT

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not all(k in ex for k in ("ci_log", "failure_class", "repo")):
                continue

            # Format prompt for GRPO
            ci_log = ex.get("ci_log", "")
            repo = ex.get("repo", "unknown")
            language = ex.get("language", "unknown")
            failure_class = ex.get("failure_class", "")

            prompt = (
                f"<|im_start|>system\n{GREENLIGHT_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Repository: {repo} ({language})\n"
                f"Known failure class: {failure_class}\n"
                f"CI Log:\n{ci_log[:6000]}\n\n"
                f"Generate the minimal fix.\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            ex["prompt"] = prompt
            ex["metadata"] = {
                "repo": ex.get("repo"),
                "failing_sha": ex.get("failing_sha"),
                "test_command": ex.get("test_command", "pytest"),
                "language": language,
            }
            examples.append(ex)

    logger.info(f"Loaded {len(examples)} RL sandbox tasks")
    return Dataset.from_list(examples)


def train(config: RLTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading SFT LoRA adapter from: {config.sft_adapter}")
    model = PeftModel.from_pretrained(base_model, config.sft_adapter, is_trainable=True)

    dataset = load_rl_dataset(config.rl_tasks_path)
    logger.info(f"RL dataset: {len(dataset)} sandbox tasks")

    reward_fn = build_reward_function(config)

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed="training/configs/deepspeed_zero3.json",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="greenlight-ci-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    logger.info("Starting GRPO training with CI sandbox reward...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"GRPO training complete. Checkpoint saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        sft_adapter: str = "./checkpoints/sft",
        output_dir: str = "./checkpoints/rl",
        rl_tasks_path: str = "./data/rl/ci_sandbox_tasks.jsonl",
        num_generations: int = 8,
    ):
        config = RLTrainingConfig(
            base_model=base_model,
            sft_adapter=sft_adapter,
            output_dir=output_dir,
            rl_tasks_path=rl_tasks_path,
            num_generations=num_generations,
        )
        train(config)

    typer.run(main)
