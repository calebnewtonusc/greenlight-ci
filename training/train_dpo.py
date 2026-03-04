"""
Stage 3: DPO (Direct Preference Optimization)
Trains GreenLight CI to prefer minimal, root-cause-addressing fixes over
over-engineered, symptom-only, or test-disabling fixes.

Run:
  deepspeed --num_gpus=18 training/train_dpo.py --deepspeed training/configs/deepspeed_zero3.json
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from synthesis.prompts import GREENLIGHT_SYSTEM_PROMPT  # noqa: E402


@dataclass
class DPOTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    rl_adapter: str = "./checkpoints/rl"
    output_dir: str = "./checkpoints/greenlight-final"

    # DPO
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    beta: float = 0.1  # DPO regularization coefficient
    max_length: int = 8192
    max_prompt_length: int = 4096

    # Data
    dpo_pairs_path: str = "./data/training/dpo_pairs.jsonl"

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    wandb_project: str = "greenlight-ci-dpo"


def load_dpo_dataset(path: str) -> Dataset:
    """Load DPO preference pairs in TRL-compatible format."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            # TRL DPOTrainer expects: prompt, chosen, rejected
            prompt = ex.get("prompt", "")
            chosen = ex.get("chosen", "")
            rejected = ex.get("rejected", "")

            if not (prompt and chosen and rejected):
                continue
            if chosen == rejected:
                continue

            examples.append(
                {
                    "prompt": f"<|im_start|>system\n{GREENLIGHT_SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                    "chosen": chosen + "<|im_end|>",
                    "rejected": rejected + "<|im_end|>",
                }
            )

    logger.info(f"Loaded {len(examples)} DPO preference pairs")
    return Dataset.from_list(examples)


def train(config: DPOTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)  # nosec B615
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        config.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    logger.info(f"Loading RL LoRA adapter from: {config.rl_adapter}")
    model = PeftModel.from_pretrained(base_model, config.rl_adapter, is_trainable=True)  # nosec B615
    model.enable_input_require_grads()

    dataset = load_dpo_dataset(config.dpo_pairs_path)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed="training/configs/deepspeed_zero3.json",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="greenlight-ci-dpo",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference with PEFT (requires TRL>=0.7)
        processing_class=tokenizer,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    logger.info("Starting DPO training on fix quality preferences...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"DPO training complete. Final model saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        rl_adapter: str = "./checkpoints/rl",
        output_dir: str = "./checkpoints/greenlight-final",
        dpo_pairs_path: str = "./data/training/dpo_pairs.jsonl",
    ):
        config = DPOTrainingConfig(
            base_model=base_model,
            rl_adapter=rl_adapter,
            output_dir=output_dir,
            dpo_pairs_path=dpo_pairs_path,
        )
        train(config)

    typer.run(main)
