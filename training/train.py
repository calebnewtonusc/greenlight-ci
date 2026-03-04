"""
Stage 1: Supervised Fine-Tuning (SFT)
Fine-tunes Qwen2.5-7B-Coder-Instruct on ~500k CI failure→fix triplets.
Uses LoRA rank 64 with DeepSpeed ZeRO-3 across 18× A6000.

Run:
  deepspeed --num_gpus=18 training/train.py --deepspeed training/configs/deepspeed_zero3.json
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from synthesis.prompts import GREENLIGHT_SYSTEM_PROMPT


@dataclass
class SFTTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    output_dir: str = "./checkpoints/sft"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 16384
    weight_decay: float = 0.01

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # Data
    training_data: str = "./data/training/ci_repair_pairs.jsonl"

    # Logging
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    wandb_project: str = "greenlight-ci-sft"


def format_training_example(example: dict) -> str:
    """
    Format a classified CI failure→fix pair as a training example.
    Uses Qwen2.5 chat format with GreenLight system prompt.
    """
    repo = example.get("repo", "unknown")
    language = example.get("language", "unknown")
    ci_log = example.get("ci_log", "")
    code_context = example.get("code_context", "")
    failure_class = example.get("failure_class", "UNKNOWN")
    failure_subclass = example.get("failure_subclass", "")
    fix_diff = example.get("fix_diff", "")
    root_cause = example.get("root_cause", "")
    fix_strategy = example.get("fix_strategy", "")
    key_evidence = example.get("key_evidence", [])

    user_msg = (
        f"Repository: {repo} ({language})\n"
        f"CI Log:\n{ci_log[:6000]}\n\n"
        f"Code context:\n{code_context[:2000] if code_context else '(not provided)'}\n\n"
        f"Analyze this CI failure and generate the minimal fix."
    )

    # Build assistant response in GreenLight format
    evidence_str = "\n".join(f"  - {e}" for e in key_evidence[:3]) if key_evidence else "  - (see log above)"
    assistant_msg = (
        f"<classify>{failure_class} — {failure_subclass}</classify>\n"
        f"<reason>\n"
        f"{root_cause if root_cause else fix_strategy}\n"
        f"\nKey evidence from log:\n{evidence_str}\n"
        f"</reason>\n"
        f"<fix>\n"
        f"{fix_diff}\n"
        f"</fix>\n"
        f"<validate>\n"
        f"Run the failing test suite after applying this patch. "
        f"Confirm CI is green on first and second run to verify stability.\n"
        f"</validate>"
    )

    return (
        f"<|im_start|>system\n{GREENLIGHT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


def load_training_data(config: SFTTrainingConfig) -> Dataset:
    """Load and format all CI repair training pairs."""
    data_path = Path(config.training_data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Run the discovery + synthesis pipeline first:\n"
            f"  python pipeline.py --stage discovery\n"
            f"  python pipeline.py --stage synthesis"
        )

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                # Only include pairs with actual fixes
                if ex.get("has_fix") and ex.get("fix_diff") and ex.get("ci_log"):
                    examples.append(ex)
            except json.JSONDecodeError:
                pass

    logger.info(f"Loaded {len(examples)} training pairs from {data_path}")

    formatted = [{"text": format_training_example(ex)} for ex in examples]
    logger.info(f"Formatted {len(formatted)} training examples")

    # Log class distribution
    from collections import Counter
    class_dist = Counter(ex.get("failure_class", "UNKNOWN") for ex in examples)
    logger.info(f"Class distribution: {dict(class_dist)}")

    return Dataset.from_list(formatted)


def train(config: SFTTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_training_data(config)
    logger.info(f"Training on {len(dataset)} examples")

    # 10% for validation
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_seq_length=config.max_seq_length,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed="training/configs/deepspeed_zero3.json",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="greenlight-ci-sft",
        dataset_text_field="text",
        packing=False,  # Packing corrupts chat format
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"SFT training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        output_dir: str = "./checkpoints/sft",
        epochs: int = 3,
        training_data: str = "./data/training/ci_repair_pairs.jsonl",
    ):
        config = SFTTrainingConfig(
            base_model=base_model,
            output_dir=output_dir,
            num_train_epochs=epochs,
            training_data=training_data,
        )
        train(config)

    typer.run(main)
