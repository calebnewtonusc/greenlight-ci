"""
GreenLight CI — Patch Generator
Generates fix patches for classified CI failures and builds DPO preference pairs.

Usage:
  python synthesis/patch_generator.py --validate-sandbox
  python synthesis/patch_generator.py --dpo-mode
"""

import json
import os
import random
from pathlib import Path

import httpx
import typer
from loguru import logger

from synthesis.prompts import (
    GREENLIGHT_SYSTEM_PROMPT,
    DPO_RANKING_SYSTEM_PROMPT,
    PATCH_QUALITY_SYSTEM_PROMPT,
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URLS = os.environ.get("VLLM_URLS", "http://localhost:8001").split(",")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")


def generate_patch(record: dict, backend: str = "claude") -> str | None:
    """Generate a fix patch for a classified CI failure."""
    ci_log = record.get("ci_log", "")
    code_context = record.get("code_context", "")
    failure_class = record.get("failure_class", "UNKNOWN")
    failure_subclass = record.get("failure_subclass", "")
    repo = record.get("repo", "unknown")
    language = record.get("language", "unknown")

    user_msg = (
        f"Repository: {repo} ({language})\n"
        f"Failure class: {failure_class} — {failure_subclass}\n\n"
        f"CI Log:\n{ci_log[:6000]}\n\n"
        f"Code context:\n{code_context[:3000] if code_context else '(not available)'}\n\n"
        f"Generate the minimal fix."
    )

    messages = [
        {"role": "system", "content": GREENLIGHT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        if backend == "vllm":
            url = random.choice(VLLM_URLS)
            resp = httpx.post(
                f"{url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                json={"model": "Qwen/Qwen2.5-72B-Instruct", "messages": messages, "max_tokens": 1024, "temperature": 0.2},
                timeout=90.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=GREENLIGHT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return msg.content[0].text.strip()
    except Exception as e:
        logger.debug(f"Patch generation error: {e}")
        return None


def build_dpo_pairs(
    classified_dir: Path,
    output_file: Path,
    backend: str,
    n_pairs: int,
):
    """
    Build DPO preference pairs:
    - chosen: minimal, root-cause-addressing fix
    - rejected: over-engineered OR wrong-class fix OR symptom-only fix
    """
    import anthropic

    all_records = []
    for f in classified_dir.rglob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass

    random.seed(42)
    random.shuffle(all_records)
    records_to_process = all_records[:n_pairs]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pairs_written = 0

    with open(output_file, "w") as out:
        for record in records_to_process:
            if not record.get("has_fix") or not record.get("fix_diff"):
                continue

            # The "chosen" fix: the actual fix from the repo (ground truth)
            chosen_fix = record["fix_diff"]

            # Generate a "rejected" fix: a plausible but worse alternative
            # Strategy: ask the model to generate a different (suboptimal) fix
            ci_log = record.get("ci_log", "")
            language = record.get("language", "unknown")

            rejected_prompt = (
                f"Generate a DIFFERENT but WORSE fix for this CI failure. "
                f"The fix should be plausible but suffer from one of these flaws: "
                f"(1) too verbose — makes unnecessary changes, "
                f"(2) fixes the symptom not the root cause, "
                f"(3) disables/skips the failing test, "
                f"(4) adds a band-aid retry without fixing the underlying issue.\n\n"
                f"CI Log:\n{ci_log[:3000]}\n\n"
                f"Repository language: {language}\n"
                f"Output ONLY the diff, no explanation."
            )

            try:
                if backend == "claude":
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    msg = client.messages.create(
                        model="claude-haiku-4-5",
                        max_tokens=512,
                        system=(
                            "You are generating deliberately suboptimal CI repair patches for "
                            "machine learning training purposes. Produce plausible but flawed fixes."
                        ),
                        messages=[{"role": "user", "content": rejected_prompt}],
                    )
                    rejected_fix = msg.content[0].text.strip()
                else:
                    url = random.choice(VLLM_URLS)
                    resp = httpx.post(
                        f"{url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                        json={"model": "Qwen/Qwen2.5-72B-Instruct",
                              "messages": [{"role": "user", "content": rejected_prompt}],
                              "max_tokens": 512, "temperature": 0.9},
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    rejected_fix = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                continue

            if not rejected_fix or len(rejected_fix) < 20:
                continue

            # Format as DPO pair (ShareGPT DPO format compatible with TRL)
            repair_prompt = (
                f"Repository: {record.get('repo', 'unknown')} ({language})\n"
                f"CI Log:\n{ci_log[:4000]}\n\nGenerate the minimal fix."
            )

            dpo_pair = {
                "id": record.get("id", "") + "_dpo",
                "prompt": repair_prompt,
                "chosen": chosen_fix,
                "rejected": rejected_fix,
                "failure_class": record.get("failure_class"),
                "source": "dpo_synthetic",
            }
            out.write(json.dumps(dpo_pair) + "\n")
            pairs_written += 1

    logger.info(f"DPO pairs built: {pairs_written} written to {output_file}")


app = typer.Typer()


@app.command()
def main(
    classified_dir: Path = typer.Option(Path("data/classified"), help="Classified pairs directory"),
    output: Path = typer.Option(Path("data/training/dpo_pairs.jsonl"), help="DPO output file"),
    backend: str = typer.Option("claude", help="Backend: claude | vllm"),
    n_pairs: int = typer.Option(50000, help="Number of DPO pairs to build"),
    dpo_mode: bool = typer.Option(False, "--dpo-mode", help="Build DPO pairs"),
    validate_sandbox: bool = typer.Option(False, "--validate-sandbox", help="Sandbox-validate generated patches"),
):
    """Generate fix patches and build DPO preference pairs."""
    if dpo_mode:
        logger.info(f"Building {n_pairs} DPO preference pairs...")
        build_dpo_pairs(classified_dir, output, backend, n_pairs)
    elif validate_sandbox:
        logger.info("Sandbox validation mode — use agents/patch_validator.py for full validation")
        logger.info("This flag prepares patches for sandbox validation.")
    else:
        logger.error("Specify --dpo-mode or --validate-sandbox")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
