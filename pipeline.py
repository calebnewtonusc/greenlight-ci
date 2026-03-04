"""
GreenLight CI Master Pipeline
Orchestrates the full data → training → evaluation pipeline.
~40 hours total on 18× A6000 + synthesis servers.

Usage:
  python pipeline.py                          # Full pipeline
  python pipeline.py --stage discovery        # Step 1: collect CI failure/fix pairs
  python pipeline.py --stage synthesis        # Step 2: classify + augment pairs
  python pipeline.py --stage train            # Step 3: 3-stage training (SFT + GRPO + DPO)
  python pipeline.py --stage eval             # Step 4: CIBench evaluation
  python pipeline.py --list                   # List all stages with time estimates
"""

import subprocess

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()


STAGES = [
    # ── Environment Check ─────────────────────────────────────────────────
    {
        "name": "check_env",
        "description": "Verify environment, GPU setup, and API keys",
        "cmd": "bash scripts/check_env.sh",
        "phase": "discovery",
        "estimated_hours": 0.1,
    },
    # ── Discovery ─────────────────────────────────────────────────────────
    {
        "name": "fetch_github_ci_logs",
        "description": "Fetch failed GitHub Actions runs + fix commits (Stream 1)",
        "cmd": "python discovery/github_ci_logs.py --repos 10000 --workers 30",
        "phase": "discovery",
        "estimated_hours": 4.0,
    },
    {
        "name": "fetch_failure_patterns",
        "description": "Collect CI history from curated open-source repos (Stream 2)",
        "cmd": "python discovery/fetch_failure_patterns.py --workers 20",
        "phase": "discovery",
        "estimated_hours": 3.0,
    },
    {
        "name": "fetch_dep_drift_corpus",
        "description": "Collect Dependabot/Renovate PRs with CI outcomes (Stream 3)",
        "cmd": "python discovery/github_ci_logs.py --dep-drift-mode --workers 20",
        "phase": "discovery",
        "estimated_hours": 2.0,
    },
    # ── Synthesis ─────────────────────────────────────────────────────────
    {
        "name": "start_vllm",
        "description": "Launch Qwen2.5-72B synthesis servers",
        "cmd": "bash scripts/start_vllm.sh",
        "phase": "synthesis",
        "estimated_hours": 0.5,
    },
    {
        "name": "classify_failures",
        "description": "Auto-classify all collected failure logs into 4-class taxonomy",
        "cmd": "python synthesis/failure_classifier.py --input data/raw/ --output data/classified/",
        "phase": "synthesis",
        "estimated_hours": 3.0,
    },
    {
        "name": "synthesize_pairs",
        "description": "Augment + synthesize additional failure→fix pairs (Stream 5)",
        "cmd": "python synthesis/synthesize_bulk.py --concurrency 32",
        "phase": "synthesis",
        "estimated_hours": 10.0,
    },
    {
        "name": "generate_patches",
        "description": "Generate and validate synthetic fix patches via sandbox",
        "cmd": "python synthesis/patch_generator.py --validate-sandbox",
        "phase": "synthesis",
        "estimated_hours": 4.0,
    },
    # ── Validation ────────────────────────────────────────────────────────
    {
        "name": "validate_pairs",
        "description": "Quality filter, MinHash dedup, confidence scoring",
        "cmd": "python synthesis/synthesize_bulk.py --validate-only",
        "phase": "validation",
        "estimated_hours": 1.5,
    },
    {
        "name": "build_dpo_pairs",
        "description": "Build preference pairs (minimal fix vs. over-engineered vs. wrong-class)",
        "cmd": "python synthesis/patch_generator.py --dpo-mode",
        "phase": "validation",
        "estimated_hours": 2.0,
    },
    {
        "name": "build_rl_tasks",
        "description": "Build sandbox-executable tasks for GRPO training",
        "cmd": "python agents/patch_validator.py --build-rl-tasks",
        "phase": "validation",
        "estimated_hours": 1.0,
    },
    # ── Training ──────────────────────────────────────────────────────────
    {
        "name": "train_sft",
        "description": "Stage 1: Supervised Fine-Tuning (8h on 18× A6000)",
        "cmd": "deepspeed --num_gpus=18 training/train.py --deepspeed training/configs/deepspeed_zero3.json",
        "phase": "train",
        "estimated_hours": 8.0,
    },
    {
        "name": "train_rl",
        "description": "Stage 2: CI-Verified GRPO (4h on 18× A6000 + sandbox pool)",
        "cmd": "deepspeed --num_gpus=18 training/train_rl.py --deepspeed training/configs/deepspeed_zero3.json",
        "phase": "train",
        "estimated_hours": 4.0,
    },
    {
        "name": "train_dpo",
        "description": "Stage 3: DPO on fix quality preferences (2h)",
        "cmd": "deepspeed --num_gpus=18 training/train_dpo.py --deepspeed training/configs/deepspeed_zero3.json",
        "phase": "train",
        "estimated_hours": 2.0,
    },
    # ── Evaluation ────────────────────────────────────────────────────────
    {
        "name": "ci_bench",
        "description": "CIBench evaluation on 200 historical failures",
        "cmd": "python evaluation/ci_bench.py --model checkpoints/greenlight-final",
        "phase": "eval",
        "estimated_hours": 4.0,
    },
    # ── Deploy ────────────────────────────────────────────────────────────
    {
        "name": "deploy",
        "description": "Launch GreenLight CI API server (Docker)",
        "cmd": "docker compose -f deploy/docker-compose.yml up -d",
        "phase": "deploy",
        "estimated_hours": 0.2,
    },
]


def run_stage(stage: dict, dry_run: bool = False) -> bool:
    """Execute a pipeline stage. Returns True on success."""
    console.print(f"\n[bold cyan]▶ {stage['name']}[/bold cyan]: {stage['description']}")
    console.print(f"  [dim]{stage['cmd']}[/dim]")

    if dry_run:
        console.print("  [yellow](dry run — skipping)[/yellow]")
        return True

    result = subprocess.run(stage["cmd"], shell=True)
    if result.returncode != 0:
        console.print(f"  [red]✗ Failed (exit {result.returncode})[/red]")
        return False

    console.print("  [green]✓ Complete[/green]")
    return True


@app.command()
def main(
    stage: str = typer.Option(
        None,
        help="Run only this phase: discovery | synthesis | validation | train | eval | deploy",
    ),
    from_stage: str = typer.Option(None, help="Resume pipeline from this stage name"),
    dry_run: bool = typer.Option(False, help="Print commands without executing"),
    list_stages: bool = typer.Option(False, "--list", help="List all stages and exit"),
):
    """GreenLight CI: full training pipeline from raw CI logs to deployed repair model."""

    if list_stages:
        table = Table(title="GreenLight CI Pipeline Stages")
        table.add_column("Stage", style="cyan")
        table.add_column("Phase")
        table.add_column("Description")
        table.add_column("Est. Hours", justify="right")
        for s in STAGES:
            table.add_row(
                str(s["name"]), str(s["phase"]), str(s["description"]), str(s["estimated_hours"])
            )
        console.print(table)
        total = sum(float(s["estimated_hours"]) for s in STAGES)
        console.print(f"\nTotal estimated: {total:.1f} hours")
        return

    stages_to_run = STAGES
    if stage:
        stages_to_run = [s for s in STAGES if s["phase"] == stage]
        if not stages_to_run:
            console.print(f"[red]Unknown phase: {stage}[/red]")
            raise typer.Exit(1)
    elif from_stage:
        names = [s["name"] for s in STAGES]
        if from_stage not in names:
            console.print(f"[red]Unknown stage: {from_stage}[/red]")
            raise typer.Exit(1)
        idx = names.index(from_stage)
        stages_to_run = STAGES[idx:]

    total_hours = sum(float(s["estimated_hours"]) for s in stages_to_run)
    console.print(
        f"\n[bold]GreenLight CI Pipeline[/bold] — {len(stages_to_run)} stages, ~{total_hours:.0f}h estimated"
    )
    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    for s in stages_to_run:
        success = run_stage(s, dry_run=dry_run)
        if not success:
            console.print(
                f"\n[red bold]Pipeline failed at stage: {s['name']}[/red bold]"
            )
            console.print(f"To resume: python pipeline.py --from-stage {s['name']}")
            raise typer.Exit(1)

    console.print("\n[green bold]Pipeline complete.[/green bold]")


if __name__ == "__main__":
    app()
