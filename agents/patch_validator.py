"""
GreenLight CI — Patch Validator
Runs generated patches in isolated Docker sandboxes and validates CI outcomes.
Also serves as the sandbox executor pool for GRPO training.

Usage:
  # Start sandbox pool for GRPO training
  python agents/patch_validator.py --sandbox-pool 8

  # Build RL task dataset from classified pairs
  python agents/patch_validator.py --build-rl-tasks

  # Validate a specific patch
  python agents/patch_validator.py --validate --patch fix.diff --repo owner/repo --sha abc123
"""

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import docker
import typer
import uvicorn
from fastapi import FastAPI
from loguru import logger

SANDBOX_IMAGE_MAP = {
    "python": "greenlight-sandbox-python:latest",
    "javascript": "greenlight-sandbox-node:latest",
    "go": "greenlight-sandbox-go:latest",
    "java": "greenlight-sandbox-java:latest",
    "ruby": "greenlight-sandbox-ruby:latest",
    "rust": "greenlight-sandbox-rust:latest",
}

DEFAULT_TEST_COMMANDS = {
    "python": "pytest --tb=short -q",
    "javascript": "npm test",
    "go": "go test ./...",
    "java": "mvn test -q",
    "ruby": "bundle exec rspec",
    "rust": "cargo test",
}


@dataclass
class SandboxResult:
    """Result of a sandbox CI execution."""

    patch_applied: bool
    first_run_passed: bool
    second_run_passed: bool
    first_run_output: str
    second_run_output: str
    execution_time_seconds: float
    reward: float  # Computed reward for GRPO


def apply_patch_in_temp(
    diff_text: str,
    repo_dir: str,
    temp_dir: str,
) -> tuple[bool, str]:
    """
    Apply a unified diff to a repo directory (copied to temp_dir).
    Returns (success, error_message).
    """
    import shutil

    # Copy repo to temp dir
    shutil.copytree(repo_dir, temp_dir, dirs_exist_ok=True)

    # Write diff to file
    diff_file = Path(temp_dir) / "greenlight.patch"
    diff_file.write_text(diff_text)

    # Apply with patch command
    result = subprocess.run(
        ["patch", "-p1", "--input", str(diff_file), "--directory", temp_dir],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return False, result.stderr
    return True, ""


def run_ci_in_docker(
    repo_path: str,
    language: str,
    test_command: str,
    timeout: int = 180,
) -> tuple[bool, str]:
    """
    Run CI tests in a Docker container with the patched codebase.
    Returns (passed, output).
    """
    client = docker.from_env()
    image = SANDBOX_IMAGE_MAP.get(language, "greenlight-sandbox-python:latest")
    cmd = test_command or DEFAULT_TEST_COMMANDS.get(language, "pytest")

    try:
        result = client.containers.run(
            image=image,
            command=f"/bin/bash -c '{cmd}'",
            volumes={repo_path: {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            remove=True,
            mem_limit="4g",
            cpu_period=100000,
            cpu_quota=200000,  # 2 CPUs
            network_disabled=True,  # Security: no network in sandbox
            timeout=timeout,
        )
        # Container exited 0 = tests passed
        return True, result.decode("utf-8", errors="replace")[:4096]
    except docker.errors.ContainerError as e:
        return False, e.stderr.decode("utf-8", errors="replace")[
            :4096
        ] if e.stderr else str(e)
    except Exception as e:
        return False, str(e)


def execute_patch(
    diff_text: str,
    repo: str,
    failing_sha: str,
    language: str,
    test_command: str,
    stability_reruns: int = 2,
) -> SandboxResult:
    """
    Full sandbox execution pipeline:
    1. Clone/checkout repo at failing SHA
    2. Apply the patch
    3. Run CI tests
    4. Run again (stability check)
    """
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="greenlight_") as tmpdir:
        sandbox_dir = Path(tmpdir) / "sandbox"

        # Clone at failing SHA
        clone_result = subprocess.run(
            [
                "git",
                "clone",
                "--depth=50",
                f"https://github.com/{repo}.git",
                str(sandbox_dir),
            ],
            capture_output=True,
            timeout=120,
        )
        if clone_result.returncode != 0:
            logger.debug(f"Clone failed for {repo}")
            return SandboxResult(
                patch_applied=False,
                first_run_passed=False,
                second_run_passed=False,
                first_run_output="Clone failed",
                second_run_output="",
                execution_time_seconds=time.time() - t0,
                reward=0.0,
            )

        # Checkout at failing SHA
        subprocess.run(
            ["git", "checkout", failing_sha],
            cwd=str(sandbox_dir),
            capture_output=True,
            timeout=30,
        )

        # Apply patch
        patch_file = Path(tmpdir) / "greenlight.patch"
        patch_file.write_text(diff_text)
        patch_result = subprocess.run(
            ["git", "apply", "--whitespace=fix", str(patch_file)],
            cwd=str(sandbox_dir),
            capture_output=True,
            timeout=30,
        )
        if patch_result.returncode != 0:
            logger.debug(
                f"Patch application failed: {patch_result.stderr.decode()[:200]}"
            )
            return SandboxResult(
                patch_applied=False,
                first_run_passed=False,
                second_run_passed=False,
                first_run_output=patch_result.stderr.decode()[:1000],
                second_run_output="",
                execution_time_seconds=time.time() - t0,
                reward=0.0,
            )

        # Run CI tests
        first_passed, first_output = run_ci_in_docker(
            str(sandbox_dir), language, test_command
        )

        second_passed = False
        second_output = ""
        if first_passed and stability_reruns >= 1:
            second_passed, second_output = run_ci_in_docker(
                str(sandbox_dir), language, test_command
            )

        # Compute reward
        if not first_passed:
            reward = 0.0
        elif first_passed and not second_passed:
            reward = 0.5
        else:
            reward = 1.0

        # Minimality adjustment
        diff_lines = len(
            [
                line
                for line in diff_text.split("\n")
                if line.startswith(("+", "-")) and not line.startswith(("---", "+++"))
            ]
        )
        if reward > 0:
            if diff_lines < 10:
                reward = min(reward + 0.1, 1.1)
            elif diff_lines > 50:
                reward = max(reward - 0.1, 0.0)

        return SandboxResult(
            patch_applied=True,
            first_run_passed=first_passed,
            second_run_passed=second_passed,
            first_run_output=first_output,
            second_run_output=second_output,
            execution_time_seconds=time.time() - t0,
            reward=reward,
        )


def build_rl_tasks(
    classified_dir: Path,
    output_file: Path,
    max_tasks: int = 10000,
    languages: list[str] | None = None,
):
    """
    Build sandbox-executable RL tasks from classified CI failure pairs.
    Only includes pairs with cloneable repos and verifiable tests.
    """
    if languages is None:
        languages = ["python", "javascript", "go"]  # Most common, best sandbox support

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_written = 0

    with open(output_file, "w") as out:
        for jsonl_file in classified_dir.rglob("*.jsonl"):
            if tasks_written >= max_tasks:
                break
            with open(jsonl_file) as f:
                for line in f:
                    if tasks_written >= max_tasks:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    lang = rec.get("language", "").lower()
                    if lang not in languages:
                        continue
                    if not rec.get("has_fix") or not rec.get("ci_log"):
                        continue
                    if not rec.get("repo") or "/" not in rec.get("repo", ""):
                        continue

                    # Build RL task
                    task = {
                        "id": rec.get("id"),
                        "repo": rec.get("repo"),
                        "failing_sha": rec.get("failing_sha"),
                        "language": lang,
                        "ci_log": rec.get("ci_log"),
                        "failure_class": rec.get("failure_class"),
                        "failure_subclass": rec.get("failure_subclass"),
                        "test_command": DEFAULT_TEST_COMMANDS.get(lang, "pytest"),
                        "ground_truth_diff": rec.get("fix_diff", ""),
                    }
                    out.write(json.dumps(task) + "\n")
                    tasks_written += 1

    logger.info(f"Built {tasks_written} RL sandbox tasks → {output_file}")


# ── REST API for GRPO training reward calls ────────────────────────────────────

rest_app = FastAPI(title="GreenLight CI Sandbox Pool", version="1.0.0")


@rest_app.get("/health")
def health():
    return {"status": "ok", "sandbox_pool": "active"}


@rest_app.post("/execute")
def execute_endpoint(body: dict):
    """Execute a patch in sandbox and return CI outcome."""
    diff = body.get("diff", "")
    repo = body.get("repo", "")
    failing_sha = body.get("failing_sha", "")
    language = body.get("language", "python")
    test_command = body.get("test_command", "")
    stability_reruns = body.get("stability_reruns", 2)

    if not diff or not repo:
        return {"error": "diff and repo required"}

    result = execute_patch(
        diff, repo, failing_sha, language, test_command, stability_reruns
    )
    return {
        "patch_applied": result.patch_applied,
        "first_run_passed": result.first_run_passed,
        "second_run_passed": result.second_run_passed,
        "execution_time": result.execution_time_seconds,
        "reward": result.reward,
    }


app = typer.Typer()


@app.command()
def main(
    sandbox_pool: int = typer.Option(
        None, help="Start sandbox pool REST API with N workers"
    ),
    port: int = typer.Option(8080, help="Sandbox pool port"),
    build_rl_tasks_flag: bool = typer.Option(
        False, "--build-rl-tasks", help="Build RL task dataset"
    ),
    classified_dir: Path = typer.Option(
        Path("data/classified"), help="Input for --build-rl-tasks"
    ),
    output: Path = typer.Option(
        Path("data/rl/ci_sandbox_tasks.jsonl"), help="Output for --build-rl-tasks"
    ),
    max_tasks: int = typer.Option(10000, help="Max RL tasks to build"),
):
    """GreenLight CI patch validator and sandbox executor."""
    if sandbox_pool:
        logger.info(f"Starting sandbox pool on port {port} with {sandbox_pool} workers")
        uvicorn.run(rest_app, host="0.0.0.0", port=port, workers=sandbox_pool)  # nosec B104
    elif build_rl_tasks_flag:
        build_rl_tasks(classified_dir, output, max_tasks)
    else:
        logger.error("Specify --sandbox-pool N or --build-rl-tasks")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
