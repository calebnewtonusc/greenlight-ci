"""
GreenLight CI — Main Repair Agent Orchestrator
Watches for CI failures, classifies them, generates fixes, and opens PRs.

Usage:
  # Fix a specific failing run
  python agents/ci_repair_agent.py --repo owner/repo --run-id 12345678

  # Watch mode: monitor and auto-fix failures
  python agents/ci_repair_agent.py --watch --repo owner/repo

  # REST API mode (receives webhooks from CI platforms)
  python agents/ci_repair_agent.py --serve --port 8000
"""

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import typer
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from core.failure_taxonomy import heuristic_classify
from synthesis.prompts import GREENLIGHT_SYSTEM_PROMPT

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GREENLIGHT_MODEL_PATH = os.environ.get(
    "GREENLIGHT_MODEL_PATH", "./checkpoints/greenlight-final"
)
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Coder-Instruct")

# Lazy-loaded model (singleton)
_model = None
_tokenizer = None


def get_model():
    """Lazy-load the GreenLight CI model (singleton)."""
    global _model, _tokenizer
    if _model is None:
        logger.info(f"Loading GreenLight CI model from {GREENLIGHT_MODEL_PATH}")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)  # nosec B615
        _tokenizer.pad_token = _tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(  # nosec B615
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model_path = Path(GREENLIGHT_MODEL_PATH)
        if model_path.exists():
            _model = PeftModel.from_pretrained(base, str(model_path))  # nosec B615
        else:
            logger.warning("No trained model found — using base model (lower quality)")
            _model = base
        _model.eval()
        logger.info("Model loaded.")
    return _model, _tokenizer


@dataclass
class RepairRequest:
    """A CI repair request."""

    repo: str
    run_id: int
    log_text: str
    language: str = "python"
    workflow_name: str = ""


@dataclass
class RepairResult:
    """The result of a CI repair attempt."""

    repo: str
    run_id: int
    failure_class: str
    failure_subclass: str
    generated_response: str
    extracted_fix: str
    pr_url: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


def fetch_ci_log(repo: str, run_id: int) -> tuple[str, str, str]:
    """Fetch CI log, language, and workflow name from GitHub API."""
    import requests
    import zipfile
    from io import BytesIO

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    # Get run metadata
    run_resp = requests.get(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
        headers=headers,
        timeout=30,
    )
    if run_resp.status_code != 200:
        return "", "unknown", ""

    run_data = run_resp.json()
    workflow_name = run_data.get("name", "")

    # Get repo language
    repo_resp = requests.get(
        f"https://api.github.com/repos/{repo}",
        headers=headers,
        timeout=30,
    )
    language = (
        repo_resp.json().get("language", "unknown")
        if repo_resp.status_code == 200
        else "unknown"
    )

    # Fetch log zip
    log_resp = requests.get(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs",
        headers=headers,
        timeout=60,
        allow_redirects=True,
    )
    if log_resp.status_code != 200:
        return "", language, workflow_name

    try:
        zf = zipfile.ZipFile(BytesIO(log_resp.content))
        log_parts = []
        for name in zf.namelist():
            if any(kw in name.lower() for kw in ["test", "build", "check"]):
                log_parts.insert(0, zf.read(name).decode("utf-8", errors="replace"))
            else:
                log_parts.append(zf.read(name).decode("utf-8", errors="replace"))
        log_text = "\n".join(log_parts)[:16000]
    except Exception:
        log_text = log_resp.text[:16000]

    return log_text, language, workflow_name


def generate_repair(request: RepairRequest) -> str:
    """Run GreenLight CI inference to generate a repair."""
    model, tokenizer = get_model()

    # Heuristic pre-classification for context
    signals = heuristic_classify(request.log_text)
    heuristic_hint = ""
    if signals:
        top = signals[0]
        subclass_str = top.failure_subclass.value if top.failure_subclass else "unknown"
        heuristic_hint = f"\n[Heuristic hint: {top.failure_class.value} — {subclass_str}, confidence {top.confidence:.2f}]"

    prompt = (
        f"<|im_start|>system\n{GREENLIGHT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Repository: {request.repo} ({request.language})\n"
        f"Workflow: {request.workflow_name}\n"
        f"{heuristic_hint}\n"
        f"CI Log:\n{request.log_text[:7000]}\n\n"
        f"Analyze this CI failure and generate the minimal fix.\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=12000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def extract_fix(response: str) -> tuple[str, str, str]:
    """Extract failure class, fix diff, and validate strategy from response."""
    classify_match = re.search(r"<classify>(.*?)</classify>", response, re.DOTALL)
    fix_match = re.search(r"<fix>(.*?)</fix>", response, re.DOTALL)
    validate_match = re.search(r"<validate>(.*?)</validate>", response, re.DOTALL)

    failure_info = classify_match.group(1).strip() if classify_match else "UNKNOWN"
    fix = fix_match.group(1).strip() if fix_match else ""
    validate = validate_match.group(1).strip() if validate_match else ""

    return failure_info, fix, validate


def open_github_pr(
    repo: str,
    fix_diff: str,
    failure_class: str,
    run_id: int,
    validate_strategy: str,
) -> str | None:
    """Open a GitHub PR with the generated fix. Returns PR URL or None."""
    import requests

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    # Get default branch
    repo_resp = requests.get(
        f"https://api.github.com/repos/{repo}", headers=headers, timeout=15
    )
    if repo_resp.status_code != 200:
        logger.error("Could not fetch repo info for PR creation")
        return None
    default_branch = repo_resp.json().get("default_branch", "main")

    # Create branch
    branch_name = f"greenlight-ci/fix-{run_id}"
    ref_resp = requests.get(
        f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}",
        headers=headers,
        timeout=15,
    )
    if ref_resp.status_code != 200:
        return None

    sha = ref_resp.json()["object"]["sha"]
    create_branch_resp = requests.post(
        f"https://api.github.com/repos/{repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{branch_name}", "sha": sha},
        timeout=15,
    )
    if create_branch_resp.status_code not in (201, 422):
        return None

    # Create PR
    pr_body = (
        f"## GreenLight CI Auto-Repair\n\n"
        f"**Failure Class**: `{failure_class}`\n"
        f"**Run ID**: {run_id}\n\n"
        f"### Fix\n"
        f"```diff\n{fix_diff}\n```\n\n"
        f"### Validation Plan\n"
        f"{validate_strategy}\n\n"
        f"---\n"
        f"*Generated by [GreenLight CI](https://github.com/calebnewtonusc/greenlight-ci)*"
    )

    pr_resp = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        headers=headers,
        json={
            "title": f"fix(ci): {failure_class} — auto-repair run {run_id}",
            "body": pr_body,
            "head": branch_name,
            "base": default_branch,
        },
        timeout=15,
    )

    if pr_resp.status_code == 201:
        return pr_resp.json().get("html_url")
    logger.error(f"PR creation failed: {pr_resp.status_code} {pr_resp.text[:200]}")
    return None


def repair(repo: str, run_id: int, open_pr: bool = True) -> RepairResult:
    """Main repair entrypoint: fetch log, generate fix, open PR."""
    logger.info(f"Fetching CI log for {repo} run {run_id}...")
    log_text, language, workflow_name = fetch_ci_log(repo, run_id)

    if not log_text:
        return RepairResult(
            repo=repo,
            run_id=run_id,
            failure_class="UNKNOWN",
            failure_subclass="",
            generated_response="",
            extracted_fix="",
            success=False,
            error="Could not fetch CI log",
        )

    request = RepairRequest(
        repo=repo,
        run_id=run_id,
        log_text=log_text,
        language=language,
        workflow_name=workflow_name,
    )

    logger.info("Generating repair...")
    t0 = time.time()
    response = generate_repair(request)
    latency = time.time() - t0
    logger.info(f"Generation complete in {latency:.1f}s")

    failure_info, fix_diff, validate = extract_fix(response)
    failure_class = (
        failure_info.split("—")[0].strip().split()[0] if failure_info else "UNKNOWN"
    )
    failure_subclass = failure_info.split("—")[1].strip() if "—" in failure_info else ""

    logger.info(f"Classified as: {failure_class} — {failure_subclass}")
    logger.info(f"Generated fix ({len(fix_diff.splitlines())} lines)")

    pr_url = None
    if open_pr and fix_diff and fix_diff != "":
        logger.info("Opening GitHub PR...")
        pr_url = open_github_pr(repo, fix_diff, failure_class, run_id, validate)
        if pr_url:
            logger.info(f"PR opened: {pr_url}")
        else:
            logger.warning("Could not open PR — check GITHUB_TOKEN permissions")

    return RepairResult(
        repo=repo,
        run_id=run_id,
        failure_class=failure_class,
        failure_subclass=failure_subclass,
        generated_response=response,
        extracted_fix=fix_diff,
        pr_url=pr_url,
        success=bool(fix_diff and len(fix_diff) > 20),
    )


# ── FastAPI server for webhook mode ────────────────────────────────────────────

rest_app = FastAPI(title="GreenLight CI API", version="1.0.0")


@rest_app.get("/health")
def health():
    return {"status": "ok", "model": GREENLIGHT_MODEL_PATH}


@rest_app.post("/repair")
async def repair_endpoint(body: dict, background_tasks: BackgroundTasks):
    """Receive a repair request and process in background."""
    repo = body.get("repo")
    run_id = body.get("run_id")
    if not repo or not run_id:
        return JSONResponse({"error": "repo and run_id required"}, status_code=400)

    def do_repair():
        result = repair(repo, int(run_id))
        logger.info(f"Repair complete: {result.pr_url or 'no PR'}")

    background_tasks.add_task(do_repair)
    return {"status": "queued", "repo": repo, "run_id": run_id}


# ── CLI ────────────────────────────────────────────────────────────────────────

app = typer.Typer()


@app.command()
def main(
    repo: str = typer.Option(None, help="GitHub repo (owner/repo)"),
    run_id: int = typer.Option(None, help="Failing workflow run ID"),
    watch: bool = typer.Option(False, help="Watch mode: poll for failures every 5 min"),
    serve: bool = typer.Option(False, help="Start REST API server"),
    port: int = typer.Option(8000, help="Port for REST API server"),
    no_pr: bool = typer.Option(False, help="Don't open PR (dry run)"),
):
    """GreenLight CI repair agent."""
    if serve:
        logger.info(f"Starting GreenLight CI API server on port {port}")
        uvicorn.run(rest_app, host="0.0.0.0", port=port)  # nosec B104
    elif watch and repo:
        logger.info(f"Watch mode: monitoring {repo} for CI failures")
        import requests

        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        processed = set()
        while True:
            resp = requests.get(
                f"https://api.github.com/repos/{repo}/actions/runs",
                headers=headers,
                params={"conclusion": "failure", "per_page": 10},
                timeout=15,
            )
            if resp.status_code == 200:
                for run in resp.json().get("workflow_runs", []):
                    rid = run["id"]
                    if rid not in processed:
                        processed.add(rid)
                        logger.info(f"New failure detected: run {rid}")
                        repair(repo, rid, open_pr=not no_pr)
            time.sleep(300)
    elif repo and run_id:
        result = repair(repo, run_id, open_pr=not no_pr)
        logger.info(f"Result: {result}")
    else:
        logger.error("Specify --repo + --run-id, --watch + --repo, or --serve")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
