"""
circleci_logs.py - CircleCI failed pipeline log discovery and collection.

Uses CircleCI API v2 to pull failed pipeline logs from public repositories.
Pairs failure logs with the fixing commit that made CI green.

Target: 50k CircleCI failure -> fix pairs.

API: https://circleci.com/api/v2/
Docs: https://circleci-public.github.io/circleci-openapi-redoc/

Usage:
    export CIRCLECI_TOKEN=your_token
    python discovery/circleci_logs.py
    python discovery/circleci_logs.py --max-projects 1000
    python discovery/circleci_logs.py --org github/python
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parents[1] / "data"
CIRCLECI_LOGS_FILE = DATA_DIR / "circleci_failure_logs.jsonl"
CIRCLECI_PROGRESS_FILE = DATA_DIR / "circleci_progress.json"

CCI_BASE = "https://circleci.com/api/v2"
GH_BASE = "https://api.github.com"

# ─── Well-known CircleCI-heavy organizations and repos ───────────────────────
KNOWN_CIRCLECI_ORGS = [
    "github/facebook",
    "github/google",
    "github/microsoft",
    "github/mozilla",
    "github/rails",
    "github/django",
    "github/nodejs",
    "github/rust-lang",
    "github/python",
    "github/pallets",  # Flask, Click
    "github/celery",
    "github/redis",
    "github/elastic",
    "github/hashicorp",
    "github/kubernetes",
    "github/prometheus",
    "github/grafana",
    "github/circleci",
    "github/travis-ci",
    "github/netlify",
]

# Well-known repos with CircleCI
KNOWN_CIRCLECI_REPOS = [
    "github/facebook/react",
    "github/facebook/relay",
    "github/rails/rails",
    "github/django/django",
    "github/pallets/flask",
    "github/pallets/click",
    "github/celery/celery",
    "github/redis/redis-py",
    "github/psf/requests",
    "github/python-poetry/poetry",
    "github/docker/compose",
    "github/hashicorp/terraform",
    "github/prometheus/prometheus",
    "github/grafana/grafana",
]

# ─── Log truncation patterns to extract useful sections ──────────────────────
ERROR_SECTION_PATTERNS = [
    re.compile(r"(Error|Exception|FAILED|FAILURE|error:).*", re.I),
    re.compile(r"(npm ERR!|yarn error|pip.*error).*", re.I),
    re.compile(r"exit code \d+", re.I),
]


def cci_get(endpoint: str, params: dict, token: str) -> dict:
    """Make authenticated CircleCI API v2 request."""
    url = f"{CCI_BASE}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Circle-Token": token,
            "Accept": "application/json",
            "User-Agent": "greenlight-ci-harvester/1.0",
        },
    )
    try:
        if not url.startswith("https://"):
            raise ValueError(f"Unsafe URL scheme: {url}")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        if hasattr(e, "code") and e.code == 429:
            time.sleep(10)
            return {}
        return {}


def gh_get(endpoint: str, params: dict, gh_token: str = "") -> dict:
    """Make GitHub API request to get commit diffs for fixes."""
    query = urllib.parse.urlencode(params)
    url = f"{GH_BASE}/{endpoint}" + (f"?{query}" if query else "")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "greenlight-ci-harvester/1.0",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        if not url.startswith("https://"):
            raise ValueError(f"Unsafe URL scheme: {url}")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def get_project_pipelines(
    project_slug: str,
    token: str,
    page_token: str = None,
    branch: str = "main",
) -> dict:
    """Get recent pipelines for a project."""
    params = {"branch": branch}
    if page_token:
        params["page-token"] = page_token
    return cci_get(f"project/{project_slug}/pipeline", params, token)


def get_pipeline_workflows(pipeline_id: str, token: str) -> list[dict]:
    """Get workflows for a pipeline."""
    data = cci_get(f"pipeline/{pipeline_id}/workflow", {}, token)
    return data.get("items", [])


def get_workflow_jobs(workflow_id: str, token: str) -> list[dict]:
    """Get jobs for a workflow."""
    data = cci_get(f"workflow/{workflow_id}/job", {}, token)
    return data.get("items", [])


def get_job_steps(project_slug: str, job_number: int, token: str) -> list[dict]:
    """Get step details for a job."""
    data = cci_get(f"project/{project_slug}/job/{job_number}", {}, token)
    return data.get("steps", [])


def get_step_logs(step_url: str, token: str) -> str:
    """Download log output for a specific step."""
    req = urllib.request.Request(
        step_url,
        headers={
            "Circle-Token": token,
            "User-Agent": "greenlight-ci-harvester/1.0",
        },
    )
    try:
        if not step_url.startswith("https://"):
            raise ValueError(f"Unsafe URL scheme: {step_url}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            # CircleCI logs can be gzipped
            if content[:2] == b"\x1f\x8b":
                import gzip

                content = gzip.decompress(content)
            log_data = json.loads(content)
            if isinstance(log_data, list):
                return "\n".join(item.get("message", "") for item in log_data)
            return str(log_data)
    except Exception:
        return ""


def extract_error_context(log: str, max_lines: int = 100) -> str:
    """Extract the most relevant error context from a log."""
    lines = log.split("\n")

    # Find lines with error signals
    error_line_indices = []
    for i, line in enumerate(lines):
        for pattern in ERROR_SECTION_PATTERNS:
            if pattern.search(line):
                error_line_indices.append(i)
                break

    if not error_line_indices:
        # No specific errors found — return last N lines
        return "\n".join(lines[-max_lines:])

    # Return context around error lines
    context_indices = set()
    for idx in error_line_indices:
        for j in range(max(0, idx - 3), min(len(lines), idx + 20)):
            context_indices.add(j)

    relevant_lines = [lines[i] for i in sorted(context_indices)]
    return "\n".join(relevant_lines[:max_lines])


def find_fixing_commit(
    repo_slug: str,
    failed_sha: str,
    branch: str,
    gh_token: str,
) -> Optional[dict]:
    """
    Find the commit on the branch that came after the failure and fixed CI.
    This is a heuristic: we look at commits after failed_sha on the branch.
    """
    if not gh_token:
        return None

    # repo_slug is "github/owner/repo" → extract "owner/repo"
    parts = repo_slug.split("/")
    if len(parts) < 3:
        return None
    owner_repo = "/".join(parts[1:3])

    # Get commits after failed_sha
    commits_data = gh_get(
        f"repos/{owner_repo}/commits",
        {"sha": branch, "per_page": 10},
        gh_token,
    )

    commits = commits_data if isinstance(commits_data, list) else []
    if not commits:
        return None

    # Find position of failed_sha in history
    failed_index = None
    for i, commit in enumerate(commits):
        if commit.get("sha", "").startswith(failed_sha[:8]):
            failed_index = i
            break

    if failed_index is None or failed_index == 0:
        return None

    # The commit before failed_sha in list (later in time) is the potential fix
    fix_commit = commits[failed_index - 1]

    # Get the diff for this commit
    sha = fix_commit.get("sha", "")
    diff_data = gh_get(f"repos/{owner_repo}/commits/{sha}", {}, gh_token)

    files = diff_data.get("files", [])
    # Filter to CI-relevant files
    ci_files = [
        f
        for f in files
        if any(
            kw in f.get("filename", "")
            for kw in [
                ".yml",
                ".yaml",
                ".json",
                "Dockerfile",
                "requirements",
                "package",
            ]
        )
    ]

    return {
        "sha": sha,
        "message": fix_commit.get("commit", {}).get("message", "")[:200],
        "author": fix_commit.get("commit", {}).get("author", {}).get("name", ""),
        "files_changed": len(files),
        "ci_files_changed": [f.get("filename") for f in ci_files[:10]],
        "diff_patch": "\n".join(f.get("patch", "")[:500] for f in ci_files[:3]),
    }


def process_failed_pipeline(
    project_slug: str,
    pipeline: dict,
    workflows: list[dict],
    token: str,
    gh_token: str,
    branch: str,
) -> Optional[dict]:
    """
    Process a failed pipeline to extract (failure_log, fix_commit) pair.
    """
    vcs = pipeline.get("vcs", {})
    failed_sha = vcs.get("revision", "")
    repo_url = vcs.get("origin_repository_url", "")

    failed_workflow = None
    for wf in workflows:
        if wf.get("status") in ("failed", "error"):
            failed_workflow = wf
            break

    if not failed_workflow:
        return None

    # Get jobs
    jobs = get_workflow_jobs(failed_workflow["id"], token)
    failed_jobs = [j for j in jobs if j.get("status") in ("failed", "blocked")]
    if not failed_jobs:
        return None

    # Get log for first failed job
    first_failed = failed_jobs[0]
    job_number = first_failed.get("job_number")
    if not job_number:
        return None

    steps = get_job_steps(project_slug, job_number, token)

    # Find failed step and get its log
    failure_log = ""
    for step in steps:
        for action in step.get("actions", []):
            if action.get("status") == "failed":
                log_url = action.get("output_url", "")
                if log_url:
                    raw_log = get_step_logs(log_url, token)
                    failure_log = extract_error_context(raw_log)
                    break
        if failure_log:
            break

    if not failure_log:
        return None

    # Find fixing commit
    fix_info = find_fixing_commit(project_slug, failed_sha, branch, gh_token)

    return {
        "platform": "circleci",
        "project_slug": project_slug,
        "repo_url": repo_url,
        "pipeline_id": pipeline.get("id"),
        "workflow_name": failed_workflow.get("name", ""),
        "failed_job_name": first_failed.get("name", ""),
        "failed_sha": failed_sha,
        "branch": branch,
        "failure_log": failure_log[:5000],
        "fix_commit": fix_info,
        "has_fix": fix_info is not None,
        "created_at": pipeline.get("created_at", ""),
    }


def load_progress() -> dict:
    if CIRCLECI_PROGRESS_FILE.exists():
        return json.loads(CIRCLECI_PROGRESS_FILE.read_text())
    return {"processed_slugs": [], "total_pairs": 0}


def save_progress(progress: dict) -> None:
    CIRCLECI_PROGRESS_FILE.write_text(json.dumps(progress))


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CIRCLECI_LOGS_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect CircleCI failure logs for GreenLight CI training"
    )
    parser.add_argument("--token", default=os.environ.get("CIRCLECI_TOKEN", ""))
    parser.add_argument("--gh-token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--max-projects", type=int, default=500)
    parser.add_argument("--max-pipelines-per-project", type=int, default=50)
    parser.add_argument(
        "--org",
        type=str,
        default=None,
        help="Specific org to crawl (e.g., github/python)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.token:
        print("Error: set CIRCLECI_TOKEN or use --token")
        return

    progress = (
        load_progress() if args.resume else {"processed_slugs": [], "total_pairs": 0}
    )
    processed_slugs = set(progress.get("processed_slugs", []))
    total_pairs = progress.get("total_pairs", 0)

    # Build project slug list
    project_slugs = []
    if args.org:
        project_slugs.append(args.org)
    else:
        project_slugs.extend(KNOWN_CIRCLECI_REPOS)

    print("=== CIRCLECI LOG HARVESTER ===")
    print(f"Projects to process: {len(project_slugs)}")
    print(f"Already processed: {len(processed_slugs)}")

    for project_slug in project_slugs[: args.max_projects]:
        if project_slug in processed_slugs:
            continue

        print(f"\n  Processing: {project_slug}")
        pairs_found = 0

        for branch in ["main", "master"]:
            page_token = None
            pipelines_fetched = 0

            while pipelines_fetched < args.max_pipelines_per_project:
                data = get_project_pipelines(
                    project_slug, args.token, page_token, branch
                )
                pipelines = data.get("items", [])
                if not pipelines:
                    break

                for pipeline in pipelines:
                    # Accept "errored" and "created" pipeline states; workflow-level
                    # status check in process_failed_pipeline determines actual failure.
                    if pipeline.get("state") not in ("errored", "created"):
                        continue

                    workflows = get_pipeline_workflows(pipeline["id"], args.token)
                    time.sleep(0.1)

                    record = process_failed_pipeline(
                        project_slug,
                        pipeline,
                        workflows,
                        args.token,
                        args.gh_token,
                        branch,
                    )
                    if record:
                        save_records([record])
                        total_pairs += 1
                        pairs_found += 1

                    time.sleep(0.2)

                pipelines_fetched += len(pipelines)
                page_token = data.get("next_page_token")
                if not page_token:
                    break
                time.sleep(0.3)

        processed_slugs.add(project_slug)
        progress = {
            "processed_slugs": list(processed_slugs),
            "total_pairs": total_pairs,
        }
        save_progress(progress)
        print(f"    +{pairs_found} pairs (total: {total_pairs})")
        time.sleep(0.5)

    print("\n=== DONE ===")
    print(f"Total CircleCI failure->fix pairs: {total_pairs}")
    print(f"Output: {CIRCLECI_LOGS_FILE}")


if __name__ == "__main__":
    main()
