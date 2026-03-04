"""
gitlab_ci_logs.py - GitLab CI failed pipeline log discovery and collection.

Uses the GitLab API to collect failed pipeline logs from public GitLab projects.
Pairs each failure log with the commit that fixed CI.

API: https://docs.gitlab.com/ee/api/
GitLab public projects endpoint: GET /projects?visibility=public&order_by=star_count

Target: 50k GitLab failure -> fix pairs.

Usage:
    export GITLAB_TOKEN=your_token
    python discovery/gitlab_ci_logs.py
    python discovery/gitlab_ci_logs.py --max-projects 1000
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
GITLAB_LOGS_FILE = DATA_DIR / "gitlab_failure_logs.jsonl"
GITLAB_PROGRESS_FILE = DATA_DIR / "gitlab_progress.json"

GITLAB_BASE = "https://gitlab.com/api/v4"

# ─── Well-known GitLab public repos ──────────────────────────────────────────
KNOWN_GITLAB_PROJECTS = [
    # GitLab-hosted open source
    "gitlab-org/gitlab",
    "gitlab-org/gitlab-runner",
    "gitlab-org/gitaly",
    "gitlab-org/gitlab-workhorse",
    "gitlab-com/www-gitlab-com",
    # Python projects
    "pgjones/quart",
    "tox-dev/tox",
    "httpie/httpie",
    # GNOME
    "gnome/gnome-shell",
    "gnome/gtk",
    "gnome/glib",
    # KDE
    "kde/plasma-desktop",
    "kde/dolphin",
    # Others
    "inkscape/inkscape",
    "blender/blender",
    "godotengine/godot",
    "libreoffice/core",
    "freedesktop/mesa",
    "gstreamer/gstreamer",
    "videolan/vlc",
    "fdroid/fdroidclient",
    "fdroid/fdroidserver",
]

# ─── Error extraction patterns ────────────────────────────────────────────────
GITLAB_ERROR_PATTERNS = [
    re.compile(r"(ERROR|FAILED|Error|error:)\s+.*", re.I),
    re.compile(r"(Job failed|Build failed|Pipeline failed)", re.I),
    re.compile(r"exit status \d+", re.I),
    re.compile(r"make\[\d+\].*Error \d+", re.I),
    re.compile(r"fatal:.*", re.I),
    re.compile(r"FAIL\s+", re.I),
]


def gl_get(endpoint: str, params: dict, token: str = "") -> dict:
    """Make GitLab API request."""
    url = f"{GITLAB_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "greenlight-ci-harvester/1.0",
        "Accept": "application/json",
    }
    if token:
        headers["PRIVATE-TOKEN"] = token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            return json.loads(data)
    except Exception as e:
        if hasattr(e, "code") and e.code == 429:
            time.sleep(15)
        return {}


def gl_get_list(endpoint: str, params: dict, token: str = "") -> list:
    """Get a list response from GitLab API (handles both dict and list responses)."""
    result = gl_get(endpoint, params, token)
    if isinstance(result, list):
        return result
    return result.get("items", result.get("data", []))


def search_public_projects(
    token: str,
    page: int = 1,
    per_page: int = 100,
    order_by: str = "star_count",
    min_stars: int = 50,
) -> list[dict]:
    """Search for public GitLab projects with CI enabled."""
    url = f"{GITLAB_BASE}/projects?" + urllib.parse.urlencode(
        {
            "visibility": "public",
            "order_by": order_by,
            "sort": "desc",
            "with_ci_enabled": "true",
            "page": page,
            "per_page": per_page,
        }
    )
    headers = {"User-Agent": "greenlight-ci-harvester/1.0"}
    if token:
        headers["PRIVATE-TOKEN"] = token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            projects = json.loads(resp.read())
    except Exception:
        return []

    if not isinstance(projects, list):
        return []

    return [p for p in projects if p.get("star_count", 0) >= min_stars]


def get_project_pipelines(
    project_id: int,
    token: str,
    status: str = "failed",
    page: int = 1,
    per_page: int = 50,
) -> list[dict]:
    """Get pipelines for a project filtered by status."""
    result = gl_get(
        f"projects/{project_id}/pipelines",
        {"status": status, "page": page, "per_page": per_page},
        token,
    )
    return result if isinstance(result, list) else []


def get_pipeline_jobs(
    project_id: int,
    pipeline_id: int,
    token: str,
) -> list[dict]:
    """Get jobs for a pipeline."""
    result = gl_get(
        f"projects/{project_id}/pipelines/{pipeline_id}/jobs",
        {"per_page": 100},
        token,
    )
    return result if isinstance(result, list) else []


def get_job_trace(project_id: int, job_id: int, token: str) -> str:
    """Download trace (log) for a job."""
    url = f"{GITLAB_BASE}/projects/{project_id}/jobs/{job_id}/trace"
    headers = {"User-Agent": "greenlight-ci-harvester/1.0"}
    if token:
        headers["PRIVATE-TOKEN"] = token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def get_pipeline_commit(
    project_id: int,
    pipeline_sha: str,
    token: str,
) -> Optional[dict]:
    """Get commit info for a pipeline."""
    return gl_get(f"projects/{project_id}/repository/commits/{pipeline_sha}", {}, token)


def find_fixing_pipeline(
    project_id: int,
    failed_pipeline: dict,
    token: str,
) -> Optional[dict]:
    """Find the pipeline that fixed CI after the failed one."""
    ref = failed_pipeline.get("ref", "main")
    failed_id = failed_pipeline.get("id", 0)

    # Get recent successful pipelines on same branch
    successful = gl_get(
        f"projects/{project_id}/pipelines",
        {"status": "success", "ref": ref, "per_page": 20},
        token,
    )
    if not isinstance(successful, list):
        return None

    # Sort by id descending so the most recent successes are checked first;
    # then find the first success that came after the failure
    successful = sorted(successful, key=lambda p: p.get("id", 0), reverse=True)
    for pipeline in successful:
        if pipeline.get("id", 0) > failed_id:
            commit_sha = pipeline.get("sha", "")
            # Get commit message
            commit_info = get_pipeline_commit(
                project_id, pipeline.get("sha", ""), token
            )
            return {
                "pipeline_id": pipeline.get("id"),
                "sha": commit_sha,
                "ref": ref,
                "commit_message": (commit_info.get("title") or "")[:200]
                if commit_info
                else "",
                "web_url": pipeline.get("web_url", ""),
                "duration": pipeline.get("duration", 0),
            }
    return None


def extract_error_context(log: str, max_lines: int = 80) -> str:
    """Extract error context from a GitLab CI job trace."""
    # Remove ANSI codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    log = ansi_escape.sub("", log)
    lines = log.split("\n")

    # Find error lines
    error_indices = []
    for i, line in enumerate(lines):
        for pattern in GITLAB_ERROR_PATTERNS:
            if pattern.search(line):
                error_indices.append(i)
                break

    if not error_indices:
        return "\n".join(lines[-max_lines:])

    context = set()
    for idx in error_indices:
        for j in range(max(0, idx - 3), min(len(lines), idx + 20)):
            context.add(j)

    return "\n".join(lines[i] for i in sorted(context))[:5000]


def process_failed_pipeline(
    project: dict,
    pipeline: dict,
    token: str,
) -> Optional[dict]:
    """Process a failed GitLab pipeline into a training record."""
    project_id = project.get("id")
    pipeline_id = pipeline.get("id")

    jobs = get_pipeline_jobs(project_id, pipeline_id, token)
    failed_jobs = [j for j in jobs if j.get("status") in ("failed",)]
    if not failed_jobs:
        return None

    # Get log for most recent failed job
    first_failed = failed_jobs[0]
    log = get_job_trace(project_id, first_failed.get("id"), token)
    if not log:
        return None

    error_context = extract_error_context(log)

    # Find fixing pipeline
    fix_info = find_fixing_pipeline(project_id, pipeline, token)
    time.sleep(0.1)

    return {
        "platform": "gitlab_ci",
        "project_path": project.get("path_with_namespace", ""),
        "project_url": project.get("web_url", ""),
        "pipeline_id": pipeline_id,
        "failed_sha": pipeline.get("sha", ""),
        "branch": pipeline.get("ref", ""),
        "failed_job_name": first_failed.get("name", ""),
        "failed_stage": first_failed.get("stage", ""),
        "failure_log": error_context,
        "fix_pipeline": fix_info,
        "has_fix": fix_info is not None,
        "language": project.get("predominant_language") or "unknown",
        "project_stars": project.get("star_count", 0),
        "created_at": pipeline.get("created_at", ""),
    }


def load_progress() -> dict:
    if GITLAB_PROGRESS_FILE.exists():
        return json.loads(GITLAB_PROGRESS_FILE.read_text())
    return {"processed_projects": [], "total_pairs": 0}


def save_progress(progress: dict) -> None:
    GITLAB_PROGRESS_FILE.write_text(json.dumps(progress))


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GITLAB_LOGS_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect GitLab CI failure logs for GreenLight CI training"
    )
    parser.add_argument("--token", default=os.environ.get("GITLAB_TOKEN", ""))
    parser.add_argument("--max-projects", type=int, default=500)
    parser.add_argument("--max-pipelines-per-project", type=int, default=30)
    parser.add_argument(
        "--discover-projects",
        action="store_true",
        help="Discover additional public projects via API search",
    )
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    progress = (
        load_progress() if args.resume else {"processed_projects": [], "total_pairs": 0}
    )
    processed_projects = set(progress.get("processed_projects", []))
    total_pairs = progress.get("total_pairs", 0)

    # Build project list
    projects_to_process: list[dict] = []

    # Start with known projects
    for path in KNOWN_GITLAB_PROJECTS:
        projects_to_process.append({"path_with_namespace": path, "id": None})

    # Optionally discover more via API
    if args.discover_projects:
        print("Discovering additional public projects...")
        for page in range(1, 6):  # up to 500 more projects
            discovered = search_public_projects(
                args.token, page=page, min_stars=args.min_stars
            )
            projects_to_process.extend(discovered)
            if len(discovered) < 100:
                break
            time.sleep(0.5)

    print("=== GITLAB CI LOG HARVESTER ===")
    print(f"Projects to process: {len(projects_to_process)}")

    for project in projects_to_process[: args.max_projects]:
        project_path = project.get("path_with_namespace", "")
        if not project_path or project_path in processed_projects:
            continue

        # Get project ID if not known
        project_id = project.get("id")
        if not project_id:
            proj_data = gl_get(
                f"projects/{urllib.parse.quote(project_path, safe='')}",
                {},
                args.token,
            )
            project_id = proj_data.get("id")
            if not project_id:
                continue
            project = {**project, **proj_data}

        print(f"\n  Processing: {project_path} (id={project_id})")
        pairs_found = 0

        pipelines = get_project_pipelines(
            project_id, args.token, "failed", per_page=args.max_pipelines_per_project
        )

        for pipeline in pipelines:
            record = process_failed_pipeline(project, pipeline, args.token)
            if record:
                save_records([record])
                total_pairs += 1
                pairs_found += 1
            time.sleep(0.2)

        processed_projects.add(project_path)
        save_progress(
            {"processed_projects": list(processed_projects), "total_pairs": total_pairs}
        )
        print(f"    +{pairs_found} pairs (total: {total_pairs})")
        time.sleep(0.5)

    print("\n=== DONE ===")
    print(f"Total GitLab CI failure->fix pairs: {total_pairs}")
    print(f"Output: {GITLAB_LOGS_FILE}")


if __name__ == "__main__":
    main()
