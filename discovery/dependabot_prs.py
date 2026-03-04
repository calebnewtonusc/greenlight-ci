"""
dependabot_prs.py - Dependabot/Renovate PR discovery for CI dependency training data.

Specifically scrapes Dependabot and Renovate bot PRs which always have:
- Exact dependency version bump (old_version -> new_version)
- CI results (pass/fail after the update)
- Structured commit format for easy parsing

Creates (old_lockfile_state, new_lockfile_state, ci_result) triples.
These are gold data for the DEP_DRIFT failure class.

Usage:
    export GITHUB_TOKEN=your_token
    python discovery/dependabot_prs.py
    python discovery/dependabot_prs.py --ecosystem npm
    python discovery/dependabot_prs.py --min-stars 100
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
DEPENDABOT_FILE = DATA_DIR / "dependabot_prs.jsonl"
DEP_PROGRESS_FILE = DATA_DIR / "dependabot_progress.json"

GH_BASE = "https://api.github.com"
GH_GRAPHQL = "https://api.github.com/graphql"

# ─── Dependabot PR title patterns ────────────────────────────────────────────
DEPENDABOT_PATTERNS = [
    re.compile(r"Bump (.+?) from ([\d.]+) to ([\d.]+)", re.I),
    re.compile(r"Update (.+?) requirement from ([\d.>=<^~]+) to ([\d.>=<^~]+)", re.I),
    re.compile(r"chore\(deps\): bump (.+?) from ([\d.]+) to ([\d.]+)", re.I),
    re.compile(r"chore\(deps-dev\): bump (.+?) from ([\d.]+) to ([\d.]+)", re.I),
    re.compile(r"Update (.+?) to ([\d.]+)", re.I),
    re.compile(r"Upgrade (.+?) from ([\d.]+) to ([\d.]+)", re.I),
]

# ─── Ecosystem detection from PR title and files ─────────────────────────────
ECOSYSTEM_INDICATORS = {
    "npm": ["package.json", "package-lock.json", "yarn.lock", "node_modules"],
    "pip": [
        "requirements.txt",
        "Pipfile",
        "Pipfile.lock",
        "setup.py",
        "pyproject.toml",
    ],
    "maven": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "cargo": ["Cargo.toml", "Cargo.lock"],
    "bundler": ["Gemfile", "Gemfile.lock"],
    "go": ["go.mod", "go.sum"],
    "composer": ["composer.json", "composer.lock"],
    "nuget": [".csproj", "packages.config", "*.nuspec"],
}

# ─── Search queries to find repos with Dependabot PRs ────────────────────────
DEPENDABOT_SEARCH_QUERIES = [
    "is:pr author:app/dependabot label:dependencies is:merged",
    "is:pr author:app/dependabot is:merged updated:>2024-01-01",
    "is:pr author:app/renovate is:merged label:dependencies",
    'is:pr title:"Bump" author:app/dependabot is:merged',
    'is:pr title:"chore(deps)" is:merged',
    "is:pr author:app/dependabot is:closed is:unmerged",  # Failed PRs = failing dep update
]

# Repos known to use Dependabot heavily
DEPENDABOT_HEAVY_REPOS = [
    "pallets/flask",
    "psf/requests",
    "django/django",
    "pytest-dev/pytest",
    "python-poetry/poetry",
    "fastapi/fastapi",
    "tiangolo/fastapi",
    "pydantic/pydantic",
    "encode/httpx",
    "encode/starlette",
    "expressjs/express",
    "nestjs/nest",
    "vercel/next.js",
    "vuejs/vue",
    "angular/angular",
    "rails/rails",
    "jekyll/jekyll",
    "github/docs",
    "microsoft/TypeScript",
    "rust-lang/cargo",
    "mozilla/pdf.js",
]


def gh_get(endpoint: str, params: dict, token: str) -> dict:
    """Make authenticated GitHub API request."""
    url = f"{GH_BASE}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "greenlight-ci-harvester/1.0",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        if hasattr(e, "code") and e.code in (403, 429):
            time.sleep(10)
        return {}


def gh_get_list(endpoint: str, params: dict, token: str) -> list:
    """Get list response from GitHub API."""
    result = gh_get(endpoint, params, token)
    if isinstance(result, list):
        return result
    return []


def parse_dep_bump(title: str) -> Optional[dict]:
    """Parse dependency version bump from PR title."""
    for pattern in DEPENDABOT_PATTERNS:
        match = pattern.search(title)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                return {
                    "package": groups[0].strip(),
                    "from_version": groups[1].strip() if len(groups) >= 3 else None,
                    "to_version": groups[-1].strip(),
                }
    return None


def detect_ecosystem(pr: dict, changed_files: list[dict]) -> str:
    """Detect the package ecosystem from PR context."""
    # Check PR title
    title = (pr.get("title") or "").lower()
    labels = [label.get("name", "").lower() for label in (pr.get("labels") or [])]
    body = (pr.get("body") or "").lower()

    # Check labels first
    for label in labels:
        for eco, _ in ECOSYSTEM_INDICATORS.items():
            if eco in label:
                return eco

    # Check changed files
    file_names = [f.get("filename", "") for f in changed_files]
    for eco, indicators in ECOSYSTEM_INDICATORS.items():
        for indicator in indicators:
            if any(indicator in fname for fname in file_names):
                return eco

    # Check title/body
    if "package.json" in body or "npm" in title:
        return "npm"
    if "requirements" in body or "pip" in title or "python" in title:
        return "pip"
    if "cargo.toml" in body or "rust" in title:
        return "cargo"
    if "gemfile" in body or "ruby" in title:
        return "bundler"
    if "go.mod" in body:
        return "go"

    return "unknown"


def get_pr_ci_status(owner: str, repo: str, pr_number: int, token: str) -> dict:
    """Get CI check results for a PR."""
    pr_data = gh_get(f"repos/{owner}/{repo}/pulls/{pr_number}", {}, token)
    head_sha = pr_data.get("head", {}).get("sha", "")
    if not head_sha:
        return {"status": "unknown", "checks": []}

    checks_data = gh_get(
        f"repos/{owner}/{repo}/commits/{head_sha}/check-runs", {}, token
    )
    check_runs = checks_data.get("check_runs", [])

    # Summarize CI results
    total = len(check_runs)
    passed = sum(1 for c in check_runs if c.get("conclusion") == "success")
    failed = sum(
        1 for c in check_runs if c.get("conclusion") in ("failure", "timed_out")
    )

    overall_status = "unknown"
    if total > 0:
        if failed > 0:
            overall_status = "failed"
        elif passed == total:
            overall_status = "passed"
        else:
            overall_status = "partial"

    return {
        "status": overall_status,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "checks": [
            {
                "name": c.get("name"),
                "conclusion": c.get("conclusion"),
                "app": c.get("app", {}).get("slug") if c.get("app") else None,
            }
            for c in check_runs[:10]
        ],
    }


def get_pr_file_diffs(owner: str, repo: str, pr_number: int, token: str) -> list[dict]:
    """Get file changes for a PR."""
    files = gh_get_list(
        f"repos/{owner}/{repo}/pulls/{pr_number}/files",
        {"per_page": 50},
        token,
    )
    # Filter to dependency files
    dep_files = []
    for f in files:
        fname = f.get("filename", "")
        if any(
            ind in fname.lower()
            for inds in ECOSYSTEM_INDICATORS.values()
            for ind in inds
        ):
            dep_files.append(
                {
                    "filename": fname,
                    "status": f.get("status"),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": (f.get("patch") or "")[:2000],
                }
            )
    return dep_files


def process_pr(
    owner: str,
    repo: str,
    pr: dict,
    token: str,
) -> Optional[dict]:
    """Process a Dependabot/Renovate PR into a training record."""
    pr_number = pr.get("number")
    title = pr.get("title", "")
    body = pr.get("body") or ""

    # Parse the dependency bump
    bump_info = parse_dep_bump(title)
    if not bump_info:
        return None

    # Get changed files
    changed_files = get_pr_file_diffs(owner, repo, pr_number, token)
    time.sleep(0.1)

    # Detect ecosystem
    ecosystem = detect_ecosystem(pr, changed_files)

    # Get CI status
    ci_status = get_pr_ci_status(owner, repo, pr_number, token)
    time.sleep(0.1)

    # Determine merge status
    merged = pr.get("merged_at") is not None
    merge_status = "merged" if merged else ("closed" if pr.get("closed_at") else "open")

    return {
        "type": "dependabot_pr",
        "repo": f"{owner}/{repo}",
        "repo_url": f"https://github.com/{owner}/{repo}",
        "pr_number": pr_number,
        "pr_url": pr.get("html_url"),
        "title": title,
        "bump_info": bump_info,
        "ecosystem": ecosystem,
        "merge_status": merge_status,
        "ci_result": ci_status,
        "ci_passed": ci_status["status"] == "passed",
        "changed_dep_files": changed_files,
        "num_dep_files_changed": len(changed_files),
        "body_snippet": body[:500] if body else "",
        "created_at": pr.get("created_at", ""),
        "merged_at": pr.get("merged_at", ""),
        "labels": [label.get("name") for label in (pr.get("labels") or [])],
        "author": (pr.get("user") or {}).get("login", ""),
    }


def get_repo_dependabot_prs(
    owner: str,
    repo: str,
    token: str,
    max_prs: int = 100,
    state: str = "all",
) -> list[dict]:
    """Get all Dependabot/Renovate PRs for a repo."""
    all_prs = []
    page = 1
    while len(all_prs) < max_prs:
        prs = gh_get_list(
            f"repos/{owner}/{repo}/pulls",
            {
                "state": state,
                "per_page": 100,
                "page": page,
                "sort": "created",
                "direction": "desc",
            },
            token,
        )
        if not prs:
            break

        # Filter to bot PRs
        bot_prs = [
            pr
            for pr in prs
            if (pr.get("user") or {}).get("login", "").lower()
            in (
                "dependabot[bot]",
                "dependabot-preview[bot]",
                "renovate[bot]",
                "renovate",
            )
        ]
        all_prs.extend(bot_prs)

        if len(prs) < 100:
            break
        page += 1
        time.sleep(0.2)

    return all_prs[:max_prs]


def load_progress() -> dict:
    if DEP_PROGRESS_FILE.exists():
        return json.loads(DEP_PROGRESS_FILE.read_text())
    return {"processed_repos": [], "total_records": 0}


def save_progress(progress: dict) -> None:
    DEP_PROGRESS_FILE.write_text(json.dumps(progress))


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEPENDABOT_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Dependabot/Renovate PRs for DEP_DRIFT training data"
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--ecosystem",
        type=str,
        default=None,
        help="Filter by ecosystem: npm, pip, cargo, maven, bundler, go",
    )
    parser.add_argument("--min-stars", type=int, default=10)
    parser.add_argument("--max-repos", type=int, default=200)
    parser.add_argument("--max-prs-per-repo", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.token:
        print("Error: set GITHUB_TOKEN or use --token")
        return

    progress = (
        load_progress() if args.resume else {"processed_repos": [], "total_records": 0}
    )
    processed_repos = set(progress.get("processed_repos", []))
    total_records = progress.get("total_records", 0)

    print("=== DEPENDABOT/RENOVATE PR HARVESTER ===")
    print(f"Processing {len(DEPENDABOT_HEAVY_REPOS)} known repos")

    for full_name in DEPENDABOT_HEAVY_REPOS[: args.max_repos]:
        if full_name in processed_repos:
            continue

        parts = full_name.split("/")
        if len(parts) != 2:
            continue
        owner, repo = parts

        print(f"\n  Processing: {full_name}")
        prs = get_repo_dependabot_prs(owner, repo, args.token, args.max_prs_per_repo)
        time.sleep(0.3)

        records = []
        for pr in prs:
            record = process_pr(owner, repo, pr, args.token)
            if record:
                if args.ecosystem and record.get("ecosystem") != args.ecosystem:
                    continue
                records.append(record)
            time.sleep(0.15)

        save_records(records)
        total_records += len(records)
        processed_repos.add(full_name)
        save_progress(
            {"processed_repos": list(processed_repos), "total_records": total_records}
        )
        print(f"    +{len(records)} dep bump records (total: {total_records})")
        time.sleep(0.5)

    print("\n=== DONE ===")
    print(f"Total Dependabot/Renovate PR records: {total_records}")
    print(f"Output: {DEPENDABOT_FILE}")


if __name__ == "__main__":
    main()
