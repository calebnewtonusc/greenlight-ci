"""
travis_ci_archive.py - Travis CI build log discovery and collection.

Travis CI has a public API for accessing build logs from public repos.
Scrapes travis-ci.com public repos (Python, Ruby, Node.js heavy).
Pairs build failure logs with the branch/commit where CI went green again.

API: https://api.travis-ci.com/
Docs: https://developer.travis-ci.com/

Usage:
    export TRAVIS_TOKEN=your_token
    python discovery/travis_ci_archive.py
    python discovery/travis_ci_archive.py --language python
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
TRAVIS_LOGS_FILE = DATA_DIR / "travis_failure_logs.jsonl"
TRAVIS_PROGRESS_FILE = DATA_DIR / "travis_progress.json"

TRAVIS_BASE = "https://api.travis-ci.com"

# ─── Popular repos with Travis CI ────────────────────────────────────────────
# These are repos known to use or have used Travis CI
TRAVIS_REPOS = [
    # Python
    "django/django",
    "pallets/flask",
    "psf/requests",
    "pypa/pip",
    "sqlalchemy/sqlalchemy",
    "pytest-dev/pytest",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "celery/celery",
    "redis/redis-py",
    "boto/boto3",
    "docker/docker-py",
    "paramiko/paramiko",
    "httpie/httpie",
    "scrapy/scrapy",
    "ansible/ansible",
    # Ruby
    "rails/rails",
    "jekyll/jekyll",
    "ruby/ruby",
    "rubocop/rubocop",
    "bundler/bundler",
    "sinatra/sinatra",
    # Node.js
    "expressjs/express",
    "lodash/lodash",
    "npm/cli",
    "eslint/eslint",
    "prettier/prettier",
    "babel/babel",
    "jestjs/jest",
    "mochajs/mocha",
    # Go
    "golang/go",
    "docker/cli",
    "kubernetes/kubernetes",
    # Java
    "spring-projects/spring-boot",
    "apache/kafka",
    "elastic/elasticsearch",
]

# ─── Error pattern for log truncation ────────────────────────────────────────
TRAVIS_ERROR_PATTERNS = [
    re.compile(r'(The command .+ exited with \d+)', re.I),
    re.compile(r'(FAILED|ERROR|error:)\s+', re.I),
    re.compile(r'Build failed', re.I),
    re.compile(r'bundler: failed', re.I),
    re.compile(r'rake aborted!', re.I),
    re.compile(r'npm ERR!', re.I),
    re.compile(r'Test failed', re.I),
]


def travis_get(endpoint: str, params: dict, token: str) -> dict:
    """Make Travis CI API request."""
    url = f"{TRAVIS_BASE}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Travis-API-Version": "3",
        "User-Agent": "greenlight-ci-harvester/1.0",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [WARN] Travis API error {endpoint}: {e}")
        return {}


def get_repo_builds(slug: str, token: str, limit: int = 50) -> list[dict]:
    """Get recent builds for a repo."""
    data = travis_get(
        f"repo/{urllib.parse.quote(slug, safe='')}/builds",
        {"limit": limit, "sort_by": "created_at:desc"},
        token,
    )
    return data.get("builds", [])


def get_build_jobs(build_id: int, token: str) -> list[dict]:
    """Get jobs for a build."""
    data = travis_get(f"build/{build_id}/jobs", {}, token)
    return data.get("jobs", [])


def get_job_log(job_id: int, token: str) -> str:
    """Get log content for a job."""
    data = travis_get(f"job/{job_id}/log", {}, token)
    # Log content can be in "content" or at a separate URL
    content = data.get("content") or data.get("log") or ""
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    return str(content)


def extract_travis_error_context(log: str, max_lines: int = 80) -> str:
    """Extract relevant error context from Travis log."""
    lines = log.split("\n")

    # Remove ANSI color codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    lines = [ansi_escape.sub('', line) for line in lines]

    # Find error lines
    error_indices = []
    for i, line in enumerate(lines):
        for pattern in TRAVIS_ERROR_PATTERNS:
            if pattern.search(line):
                error_indices.append(i)
                break

    if not error_indices:
        return "\n".join(lines[-max_lines:])

    # Build context window around errors
    context = set()
    for idx in error_indices:
        for j in range(max(0, idx - 5), min(len(lines), idx + 25)):
            context.add(j)

    return "\n".join(lines[i] for i in sorted(context))[:5000]


def find_next_passing_build(
    slug: str,
    failed_build_number: int,
    branch: str,
    token: str,
) -> Optional[dict]:
    """Find the first passing build after the failed build on the same branch."""
    # Get more builds to find the fix
    data = travis_get(
        f"repo/{urllib.parse.quote(slug, safe='')}/builds",
        {
            "branch.name": branch,
            "build.number": str(failed_build_number + 1),
            "limit": 20,
            "sort_by": "created_at:asc",
        },
        token,
    )
    builds = data.get("builds", [])

    for build in builds:
        if build.get("state") == "passed":
            return {
                "build_number": build.get("number"),
                "commit_sha": build.get("commit", {}).get("sha", "")[:12],
                "commit_message": build.get("commit", {}).get("message", "")[:200],
                "compare_url": build.get("commit", {}).get("compare_url", ""),
                "duration": build.get("duration", 0),
            }
    return None


def process_failed_build(
    slug: str,
    build: dict,
    token: str,
) -> Optional[dict]:
    """Process a single failed build into a training record."""
    if build.get("state") not in ("failed", "errored"):
        return None

    build_id = build.get("id")
    build_number = int(build.get("number", 0))
    branch = build.get("branch", {}).get("name", "main")
    commit_sha = build.get("commit", {}).get("sha", "")[:12]
    commit_message = build.get("commit", {}).get("message", "")[:200]

    # Get failed jobs
    jobs = get_build_jobs(build_id, token)
    failed_jobs = [j for j in jobs if j.get("state") in ("failed", "errored")]
    if not failed_jobs:
        return None

    # Get log for first failed job
    first_failed = failed_jobs[0]
    job_log = get_job_log(first_failed.get("id"), token)
    if not job_log:
        return None

    error_context = extract_travis_error_context(job_log)

    # Find fixing build
    fix_info = find_next_passing_build(slug, build_number, branch, token)
    time.sleep(0.1)

    # Detect language from build config
    config = build.get("config", {})
    language = config.get("language", "unknown")

    return {
        "platform": "travis_ci",
        "repo_slug": slug,
        "build_number": build_number,
        "build_id": build_id,
        "branch": branch,
        "language": language,
        "failed_job_name": first_failed.get("stage", {}).get("name", "test"),
        "failed_sha": commit_sha,
        "commit_message": commit_message,
        "failure_log": error_context,
        "fix_build": fix_info,
        "has_fix": fix_info is not None,
        "os": config.get("os", "linux"),
        "dist": config.get("dist", ""),
        "started_at": build.get("started_at", ""),
    }


def load_progress() -> dict:
    if TRAVIS_PROGRESS_FILE.exists():
        return json.loads(TRAVIS_PROGRESS_FILE.read_text())
    return {"processed_repos": [], "total_pairs": 0}


def save_progress(progress: dict) -> None:
    TRAVIS_PROGRESS_FILE.write_text(json.dumps(progress))


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAVIS_LOGS_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Travis CI failure logs for GreenLight CI training"
    )
    parser.add_argument("--token", default=os.environ.get("TRAVIS_TOKEN", ""))
    parser.add_argument("--language", type=str, default=None,
                        help="Filter repos by language (python, ruby, node)")
    parser.add_argument("--max-repos", type=int, default=200)
    parser.add_argument("--max-builds", type=int, default=100,
                        help="Max builds to check per repo")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    progress = load_progress() if args.resume else {"processed_repos": [], "total_pairs": 0}
    processed_repos = set(progress.get("processed_repos", []))
    total_pairs = progress.get("total_pairs", 0)

    # Filter repos by language if specified
    repos_to_process = TRAVIS_REPOS
    if args.language:
        lang_repos = {
            "python": [r for r in TRAVIS_REPOS if any(
                kw in r.lower() for kw in ["django", "flask", "request", "pip",
                                             "sqlalchemy", "pytest", "numpy", "pandas",
                                             "scikit", "celery", "redis", "boto",
                                             "docker", "paramiko", "httpie", "scrapy",
                                             "ansible"])],
            "ruby": [r for r in TRAVIS_REPOS if any(
                kw in r.lower() for kw in ["rails", "jekyll", "ruby", "rubocop",
                                             "bundler", "sinatra"])],
            "node": [r for r in TRAVIS_REPOS if any(
                kw in r.lower() for kw in ["express", "lodash", "npm", "eslint",
                                             "prettier", "babel", "jest", "mocha"])],
        }
        repos_to_process = lang_repos.get(args.language.lower(), TRAVIS_REPOS)

    print(f"=== TRAVIS CI LOG HARVESTER ===")
    print(f"Repos to process: {len(repos_to_process)}")

    for slug in repos_to_process[:args.max_repos]:
        if slug in processed_repos:
            continue

        print(f"\n  Processing: {slug}")
        builds = get_repo_builds(slug, args.token, limit=args.max_builds)
        time.sleep(0.3)

        pairs_found = 0
        for build in builds:
            record = process_failed_build(slug, build, args.token)
            if record:
                save_records([record])
                total_pairs += 1
                pairs_found += 1
            time.sleep(0.2)

        processed_repos.add(slug)
        save_progress({"processed_repos": list(processed_repos), "total_pairs": total_pairs})
        print(f"    +{pairs_found} pairs (total: {total_pairs})")
        time.sleep(0.5)

    print(f"\n=== DONE ===")
    print(f"Total Travis CI failure->fix pairs: {total_pairs}")
    print(f"Output: {TRAVIS_LOGS_FILE}")


if __name__ == "__main__":
    main()
