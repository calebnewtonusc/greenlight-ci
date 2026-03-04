"""
GreenLight CI — GitHub Actions CI Log Discovery
Fetches failed workflow runs from public repositories and pairs them with
the fix commits that restored CI green.

Usage:
  python discovery/github_ci_logs.py --repos 10000 --workers 30
  python discovery/github_ci_logs.py --dep-drift-mode --workers 20
  python discovery/github_ci_logs.py --repo django/django --output data/raw/django/
"""

import asyncio
import gzip
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp
import typer
from loguru import logger

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


@dataclass
class FailedRun:
    """A failed CI run paired with its fix commit."""
    repo: str
    run_id: int
    workflow_name: str
    failing_sha: str
    fix_sha: Optional[str]
    language: str
    log_text: str
    fix_diff: Optional[str]
    failure_labels: list[str]


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None) -> dict | list | None:
    """Fetch JSON from GitHub API with rate limit handling."""
    for attempt in range(3):
        try:
            async with session.get(url, headers=HEADERS, params=params) as resp:
                if resp.status == 403:
                    # Rate limit
                    reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset_at - time.time(), 1)
                    logger.warning(f"Rate limited. Waiting {wait:.0f}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    logger.debug(f"HTTP {resp.status} for {url}")
                    return None
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.debug(f"Request error (attempt {attempt + 1}): {e}")
            await asyncio.sleep(2**attempt)
    return None


async def get_repo_language(session: aiohttp.ClientSession, repo: str) -> str:
    """Get primary language of a repository."""
    data = await fetch_json(session, f"{GITHUB_API}/repos/{repo}")
    if data:
        return data.get("language", "unknown") or "unknown"
    return "unknown"


async def fetch_run_log(session: aiohttp.ClientSession, repo: str, run_id: int) -> str:
    """Download and extract CI run log (GitHub returns a zip)."""
    url = f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/logs"
    try:
        async with session.get(url, headers=HEADERS, allow_redirects=True) as resp:
            if resp.status not in (200, 302):
                return ""
            raw = await resp.read()
            # GitHub returns a zip file
            if raw[:2] == b"PK":
                zf = zipfile.ZipFile(BytesIO(raw))
                logs = []
                for name in zf.namelist():
                    # Prioritize test output files
                    if any(kw in name.lower() for kw in ["test", "build", "check", "lint"]):
                        try:
                            logs.insert(0, zf.read(name).decode("utf-8", errors="replace"))
                        except Exception:
                            pass
                    else:
                        try:
                            logs.append(zf.read(name).decode("utf-8", errors="replace"))
                        except Exception:
                            pass
                combined = "\n".join(logs)
                # Truncate to 16k chars to fit model context
                return combined[:16384]
            return raw.decode("utf-8", errors="replace")[:16384]
    except Exception as e:
        logger.debug(f"Log fetch failed for run {run_id}: {e}")
        return ""


async def get_failed_runs(
    session: aiohttp.ClientSession,
    repo: str,
    per_page: int = 100,
    max_pages: int = 5,
) -> list[dict]:
    """Fetch failed workflow runs for a repository."""
    runs = []
    page = 1
    while page <= max_pages:
        data = await fetch_json(
            session,
            f"{GITHUB_API}/repos/{repo}/actions/runs",
            params={"conclusion": "failure", "per_page": per_page, "page": page},
        )
        if not data or not data.get("workflow_runs"):
            break
        runs.extend(data["workflow_runs"])
        if len(data["workflow_runs"]) < per_page:
            break
        page += 1
    return runs


async def find_fix_commit(
    session: aiohttp.ClientSession,
    repo: str,
    failing_sha: str,
    workflow_name: str,
) -> tuple[str | None, str | None]:
    """
    Walk the commit history after `failing_sha` to find the next commit
    that restored CI green on the same workflow.
    Returns (fix_sha, fix_diff) or (None, None) if not found.
    """
    # Get commits after the failing SHA
    commits_data = await fetch_json(
        session,
        f"{GITHUB_API}/repos/{repo}/commits",
        params={"per_page": 20},
    )
    if not commits_data:
        return None, None

    # Find index of failing SHA in commit list
    shas = [c["sha"] for c in commits_data]
    if failing_sha not in shas:
        return None, None

    fail_idx = shas.index(failing_sha)
    # Check commits before the failing sha (more recent in git log order)
    for candidate in commits_data[:fail_idx]:
        candidate_sha = candidate["sha"]
        # Check if this commit has a successful run of the same workflow
        runs_data = await fetch_json(
            session,
            f"{GITHUB_API}/repos/{repo}/actions/runs",
            params={"head_sha": candidate_sha, "per_page": 10},
        )
        if not runs_data:
            continue
        for run in runs_data.get("workflow_runs", []):
            if (
                run.get("name") == workflow_name
                and run.get("conclusion") == "success"
            ):
                # Found the fix — get the diff
                diff_data = await fetch_json(
                    session,
                    f"{GITHUB_API}/repos/{repo}/compare/{failing_sha}...{candidate_sha}",
                )
                diff_text = ""
                if diff_data:
                    # GitHub compare gives file-level diffs
                    files = diff_data.get("files", [])
                    diff_parts = []
                    for f in files[:10]:  # Cap at 10 files
                        patch = f.get("patch", "")
                        if patch:
                            diff_parts.append(f"--- a/{f['filename']}\n+++ b/{f['filename']}\n{patch}")
                    diff_text = "\n".join(diff_parts)

                return candidate_sha, diff_text

    return None, None


async def process_repo(
    session: aiohttp.ClientSession,
    repo: str,
    output_dir: Path,
    dep_drift_mode: bool = False,
) -> int:
    """Process a single repository: fetch failures and find fixes."""
    language = await get_repo_language(session, repo)
    repo_slug = repo.replace("/", "__")
    output_file = output_dir / f"{repo_slug}.jsonl"

    if output_file.exists():
        logger.debug(f"Skipping {repo} (already processed)")
        return 0

    failed_runs = await get_failed_runs(session, repo)
    if not failed_runs:
        return 0

    # In dep-drift mode, only look at Dependabot/Renovate triggered runs
    if dep_drift_mode:
        failed_runs = [
            r for r in failed_runs
            if any(kw in r.get("actor", {}).get("login", "").lower()
                   for kw in ["dependabot", "renovate"])
        ]

    pairs_written = 0
    with open(output_file, "w") as f:
        for run in failed_runs[:20]:  # Cap per-repo
            run_id = run["id"]
            failing_sha = run.get("head_sha", "")
            workflow_name = run.get("name", "")

            if not failing_sha:
                continue

            log_text = await fetch_run_log(session, repo, run_id)
            if len(log_text) < 200:  # Too short to be useful
                continue

            fix_sha, fix_diff = await find_fix_commit(session, repo, failing_sha, workflow_name)

            record = {
                "id": f"github_actions_{repo_slug}_{run_id}",
                "source": "github_actions",
                "repo": repo,
                "language": language,
                "ci_platform": "github_actions",
                "run_id": run_id,
                "workflow_name": workflow_name,
                "failing_sha": failing_sha,
                "fix_sha": fix_sha,
                "ci_log": log_text,
                "fix_diff": fix_diff,
                "has_fix": fix_sha is not None,
                "dep_drift_mode": dep_drift_mode,
            }
            f.write(json.dumps(record) + "\n")
            pairs_written += 1

    logger.info(f"  {repo}: {pairs_written} pairs written to {output_file}")
    return pairs_written


async def discover_popular_repos(session: aiohttp.ClientSession, limit: int = 10000) -> list[str]:
    """Discover popular GitHub repos with active CI."""
    repos = []
    languages = ["python", "javascript", "go", "java", "ruby", "rust", "typescript"]

    per_lang = limit // len(languages)
    for lang in languages:
        page = 1
        lang_count = 0
        while lang_count < per_lang:
            data = await fetch_json(
                session,
                f"{GITHUB_API}/search/repositories",
                params={
                    "q": f"language:{lang} stars:>500 archived:false",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                repos.append(item["full_name"])
                lang_count += 1
                if lang_count >= per_lang:
                    break
            if len(data["items"]) < 100:
                break
            page += 1
            await asyncio.sleep(0.5)  # Respect rate limits

    return list(set(repos))[:limit]


async def main_async(
    repos: int,
    workers: int,
    output_dir: Path,
    dep_drift_mode: bool,
    repo: str | None,
):
    """Main async entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        if repo:
            repo_list = [repo]
        else:
            logger.info(f"Discovering top {repos} repositories...")
            repo_list = await discover_popular_repos(session, limit=repos)
            logger.info(f"Discovered {len(repo_list)} repositories")

        # Process repos in batches
        total_pairs = 0
        semaphore = asyncio.Semaphore(workers)

        async def process_with_sem(r: str) -> int:
            async with semaphore:
                try:
                    return await process_repo(session, r, output_dir, dep_drift_mode)
                except Exception as e:
                    logger.debug(f"Error processing {r}: {e}")
                    return 0

        tasks = [process_with_sem(r) for r in repo_list]
        results = await asyncio.gather(*tasks)
        total_pairs = sum(results)

    logger.info(f"Discovery complete. Total pairs collected: {total_pairs}")


app = typer.Typer()


@app.command()
def main(
    repos: int = typer.Option(10000, help="Number of top repositories to scan"),
    workers: int = typer.Option(30, help="Number of concurrent HTTP workers"),
    output: Path = typer.Option(Path("data/raw/github_actions"), help="Output directory"),
    dep_drift_mode: bool = typer.Option(False, "--dep-drift-mode", help="Only collect Dependabot/Renovate failures"),
    repo: str = typer.Option(None, help="Process a single specific repo (owner/repo)"),
):
    """Fetch failed GitHub Actions runs and paired fix commits."""
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set. Set it in .env before running.")
        raise typer.Exit(1)

    logger.info(f"Starting GitHub CI log discovery: {repos} repos, {workers} workers")
    logger.info(f"Output: {output}")
    if dep_drift_mode:
        logger.info("Dep-drift mode: only collecting Dependabot/Renovate failures")

    asyncio.run(main_async(repos, workers, output, dep_drift_mode, repo))


if __name__ == "__main__":
    app()
