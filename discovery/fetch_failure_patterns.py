"""
GreenLight CI — Open Source CI History Collection (Stream 2)
Collects longitudinal CI failure→fix chains from curated open source repositories.

For each repo, we walk the full git/CI history to identify consecutive
failure→success commit pairs and extract the diff between them.

Usage:
  python discovery/fetch_failure_patterns.py --workers 20
  python discovery/fetch_failure_patterns.py --env-mode  # env failure focus
"""

import asyncio
import json
import os
from pathlib import Path

import aiohttp
import typer
from loguru import logger

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
GITHUB_API = "https://api.github.com"

# Curated list of high-quality open source repos per language
CURATED_REPOS = {
    "python": [
        "django/django",
        "pallets/flask",
        "tiangolo/fastapi",
        "pandas-dev/pandas",
        "scikit-learn/scikit-learn",
        "numpy/numpy",
        "psf/requests",
        "sqlalchemy/sqlalchemy",
        "pytest-dev/pytest",
        "python/cpython",
        "celery/celery",
        "encode/httpx",
        "pydantic/pydantic",
        "huggingface/transformers",
        "aio-libs/aiohttp",
    ],
    "javascript": [
        "facebook/react",
        "vuejs/vue",
        "expressjs/express",
        "webpack/webpack",
        "jestjs/jest",
        "eslint/eslint",
        "nodejs/node",
        "vitejs/vite",
        "prettier/prettier",
        "babel/babel",
        "axios/axios",
        "lodash/lodash",
    ],
    "go": [
        "kubernetes/kubernetes",
        "moby/moby",
        "hashicorp/terraform",
        "gin-gonic/gin",
        "spf13/cobra",
        "go-chi/chi",
        "gofiber/fiber",
        "grpc/grpc-go",
    ],
    "java": [
        "spring-projects/spring-boot",
        "FasterXML/jackson-databind",
        "google/guava",
        "netty/netty",
        "apache/kafka",
        "elastic/elasticsearch",
    ],
    "ruby": [
        "rails/rails",
        "heartcombo/devise",
        "rspec/rspec-core",
        "rubocop/rubocop",
        "Homebrew/brew",
        "jekyll/jekyll",
    ],
    "rust": [
        "tokio-rs/tokio",
        "serde-rs/serde",
        "clap-rs/clap",
        "actix/actix-web",
        "rust-lang/rust",
        "hyperium/hyper",
    ],
}


async def fetch_json(
    session: aiohttp.ClientSession, url: str, params: dict = None
) -> dict | list | None:
    """Fetch from GitHub API with retry logic."""
    import time

    for attempt in range(3):
        try:
            async with session.get(url, headers=HEADERS, params=params) as resp:
                if resp.status == 403:
                    reset_at = int(
                        resp.headers.get("X-RateLimit-Reset", time.time() + 60)
                    )
                    wait = max(reset_at - time.time(), 1)
                    logger.debug(f"Rate limited, waiting {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception:
            await asyncio.sleep(2**attempt)
    return None


async def get_all_workflow_runs(
    session: aiohttp.ClientSession,
    repo: str,
    max_runs: int = 500,
) -> list[dict]:
    """Fetch all workflow runs for a repo (both failed and successful)."""
    all_runs = []
    for conclusion in ["failure", "success"]:
        page = 1
        while len(all_runs) < max_runs:
            data = await fetch_json(
                session,
                f"{GITHUB_API}/repos/{repo}/actions/runs",
                params={"conclusion": conclusion, "per_page": 100, "page": page},
            )
            if not data or not data.get("workflow_runs"):
                break
            runs = data["workflow_runs"]
            all_runs.extend(runs)
            if len(runs) < 100:
                break
            page += 1

    return all_runs


def build_failure_fix_chains(runs: list[dict]) -> list[tuple[dict, dict]]:
    """
    Given a list of workflow runs sorted by time, find consecutive failure→success pairs
    on the same workflow (branch: main/master).

    Returns list of (failed_run, fix_run) tuples.
    """
    # Group by workflow name and filter to main branch
    by_workflow: dict[str, list[dict]] = {}
    for run in runs:
        branch = run.get("head_branch", "")
        if branch not in ("main", "master", "develop"):
            continue
        wf = run.get("name", "")
        if wf not in by_workflow:
            by_workflow[wf] = []
        by_workflow[wf].append(run)

    chains = []
    for workflow_name, wf_runs in by_workflow.items():
        # Sort by run_number (ascending)
        wf_runs.sort(key=lambda r: r.get("run_number", 0))

        for i in range(len(wf_runs) - 1):
            current = wf_runs[i]
            next_run = wf_runs[i + 1]

            if (
                current.get("conclusion") == "failure"
                and next_run.get("conclusion") == "success"
            ):
                chains.append((current, next_run))

    return chains


async def get_commit_diff(
    session: aiohttp.ClientSession,
    repo: str,
    base_sha: str,
    head_sha: str,
    max_files: int = 15,
) -> str:
    """Get unified diff between two commits, capped at max_files."""
    data = await fetch_json(
        session,
        f"{GITHUB_API}/repos/{repo}/compare/{base_sha}...{head_sha}",
    )
    if not data:
        return ""

    files = data.get("files", [])[:max_files]
    parts = []
    for f in files:
        patch = f.get("patch", "")
        if patch:
            parts.append(f"--- a/{f['filename']}\n+++ b/{f['filename']}\n{patch}")

    return "\n".join(parts)


async def collect_repo_failure_chains(
    session: aiohttp.ClientSession,
    repo: str,
    language: str,
    output_dir: Path,
    env_mode: bool = False,
) -> int:
    """Collect failure→fix chains for a single repository."""
    repo_slug = repo.replace("/", "__")
    output_file = output_dir / f"{repo_slug}.jsonl"

    if output_file.exists():
        return 0

    all_runs = await get_all_workflow_runs(session, repo, max_runs=300)
    if not all_runs:
        return 0

    chains = build_failure_fix_chains(all_runs)
    if not chains:
        return 0

    pairs_written = 0
    with open(output_file, "w") as f:
        for failed_run, fix_run in chains[:30]:  # Cap at 30 chains per repo
            base_sha = failed_run.get("head_sha", "")
            fix_sha = fix_run.get("head_sha", "")

            if not base_sha or not fix_sha:
                continue

            diff = await get_commit_diff(session, repo, base_sha, fix_sha)
            if not diff:
                continue

            # In env_mode, only keep pairs where diff touches Dockerfile or YAML
            if env_mode and not any(
                kw in diff for kw in ["Dockerfile", ".yml", ".yaml", "apt-get", "FROM "]
            ):
                continue

            record = {
                "id": f"ci_history_{repo_slug}_{failed_run['id']}",
                "source": "ci_history",
                "repo": repo,
                "language": language,
                "ci_platform": "github_actions",
                "workflow_name": failed_run.get("name", ""),
                "failing_sha": base_sha,
                "fix_sha": fix_sha,
                "failed_run_id": failed_run["id"],
                "fix_run_id": fix_run["id"],
                "fix_diff": diff,
                "has_fix": True,
                "env_mode": env_mode,
            }
            f.write(json.dumps(record) + "\n")
            pairs_written += 1

    logger.info(f"  {repo}: {pairs_written} failure→fix chains")
    return pairs_written


async def main_async(workers: int, output_dir: Path, env_mode: bool):
    """Main async orchestrator."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten curated repos list with language labels
    all_repos: list[tuple[str, str]] = []
    for lang, repos in CURATED_REPOS.items():
        for repo in repos:
            all_repos.append((repo, lang))

    logger.info(f"Processing {len(all_repos)} curated repositories")

    semaphore = asyncio.Semaphore(workers)
    connector = aiohttp.TCPConnector(limit=workers * 2)

    async def process_with_sem(
        session: aiohttp.ClientSession, repo: str, lang: str
    ) -> int:
        async with semaphore:
            try:
                return await collect_repo_failure_chains(
                    session, repo, lang, output_dir, env_mode
                )
            except Exception as e:
                logger.debug(f"Error on {repo}: {e}")
                return 0

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_with_sem(session, r, lang) for r, lang in all_repos]
        results = await asyncio.gather(*tasks)
    total = sum(results)
    logger.info(f"Collection complete. Total failure→fix chains: {total}")


app = typer.Typer()


@app.command()
def main(
    workers: int = typer.Option(20, help="Concurrent workers"),
    output: Path = typer.Option(Path("data/raw/ci_history"), help="Output directory"),
    env_mode: bool = typer.Option(
        False, "--env-mode", help="Focus on env/Dockerfile failures"
    ),
):
    """Collect CI failure→fix chains from curated open source repositories."""
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set.")
        raise typer.Exit(1)
    asyncio.run(main_async(workers, output, env_mode))


if __name__ == "__main__":
    app()
