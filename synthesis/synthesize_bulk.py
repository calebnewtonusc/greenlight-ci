"""
GreenLight CI — Bulk Synthesis
Augments collected CI failure→fix pairs and generates synthetic Stream 5 pairs.
Also handles the final quality filtering and dataset preparation.

Usage:
  python synthesis/synthesize_bulk.py --concurrency 32
  python synthesis/synthesize_bulk.py --validate-only  # Only run quality filter
  python synthesis/synthesize_bulk.py --backend claude  # Use Claude API
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import typer  # noqa: E402
from loguru import logger  # noqa: E402

from synthesis.prompts import FAILURE_SYNTHESIS_SYSTEM_PROMPT  # noqa: E402
from core.failure_taxonomy import FailureSubClass  # noqa: E402

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URLS = os.environ.get("VLLM_URLS", "http://localhost:8001").split(",")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")

# Synthesis targets: how many pairs per sub-class
SYNTHESIS_TARGETS = {
    FailureSubClass.FLAKY_RACE_CONDITION: 3000,
    FailureSubClass.FLAKY_EXTERNAL_DEP: 4000,
    FailureSubClass.FLAKY_RESOURCE: 2000,
    FailureSubClass.FLAKY_TEST_ORDERING: 2000,
    FailureSubClass.DEP_DIRECT_BREAKING: 5000,
    FailureSubClass.DEP_TRANSITIVE_CONFLICT: 4000,
    FailureSubClass.DEP_BUILD_TOOL: 2000,
    FailureSubClass.DEP_LOCKFILE_STALE: 3000,
    FailureSubClass.ENV_BASE_IMAGE: 3000,
    FailureSubClass.ENV_RUNNER_CHANGE: 2000,
    FailureSubClass.ENV_MISSING_SYSTEM_DEP: 3000,
    FailureSubClass.ENV_SECRET_MISSING: 1000,
    FailureSubClass.LOGIC_TEST_EXPECTATION: 4000,
    FailureSubClass.LOGIC_API_CONTRACT: 4000,
    FailureSubClass.LOGIC_SCHEMA_MISMATCH: 2000,
    FailureSubClass.LOGIC_IMPORT_ERROR: 2000,
    # SECURITY_AUDIT
    FailureSubClass.SECURITY_VULN_FOUND: 400,
    FailureSubClass.SECURITY_LICENSE_VIOLATION: 200,
    FailureSubClass.SECURITY_SECRET_LEAK: 200,
    FailureSubClass.SECURITY_SBOM_MISMATCH: 100,
    # LINT_FORMATTING
    FailureSubClass.LINT_STYLE_VIOLATION: 400,
    FailureSubClass.LINT_TYPE_ERROR: 300,
    FailureSubClass.LINT_UNUSED_IMPORT: 200,
    FailureSubClass.LINT_COMPLEXITY: 100,
    # BUILD_COMPILATION
    FailureSubClass.BUILD_COMPILE_ERROR: 400,
    FailureSubClass.BUILD_MISSING_ARTIFACT: 200,
    FailureSubClass.BUILD_CACHE_INVALID: 150,
    FailureSubClass.BUILD_TOOL_INCOMPATIBLE: 250,
}

LANGUAGES = ["python", "javascript", "go", "java", "ruby", "rust"]
REPO_TYPES = [
    "web_framework",
    "data_library",
    "cli_tool",
    "database_orm",
    "authentication_service",
    "api_client",
    "testing_framework",
    "build_tool",
    "monitoring",
    "message_queue",
]


def make_synthesis_prompt(
    subclass: FailureSubClass, language: str, repo_type: str
) -> str:
    """Build a user prompt for synthesizing a specific failure scenario."""
    return (
        f"Generate a realistic CI failure scenario with these specifications:\n"
        f"- failure_subclass: {subclass.value}\n"
        f"- language: {language}\n"
        f"- repo_type: {repo_type}\n\n"
        f"The scenario must be specific and realistic — it should look like a real CI log "
        f"from a real open source project. Include actual package names, version numbers, "
        f"and error messages that would appear in this type of failure.\n\n"
        f"The fix_diff must be minimal — only the exact lines needed to fix the failure."
    )


async def synthesize_one(
    client: httpx.AsyncClient,
    subclass: FailureSubClass,
    language: str,
    repo_type: str,
    backend: str,
    vllm_url: str,
    aclient=None,
) -> dict | None:
    """Synthesize a single failure→fix pair."""
    prompt = make_synthesis_prompt(subclass, language, repo_type)

    messages = [
        {"role": "system", "content": FAILURE_SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        if backend == "vllm":
            resp = await client.post(
                f"{vllm_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                json={
                    "model": "Qwen/Qwen2.5-72B-Instruct",
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.8,  # Higher temp for diversity
                },
                timeout=90.0,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            # Claude API (async client in async context — reuse shared instance)
            msg = await aclient.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=FAILURE_SYNTHESIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()

        # Parse JSON — use find/rfind to handle nested braces correctly
        if text.startswith("{"):
            data = json.loads(text)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            data = json.loads(text[start : end + 1])

        # Validate required fields
        required = ["failure_class", "failure_subclass", "ci_log", "fix_diff"]
        if not all(k in data for k in required):
            return None

        # Add metadata
        data["id"] = f"synth_{subclass.value}_{language}_{random.randint(10000, 99999)}"
        data["source"] = "synthesized"
        data["has_fix"] = True
        data["verified_sandbox"] = False
        data["language"] = language
        data["repo_type"] = repo_type

        return data

    except Exception as e:
        logger.debug(f"Synthesis error ({subclass.value}, {language}): {e}")
        return None


def score_pair_quality(record: dict) -> float:
    """
    Heuristic quality scoring for synthesized pairs.
    Returns 0.0-1.0. Pairs below 0.6 are discarded.
    """
    score = 1.0
    ci_log = record.get("ci_log", "")
    fix_diff = record.get("fix_diff", "")

    # CI log quality checks
    if len(ci_log) < 100:
        score -= 0.4  # Too short to be realistic
    if "error" not in ci_log.lower() and "failed" not in ci_log.lower():
        score -= 0.3  # No error signal in log

    # Fix diff quality checks
    if not fix_diff or len(fix_diff) < 20:
        score -= 0.4  # Empty or trivial diff
    if "---" not in fix_diff or "+++" not in fix_diff:
        score -= 0.2  # Not a valid unified diff

    diff_lines = len(
        [
            line
            for line in fix_diff.split("\n")
            if line.startswith(("+", "-")) and not line.startswith(("---", "+++"))
        ]
    )
    if diff_lines > 100:
        score -= 0.3  # Suspiciously large diff for a CI fix
    if diff_lines == 0:
        score -= 0.5  # Empty diff

    return max(score, 0.0)


async def run_synthesis(
    output_dir: Path,
    backend: str,
    concurrency: int,
    pairs_per_subclass: int,
):
    """Run bulk synthesis for all subclasses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthesized_pairs.jsonl"

    semaphore = asyncio.Semaphore(concurrency)
    vllm_urls = VLLM_URLS if backend == "vllm" else [""]

    total_generated = 0
    total_passed = 0

    # Create the Anthropic async client once outside the synthesis loop
    aclient = None
    if backend == "claude":
        from anthropic import AsyncAnthropic

        aclient = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async with httpx.AsyncClient() as client:
        tasks = []
        for subclass in FailureSubClass:
            target = min(
                pairs_per_subclass, SYNTHESIS_TARGETS.get(subclass, pairs_per_subclass)
            )
            for _ in range(target):
                language = random.choice(LANGUAGES)
                repo_type = random.choice(REPO_TYPES)
                vllm_url = random.choice(vllm_urls)
                tasks.append((subclass, language, repo_type, vllm_url))

        random.shuffle(tasks)
        logger.info(
            f"Synthesizing {len(tasks)} pairs across {len(FailureSubClass)} sub-classes"
        )

        with open(output_file, "a") as f:

            async def synthesize_with_sem(sc, lang, rt, url):
                async with semaphore:
                    return await synthesize_one(
                        client, sc, lang, rt, backend, url, aclient
                    )

            batch_size = concurrency * 4
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                results = await asyncio.gather(
                    *[
                        synthesize_with_sem(sc, lang, rt, url)
                        for sc, lang, rt, url in batch
                    ]
                )
                for result in results:
                    if result is None:
                        continue
                    total_generated += 1
                    quality = score_pair_quality(result)
                    if quality >= 0.6:
                        result["quality_score"] = quality
                        f.write(json.dumps(result) + "\n")
                        total_passed += 1

                if i % (batch_size * 10) == 0:
                    logger.info(
                        f"  Progress: {i}/{len(tasks)} | "
                        f"Generated: {total_generated} | Passed: {total_passed}"
                    )

    logger.info(
        f"Synthesis complete: {total_passed}/{total_generated} pairs passed quality filter"
    )


def run_quality_filter(input_dir: Path, output_file: Path, min_quality: float):
    """
    Merge all classified pairs, apply quality filter, and produce final training dataset.
    """
    from datasketch import MinHash, MinHashLSH

    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for jsonl_file in input_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # Adapter: GitLab (and some Travis) records use "failure_log" key;
                    # normalize to "ci_log" so all downstream code uses a single field name.
                    if "ci_log" not in rec and "failure_log" in rec:
                        rec["ci_log"] = rec["failure_log"]
                    all_records.append(rec)
                except Exception:
                    pass

    logger.info(f"Loaded {len(all_records)} records for quality filtering")

    # MinHash deduplication
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    deduped = []

    for i, rec in enumerate(all_records):
        # Create fingerprint from log + diff content
        content = rec.get("ci_log", "") + rec.get("fix_diff", "")
        words = set(content.lower().split())

        mh = MinHash(num_perm=128)
        for w in words:
            mh.update(w.encode("utf8"))

        record_id = rec.get("id", str(i))
        try:
            result = lsh.query(mh)
            if not result:
                try:
                    lsh.insert(record_id, mh)
                    deduped.append(rec)
                except ValueError:
                    # Duplicate record_id already inserted — log and skip
                    logger.warning(
                        f"Duplicate record id skipped during LSH insert: {record_id!r}"
                    )
        except Exception:
            deduped.append(rec)

    logger.info(
        f"After deduplication: {len(deduped)} records (removed {len(all_records) - len(deduped)})"
    )

    # Apply quality filter
    passed = []
    for rec in deduped:
        conf = rec.get("classification_confidence", 0.5)
        fc = rec.get("failure_class", "UNKNOWN")
        has_fix = rec.get("has_fix", False)
        quality = rec.get("quality_score", score_pair_quality(rec))

        if conf >= min_quality and fc != "UNKNOWN" and has_fix and quality >= 0.6:
            rec["quality_score"] = quality
            passed.append(rec)

    # Shuffle for training
    random.shuffle(passed)

    with open(output_file, "w") as f:
        for rec in passed:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Final dataset: {len(passed)} pairs written to {output_file}")

    # Class distribution report
    from collections import Counter

    class_dist = Counter(r.get("failure_class", "UNKNOWN") for r in passed)
    logger.info(f"Class distribution: {dict(class_dist)}")


app = typer.Typer()


@app.command()
def main(
    output_dir: Path = typer.Option(Path("data/synthesized"), help="Output directory"),
    backend: str = typer.Option("claude", help="Backend: claude | vllm"),
    concurrency: int = typer.Option(32, help="Concurrent synthesis requests"),
    pairs_per_subclass: int = typer.Option(250, help="Target pairs per sub-class"),
    validate_only: bool = typer.Option(
        False, "--validate-only", help="Only run quality filter"
    ),
    min_quality: float = typer.Option(0.65, help="Minimum quality score"),
    input_dir: Path = typer.Option(
        Path("data/classified"), help="Input dir for validate-only mode"
    ),
    final_output: Path = typer.Option(
        Path("data/training/ci_repair_pairs.jsonl"), help="Final merged output"
    ),
):
    """Bulk synthesize and quality-filter CI failure→fix training pairs."""
    if validate_only:
        logger.info("Running quality filter only (--validate-only)")
        run_quality_filter(input_dir, final_output, min_quality)
    else:
        asyncio.run(run_synthesis(output_dir, backend, concurrency, pairs_per_subclass))
        logger.info("Running quality filter on all data...")
        run_quality_filter(
            Path("data/classified"),
            final_output,
            min_quality,
        )


if __name__ == "__main__":
    app()
