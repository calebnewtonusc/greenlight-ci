"""
GreenLight CI — Failure Classifier
Classifies raw CI failure logs into the 4-class, 12-subclass taxonomy
using a two-pass approach:
  1. Fast heuristic pre-classification (pattern matching)
  2. Deep LLM classification (Qwen2.5-72B via vLLM or Claude API)

Usage:
  python synthesis/failure_classifier.py --input data/raw/ --output data/classified/
  python synthesis/failure_classifier.py --input data/raw/github_actions/ --output data/classified/ --backend vllm
"""

import json
import os
from pathlib import Path

import httpx
import typer
from loguru import logger

from core.failure_taxonomy import (
    top_heuristic_class,
    get_fix_strategy,
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URL = os.environ.get("VLLM_URLS", "http://localhost:8001").split(",")[0]
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")


CLASSIFICATION_SYSTEM_PROMPT = """\
You are a CI failure classification expert. Given a CI log from a failing build,
you must classify the failure into exactly one of 8 primary classes and one of 28 sub-classes.

Primary classes:
- FLAKY: Non-deterministic test failures (race conditions, flaky external deps, resource limits, test ordering)
- DEP_DRIFT: Dependency version changes breaking the build (direct breaking change, transitive conflict, build tool, lockfile stale)
- ENV: Environment configuration failures (base image change, runner OS/tool change, missing system dep, missing secret)
- LOGIC: Code regression (test expectation drift, API contract break, schema mismatch, import error)
- SECURITY_AUDIT: Security scanner or audit failures (vulnerability found, license violation, secret detected, SBOM mismatch)
- LINT_FORMATTING: Linting and code formatting failures (style violation, type error, unused import, complexity threshold)
- BUILD_COMPILATION: Compilation or build system failures (compile error, missing artifact, cache invalid, build tool incompatible)
- UNKNOWN: Cannot determine the failure class from the log

Sub-classes:
1a: race_condition  1b: external_dependency  1c: resource_exhaustion  1d: test_ordering
2a: direct_dep_breaking  2b: transitive_conflict  2c: build_tool_version  2d: lockfile_stale
3a: base_image_update  3b: runner_os_tool  3c: missing_system_dep  3d: secret_env_missing
4a: test_expectation  4b: api_contract  4c: schema_mismatch  4d: import_error
5a: vulnerability_found  5b: license_violation  5c: secret_detected_in_code  5d: sbom_mismatch
6a: style_violation  6b: type_error  6c: unused_import_variable  6d: complexity_threshold
7a: compile_error  7b: missing_build_artifact  7c: build_cache_invalid  7d: build_tool_version_incompatible

Output ONLY valid JSON matching this schema:
{
  "failure_class": "<FLAKY|DEP_DRIFT|ENV|LOGIC|SECURITY_AUDIT|LINT_FORMATTING|BUILD_COMPILATION|UNKNOWN>",
  "failure_subclass": "<e.g. 2a_direct_dep_breaking>",
  "confidence": <0.0-1.0>,
  "root_cause": "<one sentence explaining the exact root cause>",
  "fix_strategy": "<one sentence describing the minimal fix>",
  "key_evidence": ["<evidence quote 1>", "<evidence quote 2>"]
}
"""


def classify_with_claude(log_text: str, heuristic_hint: str = "") -> dict | None:
    """Classify a CI log using Claude API (Anthropic)."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_msg = f"""CI Log (truncated to key section):
{log_text[:8000]}

Heuristic pre-classification hint: {heuristic_hint or "none"}

Classify this CI failure."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=CLASSIFICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = msg.content[0].text.strip()
        # Extract JSON from response
        if text.startswith("{"):
            return json.loads(text)
        # Try to find JSON block
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception as e:
        logger.debug(f"Claude classification error: {e}")
        return None


def classify_with_vllm(log_text: str, heuristic_hint: str = "") -> dict | None:
    """Classify a CI log using local vLLM server (Qwen2.5-72B)."""
    user_msg = f"""CI Log:
{log_text[:8000]}

Heuristic hint: {heuristic_hint or "none"}

Classify this CI failure."""

    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = httpx.post(
            f"{VLLM_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            json={
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("{"):
            return json.loads(text)
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception as e:
        logger.debug(f"vLLM classification error: {e}")
        return None


def classify_failure(
    record: dict,
    backend: str = "claude",
    heuristic_only: bool = False,
) -> dict:
    """
    Classify a single CI failure record.
    Returns the record enriched with classification fields.
    """
    log_text = record.get("ci_log", "")
    if not log_text:
        record["failure_class"] = "UNKNOWN"
        record["failure_subclass"] = None
        record["classification_confidence"] = 0.0
        record["classification_source"] = "no_log"
        return record

    # Step 1: Heuristic pre-classification
    h_class, h_subclass, h_confidence = top_heuristic_class(log_text)
    heuristic_hint = f"{h_class.value} ({h_subclass.value if h_subclass else 'unknown'}) — confidence {h_confidence:.2f}"

    # If heuristic confidence is very high, skip LLM
    if heuristic_only or h_confidence >= 0.90:
        record["failure_class"] = h_class.value
        record["failure_subclass"] = h_subclass.value if h_subclass else None
        record["classification_confidence"] = h_confidence
        record["classification_source"] = "heuristic"
        record["fix_strategy"] = get_fix_strategy(h_subclass) if h_subclass else ""
        record["root_cause"] = ""
        record["key_evidence"] = []
        return record

    # Step 2: Deep LLM classification
    if backend == "vllm":
        result = classify_with_vllm(log_text, heuristic_hint)
    else:
        result = classify_with_claude(log_text, heuristic_hint)

    if result and result.get("failure_class"):
        record["failure_class"] = result["failure_class"]
        record["failure_subclass"] = result.get("failure_subclass")
        record["classification_confidence"] = result.get("confidence", 0.7)
        record["classification_source"] = backend
        record["root_cause"] = result.get("root_cause", "")
        record["fix_strategy"] = result.get("fix_strategy", "")
        record["key_evidence"] = result.get("key_evidence", [])
    else:
        # Fall back to heuristic
        record["failure_class"] = h_class.value
        record["failure_subclass"] = h_subclass.value if h_subclass else None
        record["classification_confidence"] = h_confidence * 0.8  # Penalize fallback
        record["classification_source"] = "heuristic_fallback"
        record["fix_strategy"] = get_fix_strategy(h_subclass) if h_subclass else ""

    return record


def process_directory(
    input_dir: Path,
    output_dir: Path,
    backend: str,
    min_confidence: float,
    batch_size: int,
):
    """Process all JSONL files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.rglob("*.jsonl"))
    logger.info(f"Found {len(input_files)} JSONL files to classify")

    total_processed = 0
    total_passed = 0
    total_skipped = 0

    for input_file in input_files:
        output_file = output_dir / input_file.name

        records = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        classified = []
        for i, record in enumerate(records):
            if i % 100 == 0:
                logger.info(f"  {input_file.name}: {i}/{len(records)} classified")

            classified_record = classify_failure(record, backend=backend)
            total_processed += 1

            # Quality filter
            conf = classified_record.get("classification_confidence", 0.0)
            fc = classified_record.get("failure_class", "UNKNOWN")
            has_fix = classified_record.get("has_fix", False)

            if conf >= min_confidence and fc != "UNKNOWN" and has_fix:
                classified.append(classified_record)
                total_passed += 1
            else:
                total_skipped += 1

        with open(output_file, "w") as f:
            for rec in classified:
                f.write(json.dumps(rec) + "\n")

        logger.info(
            f"  {input_file.name}: {len(classified)}/{len(records)} passed quality filter"
        )

    logger.info(
        f"Classification complete: {total_passed} passed, {total_skipped} skipped "
        f"(total {total_processed})"
    )


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("data/raw"), help="Input directory with raw JSONL files"
    ),
    output_dir: Path = typer.Option(
        Path("data/classified"), help="Output directory for classified pairs"
    ),
    backend: str = typer.Option("claude", help="LLM backend: claude | vllm"),
    min_confidence: float = typer.Option(
        0.65, help="Minimum classification confidence to keep pair"
    ),
    batch_size: int = typer.Option(50, help="Batch size for LLM calls"),
):
    """Classify CI failure logs using heuristics + LLM deep analysis."""
    if backend == "claude" and not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set for claude backend")
        raise typer.Exit(1)
    if backend == "vllm" and VLLM_URL == "http://localhost:8001":
        logger.error(
            "VLLM_URL is still the default (http://localhost:8001); "
            "set VLLM_URLS env var to a running vLLM server"
        )
        raise typer.Exit(1)

    logger.info(f"Classifying failures: {input_dir} → {output_dir} (backend={backend})")
    process_directory(input_dir, output_dir, backend, min_confidence, batch_size)


if __name__ == "__main__":
    app()
