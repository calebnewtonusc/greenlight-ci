"""
CIBench — GreenLight CI Evaluation Suite
200 historical CI failures stratified across all 4 failure classes and 6 languages.

Usage:
  python evaluation/ci_bench.py --model checkpoints/greenlight-final
  python evaluation/ci_bench.py --model checkpoints/greenlight-final --failure-class DEP_DRIFT
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.failure_taxonomy import FailureClass, FailureSubClass


@dataclass
class CIBenchCase:
    """A single CIBench evaluation case."""
    id: str
    repo: str
    language: str
    failure_class: FailureClass
    failure_subclass: FailureSubClass
    ci_log: str
    correct_fix_diff: str  # Ground truth fix
    acceptable_fix_classes: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CIBenchResult:
    """Result for a single CIBench case."""
    case_id: str
    failure_class: FailureClass
    predicted_class: str
    correct_classification: bool
    generated_fix: str
    fix_applies_cleanly: bool
    fix_is_minimal: bool  # diff <= 1.5x ground truth size
    sandbox_passed: Optional[bool] = None  # None if sandbox not available
    latency_seconds: float = 0.0


@dataclass
class CIBenchSummary:
    """Aggregated CIBench evaluation results."""
    total_cases: int
    classification_accuracy: float
    fix_application_rate: float
    fix_minimality_rate: float
    sandbox_pass_rate: Optional[float]
    per_class_results: dict[str, dict]
    per_language_results: dict[str, dict]
    avg_latency: float


# ── Embedded evaluation cases (representative subset) ──────────────────────────

CIBENCH_CASES: list[CIBenchCase] = [
    CIBenchCase(
        id="dep_drift_001_psycopg3",
        repo="django/django",
        language="python",
        failure_class=FailureClass.DEP_DRIFT,
        failure_subclass=FailureSubClass.DEP_DIRECT_BREAKING,
        ci_log="""
FAILED tests/backends/test_postgresql.py::PostgreSQLTestCase::test_connect
ImportError: cannot import name 'connect' from 'psycopg' (psycopg 3.0 changed API)

Traceback (most recent call last):
  File "django/db/backends/postgresql/base.py", line 12, in <module>
    from psycopg import connect, OperationalError
ImportError: cannot import name 'connect' from 'psycopg'
""",
        correct_fix_diff="""--- a/requirements.txt
+++ b/requirements.txt
@@ -5,7 +5,7 @@
-psycopg>=3.0
+psycopg>=2.9,<3.0
""",
        notes="psycopg3 changed the public API; pin to 2.x branch",
    ),
    CIBenchCase(
        id="flaky_001_async_race",
        repo="encode/httpx",
        language="python",
        failure_class=FailureClass.FLAKY,
        failure_subclass=FailureSubClass.FLAKY_RACE_CONDITION,
        ci_log="""
FAILED tests/test_async.py::test_concurrent_requests - RuntimeError: Event loop is closed
RuntimeError: Event loop is closed
During handling of the above exception, another exception occurred:
  File "tests/conftest.py", line 45, in event_loop
    loop = asyncio.get_event_loop()
""",
        correct_fix_diff="""--- a/tests/conftest.py
+++ b/tests/conftest.py
@@ -42,6 +42,9 @@ import pytest
 @pytest.fixture
-def event_loop():
-    loop = asyncio.get_event_loop()
+def event_loop():
+    loop = asyncio.new_event_loop()
+    asyncio.set_event_loop(loop)
     yield loop
+    loop.close()
""",
        notes="Event loop reuse across tests; each test needs a fresh loop",
    ),
    CIBenchCase(
        id="env_001_ubuntu_pkg",
        repo="psf/requests",
        language="python",
        failure_class=FailureClass.ENV,
        failure_subclass=FailureSubClass.ENV_MISSING_SYSTEM_DEP,
        ci_log="""
Run apt-get install -y libssl-dev
E: Package 'libssl-dev' has no installation candidate
Error: Process completed with exit code 100.
""",
        correct_fix_diff="""--- a/.github/workflows/tests.yml
+++ b/.github/workflows/tests.yml
@@ -18,7 +18,7 @@
       - name: Install system deps
-        run: sudo apt-get install -y libssl-dev
+        run: sudo apt-get install -y libssl-dev openssl
""",
        notes="Ubuntu 24.04 renamed some ssl packages; openssl provides libssl-dev equivalent",
    ),
    CIBenchCase(
        id="logic_001_assertion",
        repo="pallets/flask",
        language="python",
        failure_class=FailureClass.LOGIC,
        failure_subclass=FailureSubClass.LOGIC_TEST_EXPECTATION,
        ci_log="""
FAILED tests/test_basic.py::test_404_response_code
AssertionError: assert 404 == 200
  - Expected 200 (route existed before refactor)
  - Got 404 (route was moved to /api/v2/users in latest PR)
""",
        correct_fix_diff="""--- a/tests/test_basic.py
+++ b/tests/test_basic.py
@@ -34,7 +34,7 @@
 def test_404_response_code(client):
-    response = client.get('/users')
+    response = client.get('/api/v2/users')
     assert response.status_code == 200
""",
        notes="Route was moved; test expectation not updated in the same PR",
    ),
]

# In production, CIBENCH_CASES is loaded from data/ci_bench/ (200 cases)
# The above is a representative embedded subset for unit testing


def load_cases_from_dir(cases_dir: Path) -> list[CIBenchCase]:
    """Load CIBench cases from JSON files in a directory."""
    global _ALL_CASES
    cases = list(CIBENCH_CASES)  # Start with embedded cases
    seen_ids = {c.id for c in cases}
    for f in cases_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            case = CIBenchCase(**data)
            if case.id not in seen_ids:
                cases.append(case)
                seen_ids.add(case.id)
        except Exception as e:
            logger.debug(f"Failed to load case {f}: {e}")
    _ALL_CASES = cases
    return cases


# Module-level cache for all cases (used by summarize_results for language lookup)
_ALL_CASES: list[CIBenchCase] = list(CIBENCH_CASES)


def load_model(model_path: str):
    """Load GreenLight CI model for evaluation."""
    base_model_name = "Qwen/Qwen2.5-7B-Coder-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if Path(model_path).exists():
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info(f"Loaded LoRA adapter from {model_path}")
    else:
        model = base_model
        logger.warning(f"No adapter found at {model_path}, using base model")

    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, case: CIBenchCase) -> str:
    """Run GreenLight CI inference on a single case."""
    from synthesis.prompts import GREENLIGHT_SYSTEM_PROMPT

    prompt = (
        f"<|im_start|>system\n{GREENLIGHT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Repository: {case.repo} ({case.language})\n"
        f"CI Log:\n{case.ci_log[:6000]}\n\n"
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
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated


def evaluate_result(case: CIBenchCase, generated: str, latency: float) -> CIBenchResult:
    """Evaluate a generated response against the ground truth."""
    # Extract classification
    classify_match = re.search(r"<classify>(.*?)</classify>", generated, re.DOTALL)
    predicted_class = "UNKNOWN"
    if classify_match:
        predicted_class = classify_match.group(1).strip().split("—")[0].strip().split()[0]

    correct_classification = predicted_class == case.failure_class.value

    # Extract fix
    fix_match = re.search(r"<fix>(.*?)</fix>", generated, re.DOTALL)
    generated_fix = fix_match.group(1).strip() if fix_match else ""

    # Check if fix applies cleanly (heuristic: valid unified diff format)
    fix_applies = (
        "---" in generated_fix
        and "+++" in generated_fix
        and len(generated_fix) > 20
    )

    # Minimality check: generated diff should be <=1.5x the ground truth diff size
    gt_lines = len([
        l for l in case.correct_fix_diff.split("\n")
        if l.startswith(("+", "-")) and not l.startswith(("---", "+++"))
    ])
    gen_lines = len([
        l for l in generated_fix.split("\n")
        if l.startswith(("+", "-")) and not l.startswith(("---", "+++"))
    ])
    is_minimal = gen_lines <= max(gt_lines * 1.5, gt_lines + 5) if gt_lines > 0 else gen_lines <= 15

    return CIBenchResult(
        case_id=case.id,
        failure_class=case.failure_class,
        predicted_class=predicted_class,
        correct_classification=correct_classification,
        generated_fix=generated_fix,
        fix_applies_cleanly=fix_applies,
        fix_is_minimal=is_minimal,
        latency_seconds=latency,
    )


def summarize_results(results: list[CIBenchResult]) -> CIBenchSummary:
    """Aggregate CIBench results into a summary."""
    from collections import defaultdict

    total = len(results)
    correct_class = sum(1 for r in results if r.correct_classification)
    applies = sum(1 for r in results if r.fix_applies_cleanly)
    minimal = sum(1 for r in results if r.fix_is_minimal)
    latencies = [r.latency_seconds for r in results]

    # Per-class breakdown
    by_class = defaultdict(list)
    for r in results:
        by_class[r.failure_class.value].append(r)

    per_class = {}
    for cls, cls_results in by_class.items():
        n = len(cls_results)
        per_class[cls] = {
            "total": n,
            "classification_accuracy": sum(1 for r in cls_results if r.correct_classification) / n,
            "fix_application_rate": sum(1 for r in cls_results if r.fix_applies_cleanly) / n,
            "fix_minimality_rate": sum(1 for r in cls_results if r.fix_is_minimal) / n,
        }

    # Per-language breakdown
    by_lang: dict = defaultdict(list)
    for r in results:
        # Retrieve language from the original case via case_id lookup
        lang = "unknown"
        for case in _ALL_CASES:
            if case.id == r.case_id:
                lang = case.language
                break
        by_lang[lang].append(r)

    per_language = {}
    for lang, lang_results in by_lang.items():
        n = len(lang_results)
        per_language[lang] = {
            "total": n,
            "classification_accuracy": sum(1 for r in lang_results if r.correct_classification) / n,
            "fix_application_rate": sum(1 for r in lang_results if r.fix_applies_cleanly) / n,
        }

    return CIBenchSummary(
        total_cases=total,
        classification_accuracy=correct_class / total if total else 0,
        fix_application_rate=applies / total if total else 0,
        fix_minimality_rate=minimal / total if total else 0,
        sandbox_pass_rate=None,
        per_class_results=per_class,
        per_language_results=per_language,
        avg_latency=sum(latencies) / len(latencies) if latencies else 0,
    )


app = typer.Typer()


@app.command()
def main(
    model_path: str = typer.Option("./checkpoints/greenlight-final", help="Path to model/adapter"),
    cases_dir: Path = typer.Option(Path("data/ci_bench"), help="Directory with CIBench case JSON files"),
    failure_class: str = typer.Option(None, help="Only evaluate specific class: FLAKY|DEP_DRIFT|ENV|LOGIC"),
    output_json: Path = typer.Option(Path("results/ci_bench_results.json"), help="Results output file"),
    max_cases: int = typer.Option(200, help="Maximum cases to evaluate"),
):
    """Run CIBench evaluation on GreenLight CI model."""
    logger.info(f"Loading CIBench cases from embedded set + {cases_dir}")
    all_cases = load_cases_from_dir(cases_dir)

    if failure_class:
        all_cases = [c for c in all_cases if c.failure_class.value == failure_class]

    cases = all_cases[:max_cases]
    logger.info(f"Evaluating on {len(cases)} cases")

    model, tokenizer = load_model(model_path)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"[{i+1}/{len(cases)}] {case.id} ({case.failure_class.value})")
        start = time.time()
        generated = run_inference(model, tokenizer, case)
        latency = time.time() - start
        result = evaluate_result(case, generated, latency)
        results.append(result)
        logger.info(
            f"  Class correct: {result.correct_classification} | "
            f"Fix applies: {result.fix_applies_cleanly} | "
            f"Minimal: {result.fix_is_minimal} | "
            f"Latency: {latency:.1f}s"
        )

    summary = summarize_results(results)

    logger.info("\n" + "="*60)
    logger.info("CIBench Results")
    logger.info("="*60)
    logger.info(f"Total cases: {summary.total_cases}")
    logger.info(f"Classification accuracy: {summary.classification_accuracy:.1%}")
    logger.info(f"Fix application rate:    {summary.fix_application_rate:.1%}")
    logger.info(f"Fix minimality rate:     {summary.fix_minimality_rate:.1%}")
    logger.info(f"Avg latency:             {summary.avg_latency:.1f}s")
    logger.info("\nPer-class breakdown:")
    for cls, stats in summary.per_class_results.items():
        logger.info(
            f"  {cls}: class_acc={stats['classification_accuracy']:.1%} "
            f"fix_apply={stats['fix_application_rate']:.1%}"
        )

    # Save results
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "summary": {
            "total_cases": summary.total_cases,
            "classification_accuracy": summary.classification_accuracy,
            "fix_application_rate": summary.fix_application_rate,
            "fix_minimality_rate": summary.fix_minimality_rate,
            "avg_latency": summary.avg_latency,
            "per_class": summary.per_class_results,
        },
        "cases": [
            {
                "id": r.case_id,
                "failure_class": r.failure_class.value,
                "predicted_class": r.predicted_class,
                "correct_classification": r.correct_classification,
                "fix_applies": r.fix_applies_cleanly,
                "fix_minimal": r.fix_is_minimal,
                "latency": r.latency_seconds,
            }
            for r in results
        ],
    }
    output_json.write_text(json.dumps(output_data, indent=2))
    logger.info(f"\nResults saved to {output_json}")


if __name__ == "__main__":
    app()
