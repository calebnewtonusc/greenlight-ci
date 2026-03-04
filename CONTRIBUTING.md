# Contributing to GreenLight CI

GreenLight CI improves with more failure→fix pairs, better evaluation scenarios, and community-contributed language support. Here's how to contribute.

---

## Contributing Failure→Fix Pairs

The highest-value contribution is verified (failure_log, fix_diff) pairs. These go directly into training data.

### Format

Create a JSONL file with this schema and open a PR to `data/community/`:

```json
{
  "source": "community",
  "repo": "owner/repo",
  "language": "python",
  "ci_platform": "github_actions",
  "failure_class": "DEP_DRIFT",
  "failure_subclass": "2a_direct_dep_breaking_change",
  "ci_log": "(paste relevant CI log section)",
  "code_context": "(paste relevant source file snippet)",
  "fix_diff": "(paste unified diff of the fix)",
  "fix_explanation": "(explain why this fixed it)",
  "verified_sandbox": false
}
```

### Requirements

- The failure must have actually occurred (not synthesized)
- The fix diff must have actually restored CI green
- CI log must include the error message and test name
- `fix_explanation` must explain the root cause, not just what changed

### Validation

All community pairs are run through `synthesis/failure_classifier.py` for automatic quality scoring. Pairs scoring below 0.7 confidence will be flagged for manual review before inclusion.

---

## Contributing CIBench Test Cases

CIBench is our evaluation suite. More test cases = more rigorous evaluation.

Add test cases in `evaluation/ci_bench.py`:

```python
CIBENCH_CASES.append(CIBenchCase(
    id="community_django_dep_drift_001",
    repo="django/django",
    language="python",
    failure_class=FailureClass.DEP_DRIFT,
    ci_log="""...""",
    correct_fix_diff="""...""",
    acceptable_fix_classes=["DEP_DRIFT"],
    notes="psycopg3 2.0 breaking API change"
))
```

Test cases must include:
- The actual failing CI log
- The actual correct fix (or a description of what constitutes a correct fix)
- The failure class label

---

## Contributing Language Support

GreenLight CI currently focuses on Python, JavaScript, Go, Java, Ruby, and Rust. Adding support for a new language means:

1. **Discovery**: Add language-specific CI log parsing in `discovery/github_ci_logs.py`
2. **Failure patterns**: Add language-specific failure patterns to `core/failure_taxonomy.py`
3. **Test examples**: Add 10+ example failure→fix pairs for the language

Start by opening a GitHub Issue with the tag `[language-support]` describing the language and your proposed approach.

---

## Code Style

- Python 3.11+ with type hints on all public functions
- Docstrings required on all classes and public methods (Google style)
- `loguru` for logging (no `print` statements in library code)
- `typer` for CLI interfaces
- `ruff` for linting (run `ruff check .` before PRs)
- Tests in `tests/` using `pytest`

---

## Development Setup

```bash
git clone https://github.com/calebnewtonusc/greenlight-ci
cd greenlight-ci
pip install -r requirements.txt
pip install -e ".[dev]"  # Dev extras: pytest, ruff, mypy

# Run tests
pytest tests/ -v

# Lint
ruff check .
mypy . --ignore-missing-imports
```

---

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Add tests for any new functionality
3. Run `ruff check .` and `pytest tests/`
4. Open a PR with a description of the change and its motivation
5. Link to any GitHub Issues the PR resolves

For training data contributions (new failure→fix pairs), no tests are required — just the JSONL file with complete schema and a PR description explaining the failure type.
