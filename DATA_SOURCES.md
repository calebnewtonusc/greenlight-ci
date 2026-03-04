# GreenLight CI — Data Sources

## Overview

GreenLight CI trains on CI failure→fix pairs — the same raw signal that DevOps engineers process manually every day. Every time CI breaks and a human commits a fix, that's a training example. We collect 500,000+ such pairs across 5 streams.

**Total target**: 500,000+ (failure_log, failure_class, fix_diff) triplets

---

## Stream 1: GitHub Actions Public Logs (40% — ~200k pairs)

**What**: Failed workflow runs from public repositories paired with the fix commits.

**How**:
1. GitHub API `/repos/{owner}/{repo}/actions/runs` — query `conclusion=failure`
2. For each failed run, fetch the full log via `/actions/runs/{run_id}/logs`
3. Walk repo commit history to find the next commit that restores CI green on the same workflow
4. Extract the diff between the failing commit and the fix commit
5. Label the failure class using `failure_classifier.py`

**Languages covered**: Python, JavaScript/TypeScript, Go, Java, Ruby, Rust, PHP, C#

**Sources**:
- Top 10,000 GitHub repositories by stars (GitHub API `q=stars:>1000`)
- Open source projects with high CI activity (>100 runs/month)
- Well-maintained projects with clear commit messages (easier supervision signal)

**Collection script**: `discovery/github_ci_logs.py`

**Estimated pairs**: 200,000 (filtering ~40% of raw pairs for quality)

---

## Stream 2: Open Source CI History (25% — ~125k pairs)

**What**: Longitudinal CI run history from major open source projects, covering complete failure→fix lifecycles.

**How**:
1. Select 2,000 well-maintained open source repos across languages
2. Fetch full git history + corresponding CI run outcomes
3. Align git commits to CI run outcomes (commit SHA → run conclusion)
4. Identify failure→fix commit chains (consecutive commits where CI goes red→green)
5. Extract structured failure contexts including stack traces, test names, and error codes

**Key projects targeted**:
- Python: `django`, `flask`, `fastapi`, `pandas`, `scikit-learn`, `numpy`
- JavaScript: `react`, `vue`, `express`, `webpack`, `jest`, `eslint`
- Go: `kubernetes`, `docker`, `terraform`, `gin`, `cobra`
- Java: `spring-boot`, `jackson`, `guava`, `netty`
- Ruby: `rails`, `devise`, `rspec`, `rubocop`
- Rust: `tokio`, `serde`, `clap`, `actix-web`

**Failure categories emphasized**: Logic regressions (most complex, needs more examples)

**Collection script**: `discovery/fetch_failure_patterns.py`

**Estimated pairs**: 125,000

---

## Stream 3: Dependency Drift Corpus (15% — ~75k pairs)

**What**: Dependabot and Renovate Bot PRs that broke CI, paired with the resolution commits.

**How**:
1. Search GitHub for Dependabot/Renovate PRs that triggered CI failures
2. Query: `author:dependabot author:renovate is:pr is:merged`
3. For each dependency update PR: fetch CI outcome, before/after lockfile, and any follow-up fix commits
4. Extract `requirements.txt`, `package.json`, `go.mod`, `Gemfile`, `Cargo.toml` diffs

**Failure sub-classes**:
- Breaking API changes in major version bumps (e.g., `boto3 1.x → 2.x`)
- Transitive dependency conflicts (requires resolution via `pip install --constraint`)
- Lockfile staleness (`package-lock.json` regeneration failures)
- Build tool version incompatibility (Python 3.9 → 3.12 syntax changes)

**Augmentation**: Cross-reference with `libraries.io` release notes to label *why* each version change broke CI.

**Collection script**: `discovery/github_ci_logs.py` (dep-drift mode)

**Estimated pairs**: 75,000

---

## Stream 4: Environment Failure Corpus (12% — ~60k pairs)

**What**: CI failures caused by environment changes — Dockerfile updates, runner version changes, missing system packages, broken action YAML.

**How**:
1. GitHub search for commits containing `Dockerfile`, `.github/workflows/*.yml` changes that preceded CI green restoration
2. Docker Hub public image changelog parsing (base image tag changes)
3. GitHub Actions Marketplace — collect action version updates that caused failures
4. CI runner changelog cross-reference (ubuntu-22.04 → ubuntu-24.04 migration failures)

**Failure sub-classes targeted**:
- `FROM ubuntu:22.04 → FROM ubuntu:24.04` breaking apt package availability
- `actions/setup-python@v4 → v5` behavior changes
- Missing `apt-get install` for native extensions (`libpq-dev`, `libssl-dev`)
- Docker layer caching invalidation causing unexpected dep upgrades
- Environment variable missing from CI secrets

**Collection script**: `discovery/fetch_failure_patterns.py` (env-mode)

**Estimated pairs**: 60,000

---

## Stream 5: Synthesized Multi-Language Pairs (8% — ~40k pairs)

**What**: LLM-synthesized failure→fix pairs covering rare failure sub-classes and underrepresented languages.

**How**:
1. For each of 12 failure sub-classes, generate synthetic CI logs and code contexts
2. Use Qwen2.5-72B to generate realistic failure scenarios + correct fixes
3. Validate in Docker sandbox: apply generated fix, run synthetic test suite, confirm CI green
4. Discard any pair where the sandbox doesn't confirm the fix

**Coverage goals**:
- Rust, C#, PHP — underrepresented in Streams 1-4
- Complex multi-file logic regressions
- Edge cases: intermittent network failures, GPU-dependent tests, platform-specific failures

**Synthesis script**: `synthesis/synthesize_bulk.py`

**Estimated pairs**: 40,000 (after sandbox validation filtering)

---

## Data Quality Pipeline

All pairs pass through a 5-step quality filter:

1. **Log completeness check**: CI log must contain at least the failing test name and error message
2. **Diff sanity check**: Fix diff must be <500 lines and apply cleanly to the failing commit
3. **Class verification**: Failure class label verified by a second model pass (ensemble agreement)
4. **Duplicate removal**: MinHash deduplication at 85% Jaccard similarity threshold
5. **Sandbox validation** (random 10% sample): Apply fix, rerun test, confirm green

**Expected retention rate**: ~65% of raw pairs pass quality filter

---

## Data Schema

Each training pair is stored as JSONL with the following schema:

```json
{
  "id": "github_actions_django_a1b2c3d4",
  "source": "github_actions",
  "repo": "django/django",
  "language": "python",
  "ci_platform": "github_actions",
  "workflow_name": "Tests",
  "failure_class": "DEP_DRIFT",
  "failure_subclass": "2a_direct_dep_breaking_change",
  "failing_tests": ["test_database.TestDatabaseBackend.test_connect"],
  "ci_log": "...(truncated to 8192 chars)...",
  "code_context": "...(relevant source file snippets)...",
  "fix_diff": "--- a/requirements.txt\n+++ b/requirements.txt\n...",
  "fix_explanation": "psycopg3 2.0 changed the connection API...",
  "before_sha": "a1b2c3d4",
  "fix_sha": "e5f6g7h8",
  "verified_sandbox": true,
  "diff_lines": 3
}
```
