# GreenLight CI Architecture

## Core Insight

CI failures have structure. Every broken build falls into one of four failure classes, and each class has a canonical fix strategy. The problem with existing tools is they treat all CI failures as the same problem — "something is broken, ask an LLM."

GreenLight CI's thesis: **the failure class determines the fix. Train a model to classify failures and generate class-specific patches.**

The analogy is medical diagnosis. A doctor who sees a symptom doesn't immediately prescribe medication — they diagnose the disease first, then apply the appropriate treatment. Prescribing the same treatment for every symptom would fail most of the time. That's exactly what general-purpose LLM-based CI fixers do.

---

## 4-Phase Product Vision

### Phase 1 — CLASSIFY + FIX (v1, Q3 2026)
Core repair agent. 4-class failure classifier + per-class patch generator. CIBench evaluation across 200 historical failures. First open-weights CI repair model.

### Phase 2 — PREDICT (v1.5, Q4 2026)
Proactive failure prediction. Before merging, GreenLight CI flags PRs likely to break CI. Understands dep compatibility matrices, test isolation patterns, and which code paths correlate with historical failures.

### Phase 3 — HARDEN (v2, Q1 2027)
Quarantine and flaky test elimination. Detects historically flaky tests across the codebase. Generates retries with proper isolation, mocks, and determinism fixes. Reduces the flaky test ratio org-wide over time.

### Phase 4 — AUTONOMOUS (v3, 2027)
Fully autonomous CI maintenance. GreenLight CI monitors all repos in an org, fixes issues before they block PRs, and learns from every failure/fix pair it generates.

---

## Failure Taxonomy (The Core Innovation)

GreenLight CI defines 4 primary failure classes and 12 sub-classes:

### Class 1: FLAKY
The test itself is non-deterministic. The code is correct but the test is unreliable.

- **1a: Race condition** — Async tests, threading, process ordering
- **1b: External dependency** — HTTP calls, database state, time-based assertions
- **1c: Resource exhaustion** — OOM, disk space, port conflicts in CI
- **1d: Test ordering dependency** — Tests that rely on shared global state

**Fix strategy**: Quarantine label + retry policy OR isolation fix OR mock injection

### Class 2: DEP_DRIFT
A dependency version changed in a way that breaks the build.

- **2a: Direct dep breaking change** — Upstream released a breaking API change
- **2b: Transitive dep conflict** — Two deps require incompatible versions of a shared dep
- **2c: Build tool version drift** — Node/Python/Go runtime version changed in CI
- **2d: Lockfile stale** — `package-lock.json` or `poetry.lock` out of sync

**Fix strategy**: Pin the breaking dep, update lockfile, or constrain the version range

### Class 3: ENV
The execution environment changed in a way that breaks the build.

- **3a: Base image update** — Docker base image changed behavior
- **3b: Runner OS/tool change** — Ubuntu 22→24, GitHub Actions runner update
- **3c: Missing system dep** — `apt-get install` needed, not in Dockerfile
- **3d: Secret/env var missing** — Required env var not configured in CI

**Fix strategy**: Pin image version, add apt deps, fix YAML config

### Class 4: LOGIC
The code change itself introduced a regression.

- **4a: Test expectation drift** — Code behavior changed correctly but test wasn't updated
- **4b: API contract break** — Internal API signature changed
- **4c: Schema mismatch** — DB migration, config schema change
- **4d: Import error** — Module restructuring broke imports

**Fix strategy**: Patch the code, update the test, or both

---

## Training Pipeline

### Data Streams → Pairs

```
Stream 1: GitHub Actions Logs (40%)
  Failed run log → commit that fixed it → (log, failure class, fix diff) triplet
  Source: GitHub API /repos/{owner}/{repo}/actions/runs with conclusion=failure

Stream 2: Open Source CI History (25%)
  Public repo git log + CI run history → (failure_log, fix_commit_diff) pairs
  Languages: Python (35%), JavaScript (25%), Go (15%), Java (10%), Ruby (10%), Rust (5%)

Stream 3: Dependency Drift Corpus (15%)
  Dependabot/Renovate PRs with before/after CI outcomes
  Includes: requirements.txt, package.json, go.mod, Gemfile, Cargo.toml

Stream 4: Environment Failure Corpus (12%)
  Dockerfile + action YAML failures + fixes
  Source: public DockerHub repos, GitHub Actions marketplace PRs

Stream 5: Synthesized Pairs (8%)
  LLM-generated failure scenarios across all 12 sub-classes, validated by sandbox execution
```

### Stage 1: Supervised Fine-Tuning (SFT)

- **Base model**: Qwen2.5-7B-Coder-Instruct
- **Data**: ~500k curated (failure_log, failure_class, fix_diff) triplets
- **LoRA**: rank 64, alpha 128, all attention + MLP layers
- **Context**: 16,384 tokens (full CI log + relevant code context)
- **Duration**: ~8 hours on 18× A6000 with DeepSpeed ZeRO-3
- **Goal**: Model learns failure taxonomy and per-class patch generation

**Training format**:
```
<|im_start|>system
You are GreenLight CI, an expert CI repair specialist. You analyze CI failure logs,
classify the root cause, and generate minimal surgical patches to restore CI green.

Always reason through: failure class → root cause → minimal fix → validation plan.

Output format:
<classify>[FLAKY|DEP_DRIFT|ENV|LOGIC] — [sub-class]</classify>
<reason>[Root cause analysis]</reason>
<fix>[Minimal patch in unified diff format]</fix>
<validate>[How to verify the fix holds on rerun]</validate>
<|im_end|>
<|im_start|>user
Repository: {repo_name} ({language})
Failure: {test_name} in {workflow_name}

CI Log:
{truncated_log}

Relevant code context:
{code_context}
<|im_end|>
<|im_start|>assistant
...
```

### Stage 2: CI-Verified Reinforcement Learning (GRPO)

The core technical innovation of GreenLight CI.

**Reward signal**: Does the generated patch make CI green? Does it hold on the second rerun?

```
reward = 0.0 (patch fails to apply)
reward = 0.0 (CI still red after patch)
reward = 0.5 (CI green on first run)
reward = 1.0 (CI green on first AND second run — stable fix)
reward += 0.1 (diff_lines < 10 — minimality bonus)
reward -= 0.1 (diff_lines > 50 — verbosity penalty)
```

**Execution harness**: Docker sandbox per patch application. Each sandbox:
1. Checks out the failing commit
2. Applies the generated diff
3. Runs the specific failing test suite
4. Reports pass/fail
5. Reruns (second stability check)

**GRPO config**: 8 generations per prompt (8 candidate patches), reward best

### Stage 3: DPO (Preference Optimization)

**Preference pairs**: Minimal correct fix vs. over-engineered fix vs. wrong-class fix

Examples:
- Flaky HTTP test: mock injection (chosen) vs. adding retry (rejected — masks the issue)
- Dep pin: constrain to `>=2.1,<3.0` (chosen) vs. pin to exact `==2.1.3` (rejected — fragile)
- Logic fix: update test expectation (chosen) vs. disable the test (rejected)

**Duration**: ~2 hours on 18× A6000

---

## Multi-Agent Orchestration

```
CI Repair Agent (Orchestrator)
├── Log Fetcher
│   ├── github_actions_fetcher()
│   ├── circleci_fetcher()
│   └── jenkins_fetcher()
├── Failure Classifier
│   ├── classify_failure_class()  ── GreenLight model call
│   ├── identify_sub_class()
│   └── extract_relevant_context()
├── Patch Generator (dispatches by class)
│   ├── FlakePatcher
│   │   ├── detect_nondeterminism()
│   │   ├── generate_mock_injection()
│   │   └── generate_retry_policy()
│   ├── DepDriftPatcher
│   │   ├── parse_dep_graph()
│   │   ├── find_compatible_version()
│   │   └── update_lockfile()
│   ├── EnvPatcher
│   │   ├── pin_base_image()
│   │   ├── add_system_dep()
│   │   └── fix_action_yaml()
│   └── LogicPatcher
│       ├── identify_failing_assertion()
│       ├── trace_code_change()
│       └── generate_code_fix()
├── Patch Validator
│   ├── apply_patch_sandbox()
│   ├── run_failing_suite()
│   └── stability_rerun()
└── PR Generator
    ├── format_minimal_diff()
    ├── write_pr_description()
    └── open_pr_via_api()
```

---

## CIBench

Custom benchmark evaluating GreenLight CI on 200 historical CI failures, stratified by:
- Failure class (50 per class)
- Language (Python, Go, Node, Java, mixed)
- Complexity (simple single-file fix vs. multi-file vs. config-only)

Metrics:
- **Fix rate**: % of failures successfully resolved (CI green)
- **Stability rate**: % of fixes that hold on second rerun
- **Classification accuracy**: % of failures correctly classified
- **Minimality**: Average diff size of successful fixes (lines changed)
- **False positive rate**: % of "fixes" that break previously passing tests

---

## Model Specification

| Property | Value |
|----------|-------|
| Base model | Qwen2.5-7B-Coder-Instruct |
| Total parameters | 7.6B |
| Trainable (LoRA) | ~168M (2.2%) |
| Context length | 16,384 tokens |
| LoRA rank | 64 |
| Output format | `<classify>` + `<reason>` + `<fix>` + `<validate>` |
| Quantization (inference) | 4-bit GPTQ |
| Serving | vLLM with PagedAttention |
| Latency (repair suggestion) | <3s |

---

## Deployment Stack

```
vLLM (model serving, 1× A6000 per worker)
    ↕
FastAPI (webhook receiver + REST API)
    ↕
Sandbox Executor (Docker pool, 8 parallel sandboxes)
    ↕
Redis (failure queue, repair session state)
    ↕
GitHub API (PR creation, CI status polling)
    ↕
Nginx (reverse proxy, SSL)
    ↕
Docker Compose (one-command deployment)
```
