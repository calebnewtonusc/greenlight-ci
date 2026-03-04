# GreenLight CI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-7B-Coder](https://img.shields.io/badge/base_model-Qwen2.5--7B--Coder-purple.svg)](https://huggingface.co/Qwen)
[![GPUs: 18x A6000](https://img.shields.io/badge/training-18×_A6000-red.svg)](https://www.nvidia.com)
[![Status: Training](https://img.shields.io/badge/status-training-orange.svg)]()

> **"CI never stays broken."**

GreenLight CI is the first trained specialist model for CI repair. It doesn't just detect failing CI — it analyzes failure taxonomy, classifies root cause (flaky test, dependency drift, environment misconfiguration, or real logic regression), and opens minimal-risk PRs that fix the specific failure with surgical precision.

This repository contains the complete dataset pipeline, training infrastructure, and deployment stack for GreenLight CI — from raw CI log ingestion to a production-ready repair agent.

---

## Why GreenLight CI Is Different

Every existing CI fixing tool takes one of two approaches: (1) re-run the job and hope it's flaky, or (2) call GPT-4 with the log and ask for advice. Neither approach actually *fixes* CI. Neither understands failure taxonomy. Neither knows when a `pip install` version pin 3 months ago is why your tests fail today.

| Capability | Dependabot | RenovateBot | Copilot Autofix | GPT-4 + log | **GreenLight CI** |
|---|---|---|---|---|---|
| Understands failure taxonomy | — | — | — | partial | **Flaky / dep-drift / env / logic — 4-class** |
| Generates minimal diffs | dep only | dep only | — | inconsistent | **Surgery-grade patches, zero bloat** |
| Trains on CI history | — | — | — | — | **500k+ failure→fix pairs** |
| Rerun stability verification | — | — | — | — | **Must fix same failure twice** |
| Handles env/config failures | — | — | — | — | **Dockerfile, action YAML, runner config** |
| Language-agnostic | — | — | partial | partial | **Python, Go, Node, Java, Rust, Ruby** |
| Free verifiable reward | — | — | — | — | **CI green = ground truth** |

---

## Architecture

```
                      ┌────────────────────────────────────────────────────┐
 CI Failure Webhook ──►│              GreenLight CI System                  │
  (GitHub Actions,     │                                                    │
   CircleCI, Jenkins)  │  ┌─────────────────────────────────────────────┐  │
                       │  │         GreenLight Model                    │  │
                       │  │  (Qwen2.5-7B-Coder + LoRA rank 64          │  │
                       │  │   SFT → GRPO → DPO, ZeRO-3 trained)        │  │
                       │  └──────────────────┬──────────────────────────┘  │
                       │                     │                              │
                       │         ┌───────────▼───────────┐                 │
                       │         │   Failure Classifier   │                 │
                       │         │  FLAKY / DEP / ENV /  │                 │
                       │         │  LOGIC (4-class)       │                 │
                       │         └───────────┬───────────┘                 │
                       │                     │                              │
                       │    ┌────────────────┼────────────────┐            │
                       │    ▼                ▼                ▼            │
                       │ ┌──────┐      ┌─────────┐      ┌─────────┐       │
                       │ │Flaky │      │Dep Drift│      │  Env /  │       │
                       │ │Retry │      │ Patcher │      │ Config  │       │
                       │ │Logic │      │         │      │ Patcher │       │
                       │ └──────┘      └─────────┘      └─────────┘       │
                       │         ┌──────────────────────────┐              │
                       │         │     Logic Fix Agent       │              │
                       │         │  (real test/code repair)  │              │
                       │         └──────────────────────────┘              │
                       │                     │                              │
                       │         ┌───────────▼───────────┐                 │
                       │         │   Patch Validator      │                 │
                       │         │  (sandbox CI rerun)    │                 │
                       │         └───────────┬───────────┘                 │
                       │                     │                              │
                       │         ┌───────────▼───────────┐                 │
                       │         │    PR Generator        │                 │
                       │         │  (minimal diff, desc)  │                 │
                       │         └───────────────────────┘                 │
                       └────────────────────────────────────────────────────┘
```

**Training data sources (5 streams, 500k+ failure→fix pairs):**
- Stream 1: GitHub Actions public logs + the commits that fixed them (40%)
- Stream 2: Open source repo CI history — failure patterns across languages (25%)
- Stream 3: Dependency drift analysis — `requirements.txt`/`package.json` pin failures (15%)
- Stream 4: Environment configuration failures — Dockerfile, runner YAML, OS mismatches (12%)
- Stream 5: Synthesized multi-language failure→fix pairs via LLM augmentation (8%)

---

## What Makes GreenLight CI Different

### The Real Problem With CI

CI failures fall into exactly four categories, and each requires a completely different fix strategy:

1. **Flaky tests** — The test itself is non-deterministic. Fix: quarantine, retry, or fix the test isolation.
2. **Dependency drift** — A package version changed upstream. Fix: pin the dep, update lockfile, or constrain the range.
3. **Environment failure** — The runner, Docker image, OS, or tool version changed. Fix: pin the image, update action versions, add apt deps.
4. **Logic regression** — Code was merged that actually broke something. Fix: patch the code or the test expectation.

Every existing tool either retries (which only helps flaky), bumps deps (which only helps drift), or asks a general LLM to "look at the log." GreenLight CI is trained to classify first, then apply the correct fix strategy.

### The Training Signal

CI green/red is the best free verifiable reward signal in software engineering. Like DeepSeek-R1's math reward (is the answer correct?), CI gives us ground truth without human raters. GreenLight CI uses this to train Stage 2 (GRPO): generate a patch, apply it in a sandbox, rerun CI, reward = pass rate improvement.

The critical addition: **rerun stability**. A fix must pass CI twice on the same failure to earn full reward. This prevents "restart the runner" patches that only mask flaky failures.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/calebnewtonusc/greenlight-ci
cd greenlight-ci
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys

# Verify environment
bash scripts/check_env.sh

# Run full pipeline (data → training → eval), ~40 hours on 18× A6000
bash scripts/run_all.sh

# Or step by step:
python pipeline.py --stage discovery    # ~8h, collect CI failure/fix pairs
python pipeline.py --stage synthesis    # ~14h, augment + classify pairs
python pipeline.py --stage train        # ~14h, SFT (8h) + GRPO (4h) + DPO (2h)
python pipeline.py --stage eval         # ~4h, CIBench evaluation
```

### Run on a Live Failing CI Job

```bash
# Point GreenLight at a specific failing run
python agents/ci_repair_agent.py \
  --repo "owner/repo" \
  --run-id 12345678 \
  --github-token $GITHUB_TOKEN

# Or watch for failures via webhook (production mode)
python agents/ci_repair_agent.py --watch --repo "owner/repo"

# REST API
curl -X POST http://localhost:8000/repair \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/repo", "run_id": 12345678}'
```

---

## CIBench Results (Target v1)

| Metric | Target | Naive Retry | Dependabot |
|--------|--------|-------------|------------|
| Overall fix rate (200 failures) | >70% | ~15% | ~20% |
| Flaky test classification accuracy | >90% | — | — |
| Dep drift fix rate | >85% | — | ~60% |
| Env failure fix rate | >75% | — | — |
| Logic regression fix rate | >55% | — | — |
| Rerun stability (fix holds 2nd run) | >95% | ~40% | ~90% |
| Avg diff lines (lower = better) | <15 | N/A | ~30 |

---

## Hardware Requirements

| Stage | Config | Estimated Time |
|-------|--------|----------------|
| Discovery | Any machine, async I/O | 6-10 hours |
| Synthesis (vLLM) | 4× A6000 per instance | 12-16 hours |
| SFT Training | 18× A6000, ZeRO-3 | 8 hours |
| GRPO Training | 18× A6000, ZeRO-3 + sandbox workers | 4 hours |
| DPO Training | 18× A6000 | 2 hours |
| Inference | 1× A6000 or A100 | <2s per repair |

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full technical architecture, failure taxonomy, training pipeline
- [DATA_SOURCES.md](DATA_SOURCES.md) — 5 training data streams with collection methodology
- [MODEL_CARD.md](MODEL_CARD.md) — Model specification, capabilities, limitations
- [ROADMAP.md](ROADMAP.md) — v1 through v3 feature roadmap
- [SETUP_GPU.md](SETUP_GPU.md) — 18× A6000 cluster configuration guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute failure/fix pairs and evaluations

---

## License

MIT License — open training pipeline, open weights (post v1 release).
