# GreenLight CI Roadmap

## v1 — CLASSIFY + FIX (Q3 2026)

Core repair agent. First trained specialist model for CI failure classification and repair.

**Goals**:
- >70% fix rate on CIBench (200 historical failures, 4 classes)
- >90% failure classification accuracy
- Rerun stability >95% (fixes hold on second run)
- Works with GitHub Actions, CircleCI, Jenkins
- Open weights on HuggingFace

**Features**:
- 3-stage trained model (SFT → CI-RL → DPO)
- 4-class failure classifier (flaky, dep-drift, env, logic)
- 12 sub-class taxonomy with specialized patch strategies
- Docker sandbox execution harness for CI validation
- GitHub Actions webhook integration
- PR auto-generation with minimal diffs
- CIBench evaluation suite (200 failures across languages)
- REST API for CI platform integrations

**Paper Target**: ICSE 2027 — "GreenLight CI: Training a CI Repair Specialist with Failure Taxonomy and Execution-Verified Reinforcement Learning"

---

## v1.5 — PREDICT (Q4 2026)

Proactive failure prediction before merges break CI.

**Goals**:
- >60% recall on "will-break-CI" prediction (pre-merge analysis)
- <10% false positive rate (low alert fatigue)
- Integration with PR review workflow

**Features**:
- Pre-merge CI risk scoring — analyze diff before merge
- Dep compatibility matrix — check if new dep versions conflict before CI runs
- Test coverage gap detection — flag code paths without test coverage
- Historical failure correlation — "this file changed → CI broke 3 of last 5 times"
- GitHub PR check integration (status check, not just comment)
- Slack/Teams notification for high-risk merges

---

## v2 — HARDEN (Q1 2027)

Quarantine and flaky test elimination at the org level.

**Goals**:
- Detect all flaky tests in a codebase within 7 days
- Reduce org-wide flaky test ratio by >80% in 30 days
- Automated quarantine + root cause PR for each flaky test

**Features**:
- Flaky test detection via multi-run statistical analysis
- Root cause classification per flaky test (race condition / external dep / resource / ordering)
- Automated isolation fix generation (mocks, retry decorators, test ordering fixes)
- Quarantine registry — track which tests are quarantined and why
- Flaky test dashboard (org-wide health view)
- CI run time reduction tracking (fixing flakyness speeds CI)

---

## v3 — AUTONOMOUS (2027)

Fully autonomous CI maintenance for entire organizations.

**Goals**:
- Zero CI failures lingering >15 minutes in monitored orgs
- GreenLight CI as the de facto CI-ops layer across cloud CI platforms
- Self-improvement: each repair teaches the model a new pattern

**Features**:
- Org-wide CI monitoring (all repos, all branches)
- Priority queue: P0 main branch failures fixed immediately
- Continual fine-tuning: each verified fix pair added to training corpus weekly
- Multi-repo dependency graph — upstream dep change → downstream CI fix cascade
- Cost analytics (CI minutes saved per fix, flaky tests eliminated)
- Enterprise API with SSO, audit log, and on-premise deployment
- IDE extension (VS Code, JetBrains) — fix CI without leaving editor

---

## Research Paper Pipeline

| Paper | Target Venue | Core Contribution |
|-------|-------------|-------------------|
| GreenLight v1 | ICSE 2027 | CI failure taxonomy + CI-RL training signal |
| Failure Prediction | FSE 2027 | Pre-merge CI risk scoring from diff analysis |
| Flaky Test Taxonomy | ASE 2027 | Statistical flaky classification at scale |
| GreenLight vs. General LLMs | ICSE 2028 | Specialist training beats scaffolded GPT-4 on CI repair |
