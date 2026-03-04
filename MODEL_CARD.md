# GreenLight CI — Model Card

## Model Overview

| Property | Value |
|----------|-------|
| **Model name** | GreenLight CI v1 |
| **Base model** | Qwen/Qwen2.5-7B-Coder-Instruct |
| **Total parameters** | 7.6B |
| **Trainable parameters (LoRA)** | ~168M (2.2%) |
| **LoRA rank** | 64 |
| **LoRA alpha** | 128 |
| **Context length** | 16,384 tokens |
| **Training stages** | SFT → GRPO → DPO |
| **Training hardware** | 18× NVIDIA A6000 48GB |
| **Training duration** | ~14 hours total |
| **License** | Apache 2.0 |
| **HuggingFace** | `calebnewtonusc/greenlight-ci-v1` (post-release) |

---

## Intended Use

GreenLight CI is designed to:

1. **Classify** CI failure logs into one of 4 root cause categories (FLAKY, DEP_DRIFT, ENV, LOGIC)
2. **Generate** minimal surgical patches that resolve the specific failure
3. **Validate** patches by predicting whether they will pass CI on rerun

**Intended users**: DevOps engineers, platform teams, CI/CD system integrators

**Intended integration**: GitHub Actions, CircleCI, Jenkins webhook systems; automated PR generation workflows

---

## Capabilities

### What GreenLight CI Does Well

- **Failure classification**: Correctly identifies the root cause class in >90% of cases (CIBench target)
- **Dependency drift repair**: Identifies the breaking version, finds a compatible pin or update strategy
- **Environment failure repair**: Pinning Docker base images, adding missing apt packages, fixing action YAML
- **Minimal diffs**: Generates patches averaging <15 lines for common failure classes
- **Multi-language**: Python, JavaScript, TypeScript, Go, Java, Ruby, Rust

### What GreenLight CI Does Not Do

- **Logic regression repair for complex business logic**: Deep algorithmic bugs require domain knowledge GreenLight CI does not have
- **Security vulnerability patching**: Use SealPatch for CVE remediation (separate specialist)
- **Test generation from scratch**: GreenLight CI patches existing tests; it does not write test suites
- **Cross-repo cascade repairs**: Does not automatically propagate fixes to downstream repositories (v2 feature)

---

## Training Data

- ~500,000 (failure_log, failure_class, fix_diff) triplets
- Sources: GitHub Actions logs, open source CI history, Dependabot PRs, environment failure corpus, synthesized pairs
- Languages: Python 35%, JavaScript/TypeScript 25%, Go 15%, Java 10%, Ruby 10%, Rust 5%
- Failure class distribution: FLAKY 30%, DEP_DRIFT 35%, ENV 20%, LOGIC 15%

---

## Evaluation

### CIBench v1 (200 historical failures)

| Failure Class | Fix Rate | Classification Acc | Stability |
|---------------|----------|-------------------|-----------|
| FLAKY | target >65% | target >92% | target >95% |
| DEP_DRIFT | target >85% | target >95% | target >98% |
| ENV | target >75% | target >90% | target >97% |
| LOGIC | target >55% | target >85% | target >90% |
| **Overall** | **target >70%** | **target >90%** | **target >95%** |

Baseline comparison: naive retry (15% fix rate), Dependabot alone (20% fix rate).

---

## Limitations and Biases

- **Training data bias**: Overrepresents Python and JavaScript ecosystems. Repair quality for Java, Rust, and C# is lower.
- **Log truncation**: CI logs longer than 16,384 tokens are truncated. Very long build logs may lose critical context.
- **Novel failure modes**: Failure classes not well-represented in training data (e.g., GPU-specific failures, exotic build systems) have lower fix rates.
- **Not a security tool**: GreenLight CI does not consider security implications of proposed patches. Patches should be reviewed before merge in security-sensitive codebases.
- **Diff minimality training**: Model strongly prefers small diffs; occasionally refuses to make necessary large-scale refactors even when that is the correct fix.

---

## Ethical Considerations

- All training data sourced from public repositories under permissive licenses (MIT, Apache, BSD)
- No private repository data used in training
- Generated patches should be reviewed by a human before automatic merge in production
- The model should not be used to auto-merge patches to `main` without human review in v1

---

## Citation

```bibtex
@inproceedings{newton2026greenlight,
  title     = {GreenLight CI: Training a CI Repair Specialist with Failure Taxonomy and Execution-Verified Reinforcement Learning},
  author    = {Newton, Caleb and others},
  booktitle = {ICSE 2027},
  year      = {2027}
}
```
