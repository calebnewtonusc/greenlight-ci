"""
GreenLight CI — Failure Taxonomy
Defines the 8-class, 28-subclass CI failure taxonomy and associated detection heuristics.

This is the central ontology of GreenLight CI. Every classifier, training example,
and patch generator is built around this taxonomy.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class FailureClass(str, Enum):
    """Primary CI failure classification (8 classes)."""

    FLAKY = "FLAKY"
    DEP_DRIFT = "DEP_DRIFT"
    ENV = "ENV"
    LOGIC = "LOGIC"
    SECURITY_AUDIT = "SECURITY_AUDIT"  # Security scanner / audit failures
    LINT_FORMATTING = "LINT_FORMATTING"  # Linting and code formatting failures
    BUILD_COMPILATION = "BUILD_COMPILATION"  # Compilation / build system failures
    UNKNOWN = "UNKNOWN"


class FailureSubClass(str, Enum):
    """28-class CI failure sub-classification."""

    # FLAKY
    FLAKY_RACE_CONDITION = "1a_race_condition"
    FLAKY_EXTERNAL_DEP = "1b_external_dependency"
    FLAKY_RESOURCE = "1c_resource_exhaustion"
    FLAKY_TEST_ORDERING = "1d_test_ordering"

    # DEP_DRIFT
    DEP_DIRECT_BREAKING = "2a_direct_dep_breaking_change"
    DEP_TRANSITIVE_CONFLICT = "2b_transitive_dep_conflict"
    DEP_BUILD_TOOL = "2c_build_tool_version"
    DEP_LOCKFILE_STALE = "2d_lockfile_stale"

    # ENV
    ENV_BASE_IMAGE = "3a_base_image_update"
    ENV_RUNNER_CHANGE = "3b_runner_os_tool_change"
    ENV_MISSING_SYSTEM_DEP = "3c_missing_system_dep"
    ENV_SECRET_MISSING = "3d_secret_env_var_missing"

    # LOGIC
    LOGIC_TEST_EXPECTATION = "4a_test_expectation_drift"
    LOGIC_API_CONTRACT = "4b_api_contract_break"
    LOGIC_SCHEMA_MISMATCH = "4c_schema_mismatch"
    LOGIC_IMPORT_ERROR = "4d_import_error"

    # SECURITY_AUDIT
    SECURITY_VULN_FOUND = "5a_vulnerability_found"
    SECURITY_LICENSE_VIOLATION = "5b_license_violation"
    SECURITY_SECRET_LEAK = "5c_secret_detected_in_code"
    SECURITY_SBOM_MISMATCH = "5d_sbom_mismatch"

    # LINT_FORMATTING
    LINT_STYLE_VIOLATION = "6a_style_violation"
    LINT_TYPE_ERROR = "6b_type_error"
    LINT_UNUSED_IMPORT = "6c_unused_import_variable"
    LINT_COMPLEXITY = "6d_complexity_threshold"

    # BUILD_COMPILATION
    BUILD_COMPILE_ERROR = "7a_compile_error"
    BUILD_MISSING_ARTIFACT = "7b_missing_build_artifact"
    BUILD_CACHE_INVALID = "7c_build_cache_invalid"
    BUILD_TOOL_INCOMPATIBLE = "7d_build_tool_version_incompatible"


@dataclass
class FailureSignal:
    """A detected signal in a CI log that suggests a failure class."""

    failure_class: FailureClass
    failure_subclass: FailureSubClass
    confidence: float
    evidence: str
    # Multi-label: a failure can belong to multiple classes simultaneously
    additional_classes: list["FailureClass"] = field(default_factory=list)


@dataclass
class CIFailure:
    """Structured representation of a CI failure."""

    repo: str
    language: str
    ci_platform: str  # github_actions | circleci | jenkins
    workflow_name: str
    failing_tests: list[str]
    raw_log: str
    failure_class: FailureClass = FailureClass.UNKNOWN
    failure_subclass: Optional[FailureSubClass] = None
    confidence: float = 0.0
    signals: list[FailureSignal] = field(default_factory=list)
    code_context: str = ""


# ── Heuristic pattern matching for pre-classification ─────────────────────────
# These patterns are used to provide fast initial classification before the
# GreenLight model runs the full deep analysis. They achieve ~70% accuracy
# and serve as features for the model's context.

FLAKY_PATTERNS = [
    (
        re.compile(r"(timeout|timed out|connection reset|connection refused)", re.I),
        FailureSubClass.FLAKY_EXTERNAL_DEP,
        0.7,
    ),
    (
        re.compile(r"(race condition|data race|concurrent|deadlock)", re.I),
        FailureSubClass.FLAKY_RACE_CONDITION,
        0.8,
    ),
    (
        re.compile(r"(out of memory|oom|disk full|no space left)", re.I),
        FailureSubClass.FLAKY_RESOURCE,
        0.8,
    ),
    (
        re.compile(r"(port \d+ already in use|address already in use)", re.I),
        FailureSubClass.FLAKY_RESOURCE,
        0.7,
    ),
    (
        re.compile(r"(test.*order|ordering.*test|setUp.*failed|fixtures)", re.I),
        FailureSubClass.FLAKY_TEST_ORDERING,
        0.5,
    ),
]

DEP_PATTERNS = [
    (
        re.compile(r"(ImportError|ModuleNotFoundError|cannot import name)", re.I),
        FailureSubClass.DEP_DIRECT_BREAKING,
        0.6,
    ),
    (
        re.compile(
            r"(conflicting dependencies|dependency conflict|incompatible)", re.I
        ),
        FailureSubClass.DEP_TRANSITIVE_CONFLICT,
        0.8,
    ),
    (
        re.compile(r"(npm ERR!|yarn error|package-lock.json)", re.I),
        FailureSubClass.DEP_LOCKFILE_STALE,
        0.6,
    ),
    (
        re.compile(
            r"(poetry.lock|Pipfile.lock|Gemfile.lock|go.sum).*out of sync", re.I
        ),
        FailureSubClass.DEP_LOCKFILE_STALE,
        0.85,
    ),
    (
        re.compile(r"(python [\d.]+ is not supported|requires python >=)", re.I),
        FailureSubClass.DEP_BUILD_TOOL,
        0.8,
    ),
    (
        re.compile(
            r"(DeprecationWarning.*removed in|API removed|deprecated.*will be removed)",
            re.I,
        ),
        FailureSubClass.DEP_DIRECT_BREAKING,
        0.75,
    ),
]

ENV_PATTERNS = [
    (
        re.compile(r"(unable to find image|image not found|manifest unknown)", re.I),
        FailureSubClass.ENV_BASE_IMAGE,
        0.9,
    ),
    (
        re.compile(
            r"(apt-get.*No such file|Package.*has no installation candidate)", re.I
        ),
        FailureSubClass.ENV_MISSING_SYSTEM_DEP,
        0.85,
    ),
    (
        re.compile(r"(E: Unable to locate package|dpkg: error)", re.I),
        FailureSubClass.ENV_MISSING_SYSTEM_DEP,
        0.85,
    ),
    (
        re.compile(r"(Error: \$\{secrets\.|secret.*not found|env.*undefined)", re.I),
        FailureSubClass.ENV_SECRET_MISSING,
        0.8,
    ),
    (
        re.compile(r"(ubuntu-[\d]+.*deprecated|runner.*not available)", re.I),
        FailureSubClass.ENV_RUNNER_CHANGE,
        0.8,
    ),
    (
        re.compile(r"(setup-python.*Node.js|actions/checkout.*version)", re.I),
        FailureSubClass.ENV_RUNNER_CHANGE,
        0.6,
    ),
]

LOGIC_PATTERNS = [
    (
        re.compile(r"(AssertionError|assert.*!=|Expected.*Got)", re.I),
        FailureSubClass.LOGIC_TEST_EXPECTATION,
        0.6,
    ),
    (
        re.compile(
            r"(TypeError.*argument|TypeError.*keyword|unexpected keyword argument)",
            re.I,
        ),
        FailureSubClass.LOGIC_API_CONTRACT,
        0.75,
    ),
    (
        re.compile(
            r"(OperationalError|ProgrammingError|column.*does not exist|table.*doesn't exist)",
            re.I,
        ),
        FailureSubClass.LOGIC_SCHEMA_MISMATCH,
        0.8,
    ),
    (
        re.compile(r"(from.*import.*cannot.*import|no module named)", re.I),
        FailureSubClass.LOGIC_IMPORT_ERROR,
        0.7,
    ),
    (
        re.compile(r"(AttributeError.*has no attribute|object has no attribute)", re.I),
        FailureSubClass.LOGIC_API_CONTRACT,
        0.7,
    ),
]

SECURITY_PATTERNS = [
    (
        re.compile(
            r"(vulnerability|CVE-\d{4}-\d+|GHSA-|audit\s+found|critical\s+severity)",
            re.I,
        ),
        FailureSubClass.SECURITY_VULN_FOUND,
        0.85,
    ),
    (
        re.compile(
            r"(license\s+violation|incompatible\s+license|GPL\s+conflict|AGPL)", re.I
        ),
        FailureSubClass.SECURITY_LICENSE_VIOLATION,
        0.80,
    ),
    (
        re.compile(
            r"(secret.*detected|token.*exposed|credential.*leak|trufflehog|gitleaks)",
            re.I,
        ),
        FailureSubClass.SECURITY_SECRET_LEAK,
        0.90,
    ),
    (
        re.compile(r"(sbom.*mismatch|sbom.*failed|cyclonedx.*error)", re.I),
        FailureSubClass.SECURITY_SBOM_MISMATCH,
        0.80,
    ),
]

LINT_PATTERNS = [
    (
        re.compile(r"(flake8|pylint|eslint|rubocop|golint|checkstyle).*error", re.I),
        FailureSubClass.LINT_STYLE_VIOLATION,
        0.85,
    ),
    (
        re.compile(r"(mypy|pytype|flow.*error|tsc.*error|type.*error)", re.I),
        FailureSubClass.LINT_TYPE_ERROR,
        0.85,
    ),
    (
        re.compile(r"(F401|unused\s+import|unused\s+variable|W0611)", re.I),
        FailureSubClass.LINT_UNUSED_IMPORT,
        0.80,
    ),
    (
        re.compile(
            r"(cyclomatic\s+complexity|too\s+complex|C901|function.*too\s+long)", re.I
        ),
        FailureSubClass.LINT_COMPLEXITY,
        0.75,
    ),
    (
        re.compile(r"(black.*would\s+reformat|prettier.*failed|gofmt.*diff)", re.I),
        FailureSubClass.LINT_STYLE_VIOLATION,
        0.90,
    ),
]

BUILD_PATTERNS = [
    (
        re.compile(
            r"(compilation.*failed|cannot\s+compile|undefined\s+reference|linker.*error)",
            re.I,
        ),
        FailureSubClass.BUILD_COMPILE_ERROR,
        0.90,
    ),
    (
        re.compile(
            r"(could\s+not\s+find\s+artifact|missing.*dist|no\s+such\s+file.*build)",
            re.I,
        ),
        FailureSubClass.BUILD_MISSING_ARTIFACT,
        0.80,
    ),
    (
        re.compile(
            r"(cache\s+miss|invalid\s+cache|stale\s+cache|cache\s+expired)", re.I
        ),
        FailureSubClass.BUILD_CACHE_INVALID,
        0.70,
    ),
    (
        re.compile(
            r"(gradle\s+version|maven\s+version|cmake.*version|make.*version).*incompatible",
            re.I,
        ),
        FailureSubClass.BUILD_TOOL_INCOMPATIBLE,
        0.80,
    ),
    (
        re.compile(r"(SyntaxError|IndentationError|invalid\s+syntax).*build", re.I),
        FailureSubClass.BUILD_COMPILE_ERROR,
        0.75,
    ),
]

ALL_PATTERNS = [
    (FLAKY_PATTERNS, FailureClass.FLAKY),
    (DEP_PATTERNS, FailureClass.DEP_DRIFT),
    (ENV_PATTERNS, FailureClass.ENV),
    (LOGIC_PATTERNS, FailureClass.LOGIC),
    (SECURITY_PATTERNS, FailureClass.SECURITY_AUDIT),
    (LINT_PATTERNS, FailureClass.LINT_FORMATTING),
    (BUILD_PATTERNS, FailureClass.BUILD_COMPILATION),
]


def heuristic_classify(log: str) -> list[FailureSignal]:
    """
    Fast heuristic pre-classification based on regex pattern matching.
    Returns a list of signals sorted by confidence (descending).
    Used as features for the GreenLight model's deep classification.
    """
    signals: list[FailureSignal] = []

    for pattern_list, failure_class in ALL_PATTERNS:
        for pattern, subclass, base_confidence in pattern_list:
            matches = pattern.findall(log)
            if matches:
                # Boost confidence if pattern matches multiple times
                confidence = min(base_confidence + 0.05 * (len(matches) - 1), 0.95)
                signals.append(
                    FailureSignal(
                        failure_class=failure_class,
                        failure_subclass=subclass,
                        confidence=confidence,
                        evidence=f"Pattern '{pattern.pattern}' matched {len(matches)} times: {matches[:3]}",
                    )
                )

    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals


def top_heuristic_class(log: str) -> tuple[FailureClass, FailureSubClass | None, float]:
    """
    Returns the highest-confidence heuristic classification.
    Returns UNKNOWN if no pattern matches with confidence > 0.5.
    """
    signals = heuristic_classify(log)
    if not signals or signals[0].confidence < 0.5:
        return FailureClass.UNKNOWN, None, 0.0
    top = signals[0]
    return top.failure_class, top.failure_subclass, top.confidence


# ── Fix strategy guidance for each failure class ───────────────────────────────

FIX_STRATEGY = {
    FailureSubClass.FLAKY_RACE_CONDITION: (
        "Add proper synchronization. Use locks, barriers, or make the test deterministic "
        "by mocking async operations. Do not add `time.sleep()` — that masks the problem."
    ),
    FailureSubClass.FLAKY_EXTERNAL_DEP: (
        "Mock external dependencies (HTTP, DB, time). Use `responses`, `httpretty`, "
        "`unittest.mock`, or VCR cassettes. Never rely on external services in CI tests."
    ),
    FailureSubClass.FLAKY_RESOURCE: (
        "Add resource cleanup in tearDown/fixture. Use dynamic port assignment. "
        "Add `@pytest.mark.flaky(reruns=3)` as a stopgap while root cause is fixed."
    ),
    FailureSubClass.FLAKY_TEST_ORDERING: (
        "Identify and remove shared state between tests. Use fresh fixtures. "
        "Add `--randomly-seed=0` to pytest to detect ordering dependencies systematically."
    ),
    FailureSubClass.DEP_DIRECT_BREAKING: (
        "Pin the breaking dependency to the last known-good version. Open a separate "
        "PR to migrate to the new API. Do NOT force-upgrade without addressing the API change."
    ),
    FailureSubClass.DEP_TRANSITIVE_CONFLICT: (
        "Add an explicit constraint for the conflicting transitive dep. Use "
        "`pip install --constraint` or add a `constraints.txt`. Regenerate lockfile."
    ),
    FailureSubClass.DEP_BUILD_TOOL: (
        "Pin the Python/Node/Go version in CI config. Add `.python-version`, "
        "`.nvmrc`, or `go.toolchain` to the repo root."
    ),
    FailureSubClass.DEP_LOCKFILE_STALE: (
        "Regenerate the lockfile: `pip-compile`, `npm install`, `poetry lock --no-update`. "
        "Commit the updated lockfile."
    ),
    FailureSubClass.ENV_BASE_IMAGE: (
        "Pin the Docker base image to a specific digest: "
        "`FROM ubuntu:22.04@sha256:...`. Avoid `:latest` tags in CI Dockerfiles."
    ),
    FailureSubClass.ENV_RUNNER_CHANGE: (
        "Pin the GitHub Actions runner to a specific Ubuntu version: "
        "`runs-on: ubuntu-22.04` (not `ubuntu-latest`). Pin action versions to tags, not `@main`."
    ),
    FailureSubClass.ENV_MISSING_SYSTEM_DEP: (
        "Add the missing package to the `apt-get install` step in the workflow YAML "
        "or Dockerfile. Check which package provides the missing library via `apt-cache search`."
    ),
    FailureSubClass.ENV_SECRET_MISSING: (
        "Add the missing secret to GitHub repository secrets. Check the workflow YAML "
        "for `${{ secrets.VAR_NAME }}` and ensure the corresponding secret is configured "
        "in Settings → Secrets and Variables → Actions."
    ),
    FailureSubClass.LOGIC_TEST_EXPECTATION: (
        "Update the test expectation to match the new correct behavior. Verify that "
        "the behavior change was intentional by reviewing the PR description."
    ),
    FailureSubClass.LOGIC_API_CONTRACT: (
        "Update callers to match the new API signature. Check the PR that changed "
        "the function signature and update all call sites. Consider adding a deprecation "
        "shim if the old signature needs to be supported."
    ),
    FailureSubClass.LOGIC_SCHEMA_MISMATCH: (
        "Add a database migration or update the schema definition. Check if a migration "
        "was added in the same PR as the code change. If not, create and apply it."
    ),
    FailureSubClass.LOGIC_IMPORT_ERROR: (
        "Update import paths to match the new module structure. Check if modules were "
        "moved or renamed in the failing commit. Update `__init__.py` exports if needed."
    ),
}


FIX_STRATEGY.update(
    {
        FailureSubClass.SECURITY_VULN_FOUND: (
            "Run `npm audit fix`, `pip-audit --fix`, or upgrade the flagged package to the patched version. "
            "Check the CVE details to verify exploitability before suppressing."
        ),
        FailureSubClass.SECURITY_LICENSE_VIOLATION: (
            "Replace the offending package with a license-compatible alternative. "
            "Add it to the license allowlist if business approval exists."
        ),
        FailureSubClass.SECURITY_SECRET_LEAK: (
            "Immediately rotate the exposed credential. Remove the secret from git history "
            "using `git filter-repo`. Add the pattern to `.gitignore` and a pre-commit hook."
        ),
        FailureSubClass.SECURITY_SBOM_MISMATCH: (
            "Regenerate the SBOM with `syft` or `cyclonedx-cli`. Ensure all dependencies are "
            "declared in the manifest before regenerating."
        ),
        FailureSubClass.LINT_STYLE_VIOLATION: (
            "Run the formatter locally: `black .`, `prettier --write .`, `rubocop -a`. "
            "Add a pre-commit hook to enforce formatting before CI."
        ),
        FailureSubClass.LINT_TYPE_ERROR: (
            "Fix the type annotation or add a type: ignore comment with explanation. "
            "Run `mypy --strict` locally to find all type errors before pushing."
        ),
        FailureSubClass.LINT_UNUSED_IMPORT: (
            "Remove unused imports. Use `autoflake --remove-all-unused-imports` or "
            "your IDE's 'Optimize Imports' feature."
        ),
        FailureSubClass.LINT_COMPLEXITY: (
            "Refactor the complex function into smaller helper functions. "
            "Target cyclomatic complexity <= 10 per function."
        ),
        FailureSubClass.BUILD_COMPILE_ERROR: (
            "Fix the compilation error shown in the log. Check for missing type annotations, "
            "undefined symbols, or incompatible API changes in recent commits."
        ),
        FailureSubClass.BUILD_MISSING_ARTIFACT: (
            "Add the missing build step that generates the artifact. Ensure build steps "
            "are ordered correctly in the CI workflow."
        ),
        FailureSubClass.BUILD_CACHE_INVALID: (
            "Clear the CI cache and retry. If this recurs, pin cache keys to "
            "lockfile hashes: `hashFiles('**/package-lock.json')`."
        ),
        FailureSubClass.BUILD_TOOL_INCOMPATIBLE: (
            "Pin the build tool version in CI: add `.gradle-version`, "
            "`mvn.version` property, or equivalent. Match the version used locally."
        ),
    }
)


def heuristic_classify_multilabel(log: str) -> list[FailureSignal]:
    """
    Multi-label heuristic classification.
    A CI failure can simultaneously be LINT + DEP_DRIFT, for example.
    Returns ALL signals with confidence > 0.5 (not just the top one).
    """
    return [s for s in heuristic_classify(log) if s.confidence >= 0.5]


def get_primary_and_secondary_classes(
    log: str,
) -> tuple[FailureClass, Optional[FailureSubClass], float, list[FailureClass]]:
    """
    Returns (primary_class, primary_subclass, confidence, additional_classes).
    Supports multi-label where a failure can be both LINT and DEP_DRIFT, etc.
    """
    signals = heuristic_classify(log)
    if not signals:
        return FailureClass.UNKNOWN, None, 0.0, []

    top = signals[0]
    # Collect all secondary classes with confidence > 0.5
    secondary = list(
        {
            s.failure_class
            for s in signals[1:]
            if s.confidence >= 0.5 and s.failure_class != top.failure_class
        }
    )

    return top.failure_class, top.failure_subclass, top.confidence, secondary


def get_fix_strategy(subclass: FailureSubClass) -> str:
    """Return the canonical fix strategy for a given failure sub-class."""
    return FIX_STRATEGY.get(
        subclass, "Unknown failure sub-class. Perform manual triage."
    )
