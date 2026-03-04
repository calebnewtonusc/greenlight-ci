"""
GreenLight CI — Domain Knowledge Base
Curated patterns, known failure signatures, and fix templates
that supplement the trained model's weights.

This knowledge base is used at inference time to augment model context
and at training time to generate high-quality synthesized examples.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KnownFailurePattern:
    """A known CI failure pattern with canonical fix template."""

    name: str
    failure_class: str
    failure_subclass: str
    signatures: list[str]  # Log substrings that identify this pattern
    languages: list[str]  # Applicable languages
    fix_template: str  # Template for the fix (with {VERSION}, {PACKAGE} etc.)
    description: str


# ── Common Python dependency failures ──────────────────────────────────────────

PYTHON_DEP_PATTERNS: list[KnownFailurePattern] = [
    KnownFailurePattern(
        name="psycopg2_to_psycopg3",
        failure_class="DEP_DRIFT",
        failure_subclass="2a_direct_dep_breaking_change",
        signatures=["cannot import name 'connect' from 'psycopg'", "psycopg 3"],
        languages=["python"],
        fix_template="Pin psycopg to <3.0 in requirements: psycopg>=2.9,<3.0",
        description="psycopg3 changed its public API; psycopg 2.x must be pinned explicitly",
    ),
    KnownFailurePattern(
        name="setuptools_pkg_resources_removal",
        failure_class="DEP_DRIFT",
        failure_subclass="2a_direct_dep_breaking_change",
        signatures=["from pkg_resources import", "pkg_resources has been removed"],
        languages=["python"],
        fix_template="Add explicit setuptools dependency or migrate to importlib.resources",
        description="setuptools 67+ removed pkg_resources; need explicit dependency or migration",
    ),
    KnownFailurePattern(
        name="poetry_lock_outdated",
        failure_class="DEP_DRIFT",
        failure_subclass="2d_lockfile_stale",
        signatures=["poetry.lock is not up to date", "run poetry lock"],
        languages=["python"],
        fix_template="Run: poetry lock --no-update && commit poetry.lock",
        description="poetry.lock is out of sync with pyproject.toml",
    ),
    KnownFailurePattern(
        name="pydantic_v1_v2_migration",
        failure_class="DEP_DRIFT",
        failure_subclass="2a_direct_dep_breaking_change",
        signatures=["validator.*is not a class", "model_validator", "pydantic"],
        languages=["python"],
        fix_template="Pin pydantic to <2.0 or migrate validators to v2 style (@model_validator)",
        description="Pydantic v2 broke v1 validator decorators; must pin or migrate",
    ),
]

# ── Common environment failures ─────────────────────────────────────────────────

ENV_PATTERNS: list[KnownFailurePattern] = [
    KnownFailurePattern(
        name="ubuntu_24_libssl",
        failure_class="ENV",
        failure_subclass="3c_missing_system_dep",
        signatures=["libssl-dev has no installation candidate", "ubuntu-24"],
        languages=["python", "ruby", "rust"],
        fix_template="Replace 'libssl-dev' with 'openssl' in apt-get install step",
        description="Ubuntu 24.04 restructured OpenSSL packages",
    ),
    KnownFailurePattern(
        name="actions_checkout_v4_node",
        failure_class="ENV",
        failure_subclass="3b_runner_os_tool_change",
        signatures=["Node.js 12 actions are deprecated", "actions/checkout@v1"],
        languages=["python", "javascript", "go", "java", "ruby"],
        fix_template="Upgrade actions/checkout to @v4, actions/setup-python to @v5",
        description="GitHub deprecated Node.js 12/16 actions; must upgrade to v4+",
    ),
    KnownFailurePattern(
        name="docker_latest_tag",
        failure_class="ENV",
        failure_subclass="3a_base_image_update",
        signatures=["FROM.*:latest", "E: Package.*no installation candidate"],
        languages=["python", "javascript", "go"],
        fix_template="Pin FROM image to specific version: FROM ubuntu:22.04 (not :latest)",
        description=":latest Docker tags change silently; pin to specific version",
    ),
]

# ── Common flaky test patterns ──────────────────────────────────────────────────

FLAKY_PATTERNS: list[KnownFailurePattern] = [
    KnownFailurePattern(
        name="asyncio_event_loop_closed",
        failure_class="FLAKY",
        failure_subclass="1a_race_condition",
        signatures=["Event loop is closed", "RuntimeError: no running event loop"],
        languages=["python"],
        fix_template="Use asyncio.new_event_loop() in pytest fixture, close in teardown",
        description="Async tests sharing event loop cause RuntimeError when loop closes",
    ),
    KnownFailurePattern(
        name="port_already_in_use",
        failure_class="FLAKY",
        failure_subclass="1c_resource_exhaustion",
        signatures=["Address already in use", "port.*already in use", "EADDRINUSE"],
        languages=["python", "javascript", "go"],
        fix_template="Use port=0 (OS-assigned) for test servers, or add ephemeral port fixture",
        description="Hardcoded ports conflict when tests run in parallel or in rapid succession",
    ),
    KnownFailurePattern(
        name="datetime_timezone_flaky",
        failure_class="FLAKY",
        failure_subclass="1b_external_dependency",
        signatures=["AssertionError.*datetime", "timezone", "utcnow"],
        languages=["python"],
        fix_template="Mock datetime.now() in tests; use freeze_gun or unittest.mock.patch",
        description="Tests comparing datetime.now() are non-deterministic across timezones",
    ),
]

# ── All patterns flat list ──────────────────────────────────────────────────────

ALL_PATTERNS: list[KnownFailurePattern] = (
    PYTHON_DEP_PATTERNS + ENV_PATTERNS + FLAKY_PATTERNS
)


def find_matching_patterns(
    log_text: str,
    language: Optional[str] = None,
) -> list[KnownFailurePattern]:
    """
    Find known patterns that match a CI log.
    Returns patterns sorted by number of matching signatures (most specific first).
    """
    matches: list[tuple[int, KnownFailurePattern]] = []

    for pattern in ALL_PATTERNS:
        if language and language.lower() not in pattern.languages:
            continue

        match_count = sum(
            1 for sig in pattern.signatures if sig.lower() in log_text.lower()
        )
        if match_count > 0:
            matches.append((match_count, pattern))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in matches]
