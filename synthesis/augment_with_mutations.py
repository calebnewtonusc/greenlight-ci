"""
augment_with_mutations.py - Synthetic CI failure data generation via deterministic mutations.

Takes real CI failure->fix pairs and generates synthetic variations by mutating:
- Package version numbers
- Python/Node/Go runtime versions
- Environment variable names
- OS/runner versions
- Docker image tags

All mutations are deterministic (seeded by content hash) so they're verifiable
and reproducible. Creates synthetic training examples that test model generalization
across version ranges.

Usage:
    python synthesis/augment_with_mutations.py --input data/
    python synthesis/augment_with_mutations.py --input data/ --mutations-per-record 3
    python synthesis/augment_with_mutations.py --dry-run --input data/
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parents[1] / "data"
AUGMENTED_FILE = DATA_DIR / "augmented_mutations.jsonl"

# ─── Version mutation tables ──────────────────────────────────────────────────
# Maps known package versions to realistic alternative versions for mutations.
# All versions here are real released versions.

PYTHON_VERSIONS = [
    "3.8", "3.8.18", "3.9", "3.9.18", "3.10", "3.10.13",
    "3.11", "3.11.7", "3.12", "3.12.1",
]

NODE_VERSIONS = [
    "16", "16.20.2", "18", "18.19.0", "20", "20.10.0", "21", "21.5.0",
]

UBUNTU_VERSIONS = [
    "ubuntu-20.04", "ubuntu-22.04", "ubuntu-latest",
    "ubuntu:20.04", "ubuntu:22.04",
]

PYTHON_BASE_IMAGES = [
    "python:3.9-slim", "python:3.10-slim", "python:3.11-slim",
    "python:3.9-alpine", "python:3.10-alpine", "python:3.11-alpine",
    "python:3.9-bullseye", "python:3.10-bullseye", "python:3.11-bookworm",
]

NODE_BASE_IMAGES = [
    "node:16-slim", "node:18-slim", "node:20-slim",
    "node:16-alpine", "node:18-alpine", "node:20-alpine",
    "node:16-bullseye", "node:18-bullseye",
]

# Common package version patterns for mutation
PACKAGE_VERSION_PATTERNS = {
    "major.minor.patch": re.compile(r'(\d+)\.(\d+)\.(\d+)'),
    "major.minor": re.compile(r'(\d+)\.(\d+)(?!\.\d)'),
    "range_gte": re.compile(r'>=(\d+\.\d+)'),
    "range_caret": re.compile(r'\^(\d+\.\d+\.\d+)'),
    "range_tilde": re.compile(r'~(\d+\.\d+\.\d+)'),
}

# ─── Mutation strategies ───────────────────────────────────────────────────────

def deterministic_rng(content: str, mutation_index: int) -> random.Random:
    """Create a deterministic RNG seeded by content hash + mutation index."""
    seed_str = f"{hashlib.sha256(content.encode()).hexdigest()}:{mutation_index}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return rng


def mutate_python_version(log: str, rng: random.Random) -> tuple[str, str]:
    """Mutate Python version references in a log."""
    python_ver_pattern = re.compile(r'python(\s+|\-|:)([23]\.\d+(?:\.\d+)?)', re.I)
    matches = list(python_ver_pattern.finditer(log))
    if not matches:
        return log, ""

    # Pick one match to mutate
    match = rng.choice(matches)
    old_version = match.group(2)
    new_version = rng.choice([v for v in PYTHON_VERSIONS if v != old_version])

    mutated = log[:match.start(2)] + new_version + log[match.end(2):]
    description = f"Python version changed from {old_version} to {new_version}"
    return mutated, description


def mutate_node_version(log: str, rng: random.Random) -> tuple[str, str]:
    """Mutate Node.js version references in a log."""
    node_ver_pattern = re.compile(r'node(\s+|\-|:)(v?1[46-9]|2[01])\.\d+(?:\.\d+)?', re.I)
    matches = list(node_ver_pattern.finditer(log))
    if not matches:
        return log, ""

    match = rng.choice(matches)
    old_version = match.group(2)
    new_version = rng.choice([v for v in NODE_VERSIONS if not v.startswith(old_version)])

    mutated = log[:match.start(2)] + new_version + log[match.end(2):]
    description = f"Node.js version changed from {old_version} to {new_version}"
    return mutated, description


def mutate_package_version(log: str, package_name: str, rng: random.Random) -> tuple[str, str]:
    """
    Mutate a specific package's version number in a log.
    Uses minor/patch version bumps for realistic mutations.
    """
    # Pattern: package_name==X.Y.Z or package_name>=X.Y or package_name: X.Y.Z
    pkg_pattern = re.compile(
        re.escape(package_name) + r'[=<>^~\s:]+(\d+)\.(\d+)\.(\d+)',
        re.I
    )
    match = pkg_pattern.search(log)
    if not match:
        return log, ""

    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
    old_version_str = f"{major}.{minor}.{patch}"

    # Generate realistic mutated version
    mutation_type = rng.choice(["minor_bump", "patch_bump", "major_constraint"])
    if mutation_type == "patch_bump":
        new_patch = patch + rng.randint(1, 5)
        new_version = f"{major}.{minor}.{new_patch}"
    elif mutation_type == "minor_bump":
        new_minor = minor + rng.randint(1, 3)
        new_version = f"{major}.{new_minor}.0"
    else:
        new_major = major + 1
        new_version = f"{new_major}.0.0"

    mutated = log[:match.start(1)] + new_version + log[match.end(3):]
    description = f"{package_name} version changed from {old_version_str} to {new_version}"
    return mutated, description


def mutate_runner_version(log: str, rng: random.Random) -> tuple[str, str]:
    """Mutate GitHub Actions runner / Ubuntu version references."""
    runner_pattern = re.compile(r'ubuntu-(\d+\.\d+|latest)', re.I)
    matches = list(runner_pattern.finditer(log))
    if not matches:
        return log, ""

    match = rng.choice(matches)
    old_runner = "ubuntu-" + match.group(1)
    new_runner = rng.choice([r for r in UBUNTU_VERSIONS if r != old_runner])

    mutated = log.replace(old_runner, new_runner, 1)
    description = f"Runner changed from {old_runner} to {new_runner}"
    return mutated, description


def mutate_docker_image(log: str, rng: random.Random) -> tuple[str, str]:
    """Mutate Docker base image tags in Dockerfile snippets."""
    from_pattern = re.compile(r'FROM\s+(python|node):([^\s\n]+)', re.I)
    match = from_pattern.search(log)
    if not match:
        return log, ""

    image_type = match.group(1).lower()
    old_tag = match.group(1) + ":" + match.group(2)

    if image_type == "python":
        new_image = rng.choice(PYTHON_BASE_IMAGES)
    else:
        new_image = rng.choice(NODE_BASE_IMAGES)

    if new_image == old_tag:
        return log, ""

    mutated = log[:match.start(1)] + new_image + log[match.end(2):]
    description = f"Docker base image changed from {old_tag} to {new_image}"
    return mutated, description


def mutate_env_var_name(log: str, rng: random.Random) -> tuple[str, str]:
    """Mutate environment variable names to simulate missing/renamed secrets."""
    env_pattern = re.compile(r'\$\{\{?\s*secrets\.(\w+)\s*\}\}?|\$(\w+_KEY|\w+_TOKEN|\w+_SECRET)')
    matches = list(env_pattern.finditer(log))
    if not matches:
        return log, ""

    match = rng.choice(matches)
    old_name = match.group(1) or match.group(2)
    # Add a suffix to simulate a renamed variable
    suffixes = ["_V2", "_NEW", "_2024", "_PROD", "_UPDATED"]
    new_name = old_name + rng.choice(suffixes)

    mutated = log.replace(old_name, new_name, 1)
    description = f"Environment variable renamed from {old_name} to {new_name}"
    return mutated, description


# ─── Mutation strategies available ────────────────────────────────────────────
MUTATION_STRATEGIES = [
    ("python_version", mutate_python_version),
    ("node_version", mutate_node_version),
    ("runner_version", mutate_runner_version),
    ("docker_image", mutate_docker_image),
    ("env_var", mutate_env_var_name),
]


def apply_mutation(
    record: dict,
    strategy_name: str,
    strategy_fn,
    mutation_index: int,
) -> Optional[dict]:
    """
    Apply a single mutation strategy to a CI failure record.
    Returns a new record with the mutated failure_log and metadata.
    """
    original_log = record.get("ci_log", "")
    if not original_log:
        return None

    rng = deterministic_rng(original_log, mutation_index)
    mutated_log, description = strategy_fn(original_log, rng)

    if not description or mutated_log == original_log:
        return None  # Mutation didn't apply

    # Create a synthetic training record
    mutated_record = {
        **record,
        "ci_log": mutated_log,
        "is_synthetic": True,
        "mutation_strategy": strategy_name,
        "mutation_description": description,
        "original_record_id": (
            record.get("pipeline_id") or
            record.get("build_id") or
            record.get("pr_number") or
            hashlib.sha256(original_log[:100].encode()).hexdigest()[:12]
        ),
        "mutation_index": mutation_index,
    }

    # The fix stays the same — the mutation changes WHAT fails but the fix pattern
    # for the class should generalize
    return mutated_record


def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if not filepath.exists():
        return records
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def save_records(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUGMENTED_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CI failure variants via deterministic mutations"
    )
    parser.add_argument("--input", type=Path, default=DATA_DIR,
                        help="Directory containing failure log JSONL files")
    parser.add_argument("--mutations-per-record", type=int, default=3,
                        help="Number of mutations to generate per source record")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample mutations without saving")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy names (default: all)")
    args = parser.parse_args()

    # Select mutation strategies
    if args.strategies:
        selected_names = set(args.strategies.split(","))
        strategies = [(n, f) for n, f in MUTATION_STRATEGIES if n in selected_names]
    else:
        strategies = MUTATION_STRATEGIES

    print(f"=== CI MUTATION AUGMENTATION ===")
    print(f"Strategies: {[s[0] for s in strategies]}")
    print(f"Mutations per record: {args.mutations_per_record}")

    # Load all CI failure records
    source_files = [
        args.input / "github_ci_failure_logs.jsonl",
        args.input / "circleci_failure_logs.jsonl",
        args.input / "travis_failure_logs.jsonl",
        args.input / "gitlab_failure_logs.jsonl",
    ]

    all_records = []
    for f in source_files:
        if f.exists():
            recs = load_jsonl(f)
            print(f"  Loaded {len(recs):>6} records from {f.name}")
            all_records.extend(recs)

    if not all_records:
        print("No source records found. Run discovery scripts first.")
        return

    print(f"\nTotal source records: {len(all_records)}")
    print(f"Generating up to {len(all_records) * args.mutations_per_record} mutations...")

    total_generated = 0
    batch = []

    for record in all_records:
        # Try each mutation strategy
        mutation_count = 0
        for mutation_index, (strategy_name, strategy_fn) in enumerate(strategies):
            if mutation_count >= args.mutations_per_record:
                break

            mutated = apply_mutation(record, strategy_name, strategy_fn, mutation_index)
            if mutated:
                batch.append(mutated)
                total_generated += 1
                mutation_count += 1

                if args.dry_run and total_generated <= 3:
                    print(f"\n  [DRY RUN] Strategy: {strategy_name}")
                    print(f"  Description: {mutated['mutation_description']}")
                    print(f"  Original (first 100 chars): {record.get('failure_log', '')[:100]}")
                    print(f"  Mutated  (first 100 chars): {mutated.get('failure_log', '')[:100]}")

        # Save in batches
        if len(batch) >= 1000 and not args.dry_run:
            save_records(batch)
            batch = []

    # Save remaining
    if batch and not args.dry_run:
        save_records(batch)

    print(f"\n=== SUMMARY ===")
    print(f"Source records: {len(all_records)}")
    print(f"Mutations generated: {total_generated}")
    if not args.dry_run:
        print(f"Output: {AUGMENTED_FILE}")
    else:
        print("(dry run - no files written)")


if __name__ == "__main__":
    main()
