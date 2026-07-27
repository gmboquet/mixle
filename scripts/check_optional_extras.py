#!/usr/bin/env python3
"""Require ``all`` to equal the union of supported runtime-feature extras."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DEVELOPMENT_EXTRAS = {"all", "docs", "lint", "test"}


def _identity(requirement: str) -> tuple[str, str, str]:
    parsed = Requirement(requirement)
    marker = str(parsed.marker) if parsed.marker is not None else ""
    return parsed.name.lower().replace("_", "-"), str(parsed.specifier), marker


def validate(path: Path = PYPROJECT) -> list[str]:
    project = tomllib.loads(path.read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    runtime_names = set(extras) - DEVELOPMENT_EXTRAS
    expected = {_identity(item) for name in runtime_names for item in extras[name]}
    actual = {_identity(item) for item in extras["all"]}
    errors = []
    if len(actual) != len(extras["all"]):
        errors.append("all contains duplicate requirement identities")
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(f"all omits runtime requirements: {missing}")
    if unexpected:
        errors.append(f"all contains requirements outside runtime extras: {unexpected}")
    if "examples" not in runtime_names:
        errors.append("external comparison examples need a declared examples extra")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    args = parser.parse_args()
    errors = validate(args.pyproject)
    if errors:
        print("\n".join(errors))
        return 1
    print("all equals the runtime-feature extra union")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
