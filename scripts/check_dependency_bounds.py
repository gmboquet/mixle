#!/usr/bin/env python3
"""Validate runtime dependency upper bounds against the reviewed drift policy."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def validate() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    policy = json.loads((ROOT / "manifests" / "dependency_drift_policy.json").read_text(encoding="utf-8"))
    errors = []
    if policy.get("artifact") != "mixle.dependency_drift_policy/v1":
        errors.append("unsupported dependency drift policy")
        return errors
    requirements = [*project["dependencies"], *project["optional-dependencies"]["all"]]
    actual: dict[str, str] = {}
    for raw in requirements:
        parsed = Requirement(raw)
        uppers = [str(item) for item in parsed.specifier if item.operator in {"<", "<="}]
        name = parsed.name.lower().replace("_", "-")
        if len(uppers) != 1:
            errors.append(f"{name} must have exactly one reviewed upper bound")
        else:
            actual[name] = uppers[0]
    expected = policy.get("dependencies")
    if actual != expected:
        errors.append(
            f"dependency drift policy differs: missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}, "
            f"changed={sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])}"
        )
    if not policy.get("decision") or not policy.get("review"):
        errors.append("dependency drift policy needs a decision and review cadence")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("\n".join(errors))
        return 1
    print("runtime dependency bounds match reviewed drift policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
