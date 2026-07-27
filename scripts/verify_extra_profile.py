#!/usr/bin/env python3
"""Import every direct dependency module in an installed optional profile."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(profile: str) -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    if profile not in extras:
        raise ValueError(f"unknown optional profile {profile!r}")
    mapping = json.loads((ROOT / "manifests" / "optional_dependency_imports.json").read_text(encoding="utf-8"))[
        "distribution_to_imports"
    ]
    imported = []
    for requirement in extras[profile]:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match is None:
            raise ValueError(f"invalid optional requirement {requirement!r}")
        distribution = match.group(0)
        if distribution not in mapping:
            raise ValueError(f"no import mapping for {distribution}")
        for module in mapping[distribution]:
            importlib.import_module(module)
            imported.append(module)
    return sorted(set(imported))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    try:
        imported = verify(args.profile)
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc))
        return 1
    print(json.dumps({"artifact": "mixle.extra_profile_imports/v1", "profile": args.profile, "imports": imported}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
