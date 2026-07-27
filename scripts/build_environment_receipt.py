#!/usr/bin/env python3
"""Record and verify the exact release build toolchain."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from packaging.requirements import Requirement


def _locked(requirements: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        pins = [item.version for item in requirement.specifier if item.operator == "=="]
        if len(pins) != 1 or len(requirement.specifier) != 1:
            raise ValueError(f"build requirement must be exactly pinned: {line}")
        locked[requirement.name] = pins[0]
    return locked


def build(requirements: Path, commit: str, source_date_epoch: int) -> dict:
    locked = _locked(requirements)
    installed = {name: importlib.metadata.version(name) for name in locked}
    if installed != locked:
        raise ValueError(f"builder differs from lock: expected={locked}, installed={installed}")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("candidate commit must be a full lowercase Git SHA")
    if source_date_epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative")
    return {
        "artifact": "mixle.build_environment/v1",
        "candidate_commit": commit,
        "packages": installed,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_date_epoch": source_date_epoch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build(
            args.requirements,
            os.environ.get("CANDIDATE_SHA", ""),
            int(os.environ.get("SOURCE_DATE_EPOCH", "-1")),
        )
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
