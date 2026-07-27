#!/usr/bin/env python3
"""Generate exact constraints from one optional profile's declared lower bounds."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def constraints(profile: str, pyproject: Path = ROOT / "pyproject.toml") -> list[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    if profile not in extras:
        raise ValueError(f"unknown optional profile {profile!r}")
    requirements = [*project["dependencies"], *extras[profile]]
    pinned: dict[tuple[str, str], str] = {}
    for raw in requirements:
        requirement = Requirement(raw)
        lower = [item.version for item in requirement.specifier if item.operator == ">="]
        unsupported = [str(item) for item in requirement.specifier if item.operator not in {">=", "<"}]
        if unsupported or len(lower) != 1:
            raise ValueError(f"{raw!r} needs exactly one >= floor and optional < upper bound")
        marker = str(requirement.marker) if requirement.marker is not None else ""
        key = (requirement.name.lower().replace("_", "-"), marker)
        rendered = f"{requirement.name}=={lower[0]}" + (f"; {marker}" if marker else "")
        previous = pinned.get(key)
        if previous is not None and previous != rendered:
            raise ValueError(f"conflicting floors for {requirement.name}: {previous!r} and {rendered!r}")
        pinned[key] = rendered
    return [pinned[key] for key in sorted(pinned)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        lines = constraints(args.profile, args.pyproject)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
