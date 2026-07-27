#!/usr/bin/env python3
"""Require two clean candidate builds to produce exactly identical artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _digests(directory: Path) -> dict[str, str]:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"no artifacts in {directory}")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def verify(left: Path, right: Path) -> dict:
    first = _digests(left)
    second = _digests(right)
    if first.keys() != second.keys():
        raise ValueError(
            f"build artifact sets differ: missing={sorted(first.keys() - second.keys())}, "
            f"extra={sorted(second.keys() - first.keys())}"
        )
    changed = sorted(name for name in first if first[name] != second[name])
    if changed:
        raise ValueError(f"candidate builds are not byte-for-byte reproducible: {changed}")
    return {
        "artifact": "mixle.reproducible_builds/v1",
        "files": first,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = verify(args.left, args.right)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
