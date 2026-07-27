#!/usr/bin/env python
"""Verify that downloaded public-index artifacts exactly match retained release hashes."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.+-]+)$")


def expected_hashes(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _LINE.fullmatch(line)
        if match is None or match.group(2) in expected:
            raise ValueError(f"invalid SHA256SUMS line {number}")
        expected[match.group(2)] = match.group(1)
    if (
        not expected
        or not any(name.endswith(".whl") for name in expected)
        or not any(name.endswith(".tar.gz") for name in expected)
    ):
        raise ValueError("SHA256SUMS must name at least one wheel and one source distribution")
    return expected


def verify(directory: Path, sums: Path) -> dict[str, str]:
    expected = expected_hashes(sums)
    actual_files = {path.name: path for path in directory.iterdir() if path.is_file()}
    if set(actual_files) != set(expected):
        raise ValueError(
            f"published artifact set differs: missing={sorted(set(expected) - set(actual_files))}, "
            f"extra={sorted(set(actual_files) - set(expected))}"
        )
    for name, path in actual_files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"published artifact digest differs: {name}")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sums", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verified = verify(args.directory, args.sums)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(sorted(verified)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
