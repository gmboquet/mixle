#!/usr/bin/env python
"""Record or verify exact-SHA dependency-profile receipts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SHA = re.compile(r"^[0-9a-f]{40}$")


def record(profile: str, candidate_sha: str) -> dict:
    if not profile or not _SHA.fullmatch(candidate_sha):
        raise ValueError("profile and full lowercase candidate SHA are required")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.splitlines()
    return {
        "artifact": "mixle.environment_receipt/v1",
        "profile": profile,
        "candidate_commit": candidate_sha,
        "python": sys.version.split()[0],
        "resolved_dependencies": sorted(line for line in freeze if line),
        "passed": True,
    }


def verify(directory: Path, policy_path: Path, candidate_sha: str) -> dict:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    expected = set(policy["profiles"])
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
    profiles = [receipt.get("profile") for receipt in receipts]
    if len(profiles) != len(set(profiles)) or set(profiles) != expected:
        raise ValueError(
            f"environment receipt profiles differ: missing={sorted(expected - set(profiles))}, "
            f"extra={sorted(set(profiles) - expected)}"
        )
    for receipt in receipts:
        if (
            receipt.get("artifact") != "mixle.environment_receipt/v1"
            or receipt.get("candidate_commit") != candidate_sha
            or receipt.get("passed") is not True
            or not receipt.get("resolved_dependencies")
        ):
            raise ValueError(f"invalid environment receipt for profile {receipt.get('profile')!r}")
    return {
        "artifact": "mixle.environment_receipt_set/v1",
        "candidate_commit": candidate_sha,
        "profiles": sorted(expected),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--profile", required=True)
    record_parser.add_argument("--candidate-sha", required=True)
    record_parser.add_argument("--out", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--directory", type=Path, required=True)
    verify_parser.add_argument("--policy", type=Path, required=True)
    verify_parser.add_argument("--candidate-sha", required=True)
    verify_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            result = record(args.profile, args.candidate_sha)
        else:
            result = verify(args.directory, args.policy, args.candidate_sha)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, subprocess.SubprocessError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
