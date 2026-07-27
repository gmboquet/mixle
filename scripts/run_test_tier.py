#!/usr/bin/env python
"""Run one named pytest tier under its hard budget and retain a timing receipt."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_BUDGETS = {
    "core": 240,
    "full": 1200,
    "optional": 1800,
    "numerical": 1800,
    "hardware": 1200,
}
_SHA = re.compile(r"^[0-9a-f]{40}$")


def run(
    tier: str,
    *,
    candidate_sha: str,
    budget_seconds: int,
    pytest_args: list[str],
) -> dict:
    if tier not in _BUDGETS:
        raise ValueError(f"unknown test tier: {tier}")
    if not _SHA.fullmatch(candidate_sha):
        raise ValueError("candidate SHA must be a full lowercase Git SHA")
    if isinstance(budget_seconds, bool) or not 0 < budget_seconds <= _BUDGETS[tier]:
        raise ValueError(f"{tier} budget must be between 1 and {_BUDGETS[tier]} seconds")
    if "-m" in pytest_args or "--markers" in pytest_args:
        raise ValueError("the tier runner owns marker selection")

    command = [sys.executable, "-m", "pytest", "-m", tier, *pytest_args]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, timeout=budget_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{tier} tier exceeded its {budget_seconds}-second budget") from exc
    duration = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{tier} tier failed with exit {completed.returncode}")
    return {
        "artifact": "mixle.test_tier_receipt/v1",
        "tier": tier,
        "candidate_commit": candidate_sha,
        "command": command,
        "budget_seconds": budget_seconds,
        "duration_seconds": duration,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=sorted(_BUDGETS), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--budget-seconds", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    try:
        receipt = run(
            args.tier,
            candidate_sha=args.candidate_sha,
            budget_seconds=args.budget_seconds,
            pytest_args=pytest_args,
        )
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
