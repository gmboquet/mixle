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

# Re-baselined 2026-08-04 against measured need, not aspiration. The suite grew from ~10k to ~15k
# collected tests over the 0.8.0 hardening cycle, and the budgets did not move with it: the hosted
# 4-vCPU Linux runners were killing the core tier at 240s with ~61% of it executed (measured need
# ~390s; Apple-silicon runners finish the same tier in ~150s) and the full tier at 1200s with ~94%
# executed (measured 1279s). A kill at 61% is not a budget, it is a guarantee of failure -- and it
# presents as an opaque teardown OSError rather than as a timeout, which cost real diagnosis time.
# The tripwire property is preserved: a genuine 2x regression from today's measured need still trips.
# Second re-baseline 2026-08-04, and this one prices in RUNNER VARIANCE, which the first did not:
# on the same day, the same core tier on the same interpreter (py3.11, ubuntu-latest) completed in
# 342s on one hosted runner and was killed at 480s on another -- a 1.4x spread with zero failing
# tests. A budget at ~1.2x the median is therefore a coin flip per lane per run. Budgets are now
# ~2x the fastest observed completion (core 342s -> 720; full 1279s -> 2400), which still trips on
# any real multiplicative regression while no longer failing lanes for drawing a slow runner.
# Full-tier correction 2026-08-04 (third revision, and the reason is an evidence error, recorded as
# D-0142): the "fastest observed full completion 1279s" that priced the previous ceiling was a
# misread -- that summary line belonged to the OPTIONAL tier. The full tier (~14k tests under
# coverage on 4-vCPU hosted runners) has never completed inside any budget this cycle; the only
# genuine datum is a kill at 2400s with 72% executed, projecting ~3300s. Its budget is now ~1.6x
# that projection. Core's 720 held: every core lane on both platforms passed inside it.
_BUDGETS = {
    "core": 900,
    "full": 5400,
    "optional": 2700,
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
