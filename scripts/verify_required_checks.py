#!/usr/bin/env python
"""Fail closed unless an exact candidate SHA has every required successful check run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def required_check_names(path: Path) -> tuple[str, ...]:
    """Read unique, nonempty check names from a line-oriented policy file."""
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not names:
        raise ValueError("required-check policy must name at least one check")
    if len(names) != len(set(names)):
        raise ValueError("required-check policy contains duplicate names")
    return names


def verify_required_checks(payload: object, required: tuple[str, ...], sha: str) -> dict[str, int]:
    """Return selected successful run IDs, or raise for missing/stale/failed evidence."""
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError("candidate SHA must be a full 40-character hexadecimal commit ID")
    if not isinstance(payload, dict) or not isinstance(payload.get("check_runs"), list):
        raise ValueError("check-run evidence must be a GitHub check-runs object")

    by_name: dict[str, list[dict]] = {name: [] for name in required}
    for run in payload["check_runs"]:
        if not isinstance(run, dict) or run.get("name") not in by_name:
            continue
        if run.get("head_sha") != sha:
            continue
        run_id = run.get("id")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 0:
            raise ValueError(f"check {run.get('name')!r} has an invalid run ID")
        by_name[run["name"]].append(run)

    selected: dict[str, int] = {}
    failures: list[str] = []
    for name, runs in by_name.items():
        if not runs:
            failures.append(f"{name}: missing for {sha}")
            continue
        latest = max(runs, key=lambda run: run["id"])
        if latest.get("status") != "completed" or latest.get("conclusion") != "success":
            failures.append(
                f"{name}: latest run {latest['id']} is "
                f"{latest.get('status')}/{latest.get('conclusion')}"
            )
            continue
        selected[name] = latest["id"]
    if failures:
        raise ValueError("release-candidate checks are not approved:\n- " + "\n- ".join(failures))
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args(argv)
    try:
        required = required_check_names(args.required)
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        selected = verify_required_checks(payload, required, args.sha)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact": "mixle.release_check_evidence/v1",
                "commit": args.sha,
                "checks": selected,
                "publication_authorized": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
