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


def _all_check_runs(payload: object) -> list[dict]:
    """Every check run in the evidence, or raise if the evidence is not demonstrably complete.

    Accepts one GitHub ``check-runs`` page object or a list of them (``gh api --paginate --slurp``).
    Every page reports the commit's ``total_count``; the runs supplied must add up to it, because
    "latest run per name" is only meaningful over ALL runs -- a truncated first page can omit the
    newest run and leave an older success standing in for a newer failure.
    """
    pages = payload if isinstance(payload, list) else [payload]
    if not pages:
        raise ValueError("check-run evidence is empty")
    runs: list[dict] = []
    totals: set[int] = set()
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("check_runs"), list):
            raise ValueError("check-run evidence must be a GitHub check-runs object")
        total = page.get("total_count")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("check-run evidence lacks GitHub's total_count")
        totals.add(total)
        runs.extend(page["check_runs"])
    if len(totals) != 1:
        raise ValueError(f"check-run evidence pages disagree on total_count: {sorted(totals)}")
    (total,) = totals
    if len(runs) != total:
        raise ValueError(
            f"check-run evidence is incomplete: GitHub reports {total} check runs for the commit "
            f"but {len(runs)} were supplied (paginate the request)"
        )
    return runs


def verify_required_checks(payload: object, required: tuple[str, ...], sha: str) -> dict[str, dict[str, object]]:
    """Return selected successful check identities, or raise for missing/stale/failed evidence."""
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError("candidate SHA must be a full 40-character hexadecimal commit ID")
    check_runs = _all_check_runs(payload)

    by_name: dict[str, list[dict]] = {name: [] for name in required}
    for run in check_runs:
        if not isinstance(run, dict) or run.get("name") not in by_name:
            continue
        if run.get("head_sha") != sha:
            continue
        run_id = run.get("id")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 0:
            raise ValueError(f"check {run.get('name')!r} has an invalid run ID")
        by_name[run["name"]].append(run)

    selected: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for name, runs in by_name.items():
        if not runs:
            failures.append(f"{name}: missing for {sha}")
            continue
        latest = max(runs, key=lambda run: run["id"])
        if latest.get("status") != "completed" or latest.get("conclusion") != "success":
            failures.append(f"{name}: latest run {latest['id']} is {latest.get('status')}/{latest.get('conclusion')}")
            continue
        details_url = latest.get("details_url")
        if not isinstance(details_url, str) or "/actions/runs/" not in details_url:
            failures.append(f"{name}: latest run {latest['id']} lacks an Actions run URL")
            continue
        selected[name] = {"check_run_id": latest["id"], "details_url": details_url}
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
