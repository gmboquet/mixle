#!/usr/bin/env python
"""Fail closed unless an exact candidate SHA has every required successful check run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

GENERATOR = "scripts/verify_required_checks.py"
# GitHub grammar: an owner is alphanumerics/hyphens not starting with a hyphen; a repository name is
# alphanumerics, ".", "_", "-" and is never "." or ".."
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*/(?!\.\.?$)[A-Za-z0-9_.-]+$")
_ACTIONS_RUN_ID = re.compile(r"^[1-9][0-9]*$")
# GitHub Actions is the only app whose check runs are the release evidence; a check run reported
# by any other app (or none) is not one of our workflow jobs, whatever its name says (SYS5-01).
_ACTIONS_APP_SLUG = "github-actions"


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


def _validated_repository(repository: object) -> str:
    """The OWNER/REPO whose check runs these are; the URL invariant below is relative to it."""
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise ValueError(
            "--repository must be OWNER/REPO in GitHub's grammar (owner: alphanumerics and '-'; repo: alphanumerics, '_', '.', '-')"
        )
    return repository


def _check_run_id(run: object, position: int) -> int:
    """The id of one supplied run, or raise naming the run if it has no valid one (SYS5-02)."""
    if not isinstance(run, dict):
        raise ValueError(f"check run at position {position} is not an object")
    run_id = run.get("id")
    # GitHub never issues id 0; the resolver requires > 0, and generator and resolver must agree
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise ValueError(f"check run at position {position} ({run.get('name')!r}) has an invalid ID {run_id!r}")
    return run_id


def _all_check_runs(payload: object) -> tuple[list[dict], int]:
    """Every check run in the evidence and GitHub's total, or raise if the evidence is not demonstrably complete.

    Accepts one GitHub ``check-runs`` page object or a list of them (``gh api --paginate --slurp``).
    Every page reports the commit's ``total_count``; the runs supplied must add up to it, because
    "latest run per name" is only meaningful over ALL runs -- a truncated first page can omit the
    newest run and leave an older success standing in for a newer failure.

    Adding up is necessary, not sufficient (SYS5-02): a run supplied twice pads the count back to
    ``total_count`` after a newer failure is dropped, so every run must carry a valid id and the
    ids must be unique across every page. That is checked BEFORE the count comparison so the
    refusal names the duplicated id even when the evidence is also short.
    """
    pages = payload if isinstance(payload, list) else [payload]
    if not pages:
        raise ValueError("check-run evidence is empty")
    runs: list[dict] = []
    seen_ids: set[int] = set()
    totals: set[int] = set()
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("check_runs"), list):
            raise ValueError("check-run evidence must be a GitHub check-runs object")
        total = page.get("total_count")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("check-run evidence lacks GitHub's total_count")
        totals.add(total)
        for run in page["check_runs"]:
            run_id = _check_run_id(run, len(runs))
            if run_id in seen_ids:
                raise ValueError(
                    f"check-run evidence has a duplicate check run ID {run_id}: a run supplied twice "
                    "cannot stand in for a missing run"
                )
            seen_ids.add(run_id)
            runs.append(run)
    if len(totals) != 1:
        raise ValueError(f"check-run evidence pages disagree on total_count: {sorted(totals)}")
    (total,) = totals
    if len(runs) != total:
        raise ValueError(
            f"check-run evidence is incomplete: GitHub reports {total} check runs for the commit "
            f"but {len(runs)} were supplied (paginate the request)"
        )
    return runs, total


def _selected_run_problem(run: dict, repository: str) -> str | None:
    """Why a selected run is not a GitHub Actions job of ``repository``, or None if it is (SYS5-01).

    GitHub's invariant, verified 68/68 on the retained candidate: an Actions check run's
    ``details_url`` is exactly ``https://github.com/{repository}/actions/runs/{run_id}/job/{id}``
    with the job segment EQUAL to the check-run id, and its ``app.slug`` is ``github-actions``.
    The former substring test (``"/actions/runs/" in details_url``) accepted any host.
    """
    app = run.get("app")
    if not isinstance(app, dict) or app.get("slug") != _ACTIONS_APP_SLUG:
        return f"was not reported by the {_ACTIONS_APP_SLUG} app"
    details_url = run.get("details_url")
    prefix = f"https://github.com/{repository}/actions/runs/"
    suffix = f"/job/{run['id']}"
    if (
        not isinstance(details_url, str)
        or not details_url.startswith(prefix)
        or not details_url.endswith(suffix)
        or _ACTIONS_RUN_ID.fullmatch(details_url[len(prefix) : len(details_url) - len(suffix)]) is None
    ):
        return f"has details_url {details_url!r}, not {prefix}<run_id>{suffix} of {repository}"
    return None


def verify_required_checks(
    payload: object, required: tuple[str, ...], sha: str, *, repository: str
) -> dict[str, dict[str, object]]:
    """Return selected successful check identities, or raise for missing/stale/failed evidence."""
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError("candidate SHA must be a full 40-character hexadecimal commit ID")
    repository = _validated_repository(repository)
    check_runs, _ = _all_check_runs(payload)

    # The per-commit check-runs endpoint returns only that commit's runs (measured: 93/93 and
    # 101/101 on two real commits). A run for any other commit therefore cannot be part of GitHub's
    # answer -- and if merely ignored, it pads the count back to total_count after a newer failure
    # is dropped, exactly as a duplicated id did (SYS5-02's neighbour). Refuse the evidence.
    for position, run in enumerate(check_runs):
        if run.get("head_sha") != sha:
            raise ValueError(
                f"check-run evidence contains run {run.get('id')} at position {position} for commit "
                f"{run.get('head_sha')!r}, not the candidate {sha}; this is not the commit's check-runs listing"
            )

    by_name: dict[str, list[dict]] = {name: [] for name in required}
    for run in check_runs:
        if run.get("name") in by_name:
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
        problem = _selected_run_problem(latest, repository)
        if problem is not None:
            failures.append(f"{name}: latest run {latest['id']} {problem}")
            continue
        selected[name] = {"check_run_id": latest["id"], "details_url": latest["details_url"]}
    if failures:
        raise ValueError("release-candidate checks are not approved:\n- " + "\n- ".join(failures))
    return selected


def evidence_record(raw: bytes, required: tuple[str, ...], sha: str, *, repository: str) -> dict[str, object]:
    """The release-check-evidence record for the EXACT check-runs bytes ``raw``, or raise.

    The record names its generator and source (repository, endpoint, sha256 of the input bytes,
    GitHub's total_count) so a consumer can re-fetch the same endpoint and re-derive the selection
    (SYS5-01). Identical input bytes yield an identical record.
    """
    repository = _validated_repository(repository)
    payload = json.loads(raw.decode("utf-8"))
    selected = verify_required_checks(payload, required, sha, repository=repository)
    _, total = _all_check_runs(payload)
    return {
        "artifact": "mixle.release_check_evidence/v1",
        "generator": GENERATOR,
        "commit": sha,
        "source": {
            "repository": repository,
            "endpoint": f"repos/{repository}/commits/{sha}/check-runs",
            "check_runs_sha256": hashlib.sha256(raw).hexdigest(),
            "check_runs_count": total,
        },
        "checks": selected,
        "publication_authorized": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    # validated in code rather than by argparse so that a missing or malformed value exits 1 like
    # every other refusal of this gate (SYS5-01)
    parser.add_argument("--repository", default=None, help="OWNER/REPO whose check runs --input holds")
    args = parser.parse_args(argv)
    try:
        repository = _validated_repository(args.repository)
        required = required_check_names(args.required)
        record = evidence_record(args.input.read_bytes(), required, args.sha, repository=repository)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
