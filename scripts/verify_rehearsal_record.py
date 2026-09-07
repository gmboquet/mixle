#!/usr/bin/env python3
"""Verify the TestPyPI rehearsal record that gates promotion, in either of its two forms.

The automated form (``mixle.testpypi_rehearsal/v1``) is the artifact publish.yml's ``testpypi``
phase writes after it downloaded the published bytes, checked them against the candidate's
SHA256SUMS, installed the wheel, and swept every public module.

The manual form (``mixle.testpypi_rehearsal_manual/v1``) exists for exactly one situation that
the automated phase cannot pass by construction: the version is already on TestPyPI from an
EARLIER candidate of the same release. TestPyPI never accepts a filename twice, and the artifact
bytes depend on the commit timestamp, so no later candidate's bytes can ever be uploaded there and
the automated digest check fails forever. The manual record is produced by
``scripts/record_manual_rehearsal.py`` from the draft release's own assets, uploaded to the draft
release by the repository owner, and named to the promote phase through the
``manual_rehearsal_asset`` input. This verifier accepts it only when every binding holds AND the
exemption's precondition is demonstrably true on TestPyPI right now; the promote job then
re-derives the smoke itself (clean install of the prepared wheel plus the import sweep) rather
than trusting the record's word for it.

Exit status 0 means accepted; anything else is a refusal with the reason on stderr.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

AUTOMATED = "mixle.testpypi_rehearsal/v1"
MANUAL = "mixle.testpypi_rehearsal_manual/v1"
MANUAL_REASON = "version-already-on-testpypi-from-an-earlier-candidate"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Refusal(Exception):
    """The record is not acceptable evidence; the message says why."""


def parse_sums(text: str) -> dict[str, str]:
    """``sha256sum`` format: ``<digest>  <filename>`` per line."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or not _SHA256.match(parts[0]):
            raise Refusal(f"malformed SHA256SUMS line: {line!r}")
        out[parts[1].lstrip("*")] = parts[0]
    if not out:
        raise Refusal("SHA256SUMS is empty")
    return out


def fetch_testpypi_files(version: str, project: str = "mixle") -> dict[str, str]:
    """``{filename: sha256}`` of what TestPyPI serves for ``project==version`` (``{}`` if absent)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", project) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*(?:[a-z0-9.]*)", version):
        raise Refusal(f"refusing to build a TestPyPI URL from {project!r} / {version!r}")
    url = f"https://test.pypi.org/pypi/{project}/{version}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "mixle-release-gate"})
    try:
        # the scheme and host are fixed above and both path segments are validated; no caller
        # input can redirect this to file:// or another scheme (bandit B310)
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise
    return {entry["filename"]: entry["digests"]["sha256"] for entry in payload.get("urls", [])}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def verify(
    record: dict,
    *,
    candidate_sha: str,
    prepare_run: int,
    sums_text: str,
    manual: bool,
    version: str | None = None,
    uploader: str | None = None,
    testpypi_files: dict[str, str] | None = None,
) -> str:
    """Return a one-line acceptance summary or raise :class:`Refusal`."""
    _require(isinstance(record, dict), "record is not a JSON object")
    _require(_SHA40.match(candidate_sha) is not None, "candidate sha must be 40 hex characters")
    sums = parse_sums(sums_text)
    sums_digest = hashlib.sha256(sums_text.encode("utf-8")).hexdigest()
    artifact = record.get("artifact")
    _require(record.get("candidate_commit") == candidate_sha, "record is bound to a different candidate commit")
    _require(record.get("prepare_run") == prepare_run, "record is bound to a different prepare run")
    _require(record.get("sums_sha256") == sums_digest, "record is bound to a different SHA256SUMS")
    _require(record.get("passed") is True, "record does not claim a passed rehearsal")

    if artifact == AUTOMATED:
        _require(not manual, "an automated record was offered where a manual one was declared")
        return "rehearsal record accepted (automated)"

    _require(artifact == MANUAL, f"unknown rehearsal record kind: {artifact!r}")
    _require(manual, "a manual rehearsal record was offered without the manual declaration")
    _require(version is not None and uploader is not None, "manual verification needs --version and --uploader")
    _require(testpypi_files is not None, "manual verification needs the live TestPyPI file listing")
    _require(record.get("reason") == MANUAL_REASON, "manual record does not state the one admissible reason")
    _require(record.get("verified_by") == uploader, "manual record was not verified by the repository owner")
    try:
        _dt.datetime.fromisoformat(str(record.get("verified_at")))
    except ValueError as exc:
        raise Refusal("manual record has no ISO-8601 verified_at") from exc
    _require(
        record.get("reproducible_builds_passed") is True,
        "manual record does not carry a passed reproducible-build check",
    )
    _require(record.get("clean_install") is True, "manual record does not claim a clean install")
    modules = record.get("import_sweep_modules")
    _require(
        isinstance(modules, int) and not isinstance(modules, bool) and modules > 0,
        "manual record has no import-sweep count",
    )

    draft = record.get("draft_release_files")
    _require(isinstance(draft, dict) and draft == sums, "manual record's draft-release digests do not equal SHA256SUMS")

    # The exemption's precondition, checked against TestPyPI itself: this version is there, every
    # candidate filename is there, and at least one of them holds DIFFERENT bytes from this
    # candidate. If the bytes were identical the automated phase would pass and must be used.
    claimed = record.get("testpypi_files")
    _require(
        isinstance(claimed, dict) and claimed == testpypi_files,
        "manual record's TestPyPI digests do not match what TestPyPI serves now",
    )
    _require(
        set(sums) <= set(testpypi_files),
        "TestPyPI does not hold every candidate filename; the automated rehearsal applies",
    )
    _require(
        any(testpypi_files[name] != digest for name, digest in sums.items()),
        "TestPyPI already holds this candidate's exact bytes; the automated rehearsal applies",
    )
    return (
        f"rehearsal record accepted (manual; {modules} modules swept; TestPyPI holds an earlier candidate's {version})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--prepare-run", type=int, required=True)
    parser.add_argument("--sums", type=Path, required=True)
    parser.add_argument(
        "--manual", action="store_true", help="the record is a manual one attached to the draft release"
    )
    parser.add_argument("--version")
    parser.add_argument("--uploader", help="login that uploaded the record; must be the repository owner")
    parser.add_argument("--testpypi-json", type=Path, help="TestPyPI project JSON to use instead of fetching (tests)")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        testpypi_files = None
        if args.manual:
            if args.version is None:
                raise Refusal("--manual needs --version")
            if args.testpypi_json is not None:
                payload = json.loads(args.testpypi_json.read_text(encoding="utf-8"))
                testpypi_files = {e["filename"]: e["digests"]["sha256"] for e in payload.get("urls", [])}
            else:
                testpypi_files = fetch_testpypi_files(args.version)
        summary = verify(
            record,
            candidate_sha=args.candidate_sha,
            prepare_run=args.prepare_run,
            sums_text=args.sums.read_text(encoding="utf-8"),
            manual=args.manual,
            version=args.version,
            uploader=args.uploader,
            testpypi_files=testpypi_files,
        )
    except (Refusal, OSError, ValueError, KeyError) as exc:
        print(f"rehearsal record refused: {exc}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
