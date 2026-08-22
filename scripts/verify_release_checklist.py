#!/usr/bin/env python3
"""Validate release-checklist status and candidate-bound DONE receipts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROW = re.compile(r"^\|\s*(?P<gate>[^|]+?)\s*\|\s*`(?P<status>[A-Z-]+)`\s*\|", re.MULTILINE)
SHA = re.compile(r"^[0-9a-f]{40}$")
FINAL = {"DONE", "EXCLUDED", "POST-RELEASE"}
KNOWN = FINAL | {"IMPLEMENTED", "HOSTED", "EXTERNAL"}


def evidence_digest(root: Path, receipts_dir: Path) -> str:
    """A digest of the candidate's tracked content EXCLUDING the receipts directory.

    Why not simply require ``candidate_commit == <the tag's SHA>``: the receipts are ordinary files in
    this repository, so a committed receipt would have to name the SHA of the commit that contains it,
    and a commit's hash is determined by its content. That requirement was therefore impossible to
    satisfy -- which is why no receipt has ever existed and why this gate had never run (D-0193).

    The only thing that differs between "the bytes the gate evidence was measured on" and "the bytes
    that get tagged" is the receipts directory itself, so that is precisely what this digest leaves
    out. Everything a gate could actually be evidence ABOUT is still bound: change any tracked file
    outside the receipts directory and every receipt stops validating. Computed from git's own index
    of tracked paths and blob hashes, so it is stable across checkouts and needs no working-tree scan.
    """
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    try:
        excluded = receipts_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        excluded = None
    entries = []
    for line in listing.splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        if excluded is not None and (path == excluded or path.startswith(excluded + "/")):
            continue
        entries.append(f"{meta.split()[1]} {path}")
    entries.sort()
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _receipt_errors(receipt: dict, gate: str, candidate_sha: str, digest: str | None = None) -> list[str]:
    errors: list[str] = []
    if receipt.get("artifact") != "mixle.release_gate_receipt/v1":
        errors.append(f"{gate}: unsupported receipt artifact")
    if receipt.get("gate") != gate:
        errors.append(f"{gate}: receipt gate identity differs")
    # The receipt names the commit its evidence was measured on. That commit is normally the
    # candidate's parent (the receipts are committed on top of it), so equality cannot be required;
    # the binding that does the work is the evidence digest below.
    if not isinstance(receipt.get("candidate_commit"), str) or SHA.fullmatch(receipt["candidate_commit"]) is None:
        errors.append(f"{gate}: receipt names no measured commit")
    if digest is not None and receipt.get("evidence_digest") != digest:
        errors.append(
            f"{gate}: receipt evidence digest {str(receipt.get('evidence_digest'))[:12]}... does not match "
            f"this candidate's content {digest[:12]}... (the gate was measured on different bytes)"
        )
    if not isinstance(receipt.get("command"), str) or not receipt["command"].strip():
        errors.append(f"{gate}: receipt has no command")
    if not isinstance(receipt.get("result"), str) or not re.search(r"\d", receipt["result"]):
        errors.append(f"{gate}: receipt result has no measured value")
    try:
        dt.date.fromisoformat(receipt["date"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{gate}: receipt date is invalid")
    return errors


def validate(
    checklist: Path, receipts: Path, candidate_sha: str, *, require_final: bool, digest: str | None = None
) -> list[str]:
    if SHA.fullmatch(candidate_sha) is None:
        return ["candidate SHA must be a full lowercase Git SHA"]
    rows = [(match.group("gate").strip(), match.group("status")) for match in ROW.finditer(checklist.read_text())]
    errors: list[str] = []
    if not rows:
        return ["release checklist contains no gate rows"]
    for gate, status in rows:
        if status not in KNOWN:
            errors.append(f"{gate}: unknown status {status}")
            continue
        if require_final and status not in FINAL:
            errors.append(f"{gate}: pre-publication status remains {status}")
        if status == "DONE":
            path = receipts / f"{re.sub(r'[^a-z0-9]+', '-', gate.lower()).strip('-')}.json"
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                errors.extend(_receipt_errors(receipt, gate, candidate_sha, digest))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{gate}: invalid or missing receipt: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root, for the evidence digest")
    args = parser.parse_args()
    try:
        digest = evidence_digest(args.root, args.receipts)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"cannot compute the candidate evidence digest: {exc}")
        return 1
    errors = validate(
        args.checklist, args.receipts, args.candidate_sha, require_final=args.require_final, digest=digest
    )
    if errors:
        print("\n".join(errors))
        return 1
    print("release checklist and candidate-bound receipts are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
