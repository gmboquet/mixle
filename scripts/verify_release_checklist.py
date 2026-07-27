#!/usr/bin/env python3
"""Validate release-checklist status and candidate-bound DONE receipts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

ROW = re.compile(r"^\|\s*(?P<gate>[^|]+?)\s*\|\s*`(?P<status>[A-Z-]+)`\s*\|", re.MULTILINE)
SHA = re.compile(r"^[0-9a-f]{40}$")
FINAL = {"DONE", "EXCLUDED", "POST-RELEASE"}
KNOWN = FINAL | {"IMPLEMENTED", "HOSTED", "EXTERNAL"}


def _receipt_errors(receipt: dict, gate: str, candidate_sha: str) -> list[str]:
    errors: list[str] = []
    if receipt.get("artifact") != "mixle.release_gate_receipt/v1":
        errors.append(f"{gate}: unsupported receipt artifact")
    if receipt.get("gate") != gate:
        errors.append(f"{gate}: receipt gate identity differs")
    if receipt.get("candidate_commit") != candidate_sha:
        errors.append(f"{gate}: receipt commit differs")
    if not isinstance(receipt.get("command"), str) or not receipt["command"].strip():
        errors.append(f"{gate}: receipt has no command")
    if not isinstance(receipt.get("result"), str) or not re.search(r"\d", receipt["result"]):
        errors.append(f"{gate}: receipt result has no measured value")
    try:
        dt.date.fromisoformat(receipt["date"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{gate}: receipt date is invalid")
    return errors


def validate(checklist: Path, receipts: Path, candidate_sha: str, *, require_final: bool) -> list[str]:
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
                errors.extend(_receipt_errors(receipt, gate, candidate_sha))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{gate}: invalid or missing receipt: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    errors = validate(args.checklist, args.receipts, args.candidate_sha, require_final=args.require_final)
    if errors:
        print("\n".join(errors))
        return 1
    print("release checklist and candidate-bound receipts are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
