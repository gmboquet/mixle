#!/usr/bin/env python3
"""Fail publication closed until license provenance has independent approval."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = ROOT / "release-checklists" / "0.8.0-license-provenance.json"


def validate(record: object, *, require_approved: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict) or record.get("artifact") != "mixle.license_provenance/v1":
        return ["unsupported license-provenance record"]
    notice = record.get("original_notice")
    if not isinstance(notice, str) or not notice:
        errors.append("original_notice must be present")
    else:
        for filename in ("LICENSE", "NOTICE"):
            if notice not in (ROOT / filename).read_text(encoding="utf-8"):
                errors.append(f"{filename} does not preserve original_notice")
    if record.get("status") not in {"pending", "approved", "blocked"}:
        errors.append("status must be pending, approved, or blocked")
    if require_approved:
        if record.get("status") != "approved":
            errors.append("independent license-provenance approval is required for publication")
        if not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip():
            errors.append("approved record must name a reviewer")
        if not isinstance(record.get("review_date"), str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", record["review_date"]
        ):
            errors.append("approved record must contain an ISO review_date")
        if not isinstance(record.get("evidence"), str) or not record["evidence"].strip():
            errors.append("approved record must cite assignment/authorization evidence")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        errors = validate(record, require_approved=args.require_approved)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    if errors:
        print("\n".join(errors))
        return 1
    print("license provenance validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
