#!/usr/bin/env python3
"""Validate the public claim inventory and reject unregistered overclaims."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "manifests" / "public_claims.json"
LEDGER = ROOT / "docs" / "claim-evidence-ledger.rst"

# These patterns identify prose that needs an explicit evidence decision. They
# intentionally focus on product-scale, performance, safety, and universality
# rather than treating every version number or example parameter as a claim.
CLAIM_PATTERNS = (
    re.compile(r"(?i)(?<![\w.])~?(?:[1-9]\d*(?:\.\d+)?|0\.\d+)\s*x(?:\b|-standard-error)"),
    re.compile(r"(?i)\b\d[\d,]*\+\s+(?:tests?|famil(?:y|ies)|models?|backends?|distributions?)\b"),
    re.compile(
        r"(?i)(?:\bmixle\b|\bthe (?:package|library|model)\b).{0,80}"
        r"\b(?:faster|slower|outperform(?:s|ed|ing)?|superior)\b"
    ),
    re.compile(
        r"(?i)(?:(?:\bmixle\b|\bthe (?:package|library|model)\b).{0,100}"
        r"\b(?:production[- ]ready|safe to deploy|frontier[- ](?:class|scale))\b|"
        r"\b(?:production[- ]ready|safe to deploy|frontier[- ](?:class|scale))\b.{0,100}"
        r"(?:\bmixle\b|\bthe (?:package|library|model)\b))"
    ),
    re.compile(r"(?i)\b(?:any|every)\s+(?:engine|backend)\b"),
)


def _load() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _surface_paths(data: dict) -> list[Path]:
    config = data["public_surfaces"]
    paths = {ROOT / name for name in config["files"]}
    for pattern in config["globs"]:
        paths.update(ROOT.glob(pattern))
    return sorted(
        path
        for path in paths
        if path.is_file()
        and not _excluded(path.relative_to(ROOT).as_posix(), config["exclude_globs"])
    )


def _prose(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return text
    try:
        return ast.get_docstring(ast.parse(text)) or ""
    except SyntaxError as exc:
        raise ValueError(f"cannot parse public example {path.relative_to(ROOT)}: {exc}") from exc


def scan(data: dict) -> list[dict[str, str | int]]:
    hits: list[dict[str, str | int]] = []
    for path in _surface_paths(data):
        relative = path.relative_to(ROOT).as_posix()
        text = _prose(path)
        lines = text.splitlines()
        for line_number, line in enumerate(lines, 1):
            for pattern in CLAIM_PATTERNS:
                for match in pattern.finditer(line):
                    context = " ".join(lines[max(0, line_number - 2) : min(len(lines), line_number + 1)])
                    lowered = context.lower()
                    if any(
                        qualifier in lowered
                        for qualifier in (
                            "does not",
                            "not automatically",
                            "not every",
                            "not measured",
                            "not production",
                            "out of scope",
                            "avoid implying",
                            "no \"safe to deploy\"",
                            "before documenting",
                            "are **not**",
                            "should not",
                            "remain torch/deepspeed territory",
                        )
                    ):
                        continue
                    hits.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "text": line.strip(),
                            "match": match.group(0),
                        }
                    )
    return hits


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("artifact") != "mixle.public_claims/v1":
        errors.append("artifact must be mixle.public_claims/v1")
    grades = set(data.get("evidence_grades", []))
    claims = data.get("claims", [])
    claim_ids = [claim.get("id") for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("claim IDs must be unique")
    ledger = LEDGER.read_text(encoding="utf-8")
    for claim in claims:
        missing = {"id", "statement", "grade", "evidence"} - claim.keys()
        if missing:
            errors.append(f"{claim.get('id', '<unknown>')}: missing {sorted(missing)}")
            continue
        if claim["grade"] not in grades:
            errors.append(f"{claim['id']}: unknown evidence grade {claim['grade']!r}")
        if not claim["evidence"]:
            errors.append(f"{claim['id']}: evidence must not be empty")
        position = ledger.find(claim["statement"])
        if position < 0:
            errors.append(f"{claim['id']}: statement is absent from the claim-evidence ledger")
        else:
            row_tail = ledger[position : position + len(claim["statement"]) + 120]
            if f"\n     - {claim['grade']}" not in row_tail:
                errors.append(f"{claim['id']}: inventory grade does not match its ledger row")

    approved = data.get("approved_occurrences", [])
    approved_keys: set[tuple[str, str]] = set()
    for item in approved:
        missing = {"claim_id", "path", "text"} - item.keys()
        if missing:
            errors.append(f"approved occurrence missing {sorted(missing)}")
            continue
        if item["claim_id"] not in claim_ids:
            errors.append(f"occurrence names unknown claim {item['claim_id']!r}")
        key = (item["path"], item["text"])
        if key in approved_keys:
            errors.append(f"duplicate approved occurrence {item['path']}: {item['text']}")
        approved_keys.add(key)

    for hit in scan(data):
        key = (str(hit["path"]), str(hit["text"]))
        if key not in approved_keys:
            errors.append(
                f"unregistered public claim at {hit['path']}:{hit['line']}: "
                f"{hit['match']!r} in {hit['text']!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="print scanner hits without validating approvals")
    args = parser.parse_args()
    data = _load()
    if args.report:
        print(json.dumps(scan(data), indent=2))
        return 0
    errors = validate(data)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"validated {len(data['claims'])} claims across {len(_surface_paths(data))} public surfaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
