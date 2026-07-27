#!/usr/bin/env python
"""Bind every required example receipt to one exact candidate and wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _contract_digest(entry: dict[str, Any]) -> str:
    encoded = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    bundle_path: Path,
    candidate_path: Path,
    wheel_metadata_path: Path,
    receipt_paths: list[Path],
) -> dict[str, Any]:
    bundle = _read(bundle_path)
    if bundle.get("artifact") != "mixle.reproduction_bundle/v2":
        raise ValueError("unsupported reproduction bundle")
    candidate = _read(candidate_path)
    if candidate.get("artifact") != "mixle.release_candidate/v1":
        raise ValueError("unsupported release-candidate record")
    commit = candidate.get("commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("candidate commit must be a full lowercase Git SHA")
    if candidate.get("version") != bundle.get("release"):
        raise ValueError("candidate version does not match the example bundle")

    wheel = _read(wheel_metadata_path)
    if not isinstance(wheel.get("filename"), str) or not wheel["filename"].endswith(".whl"):
        raise ValueError("example evidence must bind a wheel metadata record")
    digest = wheel.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("wheel metadata has no valid SHA-256")

    receipts: dict[str, dict[str, Any]] = {}
    for path in receipt_paths:
        receipt = _read(path)
        entry_id = receipt.get("entry")
        if not isinstance(entry_id, str) or entry_id in receipts:
            raise ValueError(f"duplicate or invalid example receipt: {entry_id!r}")
        receipts[entry_id] = receipt

    expected_entries = {entry["id"]: entry for entry in bundle["entries"]}
    if set(receipts) != set(expected_entries):
        raise ValueError(
            f"example receipt set is incomplete: missing={sorted(set(expected_entries) - set(receipts))}, "
            f"extra={sorted(set(receipts) - set(expected_entries))}"
        )

    examples = []
    for entry_id, entry in sorted(expected_entries.items()):
        receipt = receipts[entry_id]
        if receipt.get("artifact") != "mixle.reproduction_entry_receipt/v2" or receipt.get("passed") is not True:
            raise ValueError(f"{entry_id}: receipt is not a passing reproduction receipt")
        if receipt.get("execution_status") != "passed" or receipt.get("claim_status") != "verified":
            raise ValueError(f"{entry_id}: receipt did not separately verify execution and claimed behavior")
        if receipt.get("entry_contract_sha256") != _contract_digest(entry):
            raise ValueError(f"{entry_id}: receipt is for a different entry contract")
        if receipt.get("argv") != entry["argv"] or receipt.get("tier") != entry["tier"]:
            raise ValueError(f"{entry_id}: command or dependency tier drifted")
        duration = receipt.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
            or duration > entry["timeout_seconds"]
        ):
            raise ValueError(f"{entry_id}: invalid execution duration")
        if receipt.get("validated_output") != entry["expected"]:
            raise ValueError(f"{entry_id}: validated-output contract drifted")
        examples.append(
            {
                "id": entry_id,
                "command": entry["argv"],
                "dependency_tier": entry["tier"],
                "duration_seconds": duration,
                "execution_status": "passed",
                "claim_status": "verified",
                "acceptance_contract": entry["expected"],
                "stdout_sha256": receipt["stdout_sha256"],
                "receipt_sha256": hashlib.sha256(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest(),
            }
        )

    environment_path = ROOT / bundle["environment"]
    return {
        "artifact": "mixle.example_execution_manifest/v2",
        "complete": True,
        "candidate": {
            "commit": commit,
            "tag": candidate.get("tag"),
            "version": candidate["version"],
            "workflow_run": candidate.get("workflow_run"),
        },
        "wheel": {
            "filename": wheel["filename"],
            "sha256": digest,
            "size_bytes": wheel.get("size_bytes"),
        },
        "dependency_profile": {
            "path": bundle["environment"],
            "sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
        },
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--wheel-metadata", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(
            bundle_path=args.bundle,
            candidate_path=args.candidate,
            wheel_metadata_path=args.wheel_metadata,
            receipt_paths=args.receipt,
        )
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
