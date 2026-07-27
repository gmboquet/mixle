#!/usr/bin/env python
"""Run and validate one content-addressed reproduction-bundle entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_input(item: object) -> None:
    if not isinstance(item, dict) or set(item) != {"path", "role", "sha256"}:
        raise ValueError("bundle input must contain exactly path, role, and sha256")
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"bundle input escapes the repository: {relative}")
    target = ROOT / relative
    if not target.is_file():
        raise ValueError(f"bundle input is missing: {relative}")
    if _sha256(target) != item["sha256"]:
        raise ValueError(f"bundle input digest drifted: {relative}")


def _validate_environment(bundle: dict[str, Any]) -> None:
    path = ROOT / bundle["environment"]
    environment = json.loads(path.read_text(encoding="utf-8"))
    if environment.get("artifact") != "mixle.reproduction_environment/v1":
        raise ValueError("unsupported reproduction-environment schema")
    minimum = tuple(environment["python"]["minimum"].split("."))
    if sys.version_info[: len(minimum)] < tuple(int(part) for part in minimum):
        raise ValueError(f"Python {environment['python']['minimum']} or newer is required")
    for package, expected in environment["dependencies"].items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f"required reproduction dependency is absent: {package}") from exc
        if actual != expected:
            raise ValueError(f"{package}=={expected} is required; found {actual}")


def validate_bundle(bundle: object) -> dict[str, Any]:
    """Validate schema, closure, licenses, execution controls, and unique entry IDs."""
    if not isinstance(bundle, dict) or bundle.get("artifact") != "mixle.reproduction_bundle/v2":
        raise ValueError("unsupported reproduction-bundle schema")
    binding = bundle.get("candidate_binding")
    if not isinstance(binding, dict) or binding.get("policy") != "exact-publish-workflow-candidate":
        raise ValueError("bundle lacks exact release-candidate binding")
    required_records = binding.get("required_records")
    if not isinstance(required_records, list) or len(required_records) < 4:
        raise ValueError("bundle candidate binding is incomplete")
    for item in bundle.get("closure", []):
        _validate_input(item)
    entries = bundle.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("bundle must contain entries")
    identifiers = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    if len(identifiers) != len(entries) or len(identifiers) != len(set(identifiers)):
        raise ValueError("bundle entry IDs must be present and unique")
    for entry in entries:
        if entry.get("tier") not in {"local", "hosted-network"}:
            raise ValueError(f"{entry.get('id')}: invalid execution tier")
        timeout = entry.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 0 < timeout <= 30:
            raise ValueError(f"{entry.get('id')}: timeout must be an integer from 1 through 30 seconds")
        argv = entry.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value for value in argv):
            raise ValueError(f"{entry.get('id')}: argv must be a nonempty string list")
        if argv[0] != entry.get("script"):
            raise ValueError(f"{entry.get('id')}: argv must execute its declared script")
        for item in entry.get("inputs", []):
            _validate_input(item)
        serialized = json.dumps(entry, sort_keys=True).upper()
        if any(marker in serialized for marker in ("CONFIRM-AT-PUBLISH", "TODO", "TBD")):
            raise ValueError(f"{entry.get('id')}: unresolved placeholder")
        if not isinstance(entry.get("expected"), dict):
            raise ValueError(f"{entry.get('id')}: expected-result contract is missing")
    _validate_environment(bundle)
    return bundle


def _lookup(payload: object, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"JSON result has no field {path!r}")
        value = value[part]
    return value


def _validate_output(entry: dict[str, Any], stdout: str) -> None:
    expected = entry["expected"]
    if expected.get("format") == "text":
        actual_digest = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        if actual_digest != expected.get("stdout_sha256"):
            raise ValueError(
                f"{entry['id']}: stdout digest mismatch; expected {expected.get('stdout_sha256')}, "
                f"received {actual_digest}"
            )
        for fragment in expected.get("contains", []):
            if fragment not in stdout:
                raise ValueError(f"{entry['id']}: expected output fragment is absent: {fragment!r}")
        return
    if expected.get("format") != "json":
        raise ValueError(f"{entry['id']}: unsupported expected output format")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{entry['id']}: output is not one strict JSON document") from exc
    for assertion in expected.get("assertions", []):
        actual = _lookup(payload, assertion["path"])
        if "equals" in assertion and (type(actual) is not type(assertion["equals"]) or actual != assertion["equals"]):
            raise ValueError(
                f"{entry['id']}: {assertion['path']} expected {assertion['equals']!r}, received {actual!r}"
            )
        if "minimum" in assertion and (
            not isinstance(actual, (int, float)) or isinstance(actual, bool) or actual < assertion["minimum"]
        ):
            raise ValueError(f"{entry['id']}: {assertion['path']} is below its required minimum")
        if "maximum" in assertion and (
            not isinstance(actual, (int, float)) or isinstance(actual, bool) or actual > assertion["maximum"]
        ):
            raise ValueError(f"{entry['id']}: {assertion['path']} is above its required maximum")


def run_entry(bundle: dict[str, Any], entry_id: str) -> dict[str, Any]:
    """Execute one validated entry and return a content-addressed success receipt."""
    validate_bundle(bundle)
    matches = [entry for entry in bundle["entries"] if entry["id"] == entry_id]
    if len(matches) != 1:
        raise ValueError(f"unknown reproduction entry: {entry_id}")
    entry = matches[0]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, *entry["argv"]],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=entry["timeout_seconds"],
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{entry_id}: exceeded {entry['timeout_seconds']}-second budget") from exc
    if result.returncode != 0:
        raise ValueError(f"{entry_id}: exited {result.returncode}: {result.stderr[-2000:]}")
    duration = time.monotonic() - started
    _validate_output(entry, result.stdout)
    return {
        "artifact": "mixle.reproduction_entry_receipt/v1",
        "entry": entry_id,
        "argv": entry["argv"],
        "tier": entry["tier"],
        "duration_seconds": duration,
        "timeout_seconds": entry["timeout_seconds"],
        "entry_contract_sha256": hashlib.sha256(
            json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "validated_output": entry["expected"],
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--entry", required=True)
    args = parser.parse_args(argv)
    try:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        receipt = run_entry(bundle, args.entry)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
