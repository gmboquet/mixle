#!/usr/bin/env python
"""Run and validate one content-addressed reproduction-bundle entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


def _normalize_output(entry: dict[str, Any], stdout: str) -> str:
    """Replace the entry's declared volatile spans before the digest is taken.

    An entry whose output legitimately names the commit being reproduced cannot carry a fixed
    stdout digest -- every commit invalidates it. Rather than drop the byte check for the whole
    entry, the volatile spans are declared in the bundle and normalized away, so everything else
    still has to match byte for byte. A pattern that matches nothing is an error, so a stale rule
    fails loudly instead of quietly widening what the digest accepts.
    """
    normalized = stdout
    for rule in entry["expected"].get("volatile", []):
        if not isinstance(rule, dict) or set(rule) != {"pattern", "placeholder"}:
            raise ValueError(f"{entry['id']}: volatile rule must contain exactly pattern and placeholder")
        normalized, count = re.subn(rule["pattern"], rule["placeholder"], normalized)
        if count == 0:
            raise ValueError(f"{entry['id']}: declared volatile pattern never matched: {rule['pattern']!r}")
    return normalized


def _validate_output(entry: dict[str, Any], stdout: str) -> None:
    expected = entry["expected"]
    if expected.get("format") == "text":
        actual_digest = hashlib.sha256(_normalize_output(entry, stdout).encode("utf-8")).hexdigest()
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


_ARTIFACT_PROBE = """
import json, sys
info = {"importable": False, "installed_distribution": False}
try:
    import mixle
    info["importable"] = True
    info["package_root"] = getattr(mixle, "__file__", None) or (list(mixle.__path__) or [None])[0]
    info["version"] = getattr(mixle, "__version__", None)
except Exception as exc:
    info["import_error"] = repr(exc)
    print(json.dumps(info)); raise SystemExit(0)
try:
    import pathlib
    from importlib.metadata import distribution
    dist = distribution("mixle")
    # The distribution is only the thing that EXECUTED if the imported package actually lives
    # inside it. A source checkout on sys.path (cwd, PYTHONPATH, editable install) shadows an
    # installed wheel while `distribution("mixle")` still resolves -- that is the false-verified
    # case this probe exists to catch. Compared as resolved paths: string prefixes get this wrong
    # (and `str.rstrip("mixle")` strips a character SET, not a suffix).
    located = pathlib.Path(str(dist.locate_file("mixle"))).resolve()
    root = pathlib.Path(info.get("package_root") or ".").resolve()
    info["distribution_located_at"] = str(located)
    info["installed_distribution"] = root == located or located in root.parents
    info["distribution_version"] = dist.version
except Exception as exc:
    info["metadata_error"] = repr(exc)
try:
    from mixle.reproduction import installed_content_provenance
    content = installed_content_provenance()
    info["installed_content_verified"] = content.get("verified")
    info["installed_content_digest"] = content.get("digest")
except Exception as exc:
    info["installed_content_error"] = repr(exc)
try:
    import json as _j, pathlib
    root = pathlib.Path(info.get("package_root") or ".").parent
    prov = _j.loads((root / "_build_provenance.json").read_text(encoding="utf-8"))
    info["source_commit"] = prov.get("source_commit")
    info["source_tree"] = prov.get("source_tree")
    info["source_dirty"] = prov.get("source_dirty")
except Exception:
    info["source_commit"] = None
print(json.dumps(info))
"""


def _executing_artifact(entry: dict[str, Any]) -> dict[str, Any]:
    """Identify the mixle that actually executes the entry, under the entry's own import conditions.

    The probe runs as a FILE placed beside the entry's own script, not via ``python -c``: for
    ``python examples/foo.py`` the interpreter puts ``examples/`` on ``sys.path``, while ``-c``
    puts the working directory there instead. Those resolve ``import mixle`` differently whenever
    a source checkout is the working directory, so probing with ``-c`` would describe an import
    the entry never performs.

    Reporting the *installed* distribution without checking that the imported package lives inside
    it is how a source-shadowed tree was recorded as candidate-verified evidence (SYS-03).
    """
    script_dir = (ROOT / entry["script"]).parent
    probe_path = script_dir / "_mixle_repro_artifact_probe.py"
    try:
        probe_path.write_text(_ARTIFACT_PROBE, encoding="utf-8")
        probe = subprocess.run(
            [sys.executable, str(probe_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return json.loads(probe.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, subprocess.SubprocessError, IndexError) as exc:
        return {"importable": False, "installed_distribution": False, "probe_error": repr(exc)}
    finally:
        probe_path.unlink(missing_ok=True)


def _resolve_candidate_records(bundle: dict[str, Any], records_root: Path | None) -> dict[str, Any]:
    """Resolve the bundle's ``required_records`` against a retained-records directory.

    ``validate_bundle`` only ever checked that this list was a list of at least four strings; it
    never opened a single one, so a bundle whose records were entirely absent validated cleanly
    (SYS-03). Resolution is reported per record, and absence is a stated fact rather than silence.
    """
    required = list(bundle["candidate_binding"]["required_records"])
    if records_root is None:
        return {"records_root": None, "resolved": False, "required": required, "present": [], "missing": required}
    present, missing = [], []
    for pattern in required:
        (present if any(records_root.glob(pattern)) else missing).append(pattern)
    return {
        "records_root": str(records_root),
        "resolved": not missing,
        "required": required,
        "present": present,
        "missing": missing,
    }


def run_entry(bundle: dict[str, Any], entry_id: str, *, records_root: Path | None = None) -> dict[str, Any]:
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

    artifact = _executing_artifact(entry)
    binding = _resolve_candidate_records(bundle, records_root)
    # Execution and candidate binding are separate facts and are reported separately. The entry
    # ran and its output matched byte for byte -- that is `execution_status`, and it is true even
    # from a source checkout, which is how a local reviewer legitimately exercises the bundle.
    # `claim_status` is the stronger statement: that this receipt is evidence ABOUT THE CANDIDATE
    # ARTIFACT. It requires the executing mixle to be the installed distribution (not a shadowing
    # source tree) with verified installed content, and the bundle's own required records to
    # actually resolve. Previously it was the constant "verified" (SYS-03).
    unbound: list[str] = []
    if not artifact.get("installed_distribution"):
        unbound.append("executing mixle is not the installed distribution (source tree shadows it)")
    if artifact.get("installed_content_verified") is not True:
        unbound.append("installed content did not verify")
    if not binding["resolved"]:
        unbound.append(f"candidate records unresolved: {', '.join(binding['missing']) or 'no records root given'}")

    return {
        "artifact": "mixle.reproduction_entry_receipt/v2",
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
        "executing_artifact": artifact,
        "candidate_binding": binding,
        "execution_status": "passed",
        "claim_status": "verified" if not unbound else "unbound",
        "unbound_reasons": unbound,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--entry", required=True)
    parser.add_argument(
        "--records-root",
        type=Path,
        default=None,
        help=(
            "directory holding the retained candidate records named by the bundle's "
            "candidate_binding.required_records. Without it a receipt can still record a passing "
            "execution, but cannot claim to be candidate-bound evidence."
        ),
    )
    args = parser.parse_args(argv)
    try:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        receipt = run_entry(bundle, args.entry, records_root=args.records_root)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
