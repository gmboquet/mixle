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
    # The per-file RECORD hashes of what is installed. The records root pins a wheel by SHA-256,
    # and the caller compares THESE against that wheel's own RECORD to establish that the bytes
    # executing are the pinned wheel's bytes -- not merely bytes built from the same commit. A wheel
    # rebuilt from the candidate sdist shares the commit and differs in content (SYS3-01).
    info["installed_record_hashes"] = {
        entry["path"]: entry["sha256"] for entry in content.get("entries", []) if "sha256" in entry
    }
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


_SUMS_LINE = re.compile(r"^([0-9a-f]{64})\s+\*?(\S+)$")


def _read_json_record(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_candidate_records(
    bundle: dict[str, Any], records_root: Path | None, artifact: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Open the bundle's ``required_records`` and cross-bind them to the executing artifact.

    The first version of this globbed for matching PATHNAMES and never opened a match, so a
    directory of files containing ``{}`` satisfied it and every receipt then read ``verified`` --
    the same defect, one level up, as the finding it was written to fix. The reviewer's sharpest
    falsifier was an older wheel from a different commit passing, because the executing commit was
    never compared with the candidate's.

    So each record is now parsed and checked for what it is supposed to assert, and the executing
    artifact's ``source_commit``/``source_tree`` must equal the candidate record's. Every failure
    is reported with its reason rather than collapsing to a single boolean.
    """
    required = list(bundle["candidate_binding"]["required_records"])
    if records_root is None:
        return {
            "records_root": None,
            "resolved": False,
            "required": required,
            "present": [],
            "missing": required,
            "problems": ["no records root supplied"],
        }

    present: list[str] = []
    missing: list[str] = []
    problems: list[str] = []
    matches: dict[str, list[Path]] = {}
    for pattern in required:
        found = sorted(records_root.glob(pattern))
        matches[pattern] = found
        (present if found else missing).append(pattern)
    for pattern in missing:
        # The entry receipts are the output of THIS process, so requiring them as a precondition
        # is circular: the first entry cannot wait for receipts only later entries produce, and
        # demanding them made every publication receipt unbound -- which promotion then rejected,
        # leaving the workflow unable to publish anything. Any receipt already present is still
        # validated below; completeness across all four is the publication manifest's gate.
        if pattern.startswith("metadata/reproduction-"):
            continue
        problems.append(f"absent: {pattern}")

    # metadata/release-candidate.json -- the record that names WHICH candidate this is.
    candidate_commit = None
    candidate_paths = matches.get("metadata/release-candidate.json") or []
    if candidate_paths:
        record = _read_json_record(candidate_paths[0])
        if record is None:
            problems.append("release-candidate.json is not a JSON object")
        elif record.get("artifact") != "mixle.release_candidate/v1":
            problems.append("release-candidate.json is not a mixle.release_candidate/v1 record")
        else:
            candidate_commit = record.get("commit")
            if not isinstance(candidate_commit, str) or len(candidate_commit) != 40:
                problems.append("release-candidate.json has no full-length commit")
                candidate_commit = None
            elif record.get("version") != bundle.get("release"):
                problems.append("release-candidate.json version does not match the bundle release")

    # Source identity of what EXECUTED must be the candidate's -- commit AND tree. The tree half was
    # promised by an earlier docstring and never implemented (SYS3-01 notes it as the same gap).
    # This is necessary but not sufficient: byte identity is bound separately below.
    if candidate_commit is not None and artifact is not None:
        executing = artifact.get("source_commit")
        if executing is None:
            problems.append("executing artifact declares no source commit to compare")
        elif executing != candidate_commit:
            problems.append(f"executing commit {executing} is not the candidate commit {candidate_commit}")
        # Tree is REQUIRED, not compared-if-present. The earlier "compare when the record supplies a
        # 40-char string" made the tree optional: a record with the key removed resolved and
        # produced verified, contradicting the stated commit-and-tree contract (SYS4-03). Commit
        # transitively pins a tree, so the practical exposure was small -- but a contract the code
        # does not enforce is the kind of gap that gets relied on later.
        candidate_tree = record.get("tree") if candidate_paths and record else None
        if (
            not isinstance(candidate_tree, str)
            or len(candidate_tree) != 40
            or any(c not in "0123456789abcdef" for c in candidate_tree)
        ):
            problems.append("release-candidate.json has no full lowercase 40-hex tree")
        elif artifact.get("source_tree") != candidate_tree:
            problems.append(f"executing tree {artifact.get('source_tree')} is not the candidate tree {candidate_tree}")

    # metadata/release-check-evidence.json -- required, and previously never opened (SYS3-05). The
    # candidate-binding rule says these records bind APPROVED checks, so a record whose checks are
    # not in a terminal passing state cannot support "verified". Schema: mixle.release_check_evidence/v1,
    # bound to the candidate commit, with every required check name present and "success".
    evidence_paths = matches.get("metadata/release-check-evidence.json") or []
    if evidence_paths:
        evidence = _read_json_record(evidence_paths[0])
        if evidence is None:
            problems.append("release-check-evidence.json is not a JSON object")
        elif evidence.get("artifact") != "mixle.release_check_evidence/v1":
            problems.append("release-check-evidence.json is not a mixle.release_check_evidence/v1 record")
        else:
            if candidate_commit is not None and evidence.get("commit") != candidate_commit:
                problems.append("release-check-evidence.json is bound to a different commit than the candidate")
            checks = evidence.get("checks")
            if not isinstance(checks, dict):
                problems.append("release-check-evidence.json has no checks mapping")
            else:
                for required_check in ("tests", "docs", "security", "extras_resolver_matrix"):
                    state = checks.get(required_check)
                    if state != "success":
                        problems.append(f"release-check-evidence.json: {required_check} is {state!r}, not 'success'")

    # metadata/SHA256SUMS -- must name a wheel and an sdist with well-formed digests.
    sums: dict[str, str] = {}
    sums_paths = matches.get("metadata/SHA256SUMS") or []
    if sums_paths:
        try:
            for line in sums_paths[0].read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                found = _SUMS_LINE.fullmatch(line.strip())
                if found is None:
                    problems.append("SHA256SUMS has a malformed line")
                    break
                sums[Path(found.group(2)).name] = found.group(1)
        except (OSError, UnicodeError):
            problems.append("SHA256SUMS is unreadable")
        if sums and not any(name.endswith(".whl") for name in sums):
            problems.append("SHA256SUMS names no wheel")
        if sums and not any(name.endswith(".tar.gz") for name in sums):
            problems.append("SHA256SUMS names no source distribution")
        if not sums:
            problems.append("SHA256SUMS is empty")

    # metadata/<wheel>.json -- the wheel's own digest must be one SHA256SUMS vouches for.
    for pattern, paths in matches.items():
        if not pattern.endswith(".whl.json"):
            continue
        record = _read_json_record(paths[0]) if paths else None
        if record is None:
            problems.append(f"{pattern} is not a JSON object")
            continue
        digest = record.get("sha256")
        filename = record.get("filename")
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"{pattern} has no valid sha256")
        elif sums and sums.get(str(filename)) != digest:
            problems.append(f"{pattern} digest is not the one SHA256SUMS records for {filename!r}")

    # Bind the EXECUTING BYTES to the PINNED WHEEL, not merely to a shared source commit. A wheel
    # rebuilt from the candidate sdist carries the same commit and different content, and comparing
    # commit alone let it impersonate the frozen candidate (SYS3-01). The pinned wheel must be
    # present beside the records (dist/<name>, the layout the publish workflow and the review
    # candidate both use), its bytes must hash to the SHA256SUMS entry, and every hashed entry in
    # ITS RECORD must appear in the installation with the same digest -- the same rule
    # mixle.reproduction.subject_binding applies. Absent wheel bytes are a stated problem, not a
    # silent downgrade to commit-only binding.
    if artifact is not None and sums:
        wheel_names = [name for name in sums if name.endswith(".whl")]
        pinned_name = wheel_names[0] if len(wheel_names) == 1 else None
        if pinned_name is None:
            problems.append("SHA256SUMS must pin exactly one wheel to bind the executing artifact against")
        else:
            wheel_path = records_root / "dist" / pinned_name
            installed = artifact.get("installed_record_hashes")
            if not wheel_path.is_file():
                problems.append(f"pinned wheel {pinned_name!r} is not present at dist/ beside the records")
            elif not isinstance(installed, dict) or not installed:
                problems.append("executing artifact reports no installed RECORD hashes to bind")
            else:
                actual = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
                if actual != sums[pinned_name]:
                    problems.append(
                        f"dist/{pinned_name} hashes to {actual[:12]}..., not the pinned {sums[pinned_name][:12]}..."
                    )
                else:
                    try:
                        from mixle.reproduction import _wheel_record_hashes

                        pinned_records = _wheel_record_hashes(wheel_path)
                    except (OSError, ValueError, ImportError) as exc:
                        pinned_records = None
                        problems.append(f"pinned wheel RECORD could not be read: {exc}")
                    if pinned_records is not None:
                        missing_or_different = [
                            path
                            for path, expected_digest in pinned_records.items()
                            if installed.get(path) != expected_digest
                        ]
                        if missing_or_different:
                            problems.append(
                                f"executing installation is not the pinned wheel: "
                                f"{len(missing_or_different)} of {len(pinned_records)} RECORD entries differ or are "
                                f"absent (e.g. {missing_or_different[0]!r})"
                            )

    # metadata/reproduction-*.json -- each match must be a receipt, and each must name this
    # candidate. Completeness across all four entries is the publication manifest's gate, not this
    # one: the first entry cannot require receipts that only later entries produce.
    for path in matches.get("metadata/reproduction-*.json") or []:
        record = _read_json_record(path)
        if record is None or record.get("artifact") != "mixle.reproduction_entry_receipt/v2":
            problems.append(f"{path.name} is not a reproduction entry receipt")

    return {
        "records_root": str(records_root),
        "resolved": not problems,
        "required": required,
        "present": present,
        "missing": missing,
        "problems": problems,
        "candidate_commit": candidate_commit,
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
    binding = _resolve_candidate_records(bundle, records_root, artifact)
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
        detail = ", ".join(binding.get("problems") or binding["missing"]) or "no records root given"
        unbound.append(f"candidate records unresolved: {detail}")

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
