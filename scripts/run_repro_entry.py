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
    _required_check_names(bundle)
    _attestation_contract(bundle)
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


def _required_check_names(bundle: dict[str, Any]) -> list[str]:
    """Return the bundle's embedded publication policy: the exact check-run names approval requires."""
    names = bundle.get("candidate_binding", {}).get("required_checks")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("bundle candidate binding must name the required checks (unique, nonempty strings)")
    return list(names)


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# GitHub grammar: an owner is alphanumerics/hyphens not starting with a hyphen; a repository name is
# alphanumerics, ".", "_", "-" and is never "." or ".."
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*/(?!\.\.?$)[A-Za-z0-9_.-]+$")
_WORKFLOW_FILE = re.compile(r"^\.github/workflows/[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*\.ya?ml$")
_OIDC_ISSUER = "https://token.actions.githubusercontent.com"


def _attestation_contract(bundle: dict[str, Any]) -> dict[str, Any]:
    """The bundle's statement of who may produce the check-evidence record and how that is verified."""
    binding = bundle.get("candidate_binding", {})
    repository = binding.get("repository")
    contract = binding.get("check_evidence_attestation")
    if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
        raise ValueError("bundle candidate binding must name the GitHub repository (owner/repo)")
    if not isinstance(contract, dict):
        raise ValueError("bundle candidate binding must describe the check-evidence attestation")
    workflows = contract.get("signer_workflows")
    # exact workflow files, in a grammar with no regex metacharacters but '.', so the identity
    # pattern built from them below is exactly the intended alternatives (a prefix-only entry or a
    # name with '+' would widen the SubjectAlternativeName the verifier accepts)
    if (
        not isinstance(workflows, list)
        or not workflows
        or any(not isinstance(w, str) or not _WORKFLOW_FILE.fullmatch(w) for w in workflows)
        or len(set(workflows)) != len(workflows)
    ):
        raise ValueError("check-evidence attestation must name distinct workflow files .github/workflows/<name>.yml")
    for key in ("predicate_type", "bundle_record", "check_runs_record"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"check-evidence attestation must state {key}")
    return {"repository": repository, **contract}


def _job_url_pattern(repository: str) -> re.Pattern[str]:
    # GitHub Actions check runs point at the job page of their run, and the job segment IS the
    # check-run id (verified 68/68 on the real candidate). The reviewer's forged record used
    # https://example.invalid/actions/runs/... and passed the old substring test (SYS5-01).
    return re.compile(rf"^https://github\.com/{re.escape(repository)}/actions/runs/([1-9][0-9]*)/job/([1-9][0-9]*)$")


def _run_gh(command: list[str]) -> subprocess.CompletedProcess[str]:
    """The one call into the gh CLI; a module-local seam so tests replace it without touching the
    process-wide subprocess module (which poisoned unrelated tests once)."""
    return subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)


def _verify_check_evidence_attestation(
    record_path: Path, bundle_path: Path, contract: dict[str, Any], commit: str
) -> dict[str, Any]:
    """Verify with the gh CLI that GitHub signed ``record_path`` for THIS repository, from one of the
    bundle's signing workflows, at the candidate commit. Returns what was verified, or raises
    ValueError with the reason. Nothing else can turn a record's shape into approval evidence.

    The trust anchor is Sigstore's public-good root as gh obtains it (TUF, cached after first use):
    a "trusted root" shipped beside the record would be chosen by whoever ships the record, and gh
    would consult nothing else -- an attacker with their own CA then verifies. So no
    ``--custom-trusted-root`` is ever passed; a host that has never been online cannot verify, and
    the receipt says so rather than trusting a supplied root. ``--cert-identity-regex`` binds the
    signing identity to this repository AND its named workflows in one expression (gh refuses it
    together with ``--signer-repo``); ``--source-digest`` binds to the candidate commit;
    ``--predicate-type`` to this record kind; ``--cert-oidc-issuer`` to GitHub's OIDC issuer.
    """
    escaped = [w.replace(".", "\\.") for w in contract["signer_workflows"]]  # grammar allows no other metachar
    identity = (
        rf"^https://github\.com/{contract['repository'].replace('.', chr(92) + '.')}/(" + "|".join(escaped) + ")@"
    )
    command = [
        "gh",
        "attestation",
        "verify",
        str(record_path),
        "--bundle",
        str(bundle_path),
        "--repo",
        contract["repository"],
        "--cert-identity-regex",
        identity,
        "--cert-oidc-issuer",
        _OIDC_ISSUER,
        "--source-digest",
        commit,
        "--predicate-type",
        contract["predicate_type"],
        "--deny-self-hosted-runners",
        "--format",
        "json",
    ]
    try:
        completed = _run_gh(command)
    except FileNotFoundError as exc:
        raise ValueError("gh CLI is unavailable, so the check-evidence attestation cannot be verified") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("gh attestation verify did not finish within 120 seconds") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise ValueError(f"check-evidence attestation did not verify: {detail[-1] if detail else 'no output'}")
    try:
        results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("gh attestation verify returned no JSON result") from exc
    digest = _sha256(record_path)
    verified = []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        statement = (result.get("verificationResult") or {}).get("statement") or {}
        subjects = statement.get("subject") or []
        if any(
            (subject.get("digest") or {}).get("sha256") == digest for subject in subjects if isinstance(subject, dict)
        ):
            if statement.get("predicateType") == contract["predicate_type"]:
                verified.append(result)
    if not verified:
        raise ValueError("no verified attestation names this record's digest with the expected predicate type")
    certificate = ((verified[0].get("verificationResult") or {}).get("signature") or {}).get("certificate") or {}
    return {
        "attested": True,
        "record_sha256": digest,
        "signer_workflow": certificate.get("buildSignerURI") or certificate.get("subjectAlternativeName"),
        "source_repository": certificate.get("sourceRepositoryURI"),
        "source_digest": certificate.get("sourceRepositoryDigest"),
        "run_invocation_uri": certificate.get("runInvocationURI"),
        "verifier": "gh attestation verify --bundle (Sigstore public-good root via gh)",
    }


def _rederive_selection(payload_path: Path, required: list[str], commit: str, repository: str) -> dict[str, Any]:
    """Run the generator's own selection over the retained check-runs payload; the bundle closes over
    the generator, so this is the same code the record claims to have come from."""
    import importlib.util

    generator_path = ROOT / "scripts" / "verify_required_checks.py"
    spec = importlib.util.spec_from_file_location("_verify_required_checks_rederive", generator_path)
    if spec is None or spec.loader is None:
        raise ValueError("the check-evidence generator is not available beside this checkout")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return module.verify_required_checks(payload, tuple(required), commit, repository=repository)


def _check_evidence_problems(
    bundle: dict[str, Any],
    evidence: dict[str, Any],
    *,
    evidence_path: Path | None = None,
    records_root: Path | None = None,
    candidate_commit: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Report every way ``evidence`` fails to be THE generator's attested record approving every required
    check for this candidate, and return what was verified about it.

    Layers, each necessary: (1) the generator's shape -- every required name present as a selected
    check-run record with a distinct id and a job URL of this repository whose job segment is the id;
    (2) provenance of the selection -- the record commits to the digest of the check-runs payload it was
    derived from, that payload is retained beside it, and running the generator over it reproduces the
    same selection; (3) authenticity -- a Sigstore attestation, signed through GitHub's OIDC identity of
    one of the bundle's signing workflows at the candidate commit, names this record's digest, verified
    offline against the retained trusted root. A hand-written record can satisfy (1) and, with a
    hand-written payload, (2); it cannot satisfy (3) (SYS5-01).
    """
    problems: list[str] = []
    info: dict[str, Any] = {"attested": False}
    try:
        required = _required_check_names(bundle)
        contract = _attestation_contract(bundle)
    except ValueError as exc:
        return [str(exc)], info
    checks = evidence.get("checks")
    if not isinstance(checks, dict):
        return ["release-check-evidence.json has no checks mapping"], info
    job_url = _job_url_pattern(contract["repository"])
    seen_ids: dict[int, str] = {}
    for name in required:
        selected = checks.get(name)
        if selected is None:
            problems.append(f"release-check-evidence.json: required check {name!r} is absent")
            continue
        if not isinstance(selected, dict):
            problems.append(
                f"release-check-evidence.json: required check {name!r} is {selected!r}, "
                "not a selected check-run record (check_run_id + details_url)"
            )
            continue
        run_id = selected.get("check_run_id")
        details_url = selected.get("details_url")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
            problems.append(f"release-check-evidence.json: required check {name!r} has no valid check_run_id")
            continue
        if run_id in seen_ids:
            problems.append(
                f"release-check-evidence.json: required checks {seen_ids[run_id]!r} and {name!r} "
                f"cite the same check run {run_id}"
            )
        else:
            seen_ids[run_id] = name
        found = job_url.fullmatch(details_url) if isinstance(details_url, str) else None
        if found is None or int(found.group(2)) != run_id:
            problems.append(
                f"release-check-evidence.json: required check {name!r} does not point at a "
                f"{contract['repository']} Actions job page for check run {run_id}"
            )
    if evidence.get("publication_authorized") is not True:
        problems.append("release-check-evidence.json does not carry the generator's publication_authorized=true")

    source = evidence.get("source")
    payload_digest = source.get("check_runs_sha256") if isinstance(source, dict) else None
    if not isinstance(source, dict) or source.get("repository") != contract["repository"]:
        problems.append(f"release-check-evidence.json does not name {contract['repository']} as its source repository")
    if not isinstance(payload_digest, str) or not _HEX64.fullmatch(payload_digest):
        problems.append(
            "release-check-evidence.json does not commit to the digest of the check-runs payload it came from"
        )
        payload_digest = None

    if records_root is None or evidence_path is None or candidate_commit is None:
        problems.append("check evidence cannot be authenticated without the records root and candidate commit")
        return problems, info

    payload_path = records_root / contract["check_runs_record"]
    if not payload_path.is_file():
        problems.append(f"absent: {contract['check_runs_record']} (the check-runs payload the record was derived from)")
    elif payload_digest is not None:
        try:
            actual = _sha256(payload_path)
        except OSError as exc:
            actual = None
            problems.append(f"{contract['check_runs_record']} is unreadable: {exc}")
        if actual is None:
            pass
        elif actual != payload_digest:
            problems.append(
                f"{contract['check_runs_record']} hashes to {actual[:12]}..., not the {payload_digest[:12]}... "
                "the record commits to"
            )
        else:
            try:
                rederived = _rederive_selection(payload_path, required, candidate_commit, contract["repository"])
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                rederived = None
                problems.append(f"the retained check-runs payload does not approve this candidate: {exc}")
            if rederived is not None and rederived != checks:
                problems.append(
                    "release-check-evidence.json checks do not re-derive from the retained check-runs payload"
                )
            info["check_runs_sha256"] = actual

    bundle_path = records_root / contract["bundle_record"]
    if not bundle_path.is_file():
        problems.append(f"absent: {contract['bundle_record']} (the attestation bundle)")
    else:
        try:
            info.update(_verify_check_evidence_attestation(evidence_path, bundle_path, contract, candidate_commit))
        except (ValueError, OSError) as exc:
            problems.append(str(exc))
    try:
        info.setdefault("record_sha256", _sha256(evidence_path))
    except OSError as exc:
        problems.append(f"release-check-evidence.json is unreadable: {exc}")
    return problems, info


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
            "check_evidence": {"attested": False},
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
    # not in a terminal passing state cannot support "verified".
    #
    # The record has exactly one producer: scripts/verify_required_checks.py, run by publish.yml
    # over the GitHub check-runs of the candidate SHA. It writes nothing at all unless every check
    # named in .github/release-required-checks.txt is completed/success, and what it writes is
    #   {"artifact": "mixle.release_check_evidence/v1", "commit": <sha>,
    #    "checks": {<exact check-run name>: {"check_run_id": <int>, "details_url": <Actions URL>}, ...},
    #    "publication_authorized": true}
    # The first SYS3-05 repair validated a different, invented shape ({"tests": "success", ...}) that
    # only the hand-written review-candidate stubs ever had -- so it accepted the stubs and would
    # have rejected every real record, leaving publication unable to bind any receipt. The
    # required names now come from the bundle, which embeds the same policy file the generator
    # reads (build_repro_bundle.py parses it with the generator's own parser), and each entry is
    # checked for the generator's selected-check shape. This is necessary, not sufficient: the
    # check runs themselves are only verifiable against GitHub, which the generator does.
    check_evidence: dict[str, Any] = {"attested": False}
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
            evidence_problems, check_evidence = _check_evidence_problems(
                bundle,
                evidence,
                evidence_path=evidence_paths[0],
                records_root=records_root,
                candidate_commit=candidate_commit,
            )
            problems.extend(evidence_problems)

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
                try:
                    actual = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
                except OSError as exc:
                    actual = None
                    problems.append(f"dist/{pinned_name} is unreadable: {exc}")
                if actual is None:
                    pass
                elif actual != sums[pinned_name]:
                    problems.append(
                        f"dist/{pinned_name} hashes to {actual[:12]}..., not the pinned {sums[pinned_name][:12]}..."
                    )
                else:
                    try:
                        from mixle.reproduction import _wheel_record_hashes

                        pinned_records = _wheel_record_hashes(wheel_path)
                    except Exception as exc:  # noqa: BLE001 -- the archive is untrusted input; see below
                        # A pinned wheel with a truncated central directory escaped as an uncaught
                        # zipfile.BadZipFile traceback (exit 1) instead of an unbound receipt with its
                        # reason (SYS5-03); the first repair enumerated exception classes and missed
                        # zlib.error, lzma.LZMAError, NotImplementedError (unsupported method / version /
                        # flags) and RuntimeError (encrypted member). Reading an untrusted archive can
                        # raise anything its decompressors raise, and every one of them means the same
                        # thing here: the pinned bytes cannot be read, so the binding fails closed with
                        # the exception named. Nothing is swallowed -- the type and message are the
                        # problem string -- and nothing else in this branch can raise but the read.
                        pinned_records = None
                        problems.append(f"pinned wheel RECORD could not be read: {type(exc).__name__}: {exc}")
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
        # What was established about the check-evidence record: its digest, whether GitHub's
        # attestation for it verified (and by which workflow, at which commit), and the digest of the
        # check-runs payload it re-derives from. The manifest builder binds on these.
        "check_evidence": check_evidence,
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
