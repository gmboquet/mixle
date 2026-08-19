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


def _check_evidence_record(path: Path, commit: str, required: list[str], label: str) -> tuple[dict[str, Any], str]:
    """Read a check-evidence record and require the generator's shape for THIS commit."""
    record = _read(path)
    if record.get("artifact") != "mixle.release_check_evidence/v1":
        raise ValueError(f"{label} is not a mixle.release_check_evidence/v1 record")
    if record.get("commit") != commit:
        raise ValueError(f"{label} is bound to {record.get('commit')!r}, not candidate {commit!r}")
    checks = record.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(required):
        raise ValueError(f"{label} does not select exactly the required checks")
    if record.get("publication_authorized") is not True:
        raise ValueError(f"{label} does not carry publication_authorized=true")
    return record, hashlib.sha256(path.read_bytes()).hexdigest()


def _resolver():
    """The receipt resolver's attestation verifier -- one implementation of the gh binding, not two."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_run_repro_entry_for_manifest", ROOT / "scripts" / "run_repro_entry.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("scripts/run_repro_entry.py is not available beside this checkout")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_manifest(
    *,
    bundle_path: Path,
    candidate_path: Path,
    wheel_metadata_path: Path,
    receipt_paths: list[Path],
    check_evidence_path: Path,
    check_evidence_bundle_path: Path,
    live_check_evidence_path: Path,
    live_check_evidence_bundle_path: Path,
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

    # The manifest is the publication's statement that its receipts were bound to APPROVED checks.
    # Every receipt must have been resolved against one retained, ATTESTED check-evidence record --
    # the receipts carry its digest -- and a SECOND attested record, produced by a different run of
    # a signing workflow (in publish.yml, the promote phase's own verify-candidate step over the
    # GitHub API, minutes before this runs), must select the same check runs. Two independent
    # workflow runs having attested approval of the same check-run set is what "the approval basis
    # still holds" means here; a record of the right shape that was never issued satisfies neither
    # (SYS5-01), and one file passed twice fails the different-invocation requirement.
    required = bundle.get("candidate_binding", {}).get("required_checks")
    if not isinstance(required, list) or not required:
        raise ValueError("bundle candidate binding names no required checks")
    retained, retained_digest = _check_evidence_record(check_evidence_path, commit, required, "retained check evidence")
    live, live_digest = _check_evidence_record(live_check_evidence_path, commit, required, "live check evidence")
    # Both records approve every required check for this commit (that is what _check_evidence_record
    # established). They need NOT cite the same check-run ids: rerunning any job of a tests.yml run
    # re-materializes every job's check-run id, so two attested records of the same approval can
    # legitimately differ in ids -- requiring identity made the manifest unbuildable after any rerun.
    # What matters is that approval holds at both times, attested by two different runs; which names
    # were re-run is recorded, not hidden.
    reselected = sorted(name for name in required if live["checks"].get(name) != retained["checks"].get(name))
    # A receipt's `check_evidence.attested` is the receipt's own word. The manifest establishes the
    # facts itself, through the same gh binding the resolver uses: both records' attestations must
    # verify for THIS commit, and they must come from different workflow run invocations.
    resolver = _resolver()
    contract = resolver._attestation_contract(bundle)
    attestation = resolver._verify_check_evidence_attestation(
        check_evidence_path, check_evidence_bundle_path, contract, commit
    )
    if attestation.get("record_sha256") != retained_digest:
        raise ValueError("the verified attestation does not name the retained check-evidence record")
    live_attestation = resolver._verify_check_evidence_attestation(
        live_check_evidence_path, live_check_evidence_bundle_path, contract, commit
    )
    if live_attestation.get("record_sha256") != live_digest:
        raise ValueError("the verified attestation does not name the live check-evidence record")
    invocations = (attestation.get("run_invocation_uri"), live_attestation.get("run_invocation_uri"))
    if not all(isinstance(uri, str) and uri for uri in invocations) or invocations[0] == invocations[1]:
        raise ValueError(
            "the live check-evidence record must be attested by a different workflow run than the retained one; "
            f"got {invocations!r}"
        )

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
        # `claim_status` is a string a receipt asserts about itself. Trusting it alone let receipts
        # that carried no binding evidence at all -- and receipts produced by an artifact from a
        # different commit -- into a signed manifest. Re-derive the claim here from the evidence
        # the receipt is required to carry, and bind it to THIS candidate.
        artifact = receipt.get("executing_artifact")
        binding = receipt.get("candidate_binding")
        if not isinstance(artifact, dict) or not isinstance(binding, dict):
            raise ValueError(f"{entry_id}: receipt carries no executing-artifact or candidate-binding evidence")
        if artifact.get("installed_distribution") is not True:
            raise ValueError(f"{entry_id}: receipt was not produced by an installed distribution")
        if artifact.get("installed_content_verified") is not True:
            raise ValueError(f"{entry_id}: installed content did not verify when the receipt was produced")
        if binding.get("resolved") is not True or binding.get("problems"):
            raise ValueError(f"{entry_id}: candidate records were not resolved for this receipt")
        if artifact.get("source_commit") != commit:
            raise ValueError(
                f"{entry_id}: receipt was produced by {artifact.get('source_commit')!r}, not candidate {commit!r}"
            )
        if binding.get("candidate_commit") not in (None, commit):
            raise ValueError(f"{entry_id}: receipt is bound to a different candidate than this manifest")
        evidence = binding.get("check_evidence")
        if not isinstance(evidence, dict) or evidence.get("attested") is not True:
            raise ValueError(f"{entry_id}: receipt was not resolved against an attested check-evidence record")
        if evidence.get("record_sha256") != retained_digest:
            raise ValueError(
                f"{entry_id}: receipt was resolved against a different check-evidence record than the retained one"
            )
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
        "check_evidence": {
            "sha256": retained_digest,
            "checks": len(retained["checks"]),
            "attested": True,
            "attestation": {
                "signer_workflow": attestation.get("signer_workflow"),
                "source_digest": attestation.get("source_digest"),
                "run_invocation_uri": attestation.get("run_invocation_uri"),
            },
            "live_regeneration": {
                "sha256": live_digest,
                "signer_workflow": live_attestation.get("signer_workflow"),
                "run_invocation_uri": live_attestation.get("run_invocation_uri"),
                "approves_every_required_check": True,
                "selection_identical": not reselected,
                "reselected_checks": reselected,
            },
        },
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--wheel-metadata", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument(
        "--check-evidence",
        type=Path,
        required=True,
        help="the retained, attested check-evidence record every receipt was resolved against",
    )
    parser.add_argument(
        "--check-evidence-bundle",
        type=Path,
        required=True,
        help="the Sigstore bundle attesting the retained record (release-check-evidence.sigstore.json)",
    )
    parser.add_argument(
        "--live-check-evidence",
        type=Path,
        required=True,
        help="a check-evidence record a signing workflow produced from the GitHub API just now",
    )
    parser.add_argument(
        "--live-check-evidence-bundle",
        type=Path,
        required=True,
        help="the Sigstore bundle attesting that live record (from a different workflow run)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(
            bundle_path=args.bundle,
            candidate_path=args.candidate,
            wheel_metadata_path=args.wheel_metadata,
            receipt_paths=args.receipt,
            check_evidence_path=args.check_evidence,
            check_evidence_bundle_path=args.check_evidence_bundle,
            live_check_evidence_path=args.live_check_evidence,
            live_check_evidence_bundle_path=args.live_check_evidence_bundle,
        )
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
