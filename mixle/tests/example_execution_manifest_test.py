"""Release example evidence is complete and bound to one exact candidate wheel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_example_execution_manifest.py"
BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"


def _module():
    spec = importlib.util.spec_from_file_location("build_example_execution_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExampleExecutionManifestTest(unittest.TestCase):
    def setUp(self):
        self.builder = _module()
        self.bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.candidate = root / "candidate.json"
        self.candidate.write_text(
            json.dumps(
                {
                    "artifact": "mixle.release_candidate/v1",
                    "commit": "a" * 40,
                    "tag": "v0.8.0",
                    "version": "0.8.0",
                    "workflow_run": "123",
                }
            ),
            encoding="utf-8",
        )
        self.wheel = root / "wheel.json"
        self.wheel.write_text(
            json.dumps(
                {
                    "filename": "mixle-0.8.0-py3-none-any.whl",
                    "sha256": "b" * 64,
                    "size_bytes": 100,
                }
            ),
            encoding="utf-8",
        )
        # the retained, attested check-evidence record every receipt was resolved against, and a
        # live regeneration the caller made moments before building the manifest (SYS5-01)
        required = self.bundle["candidate_binding"]["required_checks"]
        evidence = {
            "artifact": "mixle.release_check_evidence/v1",
            "commit": "a" * 40,
            "checks": {
                name: {
                    "check_run_id": 1000 + index,
                    "details_url": f"https://github.com/gmboquet/mixle/actions/runs/{500 + index}/job/{1000 + index}",
                }
                for index, name in enumerate(required)
            },
            "publication_authorized": True,
        }
        self.check_evidence = root / "release-check-evidence.json"
        self.check_evidence.write_text(json.dumps(evidence), encoding="utf-8")
        self.live_check_evidence = root / "live-check-evidence.json"
        self.live_check_evidence.write_text(
            json.dumps(evidence, indent=1), encoding="utf-8"
        )  # same selection, other bytes
        self.evidence_digest = hashlib.sha256(self.check_evidence.read_bytes()).hexdigest()
        self.evidence_bundle = root / "release-check-evidence.sigstore.json"
        self.evidence_bundle.write_text('{"fixture": "sigstore bundle"}\n', encoding="utf-8")
        self.live_evidence_bundle = root / "live-check-evidence.sigstore.json"
        self.live_evidence_bundle.write_text('{"fixture": "sigstore bundle 2"}\n', encoding="utf-8")
        # the builder verifies BOTH records' attestations itself through the resolver's gh binding;
        # a unit test cannot mint Sigstore bundles, so the resolver it loads is given a verifier that
        # accepts exactly the two issued digests (as GitHub would) and no other, and reports the
        # retained record as signed by the prepare run and the live one by the promote run
        self.live_evidence_digest = hashlib.sha256(self.live_check_evidence.read_bytes()).hexdigest()
        signed = {
            self.evidence_digest: "https://github.com/gmboquet/mixle/actions/runs/1/attempts/1",
            self.live_evidence_digest: "https://github.com/gmboquet/mixle/actions/runs/2/attempts/1",
        }

        def verify(record_path, bundle_path, contract, commit):
            digest = hashlib.sha256(Path(record_path).read_bytes()).hexdigest()
            if digest not in signed:
                raise ValueError("check-evidence attestation did not verify: no attestation names this record")
            return {
                "attested": True,
                "record_sha256": digest,
                "signer_workflow": "https://github.com/gmboquet/mixle/.github/workflows/publish.yml@refs/tags/v0.8.0",
                "source_digest": commit,
                "run_invocation_uri": signed[digest],
            }

        self.signed = signed
        resolver = self.builder._resolver()
        resolver._verify_check_evidence_attestation = verify
        self.builder._resolver = lambda: resolver
        self.receipts = []
        for entry in self.bundle["entries"]:
            receipt = root / f"{entry['id']}.json"
            receipt.write_text(
                json.dumps(
                    {
                        "artifact": "mixle.reproduction_entry_receipt/v2",
                        "entry": entry["id"],
                        "argv": entry["argv"],
                        "tier": entry["tier"],
                        "duration_seconds": 0.1,
                        "timeout_seconds": entry["timeout_seconds"],
                        "entry_contract_sha256": self.builder._contract_digest(entry),
                        "stdout_sha256": "c" * 64,
                        "validated_output": entry["expected"],
                        "execution_status": "passed",
                        "claim_status": "verified",
                        # The evidence BEHIND claim_status. These fixtures previously omitted both
                        # blocks and still built a signed manifest, because the builder trusted the
                        # status string -- so the test suite could not have caught a receipt that
                        # claimed "verified" while carrying nothing, or one produced by a different
                        # commit entirely.
                        "executing_artifact": {
                            "installed_distribution": True,
                            "installed_content_verified": True,
                            "source_commit": "a" * 40,
                        },
                        "candidate_binding": {
                            "resolved": True,
                            "problems": [],
                            "candidate_commit": "a" * 40,
                            "check_evidence": {"attested": True, "record_sha256": self.evidence_digest},
                        },
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )
            self.receipts.append(receipt)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, receipts=None, *, check_evidence=None, live_check_evidence=None):
        return self.builder.build_manifest(
            bundle_path=BUNDLE,
            candidate_path=self.candidate,
            wheel_metadata_path=self.wheel,
            receipt_paths=self.receipts if receipts is None else receipts,
            check_evidence_path=self.check_evidence if check_evidence is None else check_evidence,
            check_evidence_bundle_path=self.evidence_bundle,
            live_check_evidence_path=self.live_check_evidence if live_check_evidence is None else live_check_evidence,
            live_check_evidence_bundle_path=self.live_evidence_bundle,
        )

    def test_complete_manifest_binds_candidate_wheel_profile_and_every_example(self):
        manifest = self.build()
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["candidate"]["commit"], "a" * 40)
        self.assertEqual(manifest["wheel"]["sha256"], "b" * 64)
        self.assertEqual(
            {example["id"] for example in manifest["examples"]},
            {entry["id"] for entry in self.bundle["entries"]},
        )
        self.assertEqual(manifest["artifact"], "mixle.example_execution_manifest/v2")
        self.assertTrue(all(example["execution_status"] == "passed" for example in manifest["examples"]))
        self.assertTrue(all(example["claim_status"] == "verified" for example in manifest["examples"]))

    def test_exit_success_without_claim_verification_fails_closed(self):
        receipt = json.loads(self.receipts[0].read_text(encoding="utf-8"))
        receipt["claim_status"] = "not_checked"
        self.receipts[0].write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "separately verify"):
            self.build()

    def test_missing_or_wrong_candidate_evidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.build(self.receipts[:-1])
        candidate = json.loads(self.candidate.read_text(encoding="utf-8"))
        candidate["commit"] = "short"
        self.candidate.write_text(json.dumps(candidate), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full lowercase Git SHA"):
            self.build()

    def test_publish_workflow_builds_and_uploads_manifest(self):
        workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
        self.assertIn("build_example_execution_manifest.py", workflow)
        self.assertIn("--out candidate/metadata/example-execution-manifest.json", workflow)
        self.assertIn("candidate/metadata/*.json", workflow)
        self.assertIn("--check-evidence candidate/metadata/release-check-evidence.json", workflow)
        self.assertIn("--live-check-evidence promotion/release-check-evidence.json", workflow)

    def test_manifest_records_the_attested_evidence_it_was_bound_to(self):
        manifest = self.build()
        self.assertEqual(
            manifest["check_evidence"],
            {
                "sha256": self.evidence_digest,
                "checks": len(self.bundle["candidate_binding"]["required_checks"]),
                "attested": True,
                "attestation": {
                    "signer_workflow": "https://github.com/gmboquet/mixle/.github/workflows/publish.yml@refs/tags/v0.8.0",
                    "source_digest": "a" * 40,
                    "run_invocation_uri": "https://github.com/gmboquet/mixle/actions/runs/1/attempts/1",
                },
                "live_regeneration": {
                    "sha256": self.live_evidence_digest,
                    "signer_workflow": "https://github.com/gmboquet/mixle/.github/workflows/publish.yml@refs/tags/v0.8.0",
                    "run_invocation_uri": "https://github.com/gmboquet/mixle/actions/runs/2/attempts/1",
                    "approves_every_required_check": True,
                    "selection_identical": True,
                    "reselected_checks": [],
                },
            },
        )


class ManifestBindsAttestedEvidenceTest(ExampleExecutionManifestTest):
    """SYS5-01: the manifest reused the resolver's ``resolved`` and never asked what the check-evidence
    record was. A hand-authored record with the right shape yielded four verified receipts and a
    complete manifest. The manifest now requires every receipt to have been resolved against ONE
    attested record (its digest travels in the receipt) and that record to select the same check runs
    as a live regeneration the caller just made."""

    def _edit_receipt(self, index, mutate):
        receipt = json.loads(self.receipts[index].read_text(encoding="utf-8"))
        mutate(receipt)
        self.receipts[index].write_text(json.dumps(receipt), encoding="utf-8")

    def test_a_receipt_not_resolved_against_an_attested_record_is_refused(self):
        self._edit_receipt(1, lambda r: r["candidate_binding"].pop("check_evidence"))
        with self.assertRaisesRegex(ValueError, "not resolved against an attested check-evidence record"):
            self.build()
        self._edit_receipt(1, lambda r: r["candidate_binding"].update({"check_evidence": {"attested": False}}))
        with self.assertRaisesRegex(ValueError, "not resolved against an attested check-evidence record"):
            self.build()

    def test_a_receipt_bound_to_a_different_record_than_the_retained_one_is_refused(self):
        self._edit_receipt(2, lambda r: r["candidate_binding"]["check_evidence"].update({"record_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "different check-evidence record than the retained one"):
            self.build()

    def test_the_manifest_verifies_the_retained_records_attestation_itself(self):
        # a retained record GitHub never signed: the receipts may say attested; the manifest checks
        retained = json.loads(self.check_evidence.read_text(encoding="utf-8"))
        self.check_evidence.write_text(
            json.dumps(retained, indent=2), encoding="utf-8"
        )  # same selection, unsigned bytes
        digest = hashlib.sha256(self.check_evidence.read_bytes()).hexdigest()
        for index in range(len(self.receipts)):
            self._edit_receipt(
                index, lambda r: r["candidate_binding"]["check_evidence"].update({"record_sha256": digest})
            )
        with self.assertRaisesRegex(ValueError, "attestation did not verify"):
            self.build()
        workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
        self.assertIn("--check-evidence-bundle candidate/metadata/release-check-evidence.sigstore.json", workflow)
        self.assertIn("--live-check-evidence-bundle promotion/release-check-evidence.sigstore.json", workflow)

    def test_the_live_record_must_be_attested_by_a_different_run_than_the_retained_one(self):
        # one file passed as both retained and live: same digest, same run invocation -> refused
        # (a self-consistent forgery, or an operator short-cut, satisfied the old checks-only compare)
        with self.assertRaisesRegex(ValueError, "different workflow run"):
            self.build(live_check_evidence=self.check_evidence)
        # a live record GitHub never signed -> refused by the live attestation, before the compare
        unsigned = Path(self.temp.name) / "unsigned-live.json"
        unsigned.write_text(
            json.dumps(json.loads(self.live_check_evidence.read_text(encoding="utf-8")), indent=3), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "attestation did not verify"):
            self.build(live_check_evidence=unsigned)
        # two genuinely attested records from different runs with the same selection -> accepted
        self.assertTrue(self.build()["check_evidence"]["live_regeneration"]["selection_identical"])

    def test_a_live_regeneration_with_reselected_check_runs_is_accepted_and_recorded(self):
        # rerunning any job re-materializes every job's check-run id (measured on run 32201448676),
        # so two attested approvals of the same commit legitimately differ in ids; identity was
        # unbuildable after any rerun. The manifest records which names were re-selected.
        live = json.loads(self.live_check_evidence.read_text(encoding="utf-8"))
        first = next(iter(live["checks"]))
        live["checks"][first]["check_run_id"] += 1
        live["checks"][first]["details_url"] = live["checks"][first]["details_url"][:-1] + "9"
        self.live_check_evidence.write_text(json.dumps(live), encoding="utf-8")
        self.signed[hashlib.sha256(self.live_check_evidence.read_bytes()).hexdigest()] = self.signed.pop(
            self.live_evidence_digest
        )
        manifest = self.build()
        self.assertFalse(manifest["check_evidence"]["live_regeneration"]["selection_identical"])
        self.assertEqual(manifest["check_evidence"]["live_regeneration"]["reselected_checks"], [first])
        # but a live record that does not approve every required check is still refused
        del live["checks"][first]
        self.live_check_evidence.write_text(json.dumps(live), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly the required checks"):
            self.build()

    def test_evidence_records_must_be_generator_shaped_for_this_candidate(self):
        retained = json.loads(self.check_evidence.read_text(encoding="utf-8"))
        for corrupt, message in (
            (lambda r: r.update({"commit": "b" * 40}), "bound to"),
            (lambda r: r["checks"].popitem(), "exactly the required checks"),
            (lambda r: r.update({"publication_authorized": "true"}), "publication_authorized"),
            (lambda r: r.update({"artifact": "other"}), "not a mixle.release_check_evidence/v1 record"),
        ):
            damaged = json.loads(json.dumps(retained))
            corrupt(damaged)
            path = Path(self.temp.name) / "damaged.json"
            path.write_text(json.dumps(damaged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                self.build(live_check_evidence=path)


class ManifestRequiresEvidenceNotAssertionTest(ExampleExecutionManifestTest):
    """SYS-03 second pass: the manifest must re-derive the claim, not trust the receipt's word.

    ``claim_status`` is a string a receipt writes about itself. The builder accepted it on faith,
    so a receipt carrying no binding evidence at all -- or one produced by an artifact from a
    different commit -- could be bound into a signed manifest.
    """

    def _rewrite(self, index, mutate):
        receipt = json.loads(self.receipts[index].read_text(encoding="utf-8"))
        mutate(receipt)
        self.receipts[index].write_text(json.dumps(receipt), encoding="utf-8")

    def test_a_receipt_without_binding_evidence_is_refused(self):
        def drop(receipt):
            receipt.pop("executing_artifact", None)
            receipt.pop("candidate_binding", None)

        self._rewrite(0, drop)
        with self.assertRaisesRegex(ValueError, "no executing-artifact or candidate-binding"):
            self.build()

    def test_a_receipt_from_another_commit_is_refused(self):
        self._rewrite(0, lambda r: r["executing_artifact"].update({"source_commit": "b" * 40}))
        with self.assertRaisesRegex(ValueError, "not candidate"):
            self.build()

    def test_a_receipt_whose_records_did_not_resolve_is_refused(self):
        self._rewrite(0, lambda r: r["candidate_binding"].update({"resolved": False}))
        with self.assertRaisesRegex(ValueError, "candidate records were not resolved"):
            self.build()

    def test_a_receipt_with_recorded_binding_problems_is_refused(self):
        self._rewrite(0, lambda r: r["candidate_binding"].update({"problems": ["digest mismatch"]}))
        with self.assertRaisesRegex(ValueError, "candidate records were not resolved"):
            self.build()

    def test_a_receipt_not_produced_by_an_installed_distribution_is_refused(self):
        self._rewrite(0, lambda r: r["executing_artifact"].update({"installed_distribution": False}))
        with self.assertRaisesRegex(ValueError, "not produced by an installed distribution"):
            self.build()


if __name__ == "__main__":
    unittest.main()
