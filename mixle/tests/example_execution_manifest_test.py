"""Release example evidence is complete and bound to one exact candidate wheel."""

from __future__ import annotations

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
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )
            self.receipts.append(receipt)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, receipts=None):
        return self.builder.build_manifest(
            bundle_path=BUNDLE,
            candidate_path=self.candidate,
            wheel_metadata_path=self.wheel,
            receipt_paths=self.receipts if receipts is None else receipts,
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


if __name__ == "__main__":
    unittest.main()
