"""Artifact and hosted release evidence fails closed at the reviewed boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(script: str):
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BoundSbomTest(unittest.TestCase):
    def test_sbom_requires_mixle_and_exact_artifact_identity(self):
        binder = _load("bind_sbom.py")
        raw = {"bomFormat": "CycloneDX", "components": [{"name": "mixle"}, {"name": "numpy"}]}
        wheel = {"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": "a" * 64, "size_bytes": 10}
        result = binder.bind(raw, wheel, "b" * 40)
        self.assertEqual(result["inventory_scope"], "isolated-wheel-environment")
        with self.assertRaises(ValueError):
            binder.bind({"bomFormat": "CycloneDX", "components": [{"name": "pip-audit"}]}, wheel, "b" * 40)


class PublishedArtifactIdentityTest(unittest.TestCase):
    def test_exact_file_set_and_hashes_are_required(self):
        verifier = _load("verify_published_artifacts.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "mixle-0.8.0-py3-none-any.whl"
            sdist = root / "mixle-0.8.0.tar.gz"
            wheel.write_bytes(b"wheel")
            sdist.write_bytes(b"sdist")
            sums = root.parent / f"{root.name}-SHA256SUMS"
            sums.write_text(
                f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n"
                f"{hashlib.sha256(sdist.read_bytes()).hexdigest()}  {sdist.name}\n",
                encoding="utf-8",
            )
            self.assertEqual(set(verifier.verify(root, sums)), {wheel.name, sdist.name})
            wheel.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                verifier.verify(root, sums)
            sums.unlink()


class EnvironmentReceiptTest(unittest.TestCase):
    def test_profile_set_must_be_complete_and_exact_sha(self):
        receipt = _load("environment_receipt.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"profiles": ["a", "b"]}), encoding="utf-8")
            receipts = root / "receipts"
            receipts.mkdir()
            for profile in ("a", "b"):
                (receipts / f"{profile}.json").write_text(
                    json.dumps(
                        {
                            "artifact": "mixle.environment_receipt/v1",
                            "profile": profile,
                            "candidate_commit": "a" * 40,
                            "resolved_dependencies": ["mixle==0.8.0"],
                            "passed": True,
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertEqual(receipt.verify(receipts, policy, "a" * 40)["profiles"], ["a", "b"])
            (receipts / "b.json").unlink()
            with self.assertRaisesRegex(ValueError, "profiles differ"):
                receipt.verify(receipts, policy, "a" * 40)


class DecisionReviewTest(unittest.TestCase):
    def test_pending_decisions_are_release_blockers(self):
        reviews = _load("verify_decision_reviews.py")
        accepted = (
            "## D-0001 · one\n"
            "- **Reviewer:** accepted — Jane Reviewer; 2026-07-27; PR #1\n"
            "## D-0002 · two\n"
            "- **Reviewer:** superseded — D-0003\n"
        )
        self.assertEqual(reviews.unresolved_reviews(accepted), [])
        self.assertEqual(
            reviews.unresolved_reviews("## D-0001 · one\n- **Reviewer:** pending — PR #1\n"),
            ["D-0001"],
        )

    def test_current_pending_reviews_are_explicitly_external_and_required(self):
        checklist = (ROOT / "release-checklists" / "0.8.0.md").read_text(encoding="utf-8")
        required = (ROOT / ".github" / "release-required-checks.txt").read_text(encoding="utf-8")
        self.assertIn("Decision review acceptance | `EXTERNAL`", checklist)
        self.assertIn("release decisions / accepted review", required.splitlines())


class HostedWorkflowContractTest(unittest.TestCase):
    def test_security_and_real_data_evidence_are_fail_closed(self):
        security = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
        tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        extras = (ROOT / ".github" / "workflows" / "extras-matrix.yml").read_text(encoding="utf-8")
        post = (ROOT / ".github" / "workflows" / "post-publish-verify.yml").read_text(encoding="utf-8")
        self.assertIn('--path "$SITE"', security)
        self.assertNotIn("continue-on-error: true", security)
        self.assertEqual(tests.count("scripts/run_required_pytest.py"), 2)
        self.assertNotIn("if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'", tests)
        self.assertIn("extras matrix / exact candidate", extras)
        self.assertIn("verify_published_artifacts.py", post)
        self.assertIn("public-artifacts/*.whl", post)


if __name__ == "__main__":
    unittest.main()
