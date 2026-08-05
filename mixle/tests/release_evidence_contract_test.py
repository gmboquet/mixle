"""Artifact and hosted release evidence fails closed at the reviewed boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

# The hard-budget core CI tier installs no optional extras, so this failed there as a bare
# ModuleNotFoundError rather than skipping -- which reads as a broken candidate instead of an
# absent optional dependency.
HAS_YAML = importlib.util.find_spec("yaml") is not None

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
        self.assertEqual(result["inventory_scope"], "isolated-wheel-environment:base")
        self.assertEqual(result["profile"], "base")
        self.assertEqual(binder.bind(raw, wheel, "b" * 40, profile="all")["profile"], "all")
        with self.assertRaises(ValueError):
            binder.bind(raw, wheel, "b" * 40, profile="unknown")
        # An inventory pip-audit vouched for is recorded as audited.
        self.assertTrue(result["mixle_component_audited"])

    def test_an_unpublished_candidate_is_synthesized_and_marked_rather_than_refused(self):
        """Supersedes the old "absent mixle must raise" contract.

        `pip-audit` builds its inventory from packages it can resolve ON PYPI and reports
        "Dependency not found on PyPI and could not be audited: mixle (0.8.0)" otherwise. For an
        unpublished candidate that is structural, so requiring the entry made the SBOM gate
        unsatisfiable for the first release of any package -- exactly when an SBOM matters most.

        The property worth protecting is that the bound SBOM describes the wheel under audit, and
        the wheel's identity comes from the authoritative --wheel-metadata. So the entry is
        synthesized from that, and the document says which of the two it was: an audited entry and a
        derived one are not the same evidence.
        """
        binder = _load("bind_sbom.py")
        wheel = {"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": "a" * 64, "size_bytes": 10, "version": "0.8.0"}
        raw = {"bomFormat": "CycloneDX", "components": [{"name": "numpy", "version": "2.4.6"}]}
        result = binder.bind(raw, wheel, "b" * 40)

        self.assertFalse(result["mixle_component_audited"])
        components = {c["name"]: c for c in result["cyclonedx"]["components"]}
        self.assertIn("mixle", components)
        properties = {p["name"]: p["value"] for p in components["mixle"]["properties"]}
        self.assertEqual(properties["mixle:audited"], "false")
        self.assertEqual(properties["mixle:source"], "wheel-metadata")
        # The synthesized entry must carry the artifact's real digest, not a placeholder.
        self.assertEqual(components["mixle"]["hashes"][0]["content"], "a" * 64)
        # And the caller's inventory is not mutated in place.
        self.assertEqual(len(raw["components"]), 1)

    def test_a_malformed_inventory_is_still_refused(self):
        binder = _load("bind_sbom.py")
        wheel = {"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": "a" * 64, "size_bytes": 10}
        with self.assertRaises(ValueError):
            binder.bind({"bomFormat": "not-cyclonedx", "components": []}, wheel, "b" * 40)
        with self.assertRaises(ValueError):
            binder.bind({"bomFormat": "CycloneDX"}, wheel, "b" * 40)  # no component list at all


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

    def test_decision_reviews_are_resolved_and_honestly_attributed(self):
        """The gate is closed, and the row must say HOW it was closed.

        The decision log was resolved 2026-08-04 by AI verification that the release owner reviewed
        and approved. The required CI check still runs the verifier on every push, so a regression to
        `pending` re-blocks the release; what this test pins is the attribution: the checklist row
        must keep stating that this was a human-approved AI review rather than the fully independent
        human review the gate originally contemplated. Deleting that sentence would upgrade the
        evidence by edit, which is exactly what this file exists to prevent.
        """
        checklist = (ROOT / "release-checklists" / "0.8.0.md").read_text(encoding="utf-8")
        required = (ROOT / ".github" / "release-required-checks.txt").read_text(encoding="utf-8")
        self.assertIn("Decision review acceptance | `IMPLEMENTED`", checklist)
        self.assertIn("human-approved AI review", checklist)
        self.assertIn("release decisions / accepted review", required.splitlines())
        decisions = (ROOT / "release-checklists" / "0.8.0-decisions.md").read_text(encoding="utf-8")
        self.assertIn("## Review record · 2026-08-04 resolution campaign", decisions)
        self.assertIn("reviewed by the release owner, who approved it", decisions)


class HostedWorkflowContractTest(unittest.TestCase):
    def test_security_and_real_data_evidence_are_fail_closed(self):
        security = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
        tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        extras = (ROOT / ".github" / "workflows" / "extras-matrix.yml").read_text(encoding="utf-8")
        post = (ROOT / ".github" / "workflows" / "post-publish-verify.yml").read_text(encoding="utf-8")
        self.assertIn('--path "$SITE"', security)
        self.assertIn("pip-audit (all runtime features)", security)
        self.assertIn("--profile all", security)
        self.assertIn("bandit (source security)", security)
        self.assertIn("gitleaks (full history)", security)
        self.assertIn("fetch-depth: 0", security)
        self.assertIn("verify_vulnerability_waivers.py", security)
        self.assertIn("accepted-waivers", security)
        self.assertNotIn("continue-on-error: true", security)
        # One wrapper remains after the no-direct-dataset-usage removal (2026-08-04) took the
        # sunspots/Adult flagship lanes and the CIFAR-driven quotient-leaf gate with it: the
        # model-asset scientist gate.
        self.assertEqual(tests.count("scripts/run_required_pytest.py"), 1)
        optional_job = tests.split("\n  optional:\n", 1)[1].split("\n  numerical:\n", 1)[0]
        # The property is "this gate cannot be skipped", and it survives the 2026-08-04 sharding:
        # the JOB carries no condition (it always runs, on every shard), the tier-run step is
        # unconditional, and the only `if:` the job may contain is the shard-0 pin on the serial
        # one-shot gates -- which makes each of them run EXACTLY once per workflow run rather than
        # once per shard. Any other condition would reintroduce a skippable gate.
        self.assertNotIn("\n    if:", optional_job)
        conditions = {line.strip() for line in optional_job.splitlines() if line.strip().startswith("if:")}
        self.assertLessEqual(conditions, {"if: matrix.shard == 0"})
        self.assertIn("--tier optional", optional_job)
        self.assertIn("--num-shards 2", optional_job)
        self.assertIn("extras matrix / exact candidate", extras)
        self.assertIn("verify_published_artifacts.py", post)
        self.assertIn("public-artifacts/*.whl", post)

    @unittest.skipUnless(HAS_YAML, "check_workflows parses the workflow YAML")
    def test_release_artifacts_and_automation_fail_closed(self):
        tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
        post = (ROOT / ".github" / "workflows" / "post-publish-verify.yml").read_text(encoding="utf-8")
        self.assertIn("dist/*.whl", tests)
        self.assertIn("dist/*.tar.gz", tests)
        self.assertEqual(tests.count("scripts/import_sweep.py"), 2)
        self.assertIn("--profile full", tests)
        self.assertIn("serialization_audit_contracts_test.py", tests)
        self.assertIn("0.8.0-build-requirements.txt", publish)
        self.assertIn("build_environment_receipt.py", publish)
        self.assertGreaterEqual(publish.count("--no-isolation"), 2)
        self.assertIn("REQUESTED_VERSION: ${{ inputs.version }}", post)
        self.assertEqual(_load("check_workflows.py").validate(), [])


if __name__ == "__main__":
    unittest.main()
