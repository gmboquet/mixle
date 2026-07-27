"""The reproduction script's claim checks are deterministic and correct (worklist E14).

``scripts/reproduce.py`` emits a receipt an external reviewer runs to reproduce mixle's headline claims. Its
value is only as good as its determinism: if the seeded checks drift between runs, the receipt cannot be
compared. This pins that -- the checks reproduce exactly, and they hold (a Gaussian fit recovers its
parameters, scalar/vectorized scores agree, serialization is score-preserving, automatic selection recovers
the family) -- and that the environment capture has the fields a reviewer needs.
"""

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_GEN = Path(__file__).resolve().parents[2] / "scripts" / "reproduce.py"


def _load():
    spec = importlib.util.spec_from_file_location("_reproduce", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReproduceReceiptTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_claim_checks_are_deterministic(self):
        self.assertEqual(self.mod.claim_checks(), self.mod.claim_checks())

    def test_claim_checks_hold(self):
        checks = self.mod.evaluate_claims(self.mod.claim_checks())
        self.assertTrue(all(check["passed"] for check in checks.values()))
        # a Gaussian fit on N(3, 2) data recovers the parameters within sampling error.
        self.assertAlmostEqual(checks["gaussian_fit_mu"]["observed"], 3.0, delta=0.2)
        self.assertAlmostEqual(checks["gaussian_fit_sigma"]["observed"], 2.0, delta=0.2)

    def test_environment_capture_has_required_fields(self):
        env = self.mod.environment()
        for field in ("python", "platform", "machine", "mixle", "numpy", "scipy", "installed_content"):
            self.assertIn(field, env)

    def test_false_or_wrong_claims_fail_the_receipt_and_process(self):
        wrong = self.mod.claim_checks()
        wrong["scalar_vectorized_agree"] = False
        wrong["auto_selects"] = "WrongEstimator"
        evaluated = self.mod.evaluate_claims(wrong)
        self.assertFalse(evaluated["scalar_vectorized_agree"]["passed"])
        self.assertFalse(evaluated["auto_selects"]["passed"])

        receipt_path = Path(self.id().replace(".", "-") + ".json")
        try:
            with (
                patch("mixle.reproduction.claim_checks", return_value=wrong),
                patch(
                    "mixle.reproduction.source_tree_provenance",
                    return_value={"artifact": "test", "verified": True},
                ),
                patch(
                    "mixle.reproduction.installed_content_provenance",
                    return_value={"artifact": "test", "verified": True},
                ),
            ):
                self.assertEqual(self.mod.main(["--source-tree", "--out", str(receipt_path)]), 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
        finally:
            receipt_path.unlink(missing_ok=True)

    def test_ambient_repository_is_not_used_for_source_provenance(self):
        provenance = self.mod.source_tree_provenance()
        self.assertEqual(Path(provenance["repository"]), _GEN.parents[1])
        self.assertNotEqual(provenance["commit"], "unknown")

    def test_wheel_receipt_requires_clean_embedded_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"

            def write_wheel(source_dirty):
                provenance = {
                    "artifact": "mixle.build_provenance/v1",
                    "source_commit": "a" * 40,
                    "source_tree": "b" * 40,
                    "source_dirty": source_dirty,
                    "source_content_sha256": "c" * 64,
                }
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("mixle-0.8.0.dist-info/METADATA", "Name: mixle\nVersion: 0.8.0\n")
                    archive.writestr("mixle/_build_provenance.json", json.dumps(provenance))

            write_wheel(False)
            receipt = self.mod.wheel_provenance(wheel)
            self.assertTrue(receipt["verified"])
            self.assertEqual(len(receipt["sha256"]), 64)

            write_wheel(True)
            with self.assertRaisesRegex(ValueError, "clean source"):
                self.mod.wheel_provenance(wheel)


if __name__ == "__main__":
    unittest.main()
