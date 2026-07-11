"""The reproduction script's claim checks are deterministic and correct (worklist E14).

``scripts/reproduce.py`` emits a receipt an external reviewer runs to reproduce mixle's headline claims. Its
value is only as good as its determinism: if the seeded checks drift between runs, the receipt cannot be
compared. This pins that -- the checks reproduce exactly, and they hold (a Gaussian fit recovers its
parameters, scalar/vectorized scores agree, serialization is score-preserving, automatic selection recovers
the family) -- and that the environment capture has the fields a reviewer needs.
"""

import importlib.util
import unittest
from pathlib import Path

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
        checks = self.mod.claim_checks()
        self.assertTrue(checks["scalar_vectorized_agree"])
        self.assertTrue(checks["serialization_score_equal"])
        self.assertEqual(checks["auto_selects"], "GaussianEstimator")
        # a Gaussian fit on N(3, 2) data recovers the parameters within sampling error.
        self.assertAlmostEqual(checks["gaussian_fit_mu"], 3.0, delta=0.2)
        self.assertAlmostEqual(checks["gaussian_fit_sigma"], 2.0, delta=0.2)

    def test_environment_capture_has_required_fields(self):
        env = self.mod.environment()
        for field in ("python", "platform", "machine", "mixle", "numpy", "scipy", "git_commit"):
            self.assertIn(field, env)


if __name__ == "__main__":
    unittest.main()
