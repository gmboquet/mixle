"""create() (F3): data (+ budget/device) to a certified model artifact."""

import unittest

import numpy as np

from mixle.inference import create
from mixle.inference.create import CreatedModel
from mixle.inference.uq import UQResult


def _plan_spend(n, seed):
    rng = np.random.RandomState(seed)
    return [(["free", "pro"][i % 2], float(20 + 80 * (i % 2) + 3 * rng.randn())) for i in range(n)]


def _scalar(n, seed):
    return [float(x) for x in np.random.RandomState(seed).normal(5, 2, n)]


class CreateTest(unittest.TestCase):
    def test_returns_a_certified_artifact(self):
        art = create(_plan_spend(300, 0), seed=0)
        self.assertIsInstance(art, CreatedModel)
        self.assertGreaterEqual(int(art.guarantee), 4)  # closed-form/EM: GLOBAL or better
        self.assertIn("No gradient descent", art.why())
        self.assertEqual(art.strategy, "structured")

    def test_calibrate_reserves_a_holdout_and_checks(self):
        art = create(_plan_spend(400, 1), calibrate=0.3, seed=1)
        self.assertIsNotNone(art.calibration)
        self.assertLess(art.provenance["n_fit"], art.provenance["n"])  # holdout carved out
        self.assertIn(art.is_calibrated(), (True, False))  # a verdict was rendered

    def test_uq_attaches_for_a_flattenable_model(self):
        art = create(_scalar(300, 0), quantify_uq=True, seed=0)
        self.assertIsNotNone(art.uq)  # scalar Gaussian → Laplace posterior

    def test_uq_degrades_gracefully_when_unflattenable(self):
        art = create(_plan_spend(300, 0), quantify_uq=True, seed=0)
        # UQ is best-effort: either a real posterior or an honest None, never a crash.
        self.assertTrue(art.uq is None or isinstance(art.uq, UQResult))
        self.assertGreaterEqual(int(art.guarantee), 4)  # everything else still holds

    def test_budget_device_constrains_to_a_smaller_model(self):
        art = create(_plan_spend(300, 0), device="rpi-zero", budget=4096, seed=0)
        self.assertEqual(art.strategy, "edge-constrained")
        self.assertEqual(art.provenance["structure"], "off")  # independence-first under an envelope
        self.assertEqual(art.provenance["device"], "'rpi-zero'")

    def test_bad_calibrate_fraction_raises(self):
        with self.assertRaises(ValueError):
            create(_scalar(100, 0), calibrate=1.5)


if __name__ == "__main__":
    unittest.main()
