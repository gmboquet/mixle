"""create() (F3): data (+ budget/device) to a certified model artifact."""

import unittest

import numpy as np

from mixle.inference import create
from mixle.inference.create import CreatedModel


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
        self.assertIsNotNone(art.uq)  # scalar Gaussian → Laplace likelihood-curvature approximation

    def test_uq_attaches_for_a_structured_bn_model(self):
        # HeterogeneousBayesianNetwork Laplace-flattening landed (bn-laplace-flatten); a structured
        # fit over a mixed categorical+numeric record now gets a parameter-curvature approximation, not a
        # graceful None -- assert the new capability, not the old gap.
        art = create(_plan_spend(300, 0), quantify_uq=True, seed=0)
        self.assertIsNotNone(art.uq)
        self.assertEqual(art.uq.kind, "parameter_likelihood_approximation")
        self.assertIn("laplace", art.uq.method.lower())
        models = art.uq.sample_models(5, seed=0)
        self.assertEqual(len(models), 5)
        self.assertGreaterEqual(int(art.guarantee), 4)  # everything else still holds

    def test_budget_device_constrains_to_a_smaller_model(self):
        art = create(_plan_spend(300, 0), device="rpi-zero", budget=4096, seed=0)
        self.assertEqual(art.strategy, "edge-constrained")
        self.assertEqual(art.provenance["structure"], "off")  # independence-first under an envelope
        self.assertEqual(art.provenance["device"], "'rpi-zero'")

    def test_bad_calibrate_fraction_raises(self):
        with self.assertRaises(ValueError):
            create(_scalar(100, 0), calibrate=1.5)

    def test_successful_run_records_every_postcondition_as_performed(self):
        art = create(_scalar(300, 0), calibrate=0.3, quantify_uq=True, seed=0)
        self.assertTrue(art.is_certified())
        self.assertEqual(art.failed_postconditions(), [])
        for name in ("calibration", "uq"):
            self.assertTrue(art.postconditions[name]["requested"])
            self.assertTrue(art.postconditions[name]["performed"])
            self.assertIsNone(art.postconditions[name]["error"])
        self.assertGreaterEqual(int(art.guarantee), 4)


class RequestedPostconditionFailureTest(unittest.TestCase):
    """MXR-080-1649: a requested calibration/UQ that raised may not vanish behind a 'certified' claim."""

    def _both_failing(self):
        import importlib
        import sys

        from mixle.inference.planning import Guarantee

        cal_module = importlib.import_module("mixle.inference.calibrate_fit")
        # mixle.inference re-exports uq() as an attribute, shadowing the submodule for attribute
        # access -- reach the module object itself through sys.modules.
        uq_module = sys.modules["mixle.inference.uq"]

        def _boom(*a, **kw):
            raise RuntimeError("postcondition backend is down")

        real_cal, real_uq = cal_module.calibration_report, uq_module.uq
        try:
            cal_module.calibration_report = _boom
            uq_module.uq = _boom
            art = create(_scalar(300, 0), calibrate=0.2, quantify_uq=True, seed=0)
        finally:
            cal_module.calibration_report = real_cal
            uq_module.uq = real_uq
        return art, Guarantee

    def test_failed_postconditions_are_recorded_and_block_certification(self):
        art, Guarantee = self._both_failing()
        self.assertIsNone(art.calibration)
        self.assertIsNone(art.uq)
        self.assertEqual(art.failed_postconditions(), ["calibration", "uq"])
        self.assertFalse(art.is_certified())
        for name in ("calibration", "uq"):
            self.assertTrue(art.postconditions[name]["requested"])
            self.assertFalse(art.postconditions[name]["performed"])
            self.assertIn("postcondition backend is down", art.postconditions[name]["error"])
        # the aggregate claim no longer outruns the conditions it was asked to establish
        self.assertEqual(art.guarantee, Guarantee.UNVERIFIED)
        # a requested-but-failed calibration is False, not the "never asked" None
        self.assertIs(art.is_calibrated(), False)

    def test_unrequested_postconditions_leave_the_artifact_certified(self):
        art = create(_scalar(300, 0), seed=0)
        self.assertTrue(art.is_certified())
        self.assertIsNone(art.is_calibrated())  # never asked -> unknown, not failed
        self.assertGreaterEqual(int(art.guarantee), 4)


if __name__ == "__main__":
    unittest.main()
