"""Calibration as a post-condition of fit (B2): the PIT test tells if the model's UQ is honest."""

import unittest

import numpy as np

import mixle.stats as st
from mixle import Model
from mixle.inference import CalibrationReport, calibration_report, optimize


class PITCalibrationTest(unittest.TestCase):
    def test_well_specified_model_is_calibrated(self):
        train = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 800)]
        hold = [float(x) for x in np.random.RandomState(1).normal(5.0, 2.0, 400)]
        rep = calibration_report(optimize(train, st.GaussianEstimator(), out=None), hold)
        self.assertEqual(rep.method, "PIT")
        self.assertTrue(rep.is_calibrated())  # PIT error within the finite-sample noise floor
        self.assertIn("calibrated", rep.note)

    def test_misspecified_model_is_flagged(self):
        r = np.random.RandomState(2)
        bim = np.concatenate([r.normal(-6, 1, 400), r.normal(6, 1, 400)]).tolist()
        r2 = np.random.RandomState(3)
        bim_h = np.concatenate([r2.normal(-6, 1, 200), r2.normal(6, 1, 200)]).tolist()
        rep = calibration_report(optimize(bim, st.GaussianEstimator(), out=None), bim_h)
        self.assertFalse(rep.is_calibrated())  # a Gaussian on bimodal data is not calibrated
        self.assertGreater(rep.pit_error, 2.5 * rep.noise_floor())
        self.assertIn("deviates", rep.note)

    def test_noise_floor_scales_with_n(self):
        rep_small = CalibrationReport(n=100, mean_log_density=0.0, pit_error=0.0)
        rep_big = CalibrationReport(n=10000, mean_log_density=0.0, pit_error=0.0)
        self.assertGreater(rep_small.noise_floor(), rep_big.noise_floor())  # smaller n -> higher floor


class NoCDFTest(unittest.TestCase):
    def test_composite_reports_na_honestly(self):
        tr = [(float(a), float(b)) for a, b in zip(np.random.randn(200), np.random.randn(200))]
        te = [(float(a), float(b)) for a, b in zip(np.random.randn(100), np.random.randn(100))]
        model = optimize(tr, st.CompositeEstimator((st.GaussianEstimator(), st.GaussianEstimator())), out=None)
        rep = calibration_report(model, te)
        self.assertIsNone(rep.pit_error)  # no scalar CDF -> PIT not applicable
        self.assertFalse(rep.is_calibrated())  # unknown -> conservatively not calibrated
        self.assertTrue(np.isfinite(rep.mean_log_density))  # but the proper score is always reported


class ModelFacadeTest(unittest.TestCase):
    def test_calibrate_opt_in_attaches_report_and_reserves_holdout(self):
        train = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 800)]
        m = Model(st.GaussianEstimator()).fit(train, calibrate=0.25)
        self.assertIsInstance(m.calibration, CalibrationReport)
        self.assertIsNotNone(m.calibration.pit_error)
        self.assertEqual(m._fit_info["n"], 600)  # 25% reserved -> fit on 600, not 800

    def test_calibrate_true_uses_quarter(self):
        train = [float(x) for x in np.random.RandomState(0).normal(0.0, 1.0, 400)]
        m = Model(st.GaussianEstimator()).fit(train, calibrate=True)
        self.assertIsNotNone(m.calibration)
        self.assertEqual(m._fit_info["n"], 300)

    def test_default_fit_has_no_calibration_cost(self):
        train = [float(x) for x in np.random.RandomState(0).normal(0.0, 1.0, 400)]
        m = Model(st.GaussianEstimator()).fit(train)
        self.assertIsNone(m.calibration)  # opt-in: off by default, no held-out data spent
        self.assertEqual(m._fit_info["n"], 400)


if __name__ == "__main__":
    unittest.main()
