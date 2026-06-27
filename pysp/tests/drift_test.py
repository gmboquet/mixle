"""Model/data drift detection and the production ModelMonitor (retrain-and-swap, DOE-driven sampling)."""

import unittest

import numpy as np

from pysp.inference import (
    ModelMonitor,
    detect_drift,
    fit_with_provenance,
    ks_statistic,
    population_stability_index,
    score_drift,
)
from pysp.stats import GaussianDistribution


class DriftMetricsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.ref = self.rng.normal(0, 1, 2000)

    def test_psi_low_for_same_high_for_shifted(self):
        same = self.rng.normal(0, 1, 1000)
        shifted = self.rng.normal(3, 1.5, 1000)
        self.assertLess(population_stability_index(self.ref, same), 0.1)
        self.assertGreater(population_stability_index(self.ref, shifted), 0.25)

    def test_ks_orders_shift(self):
        same = self.rng.normal(0, 1, 1000)
        shifted = self.rng.normal(2, 1, 1000)
        self.assertLess(ks_statistic(self.ref, same), ks_statistic(self.ref, shifted))

    def test_score_drift_signals_lower_likelihood(self):
        model, _ = fit_with_provenance(self.ref.tolist(), GaussianDistribution(0, 1).estimator(), max_its=20)
        s = score_drift(model, self.ref.tolist(), self.rng.normal(4, 1, 800).tolist())
        self.assertLess(s["mean_loglik_shift"], 0.0)  # drifted data is less likely under the model
        self.assertGreater(s["ks"], 0.2)


class DetectDriftTest(unittest.TestCase):
    def test_flags_only_when_drifted(self):
        rng = np.random.RandomState(1)
        ref = rng.normal(0, 1, 2000).tolist()
        model, _ = fit_with_provenance(ref, GaussianDistribution(0, 1).estimator(), max_its=20)
        self.assertFalse(detect_drift(model, ref, rng.normal(0, 1, 1000).tolist()).drift)
        report = detect_drift(model, ref, rng.normal(3, 1.5, 1000).tolist())
        self.assertTrue(report.drift)
        self.assertIn("value", report.per_feature)


class ModelMonitorTest(unittest.TestCase):
    def test_retrains_and_swaps_on_drift(self):
        rng = np.random.RandomState(2)
        ref = rng.normal(0, 1, 2000).tolist()
        model, _ = fit_with_provenance(ref, GaussianDistribution(0, 1).estimator(), max_its=20)
        mon = ModelMonitor(model, GaussianDistribution(0, 1).estimator(), ref)

        clean = mon.update(rng.normal(0, 1, 800).tolist(), max_its=20)
        self.assertEqual(clean["action"], "none")  # no drift -> no retrain

        drifted = mon.update(rng.normal(3, 1.0, 1500).tolist(), max_its=20)
        self.assertEqual(drifted["action"], "retrained")
        self.assertGreater(drifted["model"].mu, 0.5)  # swapped model moved toward the new data
        self.assertIsNotNone(drifted["header"])  # retrain recorded a fresh provenance header
        self.assertEqual(len(mon.history), 2)

    def test_doe_suggest_samples(self):
        rng = np.random.RandomState(3)
        ref = rng.normal(0, 1, 500).tolist()
        model, _ = fit_with_provenance(ref, GaussianDistribution(0, 1).estimator(), max_its=10)
        mon = ModelMonitor(model, GaussianDistribution(0, 1).estimator(), ref)
        pts = mon.suggest_samples([(0.0, 1.0), (-1.0, 1.0)], n=6)
        self.assertEqual(np.asarray(pts).shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
