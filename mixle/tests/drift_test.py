"""Model/data drift detection and the production Monitor (retrain-and-swap, DOE-driven sampling)."""

import unittest

import numpy as np

from mixle.inference.production import Monitor, detect_drift, fit_with_provenance, score_drift

# the per-feature drift metrics are importable but demoted from the blessed production surface
from mixle.inference.production.drift import ks_statistic, population_stability_index
from mixle.stats import GaussianDistribution


class DriftMetricsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.ref = self.rng.normal(0, 1, 2000)

    def test_psi_low_for_same_high_for_shifted(self):
        same = self.rng.normal(0, 1, 1000)
        shifted = self.rng.normal(3, 1.5, 1000)
        self.assertLess(population_stability_index(self.ref, same), 0.1)
        self.assertGreater(population_stability_index(self.ref, shifted), 0.25)

    def test_psi_detects_drift_away_from_a_constant_reference(self):
        # A constant reference feature has no quantile spread to bin against; this used to
        # unconditionally return 0.0 ("no drift") regardless of `current` -- even when current has
        # moved entirely away from the reference's one value, the most extreme drift possible.
        constant_ref = np.full(500, 7.0)
        same = np.full(300, 7.0)
        drifted = np.full(300, 50.0)
        self.assertEqual(population_stability_index(constant_ref, same), 0.0)
        self.assertGreater(population_stability_index(constant_ref, drifted), 5.0)

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


class _PartialScorer:
    """A model that scores a sentinel record as NaN (unscorable) and everything else exactly."""

    SENTINEL = "unscorable"

    def dist_to_encoder(self):
        return self

    def seq_encode(self, rows):
        return list(rows)

    def seq_log_density(self, enc):
        return np.asarray([self.log_density(r) for r in enc], dtype=float)

    def log_density(self, x):
        return float("nan") if x == self.SENTINEL else -1.0


class _ShortBatchScorer:
    """A model whose batch path always returns one fewer score than it was given records."""

    def dist_to_encoder(self):
        return self

    def seq_encode(self, rows):
        return list(rows)

    def seq_log_density(self, enc):
        return np.zeros(max(len(enc) - 1, 0), dtype=float)

    def log_density(self, x):
        return 0.0


class _BatchDownScorer:
    """A model whose batch path always raises; the per-record path scores every record fine."""

    def __init__(self):
        self.scalar_calls = 0

    def dist_to_encoder(self):
        return self

    def seq_encode(self, rows):
        return list(rows)

    def seq_log_density(self, enc):
        raise ConnectionError("batch scoring endpoint is down")

    def log_density(self, x):
        self.scalar_calls += 1
        return -1.0


class DriftEvidenceCoverageTest(unittest.TestCase):
    """MXR-080-1640: an unscorable current sample cannot be certified drift-free."""

    def test_mostly_unscorable_current_sample_is_not_declared_drift_free(self):
        model = _PartialScorer()
        ref = ["ok"] * 10
        current = ["ok"] + [_PartialScorer.SENTINEL] * 9
        report = detect_drift(model, ref, current, per_feature=False)
        self.assertAlmostEqual(report.score["fraction_unscorable_current"], 0.9, places=9)
        self.assertAlmostEqual(report.score["ks"], 0.0, places=9)
        self.assertAlmostEqual(report.score["mean_loglik_shift"], 0.0, places=9)
        self.assertTrue(report.drift)  # coverage, not the shift statistics, drives this verdict
        self.assertTrue(any("scorable" in r for r in report.reasons))

    def test_a_rise_in_the_unscorable_rate_is_itself_drift(self):
        model = _PartialScorer()
        ref = ["ok"] * 100
        # 80% still scorable, so the coverage floor is met -- but the rate rose 20 points vs reference
        current = ["ok"] * 80 + [_PartialScorer.SENTINEL] * 20
        report = detect_drift(model, ref, current, per_feature=False)
        self.assertTrue(report.drift)
        self.assertTrue(any("unscorable rate rose" in r for r in report.reasons))

    def test_fully_scorable_matching_populations_stay_clean(self):
        model = _PartialScorer()
        report = detect_drift(model, ["ok"] * 50, ["ok"] * 50, per_feature=False)
        self.assertFalse(report.drift)
        self.assertEqual(report.reasons, [])
        self.assertEqual(report.score["n_scorable_current"], 50)


class DriftOneShotPopulationTest(unittest.TestCase):
    """MXR-080-1641: each population is consumed exactly once and scoring output is size-checked."""

    def test_identical_one_shot_streams_do_not_produce_a_false_drift_verdict(self):
        model = _PartialScorer()
        report = detect_drift(model, iter([0.0, 1.0, 2.0]), iter([0.0, 1.0, 2.0]))
        self.assertFalse(report.drift)
        self.assertEqual(report.reasons, [])
        self.assertEqual(report.processed_count, 3)
        for stats in report.per_feature.values():
            self.assertTrue(np.isfinite(stats["psi"]))
            self.assertLess(stats["ks"], 1.0)

    def test_score_drift_accepts_one_shot_iterables(self):
        model = _PartialScorer()
        s = score_drift(model, iter([0.0, 1.0]), iter([0.0, 1.0]))
        self.assertEqual(s["n_reference"], 2)
        self.assertEqual(s["n_current"], 2)

    def test_batch_failure_falls_back_without_consuming_the_population(self):
        model = _BatchDownScorer()
        s = score_drift(model, iter([0.0, 1.0, 2.0]), iter([0.0, 1.0, 2.0]))
        self.assertEqual(s["n_scorable_reference"], 3)
        self.assertEqual(s["n_scorable_current"], 3)
        self.assertEqual(model.scalar_calls, 6)  # every record was actually retried, none lost

    def test_wrong_cardinality_scoring_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one score per record"):
            score_drift(_ShortBatchScorer(), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])


class ModelMonitorTest(unittest.TestCase):
    def test_retrains_and_swaps_on_drift(self):
        rng = np.random.RandomState(2)
        ref = rng.normal(0, 1, 2000).tolist()
        model, _ = fit_with_provenance(ref, GaussianDistribution(0, 1).estimator(), max_its=20)
        mon = Monitor(model, GaussianDistribution(0, 1).estimator(), ref)

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
        mon = Monitor(model, GaussianDistribution(0, 1).estimator(), ref)
        pts = mon.suggest_samples([(0.0, 1.0), (-1.0, 1.0)], n=6)
        self.assertEqual(np.asarray(pts).shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
