"""Watson axial distribution: Kummer normalizer, sampling, and ML recovery (bipolar + girdle)."""

import unittest

import numpy as np
from scipy.special import hyp1f1

from mixle.inference import estimate
from mixle.stats import WatsonDistribution
from mixle.stats.directional.watson import (
    WatsonFitError,
    WatsonSamplingError,
    _kummer_ratio,
)


class WatsonTest(unittest.TestCase):
    def setUp(self):
        self.p = 3
        self.mu = np.array([0.0, 0.0, 1.0])
        rng = np.random.RandomState(0)
        u = rng.randn(50000, self.p)
        self.uniform = u / np.linalg.norm(u, axis=1, keepdims=True)

    def test_normalizer_matches_kummer(self):
        # E_uniform[exp(kappa (mu.x)^2)] = M(1/2, p/2, kappa)
        for kappa in (5.0, -5.0):
            mc = float(np.mean(np.exp(kappa * (self.uniform @ self.mu) ** 2)))
            self.assertAlmostEqual(mc, float(hyp1f1(0.5, self.p / 2.0, kappa)), delta=0.02 * abs(mc) + 0.01)

    def test_seq_matches_scalar(self):
        d = WatsonDistribution(self.mu, 4.0)
        s = d.sampler(seed=1).sample(6)
        np.testing.assert_allclose(d.seq_log_density(s), [d.log_density(x) for x in s], atol=1e-12)

    def test_sampler_is_unit_norm_axial_and_concentrated(self):
        for kappa in (5.0, -5.0):
            d = WatsonDistribution(self.mu, kappa)
            s = d.sampler(seed=1).sample(40000)
            np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-10)
            self.assertAlmostEqual(float(np.mean((s @ self.mu) ** 2)), _kummer_ratio(kappa, self.p), delta=0.02)
            self.assertAlmostEqual(float(np.mean((s @ self.mu) > 0)), 0.5, delta=0.02)  # antipodal symmetry

    def test_mle_recovers_axis_and_kappa(self):
        for kappa in (6.0, -6.0):
            d = WatsonDistribution(self.mu, kappa)
            est = estimate(list(d.sampler(seed=2).sample(40000)), d.estimator())
            self.assertGreater(abs(float(est.mu @ self.mu)), 0.99)  # axis up to sign
            self.assertAlmostEqual(est.kappa, kappa, delta=0.6)

    def test_large_finite_concentrations_remain_finite(self):
        for kappa in (-10000.0, 10000.0):
            with self.subTest(kappa=repr(kappa)):
                d = WatsonDistribution(self.mu, kappa)
                self.assertTrue(np.isfinite(d._log_const))
                self.assertTrue(np.isfinite(_kummer_ratio(kappa, self.p)))
                self.assertTrue(np.isfinite(d.log_density(self.mu)))

    def test_parameters_are_validated_copied_and_read_only(self):
        mu = self.mu.copy()
        d = WatsonDistribution(mu, 4.0)
        mu[:] = [1.0, 0.0, 0.0]
        np.testing.assert_array_equal(d.mu, self.mu)
        with self.assertRaises(ValueError):
            d.mu[0] = 1.0
        for bad_mu in (
            [1.0],
            [2.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ):
            with self.subTest(mu=repr(bad_mu)), self.assertRaises(ValueError):
                WatsonDistribution(bad_mu, 4.0)
        for bad_kappa in (np.nan, np.inf, -np.inf):
            with self.subTest(kappa=repr(bad_kappa)), self.assertRaises(ValueError):
                WatsonDistribution(self.mu, bad_kappa)

    def test_all_scoring_paths_reject_off_sphere_observations(self):
        from mixle.engines import NUMPY_ENGINE

        d = WatsonDistribution(self.mu, 4.0)
        observations = np.asarray([[2.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            d.log_density(observations[0])
        with self.assertRaises(ValueError):
            d.seq_log_density(observations)
        with self.assertRaises(ValueError):
            d.dist_to_encoder().seq_encode(observations)
        with self.assertRaises(ValueError):
            d.backend_seq_log_density(observations, NUMPY_ENGINE)

    def test_sampler_reports_exact_rejection_diagnostics(self):
        sampler = WatsonDistribution(self.mu, 20.0).sampler(seed=1)
        sample = sampler.sample(20)
        metadata = sampler.sampling_metadata
        self.assertEqual(sample.shape, (20, 3))
        self.assertEqual(metadata["method"], "exact-rejection")
        self.assertTrue(metadata["exact"])
        self.assertEqual(metadata["accepted"], 20)
        self.assertGreaterEqual(metadata["proposed"], metadata["accepted"])

    def test_sampler_budget_failure_is_typed_and_diagnostic(self):
        sampler = WatsonDistribution(self.mu, -1.0e100).sampler(seed=1)
        with self.assertRaises(WatsonSamplingError) as raised:
            sampler.sample()
        self.assertEqual(raised.exception.accepted, 0)
        self.assertEqual(raised.exception.proposed, 10000)
        self.assertEqual(raised.exception.kappa, -1.0e100)
        self.assertEqual(raised.exception.dim, 3)

    def test_accumulator_and_estimator_reject_invalid_statistics(self):
        estimator = WatsonDistribution(self.mu, 4.0).estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([1.0, 0.0, 0.0], 1.0, None)
        before = accumulator.value()
        for weight in (-1.0, np.nan, np.inf):
            with self.subTest(weight=repr(weight)), self.assertRaises(ValueError):
                accumulator.update([0.0, 1.0, 0.0], weight, None)
            actual = accumulator.value()
            np.testing.assert_array_equal(actual[0], before[0])
            self.assertEqual(actual[1], before[1])
        with self.assertRaises(WatsonFitError):
            estimator.estimate(None, (np.zeros((3, 3)), 0.0))
        with self.assertRaises(WatsonFitError):
            estimator.estimate(None, (np.diag([1.0, 0.0, 0.0]), 1.0))
        for statistics in (
            (np.eye(3), 1.0),
            (np.diag([2.0, -1.0, 0.0]), 1.0),
            (np.full((3, 3), np.nan), 1.0),
        ):
            with self.subTest(statistics=repr(statistics)), self.assertRaises(ValueError):
                estimator.estimate(None, statistics)

    def test_isotropic_fit_is_explicitly_non_identifiable(self):
        estimator = WatsonDistribution(self.mu, 4.0).estimator()
        fitted = estimator.estimate(None, (np.eye(3) / 3.0, 1.0))
        self.assertEqual(fitted.kappa, 0.0)
        self.assertFalse(fitted.fit_metadata["identifiable_axis"])

    def test_unsupported_pseudo_count_is_rejected(self):
        with self.assertRaises(ValueError):
            WatsonDistribution(self.mu, 4.0).estimator(pseudo_count=1.0)


if __name__ == "__main__":
    unittest.main()
