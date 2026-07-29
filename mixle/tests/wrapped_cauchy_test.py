"""Wrapped Cauchy circular distribution: density vs scipy, exact sampling, mean-resultant estimation."""

import unittest

import numpy as np
from scipy.stats import kstest, wrapcauchy

from mixle.inference import estimate
from mixle.stats import (
    WrappedCauchyDistribution,
    WrappedCauchyEstimator,
    WrappedCauchyFitError,
)


class WrappedCauchyTest(unittest.TestCase):
    def setUp(self):
        self.mu, self.rho = 0.7, 0.6
        self.d = WrappedCauchyDistribution(self.mu, self.rho)

    def test_log_density_matches_scipy(self):
        th = np.array([0.1, 0.7, 2.0, -1.5])
        mine = self.d.seq_log_density(self.d.dist_to_encoder().seq_encode(th))
        ref = wrapcauchy.logpdf((th - self.mu) % (2 * np.pi), self.rho)  # scipy lives on [0, 2pi)
        np.testing.assert_allclose(mine, ref, atol=1e-10)
        np.testing.assert_allclose(mine, [self.d.log_density(t) for t in th], atol=1e-12)

    def test_density_integrates_to_one(self):
        g = np.linspace(-np.pi, np.pi, 6000)
        self.assertAlmostEqual(np.trapezoid([self.d.density(t) for t in g], g), 1.0, places=3)

    def test_sampler_matches_distribution(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertAlmostEqual(float(np.mean(np.cos(s - self.mu))), self.rho, delta=0.02)  # E[cos(theta-mu)]=rho
        self.assertGreater(kstest((s - self.mu) % (2 * np.pi), "wrapcauchy", args=(self.rho,)).pvalue, 0.01)

    def test_mean_resultant_estimator_recovers_params(self):
        est = estimate(list(self.d.sampler(seed=1).sample(40000)), self.d.estimator())
        self.assertAlmostEqual(est.mu, self.mu, delta=0.03)
        self.assertAlmostEqual(est.rho, self.rho, delta=0.03)

    def test_rho_zero_is_uniform(self):
        s = WrappedCauchyDistribution(0.0, 0.0).sampler(seed=2).sample(20000)
        self.assertLess(np.hypot(np.mean(np.cos(s)), np.mean(np.sin(s))), 0.03)  # no mean resultant

    def test_invalid_rho_raises(self):
        with self.assertRaises(ValueError):
            WrappedCauchyDistribution(0.0, 1.0)
        for parameters in ((np.nan, 0.5), (0.0, np.nan), (0.0, np.inf)):
            with self.subTest(parameters=repr(parameters)), self.assertRaises(ValueError):
                WrappedCauchyDistribution(*parameters)

    def test_near_boundary_density_stays_finite(self):
        from mixle.engines import NUMPY_ENGINE

        d = WrappedCauchyDistribution(0.0, 1.0 - 1.0e-12)
        encoded = d.dist_to_encoder().seq_encode([0.0])
        scalar = d.log_density(0.0)
        self.assertTrue(np.isfinite(scalar))
        self.assertAlmostEqual(d.seq_log_density(encoded)[0], scalar, places=8)
        self.assertAlmostEqual(
            d.backend_seq_log_density(encoded, NUMPY_ENGINE)[0],
            scalar,
            places=8,
        )

    def test_forged_encoded_observations_are_rejected(self):
        from mixle.engines import NUMPY_ENGINE

        for encoded in (
            (np.asarray([1.0]), np.asarray([1.0])),
            (np.asarray([1.0, 0.0]), np.asarray([0.0])),
            (np.asarray([np.nan]), np.asarray([0.0])),
        ):
            with self.subTest(encoded=repr(encoded)), self.assertRaises(ValueError):
                self.d.seq_log_density(encoded)
            with self.assertRaises(ValueError):
                self.d.backend_seq_log_density(encoded, NUMPY_ENGINE)

    def test_accumulator_and_estimator_reject_invalid_moments(self):
        estimator = WrappedCauchyEstimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(0.0, 1.0, None)
        before = accumulator.value()
        for weight in (-1.0, np.nan, np.inf):
            with self.subTest(weight=repr(weight)), self.assertRaises(ValueError):
                accumulator.update(1.0, weight, None)
            self.assertEqual(accumulator.value(), before)
        with self.assertRaises(WrappedCauchyFitError):
            estimator.estimate(None, (0.0, 0.0, 0.0))
        with self.assertRaises(WrappedCauchyFitError):
            estimator.estimate(None, (1.0, 0.0, 1.0))
        for statistics in (
            (2.0, 0.0, 1.0),
            (np.nan, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ):
            with self.subTest(statistics=repr(statistics)), self.assertRaises(ValueError):
                estimator.estimate(None, statistics)

    def test_uniform_fit_records_non_identifiable_direction(self):
        fitted = WrappedCauchyEstimator().estimate(None, (0.0, 0.0, 2.0))
        self.assertEqual(fitted.rho, 0.0)
        self.assertFalse(fitted.fit_metadata["identifiable_direction"])

    def test_unsupported_regularization_is_rejected(self):
        with self.assertRaises(ValueError):
            self.d.estimator(pseudo_count=1.0)
        with self.assertRaises(ValueError):
            WrappedCauchyEstimator(rho_max=0.9)


if __name__ == "__main__":
    unittest.main()
