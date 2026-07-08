"""mixle.epistemic.discrepancy: KL/JS/Wasserstein/MMD between distributions or samples (Card E1)."""

import unittest

import numpy as np

from mixle.epistemic.discrepancy import (
    discrepancy_report,
    js_divergence,
    kl_divergence,
    mmd,
    wasserstein_distance,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class KLDivergenceTest(unittest.TestCase):
    def test_kl_of_identical_gaussians_is_zero(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(0.0, 1.0)
        self.assertAlmostEqual(kl_divergence(p, q), 0.0, places=8)

    def test_kl_matches_closed_form_gaussian_formula(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(5.0, 1.0)
        mu_p, var_p, mu_q, var_q = 0.0, 1.0, 5.0, 1.0
        expected = 0.5 * (var_p / var_q + (mu_q - mu_p) ** 2 / var_q - 1.0 + np.log(var_q / var_p))
        self.assertAlmostEqual(kl_divergence(p, q), expected, places=8)


class JSDivergenceTest(unittest.TestCase):
    def test_symmetric_within_mc_tolerance(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(3.0, 1.0)
        a = js_divergence(p, q, n=20_000, seed=0)
        b = js_divergence(q, p, seed=0, n=20_000)
        self.assertAlmostEqual(a, b, delta=0.02)


class WassersteinDistanceTest(unittest.TestCase):
    def test_matches_known_toy_case(self):
        class PointMass:
            def __init__(self, value):
                self.value = value

            def sample(self, n):
                return np.full(n, self.value, dtype=np.float64)

        p = PointMass(0.0)
        q = PointMass(3.0)
        self.assertAlmostEqual(wasserstein_distance(p, q, n=100), 3.0, places=8)

    def test_multivariate_raises_not_implemented(self):
        class TwoD:
            def sample(self, n):
                return np.zeros((n, 2), dtype=np.float64)

        with self.assertRaises(NotImplementedError):
            wasserstein_distance(TwoD(), TwoD(), n=10)


class MMDTest(unittest.TestCase):
    def test_same_distribution_is_near_zero(self):
        rng = np.random.RandomState(0)
        xs = rng.normal(size=1000)
        value = mmd(xs[:500], xs[500:])
        self.assertLess(abs(value), 0.05)

    def test_different_distributions_is_clearly_positive(self):
        rng = np.random.RandomState(0)
        xs = rng.normal(size=1000)
        ys = rng.normal(loc=5.0, size=1000)
        self.assertGreater(mmd(xs, ys), 0.5)

    def test_unknown_kernel_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            mmd(np.zeros(5), np.zeros(5), kernel="polynomial")


class DiscrepancyReportTest(unittest.TestCase):
    def test_not_degraded_for_registered_closed_form_pair(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(1.0, 1.0)
        result = discrepancy_report(p, q)
        self.assertEqual(result.metric, "kl_divergence")
        self.assertFalse(result.degraded)

    def test_degraded_for_a_pair_without_a_closed_form(self):
        class SampleOnly:
            def __init__(self, loc):
                self.loc = loc

            def log_density(self, x):
                return float(-0.5 * (x - self.loc) ** 2)

            def sample(self, n):
                return np.random.RandomState(0).normal(loc=self.loc, size=n)

        result = discrepancy_report(SampleOnly(0.0), SampleOnly(1.0))
        self.assertEqual(result.metric, "kl_divergence")
        self.assertTrue(result.degraded)

    def test_explicit_metric_is_honored(self):
        result = discrepancy_report(np.zeros(50), np.ones(50), metric="mmd")
        self.assertEqual(result.metric, "mmd")
        self.assertTrue(result.degraded)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            discrepancy_report(np.zeros(5), np.zeros(5), metric="not_a_real_metric")


if __name__ == "__main__":
    unittest.main()
