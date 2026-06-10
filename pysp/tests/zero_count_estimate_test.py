"""Tests for ParameterEstimator.estimate() with empty data / all-zero weights.

Estimating from aggregated sufficient statistics whose total weight is zero
(no observations, or every observation weighted 0.0) must return finite,
valid default parameters rather than nan/inf.
"""
import unittest

import numpy as np

from pysp.stats.gamma import GammaEstimator
from pysp.stats.geometric import GeometricEstimator
from pysp.stats.intmultinomial import IntegerMultinomialEstimator
from pysp.stats.poisson import PoissonEstimator


class PoissonZeroCountTestCase(unittest.TestCase):

    def test_estimate_from_empty_suff_stats(self):
        dist = PoissonEstimator().estimate(None, (0.0, 0.0))
        self.assertTrue(np.isfinite(dist.lam))
        self.assertGreater(dist.lam, 0.0)

    def test_estimate_from_all_zero_weights(self):
        acc = PoissonEstimator().accumulator_factory().make()
        for x in [1, 2, 3]:
            acc.update(x, 0.0)
        dist = PoissonEstimator().estimate(None, acc.value())
        self.assertTrue(np.isfinite(dist.lam))
        self.assertGreater(dist.lam, 0.0)

    def test_estimate_unaffected_when_data_present(self):
        dist = PoissonEstimator().estimate(None, (4.0, 12.0))
        self.assertAlmostEqual(dist.lam, 3.0, places=12)


class GeometricZeroCountTestCase(unittest.TestCase):

    def test_estimate_from_empty_suff_stats(self):
        dist = GeometricEstimator().estimate(None, (0.0, 0.0))
        self.assertTrue(np.isfinite(dist.p))
        self.assertTrue(0.0 < dist.p < 1.0)

    def test_estimate_from_all_zero_weights(self):
        acc = GeometricEstimator().accumulator_factory().make()
        for x in [1, 4, 2]:
            acc.update(x, 0.0, None)
        dist = GeometricEstimator().estimate(None, acc.value())
        self.assertTrue(np.isfinite(dist.p))
        self.assertTrue(0.0 < dist.p < 1.0)

    def test_estimate_unaffected_when_data_present(self):
        dist = GeometricEstimator().estimate(None, (5.0, 20.0))
        self.assertAlmostEqual(dist.p, 0.25, places=12)


class IntegerMultinomialZeroCountTestCase(unittest.TestCase):

    def test_estimate_from_zero_counts_ml(self):
        dist = IntegerMultinomialEstimator().estimate(None, (0, np.zeros(3), None))
        np.testing.assert_allclose(dist.p_vec, np.ones(3) / 3.0)

    def test_estimate_from_zero_counts_zero_pseudo_count(self):
        est = IntegerMultinomialEstimator(pseudo_count=0.0)
        dist = est.estimate(None, (0, np.zeros(4), None))
        np.testing.assert_allclose(dist.p_vec, np.ones(4) / 4.0)

    def test_estimate_from_zero_counts_zero_pseudo_count_min_max(self):
        est = IntegerMultinomialEstimator(min_val=0, max_val=2, pseudo_count=0.0)
        dist = est.estimate(None, (0, np.zeros(3), None))
        np.testing.assert_allclose(dist.p_vec, np.ones(3) / 3.0)

    def test_estimate_from_zero_counts_zero_pseudo_count_suff_stat(self):
        est = IntegerMultinomialEstimator(pseudo_count=0.0, suff_stat=(0, np.zeros(3)))
        dist = est.estimate(None, (0, np.zeros(3), None))
        np.testing.assert_allclose(dist.p_vec, np.ones(3) / 3.0)

    def test_estimate_unaffected_when_data_present(self):
        dist = IntegerMultinomialEstimator().estimate(None, (0, np.array([1.0, 3.0]), None))
        np.testing.assert_allclose(dist.p_vec, [0.25, 0.75])


class GammaZeroCountTestCase(unittest.TestCase):

    def test_estimate_from_empty_suff_stats(self):
        dist = GammaEstimator(name='g').estimate(None, (0.0, 0.0, 0.0))
        self.assertTrue(np.isfinite(dist.k))
        self.assertTrue(np.isfinite(dist.theta))
        self.assertGreater(dist.k, 0.0)
        self.assertGreater(dist.theta, 0.0)
        self.assertEqual(dist.name, 'g')

    def test_estimate_from_all_zero_weights(self):
        acc = GammaEstimator().accumulator_factory().make()
        for x in [0.5, 1.5, 3.0]:
            acc.update(x, 0.0, None)
        dist = GammaEstimator().estimate(None, acc.value())
        self.assertTrue(np.isfinite(dist.k))
        self.assertTrue(np.isfinite(dist.theta))

    def test_ml_mean_matches_k_theta(self):
        rng = np.random.RandomState(7)
        x = rng.gamma(shape=2.5, scale=1.7, size=20000)
        suff_stat = (float(len(x)), x.sum(), np.log(x).sum())
        dist = GammaEstimator().estimate(None, suff_stat)
        # ML estimate preserves the sample mean: k * theta == mean(x).
        self.assertAlmostEqual(dist.k * dist.theta, x.mean(), places=10)

    def test_pseudo_count_mean_uses_pc1_adjusted_count(self):
        # With pc1 != pc2 the scale must come from the pc1-adjusted mean,
        # not the pc2-adjusted log-count.
        pc1, pc2 = 2.0, 8.0
        ss1, ss2 = 3.0, 1.0
        rng = np.random.RandomState(11)
        x = rng.gamma(shape=2.0, scale=2.0, size=1000)
        suff_stat = (float(len(x)), x.sum(), np.log(x).sum())

        est = GammaEstimator(pseudo_count=(pc1, pc2), suff_stat=(ss1, ss2))
        dist = est.estimate(None, suff_stat)

        adj_mean = (x.sum() + ss1 * pc1) / (len(x) + pc1)
        self.assertAlmostEqual(dist.k * dist.theta, adj_mean, places=10)


if __name__ == '__main__':
    unittest.main()
