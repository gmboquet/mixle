"""Tests for the configurable fixed concentration (rho) of SpearmanRankingEstimator.

The estimator computes the consensus ranking sigma by maximum likelihood but holds
the concentration rho fixed at the configured value rather than estimating it from
the data.
"""
import unittest

import numpy as np

from pysp.stats.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator


def _estimate(est, data):
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return est.estimate(None, acc.value())


class SpearmanRankingEstimatorRhoTestCase(unittest.TestCase):

    data = [[0, 1, 2, 3], [0, 1, 3, 2], [1, 0, 2, 3], [0, 1, 2, 3]]

    def test_default_rho_is_one(self):
        dist = _estimate(SpearmanRankingEstimator(4), self.data)
        self.assertEqual(dist.rho, 1.0)

    def test_configured_rho_is_used(self):
        dist = _estimate(SpearmanRankingEstimator(4, rho=2.5), self.data)
        self.assertEqual(dist.rho, 2.5)

    def test_invalid_rho_raises(self):
        with self.assertRaises(ValueError):
            SpearmanRankingEstimator(4, rho=0.0)
        with self.assertRaises(ValueError):
            SpearmanRankingEstimator(4, rho=-1.0)

    def test_estimator_round_trip_preserves_rho(self):
        dist = SpearmanRankingDistribution([0.0, 1.0, 2.0, 3.0], rho=3.5, name='s')
        est = dist.estimator()
        self.assertEqual(est.rho, 3.5)
        refit = _estimate(est, self.data)
        self.assertEqual(refit.rho, 3.5)

    def test_sigma_unaffected_by_rho(self):
        dist1 = _estimate(SpearmanRankingEstimator(4, rho=1.0), self.data)
        dist2 = _estimate(SpearmanRankingEstimator(4, rho=10.0), self.data)
        np.testing.assert_array_equal(dist1.sigma, dist2.sigma)
        np.testing.assert_array_equal(dist1.sigma, np.array([0, 1, 2, 3]))

    def test_zero_data_branch_unchanged(self):
        dist = SpearmanRankingEstimator(4, rho=2.0).estimate(None, (0.0, np.zeros(4)))
        self.assertEqual(dist.rho, 0.0)


if __name__ == '__main__':
    unittest.main()
