"""Tests for SpearmanRankingEstimator concentration (rho) estimation."""

import itertools
import unittest

import numpy as np

from mixle.stats.rankings.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator


def _estimate(est, data):
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return est.estimate(None, acc.value())


class SpearmanRankingEstimatorRhoTestCase(unittest.TestCase):
    data = [[0, 1, 2, 3], [0, 1, 3, 2], [1, 0, 2, 3], [0, 1, 2, 3]]

    def test_default_rho_is_estimated(self):
        dist = _estimate(SpearmanRankingEstimator(4), self.data)
        self.assertGreater(dist.rho, 0.0)

    def test_configured_rho_is_held_fixed(self):
        dist = _estimate(SpearmanRankingEstimator(4, rho=2.5), self.data)
        self.assertEqual(dist.rho, 2.5)

    def test_consensus_is_rank_vector_not_item_order(self):
        data = [[2, 0, 1], [2, 0, 1], [2, 1, 0]]

        dist = _estimate(SpearmanRankingEstimator(3, rho=1.0), data)

        np.testing.assert_array_equal(dist.sigma, np.asarray([2, 0, 1]))
        self.assertGreater(dist.log_density([2, 0, 1]), dist.log_density([1, 2, 0]))

    def test_invalid_rho_raises(self):
        with self.assertRaises(ValueError):
            SpearmanRankingEstimator(4, rho=-1.0)
        with self.assertRaises(ValueError):
            SpearmanRankingEstimator(4, max_rho=0.0)

    def test_distribution_estimator_estimates_rho_by_default(self):
        dist = SpearmanRankingDistribution([0.0, 1.0, 2.0, 3.0], rho=3.5, name="s")
        est = dist.estimator()
        self.assertIsNone(est.rho)
        refit = _estimate(est, self.data)
        self.assertGreater(refit.rho, 0.0)

    def test_exact_expected_sufficient_statistics_recover_rho(self):
        true_dist = SpearmanRankingDistribution([0, 1, 2], rho=0.7)
        perms = np.asarray(list(itertools.permutations(range(3))), dtype=float)
        probs = np.exp(true_dist.seq_log_density(perms))
        suff_stat = (1.0, np.dot(probs, perms))

        fitted = SpearmanRankingEstimator(3).estimate(None, suff_stat)

        np.testing.assert_array_equal(fitted.sigma, np.asarray([0, 1, 2]))
        self.assertAlmostEqual(fitted.rho, 0.7, places=10)

    def test_uniform_sufficient_statistics_estimate_zero_rho(self):
        perms = np.asarray(list(itertools.permutations(range(3))), dtype=float)
        suff_stat = (float(len(perms)), perms.sum(axis=0))

        fitted = SpearmanRankingEstimator(3).estimate(None, suff_stat)

        self.assertEqual(fitted.rho, 0.0)

    def test_concentrated_sufficient_statistics_hit_max_rho(self):
        fitted = SpearmanRankingEstimator(3, max_rho=123.0).estimate(None, (5.0, 5.0 * np.asarray([0, 1, 2])))

        self.assertEqual(fitted.rho, 123.0)

    def test_zero_data_branch_unchanged(self):
        dist = SpearmanRankingEstimator(4).estimate(None, (0.0, np.zeros(4)))
        self.assertEqual(dist.rho, 0.0)


class SpearmanRankingDimGuardTestCase(unittest.TestCase):
    # No size guard previously existed on the dim! permutation normalizer (every sibling ranking
    # distribution has one for its own exponential-cost construct) -- dim=13 alone would materialize
    # 6.2 billion rows, and dim=12 already costs ~46GB/~168s; unusable, not just slow.
    #
    # These tests use a tiny artificial max_dim (2 or 3) to exercise the guard mechanism itself --
    # raising above the configured limit is fast (it raises before ever materializing anything), but
    # SUCCEEDING requires an actual dim! materialization, so only test_default_max_dim_value_is_ten
    # below touches the real default (and only via the "exceeds it, raises immediately" direction,
    # which likewise never materializes).

    def test_default_max_dim_value_is_ten(self):
        with self.assertRaises(ValueError):
            SpearmanRankingDistribution(sigma=list(range(11)), rho=1.0)  # 11 > default max_dim=10

    def test_distribution_over_custom_max_dim_raises(self):
        with self.assertRaises(ValueError):
            SpearmanRankingDistribution(sigma=[0, 1, 2], rho=1.0, max_dim=2)  # 3 > custom max_dim=2

    def test_distribution_at_custom_max_dim_is_allowed(self):
        dist = SpearmanRankingDistribution(sigma=[0, 1, 2], rho=1.0, max_dim=3)  # 3 == custom max_dim
        self.assertTrue(np.isfinite(dist.log_const))

    def test_estimator_over_custom_max_dim_raises(self):
        with self.assertRaises(ValueError):
            SpearmanRankingEstimator(3, max_dim=2)  # 3 > custom max_dim=2 -- must raise at construction

    def test_estimator_round_trip_preserves_max_dim(self):
        # estimator()/estimate() must carry the SAME max_dim forward, not silently reset to the
        # default (the same "estimator round-trip drops metadata" bug class fixed elsewhere this
        # session) -- a distribution built with a non-default max_dim, refit through its own
        # estimator, must produce a result with that SAME max_dim, not the module default.
        dist = SpearmanRankingDistribution(sigma=[0, 1, 2], rho=1.0, max_dim=3)
        est = dist.estimator()
        self.assertEqual(est.max_dim, 3)
        refit = _estimate(est, [[0, 1, 2]])
        self.assertEqual(refit.max_dim, 3)


if __name__ == "__main__":
    unittest.main()
