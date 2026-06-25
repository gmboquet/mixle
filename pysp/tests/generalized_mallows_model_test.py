"""Generalized Mallows Model (per-stage dispersion): factorization, normalization, recovery."""

import itertools
import unittest

import numpy as np

from pysp.stats import GeneralizedMallowsModelDistribution
from pysp.stats.rankings._permutation_kernels import permutation_distance, seq_rim_code


class GMMTest(unittest.TestCase):
    def test_rim_code_sums_to_kendall(self):
        rng = np.random.RandomState(0)
        sigma0 = np.array([2, 0, 3, 1, 4])
        X = np.array([rng.permutation(5) for _ in range(60)])
        j = seq_rim_code(X, sigma0)
        kd = np.array([permutation_distance(x, sigma0, "kendall") for x in X])
        np.testing.assert_array_equal(j.sum(axis=1), kd)
        self.assertTrue(np.all(j <= np.arange(1, 5)[None, :]))  # J_i in {0..i}

    def test_density_sums_to_one(self):
        d = GeneralizedMallowsModelDistribution([2, 0, 3, 1, 4], [2.0, 1.0, 0.3, 0.1])
        self.assertAlmostEqual(sum(d.density(list(p)) for p in itertools.permutations(range(5))), 1.0, places=10)

    def test_seq_matches_scalar(self):
        d = GeneralizedMallowsModelDistribution([3, 1, 0, 2], [1.5, 0.7, 0.3])
        perms = np.array(list(itertools.permutations(range(4))))
        np.testing.assert_allclose(d.seq_log_density(perms), [d.log_density(p) for p in perms], atol=1e-12)

    def test_recovers_center_and_per_stage_theta(self):
        center = [3, 1, 4, 0, 2, 5]
        true = GeneralizedMallowsModelDistribution(center, [2.5, 1.8, 1.3, 1.0, 0.8])
        samp = true.sampler(seed=1).sample(10000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        self.assertEqual(list(fit.sigma0), center)
        np.testing.assert_allclose(fit.theta, true.theta, atol=0.2)

    def test_distinct_stage_dispersions_are_learned(self):
        # a firm-top / loose-bottom truth must produce a decreasing fitted theta profile
        true = GeneralizedMallowsModelDistribution([0, 1, 2, 3, 4, 5], [3.0, 2.2, 1.6, 1.1, 0.7])
        samp = true.sampler(seed=4).sample(8000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        self.assertTrue(np.all(np.diff(fit.theta) < 0))  # strictly decreasing dispersion recovered

    def test_validation(self):
        with self.assertRaises(ValueError):
            GeneralizedMallowsModelDistribution([0, 1, 2], [1.0])  # theta must be length n-1 = 2
        with self.assertRaises(ValueError):
            GeneralizedMallowsModelDistribution([0, 1, 2], [1.0, -1.0])  # negative dispersion


if __name__ == "__main__":
    unittest.main()
