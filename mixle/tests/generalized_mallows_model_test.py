"""Generalized Mallows Model (per-stage dispersion): factorization, normalization, recovery."""

import itertools
import math
import unittest

import numpy as np

from mixle.stats import GeneralizedMallowsModelDistribution
from mixle.stats.rankings._permutation_kernels import permutation_distance, seq_rim_code
from mixle.stats.rankings.generalized_mallows_model import (
    GeneralizedMallowsModelDataEncoder,
    GeneralizedMallowsModelEstimator,
)


class GMMTest(unittest.TestCase):
    def test_near_zero_stage_theta_is_the_uniform_limit(self):
        dist = GeneralizedMallowsModelDistribution([0, 1, 2], [1.0e-309, 1.0e-309])
        self.assertAlmostEqual(dist.log_z, math.log(6.0), places=12)
        self.assertAlmostEqual(dist.density([2, 1, 0]), 1.0 / 6.0, places=12)

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
        self.assertTrue(fit.fit_diagnostics.center_exact)
        self.assertEqual(fit.fit_diagnostics.center_algorithm, "exact_enumeration")

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

    def test_log_density_rejects_non_permutations(self):
        # A malformed x isn't just an unlikely ordering, it isn't an ordering at all: log_density
        # (and density/seq_log_density, which it's built on) must reject it rather than silently
        # returning a finite score, matching the encoder's validation above.
        dist = GeneralizedMallowsModelDistribution([0, 1, 2], [1.0, 1.0])
        with self.assertRaises(ValueError):
            dist.log_density([0, 0, 1])  # repeated item
        with self.assertRaises(ValueError):
            dist.log_density([0, 1, 5])  # out of range
        with self.assertRaises(ValueError):
            dist.log_density([0, 1])  # wrong length
        with self.assertRaises(ValueError):
            dist.log_density([-1, 1, 2])  # negative index would otherwise alias a valid rank
        with self.assertRaises(ValueError):
            dist.log_density([0.5, 1.0, 2.0])  # fractional entry would otherwise truncate to 0

    def test_density_and_seq_log_density_reject_non_permutations(self):
        dist = GeneralizedMallowsModelDistribution([0, 1, 2], [1.0, 1.0])
        with self.assertRaises(ValueError):
            dist.density([0, 0, 1])
        with self.assertRaises(ValueError):
            dist.seq_log_density(np.array([[0, 0, 1]]))
        with self.assertRaises(ValueError):
            # log_density forwards x as float (not int64) precisely so this fractional check
            # inside seq_log_density can see it; exercise seq_log_density directly too.
            dist.seq_log_density(np.array([[0.5, 1.0, 2.0]]))

    def test_encoder_rejects_non_permutations(self):
        dist = GeneralizedMallowsModelDistribution([0, 1, 2], [1.0, 1.0])
        with self.assertRaises(ValueError):
            dist.dist_to_encoder().seq_encode([[0, 1, 1]])
        with self.assertRaises(ValueError):
            # A fractional entry must not be silently truncated to an in-range integer by the
            # encoder's int cast before the permutation check ever sees it.
            dist.dist_to_encoder().seq_encode([[0.5, 1.0, 2.0]])

    def test_distribution_owns_immutable_center_and_dispersion(self):
        center = np.asarray([2, 0, 1])
        theta = np.asarray([1.0, 2.0])
        dist = GeneralizedMallowsModelDistribution(center, theta)
        center[:] = [0, 1, 2]
        theta[:] = 0.0
        np.testing.assert_array_equal(dist.sigma0, [2, 0, 1])
        np.testing.assert_array_equal(dist.theta, [1.0, 2.0])
        with self.assertRaises(ValueError):
            dist.sigma0[0] = 0
        with self.assertRaises(ValueError):
            dist.theta[0] = 0.0

    def test_encoder_identity_includes_dimension(self):
        self.assertEqual(GeneralizedMallowsModelDataEncoder(3), GeneralizedMallowsModelDataEncoder(3))
        self.assertNotEqual(GeneralizedMallowsModelDataEncoder(3), GeneralizedMallowsModelDataEncoder(4))
        self.assertIn("dim=3", str(GeneralizedMallowsModelDataEncoder(3)))

    def test_exact_support_accumulator_is_merge_order_independent(self):
        estimator = GeneralizedMallowsModelEstimator(3, reservoir=2)

        def shard(row):
            accumulator = estimator.accumulator_factory().make()
            accumulator.update(row, 1.0, None)
            return accumulator

        first, second = shard([0, 1, 2]), shard([2, 1, 0])
        left = shard([0, 1, 2]).combine(second.value())
        right = shard([2, 1, 0]).combine(first.value())
        fit_left = estimator.estimate(None, left.value())
        fit_right = estimator.estimate(None, right.value())
        np.testing.assert_array_equal(fit_left.sigma0, fit_right.sigma0)
        np.testing.assert_allclose(fit_left.theta, fit_right.theta)

    def test_accumulator_fails_before_dropping_evidence_and_copies_receipts(self):
        estimator = GeneralizedMallowsModelEstimator(3, reservoir=1)
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([0, 1, 2], 1.0, None)
        before = accumulator.value()
        with self.assertRaises(MemoryError):
            accumulator.update([2, 1, 0], 1.0, None)
        after = accumulator.value()
        self.assertEqual(after[0], before[0])
        np.testing.assert_array_equal(after[1], before[1])
        before[1][:] = 0.0
        before[2][0][:] = 0
        self.assertGreater(accumulator.value()[1].sum(), 0.0)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.asarray([[0, 1, 2]]), np.asarray([-1.0]), None)

    def test_pseudo_count_and_approximate_center_are_explicit(self):
        base = GeneralizedMallowsModelDistribution([2, 0, 1], [1.0, 0.5])
        fitted = base.estimator(pseudo_count=2.0).estimate(None, (0.0, np.zeros((3, 3)), [], []))
        self.assertTrue(fitted.fit_diagnostics.regularized)
        self.assertEqual(fitted.fit_diagnostics.pseudo_count, 2.0)
        np.testing.assert_array_equal(fitted.sigma0, base.sigma0)

        rows = [[0, 1, 2, 3], [1, 0, 2, 3]]
        accumulator = GeneralizedMallowsModelEstimator(4, reservoir=2).accumulator_factory().make()
        accumulator.seq_update(np.asarray(rows), np.ones(2), None)
        with self.assertRaisesRegex(ValueError, "exact stage-wise Mallows center search"):
            GeneralizedMallowsModelEstimator(4, center_exact_cap=3).estimate(None, accumulator.value())
        approximate = GeneralizedMallowsModelEstimator(
            4,
            center_exact_cap=3,
            allow_approximate_center=True,
        ).estimate(None, accumulator.value())
        self.assertFalse(approximate.fit_diagnostics.center_exact)
        self.assertEqual(approximate.fit_diagnostics.center_algorithm, "copeland_approximation")

    def test_controls_and_sample_size_are_validated(self):
        invalid = (
            {"dim": 1},
            {"dim": 3.5},
            {"dim": 3, "reservoir": 0},
            {"dim": 3, "pseudo_count": -1.0},
            {"dim": 3, "center_exact_cap": 1},
            {"dim": 3, "allow_approximate_center": "false"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                GeneralizedMallowsModelEstimator(**kwargs)
        sampler = GeneralizedMallowsModelDistribution([0, 1, 2]).sampler()
        with self.assertRaises(ValueError):
            sampler.sample(-1)
        with self.assertRaises(ValueError):
            sampler.sample(1.5)


if __name__ == "__main__":
    unittest.main()
