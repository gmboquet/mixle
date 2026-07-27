"""Dirichlet-multinomial: pmf vs scipy, sampling, and Minka fixed-point MLE of alpha."""

import unittest
from itertools import product

import numpy as np
from scipy.stats import dirichlet_multinomial as sdm

from mixle.inference import estimate
from mixle.stats import DirichletMultinomialDistribution
from mixle.stats.multivariate.dirichlet_multinomial import (
    DirichletMultinomialAccumulator,
    DirichletMultinomialEstimator,
    DirichletMultinomialResourceError,
)


class DirichletMultinomialTest(unittest.TestCase):
    def setUp(self):
        self.alpha = np.array([1.0, 2.0, 1.5])
        self.n = 8
        self.d = DirichletMultinomialDistribution(self.alpha, self.n)

    def test_log_pmf_matches_scipy(self):
        xs = np.array([[2, 3, 3], [8, 0, 0], [0, 4, 4], [3, 3, 2]])
        mine = self.d.seq_log_density(xs)
        ref = np.array([sdm.logpmf(x, self.alpha, self.n) for x in xs])
        np.testing.assert_allclose(mine, ref, atol=1e-10)
        np.testing.assert_allclose(mine, [self.d.log_density(x) for x in xs], atol=1e-12)

    def test_normalizes_over_support(self):
        support = [c for c in product(range(self.n + 1), repeat=3) if sum(c) == self.n]
        self.assertAlmostEqual(sum(self.d.density(np.array(c)) for c in support), 1.0, places=9)
        self.assertEqual(self.d.log_density(np.array([1, 1, 1])), -np.inf)  # total != n
        self.assertEqual(self.d.log_density(np.array([-1, 5, 4])), -np.inf)  # negative count

    def test_sampler_mean(self):
        s = self.d.sampler(seed=0).sample(40000)
        np.testing.assert_allclose(s.mean(axis=0), self.n * self.alpha / self.alpha.sum(), atol=0.06)

    def test_minka_mle_recovers_alpha(self):
        est = estimate(list(self.d.sampler(seed=1).sample(40000)), self.d.estimator())
        np.testing.assert_allclose(est.alpha, self.alpha, rtol=0.06, atol=0.06)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            DirichletMultinomialDistribution([1.0, 0.0, 1.0], 8)
        with self.assertRaises(ValueError):
            DirichletMultinomialDistribution([], 0)

    def test_alpha_is_owned_and_immutable(self):
        alpha = np.array([1.0, 2.0])
        dist = DirichletMultinomialDistribution(alpha, 2)
        before = dist.log_density([1, 1])
        alpha[:] = 100.0
        self.assertEqual(dist.log_density([1, 1]), before)
        with self.assertRaises(ValueError):
            dist.alpha[0] = 3.0

    def test_batch_shape_and_encoder_support_are_exact(self):
        with self.assertRaises(ValueError):
            self.d.seq_log_density(np.ones((2, 2)))
        encoder = self.d.dist_to_encoder()
        for invalid in (
            [[1.9, 0.1, 6.0]],
            [[-1.0, 1.0, 8.0]],
            [[1.0, 1.0, 1.0]],
            [[1.0, 1.0]],
            [[1.0, np.nan, 7.0]],
        ):
            with self.assertRaises(ValueError):
                encoder.seq_encode(invalid)

    def test_accumulator_rejects_unsupported_evidence_and_weights(self):
        accumulator = DirichletMultinomialAccumulator(2, 2)
        for invalid in ([1.9, 0.1], [-1.0, 3.0], [1.0, 0.0], [np.nan, 2.0]):
            with self.assertRaises(ValueError):
                accumulator.update(invalid, 1.0, None)
        with self.assertRaises(ValueError):
            accumulator.update([1, 1], -1.0, None)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.array([[1, 1], [2, 0]]), np.ones(1), None)

    def test_zero_trial_statistic_round_trip_preserves_logical_n(self):
        accumulator = DirichletMultinomialAccumulator(2, 0)
        accumulator.update([0, 0], 1.0, None)
        serialized = accumulator.value()
        self.assertEqual(serialized[2], 0)
        restored = DirichletMultinomialAccumulator(2, 0).from_value(serialized)
        self.assertEqual(restored.n, 0)
        self.assertEqual(restored.value()[2], 0)
        np.testing.assert_array_equal(restored.c, np.zeros((2, 1)))

    def test_sparse_fit_is_regularized_and_receipted(self):
        estimator = DirichletMultinomialEstimator(2, 2)
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(np.array([[2, 0], [2, 0]]), np.ones(2), None)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertTrue(np.all(fitted.alpha > 0.0))
        self.assertIn(1, fitted.fit_receipt.regularized_categories)
        self.assertTrue(fitted.fit_receipt.identifiable)
        self.assertGreaterEqual(len(fitted.fit_receipt.objective_history), 2)

    def test_estimator_controls_and_statistics_fail_closed(self):
        for kwargs in (
            {"dim": 0, "n": 2},
            {"dim": 2, "n": 2, "max_iter": 0},
            {"dim": 2, "n": 2, "tol": np.nan},
            {"dim": 2, "n": 2, "min_alpha": 0.0},
        ):
            with self.assertRaises((TypeError, ValueError)):
                DirichletMultinomialEstimator(**kwargs)
        estimator = DirichletMultinomialEstimator(2, 2)
        with self.assertRaises(ValueError):
            estimator.estimate(None, (np.zeros((2, 2)), 1.0, 3))
        with self.assertRaises(ValueError):
            estimator.estimate(None, (np.full((2, 2), np.nan), 1.0, 2))

    def test_nonconvergence_is_explicit(self):
        estimator = DirichletMultinomialEstimator(2, 2, max_iter=1, tol=1.0e-30)
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(np.array([[2, 0], [1, 1]]), np.ones(2), None)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertFalse(fitted.fit_receipt.converged)
        self.assertEqual(fitted.fit_receipt.iterations, 1)

    def test_allocation_and_frontier_budgets_fail_before_growth(self):
        with self.assertRaises(DirichletMultinomialResourceError):
            DirichletMultinomialAccumulator(10, 10, max_recurrence_cells=99)
        dist = DirichletMultinomialDistribution(
            [1.0, 1.0],
            3,
            max_frontier_entries=1,
        )
        enumerator = dist.enumerator()
        with self.assertRaises(DirichletMultinomialResourceError):
            next(enumerator)
        with self.assertRaises(DirichletMultinomialResourceError):
            dist.enumerator(max_recurrence_cells=1)

    def test_fractional_counts_score_neg_inf(self):
        # A per-category fractional count is not a valid Dirichlet-multinomial outcome even when the
        # (real-valued) total happens to equal n -- e.g. [1.5, 2.5, 4.0] sums to 8 -- so the sum-to-n
        # check alone is not sufficient; each entry must individually be an integer.
        d2 = DirichletMultinomialDistribution(np.array([2.0, 3.0]), 4)
        self.assertEqual(d2.log_density(np.array([1.5, 2.5])), -np.inf)
        self.assertEqual(d2.density(np.array([1.5, 2.5])), 0.0)
        seq = d2.seq_log_density(np.array([[1.5, 2.5], [2.0, 2.0], [0.5, 3.5]]))
        np.testing.assert_array_equal(seq, [-np.inf, d2.log_density(np.array([2, 2])), -np.inf])
        # NaN/inf entries are likewise rejected.
        self.assertEqual(d2.log_density(np.array([float("nan"), 4.0])), -np.inf)
        self.assertEqual(d2.log_density(np.array([float("inf"), 4.0])), -np.inf)

    def test_pseudo_count_raises_rather_than_silently_ignored(self):
        # The Minka fixed-point MLE operates on a cumulative-count recurrence statistic, not a
        # simple additive raw moment, so pseudo_count cannot be cleanly blended in the way the
        # method-of-moments estimators (Gumbel, Weibull, ...) do. Rather than silently no-op-ing
        # (previously: pseudo_count was accepted, not even stored, and had zero effect), it must
        # now raise explicitly, both via the distribution factory method and the estimator
        # constructor directly.
        with self.assertRaises(ValueError):
            self.d.estimator(pseudo_count=1.0)
        with self.assertRaises(ValueError):
            DirichletMultinomialEstimator(self.d.dim, self.n, pseudo_count=1.0)
        # pseudo_count=None (the default) must remain unaffected.
        est = self.d.estimator()
        self.assertIsInstance(est, DirichletMultinomialEstimator)


if __name__ == "__main__":
    unittest.main()
