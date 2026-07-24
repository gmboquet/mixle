"""Dirichlet-multinomial: pmf vs scipy, sampling, and Minka fixed-point MLE of alpha."""

import unittest
from itertools import product

import numpy as np
from scipy.stats import dirichlet_multinomial as sdm

from mixle.inference import estimate
from mixle.stats import DirichletMultinomialDistribution
from mixle.stats.multivariate.dirichlet_multinomial import DirichletMultinomialEstimator


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
