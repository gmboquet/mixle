"""Beta-binomial: pmf vs scipy, sampling, and method-of-moments recovery of (a, b)."""

import unittest

import numpy as np
from scipy.stats import betabinom

from mixle.inference import estimate
from mixle.stats import BetaBinomialDistribution


class BetaBinomialTest(unittest.TestCase):
    def setUp(self):
        self.n, self.a, self.b = 10, 2.0, 3.0
        self.d = BetaBinomialDistribution(self.n, self.a, self.b)

    def test_log_pmf_matches_scipy(self):
        ks = np.arange(0, self.n + 1)
        mine = self.d.seq_log_density(self.d.dist_to_encoder().seq_encode(ks))
        np.testing.assert_allclose(mine, betabinom.logpmf(ks, self.n, self.a, self.b), atol=1e-10)
        np.testing.assert_allclose(mine, [self.d.log_density(int(k)) for k in ks], atol=1e-12)

    def test_normalizes_and_support(self):
        ks = np.arange(0, self.n + 1)
        self.assertAlmostEqual(np.exp(self.d.seq_log_density(ks)).sum(), 1.0, places=9)
        self.assertEqual(self.d.log_density(self.n + 1), -np.inf)
        self.assertEqual(self.d.log_density(-1), -np.inf)

    def test_sampler_matches_pmf(self):
        s = np.array(self.d.sampler(seed=0).sample(50000))
        emp = np.bincount(s, minlength=self.n + 1) / len(s)
        ref = betabinom.pmf(np.arange(self.n + 1), self.n, self.a, self.b)
        self.assertLess(np.abs(emp - ref).max(), 0.01)

    def test_moment_estimator_recovers_params(self):
        est = estimate(list(self.d.sampler(seed=1).sample(50000)), self.d.estimator())
        self.assertAlmostEqual(est.a, self.a, delta=0.2)
        self.assertAlmostEqual(est.b, self.b, delta=0.3)
        self.assertAlmostEqual(est.a / (est.a + est.b), self.a / (self.a + self.b), delta=0.02)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            BetaBinomialDistribution(10, 0.0, 1.0)

    def test_fractional_count_scores_neg_inf(self):
        # A fractional count is not a valid beta-binomial outcome: the pmf is a ratio of Gamma/Beta
        # functions that -- unless explicitly guarded -- happily evaluates at any real-valued "count"
        # via the smooth continuation of gammaln/betaln, silently returning finite mass for k=2.5.
        self.assertEqual(self.d.log_density(2.5), -np.inf)
        self.assertEqual(self.d.density(2.5), 0.0)
        seq = self.d.seq_log_density(np.array([2.5, 2.0, 3.5, 7.0]))
        np.testing.assert_array_equal(seq, [-np.inf, self.d.log_density(2), -np.inf, self.d.log_density(7)])
        # NaN/inf counts are likewise rejected rather than propagating through gammaln.
        self.assertEqual(self.d.log_density(float("nan")), -np.inf)
        self.assertEqual(self.d.log_density(float("inf")), -np.inf)

    def test_pseudo_count_smooths_toward_prior(self):
        # estimator(pseudo_count=...) previously ignored pseudo_count entirely -- a silent no-op.
        est = self.d.estimator(pseudo_count=1.0e6)
        self.assertIsNotNone(est.suff_stat)
        fitted = est.estimate(None, (1.0, 100.0, 1.0))  # one observation far from the prior mean
        self.assertAlmostEqual(fitted.a, self.a, places=2)
        self.assertAlmostEqual(fitted.b, self.b, places=2)

        fitted_plain = self.d.estimator().estimate(None, (1.0, 100.0, 1.0))
        self.assertNotAlmostEqual(fitted_plain.a, self.a, places=1)

    def test_pseudo_count_recovers_exact_prior_moments(self):
        # feeding the stored suff_stat back through with zero real data (pure pseudo-count blend)
        # must reproduce this distribution's own (a, b) almost exactly.
        est = self.d.estimator(pseudo_count=1.0)
        fitted = est.estimate(None, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(fitted.a, self.a, places=6)
        self.assertAlmostEqual(fitted.b, self.b, places=6)
        self.assertEqual(fitted.n, self.n)


if __name__ == "__main__":
    unittest.main()
