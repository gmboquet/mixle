"""Bayesian (conjugate Beta) behavior folded onto the Bernoulli/Geometric/Binomial leaves.

Mirrors stats_bayes_gaussian_test.py: each leaf gains conjugate posterior estimation,
``expected_log_density``, and a posterior-returning ``fit`` while its MLE path stays
byte-identical. Conjugate behavior is pinned against the textbook Beta posterior closed form,
the digamma expected-log-density formula, and scalar-vs-seq self-consistency.
"""

import unittest

import numpy as np

from pysp.stats.bernoulli import BernoulliDistribution, BernoulliEstimator
from pysp.stats.beta import BetaDistribution
from pysp.stats.binomial import BinomialDistribution, BinomialEstimator
from pysp.stats.geometric import GeometricDistribution, GeometricEstimator
from pysp.utils.special import digamma


class StatsBayesBernoulliTestCase(unittest.TestCase):
    def setUp(self):
        self.psum, self.nsum = 37.0, 63.0
        self.count = self.psum + self.nsum

    def test_mle_path_unchanged(self):
        """No prior -> plain MLE point estimate; estimator/dist carry no posterior."""
        m = BernoulliEstimator().estimate(None, (self.count, self.psum))
        self.assertAlmostEqual(m.p, self.psum / self.count, places=12)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_pseudo_count_path_unchanged(self):
        """pseudo_count path is untouched by the additive conjugate machinery."""
        m = BernoulliEstimator(pseudo_count=5.0, suff_stat=0.2).estimate(None, (self.count, self.psum))
        expected = (self.psum + 5.0 * 0.2) / (self.count + 5.0)
        self.assertAlmostEqual(m.p, expected, places=12)

    def test_conjugate_posterior_closed_form(self):
        """estimate() with a Beta prior matches the textbook posterior + MAP mode."""
        a, b = 2.3, 4.7
        m = BernoulliEstimator(prior=BetaDistribution(a, b)).estimate(None, (self.count, self.psum))
        post = m.get_prior()
        self.assertAlmostEqual(post.a, a + self.psum, places=10)
        self.assertAlmostEqual(post.b, b + self.nsum, places=10)
        self.assertAlmostEqual(m.p, (self.psum + a - 1.0) / (self.count + a + b - 2.0), places=10)

    def test_expected_log_density_formula(self):
        """expected_log_density equals the digamma closed form and falls back without a prior."""
        a, b = 3.0, 5.0
        d = BernoulliDistribution(0.4, prior=BetaDistribution(a, b))
        dab = digamma(a + b)
        self.assertAlmostEqual(d.expected_log_density(True), digamma(a) - dab, places=12)
        self.assertAlmostEqual(d.expected_log_density(False), digamma(b) - dab, places=12)
        xs = np.array([True, False, True, False])
        self.assertTrue(
            np.allclose(d.seq_expected_log_density(xs), [d.expected_log_density(x) for x in xs], atol=1e-12)
        )
        d0 = BernoulliDistribution(0.4)
        self.assertAlmostEqual(d0.expected_log_density(True), d0.log_density(True), places=12)


class StatsBayesGeometricTestCase(unittest.TestCase):
    def setUp(self):
        self.count, self.sum = 80.0, 250.0

    def test_mle_path_unchanged(self):
        m = GeometricEstimator().estimate(None, (self.count, self.sum))
        self.assertAlmostEqual(m.p, self.count / self.sum, places=12)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_conjugate_posterior_closed_form(self):
        a, b = 1.7, 3.1
        m = GeometricEstimator(prior=BetaDistribution(a, b)).estimate(None, (self.count, self.sum))
        post = m.get_prior()
        self.assertAlmostEqual(post.a, a + self.count, places=10)
        self.assertAlmostEqual(post.b, b + self.sum - self.count, places=10)
        self.assertAlmostEqual(m.p, (post.a - 1.0) / (post.a + post.b - 2.0), places=10)

    def test_expected_log_density_formula(self):
        a, b = 4.0, 6.0
        d = GeometricDistribution(0.3, prior=BetaDistribution(a, b))
        ga, gb, gab = digamma(a), digamma(b), digamma(a + b)
        for x in (1, 2, 5):
            self.assertAlmostEqual(d.expected_log_density(x), (gb - gab) * (x - 1) + (ga - gab), places=12)
        self.assertEqual(d.expected_log_density(0), -np.inf)
        xs = np.array([1.0, 2.0, 5.0])
        self.assertTrue(
            np.allclose(d.seq_expected_log_density(xs), [d.expected_log_density(x) for x in xs], atol=1e-12)
        )
        d0 = GeometricDistribution(0.3)
        self.assertAlmostEqual(d0.expected_log_density(3), d0.log_density(3), places=12)


class StatsBayesBinomialTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.count, self.sum = 50.0, 230.0

    def test_mle_path_unchanged(self):
        """No prior -> existing MLE/min-max path is byte-identical."""
        m = BinomialEstimator(max_val=self.n, min_val=0).estimate(None, (self.count, self.sum, 0, self.n))
        self.assertAlmostEqual(m.p, self.sum / (self.count * self.n), places=12)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_conjugate_posterior_closed_form(self):
        a, b = 2.1, 5.5
        psum = self.sum
        fsum = self.count * self.n - psum
        m = BinomialEstimator(max_val=self.n, min_val=0, prior=BetaDistribution(a, b)).estimate(
            None, (self.count, self.sum, 0, self.n)
        )
        post = m.get_prior()
        self.assertAlmostEqual(post.a, a + psum, places=10)
        self.assertAlmostEqual(post.b, b + fsum, places=10)
        self.assertAlmostEqual(m.p, (post.a - 1.0) / (post.a + post.b - 2.0), places=10)
        self.assertEqual(m.n, self.n)

    def test_expected_log_density_formula(self):
        from pysp.utils.vector import gammaln

        a, b = 3.0, 4.0
        d = BinomialDistribution(0.4, self.n, prior=BetaDistribution(a, b))
        e1 = digamma(a) - digamma(a + b)
        e2 = digamma(b) - digamma(a + b)
        for x in (0, 3, 7, 10):
            cc = gammaln(self.n + 1) - gammaln(x + 1) - gammaln(self.n - x + 1)
            self.assertAlmostEqual(d.expected_log_density(x), cc + x * e1 + (self.n - x) * e2, places=12)
        self.assertEqual(d.expected_log_density(11), -np.inf)
        enc = d.dist_to_encoder().seq_encode([0, 3, 7, 10])
        self.assertTrue(
            np.allclose(
                d.seq_expected_log_density(enc),
                [d.expected_log_density(x) for x in (0, 3, 7, 10)],
                atol=1e-12,
            )
        )
        d0 = BinomialDistribution(0.4, self.n)
        self.assertAlmostEqual(d0.expected_log_density(3), d0.log_density(3), places=12)


if __name__ == "__main__":
    unittest.main()
