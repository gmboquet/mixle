"""Bayesian (conjugate / variational) behavior folded onto the multivariate Gaussian group.

Covers MultivariateGaussian (NormalWishart prior), DiagonalGaussian
(MultivariateNormalGamma prior), and LogGaussian (NormalGamma prior). Each test
asserts the conjugate posterior closed forms, the variational ``expected_log_density``
(with scalar-vs-seq self-consistency), and that the MLE path is unchanged when no prior
is attached.
"""

import unittest

import numpy as np

from pysp.stats.bayes.mvngamma import MultivariateNormalGammaDistribution
from pysp.stats.bayes.normgamma import NormalGammaDistribution
from pysp.stats.bayes.normwishart import NormalWishartDistribution
from pysp.stats.leaf.log_gaussian import LogGaussianDistribution, LogGaussianEstimator
from pysp.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution, DiagonalGaussianEstimator
from pysp.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from pysp.utils.special import digamma


class StatsBayesMvGaussianGroupTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(13)
        self.d = 3
        self.X = rng.randn(120, self.d) * np.array([1.5, 0.7, 2.0]) + np.array([1.0, -2.0, 0.5])
        self.count = float(self.X.shape[0])
        self.xsum = self.X.sum(axis=0)
        self.outer = self.X.T @ self.X
        self.sum2 = (self.X * self.X).sum(axis=0)
        # positive data for log-gaussian
        self.xp = np.abs(rng.randn(150)) + 0.2
        self.lx = np.log(self.xp)
        self.lsum = float(self.lx.sum())
        self.lsum2 = float((self.lx * self.lx).sum())
        self.n = float(len(self.xp))

    # ----------------------------- MVN ----------------------------------
    def test_mvn_mle_path_unchanged(self):
        """No prior -> plain MLE; no posterior carried."""
        m = MultivariateGaussianEstimator(dim=self.d).estimate(None, (self.xsum, self.outer, self.count))
        mu = self.xsum / self.count
        covar = self.outer / self.count - np.outer(mu, mu)
        self.assertTrue(np.allclose(m.mu, mu, atol=1e-12))
        self.assertTrue(np.allclose(m.covar, covar, atol=1e-12))
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_mvn_conjugate_posterior_closed_form(self):
        """NormalWishart conjugate update matches the textbook posterior."""
        m0 = np.zeros(self.d)
        kappa0, nu0 = 1e-2, self.d + 3.0
        W0 = np.eye(self.d) * 0.7
        prior = NormalWishartDistribution(m0, kappa0, W0, nu0)
        m = MultivariateGaussianEstimator(dim=self.d, prior=prior).estimate(None, (self.xsum, self.outer, self.count))
        pmu, pkappa, pW, pnu = m.get_prior().get_parameters()
        self.assertAlmostEqual(pkappa, kappa0 + self.count, places=8)
        self.assertAlmostEqual(pnu, nu0 + self.count, places=8)
        m_n = (kappa0 * m0 + self.xsum) / (kappa0 + self.count)
        self.assertTrue(np.allclose(pmu, m_n, atol=1e-8))
        xbar = self.xsum / self.count
        scatter = self.outer - self.count * np.outer(xbar, xbar)
        dmu = xbar - m0
        w_n_inv = np.linalg.inv(W0) + scatter + (kappa0 * self.count / pkappa) * np.outer(dmu, dmu)
        w_n_inv = 0.5 * (w_n_inv + w_n_inv.T)
        self.assertTrue(np.allclose(m.covar, w_n_inv / (pnu - self.d), atol=1e-8))
        self.assertTrue(np.allclose(pW, np.linalg.inv(w_n_inv), atol=1e-8))

    def test_mvn_expected_log_density_formula(self):
        """expected_log_density equals the VB E[log p] closed form; falls back without a prior."""
        m0 = np.array([0.1, -0.2, 0.3])
        kappa, nu = 2.0, self.d + 5.0
        W = np.diag([0.5, 0.8, 1.2])
        prior = NormalWishartDistribution(m0, kappa, W, nu)
        d = MultivariateGaussianDistribution(m0, np.diag([1.0, 1.0, 1.0]), prior=prior)
        e_log_det = prior.expected_log_det()
        for x in self.X[:5]:
            diff = x - m0
            e_quad = self.d / kappa + nu * float(diff @ W @ diff)
            want = 0.5 * e_log_det - 0.5 * self.d * np.log(2.0 * np.pi) - 0.5 * e_quad
            self.assertAlmostEqual(d.expected_log_density(x), want, places=10)
        self.assertTrue(
            np.allclose(
                d.seq_expected_log_density(self.X[:5]),
                [d.expected_log_density(x) for x in self.X[:5]],
                atol=1e-12,
            )
        )
        d0 = MultivariateGaussianDistribution(m0, np.eye(self.d))
        self.assertAlmostEqual(d0.expected_log_density(self.X[0]), d0.log_density(self.X[0]), places=12)

    def test_mvn_seq_eld_matches_scalar(self):
        """Full-batch seq_expected_log_density matches the per-row scalar value."""
        m0 = np.zeros(self.d)
        prior_args = (m0, 1e-2, np.eye(self.d) * 0.7, self.d + 3.0)
        sm = MultivariateGaussianEstimator(dim=self.d, prior=NormalWishartDistribution(*prior_args)).estimate(
            None, (self.xsum, self.outer, self.count)
        )
        scalar = np.asarray([sm.expected_log_density(x) for x in self.X])
        self.assertTrue(np.allclose(sm.seq_expected_log_density(self.X), scalar, atol=1e-12))

    # ------------------------- DiagonalGaussian -------------------------
    def test_dmvn_mle_path_unchanged(self):
        """No prior -> plain MLE; no posterior carried."""
        m = DiagonalGaussianEstimator(dim=self.d).estimate(None, (self.xsum, self.sum2, self.count))
        mu = self.xsum / self.count
        covar = self.sum2 / self.count - mu * mu
        self.assertTrue(np.allclose(m.mu, mu, atol=1e-12))
        self.assertTrue(np.allclose(m.covar, covar, atol=1e-12))
        self.assertFalse(m.has_conj_prior)

    def test_dmvn_conjugate_posterior_closed_form(self):
        """Per-component NormalGamma conjugate update matches the closed form."""
        mu0 = np.zeros(self.d)
        lam0 = np.ones(self.d) * 1e-2
        a0 = np.ones(self.d) * 1.1
        b0 = np.ones(self.d) * 0.9
        prior = MultivariateNormalGammaDistribution(mu0, lam0, a0, b0)
        m = DiagonalGaussianEstimator(dim=self.d, prior=prior).estimate(None, (self.xsum, self.sum2, self.count))
        pmu, plam, pa, pb = m.get_prior().get_parameters()
        self.assertTrue(np.allclose(plam, lam0 + self.count, atol=1e-8))
        self.assertTrue(np.allclose(pa, a0 + self.count / 2.0, atol=1e-8))
        self.assertTrue(np.allclose(pmu, (self.xsum + mu0 * lam0) / (lam0 + self.count), atol=1e-8))
        mean = self.xsum / self.count
        b0n = self.sum2 - mean * self.xsum
        b1n = (lam0 * self.count / plam) * (mean - mu0) ** 2
        self.assertTrue(np.allclose(pb, b0 + 0.5 * (b0n + b1n), atol=1e-8))
        self.assertTrue(np.allclose(m.covar, pb / (pa - 0.5), atol=1e-8))

    def test_dmvn_expected_log_density_formula(self):
        """expected_log_density equals the VB closed form; falls back without a prior."""
        mu0 = np.array([0.1, -0.2, 0.3])
        lam0 = np.array([2.0, 1.5, 3.0])
        a0 = np.array([4.0, 3.0, 5.0])
        b0 = np.array([1.0, 2.0, 1.5])
        prior = MultivariateNormalGammaDistribution(mu0, lam0, a0, b0)
        d = DiagonalGaussianDistribution(mu0, b0 / (a0 - 0.5), prior=prior)
        ea = np.sum((mu0 * mu0) * (a0 / b0) * 0.5 + 0.5 / lam0 + 0.5 * (np.log(b0) - digamma(a0)))
        e1 = mu0 * a0 / b0
        e2 = -0.5 * a0 / b0
        eb = -0.5 * np.log(2 * np.pi) * self.d
        for x in self.X[:5]:
            want = float(np.dot(x, e1) + np.dot(x * x, e2) - ea + eb)
            self.assertAlmostEqual(d.expected_log_density(x), want, places=10)
        self.assertTrue(
            np.allclose(
                d.seq_expected_log_density(self.X[:5]),
                [d.expected_log_density(x) for x in self.X[:5]],
                atol=1e-12,
            )
        )
        d0 = DiagonalGaussianDistribution(mu0, np.ones(self.d))
        self.assertAlmostEqual(d0.expected_log_density(self.X[0]), d0.log_density(self.X[0]), places=12)

    def test_dmvn_seq_eld_matches_scalar(self):
        """Full-batch seq_expected_log_density matches the per-row scalar value."""
        args = (np.zeros(self.d), np.ones(self.d) * 1e-2, np.ones(self.d) * 1.1, np.ones(self.d) * 0.9)
        sm = DiagonalGaussianEstimator(dim=self.d, prior=MultivariateNormalGammaDistribution(*args)).estimate(
            None, (self.xsum, self.sum2, self.count)
        )
        scalar = np.asarray([sm.expected_log_density(x) for x in self.X])
        self.assertTrue(np.allclose(sm.seq_expected_log_density(self.X), scalar, atol=1e-12))

    # ----------------------------- LogGaussian --------------------------
    def test_log_gaussian_mle_path_unchanged(self):
        """No prior -> plain MLE; no posterior carried."""
        m = LogGaussianEstimator().estimate(None, (self.lsum, self.lsum2, self.n, self.n))
        mu = self.lsum / self.n
        sigma2 = np.sum(self.lsum2 - self.lsum**2 / self.n) / self.n
        self.assertAlmostEqual(m.mu, mu, places=10)
        self.assertAlmostEqual(m.sigma2, sigma2, places=10)
        self.assertFalse(m.has_conj_prior)

    def test_log_gaussian_conjugate_posterior_closed_form(self):
        """NormalGamma conjugate update on log-scale statistics matches the closed form."""
        mu0, lam0, a0, b0 = 0.0, 1e-2, 1.2, 0.8
        prior = NormalGammaDistribution(mu0, lam0, a0, b0)
        m = LogGaussianEstimator(prior=prior).estimate(None, (self.lsum, self.lsum2, self.n, self.n))
        pmu, plam, pa, pb = m.get_prior().get_parameters()
        self.assertAlmostEqual(plam, lam0 + self.n, places=8)
        self.assertAlmostEqual(pa, a0 + self.n / 2.0, places=8)
        self.assertAlmostEqual(pmu, (self.lsum + mu0 * lam0) / (lam0 + self.n), places=8)
        mean = self.lsum / self.n
        b0n = self.lsum2 - mean * self.lsum
        b1n = (lam0 * self.n / plam) * (mean - mu0) ** 2
        self.assertAlmostEqual(pb, b0 + 0.5 * (b0n + b1n), places=8)
        self.assertAlmostEqual(m.sigma2, pb / (pa - 0.5), places=10)

    def test_log_gaussian_expected_log_density_formula(self):
        """expected_log_density equals the VB closed form at log(x), with Jacobian -log(x)."""
        prior = NormalGammaDistribution(0.3, 2.0, 4.0, 5.0)
        d = LogGaussianDistribution(0.3, 5.0 / (4.0 - 0.5), prior=prior)
        mu, lam, a, b = prior.get_parameters()
        ea = (mu * mu) * (a / b) * 0.5 + 0.5 / lam + 0.5 * (np.log(b) - digamma(a))
        e1 = mu * a / b
        e2 = -0.5 * a / b
        eb = -0.5 * np.log(2 * np.pi)
        for x in (0.3, 1.0, 2.7):
            y = np.log(x)
            self.assertAlmostEqual(d.expected_log_density(x), y * (e1 + y * e2) - ea + eb - y, places=10)
        self.assertEqual(d.expected_log_density(-1.0), -np.inf)
        lq = np.log(np.array([0.3, 1.0, 2.7]))
        self.assertTrue(
            np.allclose(d.seq_expected_log_density(lq), [d.expected_log_density(np.exp(y)) for y in lq], atol=1e-12)
        )
        d0 = LogGaussianDistribution(0.3, 2.0)
        self.assertAlmostEqual(d0.expected_log_density(1.1), d0.log_density(1.1), places=12)

    def test_log_gaussian_seq_eld_matches_scalar(self):
        """seq_expected_log_density on log-scale stats matches the per-element scalar value."""
        args = (0.0, 1e-2, 1.2, 0.8)
        sm = LogGaussianEstimator(prior=NormalGammaDistribution(*args)).estimate(
            None, (self.lsum, self.lsum2, self.n, self.n)
        )
        lq = np.log(self.xp[:10])
        scalar = np.asarray([sm.expected_log_density(np.exp(y)) for y in lq])
        self.assertTrue(np.allclose(sm.seq_expected_log_density(lq), scalar, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
