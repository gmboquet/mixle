"""Bayesian (conjugate / variational) behavior folded onto pysp.stats Gaussian.

This is the proven template for the bstats -> stats merge: a frequentist leaf gains conjugate
posterior estimation, ``expected_log_density``, and a posterior-returning ``fit`` while its MLE
path stays byte-identical. Numeric expectations mirror the historical bstats assertions.
"""

import unittest

import numpy as np

from pysp.stats import seq_encode, seq_estimate, seq_initialize
from pysp.stats.bayes.normal_gamma import NormalGammaDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution, GaussianEstimator
from pysp.utils.estimation import _data_objective_sum, _model_objective, fit, optimize
from pysp.utils.special import digamma


class StatsBayesGaussianTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(3)
        self.data = list(rng.normal(2.5, 1.5, size=500))
        self.sx = float(np.sum(self.data))
        self.sxx = float(np.sum(np.square(self.data)))
        self.n = float(len(self.data))

    def test_mle_path_unchanged(self):
        """No prior -> plain MLE point estimate; estimator carries no posterior."""
        m = optimize(self.data, GaussianEstimator(), max_its=1, out=None)
        self.assertAlmostEqual(m.mu, self.sx / self.n, places=10)
        self.assertAlmostEqual(m.sigma2, self.sxx / self.n - (self.sx / self.n) ** 2, places=10)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_conjugate_posterior_closed_form(self):
        """estimate() with a NormalGamma prior matches the textbook posterior update."""
        mu0, lam, a, b = 0.5, 3.0, 1.2, 0.8
        prior = NormalGammaDistribution(mu0, lam, a, b)
        m = GaussianEstimator(prior=prior).estimate(None, (self.sx, self.sxx, self.n, self.n))
        post = m.get_prior()
        pmu, plam, pa, pb = post.get_parameters()
        # closed forms
        self.assertAlmostEqual(plam, lam + self.n, places=8)
        self.assertAlmostEqual(pa, a + self.n / 2.0, places=8)
        self.assertAlmostEqual(pmu, (self.sx + mu0 * lam) / (lam + self.n), places=8)
        mean1 = self.sx / self.n
        mean2 = self.sx / self.n
        b0 = self.sxx - mean2 * self.sx
        b1 = (lam * self.n / plam) * (mean1 - mu0) ** 2
        self.assertAlmostEqual(pb, b + 0.5 * (b0 + b1), places=8)
        # joint MAP point estimate
        self.assertAlmostEqual(m.mu, pmu, places=10)
        self.assertAlmostEqual(m.sigma2, pb / (pa - 0.5), places=10)

    def test_expected_log_density_formula(self):
        """expected_log_density equals the VB E[log p] closed form and falls back without a prior."""
        prior = NormalGammaDistribution(0.3, 2.0, 4.0, 5.0)
        d = GaussianDistribution(0.3, 5.0 / (4.0 - 0.5), prior=prior)
        mu, lam, a, b = prior.get_parameters()
        ea = (mu * mu) * (a / b) * 0.5 + 0.5 / lam + 0.5 * (np.log(b) - digamma(a))
        e1 = mu * a / b
        e2 = -0.5 * a / b
        eb = -0.5 * np.log(2 * np.pi)
        for x in (-1.0, 0.0, 0.3, 2.7):
            self.assertAlmostEqual(d.expected_log_density(x), x * (e1 + x * e2) - ea + eb, places=10)
        xs = np.array([-1.0, 0.0, 0.3, 2.7])
        self.assertTrue(
            np.allclose(d.seq_expected_log_density(xs), [d.expected_log_density(x) for x in xs], atol=1e-12)
        )
        # no prior -> plug-in
        d0 = GaussianDistribution(0.3, 2.0)
        self.assertAlmostEqual(d0.expected_log_density(1.1), d0.log_density(1.1), places=12)

    def test_fit_recovers_parameters_and_returns_posterior(self):
        """fit() returns a posterior-bearing MAP model that recovers the generating parameters."""
        prior = NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)
        m = fit(self.data, GaussianEstimator(prior=prior), max_its=20, out=None)
        self.assertIsInstance(m.get_prior(), NormalGammaDistribution)
        self.assertAlmostEqual(m.mu, 2.5, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.sigma2), 1.5, delta=0.2)

    def test_fit_objective_monotone(self):
        """The penalized objective never decreases across fit iterations."""
        prior = NormalGammaDistribution(0.0, 1.0, 1.5, 1.5)
        est = GaussianEstimator(prior=prior)
        enc = seq_encode(self.data, est.accumulator_factory().make().acc_to_encoder())
        mm = seq_initialize(enc_data=enc, estimator=est, rng=np.random.RandomState(0), p=0.5)
        objs = [_data_objective_sum(enc, mm) + _model_objective(est, mm)]
        for _ in range(5):
            mm = seq_estimate(enc, est, mm)
            objs.append(_data_objective_sum(enc, mm) + _model_objective(est, mm))
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6), "objective decreased: %s" % objs)


if __name__ == "__main__":
    unittest.main()
