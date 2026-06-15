"""Bayesian (conjugate / variational) behavior folded onto the Gamma-prior leaf group.

Covers the conjugate-prior merge for the three leaf families whose conjugate parameter prior is
a GammaDistribution: Poisson and Exponential (Gamma prior on the rate). Each family gains
conjugate posterior estimation, ``expected_log_density``, and a posterior-returning ``fit`` while
its MLE path stays byte-identical. Conjugate behavior is pinned against the textbook Gamma
posterior closed form and the variational expected-log-density formula. Gamma itself is exercised
for MLE self-consistency (Gamma has no conjugate estimate) and as the shared conjugate prior
family.
"""

import unittest

import numpy as np
from scipy.special import gammaln

from pysp.stats import seq_encode, seq_estimate, seq_initialize
from pysp.stats.exponential import ExponentialDistribution, ExponentialEstimator
from pysp.stats.gamma import GammaDistribution, GammaEstimator
from pysp.stats.poisson import PoissonDistribution, PoissonEstimator
from pysp.utils.estimation import _data_objective_sum, _model_objective, fit, optimize
from pysp.utils.special import digamma


class StatsBayesPoissonTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.data = [int(v) for v in rng.poisson(3.0, size=400)]
        self.n = float(len(self.data))
        self.psum = float(np.sum(self.data))

    def test_mle_path_unchanged(self):
        """No prior -> plain MLE point estimate; estimator carries no posterior."""
        m = optimize(self.data, PoissonEstimator(), max_its=1, out=None)
        self.assertAlmostEqual(m.lam, self.psum / self.n, places=10)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_mle_value_matches_legacy(self):
        """The MLE estimate is byte-identical to a direct PoissonEstimator.estimate."""
        m = PoissonEstimator().estimate(None, (self.n, self.psum))
        self.assertEqual(m.lam, max(self.psum / self.n, 1.0e-12))
        self.assertIsNone(m.get_prior())

    def test_conjugate_seq_matches_scalar(self):
        """seq_expected_log_density matches the per-element scalar expected_log_density."""
        k, theta = 2.3, 1.7
        ss = (self.n, self.psum)
        sd = PoissonEstimator(prior=GammaDistribution(k, theta)).estimate(None, ss)

        xs = [0, 1, 2, 3, 7, 15]
        xsa = np.asarray(xs, dtype=float)
        enc = (xsa, gammaln(xsa + 1.0))
        scalar = np.asarray([sd.expected_log_density(x) for x in xs])
        self.assertTrue(np.allclose(sd.seq_expected_log_density(enc), scalar, atol=1e-12))

    def test_conjugate_posterior_closed_form(self):
        """estimate() with a Gamma prior matches the textbook Gamma posterior update."""
        k, theta = 2.3, 1.7
        m = PoissonEstimator(prior=GammaDistribution(k, theta)).estimate(None, (self.n, self.psum))
        pk, ptheta = m.get_prior().get_parameters()
        self.assertAlmostEqual(pk, k + self.psum, places=8)
        self.assertAlmostEqual(ptheta, theta / (self.n * theta + 1.0), places=8)
        self.assertAlmostEqual(m.lam, (pk - 1.0) * ptheta, places=10)

    def test_expected_log_density_formula(self):
        """expected_log_density equals the VB closed form and falls back without a prior."""
        k, theta = 2.3, 1.7
        d = PoissonDistribution(2.0, prior=GammaDistribution(k, theta))
        for x in (0, 1, 4, 9):
            expect = (digamma(k) + np.log(theta)) * x - k * theta - gammaln(x + 1.0)
            self.assertAlmostEqual(d.expected_log_density(x), expect, places=10)
        d0 = PoissonDistribution(2.0)
        self.assertAlmostEqual(d0.expected_log_density(3), d0.log_density(3), places=12)

    def test_fit_recovers_parameters_and_returns_posterior(self):
        """fit() returns a posterior-bearing model that recovers the generating rate."""
        prior = GammaDistribution(1.0001, 1.0e6)
        m = fit(self.data, PoissonEstimator(prior=prior), max_its=10, out=None)
        self.assertIsInstance(m.get_prior(), GammaDistribution)
        self.assertAlmostEqual(m.lam, 3.0, delta=0.4)

    def test_fit_objective_monotone(self):
        """The penalized objective never decreases across fit iterations."""
        est = PoissonEstimator(prior=GammaDistribution(2.0, 2.0))
        enc = seq_encode(self.data, est.accumulator_factory().make().acc_to_encoder())
        mm = seq_initialize(enc_data=enc, estimator=est, rng=np.random.RandomState(0), p=0.5)
        objs = [_data_objective_sum(enc, mm) + _model_objective(est, mm)]
        for _ in range(5):
            mm = seq_estimate(enc, est, mm)
            objs.append(_data_objective_sum(enc, mm) + _model_objective(est, mm))
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6), "objective decreased: %s" % objs)


class StatsBayesExponentialTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(11)
        # stats Exponential is scale-parameterized (mean = beta); generate mean 2.0
        self.data = list(rng.exponential(2.0, size=400))
        self.n = float(len(self.data))
        self.psum = float(np.sum(self.data))

    def test_mle_path_unchanged(self):
        """No prior -> plain MLE point estimate; estimator carries no posterior."""
        m = optimize(self.data, ExponentialEstimator(), max_its=1, out=None)
        self.assertAlmostEqual(m.beta, self.psum / self.n, places=10)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_mle_value_matches_legacy(self):
        """The MLE estimate is byte-identical to a direct ExponentialEstimator.estimate."""
        m = ExponentialEstimator().estimate(None, (self.n, self.psum))
        self.assertEqual(m.beta, self.psum / self.n)
        self.assertIsNone(m.get_prior())

    def test_conjugate_seq_matches_scalar(self):
        """seq_expected_log_density matches the per-element scalar expected_log_density."""
        k, theta = 2.0, 3.0
        ss = (self.n, self.psum)
        sd = ExponentialEstimator(prior=GammaDistribution(k, theta)).estimate(None, ss)

        xs = [0.0, 0.5, 1.5, 3.0]
        xsa = np.asarray(xs)
        scalar = np.asarray([sd.expected_log_density(x) for x in xs])
        self.assertTrue(np.allclose(sd.seq_expected_log_density(xsa), scalar, atol=1e-12))

    def test_conjugate_posterior_closed_form(self):
        """estimate() with a Gamma prior matches the Gamma posterior update on the rate."""
        k, theta = 2.0, 3.0
        a, b = k, 1.0 / theta
        m = ExponentialEstimator(prior=GammaDistribution(k, theta)).estimate(None, (self.n, self.psum))
        pk, ptheta = m.get_prior().get_parameters()
        n_post = self.n + a
        s_post = self.psum + b
        self.assertAlmostEqual(pk, n_post, places=8)
        self.assertAlmostEqual(ptheta, 1.0 / s_post, places=8)
        self.assertAlmostEqual(1.0 / m.beta, (n_post - 1.0) / s_post, places=10)

    def test_expected_log_density_formula(self):
        """expected_log_density equals the VB closed form and falls back without a prior."""
        k, theta = 2.0, 3.0
        a, b = k, 1.0 / theta
        e1 = -a / b
        ea = -(digamma(a) - np.log(b))
        d = ExponentialDistribution(1.0, prior=GammaDistribution(k, theta))
        for x in (0.0, 0.5, 1.5, 3.0):
            self.assertAlmostEqual(d.expected_log_density(x), e1 * x - ea, places=10)
        self.assertEqual(d.expected_log_density(-1.0), -np.inf)
        d0 = ExponentialDistribution(2.0)
        self.assertAlmostEqual(d0.expected_log_density(1.3), d0.log_density(1.3), places=12)

    def test_fit_recovers_parameters_and_returns_posterior(self):
        """fit() returns a posterior-bearing model that recovers the generating scale."""
        prior = GammaDistribution(1.0001, 1.0e6)
        m = fit(self.data, ExponentialEstimator(prior=prior), max_its=10, out=None)
        self.assertIsInstance(m.get_prior(), GammaDistribution)
        self.assertAlmostEqual(m.beta, 2.0, delta=0.3)

    def test_fit_objective_monotone(self):
        """The penalized objective never decreases across fit iterations."""
        est = ExponentialEstimator(prior=GammaDistribution(2.0, 2.0))
        enc = seq_encode(self.data, est.accumulator_factory().make().acc_to_encoder())
        mm = seq_initialize(enc_data=enc, estimator=est, rng=np.random.RandomState(0), p=0.5)
        objs = [_data_objective_sum(enc, mm) + _model_objective(est, mm)]
        for _ in range(5):
            mm = seq_estimate(enc, est, mm)
            objs.append(_data_objective_sum(enc, mm) + _model_objective(est, mm))
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6), "objective decreased: %s" % objs)


class StatsBayesGammaTestCase(unittest.TestCase):
    """Gamma has no conjugate estimate; verify the MLE stationarity and prior-family role."""

    def setUp(self):
        rng = np.random.RandomState(13)
        self.data = list(rng.gamma(2.5, 1.3, size=400))
        self.n = float(len(self.data))
        self.s = float(np.sum(self.data))
        self.sl = float(np.sum(np.log(self.data)))

    def test_get_parameters(self):
        """GammaDistribution exposes (k, theta) so it can serve as a conjugate prior."""
        d = GammaDistribution(2.0, 3.0)
        self.assertEqual(d.get_parameters(), (2.0, 3.0))

    def test_mle_satisfies_stationarity(self):
        """The MLE (k, theta) satisfies the Gamma likelihood stationarity equations.

        For a Gamma(k, theta) the MLE solves theta = mean/k and
        log(k) - digamma(k) = log(mean) - mean(log x); both hold to numerical tolerance.
        """
        sd = GammaEstimator().estimate(None, (self.n, self.s, self.sl))
        mean = self.s / self.n
        mean_log = self.sl / self.n
        self.assertAlmostEqual(sd.theta, mean / sd.k, places=10)
        self.assertAlmostEqual(np.log(sd.k) - digamma(sd.k), np.log(mean) - mean_log, places=8)

    def test_mle_path_unchanged(self):
        """optimize() recovers the generating shape/scale via plain MLE."""
        m = optimize(self.data, GammaEstimator(), max_its=1, out=None)
        self.assertAlmostEqual(m.k, 2.5, delta=0.4)
        self.assertAlmostEqual(m.theta, 1.3, delta=0.3)


if __name__ == "__main__":
    unittest.main()
