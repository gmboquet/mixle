"""Tests for pysp.ppl Bayesian inference: MAP and parameter MCMC (build slice 7)."""

import unittest

import numpy as np

from pysp.ppl import Bernoulli, Beta, Exponential, Gamma, Normal, Poisson, free
from pysp.ppl.inference import ConjugatePosterior


class PPLInferenceTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(5.0, 2.0, size=3000))

    def test_map_recovers_params(self):
        m = Normal(Normal(0, 10), free).fit(self.data, how="map")
        self.assertTrue(m.is_bound)
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.2)

    def test_auto_picks_map_when_priors(self):
        # priors present -> auto routes to map (point), no crash, recovers params
        m = Normal(Normal(0, 10), free).fit(self.data)  # how="auto"
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)

    def test_mcmc_posterior(self):
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, free).fit(self.data, how="mcmc", draws=1500, burn=800, rng=np.random.RandomState(1))
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.2)
        # decent mixing (adaptive RW targets ~0.44)
        self.assertGreater(m.result.acceptance_rate, 0.2)
        self.assertLess(m.result.acceptance_rate, 0.7)
        # posterior draws addressable by handle, name, and index
        by_handle = m.posterior(mu)
        by_name = m.posterior("mu")
        by_index = m.posterior(0)
        self.assertEqual(len(by_handle), 1500)
        np.testing.assert_array_equal(by_handle, by_name)
        np.testing.assert_array_equal(by_handle, by_index)
        # posterior mean is close to the point estimate
        self.assertAlmostEqual(by_handle.mean(), m.dist.mu, delta=0.1)
        # summary present
        s = m.result.summary()
        self.assertIn("mu", s)
        self.assertIn("q2.5", s["mu"])


class PPLHMCTestCase(unittest.TestCase):
    def test_hmc_recovers_and_mixes(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, size=3000))
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, free).fit(data, how="hmc", draws=1000, burn=500, rng=np.random.RandomState(1))
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.2)
        # HMC mixes far better than RW: high ESS relative to draws
        ess = np.atleast_1d(m.result.raw.effective_sample_size())
        self.assertGreater(ess.min(), 400)  # out of 1000 draws
        self.assertEqual(len(m.posterior("mu")), 1000)


class PPLHierarchicalTestCase(unittest.TestCase):
    def test_normal_normal_random_effects(self):
        rng = np.random.RandomState(0)
        m_true, tau_true, sigma_true = 10.0, 3.0, 1.0
        G = 200
        true_mu = rng.normal(m_true, tau_true, G)
        data = [list(rng.normal(true_mu[i], sigma_true, rng.randint(5, 20))) for i in range(G)]
        fit = Normal(Normal(0, 100).each(), free).fit(data)
        h = fit.result.hyper
        self.assertAlmostEqual(h["m"], m_true, delta=0.5)
        self.assertAlmostEqual(h["tau"], tau_true, delta=0.5)
        self.assertAlmostEqual(h["sigma"], sigma_true, delta=0.2)
        # per-group posterior means track the true group means (shrinkage estimator)
        corr = np.corrcoef(fit.result.group_means, true_mu)[0, 1]
        self.assertGreater(corr, 0.95)
        self.assertEqual(fit.result.group_means.size, G)

    def test_gamma_poisson_random_effects(self):
        rng = np.random.RandomState(0)
        G = 300
        lam = rng.gamma(4.0, 1 / 2.0, G)  # rates ~ Gamma(shape=4, rate=2), mean 2
        data = [list(rng.poisson(lam[i], rng.randint(5, 20)).astype(float)) for i in range(G)]
        fit = Poisson(Gamma(1, 1).each()).fit(data)
        self.assertAlmostEqual(fit.result.hyper["mean"], 2.0, delta=0.3)
        self.assertGreater(np.corrcoef(fit.result.group_means, lam)[0, 1], 0.85)

    def test_beta_bernoulli_random_effects(self):
        rng = np.random.RandomState(0)
        G = 300
        p = rng.beta(2.0, 5.0, G)  # p_i ~ Beta(2,5), mean 0.286
        data = [list((rng.random(rng.randint(10, 40)) < p[i]).astype(float)) for i in range(G)]
        fit = Bernoulli(Beta(1, 1).each()).fit(data)
        self.assertAlmostEqual(fit.result.hyper["mean"], 0.286, delta=0.06)
        self.assertGreater(np.corrcoef(fit.result.group_means, p)[0, 1], 0.8)


class PPLVITestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(5.0, 2.0, size=3000))

    def test_vi_recovers_params(self):
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, free).fit(self.data, how="vi", rng=np.random.RandomState(1))
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.2)
        # variational posterior draws + ELBO available
        self.assertEqual(len(m.posterior("mu")), 4000)
        self.assertTrue(np.isfinite(m.result.raw.elbo))

    def test_vi_handles_non_conjugate(self):
        # sd has a Gamma prior -> not a registered conjugate pair; VI must handle it
        m = Normal(Normal(0, 10, name="mu"), Gamma(2, 1, name="sd")).fit(
            self.data, how="vi", rng=np.random.RandomState(2)
        )
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.3)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.3)

    def test_vi_batched_target_across_supports(self):
        # the batched ADVI ELBO must broadcast priors over positive (Gamma) and unit (Beta)
        # supports, not just the real line.
        rng = np.random.RandomState(3)
        pois = list(rng.poisson(3.5, 4000).astype(float))
        mp = Poisson(Gamma(2, 1, name="rate")).fit(pois, how="vi", rng=np.random.RandomState(4))
        self.assertAlmostEqual(mp.params["rate"], 3.5, delta=0.3)

        bern = list((rng.uniform(size=4000) < 0.3).astype(float))
        mb = Bernoulli(Beta(2, 2, name="p")).fit(bern, how="vi", rng=np.random.RandomState(5))
        self.assertAlmostEqual(mb.params["p"], 0.3, delta=0.05)


class PPLConjugateTestCase(unittest.TestCase):
    def test_normal_normal_conjugate_is_exact_and_auto(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, size=4000))
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, 2.0).fit(data)  # known sd -> auto picks conjugate
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.15)
        s = m.result.summary()["mu"]
        self.assertEqual(s["posterior"], "Normal")
        # posterior draws addressable
        self.assertEqual(len(m.posterior("mu")), 4000)

    def test_poisson_gamma_conjugate(self):
        rng = np.random.RandomState(1)
        data = list(rng.poisson(3.5, size=4000).astype(float))
        m = Poisson(Gamma(2.0, 1.0, name="rate")).fit(data)
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(m.dist.lam, 3.5, delta=0.15)
        self.assertEqual(m.result.summary()["rate"]["posterior"], "Gamma")

    def test_bernoulli_beta_conjugate(self):
        rng = np.random.RandomState(2)
        data = list((rng.random(4000) < 0.3).astype(float))
        m = Bernoulli(Beta(1, 1, name="p")).fit(data)
        self.assertAlmostEqual(m.dist.p, 0.3, delta=0.03)

    def test_exponential_gamma_conjugate(self):
        rng = np.random.RandomState(3)
        data = list(rng.exponential(1.0 / 0.7, size=4000))  # rate 0.7
        m = Exponential(Gamma(2.0, 1.0, name="rate")).fit(data)
        # Exponential lowers rate->beta=1/rate; recovered rate = 1/beta
        self.assertAlmostEqual(1.0 / m.dist.beta, 0.7, delta=0.05)


if __name__ == "__main__":
    unittest.main()
