"""pysp.ppl: the expanded leaf-family surface + per-slot constraint transforms.

Covers the newly registered families (Weibull/Laplace/Logistic/Uniform/Rayleigh/Pareto/Binomial),
partial-`free` models (some slots fixed, some estimated), and unit-interval (logit) reparameterized
Bayesian inference for probability parameters.
"""

import unittest

import numpy as np

from pysp.ppl import (
    Bernoulli,
    Beta,
    Binomial,
    Gamma,
    Geometric,
    Laplace,
    Logistic,
    Mix,
    Normal,
    Pareto,
    Poisson,
    Rayleigh,
    Uniform,
    Weibull,
    free,
)


class LeafFamilyMLETestCase(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_weibull(self):
        m = Weibull(free, free).fit(list(self.rng.weibull(1.5, 4000) * 2.0))
        self.assertAlmostEqual(m.params["shape"], 1.5, delta=0.2)
        self.assertAlmostEqual(m.params["scale"], 2.0, delta=0.2)

    def test_laplace(self):
        m = Laplace(free, free).fit(list(self.rng.laplace(1.0, 2.0, 4000)))
        self.assertAlmostEqual(m.params["loc"], 1.0, delta=0.2)
        self.assertAlmostEqual(m.params["scale"], 2.0, delta=0.2)

    def test_logistic(self):
        m = Logistic(free, free).fit(list(self.rng.logistic(0.5, 1.5, 4000)))
        self.assertAlmostEqual(m.params["loc"], 0.5, delta=0.2)
        self.assertAlmostEqual(m.params["scale"], 1.5, delta=0.2)

    def test_uniform(self):
        m = Uniform(free, free).fit(list(self.rng.uniform(-2.0, 5.0, 4000)))
        self.assertAlmostEqual(m.params["low"], -2.0, delta=0.1)
        self.assertAlmostEqual(m.params["high"], 5.0, delta=0.1)

    def test_rayleigh(self):
        m = Rayleigh(free).fit(list(self.rng.rayleigh(2.0, 4000)))
        self.assertAlmostEqual(m.params["sigma"], 2.0, delta=0.2)

    def test_pareto(self):
        m = Pareto(free, free).fit(list(self.rng.pareto(3.0, 4000) + 1.0))
        self.assertAlmostEqual(m.params["alpha"], 3.0, delta=0.4)


class PartialFreeTestCase(unittest.TestCase):
    def test_binomial_fixed_n(self):
        # n is structural (fixed); only p is estimated -> partial-free MLE
        rng = np.random.RandomState(1)
        m = Binomial(10, free).fit(list(rng.binomial(10, 0.3, 6000).astype(float)))
        self.assertEqual(m.params["n"], 10)
        self.assertAlmostEqual(m.params["p"], 0.3, delta=0.02)

    def test_normal_fixed_mean(self):
        rng = np.random.RandomState(2)
        m = Normal(0.0, free).fit(list(rng.normal(0.0, 2.5, 6000)))
        self.assertAlmostEqual(m.params["mean"], 0.0, delta=1e-9)  # held fixed
        self.assertAlmostEqual(m.params["sd"], 2.5, delta=0.1)


class UnitIntervalBayesTestCase(unittest.TestCase):
    """Probability params use a logit reparameterization for gradient/MCMC inference."""

    def test_bernoulli_p_mcmc(self):
        rng = np.random.RandomState(1)
        data = list((rng.uniform(size=5000) < 0.7).astype(float))
        m = Bernoulli(Beta(2, 2, name="p")).fit(data, how="mcmc", draws=1500, burn=500, rng=np.random.RandomState(1))
        self.assertAlmostEqual(float(m.result.mean("p")), 0.7, delta=0.03)
        self.assertTrue(0.0 < m.dist.p < 1.0)

    def test_binomial_p_mcmc(self):
        rng = np.random.RandomState(2)
        data = list(rng.binomial(10, 0.4, 4000).astype(float))
        m = Binomial(10, Beta(2, 2, name="p")).fit(data, how="mcmc", draws=1200, burn=400, rng=np.random.RandomState(2))
        self.assertAlmostEqual(float(m.result.mean("p")), 0.4, delta=0.03)

    def test_weibull_shape_prior_map(self):
        rng = np.random.RandomState(3)
        m = Weibull(Gamma(2, 1, name="shape"), free).fit(list(rng.weibull(1.5, 4000) * 2.0), how="map")
        self.assertAlmostEqual(m.params["shape"], 1.5, delta=0.3)
        self.assertAlmostEqual(m.params["scale"], 2.0, delta=0.3)

    def test_negative_binomial_p_slot_is_logit_reparameterized(self):
        """NB success-probability p must live on the unit interval, not the real line.

        Regression for the slot-support bug: with p registered on 'real' the
        unconstrained optimizer drives p out of (0, 1) and the underlying
        NegativeBinomialDistribution raises. The logit ('unit') reparameterization
        keeps p in (0, 1) and recovers the truth.
        """
        from pysp.ppl import NegativeBinomial
        from pysp.ppl.core import _FAMILIES

        # The p slot (index 1) must be the logit-reparameterized 'unit' support.
        self.assertEqual(_FAMILIES["NegativeBinomial"].support, ("positive", "unit"))

        rng = np.random.RandomState(0)
        data = list(rng.negative_binomial(5, 0.4, 3000).astype(float))
        m = NegativeBinomial(5.0, Beta(2, 2, name="p")).fit(data, how="map")
        self.assertTrue(0.0 < m.dist.p < 1.0)
        self.assertAlmostEqual(m.dist.p, 0.4, delta=0.05)


class ConjugatePairsTestCase(unittest.TestCase):
    """New closed-form conjugate posteriors (exact, instant) must match MCMC."""

    def test_binomial_beta(self):
        rng = np.random.RandomState(0)
        data = list(rng.binomial(10, 0.35, 4000).astype(float))
        from pysp.ppl.inference import ConjugatePosterior

        m = Binomial(10, Beta(2, 2, name="p")).fit(data)  # auto -> conjugate
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("p")), 0.35, delta=0.02)

    def test_geometric_beta(self):
        rng = np.random.RandomState(1)
        data = list(rng.geometric(0.3, 4000).astype(float))  # k >= 1
        from pysp.ppl.inference import ConjugatePosterior

        m = Geometric(Beta(2, 2, name="p")).fit(data)
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("p")), 0.3, delta=0.02)


class ConjugateMixturePriorTestCase(unittest.TestCase):
    """A Mix(...) of conjugate priors has an exact reweighted-mixture-of-conjugates posterior."""

    def test_normal_bimodal_prior_selects_component(self):
        from pysp.ppl.inference import ConjugateMixturePosterior

        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 2000))
        m = Normal(Mix([Normal(-5, 2), Normal(5, 2)]), 1.0).fit(data)  # auto -> conjugate mixture
        self.assertIsInstance(m.result, ConjugateMixturePosterior)
        self.assertAlmostEqual(m.result.mean(), 5.0, delta=0.1)
        # the +5 component must dominate the posterior mixture weights
        self.assertGreater(m.result.weights[1], 0.99)

    def test_matches_mcmc(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 2000))
        exact = Normal(Mix([Normal(-5, 2), Normal(5, 2)]), 1.0).fit(data)
        mcmc = Normal(Mix([Normal(-5, 2), Normal(5, 2)]), 1.0).fit(
            data, how="mcmc", draws=3000, burn=1000, rng=np.random.RandomState(1)
        )
        self.assertAlmostEqual(exact.result.mean(), float(np.mean(mcmc.result.samples())), delta=0.05)

    def test_poisson_mixture_of_gammas(self):
        rng = np.random.RandomState(2)
        data = list(rng.poisson(3.5, 2000).astype(float))
        m = Poisson(Mix([Gamma(1, 1), Gamma(20, 2)])).fit(data)
        self.assertAlmostEqual(m.result.mean(), 3.5, delta=0.2)


class MultiChainDiagnosticsTestCase(unittest.TestCase):
    """Multiple chains -> Gelman-Rubin R-hat + combined ESS; process-parallel matches sequential."""

    def test_rhat_and_combined_ess(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 1500))
        m = Normal(free, free).fit(data, how="mcmc", draws=1500, burn=500, chains=4, rng=np.random.RandomState(1))
        self.assertEqual(m.result.n_chains, 4)
        for r in m.result.rhat.values():  # converged: R-hat ~ 1
            self.assertLess(r, 1.1)
        self.assertGreater(m.result.ess, 0.0)
        self.assertAlmostEqual(m.result.mean("arg0"), 5.0, delta=0.2)

    def test_ensemble_sampler(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 4000))
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, free).fit(data, how="ensemble", draws=800, burn=300, rng=np.random.RandomState(1))
        self.assertAlmostEqual(float(m.result.mean("mu")), 5.0, delta=0.1)
        # the stretch move mixes well across the walker ensemble
        ess = float(np.atleast_1d(m.result.raw.effective_sample_size()).min())
        self.assertGreater(ess, 250)

    def test_ensemble_multichain_rhat(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 2000))
        m = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="ensemble", chains=4, draws=400, burn=150, rng=np.random.RandomState(1)
        )
        self.assertEqual(m.result.n_chains, 4)
        for r in m.result.rhat.values():
            self.assertLess(r, 1.1)
        self.assertAlmostEqual(float(m.result.mean("mu")), 5.0, delta=0.2)

    def test_ensemble_parallel_chains(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(-1.0, 1.5, 2000))
        seq = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="ensemble", chains=3, parallel=False, draws=300, burn=100, rng=np.random.RandomState(7)
        )
        par = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="ensemble", chains=3, parallel=True, draws=300, burn=100, rng=np.random.RandomState(7)
        )
        self.assertAlmostEqual(seq.result.mean("mu"), par.result.mean("mu"), places=5)

    def test_process_parallel_matches_sequential(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(-1.0, 1.5, 1200))
        seq = Normal(free, free).fit(
            data, how="hmc", draws=800, burn=300, chains=3, parallel=False, rng=np.random.RandomState(7)
        )
        par = Normal(free, free).fit(
            data, how="hmc", draws=800, burn=300, chains=3, parallel=True, rng=np.random.RandomState(7)
        )
        # identical seeds + deterministic chains -> identical posterior means
        self.assertAlmostEqual(seq.result.mean("arg0"), par.result.mean("arg0"), places=6)
        self.assertAlmostEqual(seq.result.mean("arg1"), par.result.mean("arg1"), places=6)


if __name__ == "__main__":
    unittest.main()
