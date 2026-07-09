"""mixle.ppl: the expanded leaf-family surface + per-slot constraint transforms.

Covers the newly registered families (Weibull/Laplace/Logistic/Uniform/Rayleigh/Pareto/Binomial),
partial-`free` models (some slots fixed, some estimated), and unit-interval (logit) reparameterized
Bayesian inference for probability parameters.
"""

import unittest

import numpy as np

from mixle.ppl import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
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
        from mixle.ppl import NegativeBinomial
        from mixle.ppl.core import _FAMILIES

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
        from mixle.ppl.inference import ConjugatePosterior

        m = Binomial(10, Beta(2, 2, name="p")).fit(data)  # auto -> conjugate
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("p")), 0.35, delta=0.02)

    def test_geometric_beta(self):
        rng = np.random.RandomState(1)
        data = list(rng.geometric(0.3, 4000).astype(float))  # k >= 1
        from mixle.ppl.inference import ConjugatePosterior

        m = Geometric(Beta(2, 2, name="p")).fit(data)
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("p")), 0.3, delta=0.02)

    def test_normal_mean_and_variance_nig(self):
        # Normal(free, free): mean AND variance unknown -> Normal-Inverse-Gamma closed form (the most
        # common Bayesian model). Verified bit-for-bit against the stats-layer conjugate path.
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 2000))
        from mixle.ppl.inference import ConjugatePosterior

        m = Normal(free, free).fit(data, how="conjugate")
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("mu")), 5.0, delta=0.15)
        self.assertAlmostEqual(float(m.result.mean("sigma")), 2.0, delta=0.15)
        # oracle: the stats GaussianEstimator(prior=NormalGammaPrior()) closed form
        from mixle.inference import optimize
        from mixle.inference.priors import NormalGammaPrior
        from mixle.stats import GaussianEstimator

        ref = optimize(data, GaussianEstimator(prior=NormalGammaPrior()), max_its=1, out=None)
        self.assertAlmostEqual(float(m.result.mean("mu")), ref.mu, places=9)
        self.assertAlmostEqual(float(m.result.mean("sigma")), float(np.sqrt(ref.sigma2)), delta=0.02)
        # a real posterior: credible interval over mu + posterior predictive
        mus = m.result.samples("mu", n=3000, rng=np.random.RandomState(1))
        self.assertLess(np.percentile(mus, 2.5), 5.0)
        self.assertGreater(np.percentile(mus, 97.5), 5.0)

    def test_gamma_rate_gamma(self):
        rng = np.random.RandomState(5)
        data = list(rng.gamma(3.0, 1.0 / 2.0, 4000))  # Gamma(shape=3, rate=2)
        from mixle.ppl.inference import ConjugatePosterior

        m = Gamma(3.0, Gamma(2.0, 1.0, name="rate")).fit(data)  # known shape, Gamma prior on rate -> conjugate
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("rate")), 2.0, delta=0.1)

    def test_negbinomial_beta(self):
        from mixle.ppl import NegativeBinomial

        rng = np.random.RandomState(6)
        data = list(rng.negative_binomial(5, 0.4, 4000).astype(float))  # r=5 successes, success prob 0.4
        from mixle.ppl.inference import ConjugatePosterior

        m = NegativeBinomial(5, Beta(2.0, 2.0, name="p")).fit(data)  # known r, Beta prior on p -> conjugate
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("p")), 0.4, delta=0.03)

    def test_general_bridge_new_families(self):
        # families with NO hand-coded posterior in the PPL: the general exp-family bridge delegates to the
        # exp-family-map-derived stats conjugate machinery, so how='auto' picks 'conjugate' and the
        # closed-form posterior recovers truth -- no posterior math written in the PPL for these.
        from mixle.ppl import InverseGamma, InverseGaussian
        from mixle.ppl.inference import ConjugatePosterior, stats_conjugate_supported

        rng = np.random.RandomState(7)

        def weak(nm):
            return Gamma(1e-3, 1e-3, name=nm)  # weak Gamma prior on the target parameter

        # InverseGamma(shape known, scale ~ prior): posterior mean of scale recovers truth
        ig_rv = InverseGamma(4.0, weak("beta"))
        self.assertTrue(stats_conjugate_supported(ig_rv))
        self.assertEqual(ig_rv.explain_fit()["route"], "conjugate")
        m = ig_rv.fit(list(1.0 / rng.gamma(4.0, 1 / 3.0, 4000)))
        self.assertIsInstance(m.result, ConjugatePosterior)
        self.assertAlmostEqual(float(m.result.mean("beta")), 3.0, delta=0.4)

        # Pareto(scale known, tail index alpha ~ prior)
        p_rv = Pareto(1.0, weak("alpha"))
        self.assertEqual(p_rv.explain_fit()["route"], "conjugate")
        m = p_rv.fit(list(1.0 + rng.pareto(3.0, 4000)))
        self.assertAlmostEqual(float(m.result.mean("alpha")), 3.0, delta=0.4)

        # InverseGaussian(mean known, shape lambda ~ prior)
        igauss_rv = InverseGaussian(2.0, weak("lam"))
        self.assertEqual(igauss_rv.explain_fit()["route"], "conjugate")
        m = igauss_rv.fit(list(rng.wald(2.0, 3.0, 4000)))
        self.assertAlmostEqual(float(m.result.mean("lam")), 3.0, delta=0.5)

    def test_categorical_dirichlet(self):
        rng = np.random.RandomState(2)
        true = np.array([0.2, 0.3, 0.5])
        data = list(rng.choice(3, size=4000, p=true))
        from mixle.ppl.inference import ConjugatePosterior

        rv = Categorical(Dirichlet([1.0, 1.0, 1.0], name="p"))
        self.assertEqual(rv.explain_fit()["route"], "conjugate")
        m = rv.fit(data)  # auto -> conjugate (closed-form Dirichlet posterior over the K-vector)
        self.assertIsInstance(m.result, ConjugatePosterior)
        post_mean = np.asarray(m.result.mean("p"))
        self.assertEqual(post_mean.shape, (3,))
        self.assertTrue(np.allclose(post_mean, true, atol=0.03))
        # the fitted Categorical's pmap matches the posterior-mean probabilities
        self.assertTrue(np.allclose([m.dist.pmap[k] for k in range(3)], post_mean, atol=1e-9))
        # posterior-predictive draws are valid categories
        pp = np.asarray(m.result.predictive(20, np.random.RandomState(3))).ravel()
        self.assertTrue(set(np.unique(pp)).issubset({0, 1, 2}))


class ConjugateMixturePriorTestCase(unittest.TestCase):
    """A Mix(...) of conjugate priors has an exact reweighted-mixture-of-conjugates posterior."""

    def test_normal_bimodal_prior_selects_component(self):
        from mixle.ppl.inference import ConjugateMixturePosterior

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
        # Correctness here is exact seq==par reproducibility (asserted below), not parameter
        # recovery precision, so draws/burn can be cut hard -- verified bit-for-bit identical
        # across 5 seeds at this smaller budget. Most of the wall time is the fixed per-process
        # import cost paid by parallel=True's subprocess pool (unavoidable without touching
        # library code), so this mainly shrinks the parallel=False portion.
        seq = Normal(free, free).fit(
            data, how="hmc", draws=250, burn=150, chains=3, parallel=False, rng=np.random.RandomState(7)
        )
        par = Normal(free, free).fit(
            data, how="hmc", draws=250, burn=150, chains=3, parallel=True, rng=np.random.RandomState(7)
        )
        # identical seeds + deterministic chains -> identical posterior means
        self.assertAlmostEqual(seq.result.mean("arg0"), par.result.mean("arg0"), places=6)
        self.assertAlmostEqual(seq.result.mean("arg1"), par.result.mean("arg1"), places=6)


if __name__ == "__main__":
    unittest.main()
