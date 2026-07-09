"""Tests for mixle.ppl Bayesian inference: MAP and parameter MCMC (build slice 7)."""

import importlib.util
import pickle
import unittest

import numpy as np

from mixle.ppl import Bernoulli, Beta, Exponential, Field, Gamma, Group, Normal, Poisson, free
from mixle.ppl.inference import ConjugatePosterior

HAS_TORCH = importlib.util.find_spec("torch") is not None


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


class PPLSummaryDiagnosticsTestCase(unittest.TestCase):
    """summary() folds per-parameter convergence diagnostics (R-hat, ESS) into each row."""

    def test_multichain_summary_has_per_param_rhat_and_ess(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 1500))
        m = Normal(Normal(0, 10, name="mu"), free).fit(data, how="mcmc", draws=1200, burn=600, chains=4)
        row = m.result.summary()["mu"]
        for key in ("mean", "std", "q2.5", "q97.5", "r_hat", "ess_bulk", "ess_tail", "split_r_hat"):
            self.assertIn(key, row)
        self.assertLess(row["r_hat"], 1.1)  # converged
        self.assertGreater(row["ess_bulk"], 100)

    def test_single_chain_summary_back_compat(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(5.0, 2.0, 800))
        m = Normal(Normal(0, 10, name="mu"), free).fit(data, how="mcmc", draws=600, burn=300)
        row = m.result.summary()["mu"]
        self.assertIn("mean", row)
        self.assertIn("q2.5", row)  # the original keys are untouched


class PPLLaplaceTestCase(unittest.TestCase):
    """how='laplace' — a cheap Gaussian posterior at the MAP (the uncertainty rung above 'map')."""

    def test_laplace_recovers_and_quantifies(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 400))  # posterior sd of the mean ~ 2/sqrt(400) = 0.1
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, free).fit(data, how="laplace", rng=np.random.RandomState(1))
        s = m.result.summary()["mu"]
        self.assertAlmostEqual(s["mean"], 5.0, delta=0.25)
        self.assertAlmostEqual(s["std"], 0.1, delta=0.05)  # Hessian-based uncertainty is calibrated
        self.assertIn("q2.5", s)
        post = m.posterior(mu)
        self.assertLess(float(np.percentile(post, 2.5)), 5.0)
        self.assertGreater(float(np.percentile(post, 97.5)), 5.0)

    def test_laplace_mean_matches_map_point(self):
        rng = np.random.RandomState(2)
        data = list(rng.normal(-1.0, 1.5, 600))
        lap = Normal(Normal(0, 10), free).fit(data, how="laplace", rng=np.random.RandomState(3))
        mp = Normal(Normal(0, 10), free).fit(data, how="map")
        self.assertAlmostEqual(lap.posterior(0).mean(), mp.dist.mu, delta=0.1)


class PPLExplainFitTestCase(unittest.TestCase):
    """explain_fit() reports the route .fit(how='auto') will take, with honest caveats."""

    def test_routes_match_intent(self):
        self.assertEqual(Normal(free, free).explain_fit()["route"], "em")
        self.assertEqual(Normal(Normal(0, 10), 1.0).explain_fit()["route"], "conjugate")
        self.assertEqual(Poisson(Gamma(1, 1)).explain_fit()["route"], "conjugate")
        self.assertEqual(Normal(Normal(0, 10), free).explain_fit()["route"], "map")
        self.assertEqual(Normal(Normal(0, 5).each(), free).explain_fit()["route"], "hierarchical")
        self.assertEqual(Poisson(free * Field("x") + Group("g")).explain_fit()["route"], "glmm")
        self.assertEqual(Normal(free(3)[Field("g")], free).explain_fit()["route"], "indexed")
        # honest caveats are attached
        self.assertTrue(any("point estimate" in c for c in Normal(Normal(0, 10), free).explain_fit()["caveats"]))
        self.assertTrue(
            any("PQL" in c or "biased" in c for c in Poisson(free * Field("x") + Group("g")).explain_fit()["caveats"])
        )

    def test_explanation_matches_actual_fit_behavior(self):
        rng = np.random.RandomState(0)
        conj = Poisson(Gamma(2.0, 1.0))
        self.assertEqual(conj.explain_fit()["route"], "conjugate")
        m = conj.fit(list(rng.poisson(3.0, 500)))
        self.assertIsInstance(m.result, ConjugatePosterior)  # conjugate route -> closed-form posterior

        em = Normal(free, free)
        self.assertEqual(em.explain_fit()["route"], "em")
        m2 = em.fit(list(rng.normal(5.0, 2.0, 800)))
        self.assertTrue(m2.is_bound)
        self.assertAlmostEqual(m2.dist.mu, 5.0, delta=0.3)

    def test_explicit_how_reported_verbatim(self):
        p = Normal(free, free).explain_fit(how="nuts")
        self.assertEqual(p["route"], "nuts")
        self.assertIn("nuts", p["reason"])

    def test_bound_rv_explain_fit_reports_the_actual_route(self):
        # A bound RV's _args is always empty, so re-deriving the route from its structure (as if it
        # were the pre-fit expression) silently falls through to "em, no priors" regardless of what
        # actually happened. fit() must stash the real answer instead.
        rng = np.random.RandomState(0)
        conjugate = Poisson(Gamma(2.0, 1.0)).fit(list(rng.poisson(3.0, 500)))
        self.assertEqual(conjugate.explain_fit()["route"], "conjugate")

        em = Normal(free, free).fit(list(rng.normal(5.0, 2.0, 800)))
        self.assertEqual(em.explain_fit()["route"], "em")

        map_route = Normal(Normal(0, 10), free).fit(list(rng.normal(5.0, 2.0, 800)))
        self.assertEqual(map_route.explain_fit()["route"], "map")

        hierarchical = Normal(Normal(0, 5).each(), free).fit([[1.0, 1.2], [5.0, 4.8], [-1.0, -0.9]])
        self.assertEqual(hierarchical.explain_fit()["route"], "hierarchical")

    def test_bound_rv_without_cached_explanation_raises(self):
        # A model reloaded from a saved artifact (or otherwise bound without going through .fit())
        # has no stashed explanation; report that honestly rather than guessing.
        m = Normal(free, free).fit(list(np.random.RandomState(0).normal(size=50)))
        reloaded = pickle.loads(pickle.dumps(m))
        with self.assertRaises(RuntimeError):
            reloaded.explain_fit()


class PPLDeterministicExpressionTestCase(unittest.TestCase):
    """A parameter slot may be a deterministic expression over named latents (a + b, exp(a), ...);
    the leaves are sampled and the slot value is recomputed from them each evaluation."""

    def test_sum_of_two_latents_in_mean(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(3.0, 1.0, 600))  # a + b should recover ~3
        a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
        m = Normal(a + b, 1.0).fit(data, how="map")
        self.assertAlmostEqual(m.dist.mu, 3.0, delta=0.2)

    def test_linear_combination(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(5.0, 1.0, 600))  # 2a - b
        a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
        m = Normal(2.0 * a - b, 1.0).fit(data, how="map")
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.3)

    def test_deterministic_transform_as_scale(self):
        rng = np.random.RandomState(2)
        data = list(rng.normal(0.0, np.exp(0.5), 1500))  # scale = exp(a), a ~ 0.5
        a = Normal(0, 10, name="a")
        m = Normal(0.0, a.exp()).fit(data, how="map")
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), np.exp(0.5), delta=0.25)

    def test_shared_latent_across_slots(self):
        rng = np.random.RandomState(3)
        data = list(rng.normal(1.0, np.exp(1.0), 1500))  # mean = a, scale = exp(a), a ~ 1
        a = Normal(0, 10, name="a")
        m = Normal(a, a.exp()).fit(data, how="map")
        self.assertAlmostEqual(m.dist.mu, 1.0, delta=0.3)

    def test_mcmc_posterior_over_expression_leaves(self):
        rng = np.random.RandomState(4)
        data = list(rng.normal(3.0, 1.0, 600))
        a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
        m = Normal(a + b, 1.0).fit(data, how="mcmc", draws=1200, burn=600, rng=np.random.RandomState(5))
        # each leaf is addressable in the posterior, and their sum recovers the mean
        pa, pb = m.posterior(a), m.posterior(b)
        self.assertEqual(len(pa), 1200)
        self.assertAlmostEqual(pa.mean() + pb.mean(), 3.0, delta=0.3)


class PPLPotentialTestCase(unittest.TestCase):
    """A custom potential adds an arbitrary log-factor fn(*values) to the joint (Stan `target +=`)."""

    def test_potential_pulls_map_estimate(self):
        from mixle.ppl import potential

        rng = np.random.RandomState(0)
        data = list(rng.normal(2.0, 1.0, 20))  # data alone -> mean ~2.6
        a, b = Normal(0, 10, name="a"), Normal(8, 0.5, name="b")
        base = Normal(a, 1.0).fit(data, how="map").params["mean"]
        pulled = (
            Normal(a, 1.0)
            .fit(data, how="map", potentials=potential(lambda av, bv: -50.0 * (av - bv) ** 2, a, b))
            .params["mean"]
        )
        self.assertGreater(pulled, base + 0.5)  # a strong coupling to b=8 drags the estimate up

    def test_potential_introduces_auxiliary_latent(self):
        from mixle.ppl import potential

        rng = np.random.RandomState(1)
        data = list(rng.normal(2.0, 1.0, 400))  # a is pinned near 2 by the data
        a, b = Normal(0, 10, name="a"), Normal(5, 1, name="b")  # b appears ONLY in the potential
        m = Normal(a, 1.0).fit(
            data,
            how="mcmc",
            draws=2000,
            burn=1000,
            rng=np.random.RandomState(2),
            potentials=potential(lambda av, bv: -8.0 * (av - bv) ** 2, a, b),
        )
        # b is a real auxiliary latent with its own posterior, pulled from its prior mean (5) toward a (~2)
        pb = m.posterior(b)
        self.assertEqual(len(pb), 2000)
        self.assertLess(float(np.mean(pb)), 4.0)
        self.assertGreater(float(np.mean(pb)), 2.0)

    def test_potential_rejected_for_closed_form_how(self):
        from mixle.ppl import potential

        data = list(np.random.RandomState(3).normal(0.0, 1.0, 100))
        a = Normal(0, 10, name="a")
        with self.assertRaises(ValueError):
            Normal(a, 1.0).fit(data, how="conjugate", potentials=potential(lambda av: -(av**2), a))


class PPLIndexedLatentTestCase(unittest.TestCase):
    """A data-indexed latent vector theta[Field('g')] is fit per-observation (MAP)."""

    def test_recovers_latent_vector(self):
        rng = np.random.RandomState(0)
        K = 6
        theta_true = rng.normal(0.0, 5.0, K)
        labels = rng.randint(0, K, 500)
        y = rng.normal(theta_true[labels], 0.7)
        theta = free(K, name="theta")
        m = Normal(theta[Field("g")], free).fit(y, given={"g": labels})
        est = m.result.latents["theta"]
        self.assertEqual(est.shape, (K,))
        self.assertLess(float(np.max(np.abs(est - theta_true))), 0.6)
        np.testing.assert_array_equal(m.result.group_means, est)  # single-vector alias

    def test_gather_in_expression(self):
        rng = np.random.RandomState(1)
        K = 5
        theta_true = rng.normal(0.0, 3.0, K)
        labels = rng.randint(0, K, 400)
        y = rng.normal(theta_true[labels] + 10.0, 0.7)  # gather composed with a constant offset
        theta = free(K, name="theta")
        m = Normal(theta[Field("g")] + 10.0, free).fit(y, given={"g": labels})
        self.assertLess(float(np.max(np.abs(m.result.latents["theta"] - theta_true))), 0.6)

    def test_requires_index_covariate(self):
        rng = np.random.RandomState(2)
        labels = rng.randint(0, 3, 30)
        y = rng.normal(0.0, 1.0, 30)
        theta = free(3, name="theta")
        with self.assertRaises(ValueError):
            Normal(theta[Field("g")], free).fit(y)  # no given

    def test_mcmc_posterior_over_latent_vector(self):
        rng = np.random.RandomState(0)
        K = 5
        theta_true = rng.normal(0.0, 3.0, K)
        labels = rng.randint(0, K, 600)
        y = rng.normal(theta_true[labels], 0.8)
        theta = free(K, name="theta")
        m = Normal(theta[Field("g")], free).fit(
            y, given={"g": labels}, how="mcmc", draws=1500, burn=800, rng=np.random.RandomState(1)
        )
        s = m.result.summary()
        means = np.array([s[f"theta[{k}]"]["mean"] for k in range(K)])
        self.assertLess(float(np.max(np.abs(means - theta_true))), 0.6)
        # each latent has a credible interval (a posterior, not just a point)
        self.assertIn("q2.5", s["theta[0]"])
        self.assertLess(s["theta[0]"]["q2.5"], s["theta[0]"]["q97.5"])


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

    def test_indexed_flat_varying_intercepts(self):
        # 8-schools idiom: a flat observation array + a group-index covariate, fit with each(by=...).
        rng = np.random.RandomState(0)
        G = 8
        theta_true = rng.normal(5.0, 4.0, G)
        labels, y = [], []
        for g in range(G):
            n = rng.randint(20, 40)
            labels += [g] * n
            y += list(rng.normal(theta_true[g], 1.0, n))
        labels, y = np.array(labels), np.array(y)
        fit = Normal(Normal(0, 100).each(by="school"), free).fit(y, given={"school": labels})
        h = fit.result.hyper
        self.assertAlmostEqual(h["m"], float(theta_true.mean()), delta=0.6)  # mean of the group draws
        self.assertAlmostEqual(h["tau"], 4.0, delta=1.5)
        self.assertAlmostEqual(h["sigma"], 1.0, delta=0.2)
        # per-group latents recovered in sorted-label order
        gm = np.asarray(fit.result.summary()["group_means"])
        self.assertEqual(gm.shape, (G,))
        self.assertLess(float(np.max(np.abs(gm - theta_true))), 0.6)

    def test_indexed_flat_requires_and_checks_the_index(self):
        rng = np.random.RandomState(1)
        y = list(rng.normal(0.0, 1.0, 30))
        labels = np.array([0, 1] * 15)
        with self.assertRaises(ValueError):  # missing given
            Normal(Normal(0, 100).each(by="g"), free).fit(y)
        with self.assertRaises(ValueError):  # index length mismatch
            Normal(Normal(0, 100).each(by="g"), free).fit(y, given={"g": labels[:5]})

    def test_gamma_poisson_random_effects(self):
        rng = np.random.RandomState(0)
        G = 300
        lam = rng.gamma(4.0, 1 / 2.0, G)  # rates ~ Gamma(shape=4, rate=2), mean 2
        data = [list(rng.poisson(lam[i], rng.randint(5, 20)).astype(float)) for i in range(G)]
        fit = Poisson(Gamma(1, 1).each()).fit(data)
        self.assertAlmostEqual(fit.result.hyper["mean"], 2.0, delta=0.3)
        self.assertGreater(np.corrcoef(fit.result.group_means, lam)[0, 1], 0.85)

    def test_poisson_indexed_flat(self):
        # non-Normal varying intercepts on a flat array: y[i] ~ Poisson(lam[g[i]]), lam_g ~ Gamma
        rng = np.random.RandomState(0)
        G = 200
        lam = rng.gamma(4.0, 1 / 2.0, G)
        labels, y = [], []
        for g in range(G):
            n = rng.randint(5, 20)
            labels += [g] * n
            y += list(rng.poisson(lam[g], n).astype(float))
        labels, y = np.array(labels), np.array(y)
        fit = Poisson(Gamma(1, 1).each(by="g")).fit(y, given={"g": labels})
        self.assertAlmostEqual(fit.result.hyper["mean"], 2.0, delta=0.3)
        gm = np.asarray(fit.result.summary()["group_means"])
        self.assertEqual(gm.shape, (G,))
        self.assertGreater(np.corrcoef(gm, lam)[0, 1], 0.85)

    def test_bernoulli_indexed_flat(self):
        rng = np.random.RandomState(0)
        G = 200
        p = rng.beta(2.0, 5.0, G)
        labels, y = [], []
        for g in range(G):
            n = rng.randint(10, 40)
            labels += [g] * n
            y += list((rng.random(n) < p[g]).astype(float))
        labels, y = np.array(labels), np.array(y)
        fit = Bernoulli(Beta(1, 1).each(by="g")).fit(y, given={"g": labels})
        self.assertAlmostEqual(fit.result.hyper["mean"], 0.286, delta=0.06)
        self.assertGreater(np.corrcoef(np.asarray(fit.result.summary()["group_means"]), p)[0, 1], 0.8)

    def test_beta_bernoulli_random_effects(self):
        rng = np.random.RandomState(0)
        G = 300
        p = rng.beta(2.0, 5.0, G)  # p_i ~ Beta(2,5), mean 0.286
        data = [list((rng.random(rng.randint(10, 40)) < p[i]).astype(float)) for i in range(G)]
        fit = Bernoulli(Beta(1, 1).each()).fit(data)
        self.assertAlmostEqual(fit.result.hyper["mean"], 0.286, delta=0.06)
        self.assertGreater(np.corrcoef(fit.result.group_means, p)[0, 1], 0.8)


class PPLVariationalFamilyTestCase(unittest.TestCase):
    """Richer VB: full-rank Gaussian q (captures parameter correlations) and the tilted
    Renyi-alpha objective (alpha<1 is mass-covering, widening the too-narrow KL fit)."""

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_fullrank_captures_correlation(self):
        rng = np.random.RandomState(0)
        data = list(rng.gamma(3.0, 1.0 / 2.0, 300))  # Gamma(shape,rate) posterior is strongly correlated
        mf = Gamma(free, free).fit(data, how="vi", family="meanfield", steps=1500, rng=np.random.RandomState(1))
        fr = Gamma(free, free).fit(data, how="vi", family="fullrank", steps=1500, rng=np.random.RandomState(1))

        def corr(m):
            return float(np.corrcoef(m.result.samples("arg0"), m.result.samples("arg1"))[0, 1])

        self.assertLess(abs(corr(mf)), 0.2)  # mean-field forces independence
        self.assertGreater(corr(fr), 0.8)  # full-rank recovers the strong correlation
        self.assertAlmostEqual(fr.params["shape"], 3.0, delta=0.6)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_tilted_alpha_widens_posterior(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(5.0, 2.0, 400))
        kl = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="vi", alpha=1.0, steps=400, rng=np.random.RandomState(2)
        )
        iwae = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="vi", alpha=0.0, mc=16, steps=400, rng=np.random.RandomState(2)
        )
        self.assertAlmostEqual(iwae.params["mean"], 5.0, delta=0.25)
        # the tilted (importance-weighted) objective is mass-covering -> not narrower than KL
        self.assertGreaterEqual(float(np.std(iwae.posterior("mu"))), 0.9 * float(np.std(kl.posterior("mu"))))
        # objective metadata records which bound was optimized
        self.assertEqual(kl.result.raw.objective_kind, "kl_elbo")
        self.assertEqual(iwae.result.raw.objective_kind, "renyi_tilted")
        self.assertEqual(iwae.result.raw.alpha, 0.0)


class PPLAutoSamplerTestCase(unittest.TestCase):
    def test_sample_auto_picks_and_recovers(self):
        # how='sample' chooses the sampler for you (ensemble for low-dim) and recovers params
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, size=3000))
        m = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="sample", draws=600, burn=200, rng=np.random.RandomState(1)
        )
        self.assertAlmostEqual(float(m.result.mean("mu")), 5.0, delta=0.2)


class PPLNUTSTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(5.0, 2.0, size=3000))

    def test_nuts_recovers_params(self):
        m = Normal(Normal(0, 10, name="mu"), free).fit(
            self.data, how="nuts", draws=800, burn=500, rng=np.random.RandomState(1)
        )
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.2)
        # NUTS adapts its step size and reports tree depth per draw
        self.assertGreater(m.result.raw.step_size, 0.0)

    def test_nuts_high_ess_per_draw(self):
        # NUTS mixes well: effective sample size is a large fraction of the draws
        m = Normal(Normal(0, 10, name="mu"), free).fit(
            self.data, how="nuts", draws=1000, burn=500, rng=np.random.RandomState(2)
        )
        ess = float(np.atleast_1d(m.result.raw.effective_sample_size()).min())
        self.assertGreater(ess, 300)

    def test_nuts_multichain_rhat(self):
        m = Normal(Normal(0, 10, name="mu"), free).fit(
            self.data, how="nuts", draws=600, burn=400, chains=3, rng=np.random.RandomState(3)
        )
        self.assertEqual(m.result.n_chains, 3)
        for r in m.result.rhat.values():
            self.assertLess(r, 1.1)


class PPLVITestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(5.0, 2.0, size=3000))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_vi_recovers_params(self):
        mu = Normal(0, 10, name="mu")
        m = Normal(mu, free).fit(self.data, how="vi", rng=np.random.RandomState(1))
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.2)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.2)
        # variational posterior draws + ELBO available
        self.assertEqual(len(m.posterior("mu")), 4000)
        self.assertTrue(np.isfinite(m.result.raw.elbo))
        # the variational fit records its objective metadata, not a placeholder
        self.assertNotEqual(m.result.raw.elbo, 0.0)
        self.assertEqual(m.result.raw.objective_kind, "kl_elbo")
        self.assertEqual(m.result.raw.family, "meanfield")

    def test_vi_handles_non_conjugate(self):
        # sd has a Gamma prior -> not a registered conjugate pair; VI must handle it
        m = Normal(Normal(0, 10, name="mu"), Gamma(2, 1, name="sd")).fit(
            self.data, how="vi", rng=np.random.RandomState(2)
        )
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.3)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.3)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
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

    def test_minibatch_vi_scales(self):
        # stochastic minibatch VI (SGVB) recovers the same answer from a small per-step shard,
        # so VB scales to large data.
        rng = np.random.RandomState(6)
        data = list(rng.normal(5.0, 2.0, 50000))
        m = Normal(Normal(0, 10, name="mu"), free).fit(
            data, how="vi", steps=300, batch_size=256, rng=np.random.RandomState(7)
        )
        self.assertAlmostEqual(m.params["mean"], 5.0, delta=0.2)

    def test_minibatch_vi_positive_support(self):
        rng = np.random.RandomState(8)
        data = list(rng.poisson(3.5, 50000).astype(float))
        m = Poisson(Gamma(2, 1, name="rate")).fit(
            data, how="vi", steps=300, batch_size=256, rng=np.random.RandomState(9)
        )
        self.assertAlmostEqual(m.params["rate"], 3.5, delta=0.3)


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
