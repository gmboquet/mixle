"""mixle.ppl: parameter MCMC / HMC / ensemble for *composite* models (mixtures, ...).

Composites lower through the generic numerical target (collect leaf free/prior params across
the tree, rebuild a concrete model per evaluation), so mcmc / ensemble / map all work on them.
Mixtures need an identifiability constraint (ordered component means) to break label-switching
— the standard requirement, and exactly what the constraint surface provides.
"""

import importlib.util
import unittest

import numpy as np

from mixle.ppl import Dirichlet, Gamma, Markov, Mix, Normal, free

HAS_TORCH = importlib.util.find_spec("torch") is not None


class HMMStructuralParameterTestCase(unittest.TestCase):
    """An HMM's transition matrix (rows of simplices) and initial distribution are inferable."""

    def _sequences(self):
        rng = np.random.RandomState(0)
        true_t = np.array([[0.9, 0.1], [0.2, 0.8]])
        mu = [-3.0, 3.0]

        def simulate(length):
            s = rng.randint(2)
            out = []
            for _ in range(length):
                out.append(rng.normal(mu[s], 1.0))
                s = rng.choice(2, p=true_t[s])
            return out

        return [simulate(rng.randint(8, 15)) for _ in range(400)]

    def test_transition_matrix_and_initial_inferred(self):
        seqs = self._sequences()
        m0, m1 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1")
        fit = Markov([Normal(m0, 1.0), Normal(m1, 1.0)], transitions=free, initial=free).fit(
            seqs, how="ensemble", constraints=m0 < m1, draws=600, burn=250, rng=np.random.RandomState(1)
        )
        t = fit.params["transitions"]
        self.assertEqual(t.shape, (2, 2))
        self.assertTrue(np.allclose(t.sum(axis=1), 1.0))  # each row a valid simplex
        self.assertAlmostEqual(t[0, 0], 0.9, delta=0.1)  # self-transition of the low state
        self.assertAlmostEqual(t[1, 1], 0.8, delta=0.1)
        self.assertAlmostEqual(fit.result.mean("m0"), -3.0, delta=0.4)
        self.assertAlmostEqual(fit.result.mean("m1"), 3.0, delta=0.4)
        self.assertTrue(np.allclose(np.sum(fit.params["initial"]), 1.0))

    def test_em_default_still_works(self):
        seqs = self._sequences()
        m = Markov(Normal(free, free), states=2).fit(seqs)  # transitions=None -> EM
        t = m.params["transitions"]
        self.assertEqual(t.shape, (2, 2))
        self.assertTrue(np.allclose(t.sum(axis=1), 1.0))


class MixtureWeightsAsParameterTestCase(unittest.TestCase):
    """Mixture weights are an inferable simplex parameter (Gamma representation of the
    Dirichlet): a Dirichlet(alpha) prior or `free`, recovered alongside the components."""

    def setUp(self):
        rng = np.random.RandomState(0)
        n = 4000
        z = rng.uniform(size=n) < 0.7  # 70% / 30% split
        self.data = list(np.where(z, rng.normal(-3, 1, n), rng.normal(3, 1, n)))

    def test_dirichlet_prior_weights(self):
        m0, m1 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1")
        w = Dirichlet([1.0, 1.0], name="w")
        fit = Mix([Normal(m0, 1.0), Normal(m1, 1.0)], w).fit(
            self.data, how="ensemble", constraints=m0 < m1, draws=600, burn=250, rng=np.random.RandomState(1)
        )
        wts = fit.params["weights"]
        self.assertAlmostEqual(wts[0], 0.7, delta=0.05)
        self.assertAlmostEqual(wts[1], 0.3, delta=0.05)
        self.assertAlmostEqual(float(np.sum(wts)), 1.0, places=6)  # stayed on the simplex

    def test_free_weights(self):
        m0, m1 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1")
        fit = Mix([Normal(m0, 1.0), Normal(m1, 1.0)], free).fit(
            self.data, how="ensemble", constraints=m0 < m1, draws=600, burn=250, rng=np.random.RandomState(2)
        )
        wts = fit.params["weights"]
        self.assertAlmostEqual(wts[0], 0.7, delta=0.05)
        self.assertAlmostEqual(float(np.sum(wts)), 1.0, places=6)

    def test_three_component_weights(self):
        rng = np.random.RandomState(3)
        n = 4500  # 50/30/20 over means -5, 0, 5
        u = rng.uniform(size=n)
        x = np.where(u < 0.5, rng.normal(-5, 1, n), np.where(u < 0.8, rng.normal(0, 1, n), rng.normal(5, 1, n)))
        m0, m1, m2 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1"), Normal(0, 10, name="m2")
        fit = Mix([Normal(m0, 1.0), Normal(m1, 1.0), Normal(m2, 1.0)], Dirichlet([1.0, 1.0, 1.0])).fit(
            list(x),
            how="ensemble",
            constraints=(m0 < m1) & (m1 < m2),
            draws=1500,
            burn=650,
            walkers=40,  # 3-component posterior needs a larger ensemble to mix reliably
            rng=np.random.RandomState(4),
        )
        wts = fit.params["weights"]
        self.assertEqual(len(wts), 3)
        self.assertAlmostEqual(float(np.sum(wts)), 1.0, places=6)
        self.assertAlmostEqual(wts[0], 0.5, delta=0.08)
        self.assertAlmostEqual(wts[2], 0.2, delta=0.08)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class MixtureAutogradTestCase(unittest.TestCase):
    """A mixture of leaf components gets analytic Torch gradients (MixtureGradTarget): the
    autograd log-target matches the numeric one, and NUTS / full-rank VB work on it."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(np.concatenate([rng.normal(-3, 1, 1500), rng.normal(3, 1, 1500)]))

    def test_autograd_logtarget_matches_numeric(self):
        from mixle.ppl import autograd as ag
        from mixle.ppl.inference import _build_target, _init_u

        m0, m1 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1")
        mix = Mix([Normal(m0, 1.0), Normal(m1, 1.0)], free)
        g = ag.grad_target(mix, self.data)
        self.assertEqual(type(g).__name__, "MixtureGradTarget")
        lt_np, slots, *_ = _build_target(mix, self.data)
        u0 = _init_u(slots, g.dmean, g.dstd)
        for u in (u0, u0 + 0.3, u0 - 0.2):
            self.assertAlmostEqual(g.log_target(u), lt_np(u), places=3)

    def test_nuts_on_mixture(self):
        m0, m1 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1")
        fit = Mix([Normal(m0, 1.0), Normal(m1, 1.0)], free).fit(
            self.data, how="nuts", constraints=m0 < m1, draws=300, burn=250, rng=np.random.RandomState(1)
        )
        self.assertAlmostEqual(fit.result.mean("m0"), -3.0, delta=0.4)
        self.assertAlmostEqual(fit.result.mean("m1"), 3.0, delta=0.4)

    def test_fullrank_vi_on_mixture(self):
        # VB's unimodal q picks one labeling, so no ordering constraint is needed
        m0, m1 = Normal(0, 10, name="m0"), Normal(0, 10, name="m1")
        fit = Mix([Normal(m0, 1.0), Normal(m1, 1.0)], free).fit(
            self.data, how="vi", family="fullrank", steps=600, rng=np.random.RandomState(3)
        )
        means = sorted([fit.result.mean("m0"), fit.result.mean("m1")])
        self.assertAlmostEqual(means[0], -3.0, delta=0.4)
        self.assertAlmostEqual(means[1], 3.0, delta=0.4)


class CompositeMixtureSamplingTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(np.concatenate([rng.normal(-3, 1, 1500), rng.normal(3, 1, 1500)]))

    def _ordered_means(self):
        m0 = Normal(0, 10, name="m0")
        m1 = Normal(0, 10, name="m1")
        return Mix([Normal(m0, 1.0), Normal(m1, 1.0)]), m0, m1

    def test_ensemble_recovers_ordered_means(self):
        model, m0, m1 = self._ordered_means()
        fit = model.fit(
            self.data, how="ensemble", constraints=m0 < m1, draws=800, burn=300, rng=np.random.RandomState(1)
        )
        self.assertAlmostEqual(fit.result.mean("m0"), -3.0, delta=0.4)
        self.assertAlmostEqual(fit.result.mean("m1"), 3.0, delta=0.4)
        self.assertLess(fit.result.mean("m0"), fit.result.mean("m1"))

    def test_mcmc_recovers_ordered_means(self):
        model, m0, m1 = self._ordered_means()
        fit = model.fit(self.data, how="mcmc", constraints=m0 < m1, draws=1500, burn=500, rng=np.random.RandomState(2))
        self.assertAlmostEqual(fit.result.mean("m0"), -3.0, delta=0.4)
        self.assertAlmostEqual(fit.result.mean("m1"), 3.0, delta=0.4)

    def test_map_recovers_ordered_means(self):
        model, m0, m1 = self._ordered_means()
        fit = model.fit(self.data, how="map", constraints=m0 < m1)
        comps = fit.params["components"]
        self.assertAlmostEqual(comps[0]["mean"], -3.0, delta=0.4)
        self.assertAlmostEqual(comps[1]["mean"], 3.0, delta=0.4)

    def test_posterior_handle_lookup(self):
        model, m0, m1 = self._ordered_means()
        fit = model.fit(self.data, how="mcmc", constraints=m0 < m1, draws=600, burn=200, rng=np.random.RandomState(3))
        self.assertEqual(fit.result.samples("m0").shape, (600,))
        self.assertIn("m0", fit.result.summary())

    def test_positive_support_prior_in_composite(self):
        # a Gamma prior (positive support) on a component sd must reparameterize correctly
        rng = np.random.RandomState(4)
        data = list(np.concatenate([rng.normal(-3, 0.5, 1500), rng.normal(3, 2.0, 1500)]))
        m0 = Normal(0, 10, name="m0")
        m1 = Normal(0, 10, name="m1")
        s1 = Gamma(2, 1, name="s1")
        fit = Mix([Normal(m0, 0.5), Normal(m1, s1)]).fit(
            data, how="ensemble", constraints=m0 < m1, draws=1000, burn=400, rng=np.random.RandomState(5)
        )
        s1_hat = fit.result.mean("s1")
        self.assertGreater(s1_hat, 0.0)  # the log-reparameterization kept it positive
        self.assertTrue(0.8 < s1_hat < 5.0)  # a sensible positive estimate (true 2.0)


if __name__ == "__main__":
    unittest.main()
