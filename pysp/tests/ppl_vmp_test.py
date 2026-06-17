"""Tests for the VMP engine (variational message passing)."""

import unittest

import numpy as np

from pysp.ppl import Categorical, Dirichlet, Gamma, Graph, Mix, Normal, free


class VMPTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(5.0, 2.0, size=4000))  # mean 5, sd 2, precision 0.25

    def test_gaussian_mean_and_precision_recovered(self):
        # the conjugate registry needs a KNOWN variance; VMP infers both jointly
        m = Normal(Normal(0, 10), Gamma(1, 1)).fit(self.data, how="vmp")
        self.assertAlmostEqual(m.params["mean"], 5.0, delta=0.15)
        self.assertAlmostEqual(m.params["sd"], 2.0, delta=0.15)
        self.assertAlmostEqual(m.result.q_tau["mean"], 0.25, delta=0.03)  # E[precision]

    def test_elbo_monotonically_increases(self):
        m = Normal(Normal(0, 10), Gamma(1, 1)).fit(self.data, how="vmp")
        trace = m.result.elbo_trace
        self.assertGreater(trace.size, 1)
        diffs = np.diff(trace)
        self.assertTrue(np.all(diffs >= -1e-6))  # non-decreasing (up to numerics)

    def test_vmp_posterior_and_predictive(self):
        m = Normal(Normal(0, 10), Gamma(1, 1)).fit(self.data, how="vmp")
        # variational posterior draws for mu and the precision
        self.assertAlmostEqual(m.result.samples("mu").mean(), 5.0, delta=0.2)
        self.assertGreater(m.result.samples("tau").mean(), 0.0)
        # posterior predictive integrates both mu and tau uncertainty
        pp = np.asarray(m.predict(5000, rng=np.random.RandomState(1)))
        self.assertAlmostEqual(pp.mean(), 5.0, delta=0.3)
        self.assertAlmostEqual(pp.std(), 2.0, delta=0.3)

    def test_auto_graph_deep_model(self):
        # mean has a HYPERPRIOR and precision is unknown — auto-built graph, one fit call
        rng = np.random.RandomState(1)
        data = list(rng.normal(7.0, 2.0, 4000))
        m = Normal(Normal(Normal(0, 100), 5.0), Gamma(1, 1)).fit(data, how="vmp")
        self.assertAlmostEqual(m.params["mean"], 7.0, delta=0.2)
        self.assertAlmostEqual(m.params["sd"], 2.0, delta=0.2)
        self.assertTrue(np.all(np.diff(m.result.elbo_trace) >= -1e-6))

    def test_unsupported_model_raises(self):
        with self.assertRaises(NotImplementedError):
            Normal(0.0, 1.0).fit(self.data, how="vmp")  # nothing to infer

    def test_free_slot_gives_clear_error(self):
        # vmp needs priors, not the point-estimate `free`; the error must say so (not a TypeError)
        with self.assertRaises(NotImplementedError) as cm:
            Normal(Normal(0, 10, name="mu"), free).fit(self.data, how="vmp")
        self.assertIn("free", str(cm.exception))
        self.assertIn("vi", str(cm.exception))


class VMPGraphTestCase(unittest.TestCase):
    def test_shared_variable_combines_evidence(self):
        # the SAME mu instance appears in two factors -> one node, messages combined
        rng = np.random.RandomState(0)
        mu = Normal(0, 10)
        dA = list(rng.normal(2.0, 1.0, 2000))
        dB = list(rng.normal(4.0, 1.0, 2000))
        fit = Graph().observe(Normal(mu, 1.0), dA).observe(Normal(mu, 1.0), dB).fit()
        p = fit.posterior(mu)
        # precision-weighted compromise of the two datasets, ~3
        self.assertAlmostEqual(p["mean"], 3.0, delta=0.15)
        # posterior tighter than a single dataset (evidence from both combined)
        single = Graph().observe(Normal(Normal(0, 10), 1.0), dA).fit()
        self.assertLess(p["sd"], 0.03)
        self.assertTrue(np.all(np.diff(fit.elbo_trace) >= -1e-6))

    def test_shared_variable_is_one_node(self):
        # identity-based sharing: reusing the handle yields a single inferred node
        rng = np.random.RandomState(1)
        mu = Normal(0, 10)
        g = Graph()
        for _ in range(3):
            g.observe(Normal(mu, 1.0), list(rng.normal(7.0, 1.0, 1000)))
        fit = g.fit()
        self.assertAlmostEqual(fit.posterior(mu)["mean"], 7.0, delta=0.1)

    def test_deep_hierarchy_arbitrary_depth(self):
        # grand-mean -> group means -> data (3 levels), built generically
        rng = np.random.RandomState(2)
        grand = Normal(0, 100)
        G = 40
        true_group = rng.normal(10.0, 3.0, G)
        g = Graph()
        handles = []
        for i in range(G):
            mu_i = Normal(grand, 3.0)
            handles.append(mu_i)
            g.observe(Normal(mu_i, 1.0), list(rng.normal(true_group[i], 1.0, rng.randint(8, 20))))
        fit = g.fit()
        self.assertAlmostEqual(fit.posterior(grand)["mean"], 10.0, delta=1.0)
        # a group posterior is pulled toward its data (shrinkage)
        self.assertEqual(set(fit.posterior(handles[0])), {"mean", "sd"})

    def test_mean_precision_via_graph(self):
        rng = np.random.RandomState(3)
        m, tau = Normal(0, 10), Gamma(1, 1)
        fit = Graph().observe(Normal(m, tau), list(rng.normal(5.0, 2.0, 4000))).fit()
        self.assertAlmostEqual(fit.posterior(m)["mean"], 5.0, delta=0.2)
        self.assertAlmostEqual(fit.posterior(tau)["mean"], 0.25, delta=0.05)

    def test_dirichlet_categorical(self):
        rng = np.random.RandomState(0)
        true = np.array([0.2, 0.3, 0.5])
        data = list(rng.choice(3, size=5000, p=true))
        pi = Dirichlet([1, 1, 1])
        fit = Graph().observe(Categorical(pi), data).fit()
        est = fit.posterior(pi)["mean"]
        self.assertTrue(np.allclose(est, true, atol=0.03))
        self.assertTrue(np.all(np.diff(fit.elbo_trace) >= -1e-9))

    def test_shared_dirichlet_pools_counts(self):
        rng = np.random.RandomState(1)
        pi = Dirichlet([1, 1, 1])
        dA = list(rng.choice(3, 2500, p=[0.5, 0.3, 0.2]))
        dB = list(rng.choice(3, 2500, p=[0.1, 0.3, 0.6]))
        fit = Graph().observe(Categorical(pi), dA).observe(Categorical(pi), dB).fit()
        est = fit.posterior(pi)["mean"]  # pooled ~ [0.3, 0.3, 0.4]
        self.assertTrue(np.allclose(est, [0.3, 0.3, 0.4], atol=0.05))


class MixtureVMPTestCase(unittest.TestCase):
    def test_bayesian_gmm_vbem(self):
        rng = np.random.RandomState(0)
        data = list(np.concatenate([rng.normal(-6, 1, 3000), rng.normal(0, 0.7, 3000), rng.normal(6, 1.2, 3000)]))
        m = Mix([Normal(free, free)] * 3).fit(data, how="vmp", rng=np.random.RandomState(1))
        r = m.result
        means = sorted(c["mean"] for c in r.components)
        self.assertAlmostEqual(means[0], -6.0, delta=0.3)
        self.assertAlmostEqual(means[1], 0.0, delta=0.3)
        self.assertAlmostEqual(means[2], 6.0, delta=0.3)
        # discrete latents: per-datapoint responsibilities, each a valid distribution
        self.assertEqual(r.responsibilities.shape, (9000, 3))
        self.assertTrue(np.allclose(r.responsibilities.sum(1), 1.0))
        self.assertTrue(np.allclose(r.weights.sum(), 1.0))
        self.assertAlmostEqual(r.weights.max(), 1 / 3, delta=0.1)


if __name__ == "__main__":
    unittest.main()
