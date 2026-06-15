"""Smoke + correctness tests for the pysp.ppl facade (build slices 1-2)."""
import numpy as np
import unittest

from pysp.ppl import Normal, Poisson, Gamma, Exponential, Categorical, Mix, Seq, Markov, free, compare
from pysp.stats.gaussian import GaussianDistribution


class PPLCoreTestCase(unittest.TestCase):

    def test_concrete_construction_and_query(self):
        x = Normal(0.0, 1.0)
        self.assertFalse(x.has_free)
        self.assertIsInstance(x.dist, GaussianDistribution)
        # log_prob matches the raw distribution
        self.assertAlmostEqual(x.log_prob(0.0), x.dist.log_density(0.0), places=10)
        # vectorized log_prob
        lp = x.log_prob([0.0, 1.0, -1.0])
        self.assertEqual(lp.shape, (3,))

    def test_sample_shape(self):
        s = Normal(2.0, 0.5).sample(50, seed=1)
        self.assertEqual(np.asarray(s).shape, (50,))

    def test_fit_recovers_gaussian(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, size=20000))
        m = Normal(free, free).fit(data, max_its=50)
        self.assertTrue(m.is_bound)
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.1)
        self.assertAlmostEqual(np.sqrt(m.dist.sigma2), 2.0, delta=0.1)
        # a fitted RV still answers the query verbs
        self.assertEqual(np.asarray(m.sample(10)).shape, (10,))

    def test_fit_recovers_poisson(self):
        rng = np.random.RandomState(1)
        data = list(rng.poisson(3.5, size=20000).astype(float))
        m = Poisson(free).fit(data, max_its=50)
        self.assertAlmostEqual(m.dist.lam, 3.5, delta=0.1)

    def test_fit_recovers_exponential(self):
        rng = np.random.RandomState(2)
        data = list(rng.exponential(1.0 / 0.7, size=20000))  # rate 0.7 -> mean 1/0.7
        m = Exponential(free).fit(data, max_its=50)
        # fitted beta is the mean = 1/rate
        self.assertAlmostEqual(1.0 / m.dist.beta, 0.7, delta=0.05)

    def test_mixture_fit_and_posterior(self):
        rng = np.random.RandomState(3)
        a = rng.normal(-5.0, 1.0, size=8000)
        b = rng.normal(5.0, 1.0, size=8000)
        data = list(np.concatenate([a, b]))
        m = Mix([Normal(free, free), Normal(free, free)]).fit(
            data, max_its=80, rng=np.random.RandomState(7))
        self.assertTrue(m.is_bound)
        means = sorted(c.mu for c in m.dist.components)
        self.assertAlmostEqual(means[0], -5.0, delta=0.3)
        self.assertAlmostEqual(means[1], 5.0, delta=0.3)
        # responsibilities: a point far left should load the left component near 1
        post = m.posterior([-5.0, 5.0])
        self.assertEqual(post.shape, (2, 2))
        self.assertTrue(post[0].max() > 0.95)
        self.assertTrue(post[1].max() > 0.95)

    def test_sequence_fit(self):
        rng = np.random.RandomState(5)
        seqs = [list(rng.normal(2.0, 1.5, size=rng.randint(5, 15))) for _ in range(2000)]
        m = Seq(Normal(free, free)).fit(seqs, max_its=40)
        self.assertAlmostEqual(m.dist.dist.mu, 2.0, delta=0.1)
        self.assertAlmostEqual(np.sqrt(m.dist.dist.sigma2), 1.5, delta=0.1)

    def test_hmm_fit_separates_states(self):
        rng = np.random.RandomState(0)
        A = np.array([[0.9, 0.1], [0.15, 0.85]])
        mus = [-5.0, 5.0]
        data = []
        for _ in range(400):
            s = rng.randint(2)
            seq = []
            for _t in range(20):
                seq.append(rng.normal(mus[s], 1.0))
                s = rng.choice(2, p=A[s])
            data.append(seq)
        hmm = Markov(Normal(free, free), states=2).fit(
            data, max_its=60, rng=np.random.RandomState(2))
        emis = sorted(t.mu for t in hmm.dist.topics)
        self.assertAlmostEqual(emis[0], -5.0, delta=0.4)
        self.assertAlmostEqual(emis[1], 5.0, delta=0.4)

    def test_unresolved_free_raises(self):
        with self.assertRaises(ValueError):
            _ = Normal(free, free).dist  # must fit first

    def test_algebra_exp_is_lognormal(self):
        ln = Normal(0.0, 1.0).exp()
        s = np.asarray(ln.sample(200000, seed=1))
        self.assertAlmostEqual(s.mean(), np.exp(0.5), delta=0.05)   # E[lognormal]=exp(mu+sd^2/2)
        self.assertAlmostEqual(np.median(s), 1.0, delta=0.05)
        # density with Jacobian correction: lognormal(0,1) at 1 has log-density -0.5*log(2pi)
        self.assertAlmostEqual(ln.log_prob(1.0), -0.5 * np.log(2 * np.pi), places=4)

    def test_algebra_affine(self):
        y = 3 * Normal(0.0, 1.0) + 1
        s = np.asarray(y.sample(200000, seed=2))
        self.assertAlmostEqual(s.mean(), 1.0, delta=0.05)
        self.assertAlmostEqual(s.std(), 3.0, delta=0.05)

    def test_convolution_normal_normal_exact(self):
        z = Normal(0, 1) + Normal(5, 2)               # -> Normal(5, sqrt(5))
        s = np.asarray(z.sample(200000, seed=1))
        self.assertAlmostEqual(s.mean(), 5.0, delta=0.05)
        self.assertAlmostEqual(s.std(), np.sqrt(5), delta=0.05)
        # exact closed-form density (not KDE)
        self.assertAlmostEqual(z.log_prob(5.0), -0.5 * np.log(2 * np.pi * 5), places=6)

    def test_convolution_poisson_and_difference(self):
        self.assertAlmostEqual(np.mean(Poisson(2) + Poisson(3)).sample if False else
                               float(np.mean((Poisson(2) + Poisson(3)).sample(100000, seed=2))),
                               5.0, delta=0.1)
        d = Normal(5, 1) - Normal(2, 1)               # -> Normal(3, sqrt(2))
        self.assertAlmostEqual(float(np.mean(d.sample(100000, seed=3))), 3.0, delta=0.05)

    def test_rv_product_rejected(self):
        with self.assertRaises(NotImplementedError):
            _ = Normal(0, 1) * Normal(0, 1)

    def test_conditioning_truncated_normal(self):
        x = Normal(0, 1)
        q = x.given(x > 0)
        s = np.asarray(q.sample(100000, seed=4))
        self.assertGreaterEqual(s.min(), 0.0)
        self.assertAlmostEqual(s.mean(), np.sqrt(2 / np.pi), delta=0.02)   # half-normal mean
        self.assertAlmostEqual(q.prob_of_event(), 0.5, delta=0.02)
        # renormalized density: truncated = base - log P(event)
        exact = (-0.5 * np.log(2 * np.pi) - 0.5) - np.log(0.5)
        self.assertAlmostEqual(q.log_prob(1.0), exact, delta=0.02)

    def test_new_families_recover(self):
        from pysp.ppl import StudentT, LogNormal, NegativeBinomial
        rng = np.random.RandomState(0)
        st = StudentT(free, free, free).fit(list(rng.standard_t(5, size=20000) * 2 + 1), max_its=60)
        self.assertAlmostEqual(st.dist.loc, 1.0, delta=0.15)
        self.assertAlmostEqual(st.dist.scale, 2.0, delta=0.2)
        ln = LogNormal(free, free).fit(list(rng.lognormal(0.5, 0.7, size=20000)), max_its=60)
        self.assertAlmostEqual(ln.dist.mu, 0.5, delta=0.05)
        # NB needs moment-matched init (provided by the family) to recover dispersion
        truth = NegativeBinomial(5.0, 0.4)
        data = list(np.asarray(truth.sample(20000, seed=1)).astype(float))
        nb = NegativeBinomial(free, free).fit(data, max_its=200)
        self.assertAlmostEqual(nb.dist.r, 5.0, delta=1.0)
        self.assertAlmostEqual(nb.dist.p, 0.4, delta=0.08)

    def test_predict_plugin_and_bayesian(self):
        from pysp.ppl import Poisson, Gamma
        rng = np.random.RandomState(0)
        # plug-in predictive from a point fit
        pe = Normal(free, free).fit(list(rng.normal(5, 2, 5000)))
        self.assertEqual(np.asarray(pe.predict(7, rng=np.random.RandomState(1))).shape, (7,))
        # Bayesian posterior predictive integrates parameter uncertainty
        pb = Poisson(Gamma(2, 1, name="rate")).fit(list(rng.poisson(3.5, 5000).astype(float)))
        pp = np.asarray(pb.predict(5000, rng=np.random.RandomState(2)))
        self.assertAlmostEqual(pp.mean(), 3.5, delta=0.2)

    def test_params_round_trip_parameterization(self):
        # fitted params come back in the SAME parameterization used to construct
        rng = np.random.RandomState(0)
        m = Normal(free, free).fit(list(rng.normal(5, 2, 5000)))
        p = m.params
        self.assertEqual(set(p), {"mean", "sd"})          # not mu/sigma2
        self.assertAlmostEqual(p["mean"], 5.0, delta=0.1)
        self.assertAlmostEqual(p["sd"], 2.0, delta=0.1)
        g = Gamma(free, free).fit(list(rng.gamma(2.0, 1 / 0.5, 5000)))
        self.assertEqual(set(g.params), {"shape", "rate"})

    def test_composite_params_are_leak_free(self):
        rng = np.random.RandomState(0)
        d = list(np.concatenate([rng.normal(-5, 1, 4000), rng.normal(5, 1, 4000)]))
        gm = Mix([Normal(free, free), Normal(free, free)]).fit(d, rng=np.random.RandomState(1))
        p = gm.params
        self.assertEqual(set(p), {"components", "weights"})
        self.assertEqual(set(p["components"][0]), {"mean", "sd"})        # recursed, PPL vocab
        # queryable sub-models via .components
        self.assertEqual(len(gm.components), 2)
        means = sorted(c.params["mean"] for c in gm.components)
        self.assertAlmostEqual(means[0], -5.0, delta=0.3)
        # Seq read has no double-.dist leak
        sq = Seq(Normal(free, free)).fit(
            [list(rng.normal(2, 1.5, rng.randint(5, 15))) for _ in range(1500)], max_its=40)
        self.assertEqual(set(sq.params), {"element"})
        self.assertEqual(set(sq.params["element"]), {"mean", "sd"})

    def test_model_comparison(self):
        rng = np.random.RandomState(0)
        data = list(np.concatenate([rng.normal(-4, 1, 3000), rng.normal(4, 1, 3000)]))  # bimodal
        m1 = Normal(free, free).fit(data)
        m2 = Mix([Normal(free, free), Normal(free, free)]).fit(data, rng=np.random.RandomState(1))
        # the mixture fits much better
        self.assertGreater(m2.log_likelihood(data), m1.log_likelihood(data))
        self.assertLess(m2.aic(data), m1.aic(data))
        ranked = compare([m1, m2], data, by="aic")
        self.assertEqual(ranked[0]["model"], "MixtureDistribution")   # best first

    def test_moments(self):
        self.assertAlmostEqual(Normal(3, 2).mean(), 3.0, delta=0.1)
        self.assertAlmostEqual(Normal(3, 2).var(), 4.0, delta=0.2)
        self.assertAlmostEqual(Normal(0, 1).exp().mean(), np.exp(0.5), delta=0.1)   # lognormal
        # moment of a convolution
        self.assertAlmostEqual((Normal(0, 1) + Normal(5, 2)).mean(), 5.0, delta=0.1)

    def test_immutability(self):
        x = Normal(0.0, 1.0)
        with self.assertRaises(AttributeError):
            x._name = "nope"


if __name__ == "__main__":
    unittest.main()
