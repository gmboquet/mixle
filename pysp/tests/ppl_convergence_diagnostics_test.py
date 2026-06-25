"""Rank-normalized convergence diagnostics + NUTS divergence tracking (Vehtari et al. 2021)."""

import unittest
import warnings

import numpy as np

from pysp.ppl.diagnostics import bulk_ess, split_rhat, tail_ess


class RankNormalizedDiagnosticsTest(unittest.TestCase):
    def test_iid_draws(self):
        x = np.random.RandomState(0).standard_normal((4, 1000))
        self.assertAlmostEqual(split_rhat(x), 1.0, delta=0.02)
        self.assertGreater(bulk_ess(x), 3000)  # near the 4000 independent draws
        self.assertGreater(tail_ess(x), 2500)

    def test_autocorrelation_lowers_ess(self):
        rng = np.random.RandomState(0)
        ar = np.zeros((4, 1000))
        for c in range(4):
            for t in range(1, 1000):
                ar[c, t] = 0.9 * ar[c, t - 1] + rng.standard_normal()
        self.assertLess(bulk_ess(ar), 600)  # AR(0.9): theoretical ESS ~ n*(1-phi)/(1+phi) ~ 210

    def test_nonconverged_chains_flag_high_rhat(self):
        rng = np.random.RandomState(0)
        bad = rng.standard_normal((4, 1000)) + np.array([[-5], [0], [5], [10]])
        self.assertGreater(split_rhat(bad), 1.5)  # chains in different places


class NutsDivergenceTest(unittest.TestCase):
    def test_funnel_produces_divergences(self):
        from pysp.inference.mcmc import nuts

        def logp(z):
            v, x = z
            return -0.5 * (v / 3.0) ** 2 - 0.5 * (x * x) * np.exp(-v) - 0.5 * v

        def grad(z):
            v, x = z
            return np.array([-(v / 9.0) + 0.5 * (x * x) * np.exp(-v) - 0.5, -x * np.exp(-v)])

        res = nuts(logp, grad, np.array([0.0, 0.0]), num_samples=500, warmup=500, rng=np.random.RandomState(1))
        self.assertEqual(res.divergences.shape, (500,))
        self.assertGreater(int(res.divergences.sum()), 0)  # the funnel neck forces divergences

    def test_well_conditioned_target_has_few_divergences(self):
        from pysp.inference.mcmc import nuts

        res = nuts(
            lambda z: -0.5 * float(z @ z),
            lambda z: -z,
            np.zeros(3),
            num_samples=400,
            warmup=400,
            rng=np.random.RandomState(0),
        )
        self.assertLess(int(res.divergences.sum()), 20)  # a Gaussian rarely diverges


class SummaryExposesDiagnosticsTest(unittest.TestCase):
    def test_nuts_summary_carries_new_diagnostics(self):
        from pysp.ppl import Normal

        rng = np.random.RandomState(0)
        data = rng.normal(2.0, 1.5, 300)
        model = Normal(Normal(0, 10, name="mu"), Normal(0, 10, name="sig"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(data, how="nuts", draws=400, burn=400, chains=4, rng=np.random.RandomState(1))
        s = fit.summary()
        for key in ("_split_rhat", "_bulk_ess", "_tail_ess"):
            self.assertIn(key, s)
            self.assertIn("mu", s[key])
        self.assertLess(s["_split_rhat"]["mu"], 1.05)
        self.assertGreater(s["_bulk_ess"]["mu"], 100)


if __name__ == "__main__":
    unittest.main()
