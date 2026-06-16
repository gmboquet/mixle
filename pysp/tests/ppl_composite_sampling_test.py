"""pysp.ppl: parameter MCMC / HMC / ensemble for *composite* models (mixtures, ...).

Composites lower through the generic numerical target (collect leaf free/prior params across
the tree, rebuild a concrete model per evaluation), so mcmc / ensemble / map all work on them.
Mixtures need an identifiability constraint (ordered component means) to break label-switching
— the standard requirement, and exactly what the constraint surface provides.
"""

import unittest

import numpy as np

from pysp.ppl import Gamma, Mix, Normal


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
