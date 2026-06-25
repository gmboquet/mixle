"""Hierarchical priors -- a prior whose hyperparameter is itself an estimated random variable."""

import unittest
import warnings

import numpy as np

from pysp.ppl import Gamma, Normal


class HierarchicalPriorTest(unittest.TestCase):
    def _funnel(self):
        tau = Gamma(2.0, 1.0, name="tau")
        mu = Normal(0.0, tau, name="mu")  # mu | tau ~ N(0, tau): the prior's scale is random
        return Normal(mu, 1.0)

    def test_autograd_gradient_matches_finite_difference(self):
        from pysp.ppl.autograd import grad_target

        rng = np.random.RandomState(0)
        data = rng.normal(2.0, 1.0, 200)
        ag = grad_target(self._funnel(), data)
        self.assertIsNotNone(ag)  # the hierarchical target is differentiable
        u0 = np.array([0.3, -0.2])
        _, g = ag.value_and_grad(u0)
        gnum = np.zeros(2)
        eps = 1e-6
        for i in range(2):
            up, um = u0.copy(), u0.copy()
            up[i] += eps
            um[i] -= eps
            gnum[i] = (ag.log_target(up) - ag.log_target(um)) / (2 * eps)
        np.testing.assert_allclose(g, gnum, atol=1e-4)

    def test_nuts_recovers_hierarchical_posterior(self):
        rng = np.random.RandomState(0)
        data = rng.normal(2.0, 1.0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = self._funnel().fit(data, how="nuts", draws=400, burn=400, chains=2, rng=np.random.RandomState(1))
        s = fit.summary()
        self.assertAlmostEqual(s["mu"]["mean"], 2.0, delta=0.3)  # mu tracks the data mean
        self.assertGreater(s["tau"]["mean"], 0.0)  # tau is a positive scale
        self.assertLess(s["_split_rhat"]["mu"], 1.1)

    def test_numeric_path_handles_hierarchy(self):
        rng = np.random.RandomState(0)
        data = rng.normal(2.0, 1.0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = self._funnel().fit(data, how="ensemble", draws=400, burn=300, rng=np.random.RandomState(1))
        s = fit.summary()
        self.assertAlmostEqual(s["mu"]["mean"], 2.0, delta=0.4)

    def test_three_level_nesting(self):
        rng = np.random.RandomState(0)
        data = rng.normal(1.5, 1.0, 150)
        a = Gamma(2.0, 1.0, name="a")
        tau = Gamma(a, 1.0, name="tau")  # tau's shape is itself random -> 3 levels deep
        model = Normal(Normal(0.0, tau, name="mu"), 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(data, how="nuts", draws=300, burn=400, chains=2, rng=np.random.RandomState(3))
        s = fit.summary()
        for k in ("mu", "tau", "a"):
            self.assertTrue(np.isfinite(s[k]["mean"]))


if __name__ == "__main__":
    unittest.main()
