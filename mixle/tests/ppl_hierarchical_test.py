"""Hierarchical priors -- a prior whose hyperparameter is itself an estimated random variable."""

import unittest
import warnings

import numpy as np

from mixle.ppl import Gamma, Normal
from mixle.ppl.autograd import torch_available

# These checks call the autodiff path directly (grad_target / value_and_grad); that needs a torch
# backend, which the no-optional-deps CI suite does not install. Skip there rather than fail.
_HAS_AUTODIFF = torch_available()


class HierarchicalPriorTest(unittest.TestCase):
    def _funnel(self):
        tau = Gamma(2.0, 1.0, name="tau")
        mu = Normal(0.0, tau, name="mu")  # mu | tau ~ N(0, tau): the prior's scale is random
        return Normal(mu, 1.0)

    @unittest.skipUnless(_HAS_AUTODIFF, "requires a torch autodiff backend")
    def test_autograd_gradient_matches_finite_difference(self):
        from mixle.ppl.autograd import grad_target

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
            fit = self._funnel().fit(data, how="nuts", draws=150, burn=150, chains=2, rng=np.random.RandomState(1))
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
            fit = model.fit(data, how="nuts", draws=100, burn=100, chains=2, rng=np.random.RandomState(3))
        s = fit.summary()
        for k in ("mu", "tau", "a"):
            self.assertTrue(np.isfinite(s[k]["mean"]))


class NonCenteredReparamTest(unittest.TestCase):
    def _funnel(self, noncentered):
        tau = Gamma(2.0, 1.0, name="tau")
        mu = Normal(0.0, tau, name="mu")
        return Normal(mu.noncentered() if noncentered else mu, 1.0)

    @unittest.skipUnless(_HAS_AUTODIFF, "requires a torch autodiff backend")
    def test_noncentered_gradient_matches_finite_difference(self):
        from mixle.ppl.autograd import grad_target

        data = np.random.RandomState(0).normal(2.0, 1.0, 60)
        ag = grad_target(self._funnel(True), data)
        u0 = np.array([0.2, -0.3])
        _, g = ag.value_and_grad(u0)
        gn = np.zeros(2)
        eps = 1e-6
        for i in range(2):
            up, um = u0.copy(), u0.copy()
            up[i] += eps
            um[i] -= eps
            gn[i] = (ag.log_target(up) - ag.log_target(um)) / (2 * eps)
        np.testing.assert_allclose(g, gn, atol=1e-4)

    def test_noncentered_matches_centered_posterior(self):
        data = np.random.RandomState(0).normal(2.0, 1.0, 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = self._funnel(False).fit(data, how="nuts", draws=200, burn=200, chains=2, rng=np.random.RandomState(1))
            nc = self._funnel(True).fit(data, how="nuts", draws=200, burn=200, chains=2, rng=np.random.RandomState(1))
        # same model, different geometry -> same posterior
        self.assertAlmostEqual(c.summary()["mu"]["mean"], nc.summary()["mu"]["mean"], delta=0.15)
        self.assertAlmostEqual(c.summary()["tau"]["mean"], nc.summary()["tau"]["mean"], delta=0.5)

    def test_noncentered_reduces_divergences_on_sharp_funnel(self):
        data = np.random.RandomState(4).normal(0.0, 1.0, 3)  # weak data -> sharp prior funnel

        def total_div(noncentered):
            d = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for seed in range(3):
                    fit = self._funnel(noncentered).fit(
                        data, how="nuts", draws=600, burn=300, chains=2, rng=np.random.RandomState(seed)
                    )
                    d += fit.summary().get("_num_divergences", 0)
            return d

        self.assertGreater(total_div(False), 4 * total_div(True) + 10)  # non-centered is far smoother


if __name__ == "__main__":
    unittest.main()
