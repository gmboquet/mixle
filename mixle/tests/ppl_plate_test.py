"""Plate / random-intercept models sampled by NUTS (per-group latents + shared hyperparameters)."""

import unittest
import warnings

import numpy as np

from mixle.ppl import Gamma, Normal
from mixle.ppl.autograd import torch_available

# The grad-check and NUTS targets here require a torch autodiff backend (value_and_grad), which the
# no-optional-deps CI suite does not install. Skip there instead of failing.
_HAS_AUTODIFF = torch_available()


def _hier_data(seed=0):
    rng = np.random.RandomState(seed)
    thetas = rng.normal(5.0, 3.0, 8)  # mu=5, tau=3
    data = [rng.normal(t, 2.0, rng.randint(8, 15)) for t in thetas]  # sigma=2
    return data, thetas


def _model(noncentered, free_mu=False):
    loc = Normal(0, 10, name="mu") if free_mu else 5.0
    inner = Normal(loc, Gamma(2.0, 1.0, name="tau"), name="theta").each()
    if noncentered:
        inner = inner.noncentered()
    return Normal(inner, Gamma(2.0, 1.0, name="sigma"))


class GroupedTargetTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_AUTODIFF, "requires a torch autodiff backend")
    def test_gradient_matches_finite_difference(self):
        from mixle.ppl.inference import _grouped_target

        data, _ = _hier_data()
        log_target, grad, slots, _b, _dm, _ds = _grouped_target(_model(True), data, want_grad=True)
        self.assertEqual(len(slots), 10)  # tau, sigma, 8 thetas
        u0 = np.array([0.1, 0.2] + [0.3] * 8)
        g = grad(u0)
        gn = np.zeros(len(u0))
        eps = 1e-6
        for i in range(len(u0)):
            up, um = u0.copy(), u0.copy()
            up[i] += eps
            um[i] -= eps
            gn[i] = (log_target(up) - log_target(um)) / (2 * eps)
        np.testing.assert_allclose(g, gn, atol=1e-4)


class GroupedNutsTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_AUTODIFF, "requires a torch autodiff backend")
    def test_recovers_hyperparameters_and_group_latents(self):
        data, thetas = _hier_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _model(True).fit(data, how="nuts", draws=600, burn=600, chains=2, rng=np.random.RandomState(1))
        s = fit.summary()
        self.assertAlmostEqual(s["tau"]["mean"], 3.0, delta=2.0)
        self.assertAlmostEqual(s["sigma"]["mean"], 2.0, delta=0.8)
        self.assertAlmostEqual(s["theta[0]"]["mean"], thetas[0], delta=1.5)  # per-group latent recovered
        self.assertLess(max(s["_split_rhat"].values()), 1.15)

    @unittest.skipUnless(_HAS_AUTODIFF, "requires a torch autodiff backend")
    def test_free_population_mean(self):
        data, _ = _hier_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Only finiteness is asserted below (no recovery precision needed), so a short
            # chain suffices -- verified stable across 10 seeds at these settings.
            fit = _model(True, free_mu=True).fit(
                data, how="nuts", draws=100, burn=150, chains=2, rng=np.random.RandomState(2)
            )
        self.assertTrue(np.isfinite(fit.summary()["mu"]["mean"]))

    def test_numeric_ensemble_path(self):
        data, _ = _hier_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _model(True).fit(data, how="ensemble", draws=400, burn=300, rng=np.random.RandomState(2))
        self.assertAlmostEqual(fit.summary()["sigma"]["mean"], 2.0, delta=1.0)

    @unittest.skipUnless(_HAS_AUTODIFF, "requires a torch autodiff backend")
    def test_noncentered_reduces_divergences(self):
        data, _ = _hier_data(seed=3)

        def total_div(noncentered):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Both parameterizations already settle at 0 divergences on this data at the
                # original draws/burn; verified unchanged (0 vs 0, seeds 0-9) at this smaller
                # budget, so the comparison is preserved while the chains run ~3x faster.
                fit = _model(noncentered).fit(
                    data, how="nuts", draws=150, burn=200, chains=2, rng=np.random.RandomState(0)
                )
            return fit.summary().get("_num_divergences", 0)

        self.assertLessEqual(total_div(True), total_div(False) + 5)  # non-centered no worse, usually better

    def test_auto_still_uses_conjugate_em(self):
        data, _ = _hier_data()
        fit = Normal(Normal(5.0, 3.0, name="theta").each(), 2.0).fit(data, how="auto")
        self.assertTrue(fit.is_bound)  # the existing conjugate-EM hierarchical path is unchanged


if __name__ == "__main__":
    unittest.main()
