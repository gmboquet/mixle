"""Tests for the PyTorch estimation engine (pysp.stats.compute.torch_mixture).

Parity is checked against the legacy seq_* path on CPU float64; the gradient
MLE path is checked for likelihood improvement and parameter recovery.
"""

import importlib
import io
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
else:
    torch = None

from pysp.inference import seq_estimate
from pysp.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
    seq_encode,
)

if HAS_TORCH:
    from pysp.stats.compute.torch_mixture import TorchMixture
else:
    TorchMixture = None
from pysp.tests.kernels_test import make_estimator, make_mixture


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TorchEngineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_mixture()
        cls.data = cls.model.sampler(seed=1).sample(size=400)
        cls.tm = TorchMixture(cls.model)
        cls.enc = cls.tm.encode(cls.data)
        cls.legacy_enc = cls.model.dist_to_encoder().seq_encode(cls.data)

    # -- scoring parity ------------------------------------------------------

    def test_component_log_density_parity(self):
        ll_k = self.tm.seq_component_log_density(self.enc)
        for k, comp in enumerate(self.model.components):
            ll_legacy = comp.seq_log_density(comp.dist_to_encoder().seq_encode(self.data))
            self.assertTrue(
                np.allclose(ll_k[:, k], ll_legacy, atol=1.0e-10),
                "component %d max err %g" % (k, np.abs(ll_k[:, k] - ll_legacy).max()),
            )

    def test_mixture_log_density_parity(self):
        ll = self.tm.seq_log_density(self.enc)
        ll_legacy = self.model.seq_log_density(self.legacy_enc)
        self.assertTrue(np.allclose(ll, ll_legacy, atol=1.0e-10))

    def test_posterior_parity(self):
        gam = self.tm.posteriors(self.enc).cpu().numpy()
        gam_legacy = self.model.seq_posterior(self.legacy_enc)
        self.assertTrue(np.allclose(gam, gam_legacy, atol=1.0e-10))

    def test_single_distribution_parity(self):
        comp = self.model.components[0]
        tm = TorchMixture(comp)
        ll = tm.seq_log_density(tm.encode(self.data))
        ll_legacy = comp.seq_log_density(comp.dist_to_encoder().seq_encode(self.data))
        self.assertTrue(np.allclose(ll, ll_legacy, atol=1.0e-10))

    # -- EM parity -------------------------------------------------------------

    def test_em_trajectory_matches_legacy(self):
        est = make_estimator()
        chunked = seq_encode(self.data, model=self.model)

        m_torch = self.model
        m_legacy = self.model
        for _ in range(3):
            m_torch = self.tm.em_step(self.enc, est, model=m_torch)
            m_legacy = seq_estimate(chunked, est, m_legacy)

            ll_t = self.tm.seq_log_density(self.enc, model=m_torch)
            ll_l = m_legacy.seq_log_density(self.legacy_enc)
            self.assertTrue(np.allclose(ll_t, ll_l, atol=1.0e-8), "EM diverged: max err %g" % np.abs(ll_t - ll_l).max())
            self.assertTrue(
                np.allclose(np.asarray(m_torch.w, dtype=float), np.asarray(m_legacy.w, dtype=float), atol=1.0e-10)
            )

    def test_fit_converges(self):
        est = make_estimator()
        model, ll = self.tm.fit(self.enc, est, max_its=60, delta=1.0e-7, rng=np.random.RandomState(5), init_p=1.0)
        ll0 = self.tm.seq_log_density(self.enc, model=self.model).sum()
        self.assertTrue(np.isfinite(ll))
        self.assertGreater(ll, ll0 - 0.05 * abs(ll0))

    # -- gradient MLE ------------------------------------------------------------

    def test_fit_mle_improves_likelihood(self):
        # perturbed start: MLE must climb back to (at least near) the truth
        comps = []
        for c in self.model.components:
            g = c.dists[0]
            comps.append(
                CompositeDistribution((GaussianDistribution(g.mu + 2.0, g.sigma2 * 3.0),) + tuple(c.dists[1:]))
            )
        start = MixtureDistribution(comps, [1.0 / 3] * 3)

        tm = TorchMixture(start)
        enc = tm.encode(self.data)
        ll_start = tm.seq_log_density(enc).sum()
        fitted, ll_fit = tm.fit_mle(enc, max_its=400, lr=0.05, out=io.StringIO())
        ll_truth = tm.seq_log_density(enc, model=self.model).sum()

        self.assertGreater(ll_fit, ll_start)
        self.assertGreater(ll_fit, ll_truth - 0.01 * abs(ll_truth))
        self.assertIsInstance(fitted, MixtureDistribution)
        # fitted model must round-trip through the legacy scorer
        ll_check = fitted.seq_log_density(fitted.dist_to_encoder().seq_encode(self.data))
        self.assertTrue(np.all(np.isfinite(ll_check)))

    def test_fit_mle_recovers_gaussian_parameters(self):
        truth = MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(-5.0, 1.0),)),
                CompositeDistribution((GaussianDistribution(5.0, 2.0),)),
            ],
            [0.3, 0.7],
        )
        data = truth.sampler(seed=7).sample(size=3000)
        start = MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(-1.0, 4.0),)),
                CompositeDistribution((GaussianDistribution(1.0, 4.0),)),
            ],
            [0.5, 0.5],
        )
        tm = TorchMixture(start)
        enc = tm.encode(data)
        fitted, _ = tm.fit_mle(enc, max_its=800, lr=0.05, out=io.StringIO())

        mus = sorted(c.dists[0].mu for c in fitted.components)
        ws = sorted(fitted.w)
        self.assertLess(abs(mus[0] - -5.0), 0.2)
        self.assertLess(abs(mus[1] - 5.0), 0.2)
        self.assertLess(abs(ws[0] - 0.3), 0.05)

    # -- gradient MAP --------------------------------------------------------------

    def test_fit_map_zero_strength_equals_mle(self):
        truth = MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(-4.0, 1.0),)),
                CompositeDistribution((GaussianDistribution(4.0, 1.0),)),
            ],
            [0.5, 0.5],
        )
        data = truth.sampler(seed=3).sample(size=500)
        start = MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(-1.0, 4.0),)),
                CompositeDistribution((GaussianDistribution(1.0, 4.0),)),
            ],
            [0.5, 0.5],
        )
        tm = TorchMixture(start)
        enc = tm.encode(data)
        m_mle, ll_mle = tm.fit_mle(enc, max_its=400, lr=0.05, out=io.StringIO())
        m_map, lp_map = tm.fit_map(enc, prior_strength=0.0, max_its=400, lr=0.05, out=io.StringIO())
        self.assertLess(abs(ll_mle - lp_map), 1.0e-4 * abs(ll_mle))

    def test_fit_map_regularizes_variance(self):
        # tiny dataset with near-identical points: MLE collapses sigma^2 toward
        # the (tiny) sample variance; an informative NormalGamma prior holds it up
        data = [0.0, 0.01, -0.01, 0.005, -0.005, 0.0, 0.01, -0.01]
        start = GaussianDistribution(0.0, 1.0)
        tm = TorchMixture(start)
        enc = tm.encode(data)
        m_mle, _ = tm.fit_mle(enc, max_its=2000, lr=0.05, out=io.StringIO())
        prior = {"family": "normalgamma", "mu0": 0.0, "kappa": 1.0e-3, "a": 3.0, "b": 2.0}
        m_map, _ = tm.fit_map(enc, priors=prior, max_its=2000, lr=0.05, out=io.StringIO())

        self.assertLess(m_mle.sigma2, 0.01)
        self.assertGreater(m_map.sigma2, 10.0 * m_mle.sigma2)
        # MAP mode in sigma^2 coordinates: maximizing
        # (a - 1 + 0.5 + n/2) log tau - (b + SS/2) tau over sigma^2 = 1/tau gives
        # sigma^2* = (b + SS/2) / (a + n/2 - 0.5) at kappa ~ 0
        n = len(data)
        ss = sum(x * x for x in data)
        expected = (2.0 + 0.5 * ss) / (3.0 + n / 2.0 - 0.5)
        self.assertLess(abs(m_map.sigma2 - expected) / expected, 0.05)

    def test_fit_map_smooths_unseen_categories(self):
        # vocabulary contains 'c' but the data never does: MAP with a Dirichlet
        # prior keeps visibly more mass on 'c' than MLE
        comp = CompositeDistribution((CategoricalDistribution({"a": 0.6, "b": 0.3, "c": 0.1}),))
        data = [("a",)] * 60 + [("b",)] * 40
        tm = TorchMixture(comp)
        enc = tm.encode(data)
        m_mle, _ = tm.fit_mle(enc, max_its=1500, lr=0.1, out=io.StringIO())
        m_map, _ = tm.fit_map(enc, prior_strength=30.0, max_its=1500, lr=0.1, out=io.StringIO())
        p_mle = m_mle.dists[0].pmap["c"]
        p_map = m_map.dists[0].pmap["c"]
        self.assertGreater(p_map, 5.0 * max(p_mle, 1.0e-12))
        self.assertGreater(p_map, 0.01)

    def test_fit_map_full_model_runs(self):
        # default priors must compose over the deep model (composite + sequence)
        tm = TorchMixture(self.model)
        enc = tm.encode(self.data)
        fitted, lp = tm.fit_map(enc, prior_strength=1.0, max_its=150, lr=0.05, out=io.StringIO())
        self.assertTrue(np.isfinite(lp))
        ll = fitted.seq_log_density(fitted.dist_to_encoder().seq_encode(self.data))
        self.assertTrue(np.all(np.isfinite(ll)))

    # -- devices ------------------------------------------------------------------

    def test_float32_parity_loose(self):
        tm32 = TorchMixture(self.model, dtype=torch.float32)
        ll = tm32.seq_log_density(tm32.encode(self.data))
        ll_legacy = self.model.seq_log_density(self.legacy_enc)
        rel = np.abs(ll - ll_legacy) / np.maximum(np.abs(ll_legacy), 1.0)
        self.assertLess(rel.max(), 1.0e-4)

    @unittest.skipUnless(HAS_TORCH and torch.backends.mps.is_available(), "MPS not available")
    def test_mps_smoke(self):
        tm = TorchMixture(self.model, device="mps", dtype=torch.float32)
        enc = tm.encode(self.data)
        ll = tm.seq_log_density(enc)
        ll_legacy = self.model.seq_log_density(self.legacy_enc)
        rel = np.abs(ll - ll_legacy) / np.maximum(np.abs(ll_legacy), 1.0)
        self.assertLess(rel.max(), 1.0e-3)
        est = make_estimator()
        m = tm.em_step(enc, est)
        self.assertTrue(np.all(np.isfinite(tm.seq_log_density(enc, model=m))))


if __name__ == "__main__":
    unittest.main()
