"""Tests for pysp.ppl latent fields (shared GP/GMRF field + many proxies, joint fit, Laplace posterior)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pysp.ppl import (
    GP,
    RBF,
    Cox,
    CustomProxy,
    Gaussian,
    GaussianField,
    GaussianProxy,
    LogisticNicheProxy,
    Niche,
    PoissonProxy,
    RandomWalk,
    fit_field,
    free,
    joint,
)


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class GaussianFieldExactTestCase(unittest.TestCase):
    """With every factor Gaussian, the Laplace posterior is the exact posterior."""

    def test_matches_closed_form(self):
        rng = np.random.RandomState(0)
        n = 30
        field = GaussianField(np.arange(n), RandomWalk(scale=0.4, ridge=3.0), name="T")
        lam = field.precision
        t_true = np.linalg.cholesky(np.linalg.inv(lam)) @ rng.randn(n)
        a, b, sig = 0.5, -1.3, 0.2
        y = a + b * t_true + sig * rng.randn(n)

        prec = lam.copy()
        np.fill_diagonal(prec, np.diag(prec) + (b / sig) ** 2)
        cov = np.linalg.inv(prec)
        mean_cf = cov @ (b / sig**2 * (y - a))
        sd_cf = np.sqrt(np.diag(cov))

        post = fit_field(field, [GaussianProxy(y, slope=b, intercept=a, scale=sig)], how="laplace")
        mean, sd = post.posterior("T")
        self.assertLess(np.max(np.abs(mean - mean_cf)), 1e-4)
        self.assertLess(np.max(np.abs(sd - sd_cf)), 1e-8)

    def test_coverage_when_well_specified(self):
        n = 40
        field = GaussianField(np.arange(n), RandomWalk(scale=0.5, ridge=3.0), name="T")
        kchol = np.linalg.cholesky(np.linalg.inv(field.precision))
        cov_rate = []
        for s in range(25):
            rng = np.random.RandomState(s)
            t_true = kchol @ rng.randn(n)
            y = 2.0 * t_true + 0.4 * rng.randn(n)
            post = fit_field(field, [GaussianProxy(y, slope=2.0, intercept=0.0, scale=0.4)], how="laplace")
            t, sd = post.posterior("T")
            cov_rate.append(np.mean(np.abs((t - t_true) / sd) < 1.96))
        self.assertGreater(np.mean(cov_rate), 0.90)  # nominal 0.95; Laplace is exact here


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class AdditiveInformationTestCase(unittest.TestCase):
    """Two proxies on one field: the joint posterior is tighter than either proxy alone."""

    def _data(self):
        rng = np.random.RandomState(1)
        n = 50
        t = np.linspace(0, 1, n)
        t_true = 1.5 * np.cos(2.4 * t) + 0.4 * np.sin(7 * t)
        c1, sig = 1.2, 0.5
        d18 = 3.0 - c1 * t_true + sig * rng.randn(n)
        S = 60
        mu = rng.uniform(t_true.min(), t_true.max(), S)
        kap = np.exp(rng.uniform(-0.5, 1.5, S))
        logit = 1.0 - 0.5 * kap[:, None] * (t_true[None, :] - mu[:, None]) ** 2
        pres = (rng.rand(S, n) < 1 / (1 + np.exp(-logit))).astype(float)
        return n, t_true, d18, pres, c1, sig

    def test_joint_sharpens_each_proxy(self):
        n, t_true, d18, pres, c1, sig = self._data()
        field = GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=3.0), name="T")
        post = fit_field(
            field,
            [GaussianProxy(d18, slope=-c1, intercept=free, scale=sig), LogisticNicheProxy(pres)],
            how="laplace",
            max_iter=400,
        )
        _, sd_iso = post.field_posterior(include=["gauss"])
        _, sd_fos = post.field_posterior(include=["niche"])
        t_map, sd_joint = post.field_posterior()
        # joint information is the sum, so the joint posterior is no wider than either subset, anywhere
        self.assertTrue(np.all(sd_joint <= sd_iso + 1e-9))
        self.assertTrue(np.all(sd_joint <= sd_fos + 1e-9))
        self.assertLess(np.median(sd_joint), np.median(sd_iso))
        # the field shape is recovered (sign anchored by the fixed slope)
        self.assertGreater(np.corrcoef(t_map, t_true)[0, 1], 0.95)

    def test_posterior_over_any_node(self):
        n, t_true, d18, pres, c1, sig = self._data()
        field = GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=3.0), name="T")
        post = fit_field(
            field,
            [GaussianProxy(d18, slope=free, intercept=free, scale=sig), LogisticNicheProxy(pres)],
            how="laplace",
            max_iter=400,
        )
        summ = post.summary()
        self.assertIn("T", summ)
        self.assertIn("gauss.slope", summ)
        self.assertIn("niche.mu", summ)
        # every node has a finite mean and positive sd
        for node in ("T", "gauss.slope", "niche.b"):
            m, s = post.posterior(node)
            self.assertTrue(np.all(np.isfinite(m)))
            self.assertTrue(np.all(s > 0))


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class CoxProcessTestCase(unittest.TestCase):
    """A log-Gaussian Cox process: latent log-intensity field -> Poisson counts."""

    def test_recovers_intensity(self):
        rng = np.random.RandomState(2)
        n = 60
        field = GaussianField(np.arange(n), RandomWalk(scale=0.4, ridge=3.0), name="logmu")
        kchol = np.linalg.cholesky(np.linalg.inv(field.precision))
        log_intensity = kchol @ rng.randn(n)
        offset = 2.0  # baseline log-rate so counts are non-trivial
        counts = rng.poisson(np.exp(offset + log_intensity))
        post = fit_field(field, [PoissonProxy(counts, offset=offset)], how="laplace", max_iter=400)
        f_map, sd = post.posterior("logmu")
        self.assertGreater(np.corrcoef(f_map, log_intensity)[0, 1], 0.8)
        self.assertTrue(np.all(sd > 0))


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class KernelAndCustomTestCase(unittest.TestCase):
    def test_rbf_kernel_precision_is_spd(self):
        field = GaussianField(np.linspace(0, 1, 20), RBF(lengthscale=0.2, amplitude=1.0), name="g")
        eig = np.linalg.eigvalsh(field.precision)
        self.assertGreater(eig.min(), 0.0)

    def test_rbf_spatial_2d(self):
        coords = np.random.RandomState(0).rand(15, 2)
        field = GaussianField(coords, RBF(lengthscale=0.3), name="s")
        self.assertEqual(field.precision.shape, (15, 15))

    def test_custom_proxy(self):
        rng = np.random.RandomState(3)
        n = 25
        field = GaussianField(np.arange(n), RandomWalk(scale=0.5, ridge=3.0), name="T")
        t_true = np.linalg.cholesky(np.linalg.inv(field.precision)) @ rng.randn(n)
        y = t_true + 0.3 * rng.randn(n)

        def gaussian_ll(field_t, params, torch):
            resid = (torch.as_tensor(y) - field_t) / 0.3
            return -0.5 * torch.sum(resid * resid)

        post = fit_field(field, [CustomProxy(gaussian_ll, prefix="obs")], how="laplace", max_iter=300)
        t_map, sd = post.posterior("T")
        self.assertGreater(np.corrcoef(t_map, t_true)[0, 1], 0.9)


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class PPLNativeSurfaceTestCase(unittest.TestCase):
    """The equation-style GP/joint surface delegates to fit_field and gives identical results."""

    def test_joint_matches_builder(self):
        rng = np.random.RandomState(1)
        n = 50
        t = np.linspace(0, 1, n)
        t_true = 1.5 * np.cos(2.4 * t) + 0.4 * np.sin(7 * t)
        c0, c1, sig = 3.0, 1.2, 0.5
        d18 = c0 - c1 * t_true + sig * rng.randn(n)
        S = 50
        mu = rng.uniform(t_true.min(), t_true.max(), S)
        kap = np.exp(rng.uniform(-0.5, 1.5, S))
        logit = 1.0 - 0.5 * kap[:, None] * (t_true[None, :] - mu[:, None]) ** 2
        pres = (rng.rand(S, n) < 1 / (1 + np.exp(-logit))).astype(float)

        T = GP("T", index=np.arange(n), kernel=RandomWalk(scale=0.3, ridge=3.0))
        post = joint([Gaussian(d18, mean=c0 - c1 * T, sd=sig), Niche(pres, over=T)], how="laplace", max_iter=400)
        _, sd_native = post.field_posterior()

        field = GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=3.0), name="T")
        ref = fit_field(
            field,
            [GaussianProxy(d18, slope=-c1, intercept=c0, scale=sig), LogisticNicheProxy(pres)],
            how="laplace",
            max_iter=400,
        )
        _, sd_ref = ref.field_posterior()
        self.assertTrue(np.allclose(sd_native, sd_ref, atol=1e-8))

    def test_cox_surface(self):
        rng = np.random.RandomState(2)
        n = 50
        field_gp = GP("logmu", index=np.arange(n), kernel=RandomWalk(scale=0.4, ridge=3.0))
        kchol = np.linalg.cholesky(np.linalg.inv(field_gp.field.precision))
        log_intensity = kchol @ rng.randn(n)
        counts = rng.poisson(np.exp(2.0 + log_intensity))
        post = joint([Cox(counts, log_intensity=field_gp, offset=2.0)], how="laplace", max_iter=400)
        f_map, sd = post.posterior("logmu")
        self.assertGreater(np.corrcoef(f_map, log_intensity)[0, 1], 0.8)

    def test_single_field_required(self):
        A = GP("A", index=np.arange(10), kernel=RandomWalk(scale=0.5, ridge=3.0))
        B = GP("B", index=np.arange(10), kernel=RandomWalk(scale=0.5, ridge=3.0))
        y = np.zeros(10)
        with self.assertRaises(ValueError):
            joint([Gaussian(y, mean=A, sd=1.0), Gaussian(y, mean=B, sd=1.0)])


if __name__ == "__main__":
    unittest.main()
