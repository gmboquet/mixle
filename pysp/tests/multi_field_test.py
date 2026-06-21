"""FieldSystem: jointly fit several coupled latent fields (Phase 2 -- the coregionalization spine).

The exploration / paleo use case: many observations (geophysical surveys, geochemical proxies) jointly
constrain *several* coupled latent fields (ore grade + density, or temperature + salinity + pCO2).
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import RBF, CustomProxy, FieldSystem, GaussianField, GaussianProxy, fit_field


def _two_field_setup(seed=0):
    n = 30
    x = np.linspace(0.0, 1.0, n)
    rng = np.random.RandomState(seed)
    a_true, b_true = np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)
    fa = GaussianField(x, RBF(lengthscale=0.18, amplitude=1.5), name="A")
    fb = GaussianField(x, RBF(lengthscale=0.18, amplitude=1.5), name="B")
    sys = FieldSystem([fa, fb])
    ya, yb = a_true + rng.randn(n) * 0.1, b_true + rng.randn(n) * 0.1
    pa = GaussianProxy(ya, slope=1.0, intercept=0.0, scale=0.1).on("A")
    pb = GaussianProxy(yb, slope=1.0, intercept=0.0, scale=0.1).on("B")
    return sys, [pa, pb], a_true, b_true


class FieldSystemValidationTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
    def test_duplicate_field_names_raise(self):
        f = GaussianField(np.arange(5.0), RBF(), name="T")
        with self.assertRaises(ValueError):
            FieldSystem([f, GaussianField(np.arange(5.0), RBF(), name="T")])

    @unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
    def test_empty_system_raises(self):
        with self.assertRaises(ValueError):
            FieldSystem([])


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class MultiFieldTest(unittest.TestCase):
    def test_two_independent_fields_each_recover(self):
        sys, proxies, a_true, b_true = _two_field_setup()
        post = fit_field(sys, proxies, how="laplace")
        self.assertGreater(np.corrcoef(post.mean("A"), a_true)[0, 1], 0.95)
        self.assertGreater(np.corrcoef(post.mean("B"), b_true)[0, 1], 0.95)
        self.assertFalse(np.allclose(post.mean("A"), post.mean("B")))  # genuinely separate posteriors
        self.assertTrue(np.all(post.sd("A") > 0) and np.all(post.sd("B") > 0))

    def test_joint_sample_spans_both_fields(self):
        sys, proxies, _, _ = _two_field_setup()
        s = fit_field(sys, proxies, how="laplace").sample(2000, rng=1)
        self.assertEqual(set(["A", "B"]), set(s) & {"A", "B"})
        self.assertEqual(s["A"].shape, (2000, 30))
        self.assertEqual(s["B"].shape, (2000, 30))

    def test_coupling_proxy_propagates_cross_field_information(self):
        """A sensor that observes A - B couples the fields and lowers total reconstruction error."""
        sys, proxies, a_true, b_true = _two_field_setup()
        n = len(a_true)
        z = (a_true - b_true) + np.random.RandomState(1).randn(n) * 0.05

        def coupling(field_t, params, torch):
            pred = params["A"] - params["B"]
            return -0.5 * torch.sum(((torch.as_tensor(z) - pred) / 0.05) ** 2)

        base = fit_field(sys, proxies, how="laplace")
        coupled = fit_field(sys, [*proxies, CustomProxy(coupling)], how="laplace")
        err = lambda p: np.sqrt(np.mean((p.mean("A") - a_true) ** 2)) + np.sqrt(np.mean((p.mean("B") - b_true) ** 2))
        self.assertLess(err(coupled), err(base))

    def test_gauss_newton_multifield_matches_laplace(self):
        sys, proxies, _, _ = _two_field_setup()
        lap = fit_field(sys, proxies, how="laplace")
        gn = fit_field(sys, proxies, how="gauss_newton")  # both exact for the linear-Gaussian forward
        np.testing.assert_allclose(gn.mean("A"), lap.mean("A"), atol=1e-4)
        np.testing.assert_allclose(gn.mean("B"), lap.mean("B"), atol=1e-4)

    def test_coregionalization_pulls_an_unobserved_field_through_the_prior(self):
        """With data ONLY on A, a strong a-priori cross-correlation should inform B; block-diag leaves it flat."""
        sys_indep, proxies, a_true, _ = _two_field_setup()
        pa = proxies[0]  # the proxy attached to A; B has no data
        fa, fb = sys_indep.fields
        indep = fit_field(FieldSystem([fa, fb]), [pa], how="laplace")
        coreg = fit_field(FieldSystem([fa, fb], coregion=np.array([[1.0, 0.9], [0.9, 1.0]])), [pa], how="laplace")
        self.assertLess(np.max(np.abs(indep.mean("B"))), 1e-3)  # block-diagonal: B uninformed, flat at 0
        self.assertGreater(np.corrcoef(coreg.mean("B"), a_true)[0, 1], 0.8)  # coregion: B tracks A

    def test_negative_coregion_flips_the_induced_field(self):
        sys_indep, proxies, a_true, _ = _two_field_setup()
        fa, fb = sys_indep.fields
        coreg = fit_field(FieldSystem([fa, fb], coregion=np.array([[1.0, -0.9], [-0.9, 1.0]])), [proxies[0]], how="laplace")
        self.assertLess(np.corrcoef(coreg.mean("B"), a_true)[0, 1], -0.8)

    def test_icm_recovers_the_exact_conditional_scaling(self):
        """At convergence the data-less field equals (B01/B00)*A pointwise -- the intrinsic-coregion prior."""
        sys_indep, proxies, _, _ = _two_field_setup()
        fa, fb = sys_indep.fields
        coreg = fit_field(
            FieldSystem([fa, fb], coregion=np.array([[1.0, 0.7], [0.7, 1.0]])), [proxies[0]], how="laplace", max_iter=3000
        )
        np.testing.assert_allclose(np.median(coreg.mean("B") / coreg.mean("A")), 0.7, atol=0.02)

    def test_coregion_validation(self):
        f = GaussianField(np.arange(6.0), RBF(), name="A")
        g = GaussianField(np.arange(6.0), RBF(), name="B")
        with self.assertRaises(ValueError):  # not positive-definite
            FieldSystem([f, g], coregion=np.array([[1.0, 1.2], [1.2, 1.0]]))
        with self.assertRaises(ValueError):  # wrong shape for 2 fields
            FieldSystem([f, g], coregion=np.eye(3))
        with self.assertRaises(ValueError):  # fields must share a dim/index
            FieldSystem([f, GaussianField(np.arange(5.0), RBF(), name="B")], coregion=np.eye(2))

    def test_single_field_system_matches_plain_field(self):
        n = 20
        x = np.linspace(0, 1, n)
        rng = np.random.RandomState(2)
        y = np.sin(3 * x) + rng.randn(n) * 0.1
        f = GaussianField(x, RBF(lengthscale=0.2, amplitude=1.0), name="T")
        px = lambda: GaussianProxy(y, slope=1.0, intercept=0.0, scale=0.1)
        plain = fit_field(f, [px()], how="laplace")
        viasys = fit_field(FieldSystem([f]), [px().on("T")], how="laplace")
        np.testing.assert_allclose(plain.mean("T"), viasys.mean("T"), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
