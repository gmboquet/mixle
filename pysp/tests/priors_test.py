"""Tests for edge-preserving / discrete priors, complex-valued data, and multistart (phase 4)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import (
        GP,
        Differential,
        Gaussian,
        Potts,
        RandomWalk,
        TotalVariation,
        free,
        joint,
        multistart,
    )


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TotalVariationTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_preserves_edges_vs_smoothing(self):
        n = 60
        x = np.linspace(0, 1, n)
        f_true = (x > 0.5).astype(float) * 2.0
        y = f_true + 0.2 * np.random.RandomState(0).randn(n)
        jump = lambda f: np.max(np.abs(np.diff(f)))
        rmse = lambda f: np.sqrt(np.mean((f - f_true) ** 2))

        sm_fld = GP("f", index=np.arange(n), kernel=RandomWalk(scale=0.08, ridge=10.0))
        sm = joint([Gaussian(y, mean=1.0 * sm_fld, sd=0.2)]).fit(how="map").mean("f")

        tv_fld = GP("f", index=np.arange(n), kernel=RandomWalk(scale=8.0, ridge=10.0))
        tv = (
            joint([Gaussian(y, mean=1.0 * tv_fld, sd=0.2), TotalVariation(over=tv_fld, shape=(n,), weight=4.0)])
            .fit(how="map")
            .mean("f")
        )
        self.assertGreater(jump(tv), 3 * jump(sm))  # TV keeps the step; smoothing blurs it
        self.assertLess(rmse(tv), rmse(sm))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class PottsTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_pulls_toward_discrete_levels(self):
        n = 60
        x = np.linspace(0, 1, n)
        f_true = (x > 0.5).astype(float) * 2.0
        y = f_true + 0.2 * np.random.RandomState(0).randn(n)
        fld = GP("g", index=np.arange(n), kernel=RandomWalk(scale=0.3, ridge=5.0))
        g = (
            joint([Gaussian(y, mean=1.0 * fld, sd=0.3), Potts(over=fld, levels=[0.0, 2.0], weight=3.0)])
            .fit(how="map")
            .mean("g")
        )
        dist = np.minimum(np.abs(g - 0.0), np.abs(g - 2.0))
        self.assertLess(dist.mean(), 0.2)  # the field sits near one of the two materials


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ComplexDataTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_field_from_complex_observations(self):
        m = 20
        rng = np.random.RandomState(1)
        A = rng.randn(12, m) + 1j * rng.randn(12, m)
        f_true = np.sin(2 * np.pi * np.linspace(0, 1, m))
        y = A @ f_true + 0.01 * (rng.randn(12) + 1j * rng.randn(12))
        At = torch.as_tensor(A)
        fld = GP("h", index=np.arange(m), kernel=RandomWalk(scale=0.5, ridge=3.0))
        obs = Differential(y, over=fld, scale=0.01, forward=lambda p, ops: At @ p.field.to(torch.complex128))
        hm, hs = joint([obs]).fit(how="laplace").posterior("h")
        self.assertGreater(np.corrcoef(hm, f_true)[0, 1], 0.95)
        self.assertTrue(np.all(hs > 0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MultistartTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_escapes_local_minimum(self):
        t = np.linspace(0, 2, 40)
        w_true = 3.0
        y = np.sin(w_true * t) + 0.02 * np.random.RandomState(2).randn(40)
        w = free(1, name="w", support="positive")
        model = joint(
            [
                Differential(
                    y, drivers=[w], y0=0.0, t_grid=t, scale=0.02, rhs=lambda u, tt, p, ops: p.w * ops.cos(p.w * tt)
                )
            ]
        )
        bad = model.fit(how="laplace", init={"w": np.log(7.0)}).mean("w")
        best = multistart(model, [{"w": np.log(7.0)}, {"w": np.log(2.0)}], how="laplace").mean("w")
        self.assertGreater(abs(bad - w_true), 1.0)  # the bad init is stuck in a local mode
        self.assertLess(abs(best - w_true), 0.5)  # multistart finds the global one


if __name__ == "__main__":
    unittest.main()
