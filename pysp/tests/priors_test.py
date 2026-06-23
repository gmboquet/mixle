"""Tests for the edge-preserving / discrete field priors (TotalVariation, Potts) over a Gaussian forward.

The PDE-forward variants of these (complex-valued observations, multistart over a Differential forward)
moved to the pysparkplug-pde package's tests along with the PDE stack.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import GP, Gaussian, Potts, RandomWalk, TotalVariation, joint


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


if __name__ == "__main__":
    unittest.main()
