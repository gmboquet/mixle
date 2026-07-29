"""Tests for the edge-preserving / discrete field priors (TotalVariation, Potts) over a Gaussian forward.

The PDE-forward variants of these (complex-valued observations, multistart over a Differential forward)
moved to the mixle-pde package's tests along with the PDE stack.
"""

import pickle
import unittest

import numpy as np

from mixle.ppl._grid import _grid_faces
from mixle.ppl.priors import Potts, TotalVariation

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle.ppl import GP, Gaussian, RandomWalk, joint


class SpatialPriorContractTest(unittest.TestCase):
    def test_grid_rejects_invalid_shape_and_spacing(self):
        for shape in ((), (0,), (-1, 2), (2.0, 3), (True, 2)):
            with self.subTest(shape=repr(shape)), self.assertRaises((TypeError, ValueError)):
                _grid_faces(shape, 1.0)
        for spacing in (0.0, -1.0, np.inf, np.nan, [1.0, 0.0]):
            with self.subTest(spacing=repr(spacing)), self.assertRaises(ValueError):
                _grid_faces((2, 2), spacing)

    def test_degenerate_one_cell_axes_have_valid_empty_faces(self):
        one = _grid_faces((1,), 2.0)
        self.assertEqual(one["n"], 1)
        np.testing.assert_array_equal(one["boundary"], [0])
        self.assertEqual(one["interior"].size, 0)
        self.assertEqual(one["face_a"].size, 0)
        slab = _grid_faces((1, 3), [1.0, 2.0])
        self.assertEqual(slab["n"], 3)
        self.assertEqual(slab["face_a"].size, 2)
        self.assertTrue(np.all(np.isfinite(slab["face_w"])))
        self.assertTrue(np.all(slab["face_w"] > 0.0))

    def test_prior_specifications_are_validated(self):
        for kwargs in (
            {"shape": (0,), "weight": 1.0, "eps": 1e-3},
            {"shape": (2,), "weight": 0.0, "eps": 1e-3},
            {"shape": (2,), "weight": np.inf, "eps": 1e-3},
            {"shape": (2,), "weight": 1.0, "eps": 0.0},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                TotalVariation(object(), **kwargs)
        for levels in ([], [1.0], [1.0, 1.0], [0.0, np.nan]):
            with self.subTest(levels=repr(levels)), self.assertRaises((TypeError, ValueError)):
                Potts(object(), levels)
        with self.assertRaises(ValueError):
            Potts(object(), [0.0, 1.0], weight=-1.0)

    def test_penalty_proxies_round_trip_through_pickle(self):
        _, tv = TotalVariation(object(), shape=(2, 2), weight=2.0, eps=0.1)
        _, potts = Potts(object(), levels=[0.0, 2.0], weight=3.0)
        tv_back = pickle.loads(pickle.dumps(tv))
        potts_back = pickle.loads(pickle.dumps(potts))
        self.assertEqual(tv_back.prefix, "tv")
        self.assertEqual(potts_back.prefix, "potts")
        self.assertEqual(tv_back._penalty, tv._penalty)
        self.assertEqual(potts_back._penalty, potts._penalty)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_penalty_indices_follow_field_device_and_geometry(self):
        _, tv = TotalVariation(object(), shape=(2, 2), weight=2.0, eps=0.1)
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        output = tv.loglik(values, {}, torch)
        self.assertEqual(output.device, values.device)
        self.assertTrue(torch.isfinite(output))
        with self.assertRaisesRegex(ValueError, "requires 4"):
            tv.loglik(torch.tensor([0.0, 1.0]), {}, torch)
        if torch.cuda.is_available():
            cuda_values = values.cuda()
            self.assertEqual(tv.loglik(cuda_values, {}, torch).device.type, "cuda")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TotalVariationTestCase(unittest.TestCase):
    def setUp(self):
        # float64 for the MAP fits, restored afterward: a leaked float64 default breaks every
        # float32-module test that runs later in the same process (Float/Double matmul errors).
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
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
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
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
