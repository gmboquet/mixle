"""Inducing-point sparse GP (FITC): converges to the exact GP, calibrated, scalable (Phase 3)."""

import time
import unittest

import numpy as np

from mixle.models.sparse_gaussian_process import SparseGaussianProcessRegressor as SGPR
from mixle.models.sparse_gaussian_process import _kernel


def _data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(-3, 3, n))
    y = np.sin(2 * x) * np.exp(-0.1 * x**2) + rng.randn(n) * 0.1
    return x, y


def _exact_gp(x, y, xs, ls=0.5, amp=1.0, noise=0.1):
    k = _kernel(x, x, ls, amp, "rbf") + noise**2 * np.eye(len(x))
    return _kernel(xs, x, ls, amp, "rbf") @ np.linalg.solve(k, y - y.mean()) + y.mean()


class SparseGPTest(unittest.TestCase):
    def test_converges_to_exact_gp_as_inducing_points_grow(self):
        x, y = _data()
        xs = np.linspace(-2.8, 2.8, 80)
        exact = _exact_gp(x, y, xs)
        gaps = []
        for m in (10, 25, 50):
            g = SGPR(lengthscale=0.5, amplitude=1.0, noise=0.1, n_inducing=m).fit(x, y, optimize=False, seed=0)
            gaps.append(np.sqrt(np.mean((g.predict(xs) - exact) ** 2)))
        self.assertGreater(gaps[0], gaps[1])  # more inducing points -> closer to exact
        self.assertLess(gaps[-1], 1e-3)  # m=50 essentially exact

    def test_recovers_function_with_calibrated_variance(self):
        x, y = _data()
        xs = np.linspace(-2.8, 2.8, 80)
        truth = np.sin(2 * xs) * np.exp(-0.1 * xs**2)
        g = SGPR(n_inducing=40).fit(x, y, optimize=True, seed=1)
        mean, var = g.predict(xs, return_var=True)
        self.assertLess(np.sqrt(np.mean((mean - truth) ** 2)), 0.1)
        self.assertTrue(np.all(var > 0))
        self.assertLess(abs(np.mean(np.abs(mean - truth) / np.sqrt(var) < 1) - 0.68), 0.15)  # ~68% within 1 sd

    def test_scales_to_large_n(self):
        rng = np.random.RandomState(2)
        n = 50000
        x = rng.uniform(-3, 3, n)
        y = np.sin(2 * x) + rng.randn(n) * 0.1
        t = time.time()
        g = SGPR(n_inducing=80).fit(x, y, optimize=False)
        pred = g.predict(np.linspace(-2.8, 2.8, 50))
        self.assertLess(time.time() - t, 5.0)  # linear in n, not O(n^3)
        self.assertEqual(pred.shape, (50,))

    def test_multidimensional_inputs(self):
        rng = np.random.RandomState(3)
        x = rng.uniform(-2, 2, (400, 2))
        truth = np.sin(x[:, 0]) * np.cos(x[:, 1])
        y = truth + rng.randn(400) * 0.05
        g = SGPR(n_inducing=60).fit(x, y, optimize=True, seed=0)
        self.assertLess(np.sqrt(np.mean((g.predict(x) - truth) ** 2)), 0.1)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            SGPR().predict(np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
