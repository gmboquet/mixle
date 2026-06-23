"""WS-1: ADMM box-constrained least squares (augmented-Lagrangian), checked vs scipy."""

import unittest

import numpy as np
from scipy.optimize import lsq_linear, nnls

from pysp.relations import admm_bounded_least_squares


class ADMMTest(unittest.TestCase):
    def test_nnls_matches_scipy(self):
        for seed in range(40):
            r = np.random.RandomState(seed)
            m, n = r.randint(5, 15), r.randint(3, 8)
            a = r.randn(m, n)
            b = r.randn(m)
            x = admm_bounded_least_squares(a, b, 0.0, np.inf, max_iter=8000)
            xs, _ = nnls(a, b)
            with self.subTest(seed=seed):
                self.assertTrue(np.all(x >= -1e-7))  # non-negative
                self.assertAlmostEqual(np.linalg.norm(a @ x - b), np.linalg.norm(a @ xs - b), places=4)

    def test_box_matches_scipy(self):
        for seed in range(40):
            r = np.random.RandomState(seed + 100)
            m, n = r.randint(5, 15), r.randint(3, 8)
            a = r.randn(m, n)
            b = r.randn(m)
            x = admm_bounded_least_squares(a, b, -0.5, 0.5, max_iter=8000)
            ref = lsq_linear(a, b, bounds=(-0.5, 0.5)).x
            with self.subTest(seed=seed):
                self.assertTrue(np.all(x >= -0.5 - 1e-7) and np.all(x <= 0.5 + 1e-7))
                self.assertTrue(np.allclose(x, ref, atol=2e-3))

    def test_unconstrained_limit_is_least_squares(self):
        r = np.random.RandomState(0)
        a = r.randn(10, 4)
        b = r.randn(10)
        x = admm_bounded_least_squares(a, b, -1e9, 1e9, max_iter=8000)
        ols = np.linalg.lstsq(a, b, rcond=None)[0]
        self.assertTrue(np.allclose(x, ols, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
