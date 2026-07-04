"""GLM IRLS is robust to rank-deficient / collinear design (no crash on singular normal equations)."""

import unittest

import numpy as np

from mixle.inference.glm import _solve_psd, glm


class SolveRobustTest(unittest.TestCase):
    def test_solve_psd_matches_solve_when_full_rank(self):
        a = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        np.testing.assert_allclose(_solve_psd(a, b), np.linalg.solve(a, b))

    def test_solve_psd_returns_min_norm_on_singular(self):
        a = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, singular
        b = np.array([2.0, 2.0])
        x = _solve_psd(a, b)
        self.assertTrue(np.all(np.isfinite(x)))  # no raise; a finite least-squares solution
        np.testing.assert_allclose(a @ x, b, atol=1e-8)

    def test_glm_fits_with_a_duplicated_collinear_column(self):
        rng = np.random.RandomState(0)
        x0 = rng.randn(200)
        # a design with a perfectly collinear duplicate column + intercept -> singular X'WX
        X = np.column_stack([np.ones(200), x0, x0])
        y = (1.0 / (1.0 + np.exp(-(0.5 + 1.5 * x0))) > rng.rand(200)).astype(float)
        res = glm(X, y, family="binomial")  # must not raise
        self.assertTrue(np.all(np.isfinite(res.coef)))
        self.assertTrue(np.all(np.isfinite(res.se)))
        # the two collinear columns share the slope; predictions still track the signal
        pred = res.predict(X)
        self.assertGreater(np.corrcoef(pred, y)[0, 1], 0.3)


if __name__ == "__main__":
    unittest.main()
