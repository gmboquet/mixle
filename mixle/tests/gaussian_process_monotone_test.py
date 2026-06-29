"""Tests for the monotone GP prediction (isotonic projection of the posterior mean, WS-M)."""

import importlib.util
import unittest

import numpy as np

from mixle.models.gaussian_process import _pava

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PavaTest(unittest.TestCase):
    def test_already_monotone_is_unchanged(self):
        y = np.array([0.0, 1.0, 1.0, 2.5, 4.0])
        np.testing.assert_allclose(_pava(y), y)

    def test_projects_to_nondecreasing_and_preserves_mean(self):
        y = np.array([3.0, 1.0, 2.0, 0.0, 5.0])
        z = _pava(y)
        self.assertTrue(np.all(np.diff(z) >= -1e-12))  # non-decreasing
        self.assertAlmostEqual(float(z.sum()), float(y.sum()), places=10)  # PAVA preserves total mass
        # known pool: [3,1,2,0] -> mean 1.5; then 5 stays -> [1.5,1.5,1.5,1.5,5]
        np.testing.assert_allclose(z, [1.5, 1.5, 1.5, 1.5, 5.0])

    def test_idempotent(self):
        y = np.array([2.0, -1.0, 0.5, 0.5, 3.0, 2.0])
        np.testing.assert_allclose(_pava(_pava(y)), _pava(y))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class MonotoneGpTest(unittest.TestCase):
    def _fit(self):
        from mixle.models.gaussian_process import GaussianProcessRegressor

        rng = np.random.RandomState(0)
        x = np.linspace(0.0, 1.0, 40)
        y = 2.0 * x + rng.normal(0.0, 0.15, x.size)  # monotone-increasing trend + noise
        gp = GaussianProcessRegressor(lengthscale=0.3, amplitude=1.0, noise=0.2)
        gp.fit(x, y, max_its=80, out=None)
        return gp, x, y

    def test_prediction_is_monotone(self):
        gp, x, y = self._fit()
        x_new = np.linspace(0.0, 1.0, 60)
        m = gp.predict_monotone(x, y, x_new, increasing=True)
        self.assertEqual(m.shape, (60,))
        self.assertTrue(np.all(np.diff(m) >= -1e-9))

    def test_decreasing_option(self):
        gp, x, y = self._fit()
        x_new = np.linspace(0.0, 1.0, 50)
        m = gp.predict_monotone(x, y, x_new, increasing=False)
        self.assertTrue(np.all(np.diff(m) <= 1e-9))

    def test_unsorted_inputs_handled(self):
        gp, x, y = self._fit()
        x_new = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
        m = gp.predict_monotone(x, y, x_new, increasing=True)
        order = np.argsort(x_new)
        self.assertTrue(np.all(np.diff(m[order]) >= -1e-9))  # monotone in x, regardless of input order


if __name__ == "__main__":
    unittest.main()
