"""Tests for the GP kernel selection (RBF default + Matern 3/2 and 5/2)."""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


class KernelNameTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_default_is_rbf_and_aliases_resolve(self):
        from pysp.models.gaussian_process import GaussianProcessRegressor

        self.assertEqual(GaussianProcessRegressor().kernel_name, "rbf")
        self.assertEqual(GaussianProcessRegressor(kernel="Matern").kernel_name, "matern52")
        self.assertEqual(GaussianProcessRegressor(kernel="matern_3_2").kernel_name, "matern32")
        self.assertEqual(GaussianProcessRegressor(kernel="squared_exponential").kernel_name, "rbf")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_unknown_kernel_raises(self):
        from pysp.models.gaussian_process import GaussianProcessRegressor

        with self.assertRaises(ValueError):
            GaussianProcessRegressor(kernel="banana")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class KernelMathTest(unittest.TestCase):
    def _gram(self, kernel, x):
        from pysp.models.gaussian_process import GaussianProcessRegressor

        gp = GaussianProcessRegressor(lengthscale=1.0, amplitude=1.0, kernel=kernel)
        return np.asarray(gp.kernel(x, x).detach().cpu().numpy(), dtype=float)

    def test_diagonal_is_amplitude_squared_and_psd(self):
        x = np.linspace(-2.0, 2.0, 12)[:, None]
        for kernel in ("rbf", "matern32", "matern52"):
            k = self._gram(kernel, x)
            np.testing.assert_allclose(np.diag(k), 1.0, atol=1e-6)  # k(x,x) = amplitude^2 = 1
            self.assertTrue(np.allclose(k, k.T, atol=1e-8))
            eig = np.linalg.eigvalsh(k)
            self.assertGreater(eig.min(), -1e-8)  # positive semidefinite

    def test_matern_matches_closed_form_at_known_separation(self):
        # Unit lengthscale/amplitude; separation r = 1 between the two points.
        x = np.array([[0.0], [1.0]])
        r = 1.0
        m32 = self._gram("matern32", x)[0, 1]
        m52 = self._gram("matern52", x)[0, 1]
        exp_m32 = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        exp_m52 = (1 + np.sqrt(5) * r + 5.0 / 3.0 * r**2) * np.exp(-np.sqrt(5) * r)
        self.assertAlmostEqual(m32, exp_m32, places=5)
        self.assertAlmostEqual(m52, exp_m52, places=5)

    def test_matern_is_heavier_tailed_than_rbf(self):
        # At a few lengthscales out, the rougher Matern kernels retain more covariance than RBF.
        x = np.array([[0.0], [2.5]])
        rbf = self._gram("rbf", x)[0, 1]
        m32 = self._gram("matern32", x)[0, 1]
        m52 = self._gram("matern52", x)[0, 1]
        self.assertGreater(m32, rbf)
        self.assertGreater(m52, rbf)
        self.assertGreater(m32, m52)  # 3/2 is the roughest -> heaviest tail

    def test_matern_gp_recovers_a_smooth_function(self):
        from pysp.models.gaussian_process import GaussianProcessRegressor

        rng = np.random.RandomState(0)
        x = np.linspace(0.0, 1.0, 40)
        y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.05, x.size)
        gp = GaussianProcessRegressor(lengthscale=0.2, amplitude=1.0, noise=0.1, kernel="matern52")
        gp.fit(x, y, max_its=120, out=None)
        pred = gp.predict(x, y, x)
        self.assertLess(float(np.sqrt(np.mean((pred - y) ** 2))), 0.2)  # fits the data


if __name__ == "__main__":
    unittest.main()
