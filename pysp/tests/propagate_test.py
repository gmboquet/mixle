"""Forward uncertainty propagation: Monte Carlo + unscented transform (Phase 4)."""

import unittest

import numpy as np

from pysp.doe import propagate, unscented_transform


class PropagateTest(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1.0, -2.0, 0.5])
        self.mu = np.array([1.0, 2.0, 3.0])
        rng = np.random.RandomState(0)
        chol = rng.randn(3, 3)
        self.cov = chol @ chol.T + np.eye(3)
        self.f = lambda x: x @ self.a
        self.true_mean = self.a @ self.mu
        self.true_var = self.a @ self.cov @ self.a

    def test_unscented_is_exact_for_linear_models(self):
        out = propagate(self.f, self.mu, self.cov, method="unscented")
        self.assertAlmostEqual(out["mean"], self.true_mean, places=8)
        self.assertAlmostEqual(out["std"] ** 2, self.true_var, places=6)

    def test_monte_carlo_matches_linear_moments(self):
        out = propagate(self.f, self.mu, self.cov, n=200000, method="montecarlo", seed=1)
        self.assertAlmostEqual(out["mean"], self.true_mean, delta=0.05)
        self.assertAlmostEqual(out["std"], np.sqrt(self.true_var), delta=0.05)

    def test_monte_carlo_and_unscented_agree_on_mild_nonlinearity(self):
        g = lambda x: x[:, 0] ** 2 + x[:, 1]
        mu, cov = np.array([1.0, 0.0]), 0.1 * np.eye(2)
        mc = propagate(g, mu, cov, n=300000, method="montecarlo", seed=2)
        ut = propagate(g, mu, cov, method="unscented")
        self.assertAlmostEqual(mc["mean"], 1.1, delta=0.02)  # E[x1^2 + x2] = (1 + 0.1) + 0
        self.assertAlmostEqual(ut["mean"], 1.1, delta=0.02)
        self.assertAlmostEqual(mc["std"], ut["std"], delta=0.03)

    def test_monte_carlo_quantiles_are_ordered(self):
        out = propagate(self.f, self.mu, self.cov, n=50000, method="montecarlo", seed=3)
        q = out["quantiles"]
        self.assertLess(q[0.05], q[0.5])
        self.assertLess(q[0.5], q[0.95])

    def test_vector_output_covariance(self):
        h = lambda x: np.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], axis=1)
        m, c = unscented_transform(h, np.array([0.0, 0.0]), np.eye(2))
        self.assertEqual(np.shape(m), (2,))
        np.testing.assert_allclose(c, [[2.0, 0.0], [0.0, 2.0]], atol=1e-6)  # var(x0+x1)=2, uncorrelated

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            propagate(self.f, self.mu, self.cov, method="bogus")


if __name__ == "__main__":
    unittest.main()
