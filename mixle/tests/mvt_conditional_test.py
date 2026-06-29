"""Multivariate Student-t closed-form conditional (Phase B given= conditional sampling)."""

import unittest

import numpy as np

from mixle.stats import MultivariateStudentTDistribution as MVT


class MVTConditionalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.dof = 6.0
        self.mu = np.array([1.0, -2.0, 0.5, 3.0])
        L = rng.randn(4, 4)
        self.shape = L @ L.T + np.eye(4)
        self.d = MVT(self.dof, self.mu, self.shape)

    def test_conditional_equals_joint_over_marginal(self):
        """The conditional density must equal joint / marginal at every test point."""
        obs_idx, unobs_idx = [0, 2], [1, 3]
        x_o = np.array([2.0, -1.0])
        cond = self.d.condition({0: 2.0, 2: -1.0})
        marg = MVT(self.dof, self.mu[obs_idx], self.shape[np.ix_(obs_idx, obs_idx)])  # marginal over observed
        rng = np.random.RandomState(1)
        for _ in range(8):
            x_u = rng.randn(2)
            full = np.empty(4)
            full[obs_idx] = x_o
            full[unobs_idx] = x_u
            self.assertAlmostEqual(cond.log_density(x_u), self.d.log_density(full) - marg.log_density(x_o), places=9)

    def test_degrees_of_freedom_raised_by_observed_count(self):
        self.assertEqual(self.d.condition({0: 2.0, 2: -1.0}).dof, self.dof + 2)
        self.assertEqual(self.d.condition({1: 0.0}).dof, self.dof + 1)

    def test_condition_on_nothing_is_identity(self):
        c = self.d.condition({})
        np.testing.assert_allclose(c.mu, self.mu)
        np.testing.assert_allclose(c.shape, self.shape)
        self.assertEqual(c.dof, self.dof)

    def test_requires_some_unobserved(self):
        with self.assertRaises(ValueError):
            self.d.condition({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})


if __name__ == "__main__":
    unittest.main()
