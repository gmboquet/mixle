"""Gaussian conditional (`given=`-style) sampling: MVN.condition(observed)."""

import unittest

import numpy as np

from pysp.stats import MultivariateGaussianDistribution


class MvnConditionalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.mu = np.array([1.0, -2.0, 0.5, 3.0])
        a = rng.randn(4, 4)
        self.cov = a @ a.T + np.eye(4)
        self.d = MultivariateGaussianDistribution(self.mu, self.cov)

    def _ref(self, observed):
        o = sorted(observed)
        u = [i for i in range(4) if i not in observed]
        xo = np.array([observed[i] for i in o])
        s_oo, s_uo, s_uu = self.cov[np.ix_(o, o)], self.cov[np.ix_(u, o)], self.cov[np.ix_(u, u)]
        mu = self.mu[u] + s_uo @ np.linalg.solve(s_oo, xo - self.mu[o])
        cov = s_uu - s_uo @ np.linalg.solve(s_oo, s_uo.T)
        return mu, cov

    def test_matches_closed_form(self):
        c = self.d.condition({0: 2.0, 2: -1.0})
        mu_ref, cov_ref = self._ref({0: 2.0, 2: -1.0})
        self.assertEqual(c.dim, 2)
        np.testing.assert_allclose(c.mu, mu_ref, atol=1e-10)
        np.testing.assert_allclose(c.covar, cov_ref, atol=1e-10)

    def test_sampling_the_conditional(self):
        c = self.d.condition({0: 2.0, 2: -1.0})
        mu_ref, cov_ref = self._ref({0: 2.0, 2: -1.0})
        s = np.array(c.sampler(seed=1).sample(60000))
        np.testing.assert_allclose(s.mean(axis=0), mu_ref, atol=0.04)
        np.testing.assert_allclose(np.cov(s.T), cov_ref, atol=0.06)

    def test_condition_on_single_gives_three_dim(self):
        c = self.d.condition({1: 0.0})
        self.assertEqual(c.dim, 3)
        np.testing.assert_allclose(c.mu, self._ref({1: 0.0})[0], atol=1e-10)

    def test_empty_returns_full_and_all_observed_raises(self):
        np.testing.assert_allclose(self.d.condition({}).mu, self.mu)
        with self.assertRaises(ValueError):
            self.d.condition({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})


if __name__ == "__main__":
    unittest.main()
