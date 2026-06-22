"""Multivariate (mutually-exciting) Hawkes: exact likelihood, valid sampling, branching-EM recovery."""

import unittest
import warnings

import numpy as np

from pysp.inference import estimate
from pysp.stats import MultivariateHawkesProcessDistribution
from pysp.stats.leaf.multivariate_hawkes import _split


def _brute_log_density(d, ev):
    """O(n^2) reference likelihood (direct kernel sum over all earlier events)."""
    t, m = _split(ev)
    n = t.size
    mu, al, be, w = d.mu, d.alpha, d.beta, d.window
    ll = 0.0
    for i in range(n):
        lam = mu[m[i]] + sum(al[m[i], m[k]] * np.exp(-be * (t[i] - t[k])) for k in range(i))
        ll += np.log(lam)
    comp = w * mu.sum() + sum(
        al[dd, m[k]] * (1 - np.exp(-be * (w - t[k]))) / be for dd in range(d.dim) for k in range(n)
    )
    return ll - comp


class MultivariateHawkesTest(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.5, 0.3])
        self.alpha = np.array([[0.4, 0.1], [0.2, 0.5]])
        self.beta = 1.5
        self.d = MultivariateHawkesProcessDistribution(self.mu, self.alpha, self.beta, 20.0)

    def test_subcritical_spectral_radius(self):
        self.assertLess(self.d.spectral_radius, 1.0)

    def test_recursion_matches_brute_force(self):
        for seed in (3, 4, 5):
            ev = self.d.sampler(seed=seed).sample()
            self.assertAlmostEqual(self.d.log_density(ev), _brute_log_density(self.d, ev), places=8)

    def test_seq_matches_scalar(self):
        evs = self.d.sampler(seed=4).sample(3)
        np.testing.assert_allclose(self.d.seq_log_density(evs), [self.d.log_density(e) for e in evs], atol=1e-10)

    def test_sampler_validity(self):
        ev = self.d.sampler(seed=3).sample()
        self.assertTrue(all(0 <= m < 2 for _, m in ev))
        self.assertTrue(all(ev[i][0] <= ev[i + 1][0] for i in range(len(ev) - 1)))
        self.assertTrue(all(0.0 <= t <= 20.0 for t, _ in ev))

    def test_branching_em_recovers_mu_and_branching_ratios(self):
        data = self.d.sampler(seed=0).sample(200)
        m = None
        for _ in range(35):  # branching EM, iterated by the standard estimate() driver
            m = estimate(data, self.d.estimator(), m)
        np.testing.assert_allclose(m.mu, self.mu, atol=0.1)
        # the branching ratios alpha/beta are the identifiable quantities (absolute beta is harder)
        np.testing.assert_allclose(m.alpha / m.beta, self.alpha / self.beta, atol=0.06)

    def test_super_critical_warns(self):
        d = MultivariateHawkesProcessDistribution([0.5, 0.3], [[1.6, 0.2], [0.2, 1.6]], 1.0, 20.0)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            d.sampler(seed=0)
            self.assertTrue(any("super-critical" in str(x.message) for x in wl))

    def test_bad_alpha_shape_raises(self):
        with self.assertRaises(ValueError):
            MultivariateHawkesProcessDistribution([0.5, 0.3], [[0.4, 0.1, 0.0], [0.2, 0.5, 0.0]], 1.5, 20.0)


if __name__ == "__main__":
    unittest.main()
