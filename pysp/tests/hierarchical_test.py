"""Hierarchical partial-pooling normal: parameter recovery, shrinkage, and pooling gains (Phase 7)."""

import unittest

import numpy as np

from pysp.inference import estimate
from pysp.stats.hierarchical import HierarchicalNormalDistribution


def _hier_data(mu, tau, sigma, sizes, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.normal(mu, tau, len(sizes))
    groups = [rng.normal(theta[g], sigma, sizes[g]) for g in range(len(sizes))]
    return groups, theta


def _fit_hier(groups):
    """Fit a HierarchicalNormalDistribution through the pysp estimator contract."""
    return estimate(groups, HierarchicalNormalDistribution(0.0, 1.0, 1.0).estimator())


class HierarchicalNormalTest(unittest.TestCase):
    def test_recovers_population_parameters(self):
        rng = np.random.RandomState(0)
        sizes = rng.choice([2, 3, 5, 50], size=80)
        groups, _ = _hier_data(10.0, 2.0, 1.0, sizes, seed=1)
        h = _fit_hier(groups)
        self.assertAlmostEqual(h.mu, 10.0, delta=0.4)
        self.assertAlmostEqual(h.tau, 2.0, delta=0.4)
        self.assertAlmostEqual(h.sigma, 1.0, delta=0.15)

    def test_small_groups_shrink_more(self):
        groups, _ = _hier_data(0.0, 1.0, 3.0, [5] * 40, seed=2)
        h = _fit_hier(groups)
        self.assertLess(h.shrinkage(2), h.shrinkage(50))  # smaller n -> pulled more toward mu
        self.assertLess(h.shrinkage(50), 1.0)

    def test_partial_pooling_beats_no_pooling_and_complete_pooling(self):
        # strong-pooling regime: small noisy groups (sigma^2/n ~ tau^2) -> shrinkage helps a lot
        sizes = [4] * 60
        groups, theta = _hier_data(5.0, 1.5, 3.0, sizes, seed=3)
        h = _fit_hier(groups)
        ybar = np.array([g.mean() for g in groups])
        pooled = np.array([h.group_posterior(ybar[i], sizes[i])[0] for i in range(len(sizes))])
        grand = np.concatenate(groups).mean()
        rmse_no = np.sqrt(np.mean((ybar - theta) ** 2))  # no pooling: each group's own mean
        rmse_pp = np.sqrt(np.mean((pooled - theta) ** 2))  # partial pooling: shrinkage estimates
        rmse_cp = np.sqrt(np.mean((grand - theta) ** 2))  # complete pooling: one grand mean
        self.assertLess(rmse_pp, rmse_no)  # James-Stein: shrinkage dominates the per-group MLE
        self.assertLess(rmse_pp, rmse_cp)  # but real between-group variation beats full pooling

    def test_group_posterior_is_between_mean_and_data(self):
        groups, _ = _hier_data(0.0, 1.0, 2.0, [5] * 30, seed=4)
        h = _fit_hier(groups)
        m, sd = h.group_posterior(ybar=4.0, n=5)  # a group mean far from mu=0
        self.assertTrue(h.mu < m < 4.0)  # shrunk from the data toward the population mean
        self.assertGreater(sd, 0.0)


if __name__ == "__main__":
    unittest.main()
