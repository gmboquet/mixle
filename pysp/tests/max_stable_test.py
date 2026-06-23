"""Smith max-stable process for spatial extremes: extremal coefficient, margins, dependence (Phase 6)."""

import unittest

import numpy as np
from scipy.stats import norm, spearmanr

from pysp.stats.processes.max_stable import SmithMaxStable, fit_smith_maxstable


class SmithMaxStableTest(unittest.TestCase):
    def setUp(self):
        self.ms = SmithMaxStable(sigma=2.0 * np.eye(2))

    def test_extremal_coefficient_bounds_and_formula(self):
        self.assertAlmostEqual(self.ms.extremal_coefficient([0, 0]), 1.0, places=6)  # full dependence at h=0
        self.assertAlmostEqual(self.ms.extremal_coefficient([100, 0]), 2.0, places=4)  # independence far away
        a = np.sqrt(np.array([3.0, 0.0]) @ np.linalg.inv(self.ms.sigma) @ np.array([3.0, 0.0]))
        self.assertAlmostEqual(self.ms.extremal_coefficient([3, 0]), 2 * norm.cdf(a / 2))

    def test_extremal_coefficient_is_monotone(self):
        thetas = [self.ms.extremal_coefficient([h, 0]) for h in (0, 1, 2, 4, 8)]
        self.assertTrue(all(thetas[i] <= thetas[i + 1] + 1e-9 for i in range(len(thetas) - 1)))

    def test_bivariate_cdf_is_a_valid_probability(self):
        self.assertTrue(0.0 < self.ms.bivariate_cdf(1.0, 1.0, [2, 0]) < 1.0)

    def test_sampler_has_unit_frechet_margins(self):
        s = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=0).sample(4000, n_storms=150)
        np.testing.assert_allclose(np.median(s, axis=0), 1.0 / np.log(2), atol=0.2)  # unit-Frechet median

    def test_short_range_extremes_are_more_dependent(self):
        loc = np.array([[0, 0], [0.5, 0], [8, 0]])
        s = self.ms.sampler(loc, seed=0).sample(3000, n_storms=150)
        near = spearmanr(s[:, 0], s[:, 1]).correlation
        far = spearmanr(s[:, 0], s[:, 2]).correlation
        self.assertGreater(near, far)

    def test_fit_recovers_the_dependence_scale(self):
        true = SmithMaxStable(2.0**2 * np.eye(2))
        locs = np.random.RandomState(1).uniform(0, 12, (10, 2))
        fields = true.sampler(locs, seed=2).sample(500, n_storms=120)
        fit = fit_smith_maxstable(locs, fields)
        self.assertAlmostEqual(np.sqrt(fit.sigma[0, 0]), 2.0, delta=0.6)


if __name__ == "__main__":
    unittest.main()
