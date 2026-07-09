"""Kernel density / mode / intensity estimation (mixle.stats.kde)."""

import unittest

import numpy as np
from numpy import trapezoid

from mixle.analysis import (
    intensity,
    kde,
    kde_mode,
    scott_bandwidth,
    silverman_bandwidth,
)


class KDETest(unittest.TestCase):
    def test_integrates_to_one(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 3000)
        f = kde(x)
        grid = np.linspace(-7, 7, 3000)
        self.assertAlmostEqual(trapezoid(f(grid), grid), 1.0, delta=0.01)

    def test_recovers_normal_density(self):
        rng = np.random.RandomState(1)
        x = rng.normal(0, 1, 5000)
        f = kde(x)
        self.assertAlmostEqual(float(f(np.array([0.0]))[0]), 1.0 / np.sqrt(2 * np.pi), delta=0.03)

    def test_boundary_correction_reduces_edge_bias(self):
        rng = np.random.RandomState(2)
        x = rng.exponential(1.0, 5000)  # true density at 0+ is 1.0
        plain = kde(x)
        refl = kde(x, bounds=(0.0, None))
        at0 = np.array([0.02])
        # reflection is much closer to the true edge density of 1.0
        self.assertLess(float(plain(at0)[0]), 0.7)
        self.assertGreater(float(refl(at0)[0]), 0.8)
        gi = np.linspace(0, 10, 4000)
        self.assertAlmostEqual(trapezoid(refl(gi), gi), 1.0, delta=0.02)

    def test_adaptive_integrates_to_one(self):
        rng = np.random.RandomState(3)
        x = rng.standard_t(3, 3000)  # heavy tails benefit from adaptive bw
        f = kde(x, adaptive=True)
        grid = np.linspace(x.min() - 1, x.max() + 1, 4000)
        self.assertAlmostEqual(trapezoid(f(grid), grid), 1.0, delta=0.03)

    def test_bandwidth_selectors_positive(self):
        rng = np.random.RandomState(4)
        x = rng.normal(0, 2, 1000)
        self.assertGreater(silverman_bandwidth(x), 0)
        self.assertGreater(scott_bandwidth(x), 0)


class ModeTest(unittest.TestCase):
    def test_recovers_mode(self):
        rng = np.random.RandomState(0)
        x = rng.normal(5.0, 1.0, 5000)
        self.assertAlmostEqual(kde_mode(x), 5.0, delta=0.3)

    def test_bimodal_mode_at_higher_peak(self):
        rng = np.random.RandomState(1)
        x = np.concatenate([rng.normal(-3, 0.4, 2000), rng.normal(3, 0.4, 6000)])  # taller peak at +3
        self.assertAlmostEqual(kde_mode(x), 3.0, delta=0.4)

    def test_bootstrap_ci_brackets_mode(self):
        rng = np.random.RandomState(2)
        x = rng.normal(0.0, 1.0, 3000)
        out = kde_mode(x, ci=True, n_boot=40, seed=0)
        self.assertLessEqual(out["ci_low"], out["mode"])
        self.assertLessEqual(out["mode"], out["ci_high"])


class IntensityTest(unittest.TestCase):
    def test_integral_recovers_event_count(self):
        rng = np.random.RandomState(0)
        events = np.sort(rng.uniform(0, 10, 200))
        grid = np.linspace(0, 10, 1000)
        lam = intensity(events, grid, domain=(0, 10), bandwidth=0.5)
        # the intensity integrates to ~ the number of events
        self.assertAlmostEqual(trapezoid(lam, grid), 200.0, delta=20.0)

    def test_inhomogeneous_rate_tracks_density(self):
        rng = np.random.RandomState(1)
        # events concentrated near t=8
        events = np.sort(np.concatenate([rng.uniform(0, 10, 50), rng.normal(8, 0.5, 300)]))
        grid = np.array([2.0, 8.0])
        lam = intensity(events, grid, domain=(0, 10), bandwidth=0.5)
        self.assertGreater(lam[1], 3 * lam[0])  # much higher intensity at t=8


if __name__ == "__main__":
    unittest.main()
