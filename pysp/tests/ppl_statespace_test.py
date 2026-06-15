"""Tests for linear-Gaussian state-space models (Kalman/RTS + EM)."""
import unittest

import numpy as np

from pysp.ppl import AR1, LocalLevel


class StateSpaceTestCase(unittest.TestCase):

    def test_local_level_smoothing(self):
        rng = np.random.RandomState(0)
        T = 600
        level = np.cumsum(rng.normal(0, 0.3, T))
        y = level + rng.normal(0, 0.5, T)
        m = LocalLevel().fit(list(y))
        self.assertAlmostEqual(m.result.level_sd, 0.3, delta=0.15)
        self.assertAlmostEqual(m.result.obs_sd, 0.5, delta=0.15)
        # smoothing reduces error vs the raw noisy observations
        smooth_rmse = np.sqrt(np.mean((m.result.smoothed - level) ** 2))
        raw_rmse = np.sqrt(np.mean((y - level) ** 2))
        self.assertLess(smooth_rmse, raw_rmse)

    def test_ar1_recovers_phi(self):
        rng = np.random.RandomState(1)
        T = 3000
        x = np.zeros(T)
        for t in range(1, T):
            x[t] = 0.8 * x[t - 1] + rng.normal(0, 0.4)
        y = x + rng.normal(0, 0.3, T)
        m = AR1().fit(list(y))
        self.assertAlmostEqual(m.result.phi, 0.8, delta=0.1)
        self.assertEqual(m.result.forecast(5).shape, (5,))
        self.assertEqual(set(m.params), {"phi", "level_sd", "obs_sd"})


if __name__ == "__main__":
    unittest.main()
