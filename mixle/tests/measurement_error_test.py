"""Measurement-error models: SIMEX + MC propagation (mixle.inference.errors_in_variables)."""

import unittest

import numpy as np

from mixle.inference import (
    deming_regression,
    propagate_uncertainty,
    simex,
)


def _ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(X, y, rcond=None)[0]


class SimexTest(unittest.TestCase):
    def test_reduces_attenuation_bias(self):
        rng = np.random.RandomState(0)
        n = 3000
        x_star = rng.normal(0, 1, n)
        beta = 2.0
        y = 1.0 + beta * x_star + rng.normal(0, 0.5, n)
        sigma_u = 0.8
        x = x_star + rng.normal(0, sigma_u, n)
        naive = _ols(x, y)
        sim = simex(_ols, x, y, sigma_u, n_sims=80, seed=1)
        # SIMEX moves the slope from the attenuated naive estimate back toward the truth
        self.assertLess(abs(sim["estimate"][1] - beta), abs(naive[1] - beta))
        self.assertGreater(sim["estimate"][1], naive[1])
        self.assertAlmostEqual(sim["estimate"][0], 1.0, delta=0.1)  # intercept ~unbiased

    def test_curve_shape(self):
        rng = np.random.RandomState(1)
        x_star = rng.normal(0, 1, 1000)
        y = 2.0 * x_star + rng.normal(0, 0.5, 1000)
        x = x_star + rng.normal(0, 0.6, 1000)
        sim = simex(_ols, x, y, 0.6, n_sims=40, seed=2)
        # adding more noise attenuates the slope further (curve decreasing in lambda)
        self.assertTrue(sim["curve"][-1, 1] < sim["curve"][0, 1])

    def test_deming_still_works(self):
        # sanity that the existing deming_regression export is intact
        rng = np.random.RandomState(2)
        x_star = rng.normal(0, 1, 2000)
        x = x_star + rng.normal(0, 0.5, 2000)
        y = 1.0 + 2.0 * x_star + rng.normal(0, 0.5, 2000)
        fit = deming_regression(x, y, variance_ratio=1.0)
        self.assertAlmostEqual(fit.slope, 2.0, delta=0.3)


class PropagateTest(unittest.TestCase):
    def test_vectorized_function(self):
        rng = np.random.RandomState(0)
        samples = rng.normal(3.0, 1.0, (10000, 1))
        res = propagate_uncertainty(lambda s: s[:, 0] ** 2, samples)
        # E[x^2] = mean^2 + var = 9 + 1 = 10
        self.assertAlmostEqual(float(res["mean"]), 10.0, delta=0.3)

    def test_per_row_function(self):
        rng = np.random.RandomState(1)
        samples = rng.normal(0.0, 1.0, (5000, 2))
        # a function that only works on a single row (not vectorized over axis 0)
        res = propagate_uncertainty(lambda row: float(np.linalg.norm(row)), samples)
        self.assertEqual(res["samples"].shape[0], 5000)
        self.assertGreater(float(res["mean"]), 0.0)

    def test_quantiles_ordered(self):
        rng = np.random.RandomState(2)
        samples = rng.normal(0.0, 1.0, (4000, 1))
        res = propagate_uncertainty(lambda s: s[:, 0], samples, quantiles=(0.1, 0.5, 0.9))
        q = res["quantiles"].ravel()
        self.assertTrue(q[0] < q[1] < q[2])


if __name__ == "__main__":
    unittest.main()
