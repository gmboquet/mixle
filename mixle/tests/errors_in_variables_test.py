"""Errors-in-variables (Deming) regression: unbiased slope under a noisy predictor (Phase 6)."""

import unittest

import numpy as np

from mixle.inference.errors_in_variables import deming_regression


class DemingRegressionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.a, self.b = 2.0, 3.0
        self.xstar = rng.uniform(0, 10, 500)  # true predictor (e.g. true depth / location)
        self.sig_x, self.sig_y = 1.2, 1.0
        self.x = self.xstar + rng.randn(500) * self.sig_x  # predictor observed with error
        self.y = self.a + self.b * self.xstar + rng.randn(500) * self.sig_y
        self.ols = np.polyfit(self.x, self.y, 1)[0]

    def test_recovers_slope_that_ols_attenuates(self):
        self.assertLess(self.ols, self.b - 0.2)  # OLS is biased toward zero (regression dilution)
        fit = deming_regression(self.x, self.y, variance_ratio=self.sig_y**2 / self.sig_x**2)
        self.assertAlmostEqual(fit.slope, self.b, delta=0.2)
        self.assertLess(abs(fit.slope - self.b), abs(self.ols - self.b))

    def test_recovers_latent_predictor(self):
        fit = deming_regression(self.x, self.y, variance_ratio=self.sig_y**2 / self.sig_x**2)
        rmse_obs = np.sqrt(np.mean((self.x - self.xstar) ** 2))
        rmse_lat = np.sqrt(np.mean((fit.x_latent - self.xstar) ** 2))
        self.assertLess(rmse_lat, rmse_obs)  # the recovered x* is closer to the truth than the noisy input

    def test_large_variance_ratio_recovers_ols(self):
        fit = deming_regression(self.x, self.y, variance_ratio=1e6)
        self.assertAlmostEqual(fit.slope, self.ols, delta=1e-3)

    def test_conditional_mean_on_true_values(self):
        fit = deming_regression(self.x, self.y, variance_ratio=1.0)
        np.testing.assert_allclose(
            fit.conditional_mean(np.array([0.0, 1.0])), [fit.intercept, fit.intercept + fit.slope]
        )


if __name__ == "__main__":
    unittest.main()
