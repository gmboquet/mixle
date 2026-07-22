"""Errors-in-variables (Deming) regression: unbiased slope under a noisy predictor (Phase 6)."""

import unittest

import numpy as np

from mixle.inference.errors_in_variables import deming_regression, propagate_uncertainty, simex


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

    def test_rejects_degenerate_input_instead_of_dividing_by_zero(self):
        with self.assertRaises(ValueError):
            deming_regression([1.0], [2.0])  # n=1
        with self.assertRaises(ValueError):
            deming_regression([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])  # constant x

    def test_uncorrelated_nonconstant_data_gives_slope_zero_not_inf(self):
        # x varies but is exactly uncorrelated with y: sxy == 0 divides the old formula by zero.
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        fit = deming_regression(x, y)
        self.assertEqual(fit.slope, 0.0)
        self.assertTrue(np.isfinite(fit.intercept))


class PropagateUncertaintyTest(unittest.TestCase):
    def test_scalar_vectorized_function_matches_per_row_loop(self):
        samples = np.random.RandomState(0).normal(3.0, 1.0, (500, 1))
        out = propagate_uncertainty(lambda s: s[:, 0] ** 2, samples)
        np.testing.assert_allclose(out["mean"], np.mean(samples[:, 0] ** 2))

    def test_per_row_function_with_unqualified_reduction_is_not_silently_wrong(self):
        # A realistic per-draw function that forgot `axis=`: intended to normalize EACH row to sum
        # to 1, but summing the whole (n, d) array at once when handed the full batch. Before the
        # fix, matching the outer shape alone was accepted as "vectorized, correct" -- silently
        # dividing every row by the GLOBAL sum instead of falling back to the (correct) per-row loop.
        def bad_normalizer(row_or_batch):
            return row_or_batch / row_or_batch.sum()

        rng = np.random.RandomState(0)
        samples = rng.uniform(0.1, 1.0, (50, 4))
        out = propagate_uncertainty(bad_normalizer, samples)
        want = samples / samples.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(out["mean"], want.mean(axis=0))
        np.testing.assert_allclose(out["samples"], want)


class SimexTest(unittest.TestCase):
    def test_naive_key_is_the_direct_fit_not_a_noisy_refit_at_lambdas_zero(self):
        # "naive" is documented as the lambda=0 (no added noise) estimate -- before the fix it was
        # curve[0], the average of n_sims NOISY refits at whatever lambdas[0] happens to be. That
        # coincides with the true lambda=0 fit only when the caller's lambdas array happens to
        # start at exactly 0.0 (the default does); a custom lambdas array that skips 0 entirely
        # (as here) exposes the difference directly.
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 500)
        y = 2.0 * x + rng.normal(0, 0.3, 500)

        def fit_slope(xx, yy):
            xx = np.asarray(xx, dtype=float).ravel()
            return np.array([np.sum(xx * yy) / np.sum(xx * xx)])

        theta0 = fit_slope(x, y)
        result = simex(fit_slope, x, y, sigma_u=0.3, lambdas=np.array([0.5, 1.0, 1.5]), n_sims=200, seed=1)

        np.testing.assert_allclose(result["naive"], theta0, atol=1e-10)
        self.assertFalse(np.allclose(result["naive"], result["curve"][0]))


if __name__ == "__main__":
    unittest.main()
