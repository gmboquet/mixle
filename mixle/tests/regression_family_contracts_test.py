"""Regression contracts for GLM, ordinal, and errors-in-variables models."""

import unittest

import numpy as np

from mixle.inference.errors_in_variables import deming_regression, propagate_uncertainty, simex
from mixle.inference.glm import elastic_net, glm, quantile_regression, ridge_regression, robust_regression
from mixle.inference.ordinal import ordinal_regression, somers_d


class GLMContractsTest(unittest.TestCase):
    def setUp(self):
        self.x = np.column_stack([np.ones(8), np.linspace(-1.0, 1.0, 8)])

    def test_response_support_is_enforced(self):
        cases = [
            ("binomial", np.array([0, 1, 0, 1, 0, 1, 0, 1.1])),
            ("poisson", np.array([0, 1, 2, 3, 4, 5, 6, 1.5])),
            ("negativebinomial", np.array([0, 1, 2, 3, 4, 5, 6, -1])),
            ("gamma", np.array([1, 2, 3, 4, 5, 6, 7, 0])),
            ("inverse_gaussian", np.array([1, 2, 3, 4, 5, 6, 7, -1])),
        ]
        for family, y in cases:
            with self.subTest(family=family), self.assertRaises(ValueError):
                glm(self.x, y, family=family)

    def test_data_weights_offsets_and_controls_are_validated(self):
        y = np.arange(8.0)
        nonfinite_x = self.x.copy()
        nonfinite_x[0, 0] = np.nan
        invalid_calls = [
            lambda: glm(self.x[:, 0], y),
            lambda: glm(self.x, y[:, None]),
            lambda: glm(nonfinite_x, y),
            lambda: glm(self.x, y, weights=np.ones(7)),
            lambda: glm(self.x, y, weights=np.array([1, 1, 1, 1, 1, 1, 1, -1])),
            lambda: glm(self.x, y, weights=np.zeros(8)),
            lambda: glm(self.x, y, offset=np.ones(7)),
            lambda: glm(self.x, y, max_iter=0),
            lambda: glm(self.x, y, tol=0),
            lambda: glm(self.x, y, link="missing"),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_unconverged_irls_is_not_returned_as_a_fit(self):
        with self.assertRaises(RuntimeError):
            glm(self.x, np.arange(8.0), max_iter=1)

    def test_rank_and_convergence_are_explicit(self):
        x = np.column_stack([np.ones(8), np.arange(8.0), np.arange(8.0)])
        result = glm(x, np.arange(8.0) + 0.1 * np.sin(np.arange(8.0)))
        self.assertTrue(result.converged)
        self.assertEqual(result.rank, 2)

    def test_inverse_gaussian_uses_its_actual_density(self):
        y = np.array([0.7, 1.2, 1.8, 2.5, 3.7, 5.1, 6.9, 9.0])
        result = glm(self.x, y, family="inverse_gaussian")
        expected = np.sum(
            -0.5
            * (
                np.log(2.0 * np.pi * result.dispersion)
                + 3.0 * np.log(y)
                + (y - result.fitted) ** 2 / (result.dispersion * y * result.fitted**2)
            )
        )
        self.assertAlmostEqual(result.log_likelihood, expected)
        self.assertTrue(np.isfinite(result.aic))

    def test_information_criteria_require_an_actual_likelihood(self):
        result = glm(self.x, np.linspace(0.1, 0.9, 8), family="binomial")
        self.assertIsNone(result.log_likelihood)
        with self.assertRaises(ValueError):
            _ = result.aic
        with self.assertRaises(ValueError):
            _ = result.bic


class OtherRegressionContractsTest(unittest.TestCase):
    def setUp(self):
        self.x = np.column_stack([np.ones(20), np.linspace(-1.0, 1.0, 20)])
        self.y = self.x @ np.array([1.0, 2.0]) + 0.05 * np.sin(np.arange(20.0))

    def test_penalized_models_validate_data_and_report_status(self):
        with self.assertRaises(ValueError):
            ridge_regression(self.x, self.y[:-1])
        with self.assertRaises(ValueError):
            ridge_regression(self.x, self.y, alpha=np.nan)
        with self.assertRaises(RuntimeError):
            elastic_net(self.x, self.y, max_iter=1, tol=1e-15)
        result = ridge_regression(self.x, self.y)
        self.assertTrue(result.converged)
        self.assertEqual(result.rank, 1)

    def test_robust_and_quantile_controls_are_validated(self):
        with self.assertRaises(ValueError):
            robust_regression(self.x, self.y, method="unknown")
        with self.assertRaises(ValueError):
            robust_regression(self.x, self.y, c=0)
        contaminated = self.y.copy()
        contaminated[0] += 100
        with self.assertRaises(RuntimeError):
            robust_regression(self.x, contaminated, max_iter=1, tol=1e-15)
        with self.assertRaises(ValueError):
            quantile_regression(self.x, self.y, eps=0)


class OrdinalContractsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(4)
        self.x = rng.normal(size=(80, 2))
        self.y = np.digitize(self.x @ np.array([0.8, -0.4]) + rng.logistic(size=80), [-0.5, 0.7])

    def test_labels_are_not_truncated_or_reindexed(self):
        for bad_y in (
            self.y.astype(float) + 0.25,
            np.where(self.y == 1, 2, self.y),
            self.y - 1,
            np.where(np.arange(self.y.size) == 0, np.nan, self.y),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                ordinal_regression(self.x, bad_y)

    def test_design_link_rank_and_control_are_validated(self):
        with self.assertRaises(ValueError):
            ordinal_regression(self.x, self.y[:-1])
        with self.assertRaises(ValueError):
            ordinal_regression(self.x, self.y, link="cloglog")
        with self.assertRaises(ValueError):
            ordinal_regression(self.x, self.y, max_iter=0)
        with self.assertRaises(ValueError):
            ordinal_regression(np.column_stack([self.x[:, 0], self.x[:, 0]]), self.y)

    def test_fit_reports_convergence_rank_and_safe_prediction(self):
        result = ordinal_regression(self.x, self.y)
        self.assertTrue(result.converged)
        self.assertGreater(result.n_iter, 0)
        self.assertEqual(result.rank, 2)
        with self.assertRaises(ValueError):
            result.predict_proba(np.ones((2, 3)))

    def test_somers_dependent_axis_is_validated(self):
        with self.assertRaises(ValueError):
            somers_d([1, 2, 3], [1, 2, 3], dependent="z")


class MeasurementErrorContractsTest(unittest.TestCase):
    def test_deming_support_and_identifiability_are_validated(self):
        invalid = [
            lambda: deming_regression([1, 2], [1]),
            lambda: deming_regression([[1, 2]], [[1, 2]]),
            lambda: deming_regression([1, np.nan], [1, 2]),
            lambda: deming_regression([1, 2], [1, 2], variance_ratio=0),
            lambda: deming_regression([-1, 0, 1, 0], [0, 10, 0, -10]),
        ]
        for call in invalid:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_simex_validates_design_and_extrapolation(self):
        fit = lambda xx, yy: np.array([np.mean(xx), np.mean(yy)])
        x, y = np.arange(6.0), np.arange(6.0)
        invalid = [
            lambda: simex(fit, x, y[:-1], 1),
            lambda: simex(fit, x, y, -1),
            lambda: simex(fit, x, y, 1, lambdas=np.array([-1.0, 0.0, 1.0])),
            lambda: simex(fit, x, y, 1, lambdas=np.array([0.0, 1.0]), extrapolation="quadratic"),
            lambda: simex(fit, x, y, 1, n_sims=0),
            lambda: simex(fit, x, y, 1, extrapolation="cubic"),
        ]
        for call in invalid:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_simex_rejects_inconsistent_or_nonfinite_estimates(self):
        x, y = np.arange(6.0), np.arange(6.0)
        with self.assertRaises(ValueError):
            simex(lambda xx, yy: np.array([np.nan]), x, y, 1, n_sims=1)

    def test_uncertainty_propagation_validates_draws_levels_and_outputs(self):
        invalid = [
            lambda: propagate_uncertainty(lambda value: value, np.array([[1.0]])),
            lambda: propagate_uncertainty(lambda value: value, np.array([[1.0], [np.nan]])),
            lambda: propagate_uncertainty(lambda value: value, np.array([[1.0], [2.0]]), quantiles=(-0.1,)),
            lambda: propagate_uncertainty(lambda value: np.nan, np.array([[1.0], [2.0]])),
        ]
        for call in invalid:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
