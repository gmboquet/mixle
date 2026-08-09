"""Contracts separating forecast/selection guarantees and exact empirical tail risk."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mixle.inference.price_forecast import forecast_price
from mixle.inference.risk import conditional_value_at_risk
from mixle.inference.select import select_best


class PriceForecastCalibrationContractTest(unittest.TestCase):
    def test_each_forecast_lead_uses_residuals_from_that_same_lead(self):
        history = np.arange(20.0) ** 2
        captured = []

        def fake_forecast(model, observed, horizon, level, seed, keep_samples):
            samples = np.zeros((horizon, 5)) if keep_samples else None
            return SimpleNamespace(mean=np.arange(1.0, horizon + 1.0), samples=samples)

        def fake_split(cal_pred, cal_y, test_pred, **kwargs):
            captured.append((np.asarray(cal_pred), np.asarray(cal_y), np.asarray(test_pred)))
            return test_pred - 1.0, test_pred + 1.0

        with (
            patch("mixle.inference.price_forecast.forecast", side_effect=fake_forecast),
            patch("mixle.inference.price_forecast.split_conformal", side_effect=fake_split),
        ):
            result = forecast_price(object(), history, horizon=2, cal_frac=0.5, model_fit_length=0)

        self.assertEqual(len(captured), 2)
        origins = np.arange(10, 18)
        np.testing.assert_array_equal(captured[0][1], history[origins])
        np.testing.assert_array_equal(captured[1][1], history[origins + 1])
        self.assertEqual(result.calibration_count, 8)
        self.assertEqual(result.interval_method, "horizon_matched_split_conformal")
        self.assertTrue(any("held_out_exchangeability" in a for a in result.coverage_assumptions))

    def test_forecast_controls_and_history_shape_are_exact(self):
        with self.assertRaises(TypeError):
            forecast_price(object(), [1.0, 2.0], horizon=1.5, model_fit_length=0)
        with self.assertRaises(ValueError):
            forecast_price(object(), [[1.0], [2.0]], horizon=1, model_fit_length=0)


class SelectionGuaranteeContractTest(unittest.TestCase):
    def test_score_spread_flag_is_explicitly_heuristic(self):
        result = select_best([1.0, 2.0, 10.0], score=float, heuristic_alpha=0.1)
        self.assertEqual(result.confidence_method, "normal_score_spread_heuristic")
        self.assertFalse(result.coverage_guarantee)

    def test_nonfinite_verifier_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            select_best([1.0, 2.0], score=lambda value: np.nan if value == 2.0 else value)


class ExpectedShortfallContractTest(unittest.TestCase):
    def test_var_boundary_ties_receive_only_the_required_fractional_mass(self):
        outcomes = -np.array([10.0, 5.0, 5.0, 5.0, 0.0])
        self.assertAlmostEqual(
            conditional_value_at_risk(outcomes, alpha=0.6, min_tail=1),
            7.5,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
