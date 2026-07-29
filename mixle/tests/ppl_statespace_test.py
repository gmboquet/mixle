"""Tests for linear-Gaussian state-space models (Kalman/RTS + EM)."""

import unittest

import numpy as np

from mixle.ppl import AR1, LocalLevel
from mixle.ppl.statespace import _kalman_smooth


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
        self.assertEqual(
            set(m.params),
            {"phi", "level_sd", "obs_sd", "initial_mean", "initial_sd"},
        )

    def test_returned_moments_match_returned_parameters(self):
        rng = np.random.RandomState(2)
        T = 80
        x = np.zeros(T)
        for t in range(1, T):
            x[t] = 0.65 * x[t - 1] + rng.normal(0, 0.5)
        y = x + rng.normal(0, 0.4, T)

        m = AR1().fit(list(y), max_its=4, tol=0.0)
        res = m.result
        xs, ps, _, ll = _kalman_smooth(
            y,
            res.phi,
            res.level_sd**2,
            res.obs_sd**2,
            res.initial_mean,
            res.initial_sd**2,
        )

        self.assertAlmostEqual(res.loglik, ll, places=10)
        self.assertTrue(np.allclose(res.smoothed, xs))
        self.assertTrue(np.allclose(res.smoothed_sd**2, ps))

    def test_fit_controls_and_termination_receipt(self):
        y = [0.1, -0.2, 0.3, 0.0]
        for kwargs, error in [
            ({"max_its": 0}, ValueError),
            ({"max_its": 1.5}, TypeError),
            ({"delta": -1.0}, ValueError),
            ({"delta": np.nan}, ValueError),
            ({"tol": np.inf}, ValueError),
        ]:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(error):
                AR1().fit(y, **kwargs)

        limited = AR1().fit(y, max_its=2, delta=0.0).result
        self.assertFalse(limited.converged)
        self.assertEqual(limited.iterations, 2)
        self.assertEqual(limited.termination_reason, "iteration_limit")
        self.assertEqual(len(limited.objective_trace), 3)
        self.assertEqual(limited.loglik, limited.objective_trace[-1])
        self.assertEqual(limited.termination["objective"], limited.loglik)

        converged = AR1().fit(y, max_its=2, delta=1e9).result
        self.assertTrue(converged.converged)
        self.assertEqual(converged.iterations, 1)
        self.assertEqual(converged.termination_reason, "objective_tolerance")

    def test_input_and_missing_data_policy(self):
        with self.assertRaises(ValueError):
            AR1().fit([0.0, np.nan, 1.0])
        with self.assertRaises(ValueError):
            AR1().fit([0.0, np.inf, 1.0], missing="marginalize")
        with self.assertRaises(ValueError):
            AR1().fit([[0.0], [1.0]])
        with self.assertRaises(ValueError):
            AR1().fit([np.nan, np.nan], missing="marginalize")

        result = (
            AR1()
            .fit(
                [np.nan, 0.0, 0.2, np.nan, -0.1],
                missing="marginalize",
                max_its=2,
                delta=0.0,
            )
            .result
        )
        self.assertEqual(result.smoothed.shape, (5,))
        self.assertTrue(np.all(np.isfinite(result.smoothed)))

    def test_forecast_requires_a_positive_integer_horizon(self):
        result = LocalLevel().fit([0.0, 0.1], max_its=1, delta=0.0).result
        for horizon, error in [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)]:
            with self.subTest(horizon=repr(horizon)), self.assertRaises(error):
                result.forecast(horizon)


if __name__ == "__main__":
    unittest.main()
