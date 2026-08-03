"""Tests for linear-Gaussian state-space models (Kalman/RTS + EM)."""

import unittest
from unittest import mock

import numpy as np

from mixle.ppl import AR1, LocalLevel
from mixle.ppl import statespace as statespace_module
from mixle.ppl.statespace import StateSpaceResult, _kalman_em, _kalman_smooth


class _DownhillSmoother:
    """A stand-in E-step whose objective walks DOWN by a fixed amount every call.

    EM's own E/M steps are non-decreasing, so a downhill objective has to be injected to exercise
    the stopping rule against one (MXR-080-1897). The returned path is a legal smoothing result
    (finite, non-empty, non-negative variances), so only the objective's direction is unusual.
    """

    def __init__(self, start: float, step: float):
        self.value = float(start)
        self.step = float(step)

    def __call__(self, y, phi, q, r, x0, P0):
        size = np.asarray(y, dtype=float).size
        loglik = self.value
        self.value -= self.step
        return np.zeros(size), np.ones(size), np.zeros(size), loglik


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


class StateSpaceStoppingSemanticsTest(unittest.TestCase):
    """MXR-080-1897: EM used ``abs(next_ll - ll) < tol``, so a DECREASE counted as convergence."""

    def test_em_refuses_to_call_a_material_objective_decrease_convergence(self):
        # Reproduction: an objective that walks down by 0.5 a step. The old rule's own test,
        # abs(change) < tol, is satisfied on the very first iteration for any tol > 0.5, so it
        # returned converged=True / "objective_tolerance" for a fit that got strictly worse.
        smoother = _DownhillSmoother(start=-10.0, step=0.5)
        with mock.patch.object(statespace_module, "_kalman_smooth", smoother):
            with self.assertWarnsRegex(RuntimeWarning, "objective decreased"):
                result = _kalman_em([0.0, 1.0, 2.0], True, 4, 1.0)

        change = result.objective_trace[1] - result.objective_trace[0]
        self.assertEqual(change, -0.5)
        self.assertLess(abs(change), 1.0)  # exactly what the old absolute rule accepted as converged
        self.assertFalse(result.converged)
        self.assertEqual(result.termination_reason, "iteration_limit")
        self.assertEqual(result.iterations, 4)
        # disclosed on the receipt rather than swallowed
        self.assertFalse(result.monotone)
        self.assertEqual(result.max_objective_decrease, -0.5)
        self.assertFalse(result.termination["monotone"])
        self.assertEqual(result.summary()["max_objective_decrease"], -0.5)

    def test_em_still_converges_on_a_roundoff_scale_decrease(self):
        # The other half of the same rule, and the reason a decrease is not simply raised on: a
        # plateaued EM jitters downward by float roundoff. Measured over ~14,000 real AR1/LocalLevel
        # fits the worst step decrease was 1.4e-15 relative to |loglik|, so a decrease that small is
        # convergence, not a defect, and must keep converging. (Sweeping 3,200 real fits across four
        # tolerances, the new rule reproduced the old rule's stop iteration every single time.)
        smoother = _DownhillSmoother(start=-1000.0, step=1e-13)  # 1e-16 relative: inside the slack
        with mock.patch.object(statespace_module, "_kalman_smooth", smoother):
            result = _kalman_em([0.0, 1.0, 2.0], True, 20, 1e-6)

        self.assertLess(result.objective_trace[1], result.objective_trace[0])
        self.assertTrue(result.converged)
        self.assertEqual(result.termination_reason, "objective_tolerance")
        self.assertEqual(result.iterations, 1)
        self.assertTrue(result.monotone)  # within the roundoff allowance
        self.assertLess(result.max_objective_decrease, 0.0)  # still recorded verbatim

    def test_real_fits_report_a_monotone_objective(self):
        # The claim the stopping rule now makes, checked against real EM rather than a stand-in.
        rng = np.random.RandomState(5)
        for tag, y in (
            ("random_walk", np.cumsum(rng.normal(0, 0.3, 120)) + rng.normal(0, 0.5, 120)),
            ("white_noise", rng.normal(0, 1.0, 40)),
            ("near_constant", np.full(30, 2.5) + rng.normal(0, 1e-7, 30)),
        ):
            for phi_free in (True, False):
                with self.subTest(series=tag, phi_free=phi_free):
                    result = _kalman_em(y, phi_free, 200, 0.0)
                    self.assertTrue(result.monotone)
                    self.assertGreaterEqual(result.max_objective_decrease, -1e-9 * max(1.0, abs(result.loglik)))


class StateSpaceResultReceiptTest(unittest.TestCase):
    """MXR-080-1897: the result coerced constructor fields, aliased arrays, and let the reported
    log likelihood float free of the trace and verdict it came from."""

    @staticmethod
    def _receipt(**overrides):
        """A minimal internally consistent receipt, with the field under test overridden."""
        fields = {
            "phi": 0.5,
            "q": 1.0,
            "r": 1.0,
            "x0": 0.0,
            "P0": 1.0,
            "smoothed": [0.0, 1.0],
            "smoothed_var": [1.0, 1.0],
            "loglik": -3.0,
            "converged": True,
            "iterations": 1,
            "objective_trace": [-4.0, -3.0],
            "termination_reason": "objective_tolerance",
        }
        fields.update(overrides)
        return fields

    def test_result_arrays_are_copied_and_sealed(self):
        # Reproduction: np.asarray returns the caller's own float64 array, so every check in the
        # constructor described an array the caller could rewrite one statement later.
        path = np.array([0.0, 1.0, 2.0])
        variance = np.array([1.0, 1.0, 1.0])
        result = StateSpaceResult(**self._receipt(smoothed=path, smoothed_var=variance))

        self.assertIsNot(result.smoothed, path)
        path[0] = 999.0  # used to show up as result.smoothed[0] == 999.0
        self.assertEqual(result.smoothed[0], 0.0)
        with self.assertRaises(ValueError):
            result.smoothed[0] = np.nan
        with self.assertRaises(ValueError):
            result.smoothed_sd[0] = -5.0

    def test_fitted_result_geometry_cannot_be_rewritten_through_the_result(self):
        result = AR1().fit([0.1, -0.2, 0.3, 0.0], max_its=2, delta=0.0).result
        with self.assertRaises(ValueError):
            result.smoothed[0] = np.nan  # used to succeed, leaving a validated path holding NaN
        with self.assertRaises(ValueError):
            result.smoothed_sd[0] = -5.0
        self.assertTrue(np.all(np.isfinite(result.smoothed)))

    def test_reported_loglik_must_be_the_objective_the_trace_ended_on(self):
        # Reproduction: loglik=1e9 alongside a trace ending at -3.0 was accepted, and summary(),
        # termination["objective"] and every downstream reader repeated the 1e9.
        with self.assertRaisesRegex(ValueError, "not the objective the fit ended on"):
            StateSpaceResult(**self._receipt(loglik=1e9))

    def test_verdict_must_match_the_named_stopping_rule(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            StateSpaceResult(**self._receipt(termination_reason="iteration_limit"))
        with self.assertRaisesRegex(ValueError, "not a declared state-space stopping rule"):
            StateSpaceResult(**self._receipt(termination_reason="looks_fine"))

    def test_constructor_fields_are_not_coerced(self):
        # Reproduction, all previously accepted: phi="0.5" parsed to 0.5, converged="no" became
        # True (bool("no") is True), iterations=1.9 truncated to 1, and str(object()) became a
        # termination reason.
        for label, override, error in (
            ("phi as str", {"phi": "0.5"}, TypeError),
            ("q as bool", {"q": True}, TypeError),
            ("q non-positive", {"q": 0.0}, ValueError),
            ("converged as str", {"converged": "no"}, TypeError),
            ("iterations as float", {"iterations": 1.9}, TypeError),
            ("reason as object", {"termination_reason": object()}, TypeError),
        ):
            with self.subTest(field=label), self.assertRaises(error):
                StateSpaceResult(**self._receipt(**override))


if __name__ == "__main__":
    unittest.main()
