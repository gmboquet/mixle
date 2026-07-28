"""Tests for constrained Bayesian optimization (WS-E).

``probability_of_feasibility`` is torch-free and tested directly; the GP-surrogate
``propose_next_constrained`` / ``constrained_minimize`` paths require torch and are skipped without it.
"""

import importlib.util
import unittest
from unittest import mock

import numpy as np

from mixle.doe import constrained as constrained_module
from mixle.doe import (
    constrained_minimize,
    probability_of_feasibility,
    propose_next_constrained,
)
from mixle.doe.constrained import _best_feasible, _predict_std

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ProbabilityOfFeasibilityTest(unittest.TestCase):
    def test_single_constraint_monotone_in_mean(self):
        # Lower predicted constraint value -> more likely feasible (c <= 0).
        pf = probability_of_feasibility(mean=np.array([[-2.0], [0.0], [2.0]]), std=np.ones((3, 1)))
        self.assertTrue(np.all((pf >= 0.0) & (pf <= 1.0)))
        self.assertGreater(pf[0], pf[1])
        self.assertGreater(pf[1], pf[2])
        np.testing.assert_allclose(pf[1], 0.5, atol=1e-9)  # mean exactly on the boundary

    def test_zero_std_is_deterministic(self):
        pf = probability_of_feasibility(mean=np.array([[-1.0], [1.0]]), std=np.zeros((2, 1)))
        np.testing.assert_array_equal(pf, np.array([1.0, 0.0]))

    def test_multiple_constraints_multiply(self):
        # Two independent constraints each at the boundary -> 0.5 * 0.5 = 0.25.
        pf = probability_of_feasibility(mean=np.zeros((1, 2)), std=np.ones((1, 2)))
        np.testing.assert_allclose(pf, [0.25], atol=1e-9)

    def test_one_dimensional_input_is_n_points_not_n_constraints(self):
        # Regression test for MXR-080-0173: a length-N mean/std used to be reshaped to (1, N) via
        # np.atleast_2d -- one point under N constraints -- silently returning a single joint
        # probability (0.13348 for this exact input) instead of N per-point probabilities under one
        # constraint. It must now be reshaped to (N, 1): N points, one constraint each.
        pf = probability_of_feasibility(mean=[-1.0, 1.0], std=[1.0, 1.0])
        self.assertEqual(pf.shape, (2,))
        # Phi(1) for the point with mean=-1 (comfortably feasible), Phi(-1) for mean=+1 (unlikely).
        np.testing.assert_allclose(pf, [0.8413447460685429, 0.15865525393145707], atol=1e-9)

    def test_two_dimensional_multi_point_multi_constraint_unaffected(self):
        # Negative control: a genuine (n_points, n_constraints) array with BOTH n_points > 1 and
        # n_constraints > 1 must keep computing correctly -- the 1-D reinterpretation fix must not
        # disturb already-2-D input.
        mean = np.array([[-1.0, 0.0], [1.0, -2.0]])
        std = np.ones((2, 2))
        pf = probability_of_feasibility(mean, std)
        # point 0: Phi(1) * Phi(0); point 1: Phi(-1) * Phi(2).
        np.testing.assert_allclose(pf, [0.42067237303427146, 0.15504582597024455], atol=1e-9)

    def test_mismatched_mean_std_shapes_raise(self):
        with self.assertRaises(ValueError):
            probability_of_feasibility(mean=np.zeros((3, 1)), std=np.ones((4, 1)))
        with self.assertRaises(ValueError):
            probability_of_feasibility(mean=np.zeros((3, 2)), std=np.ones((3, 1)))

    def test_negative_std_raises(self):
        with self.assertRaises(ValueError):
            probability_of_feasibility(mean=np.zeros((2, 1)), std=np.array([[1.0], [-0.1]]))

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            probability_of_feasibility(mean=np.array([[np.nan], [0.0]]), std=np.ones((2, 1)))
        with self.assertRaises(ValueError):
            probability_of_feasibility(mean=np.zeros((2, 1)), std=np.array([[1.0], [np.nan]]))

    def test_zero_constraint_columns_raise(self):
        with self.assertRaises(ValueError):
            probability_of_feasibility(mean=np.zeros((3, 0)), std=np.zeros((3, 0)))


class _StubGP:
    def __init__(self, mean=None, covariance=None):
        self.mean = mean
        self.covariance = covariance

    def predict(self, x, y, points, return_cov=True):
        n = len(points)
        mean = np.zeros(n) if self.mean is None else self.mean
        covariance = np.eye(n) if self.covariance is None else self.covariance
        return mean, covariance


class ConstrainedEvidenceContractTest(unittest.TestCase):
    def test_predict_std_rejects_indefinite_posterior(self):
        covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            _predict_std(_StubGP(covariance=covariance), np.zeros((1, 1)), np.zeros(1), np.zeros((2, 1)))

    def test_natural_one_constraint_vector_is_accepted_by_proposal(self):
        with mock.patch.object(constrained_module, "_fit_surrogate", return_value=_StubGP()):
            point = propose_next_constrained(
                np.array([[0.0], [1.0]]),
                np.array([0.0, 1.0]),
                np.array([-1.0, -1.0]),
                [(0.0, 1.0)],
                n_candidates=4,
                seed=0,
            )
        self.assertEqual(point.shape, (1,))

    def test_acquisition_requires_one_finite_score_per_candidate(self):
        arguments = (
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([-1.0, -1.0]),
            [(0.0, 1.0)],
        )
        acquisitions = (
            lambda mean, std, best, **kwargs: np.zeros((len(mean), 1)),
            lambda mean, std, best, **kwargs: np.full(len(mean), np.nan),
        )
        for acquisition in acquisitions:
            with mock.patch.object(constrained_module, "_fit_surrogate", return_value=_StubGP()):
                with self.assertRaises(ValueError):
                    propose_next_constrained(*arguments, n_candidates=4, acq=acquisition, seed=0)

    def test_nonfinite_constraint_evidence_cannot_select_an_incumbent(self):
        with self.assertRaisesRegex(ValueError, "constraint observations"):
            _best_feasible(np.array([0.0, 1.0]), np.array([[np.nan], [1.0]]))

    def test_invalid_iteration_and_candidate_budgets_are_rejected(self):
        for invalid in (-2, 0.9, True, np.bool_(True)):
            with self.assertRaises((TypeError, ValueError)):
                constrained_minimize(
                    lambda point: float(point[0]),
                    [lambda point: float(point[0])],
                    [(0.0, 1.0)],
                    n_init=1,
                    n_iter=invalid,
                )
        for invalid in (0, 2.5, True, np.bool_(True)):
            with self.assertRaises((TypeError, ValueError)):
                constrained_minimize(
                    lambda point: float(point[0]),
                    [lambda point: float(point[0])],
                    [(0.0, 1.0)],
                    n_init=1,
                    n_iter=0,
                    n_candidates=invalid,
                )

    def test_nonfinite_constraint_call_is_separate_failed_evidence(self):
        result = constrained_minimize(
            lambda point: float(point[0]),
            [lambda _point: np.nan],
            [(0.0, 1.0)],
            n_init=1,
            n_iter=0,
            seed=0,
        )
        self.assertEqual(result.x.shape, (0, 1))
        self.assertEqual(result.y.shape, (0,))
        self.assertEqual(result.c.shape, (0, 1))
        self.assertIsNone(result.best_x)
        self.assertEqual(result.n_evaluations, 1)
        self.assertEqual(len(result.failed_evaluations), 1)
        self.assertEqual(result.stopped_reason, "objective_or_constraint_failed")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ConstrainedLoopTest(unittest.TestCase):
    def test_propose_next_constrained_in_bounds(self):
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform(-2.0, 2.0, size=(8, 2))
        y = np.sum(x**2, axis=1)
        c = (x[:, 0] + x[:, 1] - 1.0).reshape(-1, 1)  # constraint: x0 + x1 <= 1
        nxt = np.asarray(
            propose_next_constrained(x, y, c, bounds, n_candidates=128, seed=1, fit_kwargs={"max_its": 60})
        )
        self.assertEqual(nxt.shape, (2,))
        self.assertTrue(np.all(nxt >= -2.0) and np.all(nxt <= 2.0))

    def test_mismatched_constraint_rows_raise(self):
        with self.assertRaises(ValueError):
            propose_next_constrained(
                np.zeros((4, 2)), np.zeros(4), np.zeros((3, 1)), [(0.0, 1.0), (0.0, 1.0)], n_candidates=8
            )

    def test_constrained_minimum_respects_an_active_constraint(self):
        # Minimize (x-2)^2 subject to x <= 0: unconstrained optimum is x=2 (infeasible);
        # the constrained optimum sits at the boundary x=0.
        bounds = [(-3.0, 3.0)]

        def objective(p):
            return float((p[0] - 2.0) ** 2)

        def constraint(p):
            return float(p[0])  # feasible when x <= 0

        result = constrained_minimize(
            objective,
            [constraint],
            bounds,
            n_init=6,
            n_iter=20,
            seed=0,
            n_candidates=256,
            fit_kwargs={"max_its": 60},
        )
        self.assertEqual(result.c.shape, (26, 1))
        self.assertEqual(result.feasible.shape, (26,))
        self.assertTrue(np.any(result.feasible))  # found feasible points
        self.assertLessEqual(result.best_x[0], 1e-6)  # best feasible point honors x <= 0
        self.assertLess(result.best_y, 4.5)  # and beats a far-from-boundary feasible value like x=-1 -> 9

    def test_requires_at_least_one_constraint(self):
        with self.assertRaises(ValueError):
            constrained_minimize(lambda p: float(p[0]), [], [(0.0, 1.0)], n_init=3, n_iter=1)


if __name__ == "__main__":
    unittest.main()
