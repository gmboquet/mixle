"""amplify_and_capture: mode-2 capture of an amplified teacher + collapse-monitored round 2 guided by
the captured student (workstream D10, AMPLIFY-a research spike). Proven against the same closed-form
oracle doe_oracle_test.py uses, per the plan's own build order (no domain oracle exists yet)."""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")  # optimize_under_oracle's BayesianOptimizer surrogate needs GaussianProcessRegressor

from mixle.doe.amplify import StudentTeacher, amplify_and_capture, fit_student  # noqa: E402
from mixle.doe.oracle import OracleResult, VerifiableOracle, optimize_under_oracle  # noqa: E402

_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]


def _quadratic_bowl_oracle(target, seed=0):
    def score_fn(x):
        d2 = float(np.sum((np.asarray(x, dtype=float) - target) ** 2))
        return OracleResult(score=-d2, receipt={"target_dist2": d2}, cost=1.0)

    return VerifiableOracle(name="quadratic_bowl", tier="executable", score_fn=score_fn, fidelity="exact, noiseless")


class AmplifyAndCaptureTest(unittest.TestCase):
    def test_round1_beats_a_single_ungrounded_guess(self):
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        self.assertFalse(report.stopped_early)
        self.assertTrue(report.beats_single_input)
        self.assertGreater(report.round1.best_score, report.baseline_single_input_score)

    def test_student_is_captured_only_from_oracle_verified_history(self):
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        run1 = optimize_under_oracle(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)
        student = fit_student(run1, degree=2)

        self.assertIsInstance(student, StudentTeacher)
        # the student is directly callable, candidate -> predicted score, like any other teacher here
        prediction = student(np.array([2.0, -1.0]))
        self.assertIsInstance(prediction, float)
        # a point near the true optimum should predict a higher score than a point far from it
        far_prediction = student(np.array([-5.0, 5.0]))
        self.assertGreater(prediction, far_prediction)

    def test_round2_at_matched_budget_beats_or_matches_round1(self):
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        self.assertIsNotNone(report.round2)
        self.assertEqual(len(report.round2.run.history), len(report.round1.run.history))  # matched budget
        self.assertTrue(report.round2_beats_round1)
        self.assertGreaterEqual(report.round2.best_score, report.round1.best_score)

    def test_collapse_monitor_is_run_and_reused_not_reimplemented(self):
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        self.assertIsNotNone(report.collapse)
        self.assertTrue(report.collapse.ok)
        self.assertEqual(report.collapse.scores, [report.round1.best_score, report.round2.best_score])

    def test_no_student_self_grade_ever_enters_round2_history(self):
        """The load-bearing assertion: every OracleResult in round 2's history came from the REAL
        oracle's score_fn, never from the student's prediction."""
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        for candidate in report.round2.run.history:
            expected = oracle(candidate.x)
            self.assertAlmostEqual(candidate.result.score, expected.score, places=9)
            self.assertEqual(candidate.result.receipt, expected.receipt)

    def test_a_search_that_cannot_beat_a_single_guess_stops_honestly(self):
        # a constant-score oracle: no search can ever beat a single guess -- nothing to capture
        constant_oracle = VerifiableOracle(
            name="constant", tier="executable", score_fn=lambda x: OracleResult(score=0.0), fidelity="degenerate"
        )
        report = amplify_and_capture(constant_oracle, _BOUNDS, n_init=3, n_iter=3, seed=0)

        self.assertTrue(report.stopped_early)
        self.assertFalse(report.beats_single_input)
        self.assertIsNone(report.round2)
        self.assertIsNone(report.student)
        self.assertIn("nothing to capture", report.reason)


if __name__ == "__main__":
    unittest.main()
