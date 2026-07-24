"""amplify_and_capture: mode-2 capture of an amplified teacher + collapse-monitored round 2 guided by
the captured student (workstream D10, AMPLIFY-a research spike). Proven against the same closed-form
oracle doe_oracle_test.py uses, per the plan's own build order (no domain oracle exists yet).

MXR-080-0161 regression coverage: the original "beats a single input" gate compared round 1's
best-of-budget against ONE extra random draw -- a multiple-comparisons trap in which passage became
increasingly automatic as the budget grew even for a search with provably zero real skill (empirically
confirmed against the unfixed code: pass rate for a search with i.i.d.-noise, x-independent scores tracks
the exact closed form N / (N+1) for max-of-N-iid vs one more iid draw, e.g. ~0.92 at budget 12), and the
extra baseline draw was never counted against the stated budget. Fixed here with a budget-matched random
baseline plus a one-sided permutation test on best-of-budget; see mixle/doe/amplify.py's docstrings for
the full statistical reasoning.
"""

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


def _zero_skill_oracle(replicate_seed):
    """An oracle whose score is i.i.d. noise, independent of ``x``: no proposal strategy, however
    guided, can do better than chance against it. A search run against this oracle has, by
    construction, exactly zero real skill -- any 'pass' of the beats_baseline gate is attributable
    purely to how the comparison is structured, which is exactly MXR-080-0161's null hypothesis."""
    rng = np.random.default_rng(replicate_seed)

    def score_fn(x):
        return OracleResult(score=float(rng.normal()), receipt={}, cost=1.0)

    return VerifiableOracle(
        name="zero_skill_noise", tier="executable", score_fn=score_fn, fidelity="pure noise, independent of x"
    )


class AmplifyAndCaptureTest(unittest.TestCase):
    def test_round1_beats_a_budget_matched_random_baseline(self):
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        self.assertFalse(report.stopped_early)
        self.assertTrue(report.beats_baseline)
        self.assertGreater(report.round1.best_score, report.baseline.best_score)
        self.assertLess(report.baseline_p_value, report.significance)

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

    def test_baseline_oracle_call_budget_matches_round1s_not_a_single_extra_draw(self):
        """MXR-080-0161 budget-accounting regression: the baseline used to be exactly one draw,
        uncounted against the stated n_init + n_iter budget. It must now spend exactly as many
        oracle calls as round 1 did, each receipted in its own DesignRun."""
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        self.assertEqual(report.baseline.run.oracle_calls, 12)
        self.assertEqual(report.baseline.run.oracle_calls, report.round1.run.oracle_calls)

    def test_skilled_search_reliably_clears_the_fixed_gate(self):
        """Negative control for 0161's fix: a genuinely skilled search (real oracle feedback driving
        Bayesian optimization on an easy, well-specified landscape) should still reliably clear the
        fixed, budget-matched, permutation-tested gate -- the fix must not be so conservative that it
        also rejects real search skill. (Empirically ~19/20 at this budget; 7/10 leaves comfortable
        margin against any single-run variance while still being far above the 5% nominal rate chance
        alone would produce.)"""
        n_seeds = 10
        passes = 0
        for seed in range(n_seeds):
            oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
            report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=seed)
            passes += int(report.beats_baseline)
        self.assertGreaterEqual(passes, 7, f"only {passes}/{n_seeds} skilled-search seeds cleared the gate")

    def test_beats_baseline_false_positive_rate_matches_nominal_significance_not_inflated(self):
        """MXR-080-0161 Monte Carlo regression. Against the unfixed code, a search with PROVABLY zero
        real skill (scores i.i.d. noise, independent of x) passed the old gate increasingly often as
        the budget grew, matching the exact multiple-comparisons closed form for N iid draws vs one
        more: P(max of N > one more) = N / (N+1) -- confirmed empirically (~0.92 at budget 12, ~0.94
        at budget 15). Against the same zero-skill oracle, the FIXED gate's pass rate must instead
        track its nominal significance level, not run away toward 1 as the budget grows."""
        significance = 0.1
        n_replicates = 120
        budget = 8  # n_init=8, n_iter=0: old gate's exact null pass rate here would be 8/9 ~ 0.889
        passes = 0
        for i in range(n_replicates):
            oracle = _zero_skill_oracle(replicate_seed=30_000 + i)
            report = amplify_and_capture(oracle, _BOUNDS, n_init=budget, n_iter=0, significance=significance, seed=i)
            passes += int(report.beats_baseline)
        pass_rate = passes / n_replicates
        self.assertLess(
            pass_rate,
            0.3,
            f"pass rate {pass_rate:.3f} looks multiple-comparisons-inflated for nominal {significance} "
            f"(the old, unmatched-budget gate's exact null rate at this budget is {budget / (budget + 1):.3f})",
        )

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

    def test_a_search_that_cannot_beat_a_budget_matched_baseline_stops_honestly(self):
        # a constant-score oracle: no search can ever beat a fair baseline -- nothing to capture
        constant_oracle = VerifiableOracle(
            name="constant", tier="executable", score_fn=lambda x: OracleResult(score=0.0), fidelity="degenerate"
        )
        report = amplify_and_capture(constant_oracle, _BOUNDS, n_init=3, n_iter=3, seed=0)

        self.assertTrue(report.stopped_early)
        self.assertFalse(report.beats_baseline)
        self.assertIsNone(report.round2)
        self.assertIsNone(report.student)
        self.assertIn("nothing to capture", report.reason)


if __name__ == "__main__":
    unittest.main()
