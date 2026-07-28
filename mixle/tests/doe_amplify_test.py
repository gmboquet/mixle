"""amplify_and_capture: mode-2 capture of an amplified teacher + collapse-monitored round 2 guided by
the captured student (workstream D10, AMPLIFY-a research spike). Proven against the same closed-form
oracle doe_oracle_test.py uses, per the plan's own build order (no domain oracle exists yet).

MXR-080-0161/0162 regression coverage: the original "beats a single input" gate compared round 1's
best-of-budget against ONE extra random draw -- a multiple-comparisons trap in which passage became
increasingly automatic as the budget grew even for a search with provably zero real skill, and the extra
baseline draw was never counted against the stated budget. And round 2's candidate pool used to include
round 1's own oracle-verified points, spending round 2's budget re-verifying them and tautologically
guaranteeing round2_beats_round1 (confirmed against the unfixed code: 100/100 True, 100/100 pool
contamination, seed 0's round1-best point literally re-selected into round2 with an identical score).
Both are fixed in mixle/doe/amplify.py: a budget-matched random baseline plus a one-sided permutation
test on best-of-budget for the first, and a round-2 pool restricted to fresh candidates for the second.
See that module's docstrings for the full statistical reasoning.

MXR-080-0189 residual-gap coverage (AmplifyAbstentionHandlingTest): oracle.py's own fix made
optimize_under_oracle skip an abstained (e.g. timed-out) call's -inf placeholder when fitting its GP, but
two consumers downstream of that fix, both inside this module, still read raw scores unfiltered --
the permutation test (fed run1.scores()/baseline_run.scores() directly) and fit_student (fit against
run.history directly). Neither is exercised by the 0161/0162 tests above, since none of their oracles set
a timeout. Confirmed pre-fix: an abstained call's -inf measurably shifted the permutation test's p-value,
and a single abstained candidate anywhere in fit_student's input history drove its entire captured-student
coefficient vector to nan.
"""

import time
import unittest
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("torch")  # optimize_under_oracle's BayesianOptimizer surrogate needs GaussianProcessRegressor

import mixle.doe.amplify as amplify_module  # noqa: E402
from mixle.doe.amplify import StudentTeacher, amplify_and_capture, fit_student  # noqa: E402
from mixle.doe.oracle import (  # noqa: E402
    DesignCandidate,
    DesignRun,
    OracleResult,
    VerifiableOracle,
    optimize_under_oracle,
)

_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]


def _quadratic_bowl_oracle(target, seed=0):
    def score_fn(x):
        d2 = float(np.sum((np.asarray(x, dtype=float) - target) ** 2))
        return OracleResult(score=-d2, receipt={"target_dist2": d2}, cost=1.0)

    return VerifiableOracle(name="quadratic_bowl", tier="executable", score_fn=score_fn, fidelity="exact, noiseless")


def _flaky_quadratic_bowl_oracle(target, hang_on_calls, timeout=0.1):
    """Like ``_quadratic_bowl_oracle``, but the given 1-indexed oracle-call numbers hang past
    ``timeout`` and so abstain deterministically (matching doe_oracle_test.py's counter-based
    ``every_third_call_hangs`` pattern) -- a call COUNT is deterministic across calls to this oracle even
    though BayesianOptimizer's proposed x values are not perfectly reproducible from seed alone across
    process contexts (see test_round2_pool_excludes_round1s_own_points above)."""
    calls = {"n": 0}

    def score_fn(x):
        calls["n"] += 1
        if calls["n"] in hang_on_calls:
            time.sleep(2.0)
        d2 = float(np.sum((np.asarray(x, dtype=float) - target) ** 2))
        return OracleResult(score=-d2, receipt={"target_dist2": d2}, cost=1.0)

    return VerifiableOracle(name="flaky_bowl", tier="executable", score_fn=score_fn, timeout=timeout)


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

    def test_round2_pool_excludes_round1s_own_points(self):
        """MXR-080-0162 structural regression: round 1's points must never appear in round 2's
        oracle-verified candidates. Against the unfixed code, at this exact budget (n_init=4/n_iter=8,
        a real Bayesian-optimization search that converges close enough to the target that the student
        ranks round 1's own points highly), this was violated in 40/40 sampled seeds -- round 1's best
        point was re-selected into round 2's history with an identical (re-verified,
        deterministic-oracle) score every time. Bayesian optimization's proposal step is not perfectly
        reproducible from ``seed`` alone across process contexts (a pre-existing property of the
        surrogate this module builds on, unrelated to this fix), so this checks every seed that
        actually reaches round 2 rather than asserting on one -- the disjointness must hold for all of
        them, and at least one must reach round 2 or the check below is vacuous."""
        saw_round2 = False
        for seed in range(6):
            oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
            report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=seed)
            if report.round2 is None:
                continue
            saw_round2 = True
            self.assertEqual(len(report.round2.run.history), len(report.round1.run.history))  # matched budget
            round1_xs = {tuple(np.round(x, 12)) for x in report.round1.xs}
            round2_xs = {tuple(np.round(x, 12)) for x in report.round2.xs}
            self.assertEqual(round1_xs & round2_xs, set(), f"seed {seed}: round 2 re-offered a round 1 point")
        self.assertTrue(saw_round2, "no seed in range(6) reached round 2; the check above never ran")

    def test_round2_beats_round1_is_no_longer_tautological_for_a_misspecified_student(self):
        """MXR-080-0162 non-circularity regression. Against the unfixed code, round2_beats_round1 was
        True 100/100 sampled runs -- tautological, since round 1's own best point was always present
        (and re-verified) in round 2's pool, floor-ing round2.best_score at round1.best_score by
        construction. With round 1's points excluded, round 2 must stand on its own: a student that
        cannot represent the oracle's true shape (a straight line fit to a curved bowl -- textbook
        underfitting, not a numerical edge case) generalizes badly to the fresh candidate pool, and
        round 2 can now legitimately score worse than round 1. n_init=12/n_iter=0/seed=7 is a fully
        deterministic case (no Bayesian-optimization/GP step at n_iter=0, so nothing here depends on
        process call history): round 1 clears the baseline gate, making round 2 eligible to run at
        all, which is what makes this a real demonstration of round 2 losing, not a case that never
        got the chance to win."""
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=12, n_iter=0, degree=1, seed=7)

        self.assertTrue(report.beats_baseline)
        self.assertIsNotNone(report.round2)
        self.assertFalse(report.round2_beats_round1)
        self.assertLess(report.round2.best_score, report.round1.best_score)
        self.assertFalse(report.collapse.ok)
        self.assertEqual(report.collapse.reason, "score_decreased")
        # the retained incumbent is bookkeeping, separate from the (honestly failing) round-2 claim
        self.assertEqual(report.incumbent_best_score, report.round1.best_score)

    def test_genuinely_improving_students_round2_beats_round1_still_reports_true(self):
        """Negative control for 0162's fix: same deterministic scenario as the misspecified-student
        test above, differing only in ``degree`` -- a student whose functional form actually matches
        the oracle's (a degree-2 fit to a quadratic bowl) generalizes well to the fresh pool, and
        round2_beats_round1 must still be able to report True honestly. The fix must not make this
        metric impossible to satisfy, only stop guaranteeing it by construction."""
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=12, n_iter=0, degree=2, seed=7)

        self.assertTrue(report.beats_baseline)
        self.assertIsNotNone(report.round2)
        self.assertTrue(report.round2_beats_round1)
        self.assertGreaterEqual(report.round2.best_score, report.round1.best_score)
        self.assertEqual(report.incumbent_best_score, report.round2.best_score)

    def test_collapse_monitor_is_run_and_reused_not_reimplemented(self):
        oracle = _quadratic_bowl_oracle(target=np.array([2.0, -1.0]))
        report = amplify_and_capture(oracle, _BOUNDS, n_init=12, n_iter=0, degree=2, seed=7)

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


class AmplifyAbstentionHandlingTest(unittest.TestCase):
    """MXR-080-0189 residual-gap regression coverage (see the module docstring): an abstained oracle
    call's -inf placeholder must never reach the permutation test or fit_student as if it were a genuine
    observation, even though run1.history/baseline_run.history still keep it for the receipted record.
    """

    def test_permutation_test_never_sees_an_abstained_placeholder_score(self):
        """Direct assertion on the actual mechanism, not just DesignRun's own bookkeeping: spy on the
        real scipy permutation_test call inside amplify_and_capture and inspect exactly what arrays it
        was given."""
        oracle = _flaky_quadratic_bowl_oracle(np.array([2.0, -1.0]), hang_on_calls={2, 6})
        captured = {}
        real_permutation_test = amplify_module.permutation_test

        def spy(samples, *args, **kwargs):
            captured["samples"] = [np.asarray(s, dtype=float) for s in samples]
            return real_permutation_test(samples, *args, **kwargs)

        with mock.patch("mixle.doe.amplify.permutation_test", side_effect=spy):
            report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=0, seed=0)

        # the abstentions really happened and are really kept in the receipted record (not hidden)...
        self.assertGreaterEqual(sum(1 for c in report.round1.run.history if c.result.abstained), 1)
        self.assertGreaterEqual(sum(1 for c in report.baseline.run.history if c.result.abstained), 1)
        self.assertIn(float("-inf"), report.round1.run.scores().tolist())
        self.assertIn(float("-inf"), report.baseline.run.scores().tolist())

        # ...but neither array actually handed to the permutation test contains a -inf placeholder, and
        # neither array silently dropped a genuine observation either.
        run1_fed, baseline_fed = captured["samples"]
        self.assertTrue(np.all(np.isfinite(run1_fed)))
        self.assertTrue(np.all(np.isfinite(baseline_fed)))
        self.assertEqual(len(run1_fed), sum(1 for c in report.round1.run.history if not c.result.abstained))
        self.assertEqual(len(baseline_fed), sum(1 for c in report.baseline.run.history if not c.result.abstained))

    def test_fit_student_ignores_an_abstained_candidate_and_matches_the_genuine_only_fit(self):
        rng = np.random.RandomState(0)
        target = np.array([2.0, -1.0])
        run = DesignRun(oracle_name="t", oracle_tier="executable", oracle_fidelity=None)
        for _ in range(8):
            x = rng.uniform(-5, 5, size=2)
            d2 = float(np.sum((x - target) ** 2))
            run.append(DesignCandidate(x=x, result=OracleResult(score=-d2, receipt={}, cost=1.0)))
        genuine_only_student = fit_student(run, degree=2)
        self.assertTrue(np.all(np.isfinite(genuine_only_student.coef)))

        # Append ONE abstained candidate to the SAME history, exactly as optimize_under_oracle would
        # after a timeout -- confirmed pre-fix this alone drove the entire coef vector to nan.
        run.append(
            DesignCandidate(
                x=rng.uniform(-5, 5, size=2),
                result=OracleResult(
                    score=float("-inf"),
                    receipt={"reason": "test abstention"},
                    abstained=True,
                    cost=0.0,
                ),
            )
        )
        student_with_abstention_present = fit_student(run, degree=2)

        np.testing.assert_allclose(genuine_only_student.coef, student_with_abstention_present.coef)
        self.assertTrue(np.all(np.isfinite(student_with_abstention_present.coef)))

    def test_fit_student_raises_a_specific_error_when_every_candidate_abstained(self):
        run = DesignRun(oracle_name="t", oracle_tier="executable", oracle_fidelity=None)
        run.append(
            DesignCandidate(
                x=np.array([0.0, 0.0]),
                result=OracleResult(
                    score=float("-inf"),
                    receipt={"reason": "test abstention"},
                    abstained=True,
                    cost=0.0,
                ),
            )
        )
        with self.assertRaises(ValueError) as ctx:
            fit_student(run, degree=2)
        self.assertIn("abstained", str(ctx.exception))

    def test_end_to_end_amplify_and_capture_survives_abstentions_with_a_finite_student(self):
        """Integration-level negative control: a full amplify_and_capture run against a flaky oracle
        must still produce a computed p-value and a finite-coefficient student, never a nan student or a
        crash, even though part of the spent budget abstained. n_init=4/n_iter=8/seed=0 is the same
        reliable-pass configuration test_round1_beats_a_budget_matched_random_baseline above uses (a
        real Bayesian-optimization search, not just a random initial design), so beats_baseline and
        hence student are deterministically non-None here -- this assertion is not conditional on the
        gate happening to clear, unlike a weaker n_iter=0 setup where it would be only vacuously true
        whenever the gate did not clear."""
        oracle = _flaky_quadratic_bowl_oracle(np.array([2.0, -1.0]), hang_on_calls={5, 18})
        report = amplify_and_capture(oracle, _BOUNDS, n_init=4, n_iter=8, seed=0)

        self.assertGreaterEqual(sum(1 for c in report.round1.run.history if c.result.abstained), 1)
        self.assertGreaterEqual(sum(1 for c in report.baseline.run.history if c.result.abstained), 1)
        self.assertTrue(np.isfinite(report.baseline_p_value))
        self.assertTrue(report.beats_baseline)
        self.assertIsNotNone(report.student)
        self.assertTrue(np.all(np.isfinite(report.student.coef)))


if __name__ == "__main__":
    unittest.main()
