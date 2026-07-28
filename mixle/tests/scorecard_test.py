"""System scorecard (mixle.system.scorecard), CARD SCORE-a: evaluate() over a fixed set; catch a worsening round."""

import unittest

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.system import Query, System, SystemConfig, SystemScorecard, detect_regression, evaluate
from mixle.system.scorecard import question_set_identity


class EvaluateTest(unittest.TestCase):
    def test_quality_and_grounded_fraction_on_a_perfect_system(self):
        system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        question_set = [(Query("q1"), "yes"), (Query("q2"), "yes")]
        card = evaluate(system, question_set)
        self.assertEqual(card.quality, 1.0)
        self.assertEqual(card.grounded_fraction, 1.0)
        self.assertEqual(card.realized_cost, 2.0)
        self.assertEqual(card.n, 2)

    def test_quality_reflects_wrong_answers(self):
        system = System(SystemConfig(teacher=lambda p: "no idea"))
        question_set = [(Query("q1"), "yes"), (Query("q2"), "yes")]
        card = evaluate(system, question_set)
        self.assertEqual(card.quality, 0.0)

    def test_empty_question_set_is_honest_not_a_div_by_zero_crash(self):
        system = System(SystemConfig(teacher=lambda p: "x"))
        card = evaluate(system, [])
        self.assertEqual(card.n, 0)
        self.assertEqual(card.quality, 0.0)

    def test_teacher_down_answers_are_not_counted_grounded(self):
        def broken_teacher(prompt):
            raise ConnectionError("down")

        store = Substrate()
        store.put(SubstrateItem(kind="text", text="yes it is confirmed"))
        system = System(SystemConfig(teacher=broken_teacher, store=store))
        card = evaluate(system, [(Query("q1"), "yes")])
        self.assertEqual(card.grounded_fraction, 0.0)


class EvaluateDoesNotLeakIntoTrainingTest(unittest.TestCase):
    """CRITICAL: evaluate() must be a genuine read-only snapshot. If it were not, a successful teacher
    answer to a held-out question would land in System._harvest as a side effect of just answering it,
    and a later improve() call could promote it into the captured cache -- after which re-evaluating
    that same "held-out" set would be trivially free and perfect, because the system would have
    memorized the test answers from having been evaluated on them once already."""

    def test_evaluating_a_held_out_question_does_not_populate_the_harvest(self):
        system = System(SystemConfig(teacher=lambda p: "ok"))
        self.assertEqual(system._harvest, {})

        evaluate(system, [(Query("secret-test-question"), "ok")])

        self.assertEqual(system._harvest, {})

    def test_a_generous_improve_after_evaluation_still_has_nothing_to_promote(self):
        system = System(SystemConfig(teacher=lambda p: "ok"))
        evaluate(system, [(Query("secret-test-question"), "ok")])

        report = system.improve(1000)

        self.assertEqual(report["status"], "nothing_to_improve")
        self.assertEqual(system._captured, {})

    def test_reevaluating_the_same_held_out_set_costs_the_same_every_time(self):
        # if evaluation ever leaked into the cache, the SECOND evaluate() over the same set would show
        # realized_cost collapsing towards 0 (served free from the captured cache); it must not move.
        system = System(SystemConfig(teacher=lambda p: "ok"))
        question_set = [(Query("secret-test-question"), "ok")]

        first = evaluate(system, question_set)
        system.improve(1000)  # even a very generous improve() must find nothing to promote
        second = evaluate(system, question_set)

        self.assertEqual(first.realized_cost, 1.0)
        self.assertEqual(second.realized_cost, 1.0)
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 0)  # eval never touches this

    # negative control: evaluate() must still faithfully exercise the real answering path -- read-only
    # must not turn evaluation into a no-op that always reports perfect (or always zero) quality
    def test_evaluate_still_reflects_real_answer_quality(self):
        system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        card = evaluate(system, [(Query("q1"), "yes")])
        self.assertEqual(card.quality, 1.0)
        self.assertEqual(card.grounded_fraction, 1.0)
        self.assertEqual(card.realized_cost, 1.0)


class RegressionDetectionTest(unittest.TestCase):
    def test_a_deliberately_worsening_round_is_caught(self):
        question_set = [(Query("q1"), "yes"), (Query("q2"), "yes")]

        good_system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        baseline = evaluate(good_system, question_set)

        bad_system = System(SystemConfig(teacher=lambda p: "no idea"))
        worse = evaluate(bad_system, question_set)

        report = detect_regression(baseline, worse)
        self.assertTrue(report.regressed)
        self.assertTrue(any("quality regressed" in r for r in report.reasons))

    def test_a_grounded_fraction_drop_is_caught(self):
        question_set = [(Query("q1"), "yes")]
        good_system = System(SystemConfig(teacher=lambda p: "yes"))
        baseline = evaluate(good_system, question_set)

        def broken_teacher(prompt):
            raise ConnectionError("down")

        store = Substrate()
        store.put(SubstrateItem(kind="text", text="yes indeed"))
        degraded_system = System(SystemConfig(teacher=broken_teacher, store=store))
        current = evaluate(degraded_system, question_set)

        report = detect_regression(baseline, current)
        self.assertTrue(report.regressed)
        self.assertTrue(any("grounded_fraction regressed" in r for r in report.reasons))

    def test_a_non_regressing_round_is_not_flagged(self):
        question_set = [(Query("q1"), "yes")]
        system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        baseline = evaluate(system, question_set)
        current = evaluate(system, question_set)
        report = detect_regression(baseline, current)
        self.assertFalse(report.regressed)
        self.assertEqual(report.reasons, [])

    def test_an_improving_round_is_not_flagged(self):
        question_set = [(Query("q1"), "yes")]
        bad_system = System(SystemConfig(teacher=lambda p: "no idea"))
        baseline = evaluate(bad_system, question_set)
        good_system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        improved = evaluate(good_system, question_set)
        report = detect_regression(baseline, improved)
        self.assertFalse(report.regressed)


class NoFalseCalibrationMetricTest(unittest.TestCase):
    """MXR-080-1691: `calibration` was `quality` copied verbatim -- a deterministic system with no
    confidence at all scored a perfect "calibration" -- and detect_regression never compared it, so a
    1.0 -> 0.0 drop on the field was not a regression. A metric that measures nothing and gates nothing
    must not be reported at all."""

    def test_scorecards_do_not_report_a_calibration_number(self):
        system = System(SystemConfig(teacher=lambda p: "yes"))
        card = evaluate(system, [(Query("q1"), "yes")])
        self.assertNotIn("calibration", card.to_dict())
        self.assertFalse(hasattr(card, "calibration"))

    def test_calibration_is_not_silently_accepted_as_a_constructor_field(self):
        with self.assertRaises(TypeError):
            SystemScorecard(quality=1.0, calibration=1.0, realized_cost=0.0, grounded_fraction=1.0, n=1)


class ComparisonFailsClosedTest(unittest.TestCase):
    """MXR-080-1692: the regression gate failed OPEN on evidence it could not actually compare -- a
    NaN-metric card, a NaN tolerance, and a zero-case card against a 100-case baseline each returned
    `regressed=False`, which reads as "this round is fine"."""

    def _card(self, **kw):
        base = dict(quality=1.0, realized_cost=0.0, grounded_fraction=1.0, n=2, question_set_id="set-a")
        base.update(kw)
        return SystemScorecard(**base)

    def test_nonfinite_and_out_of_range_metrics_are_rejected_at_construction(self):
        for bad in ({"quality": float("nan")}, {"quality": 1.5}, {"grounded_fraction": -0.1}):
            with self.assertRaises(ValueError):
                self._card(**bad)
        with self.assertRaises(ValueError):
            self._card(realized_cost=float("nan"))
        with self.assertRaises(ValueError):
            self._card(realized_cost=-1.0)
        with self.assertRaises(ValueError):
            self._card(n=-1)

    def test_a_nan_tolerance_cannot_silently_disable_the_gate(self):
        baseline = self._card()
        worse = self._card(quality=0.0, realized_cost=99.0, grounded_fraction=0.0)
        self.assertTrue(detect_regression(baseline, worse).regressed)  # sanity: it does catch this
        with self.assertRaises(ValueError):
            detect_regression(baseline, worse, tolerance=float("nan"))
        with self.assertRaises(ValueError):
            detect_regression(baseline, worse, tolerance=-1.0)

    def test_a_card_from_a_different_held_out_set_is_incomparable_not_accepted(self):
        baseline = self._card()
        other_set = self._card(question_set_id="set-b")
        report = detect_regression(baseline, other_set)
        self.assertFalse(report.comparable)
        self.assertTrue(report.regressed)

    def test_a_zero_case_card_does_not_pass_as_a_clean_round(self):
        baseline = self._card(n=100)
        empty = self._card(n=0)
        report = detect_regression(baseline, empty)
        self.assertFalse(report.comparable)
        self.assertTrue(report.regressed)
        self.assertTrue(any("case counts" in r for r in report.reasons))

    def test_unidentified_cards_are_incomparable(self):
        report = detect_regression(self._card(question_set_id=""), self._card(question_set_id=""))
        self.assertFalse(report.comparable)
        self.assertTrue(report.regressed)

    def test_evaluate_binds_the_card_to_its_question_set_and_scorer(self):
        system = System(SystemConfig(teacher=lambda p: "yes"))
        set_a = [(Query("q1"), "yes")]
        set_b = [(Query("q2"), "yes")]
        card_a = evaluate(system, set_a)
        card_b = evaluate(system, set_b)
        self.assertEqual(card_a.question_set_id, question_set_identity(set_a))
        self.assertNotEqual(card_a.question_set_id, card_b.question_set_id)
        self.assertFalse(detect_regression(card_a, card_b).comparable)
        # a laxer judge on the same questions is different evidence too
        lax = evaluate(system, set_a, scorer=lambda reply, expected: True)
        self.assertNotEqual(card_a.question_set_id, lax.question_set_id)
        # and the honest same-set comparison still works
        self.assertTrue(detect_regression(card_a, evaluate(system, set_a)).comparable)


if __name__ == "__main__":
    unittest.main()
