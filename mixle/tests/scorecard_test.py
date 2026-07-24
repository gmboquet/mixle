"""System scorecard (mixle.system.scorecard), CARD SCORE-a: evaluate() over a fixed set; catch a worsening round."""

import unittest

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.system import Query, System, SystemConfig, detect_regression, evaluate


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


if __name__ == "__main__":
    unittest.main()
