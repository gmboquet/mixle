"""System scorecard (mixle.scorecard), CARD SCORE-a: evaluate() over a fixed set; catch a worsening round."""

import unittest

from mixle.scorecard import detect_regression, evaluate
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.system import Query, System, SystemConfig


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
