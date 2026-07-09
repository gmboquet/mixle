"""improve_by_regret: heuristic effort allocation across improvement options, trusting only realized
scorecard gain (workstream META-a)."""

import unittest

from mixle.meta import ImprovementOption, improve_by_regret
from mixle.system import Query, System, SystemConfig

_QUESTIONS = [(Query("Q1"), "A1"), (Query("Q2"), "A2"), (Query("Q3"), "A3")]


def _make_system(knowledge: dict) -> System:
    def teacher(prompt: str) -> str:
        return knowledge.get(prompt, "unknown")

    return System(SystemConfig(teacher=teacher))


class ImproveByRegretTest(unittest.TestCase):
    def test_runs_highest_regret_per_dollar_first_regardless_of_input_order(self):
        knowledge: dict = {}
        system = _make_system(knowledge)
        options = [
            ImprovementOption(name="low_value", cost=1.0, run=lambda: knowledge.update(Q2="A2"), estimated_regret=0.2),
            ImprovementOption(name="high_value", cost=1.0, run=lambda: knowledge.update(Q1="A1"), estimated_regret=0.9),
        ]
        report = improve_by_regret(system, _QUESTIONS, options, budget=10.0)

        self.assertEqual(report.order, ["high_value", "low_value"])  # ran in regret-per-dollar order
        self.assertIsNone(report.stopped_on_regression)
        self.assertAlmostEqual(report.scorecard_before.quality, 0.0)
        self.assertAlmostEqual(report.scorecard_after.quality, 2 / 3)

    def test_measures_realized_gain_not_the_estimate(self):
        knowledge: dict = {}
        system = _make_system(knowledge)
        # a wildly over-optimistic estimate does not change what gets MEASURED after it runs
        options = [
            ImprovementOption(
                name="accumulate_q1", cost=1.0, run=lambda: knowledge.update(Q1="A1"), estimated_regret=999.0
            )
        ]
        report = improve_by_regret(system, _QUESTIONS, options, budget=10.0)

        self.assertAlmostEqual(report.realized_gain_per_dollar["accumulate_q1"], 1 / 3)

    def test_skips_options_that_do_not_fit_the_budget(self):
        knowledge: dict = {}
        system = _make_system(knowledge)
        options = [
            ImprovementOption(name="first", cost=1.0, run=lambda: knowledge.update(Q1="A1"), estimated_regret=0.9),
            ImprovementOption(name="second", cost=1.0, run=lambda: knowledge.update(Q2="A2"), estimated_regret=0.5),
        ]
        report = improve_by_regret(system, _QUESTIONS, options, budget=1.0)

        self.assertEqual(report.order, ["first"])
        self.assertEqual(report.skipped, ["second"])
        self.assertEqual(report.spent, 1.0)

    def test_stops_immediately_on_a_regressing_option_never_absorbing_it_silently(self):
        knowledge: dict = {"Q1": "A1"}  # start already answering Q1 correctly
        system = _make_system(knowledge)
        options = [
            ImprovementOption(name="good", cost=1.0, run=lambda: knowledge.update(Q2="A2"), estimated_regret=0.9),
            ImprovementOption(name="bad_regresses", cost=1.0, run=knowledge.clear, estimated_regret=0.5),
        ]
        report = improve_by_regret(system, _QUESTIONS, options, budget=10.0)

        self.assertEqual(report.order, ["good", "bad_regresses"])
        self.assertIsNotNone(report.stopped_on_regression)
        self.assertTrue(report.stopped_on_regression.regressed)
        self.assertIn("quality regressed", report.stopped_on_regression.reasons[0])
        # the final scorecard reflects the regression honestly -- it is not silently discarded/reverted
        self.assertAlmostEqual(report.scorecard_after.quality, 0.0)


if __name__ == "__main__":
    unittest.main()
