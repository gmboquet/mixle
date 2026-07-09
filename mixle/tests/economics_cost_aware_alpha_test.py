"""select_alpha_for_cost: cost-aware threshold selection (workstream B2) -- recommend_route actually
connected to the calibration step, choosing alpha from a CostModel target instead of a fixed default.
"""

import unittest

from mixle.task.economics import CostModel, RoutePlan, recommend_route, select_alpha_for_cost


class _FakeCalibratedModel:
    """A minimal stand-in for CalibratedTaskModel: a mutable alpha, a no-op calibrate(), and an
    escalation_rate() that reads from a known alpha -> p_escalate curve (tighter alpha = more escalation,
    the real conformal relationship) -- so the sweep has a genuine, checkable optimum."""

    def __init__(self, escalation_by_alpha: dict) -> None:
        self.alpha = None
        self.qhat = None
        self._curve = escalation_by_alpha

    def calibrate(self, texts, labels):
        self.qhat = f"qhat@alpha={self.alpha}"  # stands in for a real recalibration
        return self

    def escalation_rate(self, texts):
        return self._curve[self.alpha]


class SelectAlphaForCostTest(unittest.TestCase):
    def test_picks_the_alpha_with_the_lowest_recommend_route_cost(self):
        # tight alpha (0.01) escalates almost everything (expensive frontier calls dominate);
        # loose alpha (0.3) escalates almost nothing but risks quality -- 0.1 is the sweet spot here.
        curve = {0.01: 0.9, 0.05: 0.5, 0.1: 0.15, 0.15: 0.2, 0.2: 0.4, 0.3: 0.7}
        model = _FakeCalibratedModel(curve)
        cost = CostModel(c_frontier=1.0, c_local=0.01, c_label=0.05, train_cost=1.0)

        best_alpha, best_plan, plans = select_alpha_for_cost(
            model, ["cal text"], ["cal label"], ["probe text"], cost, volume=10_000, n_label=200
        )

        self.assertEqual(best_alpha, 0.1)
        self.assertIsInstance(best_plan, RoutePlan)
        self.assertEqual(set(plans), {0.01, 0.05, 0.1, 0.15, 0.2, 0.3})
        # every candidate's plan is independently reproducible via recommend_route directly
        for alpha, plan in plans.items():
            expected = recommend_route(cost, volume=10_000, n_label=200, p_escalate=curve[alpha])
            self.assertEqual(plan.total, expected.total)
        # the model is left calibrated at the winning alpha, not the last swept one
        self.assertEqual(model.alpha, 0.1)
        self.assertEqual(model.qhat, "qhat@alpha=0.1")

    def test_all_plans_beat_frontier_only_when_escalation_is_ever_cheap_enough(self):
        curve = {0.05: 0.5, 0.1: 0.1, 0.2: 0.05}
        model = _FakeCalibratedModel(curve)
        cost = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.01, train_cost=0.0)

        best_alpha, best_plan, _ = select_alpha_for_cost(
            model, [], [], [], cost, volume=5_000, n_label=100, alphas=(0.05, 0.1, 0.2)
        )
        self.assertEqual(best_alpha, 0.2)
        self.assertGreater(best_plan.savings_vs_frontier, 0)

    def test_custom_alpha_grid_is_honored(self):
        curve = {0.02: 0.6, 0.4: 0.05}
        model = _FakeCalibratedModel(curve)
        cost = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.0, train_cost=0.0)

        _, _, plans = select_alpha_for_cost(model, [], [], [], cost, volume=1_000, n_label=0, alphas=(0.02, 0.4))
        self.assertEqual(set(plans), {0.02, 0.4})


if __name__ == "__main__":
    unittest.main()
