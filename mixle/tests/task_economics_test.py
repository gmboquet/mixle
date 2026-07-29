"""Cost economics (mixle.task.economics): break-even, cascade cost, and the cheapest-route recommendation.

Pure arithmetic -- no torch -- so it runs in the base suite.
"""

import math
import unittest

from mixle.task.economics import (
    CostModel,
    break_even_volume,
    cascade_cost_per_request,
    recommend_route,
)


class CascadeCostTest(unittest.TestCase):
    def test_cost_interpolates_local_to_frontier(self):
        c = CostModel(c_frontier=1.0, c_local=0.01)
        self.assertAlmostEqual(cascade_cost_per_request(c, 0.0), 0.01)
        self.assertAlmostEqual(cascade_cost_per_request(c, 1.0), 1.01)
        self.assertLess(cascade_cost_per_request(c, 0.2), cascade_cost_per_request(c, 0.5))

    def test_costs_and_probabilities_are_finite_nonnegative_and_bounded(self):
        for kwargs in (
            {"c_frontier": -1.0},
            {"c_frontier": float("nan")},
            {"c_frontier": 1.0, "c_local": float("inf")},
            {"c_frontier": True},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(ValueError):
                CostModel(**kwargs)
        c = CostModel(c_frontier=1.0)
        for probability in (-0.1, 1.1, float("nan"), True):
            with self.subTest(probability=repr(probability)), self.assertRaises(ValueError):
                cascade_cost_per_request(c, probability)


class BreakEvenTest(unittest.TestCase):
    def test_local_only_break_even(self):
        # setup 100*0.02 + 5 = 7 ; saving per request 1.0 - 0.0 = 1.0 -> break-even at 7 requests
        c = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.02, train_cost=5.0)
        self.assertAlmostEqual(break_even_volume(c, n_label=100), 7.0)

    def test_no_break_even_when_local_not_cheaper(self):
        c = CostModel(c_frontier=1.0, c_local=1.0, c_label=0.0, train_cost=10.0)
        self.assertTrue(math.isinf(break_even_volume(c, n_label=0)))

    def test_escalation_raises_break_even(self):
        c = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.0, train_cost=10.0)
        lo = break_even_volume(c, n_label=0, p_escalate=0.1)
        hi = break_even_volume(c, n_label=0, p_escalate=0.5)
        self.assertLess(lo, hi)  # escalating more often pays back slower


class RecommendTest(unittest.TestCase):
    def test_low_volume_prefers_frontier(self):
        c = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.05, train_cost=100.0)
        plan = recommend_route(c, volume=10, n_label=200, p_escalate=0.1)
        self.assertEqual(plan.route, "frontier_only")  # setup (110) dwarfs 10 requests
        self.assertEqual(plan.savings_vs_frontier, 0.0)

    def test_high_volume_prefers_cascade(self):
        c = CostModel(c_frontier=1.0, c_local=0.001, c_label=0.05, train_cost=100.0)
        plan = recommend_route(c, volume=1_000_000, n_label=200, p_escalate=0.1)
        self.assertEqual(plan.route, "cascade")
        self.assertGreater(plan.savings_vs_frontier, 0.0)
        # cascade per-request ~ c_local + 0.1*c_frontier = 0.101, well under frontier 1.0
        self.assertLess(plan.per_request, 0.2)

    def test_escalation_cap_drops_cascade(self):
        c = CostModel(c_frontier=1.0, c_local=0.001, c_label=0.0, train_cost=1.0)
        # p_escalate 0.4 exceeds the 0.2 cap -> cascade removed, frontier_only remains
        plan = recommend_route(c, volume=10_000, n_label=100, p_escalate=0.4, max_escalation=0.2)
        self.assertEqual(plan.route, "frontier_only")
        self.assertNotIn("cascade", plan.options)
        self.assertNotIn("local_only", plan.options)

    def test_local_only_requires_an_explicit_zero_escalation_certificate(self):
        c = CostModel(c_frontier=1.0, c_local=0.01)
        ordinary = recommend_route(c, volume=10_000, n_label=0, p_escalate=0.0)
        self.assertNotIn("local_only", ordinary.options)
        certified = recommend_route(
            c,
            volume=10_000,
            n_label=0,
            p_escalate=0.0,
            local_only_certified=True,
        )
        self.assertEqual(certified.route, "local_only")
        with self.assertRaises(ValueError):
            recommend_route(
                c,
                volume=10_000,
                n_label=0,
                p_escalate=0.1,
                local_only_certified=True,
            )

    def test_volume_and_label_count_must_be_exact_nonnegative_integers(self):
        c = CostModel(c_frontier=1.0)
        for kwargs in (
            {"volume": -1, "n_label": 0},
            {"volume": 1.5, "n_label": 0},
            {"volume": 1, "n_label": True},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(ValueError):
                recommend_route(c, p_escalate=0.1, **kwargs)


if __name__ == "__main__":
    unittest.main()
