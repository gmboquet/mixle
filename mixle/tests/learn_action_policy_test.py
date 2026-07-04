"""learn_action_policy (J3): a learned reasoner acquisition scorer with never-worse fallback."""

import unittest

from mixle.inference import LearnedAcquisition, learn_action_policy
from mixle.substrate.act import Action, investigate, score_action


def _history():
    rows = []
    for _ in range(20):
        rows.append(({"kind": "compute", "cost": 1.0, "overlap": 0.6}, "compute", {"value": 1.0}))
        rows.append(({"kind": "simulate", "cost": 2.0, "overlap": 0.6}, "simulate", {"value": 0.0}))
    return rows


def _compute(q="forecast spend plan"):
    return Action("c", "compute", run=lambda q: ["x"], cost=1.0, description="forecast spend plan")


def _simulate():
    return Action("s", "simulate", run=lambda q: ["y"], cost=2.0, description="forecast spend plan")


class LearnActionPolicyTest(unittest.TestCase):
    def test_learns_which_action_kind_pays_off(self):
        policy = learn_action_policy(_history())
        self.assertIsInstance(policy, LearnedAcquisition)
        q = "forecast spend plan"
        # history says compute yields, simulate does not -> learned prefers compute more sharply than static
        self.assertGreater(policy(_compute(), q), policy(_simulate(), q))
        self.assertLess(policy(_simulate(), q), score_action(_simulate(), q))  # demoted below the lexical prior

    def test_defers_to_static_when_history_is_thin(self):
        policy = learn_action_policy(_history()[:2], min_neighbors=4)
        q = "forecast spend plan"
        self.assertEqual(policy(_compute(), q), score_action(_compute(), q))

    def test_usable_as_investigate_scorer(self):
        policy = learn_action_policy(_history())
        inv = investigate(
            "forecast spend plan",
            [_compute(), _simulate()],
            lambda q, ctx: f"A[{ctx[:10]}]",
            scorer=policy,
            min_confidence=0.0,
        )
        self.assertEqual(inv.steps[0].kind, "compute")  # learned order fires the productive action first

    def test_empty_rows_raise(self):
        with self.assertRaises(ValueError):
            learn_action_policy([])

    def test_expected_yield_none_when_thin(self):
        policy = learn_action_policy(_history()[:3], min_neighbors=4)
        self.assertIsNone(policy.expected_yield({"kind": "compute", "cost": 1.0, "overlap": 0.6}))


if __name__ == "__main__":
    unittest.main()
