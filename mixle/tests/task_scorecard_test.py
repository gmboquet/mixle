"""scorecard(): the measured receipts — tiny model vs the teacher it replaces."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _route(t):
    if t["amount"] > 500 and t["kind"] == "refund":
        return "finance-escalation"
    if t["kind"] in ("refund", "billing"):
        return "billing"
    return "support"


def _tickets(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "billing", "question", "bug"]
    return [
        {
            "kind": kinds[rng.randint(0, 4)],
            "amount": float(rng.gamma(2.0, 150.0)),
            "region": ["us", "eu"][rng.randint(0, 2)],
        }
        for _ in range(n)
    ]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ScorecardTest(unittest.TestCase):
    def test_receipts_are_measured_and_honest(self):
        from mixle.task import scorecard, solve

        sol = solve(_route, _tickets(400), alpha=0.1, ood=None, seed=0, epochs=300)
        card = scorecard(
            sol, _route, _tickets(200, seed=9), student_cost=0.0001, teacher_cost=0.03, task="ticket routing"
        )

        # end-to-end can never be worse than local-only: escalations are answered by the teacher
        self.assertGreaterEqual(card.end_to_end_accuracy, card.local_agreement - 1e-9)
        self.assertGreater(card.local_agreement, 0.85)
        self.assertLessEqual(card.escalation_rate, 1.0)
        self.assertGreater(card.student_p50_ms, 0.0)
        self.assertGreater(card.teacher_p50_ms, 0.0)

        # blended cost prices escalations at the teacher's rate — always <= frontier-only
        self.assertLessEqual(card.student_cost_per_1k, card.teacher_cost_per_1k + 1e-9)

        table = card.table()
        for needle in ("end-to-end accuracy", "escalation rate", "cost / 1k requests", "ticket routing"):
            self.assertIn(needle, table)
        self.assertEqual(card.as_dict()["n_test"], 200)


if __name__ == "__main__":
    unittest.main()
