"""Router cost accounting includes every attempted tier."""

import unittest

from mixle.task.calibrate import ESCALATE
from mixle.task.router import Router


class _Decider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def decide(self, _request):
        if self.error is not None:
            raise self.error
        return self.result


def _teacher(rows):
    return [f"frontier:{row}" for row in rows]


class RouterAccountingTest(unittest.TestCase):
    def test_escalated_tiers_are_paid_before_the_answering_tier(self):
        router = Router(
            [
                ("first", _Decider(ESCALATE), 1.0),
                ("second", _Decider("answer"), 2.0),
                ("frontier", _teacher, 10.0),
            ]
        )
        self.assertEqual(router("request"), "answer")
        report = router.report()
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["realized_cost"], 3.0)
        self.assertEqual([tier["attempted"] for tier in report["tiers"]], [1, 1, 0])
        self.assertEqual([tier["answered"] for tier in report["tiers"]], [0, 1, 0])

    def test_failed_tier_attempt_is_not_free(self):
        router = Router(
            [
                ("broken", _Decider(error=RuntimeError("boom")), 1.0),
                ("second", _Decider("answer"), 2.0),
                ("frontier", _teacher, 10.0),
            ]
        )
        self.assertEqual(router("request"), "answer")
        report = router.report()
        self.assertEqual(report["realized_cost"], 3.0)
        self.assertEqual(report["tiers"][0]["failed"], 1)
        self.assertEqual(len(router.stats.degraded), 1)

    def test_malformed_frontier_response_is_a_paid_failed_attempt(self):
        router = Router(
            [
                ("local", _Decider(ESCALATE), 1.0),
                ("frontier", lambda _rows: [], 10.0),
            ]
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            router("request")
        report = router.report()
        self.assertEqual(report["realized_cost"], 11.0)
        self.assertEqual(report["tiers"][-1]["failed"], 1)

    def test_names_costs_and_order_are_validated(self):
        valid_local = _Decider(ESCALATE)
        invalid_tiers = (
            [("same", valid_local, 1.0), ("same", _teacher, 2.0)],
            [("local", valid_local, 2.0), ("frontier", _teacher, 1.0)],
            [("local", valid_local, -1.0), ("frontier", _teacher, 2.0)],
            [("local", valid_local, float("nan")), ("frontier", _teacher, 2.0)],
            [("local", valid_local, True), ("frontier", _teacher, 2.0)],
        )
        for tiers in invalid_tiers:
            with self.subTest(tiers=tiers), self.assertRaises(ValueError):
                Router(tiers)


if __name__ == "__main__":
    unittest.main()
