"""decide() contracts for non-classification solve shapes (workstream B3): Router tiers are not
classifier-only -- RegressionSolution/StructuredSolution/MultiLabelSolution all expose the same
CalibratedTaskModel-shaped decide(x) -> value | ESCALATE contract Router requires of a tier.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.task.calibrate import ESCALATE
from mixle.task.multilabel import MultiLabelSolution
from mixle.task.router import Router
from mixle.task.structured_out import StructuredSolution


def _price(item):
    base = {"basic": 20.0, "pro": 80.0, "max": 150.0}[item["kind"]]
    return base + 0.5 * item["size"] + 0.001 * item["size"] ** 2


def _items(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["basic", "pro", "max"]
    return [{"kind": kinds[rng.randint(0, 3)], "size": float(rng.uniform(0, 100))} for _ in range(n)]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RegressionSolutionDecideTest(unittest.TestCase):
    def test_decide_matches_answers_locally_and_never_calls_the_teacher_itself(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(500), tol=20.0, alpha=0.1, seed=0, epochs=400)
        self.assertTrue(sol.answers_locally)

        before_escalated = sol.n_escalated
        result = sol.decide(_items(1, seed=9)[0])
        self.assertIsInstance(result, float)
        self.assertEqual(sol.n_escalated, before_escalated)  # decide() doesn't touch escalation bookkeeping

        sol.tol = 1e-12  # nothing can meet an impossible tolerance
        self.assertFalse(sol.answers_locally)
        self.assertIs(sol.decide(_items(1, seed=9)[0]), ESCALATE)

    def test_router_accepts_a_regression_solution_as_a_tier(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(500), tol=20.0, alpha=0.1, seed=0, epochs=400)
        self.assertTrue(sol.answers_locally)

        router = Router(tiers=[("regression_tier", sol, 0.0), ("frontier", _price, 1.0)])
        out = router(_items(1, seed=9)[0])
        self.assertIsInstance(out, float)
        self.assertEqual(router.stats.tiers[0].answered, 1)  # answered locally, never reached the frontier
        self.assertEqual(router.stats.tiers[1].answered, 0)


class StructuredSolutionDecideTest(unittest.TestCase):
    def test_decide_is_an_alias_of_try_local(self):
        sol = StructuredSolution(fields_cat={}, fields_num={}, teacher=lambda xs: [{} for _ in xs])
        self.assertEqual(sol.decide({"anything": 1}), {})
        self.assertEqual(sol.decide({"anything": 1}), sol.try_local({"anything": 1}))


class MultiLabelSolutionDecideTest(unittest.TestCase):
    def test_decide_is_an_alias_of_try_local(self):
        sol = object.__new__(MultiLabelSolution)
        sol.labels = ["a", "b"]
        sol.upper_absent = np.array([0.7, 0.7])
        sol.lower_present = np.array([0.3, 0.3])
        sol._scores = lambda xs: np.array([[0.9, 0.1]])  # confidently present / confidently absent

        self.assertEqual(sol.decide("x"), ["a"])
        self.assertEqual(sol.decide("x"), sol.try_local("x"))

        sol._scores = lambda xs: np.array([[0.5, 0.5]])  # ambiguous on both labels
        self.assertIs(sol.decide("x"), ESCALATE)


if __name__ == "__main__":
    unittest.main()
