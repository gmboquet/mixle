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


class _FixedStudent:
    def __init__(self, answer):
        self.answer = answer

    def decide(self, _request):
        return self.answer


class ScorecardAccountingTest(unittest.TestCase):
    def test_all_escalations_pay_student_plus_teacher_and_do_not_invent_accuracy(self):
        from mixle.task import scorecard

        calls = []

        def teacher(request):
            calls.append(request)
            return "teacher-answer"

        card = scorecard(
            _FixedStudent(None),
            teacher,
            [1, 2],
            task_truth=["actual-answer", "actual-answer"],
            student_cost=1.0,
            teacher_cost=10.0,
        )
        self.assertEqual(calls, [1, 2])  # labels and timing share the same teacher calls
        self.assertEqual(card.escalation_rate, 1.0)
        self.assertEqual(card.student_cost_per_1k, 11_000.0)
        self.assertEqual(card.teacher_cost_per_1k, 10_000.0)
        self.assertEqual(card.teacher_agreement, 1.0)
        self.assertEqual(card.end_to_end_accuracy, 0.0)

    def test_no_truth_means_accuracy_is_unmeasured_not_teacher_agreement(self):
        from mixle.task import scorecard

        card = scorecard(_FixedStudent("local"), lambda _x: "teacher", [1, 2])
        self.assertIsNone(card.end_to_end_accuracy)
        self.assertEqual(card.teacher_agreement, 0.0)
        self.assertEqual(card.truth_source, "not_measured")
        self.assertIn("not measured", card.table())

    def test_truth_panel_requires_exact_cardinality(self):
        from mixle.task import scorecard

        with self.assertRaisesRegex(ValueError, "exactly one"):
            scorecard(_FixedStudent("local"), lambda _x: "teacher", [1, 2], task_truth=["only-one"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ScorecardTest(unittest.TestCase):
    def test_receipts_are_measured_and_honest(self):
        from mixle.task import scorecard, solve

        sol = solve(_route, _tickets(400), alpha=0.1, ood=None, seed=0, epochs=300)
        card = scorecard(
            sol,
            _route,
            _tickets(200, seed=9),
            task_truth=_route,
            student_cost=0.0001,
            teacher_cost=0.03,
            task="ticket routing",
        )

        self.assertGreaterEqual(card.end_to_end_accuracy, card.local_accuracy - 1e-9)
        self.assertGreaterEqual(card.teacher_agreement, card.local_agreement - 1e-9)
        self.assertGreater(card.local_agreement, 0.85)
        self.assertLessEqual(card.escalation_rate, 1.0)
        self.assertGreater(card.student_p50_ms, 0.0)
        self.assertGreater(card.teacher_p50_ms, 0.0)

        # Every request pays for its local attempt; escalations additionally pay the teacher.
        self.assertLessEqual(card.student_cost_per_1k, card.teacher_cost_per_1k + 1e-9)

        table = card.table()
        for needle in ("end-to-end accuracy", "escalation rate", "cost / 1k requests", "ticket routing"):
            self.assertIn(needle, table)
        self.assertEqual(card.as_dict()["n_test"], 200)


def _price(t):
    return 20.0 + 0.5 * t["amount"] + (30.0 if t["region"] == "eu" else 0.0)


def _flags(t):
    out = []
    if t["amount"] > 400:
        out.append("high-value")
    if t["kind"] in ("refund", "billing"):
        out.append("money")
    if t["region"] == "eu":
        out.append("eu-rules")
    return out


def _enrich(t):
    return {"team": "billing" if t["kind"] in ("refund", "billing") else "support", "price": _price(t)}


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ScorecardShapesTest(unittest.TestCase):
    """Receipts for the other solve shapes — agreement means each shape's own promise."""

    def test_regression_receipts_use_tol_agreement(self):
        from mixle.task import scorecard, solve_regression

        sol = solve_regression(_price, _tickets(240), tol=1e6, alpha=0.1, seed=0, epochs=300)
        self.assertTrue(sol.answers_locally)
        card = scorecard(sol, _price, _tickets(120, seed=9), task_truth=_price, task="pricing")
        # tol is astronomically generous, so every local answer is within-tol by construction
        self.assertEqual(card.escalation_rate, 0.0)
        self.assertEqual(card.local_agreement, 1.0)
        self.assertEqual(card.end_to_end_accuracy, 1.0)
        self.assertIsNotNone(card.artifact_bytes)
        self.assertGreater(card.artifact_bytes, 0)

    def test_regression_impossible_tol_escalates_everything(self):
        import math

        from mixle.task import scorecard
        from mixle.task.regress import solve_regression

        sol = solve_regression(_price, _tickets(240), tol=1e6, alpha=0.1, seed=0, epochs=60)
        sol.tol = 1e-6  # a promise the student can't meet -> the gate refuses every input
        card = scorecard(
            sol,
            _price,
            _tickets(120, seed=9),
            task_truth=_price,
            task="pricing-tight",
        )
        self.assertEqual(card.escalation_rate, 1.0)
        self.assertEqual(card.end_to_end_accuracy, 1.0)  # the teacher answered everything
        self.assertTrue(math.isnan(card.local_agreement))  # no local answers to judge

    def test_multilabel_receipts_exact_set_agreement(self):
        from mixle.task import scorecard, solve_multilabel

        sol = solve_multilabel(_flags, _tickets(280), alpha=0.1, seed=0, epochs=300)
        card = scorecard(
            sol,
            _flags,
            _tickets(120, seed=9),
            task_truth=_flags,
            student_cost=0.0001,
            teacher_cost=0.03,
            task="flags",
        )
        self.assertGreaterEqual(card.end_to_end_accuracy, card.local_accuracy - 1e-9)
        self.assertGreater(card.end_to_end_accuracy, 0.75)  # amount≈400 boundary is genuinely hard
        self.assertLessEqual(card.student_cost_per_1k, card.teacher_cost_per_1k + 1e-9)
        self.assertIsNotNone(card.artifact_bytes)

    def test_structured_receipts_per_field_promise(self):
        from mixle.task import scorecard, solve_structured

        sol = solve_structured(_enrich, _tickets(240), tol=1e6, alpha=0.1, seed=0, epochs=300)
        card = scorecard(
            sol,
            _enrich,
            _tickets(120, seed=9),
            task_truth=_enrich,
            task="enrich",
        )
        self.assertGreater(card.end_to_end_accuracy, 0.85)
        self.assertGreaterEqual(card.end_to_end_accuracy, card.local_accuracy - 1e-9)
        self.assertIsNotNone(card.artifact_bytes)
        self.assertGreater(card.artifact_bytes, 0)


if __name__ == "__main__":
    unittest.main()
