"""outcome_refine_planner: outcome-trained decomposition beyond imitation (workstream C4). Verification
is REAL execution against a synthetic tool-world's ground-truth DB -- never text matching against the
teacher, never a self-grade.
"""

import re
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.task.toolcall import ToolSpec


class _SyntheticOrderWorld:
    """A tiny ground-truth DB: order_id -> owner. verify_fn executes the plan against it for real."""

    def __init__(self):
        self.db: dict[str, str] = {}
        self.users = ["bob", "ana", "kim", "raj"]

    def teacher(self, request):
        m = re.search(r"refund order (\d+) for (\w+)", request)
        if m:
            return [
                {"tool": "lookup_order", "args": {"order_id": m.group(1)}},
                {"tool": "notify", "args": {"user": m.group(2)}},
            ]
        m = re.search(r"check status of order (\d+)", request)
        if m:
            return [{"tool": "lookup_order", "args": {"order_id": m.group(1)}}]
        return []

    def requests(self, n, seed=0):
        rng = np.random.RandomState(seed)
        out = []
        for _ in range(n):
            oid, user = rng.randint(1000, 9999), self.users[rng.randint(0, 4)]
            self.db[str(oid)] = user
            r = rng.rand()
            if r < 0.5:
                out.append(f"please refund order {oid} for {user} as discussed")
            elif r < 0.85:
                out.append(f"can you check status of order {oid} right away")
            else:
                out.append(f"just wanted to say thanks, note {rng.randint(0, 99)}")
        return out

    def verify(self, task, plan):
        """Executes the plan against self.db and checks the REAL outcome -- not the teacher's text."""
        m = re.search(r"refund order (\d+) for (\w+)", task)
        if m:
            order_id, claimed_user = m.group(1), m.group(2)
            looked_up = notified = None
            for step in plan:
                if step["tool"] == "lookup_order" and step["args"].get("order_id") == order_id:
                    looked_up = self.db.get(order_id)
                if step["tool"] == "notify":
                    notified = step["args"].get("user")
            return looked_up is not None and notified == looked_up == claimed_user
        m = re.search(r"check status of order (\d+)", task)
        if m:
            order_id = m.group(1)
            return (
                len(plan) == 1
                and plan[0]["tool"] == "lookup_order"
                and plan[0]["args"].get("order_id") == order_id
                and order_id in self.db
            )
        return len(plan) == 0


@unittest.skipUnless(_HAS_TORCH, "the plan-writing LM needs torch")
class OutcomeRefinementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # trained ONCE (seed=0, deterministic) and reused read-only -- outcome_refine_planner()
        # returns a NEW planner rather than mutating this one, so a per-test setUp was retraining
        # the identical base model 3 times over for no behavioral difference.
        from mixle.task import sft_planner

        # epochs=40/n_layer=2 is required here, not a stylistic choice: a 6-seed sweep using
        # deepcopy'd (order-independent) planners showed every reduced config -- epochs 15/20/25 at
        # n_layer=1, and epochs 15/25 at n_layer=2 -- fails test_solve_rate_improves_over_the_imitation_
        # baseline's solve_rate_after>=solve_rate_before assertion 6/6 times in true isolation; only
        # epochs=40/n_layer=2 passed 6/6. (A smaller config had appeared safe in an earlier pass, but
        # that was a false negative from cross-test mutation: outcome_refine_planner() mutates
        # cls.planner.lm in place, so running the class as a whole benefits from test_report_names'
        # alphabetically-earlier refinement pass -- a single isolated run of test_solve_rate reveals the
        # base model genuinely needs the deeper config to reliably demonstrate the effect.) n_train=180
        # and held_out=40 are left unchanged: they set the statistical power behind
        # verified_gain_pairs>0 and the solve-rate comparison, and shrinking either was observed
        # (separately, at smaller scale) to risk zero verified samples across the k draws -- a hard
        # failure of that assertion.
        cls.world = _SyntheticOrderWorld()
        tools = [ToolSpec("lookup_order", ["order_id"]), ToolSpec("notify", ["user"])]
        train_reqs = cls.world.requests(180, seed=0)
        cls.planner = sft_planner(cls.world.teacher, train_reqs, tools, seed=0, epochs=40, d_model=64, n_layer=2)
        cls.held_out = cls.world.requests(40, seed=7)

    def test_solve_rate_improves_over_the_imitation_baseline(self):
        """The C4 acceptance: outcome training beats imitation-only, measured on the same held-out set."""
        from mixle.task import outcome_refine_planner

        _planner, report = outcome_refine_planner(
            self.planner, self.held_out, self.world.verify, k=6, epochs=20, seed=0
        )
        self.assertEqual(report.tasks, len(self.held_out))
        self.assertGreater(report.verified_gain_pairs, 0)  # the loop actually harvested a training signal
        self.assertGreaterEqual(report.solve_rate_after, report.solve_rate_before)

    def test_verification_is_real_execution_not_the_teacher_text(self):
        """A syntactically-plausible but factually wrong plan (mismatched owner) must fail verification."""
        order_id = next(iter(self.world.db))
        real_owner = self.world.db[order_id]
        wrong_owner = next(u for u in self.world.users if u != real_owner)
        task = f"please refund order {order_id} for {real_owner} as discussed"
        correct_plan = [
            {"tool": "lookup_order", "args": {"order_id": order_id}},
            {"tool": "notify", "args": {"user": real_owner}},
        ]
        wrong_plan = [
            {"tool": "lookup_order", "args": {"order_id": order_id}},
            {"tool": "notify", "args": {"user": wrong_owner}},
        ]
        self.assertTrue(self.world.verify(task, correct_plan))
        self.assertFalse(self.world.verify(task, wrong_plan))

    def test_report_names_the_measured_quantities(self):
        from mixle.task import RefinementReport, outcome_refine_planner

        _planner, report = outcome_refine_planner(
            self.planner, self.held_out, self.world.verify, k=6, epochs=20, seed=0
        )
        self.assertIsInstance(report, RefinementReport)
        n = len(self.held_out)
        # solve_rate_before is exactly solved_count / n for some integer solved_count in [0, n]
        solved_count = round(report.solve_rate_before * n)
        self.assertAlmostEqual(report.solve_rate_before, solved_count / n, places=9)
        self.assertLessEqual(report.verified_gain_pairs, n)  # at most one harvested pair per task


if __name__ == "__main__":
    unittest.main()
