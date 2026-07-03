"""distill_planner: tiny models that decompose a request into verified steps."""

import re
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _teacher(request):
    """The 'frontier planner' being distilled: request -> multi-step plan (rule-based stand-in)."""
    m = re.search(r"refund order (\d+) for (\w+)", request)
    if m:
        oid, user = m.group(1), m.group(2)
        return [
            {"tool": "lookup_order", "args": {"order_id": oid}},
            {"tool": "create_ticket", "args": {"kind": "refund", "order_id": oid}},
            {"tool": "notify", "args": {"user": user}},
        ]
    m = re.search(r"check status of order (\d+)", request)
    if m:
        return [{"tool": "lookup_order", "args": {"order_id": m.group(1)}}]
    m = re.search(r"message (\w+) saying", request)
    if m:
        return [{"tool": "notify", "args": {"user": m.group(1)}}]
    return []


def _requests(n, seed=0):
    rng = np.random.RandomState(seed)
    users = ["bob", "ana", "kim", "raj", "lee"]
    out = []
    for _ in range(n):
        r = rng.rand()
        oid, user = rng.randint(1000, 9999), users[rng.randint(0, 5)]
        if r < 0.45:
            out.append(f"please refund order {oid} for {user} as discussed")
        elif r < 0.75:
            out.append(f"can you check status of order {oid} right away")
        elif r < 0.9:
            out.append(f"message {user} saying the fix shipped")
        else:
            out.append(f"just wanted to say thanks, note {rng.randint(0, 99)}")
    return out


TOOLS = None  # filled in setUpModule to avoid import-at-collect issues


def setUpModule():
    global TOOLS
    from mixle.task import ToolSpec

    TOOLS = [
        ToolSpec("lookup_order", ["order_id"]),
        ToolSpec("create_ticket", ["kind", "order_id"]),
        ToolSpec("notify", ["user"]),
    ]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PlannerTest(unittest.TestCase):
    def test_decomposes_verifies_and_escalates(self):
        from mixle.task import distill_planner

        planner = distill_planner(
            _teacher,
            _requests(300),
            TOOLS,
            seed=0,
            selector_kw={"ood": None, "epochs": 250},
            extractor_kw={"epochs": 40},
        )
        # plan-level holdout agreement: exact tool+args match, in order, on unseen requests
        self.assertGreater(planner.plan_agreement, 0.6)

        # a 3-step decomposition, executed and verified step by step
        calls = []
        execute = {
            "lookup_order": lambda order_id: calls.append(("lookup", order_id)) or {"status": "ok"},
            "create_ticket": lambda kind, order_id: calls.append(("ticket", kind, order_id)) or "T-1",
            "notify": lambda user: calls.append(("notify", user)) or "sent",
        }
        got = planner("please refund order 4242 for kim as discussed", execute=execute)
        want = _teacher("please refund order 4242 for kim as discussed")
        # escalated or not, the returned plan must be the CORRECT plan and fully executed
        self.assertEqual([s["tool"] for s in got["plan"]], [s["tool"] for s in want])
        self.assertEqual(len(got["results"]), len(want))
        self.assertEqual(calls[0], ("lookup", "4242"))

        # chatter decomposes to the empty plan (STOP at step 0) — the plan-vs-direct decision for free
        small = planner("just wanted to say thanks, note 7")
        self.assertEqual(small["plan"], [])

        rep = planner.report()
        self.assertEqual(rep["requests"], 2)
        self.assertEqual(rep["harvested_traces"], rep["escalated"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PlannerPersistenceTest(unittest.TestCase):
    def test_save_load_plans_identically(self):
        import tempfile

        from mixle.task import Planner, distill_planner

        planner = distill_planner(
            _teacher,
            _requests(250),
            TOOLS,
            seed=0,
            selector_kw={"ood": None, "epochs": 200},
            extractor_kw={"epochs": 30},
        )
        fresh = _requests(50, seed=6)
        want = [planner(r) for r in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = planner.save(d + "/planner")
            back = Planner.load(path, _teacher)
            got = [back(r) for r in fresh]
        self.assertEqual(got, want)
        self.assertAlmostEqual(back.plan_agreement, planner.plan_agreement, places=6)


if __name__ == "__main__":
    unittest.main()
