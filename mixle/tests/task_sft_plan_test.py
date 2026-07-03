"""sft_planner: a plan-writing LM behind the parse/spec/copy-fidelity gate — never silently wrong."""

import re
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.task.sft_plan import _parse_plan, _plans_match, _serialize_plan


def _teacher(request):
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


def _requests(n, seed=0):
    rng = np.random.RandomState(seed)
    users = ["bob", "ana", "kim", "raj"]
    out = []
    for _ in range(n):
        oid, user = rng.randint(1000, 9999), users[rng.randint(0, 4)]
        r = rng.rand()
        if r < 0.5:
            out.append(f"please refund order {oid} for {user} as discussed")
        elif r < 0.85:
            out.append(f"can you check status of order {oid} right away")
        else:
            out.append(f"just wanted to say thanks, note {rng.randint(0, 99)}")
    return out


class PlanGrammarTest(unittest.TestCase):
    def test_serialize_parse_round_trip(self):
        plan = [
            {"tool": "lookup_order", "args": {"order_id": "4242"}},
            {"tool": "notify", "args": {"user": "kim"}},
        ]
        self.assertEqual(_parse_plan(_serialize_plan(plan)), plan)
        self.assertEqual(_parse_plan(_serialize_plan([])), [])

    def test_malformed_text_is_rejected_not_guessed(self):
        for bad in ("lookup_order(order_id=", "notify user=kim)", "do(x=1) & do(y=2)", "notify(=kim)"):
            self.assertIsNone(_parse_plan(bad + "\n"))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class CopyFidelityGateTest(unittest.TestCase):
    def test_copied_values_must_occur_in_the_request(self):
        from mixle.task import ToolSpec
        from mixle.task.sft_plan import GenerativePlanner, _CharCodec

        gp = GenerativePlanner(
            lm=None,
            codec=_CharCodec(["x"]),
            tools={"lookup_order": ToolSpec("lookup_order", ["order_id"])},
            teacher=_teacher,
            plan_agreement=0.0,
        )
        req = "please refund order 4242 for kim as discussed"
        good = [{"tool": "lookup_order", "args": {"order_id": "4242"}}]
        drifted = [{"tool": "lookup_order", "args": {"order_id": "4202"}}]  # the silent copy error
        self.assertTrue(gp._validate(good, req))
        self.assertFalse(gp._validate(drifted, req))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SftPlannerTest(unittest.TestCase):
    def test_generates_verified_plans_never_silently_wrong(self):
        from mixle.task import ToolSpec, sft_planner

        tools = [ToolSpec("lookup_order", ["order_id"]), ToolSpec("notify", ["user"])]
        planner = sft_planner(_teacher, _requests(180), tools, seed=0, epochs=40, d_model=64, n_layer=2)

        specs = {t.name: t for t in tools}
        silent_wrong = 0
        for r in _requests(40, seed=7):
            out = planner(r)
            if not out["escalate"] and not _plans_match(out["plan"], _teacher(r), specs):
                silent_wrong += 1
        self.assertEqual(silent_wrong, 0)  # THE invariant: the gate lets no wrong plan out
        rep = planner.report()
        self.assertEqual(rep["requests"], 40)
        self.assertEqual(rep["harvested_traces"], rep["escalated"])


if __name__ == "__main__":
    unittest.main()
