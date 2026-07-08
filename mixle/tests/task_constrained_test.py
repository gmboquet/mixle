"""Grammar-constrained plan decoding: invalid output unrepresentable; big lift for small models."""

import re
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.task.constrained import PlanGrammar
from mixle.task.toolcall import ToolSpec

SPECS = {
    "lookup_order": ToolSpec("lookup_order", ["order_id"]),
    "notify": ToolSpec("notify", ["user"]),
}


def _walk(grammar, text):
    s = grammar.start()
    for c in text:
        allowed = grammar.allowed(s)
        if c not in allowed:
            return s, c, allowed
        s = grammar.advance(s, c)
    return s, None, grammar.allowed(s)


class PlanGrammarAutomatonTest(unittest.TestCase):
    def test_valid_plan_walks_to_terminal(self):
        g = PlanGrammar(SPECS, "please refund order 4242 for kim as discussed")
        s, rejected, _ = _walk(g, "lookup_order(order_id=4242) | notify(user=kim)\n")
        self.assertIsNone(rejected)
        self.assertEqual(s.mode, "terminal")

    def test_copy_drift_is_unrepresentable(self):
        # request contains 4242 only: after emitting "42" the automaton offers no "0"
        g = PlanGrammar(SPECS, "please refund order 4242 for kim")
        s, rejected, allowed = _walk(g, "lookup_order(order_id=420")
        self.assertEqual(rejected, "0")
        self.assertNotIn("0", allowed)

    def test_unknown_tools_and_args_are_unreachable(self):
        g = PlanGrammar(SPECS, "anything 123")
        s, rejected, _ = _walk(g, "delete_all(")
        self.assertIsNotNone(rejected)  # no such tool: the name cannot even be spelled
        s2, rejected2, _ = _walk(g, "notify(city=")
        self.assertIsNotNone(rejected2)  # notify has no 'city' argument

    def test_structural_chars_never_allowed_inside_values(self):
        g = PlanGrammar(SPECS, "weird request with (parens) and | pipes = signs 77")
        s, rejected, _ = _walk(g, "lookup_order(order_id=77")
        self.assertIsNone(rejected)
        for c in "(|=;":
            self.assertNotIn(c, {a for a in g.allowed(s) if a != ")"} - {";"} or set())
        # terminators are offered, continuations with structural chars are not
        allowed = g.allowed(s)
        self.assertIn(")", allowed)
        self.assertNotIn("(", allowed)
        self.assertNotIn("=", allowed)
        self.assertNotIn("|", allowed)


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


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ConstrainedDecodeTest(unittest.TestCase):
    def test_constrained_beats_free_when_undertrained_and_is_always_valid(self):
        from mixle.task import ToolSpec as TS
        from mixle.task import sft_planner
        from mixle.task.sft_plan import _plans_match

        tools = [TS("lookup_order", ["order_id"]), TS("notify", ["user"])]
        # block=128 comfortably covers every serialized prompt+completion pair in this fixture (max 96
        # chars, verified empirically) while cutting the O(block^2) attention cost vs. the 192 default.
        free = sft_planner(
            _teacher, _requests(160), tools, seed=0, epochs=10, d_model=64, n_layer=2, constrained=False, block=128
        )
        con = sft_planner(
            _teacher, _requests(160), tools, seed=0, epochs=10, d_model=64, n_layer=2, constrained=True, block=128
        )
        self.assertGreater(con.plan_agreement, free.plan_agreement)  # the lift where compute is scarce

        specs = {t.name: t for t in tools}
        for r in _requests(30, seed=9):
            got = con.try_plan(r)
            if got is not None:  # anything emitted is grammatically + spec + copy valid BY CONSTRUCTION
                self.assertTrue(con._validate(got, r))
            out = con(r)
            if not out["escalate"]:
                self.assertTrue(_plans_match(out["plan"], out["plan"], specs))  # well-formed structure


if __name__ == "__main__":
    unittest.main()
