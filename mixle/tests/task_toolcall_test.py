"""distill_tool_caller: a tiny model that does function calling, with escalation honesty."""

import re
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _teacher(request):
    """The 'frontier' being distilled: parses requests into tool calls (rule-based here, an LLM in life)."""
    m = re.search(r"weather (?:in|for) (\w+)", request)
    if m:
        return {"tool": "get_weather", "args": {"city": m.group(1)}}
    m = re.search(r"ticket .* kind (\w+) .* amount (\d+)", request)
    if m:
        return {"tool": "create_ticket", "args": {"kind": m.group(1), "amount": m.group(2)}}
    m = re.search(r"search for (.+)$", request)
    if m:
        return {"tool": "search", "args": {"query": m.group(1)}}
    return {"tool": None, "args": {}}


def _requests(n, seed=0):
    rng = np.random.RandomState(seed)
    cities = ["paris", "tokyo", "denver", "oslo", "lima"]
    kinds = ["refund", "billing", "bug"]
    out = []
    for _ in range(n):
        r = rng.rand()
        if r < 0.3:
            out.append(f"please tell me the weather in {cities[rng.randint(0, 5)]} today")
        elif r < 0.6:
            out.append(f"open a ticket for me kind {kinds[rng.randint(0, 3)]} with amount {rng.randint(10, 900)} now")
        elif r < 0.85:
            out.append(f"can you search for item {rng.randint(1000, 9999)}")
        else:
            out.append(f"thanks for the help, note {rng.randint(0, 99)}")
    return out


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ToolCallerTest(unittest.TestCase):
    def test_distills_selection_and_arguments_with_escalation(self):
        from mixle.task import ToolSpec, distill_tool_caller

        tools = [
            ToolSpec("get_weather", ["city"]),
            ToolSpec("create_ticket", ["kind", "amount"]),
            ToolSpec("search", ["query"]),
        ]
        tc = distill_tool_caller(
            _teacher,
            _requests(300),
            tools,
            seed=0,
            selector_kw={"ood": None, "epochs": 250},
            extractor_kw={"epochs": 40},
        )
        self.assertGreater(tc.selection_agreement, 0.85)

        # a clean weather request -> a locally-emitted, correctly-argued call
        out = tc("please tell me the weather in tokyo today")
        if not out["escalate"]:
            self.assertEqual(out["tool"], "get_weather")
            self.assertEqual(out["args"], {"city": "tokyo"})

        # no-op requests pass through as tool=None (locally or via the teacher — never a fabricated call)
        out2 = tc("thanks for the help, note 5")
        self.assertIsNone(out2["tool"])

        # end-to-end over fresh traffic: EVERY emitted call is valid; escalations carry the teacher's call
        wrong_calls = 0
        for r in _requests(120, seed=9):
            got = tc(r)
            want = _teacher(r)
            if got["escalate"]:
                self.assertEqual(got["tool"], want["tool"])  # escalation = the teacher's own call
            elif got["tool"] is not None:
                spec = {t.name: t for t in tools}[got["tool"]]
                self.assertTrue(all(got["args"].get(a) for a in spec.required_args))  # never malformed
                wrong_calls += int(got["tool"] != want["tool"])
        self.assertLess(wrong_calls / 120, 0.12)  # alpha-bounded selection risk

        rep = tc.report()
        self.assertEqual(rep["requests"], 2 + 120)
        self.assertEqual(rep["harvested_traces"], rep["escalated"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ToolCallerPersistenceTest(unittest.TestCase):
    def test_save_load_serves_identically(self):
        import tempfile

        from mixle.task import ToolCaller, ToolSpec, distill_tool_caller

        tools = [
            ToolSpec("get_weather", ["city"]),
            ToolSpec("create_ticket", ["kind", "amount"]),
            ToolSpec("search", ["query"]),
        ]
        tc = distill_tool_caller(
            _teacher,
            _requests(250),
            tools,
            seed=0,
            selector_kw={"ood": None, "epochs": 200},
            extractor_kw={"epochs": 30},
        )
        fresh = _requests(60, seed=5)
        want = [tc(r) for r in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = tc.save(d + "/caller")
            back = ToolCaller.load(path, _teacher)
            got = [back(r) for r in fresh]
        self.assertEqual(got, want)  # identical calls, escalations included, in a fresh process
        self.assertAlmostEqual(back.selection_agreement, tc.selection_agreement, places=6)


if __name__ == "__main__":
    unittest.main()
