"""harvest_agent_traces: the agent's own history becomes teacher traces for the distillers."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _convo(cid, turns):
    """turns: list of (role, text, tool_uses)."""
    msgs = []
    for role, text, uses in turns:
        content = []
        if text:
            content.append({"type": "text", "text": text})
        for name, args in uses:
            content.append({"type": "tool_use", "id": f"tu_{len(msgs)}", "name": name, "input": args})
        msgs.append({"role": role, "content": content})
    return {"id": cid, "title": "t", "createdAt": 0, "updatedAt": 0, "messages": msgs}


def _store(tmp):
    docs = [
        _convo(
            "c1",
            [
                ("user", "check the weather in tokyo please", []),
                ("assistant", "", [("get_weather", {"city": "tokyo"})]),
                ("assistant", "It is sunny in tokyo.", []),
                ("user", "thanks!", []),
                ("assistant", "You're welcome.", []),
            ],
        ),
        _convo(
            "c2",
            [
                ("user", "refund order 4242 for kim", []),
                ("assistant", "", [("lookup_order", {"order_id": "4242"}), ("notify", {"user": "kim", "cc": "ops"})]),
                ("assistant", "Done.", []),
            ],
        ),
        _convo(
            "c3",
            [
                ("user", "check the weather in oslo now", []),
                ("assistant", "", [("get_weather", {"city": "oslo"})]),
                ("assistant", "Cold.", []),
            ],
        ),
    ]
    for d in docs:
        (Path(tmp) / f"{d['id']}.json").write_text(json.dumps(d))


class TraceHarvestTest(unittest.TestCase):
    def test_harvests_requests_plans_and_infers_specs(self):
        from mixle.task import harvest_agent_traces

        with tempfile.TemporaryDirectory() as tmp:
            _store(tmp)
            traces = harvest_agent_traces(tmp)

        self.assertEqual(len(traces), 4)  # incl. the no-op "thanks!" turn
        by_req = {t.request: t for t in traces.traces}
        self.assertEqual([s["tool"] for s in by_req["refund order 4242 for kim"].plan], ["lookup_order", "notify"])
        self.assertEqual(by_req["thanks!"].plan, [])
        self.assertEqual(by_req["check the weather in tokyo please"].reply, "It is sunny in tokyo.")

        specs = {t.name: t for t in traces.tool_specs()}
        self.assertEqual(specs["get_weather"].args, ["city"])
        self.assertEqual(specs["get_weather"].required, ["city"])
        # notify was called once with cc: union has it, and (single call) it is also required
        self.assertEqual(specs["notify"].args, ["cc", "user"])

        call = traces.call_teacher()
        self.assertEqual(call("refund order 4242 for kim")["tool"], "lookup_order")
        self.assertEqual(call("thanks!"), {"tool": None, "args": {}})
        plan = traces.plan_teacher()
        self.assertEqual(len(plan("refund order 4242 for kim")), 2)

    def test_unreadable_files_are_skipped(self):
        from mixle.task import harvest_agent_traces

        with tempfile.TemporaryDirectory() as tmp:
            _store(tmp)
            (Path(tmp) / "junk.json").write_text("{not json")
            traces = harvest_agent_traces(tmp)
        self.assertEqual(len(traces), 4)

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_harvested_traces_feed_the_distillers(self):
        import numpy as np

        from mixle.task import distill_tool_caller, harvest_agent_traces

        rng = np.random.RandomState(0)
        cities = ["tokyo", "oslo", "paris", "lima"]
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(120):  # a synthetic history large enough to distill from
                city = cities[rng.randint(0, 4)]
                if rng.rand() < 0.75:
                    doc = _convo(
                        f"w{i}",
                        [
                            ("user", f"check the weather in {city} please, ref {rng.randint(10, 99)}", []),
                            ("assistant", "", [("get_weather", {"city": city})]),
                            ("assistant", "ok", []),
                        ],
                    )
                else:
                    doc = _convo(
                        f"n{i}",
                        [
                            ("user", f"thanks for the help, note {rng.randint(0, 99)}", []),
                            ("assistant", "welcome", []),
                        ],
                    )
                (Path(tmp) / f"{doc['id']}.json").write_text(json.dumps(doc))
            traces = harvest_agent_traces(tmp)

        tc = distill_tool_caller(
            traces.call_teacher(),
            traces.requests(),
            traces.tool_specs(),
            seed=0,
            selector_kw={"ood": None, "epochs": 150},
            extractor_kw={"epochs": 30},
        )
        self.assertGreater(tc.selection_agreement, 0.8)  # the agent's own history taught the tiny model

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_harvested_traces_feed_the_plan_writer(self):
        """workstream C: harvest_agent_traces -> sft_planner, exactly the module docstring's own example."""
        import numpy as np

        from mixle.task import harvest_agent_traces, sft_planner

        rng = np.random.RandomState(1)
        cities = ["tokyo", "oslo", "paris", "lima"]
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(120):
                city = cities[rng.randint(0, 4)]
                doc = _convo(
                    f"w{i}",
                    [
                        ("user", f"check the weather in {city} please, ref {rng.randint(10, 99)}", []),
                        ("assistant", "", [("get_weather", {"city": city})]),
                        ("assistant", "ok", []),
                    ],
                )
                (Path(tmp) / f"{doc['id']}.json").write_text(json.dumps(doc))
            traces = harvest_agent_traces(tmp)

        # block=96 comfortably covers every serialized prompt+completion pair here (max 69 chars, verified
        # empirically), cutting the default block=192's O(block^2) attention cost; epochs=25 keeps the same
        # plan_agreement (0.9167) as the original epochs=40 (verified the epochs=18->20 boundary is where it
        # drops off), so 25 keeps a solid margin while training ~3.5x faster overall.
        planner = sft_planner(
            traces.plan_teacher(),
            traces.requests(min_steps=1),
            traces.tool_specs(),
            seed=0,
            epochs=25,
            d_model=64,
            n_layer=2,
            block=96,
        )
        self.assertGreater(planner.plan_agreement, 0.8)  # the agent's own history taught the plan writer


if __name__ == "__main__":
    unittest.main()
