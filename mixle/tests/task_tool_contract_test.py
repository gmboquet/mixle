"""Fail-closed function-call and planner schema contracts."""

import unittest
from types import SimpleNamespace

from mixle.task.plan import Planner
from mixle.task.sft_plan import GenerativePlanner, _parse_plan
from mixle.task.toolcall import ToolCaller, ToolSpec, _tool_spec_map


def _selector(tool):
    model = SimpleNamespace(decide=lambda _request: tool)
    return SimpleNamespace(cascade=SimpleNamespace(model=model))


def _sequence_selector(*tools):
    remaining = iter(tools)
    model = SimpleNamespace(decide=lambda _request: next(remaining))
    return SimpleNamespace(cascade=SimpleNamespace(model=model))


class ToolSpecContractTest(unittest.TestCase):
    def test_duplicate_tools_and_invalid_argument_schemas_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate tool"):
            _tool_spec_map([ToolSpec("ping", []), ToolSpec("ping", [])])
        with self.assertRaises(ValueError):
            ToolSpec("ping", ["value"], required=["missing"])
        with self.assertRaises(ValueError):
            ToolSpec("ping", ["value", "value"])

    def test_required_arguments_are_presence_based(self):
        spec = ToolSpec("set_flag", ["value"])
        for value in (0, False, "", [], {}):
            with self.subTest(value=value):
                caller = ToolCaller(
                    selector=_selector("set_flag"),
                    extractors={"set_flag": lambda _request, value=value: {"value": value}},
                    tools={"set_flag": spec},
                    teacher=lambda _request: {"tool": None, "args": {}},
                    selection_agreement=1.0,
                )
                self.assertEqual(caller.try_local("request"), {"tool": "set_flag", "args": {"value": value}})

    def test_argless_tool_needs_no_extractor(self):
        spec = ToolSpec("ping", [])
        caller = ToolCaller(
            selector=_selector("ping"),
            extractors={},
            tools={"ping": spec},
            teacher=lambda _request: {"tool": None, "args": {}},
            selection_agreement=1.0,
        )
        self.assertEqual(caller.try_local("ping"), {"tool": "ping", "args": {}})

        planner = Planner(
            selector=_sequence_selector("ping", "__stop__"),
            extractors={},
            tools={"ping": spec},
            teacher=lambda _request: [],
            plan_agreement=1.0,
            max_steps=2,
        )
        executed = []
        self.assertEqual(
            planner.try_plan("ping", execute={"ping": lambda: executed.append(True)}),
            {"plan": [{"tool": "ping", "args": {}}], "results": [None]},
        )
        self.assertEqual(executed, [True])

    def test_planner_never_replays_committed_steps_after_partial_failure(self):
        tools = {"first": ToolSpec("first", []), "second": ToolSpec("second", [])}
        teacher_calls = []
        planner = Planner(
            selector=_sequence_selector("first", "second", "__stop__"),
            extractors={},
            tools=tools,
            teacher=lambda request: (
                teacher_calls.append(request) or [{"tool": "first", "args": {}}, {"tool": "second", "args": {}}]
            ),
            plan_agreement=1.0,
            max_steps=3,
        )
        committed = []

        def fail():
            raise RuntimeError("boom")

        result = planner(
            "do it",
            execute={"first": lambda: committed.append("first") or "ok", "second": fail},
        )
        self.assertTrue(result["partial"])
        self.assertTrue(result["escalate"])
        self.assertEqual(result["committed_steps"], 1)
        self.assertEqual(committed, ["first"])
        self.assertEqual(teacher_calls, [])  # no fallback plan can replay the committed action

    def test_complete_execution_policy_and_teacher_plan_are_validated_before_actions(self):
        tools = {"first": ToolSpec("first", []), "second": ToolSpec("second", ["value"])}
        committed = []
        planner = Planner(
            selector=_sequence_selector("first", "second", "__stop__"),
            extractors={"second": lambda _request: {"value": 1}},
            tools=tools,
            teacher=lambda _request: [{"tool": "second", "args": {}}],
            plan_agreement=1.0,
            max_steps=3,
        )
        with self.assertRaisesRegex(ValueError, "no callable"):
            planner.try_plan("do it", execute={"first": lambda: committed.append("first")})
        self.assertEqual(committed, [])

        planner.selector = _selector(None)
        with self.assertRaisesRegex(ValueError, "missing required"):
            planner("fallback", execute={"second": lambda value: committed.append(value)})
        self.assertEqual(committed, [])


class GenerativePlanContractTest(unittest.TestCase):
    def test_falsey_required_value_is_valid_and_duplicate_argument_text_is_not(self):
        spec = ToolSpec("set_flag", ["value"])
        planner = GenerativePlanner(
            lm=None,
            codec=None,
            tools={"set_flag": spec},
            teacher=lambda _request: [],
            plan_agreement=1.0,
        )
        self.assertTrue(planner._validate([{"tool": "set_flag", "args": {"value": 0}}], "set flag to 0"))
        self.assertIsNone(_parse_plan("set_flag(value=0; value=1)\n"))


if __name__ == "__main__":
    unittest.main()
