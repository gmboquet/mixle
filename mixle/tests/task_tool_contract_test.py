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
