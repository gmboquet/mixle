"""IC-5 conformance: the frozen trace/receipt envelope (notes/exec/contracts.md).

Landed alongside E7 (cross-chain provenance receipt), which is the first real consumer of
``validate_trace_record`` in this repository.
"""

import pytest

from mixle.task.replay import ExecutionTrace, TraceStep
from mixle.task.trace_record import STEP_KEYS, TRACE_KEYS, from_execution_trace, validate_trace_record


def test_frozen_keys():
    assert TRACE_KEYS == ("prompt", "steps", "outcome", "provenance")
    assert STEP_KEYS == ("tool", "args", "result", "model", "verdict")


def test_validate_accepts_a_minimal_record():
    validate_trace_record(
        {"prompt": "p", "steps": [{"tool": "t", "args": {}, "result": 1}], "outcome": "ok", "provenance": {}}
    )


def test_validate_rejects_missing_top_key():
    with pytest.raises(ValueError):
        validate_trace_record({"prompt": "p", "steps": [], "outcome": None})


def test_validate_rejects_bad_step():
    with pytest.raises(ValueError):
        validate_trace_record({"prompt": "p", "steps": [{"tool": "t"}], "outcome": None, "provenance": {}})


def test_execution_trace_conversion_preserves_steps_results_and_seeds():
    trace = ExecutionTrace(
        request="demo",
        steps=[TraceStep(tool="sample", args={"n": 0}, seed=7, result={"value": False})],
    )
    record = from_execution_trace(trace, outcome="ok", provenance={"run_id": "r1"})
    validate_trace_record(record)
    assert record["prompt"] == "demo"
    assert record["steps"] == [
        {
            "tool": "sample",
            "args": {"n": 0},
            "result": {"value": False},
            "model": None,
            "verdict": None,
        }
    ]
    assert record["provenance"]["run_id"] == "r1"
    assert record["provenance"]["step_seeds"] == [7]


def test_validate_rejects_unknown_keys_and_bad_container_types():
    with pytest.raises(ValueError):
        validate_trace_record({"prompt": "p", "steps": [], "outcome": None, "provenance": {}, "unexpected": True})
    with pytest.raises(ValueError):
        validate_trace_record({"prompt": "p", "steps": {}, "outcome": None, "provenance": {}})
