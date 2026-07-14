"""IC-5 conformance: the frozen trace/receipt envelope (notes/exec/contracts.md).

Landed alongside E7 (cross-chain provenance receipt), which is the first real consumer of
``validate_trace_record`` in this repository.
"""

import pytest

from mixle.task.trace_record import STEP_KEYS, TRACE_KEYS, validate_trace_record


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
