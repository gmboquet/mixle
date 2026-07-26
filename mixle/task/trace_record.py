"""IC-5 — the frozen trace/receipt envelope the dataset foundry (M4) mines (work-plan §5, DR-ALG M4).

Frozen top-level keys: ``prompt, steps, outcome, provenance``; each step: ``tool, args, result, model, verdict``.
``validate_trace_record`` enforces the names so a producer (executor/gateway/receipt) and the consumer (M4 trace
miner) never drift. ``from_execution_trace`` lifts a core ``mixle.task.replay.ExecutionTrace`` into the envelope.

This module was specified (frozen) in ``notes/exec/contracts.md`` §IC-5. E7
(cross-chain provenance receipt) uses ``validate_trace_record`` to shape its
lineage receipt, while ``from_execution_trace`` performs the public lossless
lift from replay traces into the frozen envelope.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TraceStepRecord(TypedDict, total=False):
    tool: str
    args: dict[str, Any]
    result: Any
    model: str | None
    verdict: dict[str, Any] | None  # an IC-6 Verdict as a dict, when the step was verified


class TraceRecord(TypedDict, total=False):
    prompt: str
    steps: list[TraceStepRecord]
    outcome: Any
    provenance: dict[str, Any]


TRACE_KEYS = ("prompt", "steps", "outcome", "provenance")
STEP_KEYS = ("tool", "args", "result", "model", "verdict")


def validate_trace_record(d: dict[str, Any]) -> None:
    """Validate the complete frozen trace-record schema."""
    if not isinstance(d, dict):
        raise ValueError("trace record must be a dictionary")
    missing = [k for k in TRACE_KEYS if k not in d]
    if missing:
        raise ValueError(f"trace record missing frozen keys: {missing}")
    extra = [k for k in d if k not in TRACE_KEYS]
    if extra:
        raise ValueError(f"trace record has unknown keys: {extra}")
    if not isinstance(d["prompt"], str):
        raise ValueError("trace record prompt must be a string")
    if not isinstance(d["steps"], list):
        raise ValueError("trace record steps must be a list")
    if not isinstance(d["provenance"], dict):
        raise ValueError("trace record provenance must be a dictionary")
    for i, s in enumerate(d["steps"]):
        if not isinstance(s, dict):
            raise ValueError(f"step {i} must be a dictionary")
        for k in ("tool", "args", "result"):
            if k not in s:
                raise ValueError(f"step {i} missing frozen key {k!r}")
        extra_step = [k for k in s if k not in STEP_KEYS]
        if extra_step:
            raise ValueError(f"step {i} has unknown keys: {extra_step}")
        if not isinstance(s["tool"], str) or not s["tool"]:
            raise ValueError(f"step {i} tool must be a non-empty string")
        if not isinstance(s["args"], dict):
            raise ValueError(f"step {i} args must be a dictionary")
        if s.get("model") is not None and not isinstance(s["model"], str):
            raise ValueError(f"step {i} model must be a string or None")
        if s.get("verdict") is not None and not isinstance(s["verdict"], dict):
            raise ValueError(f"step {i} verdict must be a dictionary or None")


def from_execution_trace(trace: Any, *, outcome: Any = None, provenance: dict[str, Any] | None = None) -> TraceRecord:
    """Lift a `mixle.task.replay.ExecutionTrace` into the frozen envelope (fills ``model``/``verdict`` as None)."""
    from mixle.task.replay import ExecutionTrace

    if not isinstance(trace, ExecutionTrace):
        raise TypeError("trace must be an ExecutionTrace")
    if provenance is not None and not isinstance(provenance, dict):
        raise TypeError("provenance must be a dictionary or None")
    prov = dict(provenance or {})
    prov.setdefault("source_type", "mixle.task.replay.ExecutionTrace")
    prov.setdefault("step_seeds", [step.seed for step in trace.steps])
    record: TraceRecord = {
        "prompt": trace.request,
        "steps": [
            {
                "tool": step.tool,
                "args": dict(step.args),
                "result": step.result,
                "model": None,
                "verdict": None,
            }
            for step in trace.steps
        ],
        "outcome": outcome,
        "provenance": prov,
    }
    validate_trace_record(record)
    return record
