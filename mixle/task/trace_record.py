"""IC-5 — the frozen trace/receipt envelope the dataset foundry (M4) mines (work-plan §5, DR-ALG M4).

Frozen top-level keys: ``prompt, steps, outcome, provenance``; each step: ``tool, args, result, model, verdict``.
``validate_trace_record`` enforces the names so a producer (executor/gateway/receipt) and the consumer (M4 trace
miner) never drift. ``from_execution_trace`` lifts a core ``mixle.task.replay.ExecutionTrace`` into the envelope.

This module was specified (frozen) in ``notes/exec/contracts.md`` §IC-5 as a Wave-0 deliverable, but had not
actually landed in this repository yet -- E7 (cross-chain provenance receipt) requires ``validate_trace_record``
to shape its lineage receipt, so this lands the frozen stub verbatim. Only ``from_execution_trace``'s body
remains a stub: that lift is M4a's job, not E7's.
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
    """Raise ``ValueError`` if ``d`` is missing a frozen top-level key or any step lacks ``tool``/``args``/``result``."""
    missing = [k for k in TRACE_KEYS if k not in d]
    if missing:
        raise ValueError(f"trace record missing frozen keys: {missing}")
    for i, s in enumerate(d.get("steps") or []):
        for k in ("tool", "args", "result"):
            if k not in s:
                raise ValueError(f"step {i} missing frozen key {k!r}")


def from_execution_trace(trace: Any, *, outcome: Any = None, provenance: dict[str, Any] | None = None) -> TraceRecord:
    """Lift a `mixle.task.replay.ExecutionTrace` into the frozen envelope (fills ``model``/``verdict`` as None)."""
    raise NotImplementedError("M4a")
