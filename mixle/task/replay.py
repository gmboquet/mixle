"""Replayable execution traces -- record each step an executor took (tool, args, seed, result) as a
plain JSON-able object, and re-run it later to prove the run was deterministic.

A step is only trustworthy to replay if every source of randomness it used is named and captured --
that is the whole point of recording ``seed`` per step rather than trusting global RNG state. ``replay``
re-invokes each step's registered tool with the same args and seed and returns a new
:class:`ExecutionTrace`; ``diff`` is the per-step comparison (bit-identical or not), never a
silent pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    """One recorded step: the tool name, the args it ran with, the seed (if any), and its result."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    result: Any = None

    def to_json(self) -> dict[str, Any]:
        """Serialize this trace step to JSON-compatible data."""
        return {"tool": self.tool, "args": self.args, "seed": self.seed, "result": self.result}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TraceStep:
        """Reconstruct a trace step from JSON-compatible data."""
        return cls(tool=d["tool"], args=dict(d.get("args") or {}), seed=d.get("seed"), result=d.get("result"))


@dataclass
class ExecutionTrace:
    """An ordered list of :class:`TraceStep` -- JSON-serializable, so it can be stored (e.g. as a
    ``mixle.substrate`` ``"trace"`` item) and replayed in a fresh process."""

    request: str
    steps: list[TraceStep] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize the full execution trace to JSON-compatible data."""
        return {"request": self.request, "steps": [s.to_json() for s in self.steps]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ExecutionTrace:
        """Reconstruct an execution trace from JSON-compatible data."""
        return cls(request=d["request"], steps=[TraceStep.from_json(s) for s in d.get("steps") or []])

    def dumps(self) -> str:
        """Serialize the execution trace to a stable JSON string."""
        return json.dumps(self.to_json(), sort_keys=True)


def record_step(
    tools: dict[str, Callable[..., Any]], tool: str, args: dict[str, Any], *, seed: int | None = None
) -> TraceStep:
    """Run ``tools[tool]`` once with ``args`` (and ``seed``, if the tool accepts one), recording the result."""
    fn = tools[tool]
    call_args = dict(args)
    if seed is not None:
        call_args["seed"] = seed
    result = fn(**call_args)
    return TraceStep(tool=tool, args=dict(args), seed=seed, result=result)


def replay(trace: ExecutionTrace, tools: dict[str, Callable[..., Any]]) -> ExecutionTrace:
    """Re-execute every step of ``trace`` against ``tools`` with the exact same args and seed."""
    replayed = [record_step(tools, step.tool, step.args, seed=step.seed) for step in trace.steps]
    return ExecutionTrace(request=trace.request, steps=replayed)


def diff(a: ExecutionTrace, b: ExecutionTrace) -> list[tuple[int, str]]:
    """Indices + tool names where ``a`` and ``b`` disagree (JSON-serialized result comparison)."""
    mismatches = []
    for i, (sa, sb) in enumerate(zip(a.steps, b.steps)):
        if sa.tool != sb.tool or json.dumps(sa.result, sort_keys=True) != json.dumps(sb.result, sort_keys=True):
            mismatches.append((i, sa.tool))
    if len(a.steps) != len(b.steps):
        mismatches.append((min(len(a.steps), len(b.steps)), "length_mismatch"))
    return mismatches


def is_bit_identical_replay(trace: ExecutionTrace, tools: dict[str, Callable[..., Any]]) -> bool:
    """Replay ``trace`` and return whether every step reproduces exactly."""
    return not diff(trace, replay(trace, tools))
