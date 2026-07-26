"""Replayable execution traces -- record each step an executor took (tool, args, seed, result) as a
plain JSON-able object, and re-run it later to prove the run was deterministic.

A step is only trustworthy to replay if every source of randomness it used is named and captured --
that is the whole point of recording ``seed`` per step rather than trusting global RNG state. ``replay``
re-invokes each step's registered tool with the same args and seed and returns a new
:class:`ExecutionTrace`; ``diff`` is the per-step comparison (bit-identical or not), never a
silent pass.
"""

from __future__ import annotations

import inspect
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": list(random.getstate()),
        "numpy": {
            "kind": np_state[0],
            "keys": np_state[1].tolist(),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
    }


def _nested_tuple(value: Any) -> Any:
    return tuple(_nested_tuple(item) for item in value) if isinstance(value, list) else value


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or not isinstance(state.get("numpy"), dict):
        raise ValueError("trace RNG state is malformed")
    random.setstate(_nested_tuple(state["python"]))
    np_state = state["numpy"]
    np.random.set_state(
        (
            str(np_state["kind"]),
            np.asarray(np_state["keys"], dtype=np.uint32),
            int(np_state["position"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        )
    )


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("replay identity must be canonical JSON data") from exc

@dataclass
class TraceStep:
    """One recorded step: the tool name, the args it ran with, the seed (if any), and its result."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    result: Any = None
    action: dict[str, Any] | None = None
    state_before: Any = None
    state_after: Any = None
    rng_state_before: dict[str, Any] | None = None
    rng_state_after: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize this trace step to JSON-compatible data."""
        return {
            "tool": self.tool,
            "args": self.args,
            "seed": self.seed,
            "result": self.result,
            "action": self.action,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "rng_state_before": self.rng_state_before,
            "rng_state_after": self.rng_state_after,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TraceStep:
        """Reconstruct a trace step from JSON-compatible data."""
        return cls(
            tool=d["tool"],
            args=dict(d.get("args") or {}),
            seed=d.get("seed"),
            result=d.get("result"),
            action=d.get("action"),
            state_before=d.get("state_before"),
            state_after=d.get("state_after"),
            rng_state_before=d.get("rng_state_before"),
            rng_state_after=d.get("rng_state_after"),
        )


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
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot verify whether tool {tool!r} accepts seed") from exc
        accepts_seed = "seed" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if not accepts_seed:
            raise ValueError(f"tool {tool!r} does not accept a seed argument")
        call_args["seed"] = seed
    before = _rng_state()
    result = fn(**call_args)
    return TraceStep(
        tool=tool,
        args=dict(args),
        seed=seed,
        result=result,
        rng_state_before=before,
        rng_state_after=_rng_state(),
    )


def replay(trace: ExecutionTrace, tools: dict[str, Callable[..., Any]]) -> ExecutionTrace:
    """Re-execute every step of ``trace`` against ``tools`` with the exact same args and seed."""
    caller_state = _rng_state()
    replayed: list[TraceStep] = []
    try:
        for step in trace.steps:
            if step.rng_state_before is None:
                raise ValueError("trace step has no captured RNG state")
            _restore_rng_state(step.rng_state_before)
            replayed_step = record_step(tools, step.tool, step.args, seed=step.seed)
            replayed_step.action = step.action
            replayed.append(replayed_step)
    finally:
        _restore_rng_state(caller_state)
    return ExecutionTrace(request=trace.request, steps=replayed)


def diff(a: ExecutionTrace, b: ExecutionTrace) -> list[tuple[int, str]]:
    """Indices + tool names where ``a`` and ``b`` disagree (JSON-serialized result comparison)."""
    mismatches = []
    if a.request != b.request:
        mismatches.append((-1, "request_mismatch"))
    if not a.steps and not b.steps:
        mismatches.append((0, "empty_trace"))
    for i, (sa, sb) in enumerate(zip(a.steps, b.steps)):
        if _canonical(sa.to_json()) != _canonical(sb.to_json()):
            mismatches.append((i, sa.tool))
    if len(a.steps) != len(b.steps):
        mismatches.append((min(len(a.steps), len(b.steps)), "length_mismatch"))
    return mismatches


def is_bit_identical_replay(trace: ExecutionTrace, tools: dict[str, Callable[..., Any]]) -> bool:
    """Replay ``trace`` and return whether every step reproduces exactly."""
    return not diff(trace, replay(trace, tools))
