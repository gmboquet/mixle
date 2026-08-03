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

from mixle.utils.immutable import detach_receipt_container


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
    # MXR-080-1892: the step's OUTCOME identity, recorded in the same construction as the RNG states so
    # a failure is trace data rather than a hole in the trace. ``None`` means a recorder that predates
    # the field said nothing; True/False are a positive claim about how the call ended.
    succeeded: bool | None = None

    def __post_init__(self) -> None:
        # MXR-080-1892: args arrived as a caller-owned dict and were stored by reference (``dict(args)``
        # in record_step is one level deep), so mutating a nested value after recording rewrote a step
        # that had already been used as replay evidence. Detached, not frozen: these fields are
        # round-tripped through JSON and compared by ``diff``, so their concrete types are load-bearing.
        object.__setattr__(self, "args", detach_receipt_container(self.args))
        object.__setattr__(self, "result", detach_receipt_container(self.result))
        object.__setattr__(self, "action", detach_receipt_container(self.action))
        object.__setattr__(self, "state_before", detach_receipt_container(self.state_before))
        object.__setattr__(self, "state_after", detach_receipt_container(self.state_after))

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
            "succeeded": self.succeeded,
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
            succeeded=d.get("succeeded"),
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


def _resolve_call(
    tools: dict[str, Callable[..., Any]], tool: str, args: dict[str, Any], seed: int | None
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Look the tool up and build its call arguments, raising on a harness problem.

    Split out of :func:`record_step` so :func:`replay` can distinguish a REPRODUCED tool failure (which
    it records as trace data) from a broken replay harness -- an unregistered tool or one that cannot
    take the recorded seed -- which must still raise (MXR-080-1892).
    """
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
    return fn, call_args


def record_step(
    tools: dict[str, Callable[..., Any]], tool: str, args: dict[str, Any], *, seed: int | None = None
) -> TraceStep:
    """Run ``tools[tool]`` once with ``args`` (and ``seed``, if the tool accepts one), recording the result."""
    fn, call_args = _resolve_call(tools, tool, args, seed)
    before = _rng_state()
    result = fn(**call_args)
    return TraceStep(
        tool=tool,
        args=dict(args),
        seed=seed,
        result=result,
        rng_state_before=before,
        rng_state_after=_rng_state(),
        succeeded=True,
    )


def failure_result(exc: BaseException) -> dict[str, Any]:
    """The canonical JSON shape a failed step records as its result.

    One shape for every recorder (MXR-080-1892): the orchestrator built this dict inline while replay
    let the exception escape, so a failure recorded one way could never be compared with a failure
    reproduced the other way.
    """
    return {"error": {"type": type(exc).__name__, "message": str(exc)}}


def replay(trace: ExecutionTrace, tools: dict[str, Callable[..., Any]]) -> ExecutionTrace:
    """Re-execute every step of ``trace`` against ``tools`` with the exact same args and seed.

    A step whose tool raises is RECORDED as a failed step rather than aborting the replay
    (MXR-080-1892): a recorded failure that cannot be re-run is not reproducible evidence, and the
    orchestrator's own traces are mostly interesting precisely where a step failed. The error is
    recorded in the same :func:`failure_result` shape the recorder used, so the two are comparable.

    Deliberately NOT re-derived: ``state_before``/``state_after``. Replay calls ``tools[tool](**args)``;
    it never runs the world that produced those snapshots, so recomputing them is impossible and
    comparing recorded-vs-absent would report a mismatch on every orchestrated trace. They are carried
    forward as recorded context exactly like ``action``. What replay proves is the TOOL's determinism --
    same args, same seed, same entry RNG state -> same result and same exit RNG state -- not the world's.
    """
    caller_state = _rng_state()
    replayed: list[TraceStep] = []
    try:
        for step in trace.steps:
            if step.rng_state_before is None:
                raise ValueError("trace step has no captured RNG state")
            # resolved BEFORE restoring anything: a missing tool is a broken harness, not a repro
            fn, call_args = _resolve_call(tools, step.tool, step.args, step.seed)
            _restore_rng_state(step.rng_state_before)
            before = _rng_state()
            try:
                result, succeeded = fn(**call_args), True
            except Exception as exc:  # noqa: BLE001 - a tool failure is the outcome being reproduced
                result, succeeded = failure_result(exc), False
            replayed_step = TraceStep(
                tool=step.tool,
                args=dict(step.args),
                seed=step.seed,
                result=result,
                rng_state_before=before,
                rng_state_after=_rng_state(),
                succeeded=succeeded,
            )
            replayed_step.action = step.action
            replayed_step.state_before = step.state_before
            replayed_step.state_after = step.state_after
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
