"""``orchestrate`` -- the minimal controller loop: plan a step, execute it against a world, re-plan on
a failed or atypical step, and stop on low confidence, world completion, or budget exhaustion.

``plan_model`` is any ``(request, history) -> step | None`` callable -- a :class:`~mixle.task.plan.Planner`
step, a :class:`~mixle.task.sft_plan.GenerativePlanner` decode, or a test double; ``None`` (or a step whose
``tool`` is ``None``/``"__stop__"``) means STOP. ``world`` is kept behind the :class:`World` protocol
rather than importing a concrete environment directly. Any object with ``step``/``done``/``score``
can plug in.

Every executed (or failed) step is appended to the returned trace as a
:class:`~mixle.task.replay.TraceStep`, so :func:`mixle.task.replay.replay` can later re-run the same
episode against the same ``world.step`` for a bit-identical-replay check.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.task.replay import ExecutionTrace, TraceStep

_STOP_TOOLS = (None, "__stop__", "STOP")
_NO_TOOL_KEY = object()  # distinguishes an EXPLICIT tool=None from a schema with no "tool" key at all


@runtime_checkable
class World(Protocol):
    """The minimal environment contract ``orchestrate`` needs."""

    def step(self, action: dict[str, Any]) -> Any:
        """Apply one action and return the environment's step result."""
        ...

    @property
    def done(self) -> bool:
        """Whether the environment has reached a terminal state."""
        ...

    def score(self) -> Any:
        """Return the environment's current score or outcome metric."""
        ...


@dataclass
class OrchestrationResult:
    """Final answer, execution trace, and stop reason from an orchestration run."""

    answer: Any
    trace: ExecutionTrace
    stopped_reason: str


def _is_stop(step: dict[str, Any] | None) -> bool:
    """A step is STOP only when it is ``None`` or has an EXPLICIT ``tool`` key set to a stop value.
    A step whose schema has no ``"tool"`` key at all (for example ``{"type": ..., "cell": ...}``)
    is a real action, not a stop -- ``dict.get("tool")`` alone can't tell those apart, since a missing
    key and an explicit ``tool=None`` both return ``None``."""
    return step is None or step.get("tool", _NO_TOOL_KEY) in _STOP_TOOLS


def _tool_name(step: dict[str, Any]) -> str:
    """The identifier :class:`~mixle.task.replay.TraceStep` records for this step: ``"tool"`` when the
    schema has one (the common case), else ``"type"`` for action-kind schemas -- rather than a
    bare ``step["tool"]`` KeyError far from the real cause when a world uses neither."""
    if "tool" in step:
        return step["tool"]
    if "type" in step:
        return step["type"]
    raise KeyError(
        f"orchestrate() cannot name this step for the trace: it has neither a 'tool' nor a 'type' key ({step!r})"
    )


def _snapshot(world: World) -> Any:
    snapshot = getattr(world, "snapshot", None)
    if callable(snapshot):
        return copy.deepcopy(snapshot())
    return None


def _validated_step(step: Any) -> dict[str, Any] | None:
    if step is None:
        return None
    if not isinstance(step, dict):
        raise ValueError("plan_model must return a step dictionary or None")
    if _is_stop(step):
        return None
    _tool_name(step)
    confidence = step.get("confidence", 1.0)
    if not isinstance(confidence, (int, float)) or not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("step confidence must be finite and in [0, 1]")
    return copy.deepcopy(step)


def _attempt(world: World, step: dict[str, Any]) -> tuple[TraceStep, bool, Any]:
    before = _snapshot(world)
    try:
        observation = world.step(copy.deepcopy(step))
    except Exception as exc:  # noqa: BLE001 - external world failures are returned as evidence
        result = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        return (
            TraceStep(
                tool=_tool_name(step),
                args=copy.deepcopy(step.get("args") or {}),
                result=result,
                action=copy.deepcopy(step),
                state_before=before,
                state_after=_snapshot(world),
            ),
            False,
            exc,
        )
    return (
        TraceStep(
            tool=_tool_name(step),
            args=copy.deepcopy(step.get("args") or {}),
            result=observation,
            action=copy.deepcopy(step),
            state_before=before,
            state_after=_snapshot(world),
        ),
        True,
        observation,
    )


def orchestrate(
    question: str,
    plan_model: Callable[[str, list[dict[str, Any]]], dict[str, Any] | None],
    world: World,
    *,
    budget: int,
    confidence_threshold: float | None = None,
) -> OrchestrationResult:
    """Plan one step at a time against ``plan_model``, execute it on ``world``, re-plan once on a
    failed step, and stop on an explicit STOP, low confidence, world completion, or budget exhaustion."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if confidence_threshold is not None and (
        not np.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    if not callable(plan_model):
        raise TypeError("plan_model must be callable")

    trace = ExecutionTrace(request=str(question))
    history: list[dict[str, Any]] = []
    stopped_reason = "budget_exhausted"
    attempts = 0

    while attempts < budget:
        if world.done:
            stopped_reason = "world_done"
            break

        raw_step = plan_model(question, history)
        step = _validated_step(raw_step)
        if step is None:
            stopped_reason = "plan_stop"
            break
        if confidence_threshold is not None and float(step.get("confidence", 1.0)) < confidence_threshold:
            stopped_reason = "low_confidence"
            break

        attempts += 1  # reserve the action budget before entering user-controlled code
        trace_step, succeeded, outcome = _attempt(world, step)
        trace.steps.append(trace_step)
        if not succeeded:
            if not bool(getattr(world, "failure_atomic", False)):
                stopped_reason = "partial_failure"
                break
            if attempts >= budget:
                stopped_reason = "budget_exhausted"
                break
            retry_history = [
                *history,
                {**step, "error": trace_step.result["error"], "attempt": attempts},
            ]
            retry_step = _validated_step(plan_model(question, retry_history))
            if retry_step is None:
                stopped_reason = "replan_failed"
                break
            if confidence_threshold is not None and float(retry_step.get("confidence", 1.0)) < confidence_threshold:
                stopped_reason = "low_confidence"
                break
            attempts += 1
            retry_trace, retry_succeeded, retry_outcome = _attempt(world, retry_step)
            trace.steps.append(retry_trace)
            if not retry_succeeded:
                stopped_reason = "replan_failed"
                break
            step = retry_step
            outcome = retry_outcome

        history.append({**step, "result": outcome, "attempt": attempts})

    answer = world.score()
    return OrchestrationResult(answer=answer, trace=trace, stopped_reason=stopped_reason)
