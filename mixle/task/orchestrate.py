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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
    stopped_reason: str  # "plan_stop" | "budget_exhausted" | "world_done" | "low_confidence" | "replan_failed"


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
    trace = ExecutionTrace(request=question)
    history: list[dict[str, Any]] = []
    stopped_reason = "budget_exhausted"

    for _ in range(int(budget)):
        if world.done:
            stopped_reason = "world_done"
            break

        step = plan_model(question, history)
        if _is_stop(step):
            stopped_reason = "plan_stop"
            break
        if confidence_threshold is not None and float(step.get("confidence", 1.0)) < confidence_threshold:
            stopped_reason = "low_confidence"
            break

        try:
            observation = world.step(step)
        except Exception as exc:
            # atypical/failed step: record the failure, then give the plan model one chance to re-plan
            # around it before giving up -- the trace shows what actually happened, not just the outcome
            trace.steps.append(TraceStep(tool=_tool_name(step), args=step.get("args", {}), result={"error": str(exc)}))
            retry_history = [*history, {**step, "error": str(exc)}]
            retry_step = plan_model(question, retry_history)
            if _is_stop(retry_step):
                stopped_reason = "replan_failed"
                break
            try:
                observation = world.step(retry_step)
            except Exception as retry_exc:
                trace.steps.append(
                    TraceStep(
                        tool=_tool_name(retry_step), args=retry_step.get("args", {}), result={"error": str(retry_exc)}
                    )
                )
                stopped_reason = "replan_failed"
                break
            step = retry_step

        trace.steps.append(TraceStep(tool=_tool_name(step), args=step.get("args", {}), result=observation))
        history.append({**step, "result": observation})
    else:
        stopped_reason = "budget_exhausted"

    answer = world.score()
    return OrchestrationResult(answer=answer, trace=trace, stopped_reason=stopped_reason)
