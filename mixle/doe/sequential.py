"""Sequential design loop -- the one orchestration primitive both adaptive-design experiments proved
was missing, extracted as a small, composable driver instead of hand-rolled per demo.

Every adaptive-design pipeline built end-to-end (adaptive groundwater monitoring, adaptive gravity
survey design) hand-wrote the *same* stateful loop: fit a model to the data so far -> summarize its
uncertainty -> ask a controller whether the uncertainty is tight enough to stop or another sample is
needed -> if continuing, use a design criterion to propose the next sample -> acquire it, append,
repeat. This is that loop, parameterized. It is deliberately NOT a general workflow engine (branching
DAGs, autonomous pipeline composition from a natural-language goal): that broader orchestration layer
is a real but separate design question whose scope depends on intended usage, left open on purpose.
This is only the part that is unambiguously right regardless of that answer, because a sequential
experimental-design loop is a real thing people run either way, and both demos proved its absence
forces hand-rolling.

It composes the rest of this codebase's decision machinery rather than reinventing it:

  * ``should_continue`` is any ``(history) -> {"keep_going": bool, "reason": str}`` callable -- e.g.
    wrap :func:`mixle.analysis.real_options.voi_stopping_decision` (stop when the value of the next
    sample drops below its cost) or an LLM controller via
    ``mixle_mlops.core.decisions.structured_decision`` (a forced, un-self-contradictable STOP/CONTINUE).
  * ``propose`` is any ``(state, history) -> action`` -- e.g. wrap
    :func:`mixle.doe.active.expected_information_gain_linear` / a monitoring-network design.
  * a calibration guard (``mixle.inference.calibration_gate``) can be dropped into ``summarize`` so a
    round whose posterior is miscalibrated is flagged in the record the controller sees.

The audit trail (:class:`SequentialDesignResult`) is meant to be trustworthy even when a round goes
wrong: ``summary``/``decision`` records are detached portable JSON values, and every callback,
validation, and history-copy failure is recorded as an explicit failed round before the selected
``on_error`` policy returns or raises -- never a silent, traceless abort.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DesignRound",
    "SequentialDesignError",
    "SequentialDesignResult",
    "sequential_design",
]

_ON_ERROR_POLICIES = ("raise", "record_and_stop")


@dataclass
class DesignRound:
    """One round of a sequential design: the fitted ``state``, its uncertainty ``summary``, the
    controller ``decision`` that round, and the ``proposed_action`` chosen next (``None`` on the final
    round, where the loop stopped and proposed nothing -- or on a round that failed before a proposal
    was made, see below).

    ``summary`` and ``decision`` are the small, JSON-friendly audit fields: what :func:`sequential_design`
    stores is always a deep copy of what the callback returned, and what later callbacks see via
    ``history`` is always a deep copy of what is stored, so no callback (nor any code holding a
    reference to a value it once returned) can retroactively rewrite a past round's record. ``state``
    and ``proposed_action`` are shared by reference -- they may be large or otherwise-uncopyable fit
    artifacts (a GP posterior, a torch module) that callbacks should treat as read-only by convention.

    If any callback, record validation, or history-copy stage fails during this round, that failure is
    recorded rather than left implicit: ``failed`` is ``True``, ``failed_step`` names the exact stage,
    and ``error`` is a short ``f"{type}: {message}"`` description of the exception. A failed round's
    ``state``/``summary``/``decision`` reflect whatever completed before the failure.
    """

    index: int
    state: Any
    summary: dict[str, Any]
    decision: dict[str, Any]
    proposed_action: Any = None
    failed: bool = False
    failed_step: str | None = None  # Exact callback/validation/history-copy stage, only when failed.
    error: str | None = None  # f"{type(exc).__name__}: {exc}", set only when failed


@dataclass
class SequentialDesignResult:
    """The full audit trail of a sequential design: every round in order, plus why it stopped."""

    rounds: list[DesignRound] = field(default_factory=list)
    stopped_reason: str = ""  # "controller_stop" | "budget_exhausted" | "no_proposal" | "callback_error"

    @property
    def final_state(self) -> Any:
        return self.rounds[-1].state if self.rounds else None

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)


class SequentialDesignError(RuntimeError):
    """Raised when a callback, validation, or history-copy stage fails under ``on_error="raise"``.

    ``result`` is the partial audit trail up to and including the failed round (see
    :attr:`DesignRound.failed`), so a caller that wants both fail-fast propagation and the ability to
    inspect what happened can do::

        try:
            sequential_design(..., on_error="raise")
        except SequentialDesignError as exc:
            log(exc.result.rounds)  # the audit trail survives the raise
            raise

    The original callback exception is available as ``__cause__`` (standard exception chaining).
    """

    def __init__(self, message: str, *, result: SequentialDesignResult) -> None:
        super().__init__(message)
        self.result = result


def _copy_safe_history(rounds: list[DesignRound]) -> list[DesignRound]:
    """A copy-safe view of the audit trail to hand to ``should_continue``/``propose``: fresh
    :class:`DesignRound` wrappers with their own deep copies of ``summary``/``decision``, so a callback
    that reaches into ``history`` and mutates a past round's dict cannot touch the stored record.
    ``state`` and ``proposed_action`` are still shared by reference (see :class:`DesignRound`)."""
    return [
        DesignRound(
            index=r.index,
            state=r.state,
            summary=copy.deepcopy(r.summary),
            decision=copy.deepcopy(r.decision),
            proposed_action=r.proposed_action,
            failed=r.failed,
            failed_step=r.failed_step,
            error=r.error,
        )
        for r in rounds
    ]


def _validate_json_value(value: Any, path: str, active: set[int]) -> None:
    """Require the exact portable JSON value model, including finite numeric values and string keys."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers.")
        return
    if type(value) not in (dict, list):
        raise TypeError(f"{path} contains non-JSON value {type(value).__name__}.")
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path} contains a circular reference.")
    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} contains non-string object key {key!r}.")
                _validate_json_value(item, f"{path}.{key}", active)
        else:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active)
    finally:
        active.remove(identity)


def _canonical_json_dict(value: Any, label: str) -> dict[str, Any]:
    """Validate one audit record and return a detached canonical JSON copy."""
    if type(value) is not dict:
        raise TypeError(f"{label} must return a dict, got {type(value).__name__}")
    _validate_json_value(value, label, set())
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(payload)
    if type(decoded) is not dict:  # Defensive: the top-level check above already establishes this.
        raise RuntimeError(f"{label} JSON normalization did not produce an object.")
    return decoded


def _validate_decision(decision: Any) -> dict[str, Any]:
    """Validate and detach the controller decision before any proposal or acquisition side effect."""
    record = _canonical_json_dict(decision, "should_continue()")
    if "keep_going" not in record or type(record["keep_going"]) is not bool:
        raise TypeError("should_continue() decision requires an actual Boolean 'keep_going' field.")
    if type(record.get("reason")) is not str or not record["reason"].strip():
        raise ValueError("should_continue() decision requires a nonempty string 'reason' field.")
    return record


def _handle_callback_failure(
    result: SequentialDesignResult,
    index: int,
    step: str,
    exc: Exception,
    on_error: str,
    *,
    existing_round: DesignRound | None,
) -> SequentialDesignResult:
    """Record an explicit failure round for a named stage exception (appending a new round if
    ``fit`` failed before this round had one, otherwise marking the round already appended this
    iteration), set ``stopped_reason``, and then either re-raise (per ``on_error="raise"``) or return
    the partial result (per ``on_error="record_and_stop"``)."""
    message = f"{type(exc).__name__}: {exc}"
    if existing_round is None:
        failed_round = DesignRound(index=index, state=None, summary={}, decision={})
        result.rounds.append(failed_round)
    else:
        failed_round = existing_round
    failed_round.failed = True
    failed_round.failed_step = step
    failed_round.error = message
    result.stopped_reason = "callback_error"

    if on_error == "raise":
        raise SequentialDesignError(
            f"sequential_design round {index} failed in {step}(): {message}", result=result
        ) from exc
    return result


def sequential_design(
    initial_data: Any,
    *,
    fit: Callable[[Any], Any],
    summarize: Callable[[Any, int], dict[str, Any]],
    should_continue: Callable[[list[DesignRound]], dict[str, Any]],
    propose: Callable[[Any, list[DesignRound]], Any],
    acquire: Callable[[Any], Any],
    combine: Callable[[Any, Any], Any],
    max_rounds: int,
    on_error: str = "raise",
) -> SequentialDesignResult:
    """Run a stateful sequential experimental-design loop and return its full audit trail.

    Each round: ``fit(data) -> state``; ``summarize(state, round_index) -> uq_summary``;
    ``should_continue(history) -> {"keep_going": bool, "reason": str, ...}``. If the controller stops
    (or the round budget is hit), the loop ends. Otherwise ``propose(state, history) -> action`` picks
    the next sample, ``acquire(action) -> new_data`` obtains it (a real simulation/survey/measurement),
    and ``combine(data, new_data) -> data`` folds it in for the next round.

    Args:
        initial_data: the starting dataset (whatever ``fit`` consumes).
        fit: data -> state (e.g. a posterior). Called once per round on the accumulated data.
        summarize: (state, round_index) -> a JSON-friendly uncertainty summary dict. This is what the
            controller sees, so put the decision-relevant numbers (and any calibration flag) here.
        should_continue: (history-so-far, including the current round's state/summary) -> a dict with
            an actual Boolean ``"keep_going"`` and a nonempty string ``"reason"``. Everything else in
            the dict is recorded as the round's decision.
            ``history`` is a copy-safe view (see :class:`DesignRound`) -- mutating it does not affect
            the returned audit trail.
        propose: (state, history) -> the next action/design point. Return ``None`` to stop the loop
            even though the controller wanted to continue (no admissible next sample) --
            ``stopped_reason`` is then ``"no_proposal"``. ``history`` is the same copy-safe view.
        acquire: action -> the new observation(s) from actually taking that sample.
        combine: (data, new_data) -> the updated dataset for the next round.
        max_rounds: maximum number of adaptive acquisitions after the initial fit. The audit trail
            therefore contains at most max_rounds + 1 fitted states: index 0 for the initial data,
            followed by one state per acquired sample.
        on_error: what to do when any callback, audit-record validation, or history-copy stage raises.
            In every case the failure is first recorded as an explicit failed round (see
            :attr:`DesignRound.failed`) so the audit trail always shows what happened -- the policy only
            controls what happens *after* that:

            * ``"raise"`` (default): re-raise as :class:`SequentialDesignError` (chained to the
              original exception via ``__cause__``), carrying the partial result as ``.result``.
              Matches the historical behavior of letting the exception propagate, but the caller can
              now still recover the audit trail up to the failure by catching it.
            * ``"record_and_stop"``: swallow the exception and return the partial result normally, with
              ``stopped_reason == "callback_error"`` and the failed round's ``failed``/``failed_step``/
              ``error`` describing what happened.

            There is deliberately no "continue past a failed round" option: a round that fails in
            ``propose``/``acquire``/``combine`` never obtained new data, so the next round would just
            re-fit on the same ``data`` -- not meaningful forward progress. Retry that yourself (e.g. by
            calling :func:`sequential_design` again) if that is what you want.

    Returns:
        A :class:`SequentialDesignResult` -- every :class:`DesignRound` in order plus ``stopped_reason``.
    """
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 0:
        raise ValueError("max_rounds must be a nonnegative integer")
    if on_error not in _ON_ERROR_POLICIES:
        raise ValueError(f"on_error must be one of {_ON_ERROR_POLICIES}, got {on_error!r}")

    result = SequentialDesignResult()
    data = initial_data

    for i in range(max_rounds + 1):  # round 0 is the initial fit before any adaptive sample
        try:
            state = fit(data)
        except Exception as exc:  # noqa: BLE001 - a callback's exception type is unknown; record+policy decide next
            return _handle_callback_failure(result, i, "fit", exc, on_error, existing_round=None)

        this_round = DesignRound(index=i, state=state, summary={}, decision={}, proposed_action=None)
        result.rounds.append(this_round)

        try:
            summary = summarize(state, i)
        except Exception as exc:  # noqa: BLE001 - callback failure is recorded with its exact stage
            return _handle_callback_failure(result, i, "summarize", exc, on_error, existing_round=this_round)
        try:
            this_round.summary = _canonical_json_dict(summary, "summarize()")
        except Exception as exc:  # noqa: BLE001 - record-contract failure follows the selected policy
            return _handle_callback_failure(
                result, i, "summary_validation", exc, on_error, existing_round=this_round
            )

        try:
            decision_history = _copy_safe_history(result.rounds)
        except Exception as exc:  # noqa: BLE001 - history materialization is an auditable stage
            return _handle_callback_failure(
                result, i, "decision_history", exc, on_error, existing_round=this_round
            )
        try:
            decision = should_continue(decision_history)
        except Exception as exc:  # noqa: BLE001 - callback failure is recorded with its exact stage
            return _handle_callback_failure(
                result, i, "should_continue", exc, on_error, existing_round=this_round
            )
        try:
            this_round.decision = _validate_decision(decision)
        except Exception as exc:  # noqa: BLE001 - side effects remain gated behind decision validation
            return _handle_callback_failure(
                result, i, "decision_validation", exc, on_error, existing_round=this_round
            )

        if not this_round.decision["keep_going"]:
            result.stopped_reason = "controller_stop"
            return result
        if i >= max_rounds:
            result.stopped_reason = "budget_exhausted"
            return result

        try:
            proposal_history = _copy_safe_history(result.rounds)
        except Exception as exc:  # noqa: BLE001 - history materialization is an auditable stage
            return _handle_callback_failure(
                result, i, "proposal_history", exc, on_error, existing_round=this_round
            )
        try:
            action = propose(state, proposal_history)
        except Exception as exc:  # noqa: BLE001 - a callback's exception type is unknown; record+policy decide next
            return _handle_callback_failure(result, i, "propose", exc, on_error, existing_round=this_round)

        if action is None:
            result.stopped_reason = "no_proposal"
            return result
        this_round.proposed_action = action

        try:
            acquired = acquire(action)
        except Exception as exc:  # noqa: BLE001 - a callback's exception type is unknown; record+policy decide next
            return _handle_callback_failure(result, i, "acquire", exc, on_error, existing_round=this_round)

        try:
            data = combine(data, acquired)
        except Exception as exc:  # noqa: BLE001 - a callback's exception type is unknown; record+policy decide next
            return _handle_callback_failure(result, i, "combine", exc, on_error, existing_round=this_round)

    result.stopped_reason = "budget_exhausted"
    return result
