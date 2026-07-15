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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DesignRound", "SequentialDesignResult", "sequential_design"]


@dataclass
class DesignRound:
    """One round of a sequential design: the fitted ``state``, its uncertainty ``summary``, the
    controller ``decision`` that round, and the ``proposed_action`` chosen next (``None`` on the final
    round, where the loop stopped and proposed nothing)."""

    index: int
    state: Any
    summary: dict[str, Any]
    decision: dict[str, Any]
    proposed_action: Any = None


@dataclass
class SequentialDesignResult:
    """The full audit trail of a sequential design: every round in order, plus why it stopped."""

    rounds: list[DesignRound] = field(default_factory=list)
    stopped_reason: str = ""  # "controller_stop" | "budget_exhausted" | "no_proposal"

    @property
    def final_state(self) -> Any:
        return self.rounds[-1].state if self.rounds else None

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)


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
        should_continue: (history-so-far, including the current round's state/summary) -> a dict with a
            truthy ``"keep_going"``. Everything else in the dict is recorded as the round's decision.
        propose: (state, history) -> the next action/design point. Return ``None`` to stop the loop
            even though the controller wanted to continue (no admissible next sample) --
            ``stopped_reason`` is then ``"no_proposal"``.
        acquire: action -> the new observation(s) from actually taking that sample.
        combine: (data, new_data) -> the updated dataset for the next round.
        max_rounds: hard cap on rounds (the loop always terminates).

    Returns:
        A :class:`SequentialDesignResult` -- every :class:`DesignRound` in order plus ``stopped_reason``.
    """
    result = SequentialDesignResult()
    data = initial_data

    for i in range(int(max_rounds) + 1):  # +1: round 0 is the initial fit before any adaptive sample
        state = fit(data)
        summary = summarize(state, i)
        this_round = DesignRound(index=i, state=state, summary=summary, decision={}, proposed_action=None)
        result.rounds.append(this_round)

        decision = should_continue(result.rounds)
        this_round.decision = decision

        if not decision.get("keep_going", False):
            result.stopped_reason = "controller_stop"
            return result
        if i >= int(max_rounds):
            result.stopped_reason = "budget_exhausted"
            return result

        action = propose(state, result.rounds)
        if action is None:
            result.stopped_reason = "no_proposal"
            return result
        this_round.proposed_action = action
        data = combine(data, acquire(action))

    result.stopped_reason = "budget_exhausted"
    return result
