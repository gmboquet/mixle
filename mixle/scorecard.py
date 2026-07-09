"""System-level scorecard for held-out question sets.

Named ``SystemScorecard`` (not ``Scorecard``) to avoid colliding with
:class:`mixle.task.scorecard.Scorecard` -- a different, narrower comparison (one student solution vs its
teacher on a task); this one evaluates a whole :class:`~mixle.system.System` (teacher, captured cache,
degraded modes, budget, all of it) end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.system import Query, System


def _default_scorer(reply: str | None, expected: str) -> bool:
    return reply is not None and expected.strip().lower() in reply.strip().lower()


@dataclass
class SystemScorecard:
    """One evaluation of a :class:`~mixle.system.System` against a fixed held-out question set."""

    quality: float
    calibration: float
    realized_cost: float
    grounded_fraction: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scorecard into primitive fields."""
        return dict(self.__dict__)


def evaluate(
    system: System,
    question_set: Sequence[tuple[Query, str]],
    *,
    scorer: Callable[[str | None, str], bool] | None = None,
) -> SystemScorecard:
    """Evaluate ``system`` over a fixed ``[(query, expected_answer), ...]`` set.

    * ``quality`` -- fraction of questions ``scorer`` judges correct (default: case-insensitive
      substring match of ``expected`` in the reply).
    * ``grounded_fraction`` -- fraction answered WITHOUT a degraded mode (teacher or captured, not a
      store-only fallback / refusal / failure): the fraction of answers you can actually trust came
      from a real answer path, not a fault-boundary guess.
    * ``realized_cost`` -- total spend units (:meth:`~mixle.spend.Spend.total_units`) across the set.
    * ``calibration`` -- a coarse proxy (currently equal to ``quality``) until ``answer`` carries a real
      per-answer confidence to calibrate against; extend THIS function, not call sites, once that lands.
    """
    n = len(question_set)
    if n == 0:
        return SystemScorecard(quality=0.0, calibration=0.0, realized_cost=0.0, grounded_fraction=0.0, n=0)
    scorer = scorer or _default_scorer
    correct = 0
    grounded = 0
    cost = 0.0
    for query, expected in question_set:
        reply, receipt = system.answer(query)
        if scorer(reply, expected):
            correct += 1
        if receipt.get("status") == "answered" and receipt.get("degraded_mode") is None:
            grounded += 1
        spend = receipt.get("spend") or {}
        cost += float(spend.get("frontier_calls", 0)) + float(spend.get("oracle_calls", 0))
    quality = correct / n
    return SystemScorecard(
        quality=quality, calibration=quality, realized_cost=cost, grounded_fraction=grounded / n, n=n
    )


@dataclass
class RegressionReport:
    """Whether ``current`` is worse than ``baseline`` on any tracked axis, and exactly why."""

    regressed: bool
    reasons: list[str] = field(default_factory=list)


def detect_regression(
    baseline: SystemScorecard, current: SystemScorecard, *, tolerance: float = 1e-9
) -> RegressionReport:
    """Compare two scorecards from the SAME held-out set across improve-rounds.

    Never silently accepts a round that answers worse, less groundedly, or for more cost than the round
    before it -- each tracked axis that got worse (beyond ``tolerance``) is named in ``reasons``.
    """
    reasons: list[str] = []
    if current.quality < baseline.quality - tolerance:
        reasons.append(f"quality regressed: {baseline.quality:.3f} -> {current.quality:.3f}")
    if current.grounded_fraction < baseline.grounded_fraction - tolerance:
        reasons.append(
            f"grounded_fraction regressed: {baseline.grounded_fraction:.3f} -> {current.grounded_fraction:.3f}"
        )
    if current.realized_cost > baseline.realized_cost + tolerance:
        reasons.append(f"realized_cost increased: {baseline.realized_cost:.3f} -> {current.realized_cost:.3f}")
    return RegressionReport(regressed=bool(reasons), reasons=reasons)
