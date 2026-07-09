"""Heuristic allocation of improvement effort against a held-out scorecard.

Given a :class:`~mixle.system.System` and a set of
:class:`ImprovementOption` actions, each with an estimated cost and estimated
recoverable scorecard quality, this module spends the budget on the highest
estimated gain-per-dollar option first. Only realized gain is trusted:
:func:`~mixle.scorecard.evaluate` remeasures the scorecard before and after
each option runs, and :func:`~mixle.scorecard.detect_regression` stops the
allocation immediately when a round regresses.

This module provides a deterministic heuristic baseline. A learned meta-policy
should replace it only after it demonstrates better realized scorecard gain per
dollar under the same measurement protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mixle.scorecard import RegressionReport, SystemScorecard, detect_regression, evaluate
from mixle.system import Query, System


@dataclass
class ImprovementOption:
    """Candidate improvement action with estimated cost and recoverable quality gain."""

    name: str
    cost: float
    run: Callable[[], None]
    estimated_regret: float  # prior estimate of recoverable scorecard-quality gain

    @property
    def regret_per_dollar(self) -> float:
        """Estimated recoverable gain per unit cost."""
        return self.estimated_regret / self.cost if self.cost > 0 else self.estimated_regret


@dataclass
class MetaImprovementReport:
    """Execution report for a budgeted meta-improvement run."""

    order: list[str] = field(default_factory=list)  # options run, in execution order
    skipped: list[str] = field(default_factory=list)  # over budget and not attempted
    scorecard_before: SystemScorecard | None = None
    scorecard_after: SystemScorecard | None = None
    realized_gain_per_dollar: dict[str, float] = field(default_factory=dict)  # measured after each option
    spent: float = 0.0
    stopped_on_regression: RegressionReport | None = None


def improve_by_regret(
    system: System,
    question_set: Sequence[tuple[Query, str]],
    options: Sequence[ImprovementOption],
    *,
    budget: float,
) -> MetaImprovementReport:
    """Run options by estimated gain per dollar and stop on measured regression."""
    ordered = sorted(options, key=lambda o: o.regret_per_dollar, reverse=True)
    report = MetaImprovementReport(scorecard_before=evaluate(system, question_set))
    current = report.scorecard_before
    spent = 0.0

    for opt in ordered:
        if spent + opt.cost > budget:
            report.skipped.append(opt.name)
            continue

        before = current
        opt.run()
        after = evaluate(system, question_set)
        report.realized_gain_per_dollar[opt.name] = (
            (after.quality - before.quality) / opt.cost if opt.cost > 0 else (after.quality - before.quality)
        )
        report.order.append(opt.name)
        spent += opt.cost
        current = after

        regression = detect_regression(before, after)
        if regression.regressed:
            report.stopped_on_regression = regression
            break

    report.scorecard_after = current
    report.spent = spent
    return report
