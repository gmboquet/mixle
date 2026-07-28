"""Heuristic allocation of improvement effort against a held-out scorecard.

Given a :class:`~mixle.system.core.System` and a set of
:class:`ImprovementOption` actions, each with an estimated cost and estimated
recoverable scorecard quality, this module spends the budget on the highest
estimated gain-per-dollar option first. Only realized gain is trusted:
:func:`~mixle.system.scorecard.evaluate` remeasures the scorecard before and after
each option runs, and :func:`~mixle.system.scorecard.detect_regression` stops the
allocation immediately when a round regresses.

This module provides a deterministic heuristic baseline. A learned meta-policy
should replace it only after it demonstrates better realized scorecard gain per
dollar under the same measurement protocol.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mixle.system.core import Query, System
from mixle.system.scorecard import RegressionReport, SystemScorecard, detect_regression, evaluate

__all__ = ["ImprovementOption", "MetaImprovementReport", "improve_by_regret"]


def _finite(name: str, value: object) -> float:
    """A finite real economic quantity, or ``ValueError``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass
class ImprovementOption:
    """Candidate improvement action with estimated cost and recoverable quality gain.

    ``cost`` must be finite and nonnegative and ``estimated_regret`` finite. Neither was checked, so
    under ``budget=0`` an option costing ``-5`` ran and reported ``spent=-5`` (running work *created*
    budget), and a NaN-cost option ran too, because the over-budget comparison ``spent + cost > budget``
    is false for NaN -- the ceiling failed open -- and left ``spent=NaN`` behind.
    """

    name: str
    cost: float
    run: Callable[[], None]
    estimated_regret: float  # prior estimate of recoverable scorecard-quality gain

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"improvement option name must be a non-empty string, got {self.name!r}")
        cost = _finite(f"improvement option {self.name!r} cost", self.cost)
        if cost < 0.0:
            raise ValueError(f"improvement option {self.name!r} cost must be nonnegative, got {self.cost!r}")
        self.cost = cost
        self.estimated_regret = _finite(f"improvement option {self.name!r} estimated_regret", self.estimated_regret)
        if not callable(self.run):
            raise ValueError(f"improvement option {self.name!r} run must be callable")

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
    """Run options by estimated gain per dollar and stop on measured regression.

    ``budget`` must be finite and nonnegative, and option names must be unique -- ``order``,
    ``skipped`` and ``realized_gain_per_dollar`` are all keyed by name, so duplicates silently
    overwrite each other's measured result.

    Note what this does NOT do: ``opt.run()`` mutates the live ``system`` before any measurement, and
    a regression detected afterwards only stops *further* options -- the regressing change stays
    applied. Callers who need rejection to restore the previous state must run options against a
    system they can themselves discard. ``report.stopped_on_regression`` names the round that
    regressed so that decision can be made.
    """
    budget = _finite("budget", budget)
    if budget < 0.0:
        raise ValueError(f"budget must be nonnegative, got {budget!r}")
    options = tuple(options)
    names = [opt.name for opt in options]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"improvement option names must be unique; repeated: {duplicates}")
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
