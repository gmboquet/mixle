"""``improve_by_regret`` -- heuristic improvement-effort allocator (workstream META-a research spike).

Given a :class:`~mixle.system.System` and a set of :class:`ImprovementOption` (an amplify/refine/
accumulate-style action, each with an estimated cost and an estimated "verified regret" -- how much
scorecard quality it could plausibly recover), spend the budget on the highest regret-per-dollar option
first. Only REALIZED gain is trusted: :func:`~mixle.scorecard.evaluate` re-measures the scorecard before
and after each option actually runs, and the moment :func:`~mixle.scorecard.detect_regression` flags a
worsening round, the whole allocation stops -- never silently absorbing a regression hoping a
later, cheaper-looking option fixes it.

Kill criterion (per the plan, stated before any learned variant is attempted): a learned meta-policy
must beat this heuristic's REALIZED scorecard-gain-per-dollar before it replaces it. This card ships
only the validated heuristic baseline; no learned policy is claimed or built here -- the heuristic has
not yet been beaten, so per the plan's build order nothing has earned the right to replace it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mixle.scorecard import RegressionReport, SystemScorecard, detect_regression, evaluate
from mixle.system import Query, System


@dataclass
class ImprovementOption:
    """One candidate improvement action -- amplify, refine, accumulate, or anything else that mutates
    the system/store and is expected to raise scorecard quality by roughly ``estimated_regret``."""

    name: str
    cost: float
    run: Callable[[], None]
    estimated_regret: float  # a prior estimate of recoverable scorecard-quality gain; never trusted blindly

    @property
    def regret_per_dollar(self) -> float:
        return self.estimated_regret / self.cost if self.cost > 0 else self.estimated_regret


@dataclass
class MetaImprovementReport:
    order: list[str] = field(default_factory=list)  # options actually run, in the order they ran
    skipped: list[str] = field(default_factory=list)  # over budget -- never attempted
    scorecard_before: SystemScorecard | None = None
    scorecard_after: SystemScorecard | None = None
    realized_gain_per_dollar: dict[str, float] = field(default_factory=dict)  # measured, not estimated
    spent: float = 0.0
    stopped_on_regression: RegressionReport | None = None


def improve_by_regret(
    system: System,
    question_set: Sequence[tuple[Query, str]],
    options: Sequence[ImprovementOption],
    *,
    budget: float,
) -> MetaImprovementReport:
    """Run ``options`` highest-regret-per-dollar first, within ``budget``, measuring the scorecard
    before and after EACH one and stopping immediately if a run regresses it."""
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
