"""Pluggable reweighting strategies, at a declared verifiability tier.

Names the program plan's "in order of preference: (1) certified simulator likelihoods; (2) epistemic
synthesis + discrepancy; (3) amortized neural estimator" list as a typed seam
(:class:`LikelihoodStrategy`) so :meth:`mixle.epistemic.portfolio.HypothesisPortfolio.reweight` doesn't
care which one produced a number, and so a real simulator integration (explicitly out of scope for
this plan -- see ``notes/epistemic-loop-integration-workplan.md`` §6) has a documented interface to
implement against today. ``tier`` reuses :data:`mixle.doe.oracle.VERIFIABILITY_TIERS` verbatim rather
than inventing a second vocabulary -- the same tiers :mod:`mixle.substrate.belief`'s evidence entries
already use.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from mixle.doe.oracle import VERIFIABILITY_TIERS
from mixle.epistemic.discrepancy import discrepancy_report
from mixle.epistemic.portfolio import Hypothesis


@runtime_checkable
class LikelihoodStrategy(Protocol):
    """A ``(hypothesis, observation) -> likelihood`` callable that declares its verifiability ``tier``."""

    tier: str

    def __call__(self, hypothesis: Hypothesis, observation: Any) -> float: ...


def _check_tier(tier: str) -> None:
    if tier not in VERIFIABILITY_TIERS:
        raise ValueError(
            f"likelihood tier {tier!r} is not a recognized verifiability tier {sorted(VERIFIABILITY_TIERS)} "
            "(mixle.doe.oracle.VERIFIABILITY_TIERS) -- 'self-graded by a model' is not a valid tier."
        )


class DiscrepancyLikelihood:
    """Likelihood from :func:`mixle.epistemic.discrepancy.discrepancy_report`: ``exp(-discrepancy / temperature)``.

    ``predict_fn(hypothesis) -> predicted_observation`` is the hypothesis's epistemic-synthesis step
    (program plan §3.7's "for each live hypothesis, generate the observation you would expect to
    see"); this class only does the comparison, not the prediction. ``tier`` is a required constructor
    argument rather than something inferred from ``discrepancy_report``'s ``degraded`` flag, because
    whether ``predict_fn`` itself calls a certified simulator under the hood is invisible to the
    discrepancy computation -- inferring it here would risk silently misreporting a tier
    (``notes/epistemic-loop-integration-workplan.md`` §5 Q2).
    """

    def __init__(self, predict_fn: Callable[[Hypothesis], Any], *, tier: str, temperature: float = 1.0) -> None:
        _check_tier(tier)
        self.predict_fn = predict_fn
        self.tier = tier
        self.temperature = float(temperature)

    def __call__(self, hypothesis: Hypothesis, observation: Any) -> float:
        predicted = self.predict_fn(hypothesis)
        result = discrepancy_report(predicted, observation)
        return math.exp(-result.value / self.temperature)


class CallableLikelihood:
    """Wrap any plain ``fn(hypothesis, observation) -> float`` as a :class:`LikelihoodStrategy`."""

    def __init__(self, fn: Callable[[Hypothesis, Any], float], *, tier: str) -> None:
        _check_tier(tier)
        self.fn = fn
        self.tier = tier

    def __call__(self, hypothesis: Hypothesis, observation: Any) -> float:
        return float(self.fn(hypothesis, observation))


__all__ = ["LikelihoodStrategy", "DiscrepancyLikelihood", "CallableLikelihood"]
