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
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            # Caught here, at construction, rather than left to surface later inside exp(-discrepancy /
            # temperature): 0 divides by zero at scoring time, far from this mistake; negative flips the
            # sign in the exponent and silently reverses the discrepancy ordering (a worse match would
            # score a *higher* likelihood); NaN/inf propagate into a NaN or a discrepancy-blind constant
            # likelihood instead of failing loudly.
            raise ValueError(
                f"DiscrepancyLikelihood temperature must be a finite, strictly positive number, got {temperature!r}."
            )
        self.predict_fn = predict_fn
        self.tier = tier
        self.temperature = temperature

    def __call__(self, hypothesis: Hypothesis, observation: Any) -> float:
        predicted = self.predict_fn(hypothesis)
        result = discrepancy_report(predicted, observation)
        likelihood = math.exp(-result.value / self.temperature)
        if not math.isfinite(likelihood) or likelihood < 0:
            # Internal-consistency check on our own output before it reaches portfolio reweighting/
            # normalization: a NaN or infinite discrepancy value (e.g. NaN/inf leaking in from
            # predict_fn or the observation) would otherwise pass through exp() as a NaN or infinite
            # "likelihood" silently, rather than failing where the bad input actually originated.
            raise ValueError(
                f"DiscrepancyLikelihood produced a non-finite or negative likelihood {likelihood!r} from "
                f"discrepancy value {result.value!r} (metric={result.metric!r}); refusing to return it."
            )
        return likelihood


class CallableLikelihood:
    """Wrap any plain ``fn(hypothesis, observation) -> float`` as a :class:`LikelihoodStrategy`.

    ``fn`` is arbitrary caller code, so its return value is checked here before it is presented as a
    likelihood: finite and non-negative, the same contract
    :meth:`DiscrepancyLikelihood.__call__` already enforces on its own output. ``0.0`` is legitimate
    ("this hypothesis gives the observation no support"); a negative, NaN, or infinite value is not,
    and downstream it does not fail loudly -- it moves belief mass and produces out-of-range surprise
    scores that look like ordinary numbers. This wrapper is the boundary where the mistake is still
    attributable to the function that made it.
    """

    def __init__(self, fn: Callable[[Hypothesis, Any], float], *, tier: str) -> None:
        _check_tier(tier)
        self.fn = fn
        self.tier = tier

    def __call__(self, hypothesis: Hypothesis, observation: Any) -> float:
        likelihood = float(self.fn(hypothesis, observation))
        if not math.isfinite(likelihood) or likelihood < 0:
            raise ValueError(
                f"CallableLikelihood's wrapped {getattr(self.fn, '__name__', type(self.fn).__name__)!r} returned "
                f"{likelihood!r} for hypothesis {hypothesis.id!r}; a likelihood must be finite and non-negative "
                "(0.0 is allowed and means 'no support')."
            )
        return likelihood


__all__ = ["LikelihoodStrategy", "DiscrepancyLikelihood", "CallableLikelihood"]
