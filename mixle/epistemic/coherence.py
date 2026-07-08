"""Program plan §2's three CI-checkable coherence properties, as plain testable functions.

"Checked continuously in CI, not asserted" -- these compute a violation magnitude (or boolean) over a
:class:`~mixle.epistemic.portfolio.HypothesisPortfolio` and a caller-supplied likelihood; nothing here
is wired into an enforcement path, matching the program plan's own framing. Where a fitting primitive
already exists (:func:`mixle.data.exchangeability.exchangeability_check`), it's a dataset-shaped
permutation test over row order in a flat table, not over a portfolio-update trajectory -- the shapes
don't line up cleanly enough to delegate directly, so :func:`exchangeability_violation` below is a
thin, purpose-built adapter using the same permutation-test idea rather than a forced reuse.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.epistemic.portfolio import HypothesisPortfolio


def _as_rng(rng: Any) -> np.random.RandomState:
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


def exchangeability_violation(
    portfolio0: HypothesisPortfolio,
    observations: Sequence[Any],
    likelihood: Callable[[Any, Any], float],
    *,
    n_permutations: int = 20,
    rng: Any = None,
) -> float:
    """Max posterior-weight deviation across random reorderings of ``observations`` (program plan §2(i)).

    Sequentially reweights ``portfolio0`` through ``observations`` in its given order, then again
    through ``n_permutations`` random permutations of the same set; returns the largest per-hypothesis
    (or open-world) weight deviation seen across permutations. Zero (up to float noise) for a
    likelihood whose per-step reweighting is a pure multiplicative update with no hidden order
    dependence -- successive Bayesian multiply-and-renormalize steps commute exactly in that case.
    """
    rng = _as_rng(rng)

    def _run(seq: Sequence[Any]) -> tuple[np.ndarray, float]:
        port = portfolio0
        for obs in seq:
            port = port.reweight(obs, likelihood)
        return port.weights, port.w_open

    base_weights, base_open = _run(observations)
    obs_list = list(observations)
    max_deviation = 0.0
    for _ in range(n_permutations):
        permuted = [obs_list[i] for i in rng.permutation(len(obs_list))]
        weights, w_open = _run(permuted)
        deviation = max(float(np.max(np.abs(weights - base_weights))), abs(w_open - base_open))
        max_deviation = max(max_deviation, deviation)
    return float(max_deviation)


def martingale_violation(
    portfolio: HypothesisPortfolio,
    observation_sampler: Callable[[np.random.RandomState], Any],
    likelihood: Callable[[Any, Any], float],
    *,
    n: int = 1000,
    rng: Any = None,
) -> float:
    """``|E[w_{t+1} | B_t] - w_t|`` under the model's own predictive (program plan §2(ii)).

    ``observation_sampler(rng) -> observation`` must draw from the portfolio's *own* predictive
    distribution (e.g. sample a hypothesis proportional to its current weight, then simulate one
    observation from it) -- the martingale property is a statement about self-consistency under the
    model's own predictive measure, not about any particular real data-generating process. Returns the
    largest per-hypothesis (or open-world) deviation between the prior weight and the weight averaged
    over ``n`` resampled predictive observations.
    """
    rng = _as_rng(rng)
    accum_weights = np.zeros_like(portfolio.weights)
    accum_open = 0.0
    for _ in range(n):
        observation = observation_sampler(rng)
        updated = portfolio.reweight(observation, likelihood)
        accum_weights += updated.weights
        accum_open += updated.w_open
    mean_weights = accum_weights / n
    mean_open = accum_open / n
    return float(max(float(np.max(np.abs(mean_weights - portfolio.weights))), abs(mean_open - portfolio.w_open)))


def evidence_conservation_violation(
    portfolio0: HypothesisPortfolio,
    observation: Any,
    likelihood: Callable[[Any, Any], float],
) -> bool:
    """Whether re-ingesting the identical ``observation`` a second time changes the posterior further.

    Program plan §2(iii): "the same underlying measurement, ingested twice through different routes,
    updates once." This function tests the math given an *already-deduped* input path -- it does not
    itself provide the content-addressed dedup key that real conservation needs (program plan §3.1 is
    where that lives, a storage-layer concern outside this plan's scope). Concretely: a plain
    ``likelihood`` with no memory of what it has already seen WILL show a violation here (double
    application double-counts the evidence, changing the weights again) -- that is the honest, correct
    outcome for an undeduped path, not a bug in this function. A ``likelihood`` that is itself
    dedup-aware (e.g. returns a neutral ``1.0`` for an ``observation`` it has already scored, tracked
    by identity/content-key in a closure the caller owns) shows no violation, demonstrating the property
    holds once dedup is actually wired in.
    """
    once = portfolio0.reweight(observation, likelihood)
    twice = once.reweight(observation, likelihood)
    weights_match = bool(np.allclose(once.weights, twice.weights, atol=1e-9))
    open_match = abs(once.w_open - twice.w_open) < 1e-9
    return not (weights_match and open_match)


__all__ = ["exchangeability_violation", "martingale_violation", "evidence_conservation_violation"]
