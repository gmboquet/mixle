"""One epistemic-loop step: OBSERVE -> UPDATE -> ABDUCE (on surprise) -> PREDICT -> DISCRIMINATE -> ACT.

:func:`step` is a pure function the caller drives from their own loop (interactive, scripted, or
agentic -- this module doesn't prescribe which). It is the integration point: everything in
:mod:`~mixle.epistemic.discrepancy`, :mod:`~mixle.epistemic.portfolio`, and
:mod:`~mixle.epistemic.likelihood` is a building block; this is where they compose. There is
deliberately no multi-step ``run_until(...)`` driver and no persistence beyond one
:class:`EpistemicStep`'s own fields here -- the program plan's "episode"/investigation-trace concept
(§4.1) is training-data machinery, out of scope for this plan (see
``notes/epistemic-loop-integration-workplan.md`` §6).

ACT's expected-information-gain scoring does **not** call
:func:`mixle.doe.active.expected_information_gain_nmc` directly: that function's nested-Monte-Carlo
estimator is written against a *continuous* numpy parameter space (``prior_sampler(rng, n) -> (n, k)
array``), while a :class:`~mixle.epistemic.portfolio.HypothesisPortfolio` is a *discrete* weighted set
of arbitrary typed hypothesis payloads. Forcing the portfolio through that interface would mean either
requiring every hypothesis payload to be a numpy vector (defeating the point of a typed portfolio) or
building a lossy adapter. Instead, ``_portfolio_eig_nmc`` below is the same nested-Monte-Carlo EIG
estimator (Ryan 2003), rewritten one level down against the portfolio's own discrete weighted draws
-- same math, the right data shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.epistemic.likelihood import LikelihoodStrategy
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio


def _as_rng(rng: Any) -> np.random.RandomState:
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


@dataclass(frozen=True)
class EpistemicStep:
    """The full outcome of one loop iteration -- everything :class:`~mixle.epistemic.journal.EpistemicJournal` logs."""

    observation: Any
    portfolio_before: HypothesisPortfolio
    portfolio_after: HypothesisPortfolio
    surprise: float
    next_action: Any | None
    next_action_eig: float | None


def _add_hypothesis(
    portfolio: HypothesisPortfolio, new_hypothesis: Hypothesis, *, floor_weight: float = 1e-3
) -> HypothesisPortfolio:
    """Insert a brand-new hypothesis, funding its floor weight out of ``w_open`` (mass-conserving)."""
    take = min(float(floor_weight), portfolio.w_open)
    hyps = portfolio.hypotheses + (new_hypothesis,)
    weights = np.append(portfolio.weights, take)
    return HypothesisPortfolio(hyps, weights, portfolio.w_open - take)


def _portfolio_eig_nmc(
    portfolio: HypothesisPortfolio,
    action: Any,
    likelihood: Callable[[Hypothesis, Any], float],
    simulate_fn: Callable[[Hypothesis, Any, np.random.RandomState], Any],
    rng: np.random.RandomState,
    *,
    n_outer: int,
    n_inner: int,
) -> float:
    """Nested-MC EIG of ``action`` against the portfolio's own discrete weighted hypothesis set.

    ``EIG = E_{h, y}[ log p(y|h,a) - log sum_h' w_h' p(y|h',a) ]``, estimated by drawing ``n_outer``
    hypotheses from the (renormalized active) portfolio, simulating one observation each via
    ``simulate_fn``, and estimating the log-evidence denominator from ``n_inner`` further draws --
    the discrete-portfolio analogue of :func:`mixle.doe.active.expected_information_gain_nmc`.
    """
    active = [(w, h) for w, h in zip(portfolio.weights, portfolio.hypotheses) if h.active]
    if not active:
        return 0.0
    weights = np.array([w for w, _ in active], dtype=np.float64)
    weights = weights / weights.sum()
    hyps = [h for _, h in active]
    total = 0.0
    outer_idx = rng.choice(len(hyps), size=n_outer, p=weights)
    for i in outer_idx:
        h = hyps[i]
        y = simulate_fn(h, action, rng)
        ll_true = math.log(max(float(likelihood(h, y)), 1e-300))
        inner_idx = rng.choice(len(hyps), size=n_inner, p=weights)
        liks = np.array([float(likelihood(hyps[j], y)) for j in inner_idx], dtype=np.float64)
        log_evidence = math.log(max(float(np.mean(liks)), 1e-300))
        total += ll_true - log_evidence
    return float(total / n_outer)


def step(
    portfolio: HypothesisPortfolio,
    observation: Any,
    likelihood: LikelihoodStrategy,
    *,
    action_space: Sequence[Any] | None = None,
    simulate_fn: Callable[[Hypothesis, Any, np.random.RandomState], Any] | None = None,
    cost_fn: Callable[[Any], float] | None = None,
    lam: float = 1.0,
    surprise_threshold: float | None = None,
    propose_fn: Callable[[HypothesisPortfolio], Hypothesis | None] | None = None,
    n_outer: int = 64,
    n_inner: int = 64,
    rng: Any = None,
) -> EpistemicStep:
    """One loop iteration: reweight on ``observation``, optionally abduce on surprise, optionally act.

    UPDATE: ``portfolio.reweight(observation, likelihood)``. ABDUCE: only when ``surprise_threshold``
    is set and the portfolio's :meth:`~HypothesisPortfolio.surprise_score` on ``observation`` meets or
    exceeds it, ``propose_fn(updated_portfolio)`` is called; a non-``None`` return is folded in via
    :func:`_add_hypothesis` (program plan §3.5's surprise trigger, at the scope this plan covers --
    schema-expansion / human-checkpoint semantics are not modeled here). ACT: when ``action_space`` is
    given, each candidate is scored by ``EIG(a) - lam * cost_fn(a)`` (program plan §2's ``a* =
    argmax_a EIG(a) - lambda*cost(a)``) via ``_portfolio_eig_nmc`` against the *updated* portfolio, and
    the argmax is returned; ``action_space=None`` is a valid "just update the belief" call and returns
    ``next_action=None``. Raises :class:`ValueError` if ``action_space`` is given without
    ``simulate_fn`` -- EIG estimation needs a way to generate a predicted observation per hypothesis
    per action, and there's no honest default for that.
    """
    surprise = portfolio.surprise_score(observation, likelihood)
    updated = portfolio.reweight(observation, likelihood)
    if surprise_threshold is not None and surprise >= surprise_threshold and propose_fn is not None:
        new_hypothesis = propose_fn(updated)
        if new_hypothesis is not None:
            updated = _add_hypothesis(updated, new_hypothesis)

    next_action: Any | None = None
    next_action_eig: float | None = None
    if action_space is not None:
        if simulate_fn is None:
            raise ValueError("action_space requires simulate_fn(hypothesis, action, rng) for EIG estimation")
        rng_ = _as_rng(rng)
        best_score = -math.inf
        for candidate in action_space:
            eig = _portfolio_eig_nmc(
                updated, candidate, likelihood, simulate_fn, rng_, n_outer=n_outer, n_inner=n_inner
            )
            cost = float(cost_fn(candidate)) if cost_fn is not None else 0.0
            score = eig - lam * cost
            if score > best_score:
                best_score, next_action, next_action_eig = score, candidate, eig

    return EpistemicStep(
        observation=observation,
        portfolio_before=portfolio,
        portfolio_after=updated,
        surprise=surprise,
        next_action=next_action,
        next_action_eig=next_action_eig,
    )


__all__ = ["EpistemicStep", "step"]
