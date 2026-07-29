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
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio, _checked_likelihood

# `_checked_likelihood` is imported rather than re-implemented: "finite and non-negative, 0.0 allowed"
# is one contract, and the loop must not police it differently from the portfolio operations it feeds.


def _as_rng(rng: Any) -> np.random.RandomState:
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


def _sample_budget(value: Any, name: str) -> int:
    """Validate a Monte Carlo budget as an exact positive integer.

    ``n_outer=0`` divided by zero, a negative count raised an incidental NumPy error from inside
    ``rng.choice``, and ``n_inner=0`` made ``np.mean`` of an empty slice warn and return NaN -- which
    then became a NaN EIG that :func:`step`'s ``score > best_score`` comparison silently discarded,
    turning invalid evidence into "no action was worth taking". A fractional or non-numeric budget is
    equally not a number of draws, so the check is exactness, not just positivity.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}") from None
    if count != value or count < 1:
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    return count


def _nonneg_finite(value: Any, name: str) -> float:
    """Validate a finite, non-negative economic scalar (an action cost, or the ``lam`` tradeoff)."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
    return number


class _OnceLikelihood:
    """Memoize ``likelihood`` per hypothesis, for one observation, for the duration of one step.

    :func:`step` used to run every active likelihood twice against the same observation: once through
    :meth:`~HypothesisPortfolio.surprise_score` and again through
    :meth:`~HypothesisPortfolio.reweight`. For a deterministic likelihood that is merely wasteful --
    it doubles simulator/model cost on exactly the calls a real integration makes expensive. For a
    *stateful or stochastic* one it is a correctness bug: surprise was computed from one set of
    values and the posterior from a different set, so the step reported an internally inconsistent
    pair (a two-hypothesis alternating likelihood produced a surprise from one pair and a
    ``[1/3, 2/3]`` posterior from the other).

    Caching on ``hypothesis.id`` is exact here: ids are unique within a portfolio (the constructor
    enforces it), both consumers walk the same active set of that one portfolio, and an instance is
    scoped to a single observation in a single :func:`step` call. Values are validated on the way in
    so the one evaluation that does happen is also the one place a bad likelihood is attributed.
    """

    def __init__(self, likelihood: Callable[[Hypothesis, Any], float]) -> None:
        self._likelihood = likelihood
        self._cache: dict[str, float] = {}

    def __call__(self, hypothesis: Hypothesis, observation: Any) -> float:
        cached = self._cache.get(hypothesis.id)
        if cached is None:
            cached = _checked_likelihood(
                self._likelihood(hypothesis, observation), source=f"likelihood for hypothesis {hypothesis.id!r}"
            )
            self._cache[hypothesis.id] = cached
        return cached


@dataclass(frozen=True)
class EpistemicStep:
    """The full outcome of one loop iteration -- everything :class:`~mixle.epistemic.journal.EpistemicJournal` logs."""

    observation: Any
    portfolio_before: HypothesisPortfolio
    portfolio_after: HypothesisPortfolio
    surprise: float
    next_action: Any | None
    next_action_eig: float | None

    def __post_init__(self) -> None:
        """Reject metrics that cannot mean anything, at the point they are recorded.

        A step is what the journal attests to, and the journal's job is to make a decision trail
        auditable. Nothing checked these numbers, so a step carrying ``surprise=NaN`` and
        ``next_action_eig=inf`` journaled cleanly, serialized as strict JSON, round-tripped, and
        passed ``verify()`` -- fully certified evidence for a measurement that never happened.
        Hash-chain integrity says the record was not altered after the fact; it says nothing about
        whether the number was meaningful when written, and only this constructor is positioned to.

        ``surprise`` is a bounded score in ``[0, 1]`` (see
        :meth:`~mixle.epistemic.portfolio.HypothesisPortfolio.surprise_score`). ``next_action_eig``
        is an expected information gain in nats: non-negative and finite, or ``None`` when no action
        was scored -- which is the honest way to record "not measured", unlike NaN.
        """
        # Both bounds carry the same 1e-9 slack the portfolio's own invariants use. Surprise and EIG
        # are computed as differences of entropies in floating point, so a quantity that is
        # mathematically exactly 0 or exactly 1 routinely lands a few ulp outside -- a real EIG probe
        # here produced -1.4e-17. Rejecting that would be refusing arithmetic, not catching an error;
        # the guard is for values that are genuinely not measurements.
        tol = 1e-9
        surprise = float(self.surprise)
        if not math.isfinite(surprise) or not -tol <= surprise <= 1.0 + tol:
            raise ValueError(f"surprise must be a finite score in [0, 1], got {self.surprise!r}")
        if self.next_action_eig is not None:
            eig = float(self.next_action_eig)
            if not math.isfinite(eig) or eig < -tol:
                raise ValueError(
                    f"next_action_eig must be a finite non-negative information gain (or None when "
                    f"no action was scored), got {self.next_action_eig!r}"
                )


def _add_hypothesis(
    portfolio: HypothesisPortfolio, new_hypothesis: Hypothesis, *, floor_weight: float = 1e-3
) -> HypothesisPortfolio:
    """Insert a brand-new hypothesis with a real, revivable prior of exactly ``floor_weight``.

    Funding is taken from ``w_open`` first -- that reserved "none of the above" mass is precisely what
    a newly abduced hypothesis is claiming a piece of, so spending it there is the mass-conserving
    move :meth:`~HypothesisPortfolio.resurrect` already makes. But ``w_open`` is not always available
    to spend: a portfolio constructed *closed* (``w_open == 0``) can never grow it, because
    :meth:`~HypothesisPortfolio.reweight` scales ``w_open`` multiplicatively by the open-world
    baseline, and ``0 * anything == 0``. Capping the take at ``w_open`` alone therefore inserted the
    new hypothesis active with weight ``0.0`` while the incumbent kept weight ``1.0`` -- and since
    updates are multiplicative, a zero prior is permanent. The advertised escape from model
    misspecification did nothing at exactly the moment it was invoked.

    So any shortfall is transferred from the incumbents instead, diluting them proportionally: they
    are the hypotheses whose collective failure to explain the observation triggered abduction in the
    first place, and each keeps its *relative* standing. The shortfall is always affordable --
    ``sum(active) == 1 - w_open`` and the shortfall is ``floor_weight - w_open`` with
    ``floor_weight < 1`` -- so this never drives an incumbent negative.
    """
    floor = float(floor_weight)
    if not math.isfinite(floor) or floor <= 0.0 or floor >= 1.0:
        # A zero/negative floor reproduces the inert-hypothesis bug by construction, and >= 1 would
        # demand more mass than the whole portfolio has.
        raise ValueError(f"floor_weight must be a finite value in (0, 1), got {floor_weight!r}")
    from_open = min(floor, portfolio.w_open)
    shortfall = floor - from_open
    weights = np.asarray(portfolio.weights, dtype=np.float64)
    if shortfall > 0.0:
        active_total = float(weights.sum())
        weights = weights * ((active_total - shortfall) / active_total)
    hyps = portfolio.hypotheses + (new_hypothesis,)
    return HypothesisPortfolio(hyps, np.append(weights, floor), portfolio.w_open - from_open)


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

    The "in play" set is hypotheses that are both ``active`` *and* carry positive weight, not
    ``active`` alone: :class:`~mixle.epistemic.portfolio.HypothesisPortfolio`'s constructor only
    forces weight ``0.0`` for *inactive* hypotheses, so an ``active=True`` hypothesis with weight
    exactly ``0.0`` is legal (e.g. right after :meth:`~HypothesisPortfolio.reweight` collapses every
    likelihood and moves all mass to ``w_open``, per that method's own documented "honest ...
    outcome"). Filtering by ``active`` alone would let such a hypothesis through, its lone weight of
    ``0.0`` would renormalize as ``0 / 0``, and the resulting NaN sampling distribution would blow up
    ``rng.choice``. When nothing is both active and positively weighted -- including that
    all-mass-in-``w_open`` case -- there is nothing left to discriminate between, so this returns
    ``0.0``: the same "nothing to compare" contract already used below for an empty ``active`` set.

    ``w_open`` is not dropped. The nested-MC sum below can only be taken over hypotheses that have a
    predictive -- ``simulate_fn`` needs a hypothesis to simulate from, and the reserved open-world
    slot by definition has no model -- so what that sum estimates is ``EIG | the model set is
    complete``. Returning it unqualified made otherwise identical portfolios with ``w_open = 0.99``
    and ``w_open = 0.0`` report exactly the same EIG, which is how a caller ends up commissioning a
    confident discriminating experiment while 99% of its belief mass says the thing worth
    discriminating between is not in the set. The conditional estimate is therefore weighted by
    ``1 - w_open``, the probability that closed world obtains at all: the open-world branch is
    credited with zero expected discrimination, which is the honest floor for an outcome nothing in
    the portfolio can predict. The result is an unconditional EIG that is comparable across
    portfolios, and it goes to ``0`` as belief concentrates on "none of the above".
    """
    n_outer = _sample_budget(n_outer, "n_outer")
    n_inner = _sample_budget(n_inner, "n_inner")
    closed_mass = 1.0 - portfolio.w_open
    active = [(w, h) for w, h in zip(portfolio.weights, portfolio.hypotheses) if h.active and w > 0.0]
    if not active or closed_mass <= 0.0:
        return 0.0
    weights = np.array([w for w, _ in active], dtype=np.float64)
    weights = weights / weights.sum()
    hyps = [h for _, h in active]
    total = 0.0
    outer_idx = rng.choice(len(hyps), size=n_outer, p=weights)
    for i in outer_idx:
        h = hyps[i]
        y = simulate_fn(h, action, rng)
        ll_true = math.log(max(_checked_likelihood(likelihood(h, y), source=f"likelihood for {h.id!r}"), 1e-300))
        inner_idx = rng.choice(len(hyps), size=n_inner, p=weights)
        liks = np.array(
            [_checked_likelihood(likelihood(hyps[j], y), source=f"likelihood for {hyps[j].id!r}") for j in inner_idx],
            dtype=np.float64,
        )
        log_evidence = math.log(max(float(np.mean(liks)), 1e-300))
        total += ll_true - log_evidence
    return float(closed_mass * total / n_outer)


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

    UPDATE evaluates each likelihood exactly once (see :class:`_OnceLikelihood`): surprise and the
    posterior are derived from the same evidence, so a stateful or stochastic likelihood can no longer
    make one step report two different beliefs, and an expensive likelihood is not paid for twice.

    The ACT economics are validated rather than trusted: ``lam`` and every ``cost_fn`` result must be
    finite and non-negative, and ``n_outer``/``n_inner`` must be exact positive integers. Without
    those checks a NaN cost or EIG made ``score > best_score`` false for every candidate, so a
    one-candidate action space returned "no action worth taking" instead of reporting that its
    evidence was invalid; a negative cost was scored as a reward, and a negative ``lam`` inverted the
    declared cost penalty into a bonus.
    """
    once = _OnceLikelihood(likelihood)
    surprise = portfolio.surprise_score(observation, once)
    updated = portfolio.reweight(observation, once)
    if surprise_threshold is not None and surprise >= surprise_threshold and propose_fn is not None:
        new_hypothesis = propose_fn(updated)
        if new_hypothesis is not None:
            updated = _add_hypothesis(updated, new_hypothesis)

    next_action: Any | None = None
    next_action_eig: float | None = None
    if action_space is not None:
        if simulate_fn is None:
            raise ValueError("action_space requires simulate_fn(hypothesis, action, rng) for EIG estimation")
        lam_ = _nonneg_finite(lam, "lam")
        n_outer = _sample_budget(n_outer, "n_outer")
        n_inner = _sample_budget(n_inner, "n_inner")
        rng_ = _as_rng(rng)
        best_score = -math.inf
        for candidate in action_space:
            eig = _portfolio_eig_nmc(
                updated, candidate, likelihood, simulate_fn, rng_, n_outer=n_outer, n_inner=n_inner
            )
            if not math.isfinite(eig):
                raise ValueError(f"EIG estimate for action {candidate!r} is not finite: {eig!r}")
            cost = _nonneg_finite(cost_fn(candidate), f"cost_fn({candidate!r})") if cost_fn is not None else 0.0
            score = eig - lam_ * cost
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
