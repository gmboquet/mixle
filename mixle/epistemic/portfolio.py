"""``HypothesisPortfolio`` -- a weighted set of typed hypotheses plus an explicit open-world mass.

The program-plan's ``H_t = {(h_i, w_i)} u {(_|_, w_|_)}``: a sequential-Monte-Carlo particle cloud
generalized from :func:`mixle.inference.mcmc.particle_filter`'s numeric-state-only particles to
*arbitrary typed hypothesis payloads*, with the reserved "none of the above" mass carried as a
first-class field rather than folded into the particle list. Weights always satisfy ``w_open +
sum(active weights) == 1`` -- every mutating method returns a *new* portfolio with the invariant
already restored, and the constructor validates it on every construction (no silent drift).

Pruning (:meth:`prune`) never deletes a hypothesis, only deactivates it -- the same
never-truly-forget philosophy already used by :mod:`mixle.substrate.belief`'s cascading retraction --
and its freed mass folds into ``w_open``: a pruned hypothesis was one we could no longer defend, which
is exactly what growing the "we don't currently have an explanation" mass means (see
``notes/epistemic-loop-integration-workplan.md`` §5 Q1). :meth:`resample` delegates to the same
systematic/multinomial resampling math :func:`mixle.inference.mcmc.particle_filter` uses, applied only
to the active mass -- ``w_open`` is untouched by resampling, since it isn't a particle to resample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import numpy as np

_RESAMPLE_METHODS = frozenset({"systematic", "multinomial"})
"""Resamplers :meth:`HypothesisPortfolio.resample` implements, checked before any early return."""


@dataclass(frozen=True)
class Hypothesis:
    """One typed hypothesis in a portfolio. ``payload`` is opaque to the portfolio itself."""

    id: str
    payload: Any
    active: bool = True


def _as_rng(rng: Any) -> np.random.RandomState:
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


def _checked_likelihood(value: Any, *, source: str) -> float:
    """Validate one likelihood before it is allowed to move belief mass: finite and ``>= 0``.

    A likelihood is a non-negative density/probability; nothing else has a Bayesian meaning here.
    Exactly ``0.0`` IS legitimate and deliberately allowed -- it is the "this hypothesis assigns the
    observation no support" case :meth:`HypothesisPortfolio.reweight` documents an explicit
    all-mass-to-``w_open`` outcome for. A *negative* value is not: it makes the unnormalized total
    smaller, so a portfolio fed two likelihoods of ``-1`` used to trip that same ``total <= 0`` branch
    and report the honest "nothing explains this" outcome for what was really invalid evidence, and it
    drives :meth:`HypothesisPortfolio.surprise_score`'s ``baseline / (baseline + weighted_lik)`` to
    ``inf`` (at ``-1``) or negative (below ``-1``) despite that method promising ``[0, 1)``. NaN and
    ``inf`` are equally meaningless and propagate silently. Invalid evidence must fail closed rather
    than be reinterpreted as a valid-but-uninformative observation.
    """
    lik = float(value)
    if not np.isfinite(lik) or lik < 0.0:
        raise ValueError(
            f"{source} returned {lik!r}; a likelihood must be a finite, non-negative number "
            "(0.0 is allowed and means 'no support', but negative/NaN/inf evidence is refused rather "
            "than silently reinterpreted)"
        )
    return lik


class HypothesisPortfolio:
    """A weighted, typed hypothesis set with an explicit reserved open-world mass ``w_open``.

    ``weights`` is a private *copy* of what the constructor was given, exposed read-only: the
    constructor's invariant checks run once, and a writable or caller-aliased array would let them be
    undone immediately afterwards. Build a new portfolio (or use :meth:`reweight`/:meth:`prune`/
    :meth:`resample`/:meth:`resurrect`, which all return one) to change weights; ``weights.copy()``
    gives a writable array when a caller genuinely wants scratch space.
    """

    def __init__(self, hypotheses: Sequence[Hypothesis], weights: np.ndarray, w_open: float = 0.0) -> None:
        self.hypotheses: tuple[Hypothesis, ...] = tuple(hypotheses)
        # `np.array(..., copy=True)` + `writeable = False`, not `np.asarray`: every check below runs
        # once, at construction, and the class's whole contract is that a constructed portfolio
        # SATISFIES them. `asarray` left the caller's own array aliased and the result writable, so
        # both `caller_array[:] = ...` and `portfolio.weights[:] = ...` silently rewrote a validated
        # belief state afterwards -- a [0.5, 0.5] portfolio became [2.0, -1.0] with w_open still 0,
        # violating non-negativity and the sum-to-1 invariant with no check ever re-run. Copying
        # decouples the caller's array; freezing makes the invariant hold for the object's lifetime
        # rather than only for the instant it was checked. Every mutating method already rebuilds
        # weights via `.copy()`/`np.zeros`/`np.append` and returns a NEW portfolio, so nothing
        # internal writes through this array.
        self.weights: np.ndarray = np.array(weights, dtype=np.float64)
        self.weights.flags.writeable = False
        self.w_open: float = float(w_open)
        if self.weights.shape != (len(self.hypotheses),):
            raise ValueError(f"weights must have shape ({len(self.hypotheses)},), got {self.weights.shape}")
        ids = [h.id for h in self.hypotheses]
        if len(set(ids)) != len(ids):
            # resample()'s and resurrect()'s docstrings both promise every hypothesis id stays unique
            # (resurrect reactivates the FIRST match, silently ignoring the rest on a collision), so a
            # duplicate is a real invariant violation, not a cosmetic one -- reject it here rather than
            # letting it corrupt lookups downstream.
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"hypothesis ids must be unique, got duplicate(s): {dupes!r}")
        if not np.isfinite(self.w_open) or self.w_open < -1e-9 or self.w_open > 1 + 1e-9:
            # `nan < -1e-9` and `nan > 1 + 1e-9` are both always False, so a NaN w_open used to sail
            # through this range check untouched and propagate into reweight()/resample()/
            # surprise_score() as silently corrupted state.
            raise ValueError(f"w_open must be a finite value in [0, 1], got {self.w_open}")
        for h, w in zip(self.hypotheses, self.weights):
            if not np.isfinite(w) or w < -1e-9:
                # `nan < -1e-9` is always False, so a NaN weight used to sail through this check
                # untouched -- same blind spot as w_open above.
                raise ValueError(f"hypothesis {h.id!r} has a non-finite or negative weight: {w}")
            if not h.active and w != 0.0:
                raise ValueError(f"inactive hypothesis {h.id!r} must carry weight 0.0, got {w}")
        total = float(self.weights.sum()) + self.w_open
        if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
            # Belt-and-suspenders alongside the finiteness checks above: weights/w_open are already
            # validated finite by this point, so this can only trip on overflow to +-inf from summing
            # extreme-but-individually-finite weights -- which `abs(total - 1.0) > 1e-6` already catches
            # correctly (unlike NaN, inf compares fine) -- but spelling out finiteness keeps the
            # invariant self-documenting rather than an emergent property of float overflow semantics.
            raise ValueError(f"sum(weights) + w_open must equal 1.0, got {total}")

    def __len__(self) -> int:
        return len(self.hypotheses)

    def active_mask(self) -> np.ndarray:
        return np.array([h.active for h in self.hypotheses], dtype=bool)

    def reweight(
        self,
        observation: Any,
        likelihood_fn: Callable[[Hypothesis, Any], float],
        *,
        open_world_likelihood: Callable[[Any], float] | None = None,
    ) -> HypothesisPortfolio:
        """Bayesian-reweight every active hypothesis by ``likelihood_fn(h, observation)``.

        ``open_world_likelihood(observation)`` reweights ``w_open`` too; it defaults to a flat
        constant baseline of ``1.0`` -- an implicit "moderately plausible, independent of how badly
        the current hypotheses fit" prior -- which is what makes the surprise mechanism work without
        extra wiring: when every active hypothesis's likelihood collapses toward zero on an
        out-of-support observation, the (unchanged) open-world baseline dominates the renormalization
        and ``w_open`` grows on its own, exactly the "the residual resists the current hypothesis
        schema" signal the program plan's surprise trigger names. If every likelihood (including the
        open-world baseline) is zero, all mass moves to ``w_open`` -- the honest "nothing, including
        the reserved slot, explains this" outcome, rather than raising or producing NaNs.

        Every likelihood (including the open-world one) must be finite and non-negative -- see
        :func:`_checked_likelihood`. That all-zero branch is reserved for genuinely unsupported
        observations; invalid evidence raises :class:`ValueError` instead of being routed through it.
        """
        active = self.active_mask()
        liks = np.zeros(len(self.hypotheses), dtype=np.float64)
        for i, h in enumerate(self.hypotheses):
            if h.active:
                liks[i] = _checked_likelihood(
                    likelihood_fn(h, observation), source=f"likelihood_fn for hypothesis {h.id!r}"
                )
        open_lik = (
            _checked_likelihood(open_world_likelihood(observation), source="open_world_likelihood")
            if open_world_likelihood is not None
            else 1.0
        )
        new_active_unnorm = self.weights[active] * liks[active] if active.any() else np.array([])
        new_open_unnorm = self.w_open * open_lik
        total = float(new_active_unnorm.sum()) + new_open_unnorm
        if total <= 0:
            return HypothesisPortfolio(self.hypotheses, np.zeros(len(self.hypotheses)), w_open=1.0)
        new_weights = np.zeros(len(self.hypotheses), dtype=np.float64)
        new_weights[active] = new_active_unnorm / total
        return HypothesisPortfolio(self.hypotheses, new_weights, w_open=new_open_unnorm / total)

    def resample(
        self, *, method: str = "systematic", ess_threshold: float = 0.5, rng: Any = None
    ) -> HypothesisPortfolio:
        """Resample the active particle set if effective sample size drops below ``ess_threshold * n``.

        ``w_open`` is untouched -- it is a reserved mass, not a particle. Resampled duplicates of the
        same source hypothesis get id-suffixed copies (``"h2"``, ``"h2#1"``, ...) so every hypothesis
        id in the returned portfolio stays unique, which :meth:`resurrect`/the journal rely on.

        ``method`` and ``ess_threshold`` are validated up front, before any early return
        (MXR-080-1759). Validating them lazily -- at the point each is first *used* -- made the
        contract depend on the current weights: an unknown ``method`` was silently accepted whenever
        ESS happened to clear the threshold, so the identical call raised or did not depending on
        data. ``ess_threshold`` compares against ``ess / n``, which lies in ``(0, 1]``, so the
        meaningful domain is ``[0, 1]`` (``0`` disables resampling, ``1`` resamples anything short of
        uniform weights); a negative threshold silently disabled resampling, and NaN made every
        comparison false and forced an unrequested resample on every call.

        Raises:
            ValueError: if ``method`` is not a known resampler, or ``ess_threshold`` is not a finite
                number in ``[0, 1]``.
        """
        if method not in _RESAMPLE_METHODS:
            raise ValueError(f"unknown resample method {method!r}; expected one of {sorted(_RESAMPLE_METHODS)}")
        if isinstance(ess_threshold, (bool, np.bool_)) or not isinstance(ess_threshold, Real):
            raise ValueError(f"ess_threshold must be a real scalar in [0, 1], got {ess_threshold!r}")
        ess_threshold = float(ess_threshold)
        if not np.isfinite(ess_threshold) or not 0.0 <= ess_threshold <= 1.0:
            raise ValueError(f"ess_threshold must be finite and in [0, 1], got {ess_threshold!r}")
        rng = _as_rng(rng)
        active_idx = [i for i, h in enumerate(self.hypotheses) if h.active]
        if len(active_idx) <= 1:
            return self
        w_active = self.weights[active_idx]
        total_active = float(w_active.sum())
        if total_active <= 0:
            return self
        p = w_active / total_active
        ess = 1.0 / float(np.sum(p**2)) if np.sum(p**2) > 0 else 0.0
        n = len(active_idx)
        if ess >= ess_threshold * n:
            return self
        if method == "systematic":
            positions = (rng.uniform() + np.arange(n)) / n
            chosen = np.searchsorted(np.cumsum(p), positions)
        else:  # "multinomial" -- the only other member of _RESAMPLE_METHODS, checked on entry
            chosen = rng.choice(n, size=n, p=p)
        new_hyps = list(self.hypotheses)
        new_weights = self.weights.copy()
        # Suffixes have to dodge the ids this resample does *not* get to reassign -- the inactive
        # slots, which are copied through untouched. Counting only the duplicates produced within
        # this pass minted an id that was already sitting in one of those slots, and the
        # constructor's own uniqueness check then rejected the portfolio: resample -> prune ->
        # resample is an ordinary particle-filter cycle, and the second resample raised
        # "hypothesis ids must be unique" on a portfolio this method itself had produced.
        taken = {h.id for i, h in enumerate(self.hypotheses) if i not in set(map(int, active_idx))}
        counts: dict[str, int] = {}
        for slot, pick in zip(active_idx, chosen):
            src = self.hypotheses[active_idx[int(pick)]]
            new_id = src.id  # the first copy keeps the source id; later ones get "#1", "#2", ...
            while new_id in taken:
                counts[src.id] = counts.get(src.id, 0) + 1
                new_id = f"{src.id}#{counts[src.id]}"
            taken.add(new_id)
            new_hyps[slot] = Hypothesis(id=new_id, payload=src.payload, active=True)
            new_weights[slot] = total_active / n
        return HypothesisPortfolio(new_hyps, new_weights, self.w_open)

    def prune(self, *, min_weight: float) -> HypothesisPortfolio:
        """Deactivate (never delete) active hypotheses below ``min_weight``; their mass folds into ``w_open``.

        ``min_weight`` must be finite and non-negative. ``weight < NaN`` is false for every weight,
        so a NaN threshold pruned nothing at all and returned an unchanged portfolio -- outwardly
        identical to "no hypothesis was below the threshold", which is the one reading that makes a
        caller move on.
        """
        if not np.isfinite(min_weight) or float(min_weight) < 0.0:
            raise ValueError(f"min_weight must be a finite non-negative threshold, got {min_weight!r}")
        new_hyps = list(self.hypotheses)
        new_weights = self.weights.copy()
        freed = 0.0
        for i, h in enumerate(self.hypotheses):
            if h.active and new_weights[i] < min_weight:
                freed += float(new_weights[i])
                new_weights[i] = 0.0
                new_hyps[i] = replace(h, active=False)
        return HypothesisPortfolio(new_hyps, new_weights, self.w_open + freed)

    def resurrect(self, hypothesis_id: str, *, floor_weight: float = 1e-3) -> HypothesisPortfolio:
        """Reactivate a deactivated hypothesis, taking its floor weight out of ``w_open`` (mass-conserving)."""
        idx = next((i for i, h in enumerate(self.hypotheses) if h.id == hypothesis_id), None)
        if idx is None:
            raise KeyError(f"no hypothesis with id {hypothesis_id!r}")
        if self.hypotheses[idx].active:
            return self
        take = min(float(floor_weight), self.w_open)
        new_hyps = list(self.hypotheses)
        new_hyps[idx] = replace(self.hypotheses[idx], active=True)
        new_weights = self.weights.copy()
        new_weights[idx] = take
        return HypothesisPortfolio(new_hyps, new_weights, self.w_open - take)

    def surprise_score(self, observation: Any, likelihood_fn: Callable[[Hypothesis, Any], float]) -> float:
        """Joint improbability of ``observation`` under every active hypothesis, in ``[0, 1)``.

        ``baseline / (baseline + weighted_mean_likelihood)`` against the same flat ``baseline = 1.0``
        :meth:`reweight` uses by default -- close to 0 when some active hypothesis explains the
        observation well, close to 1 when every active hypothesis assigns it near-zero likelihood
        (program plan §3.5's "improbable under every live hypothesis" surprise condition). A heuristic
        scalar, not a calibrated probability -- callers threshold it, this method just computes it.

        The ``[0, 1)`` range is only a range if every likelihood is finite and non-negative, which
        :func:`_checked_likelihood` enforces here: ``weighted_lik`` of ``-1`` makes the denominator
        zero and the score ``inf``, anything below ``-1`` makes it negative, NaN propagates, and ``inf``
        collapses it to exactly ``0``. Each of those is a number outside the advertised range that a
        caller's threshold test would nonetheless silently accept.
        """
        active = [(w, h) for w, h in zip(self.weights, self.hypotheses) if h.active]
        if not active:
            return 1.0
        total_w = sum(w for w, _ in active)
        if total_w <= 0:
            return 1.0
        weighted_lik = (
            sum(
                w * _checked_likelihood(likelihood_fn(h, observation), source=f"likelihood_fn for hypothesis {h.id!r}")
                for w, h in active
            )
            / total_w
        )
        baseline = 1.0
        return float(baseline / (baseline + weighted_lik))

    def to_dict(self) -> dict:
        return {
            "hypotheses": [{"id": h.id, "payload": h.payload, "active": h.active} for h in self.hypotheses],
            "weights": self.weights.tolist(),
            "w_open": self.w_open,
        }

    @classmethod
    def from_dict(cls, d: dict, *, payload_codec: Callable[[Any], Any] | None = None) -> HypothesisPortfolio:
        hyps = [
            Hypothesis(
                id=item["id"],
                payload=payload_codec(item["payload"]) if payload_codec else item["payload"],
                active=item["active"],
            )
            for item in d["hypotheses"]
        ]
        weights = np.array(d["weights"], dtype=np.float64)
        return cls(hyps, weights, float(d["w_open"]))


__all__ = ["Hypothesis", "HypothesisPortfolio"]
