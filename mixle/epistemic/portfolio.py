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
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Hypothesis:
    """One typed hypothesis in a portfolio. ``payload`` is opaque to the portfolio itself."""

    id: str
    payload: Any
    active: bool = True


def _as_rng(rng: Any) -> np.random.RandomState:
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


class HypothesisPortfolio:
    """A weighted, typed hypothesis set with an explicit reserved open-world mass ``w_open``."""

    def __init__(self, hypotheses: Sequence[Hypothesis], weights: np.ndarray, w_open: float = 0.0) -> None:
        self.hypotheses: tuple[Hypothesis, ...] = tuple(hypotheses)
        self.weights: np.ndarray = np.asarray(weights, dtype=np.float64)
        self.w_open: float = float(w_open)
        if self.weights.shape != (len(self.hypotheses),):
            raise ValueError(f"weights must have shape ({len(self.hypotheses)},), got {self.weights.shape}")
        if self.w_open < -1e-9 or self.w_open > 1 + 1e-9:
            raise ValueError(f"w_open must be in [0, 1], got {self.w_open}")
        for h, w in zip(self.hypotheses, self.weights):
            if not h.active and w != 0.0:
                raise ValueError(f"inactive hypothesis {h.id!r} must carry weight 0.0, got {w}")
            if w < -1e-9:
                raise ValueError(f"hypothesis {h.id!r} has negative weight {w}")
        total = float(self.weights.sum()) + self.w_open
        if abs(total - 1.0) > 1e-6:
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
        """
        active = self.active_mask()
        liks = np.zeros(len(self.hypotheses), dtype=np.float64)
        for i, h in enumerate(self.hypotheses):
            if h.active:
                liks[i] = float(likelihood_fn(h, observation))
        open_lik = float(open_world_likelihood(observation)) if open_world_likelihood is not None else 1.0
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
        """
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
        elif method == "multinomial":
            chosen = rng.choice(n, size=n, p=p)
        else:
            raise ValueError(f"unknown resample method {method!r}")
        new_hyps = list(self.hypotheses)
        new_weights = self.weights.copy()
        seen: dict[str, int] = {}
        for slot, pick in zip(active_idx, chosen):
            src = self.hypotheses[active_idx[int(pick)]]
            count = seen.get(src.id, 0)
            seen[src.id] = count + 1
            new_id = src.id if count == 0 else f"{src.id}#{count}"
            new_hyps[slot] = Hypothesis(id=new_id, payload=src.payload, active=True)
            new_weights[slot] = total_active / n
        return HypothesisPortfolio(new_hyps, new_weights, self.w_open)

    def prune(self, *, min_weight: float) -> HypothesisPortfolio:
        """Deactivate (never delete) active hypotheses below ``min_weight``; their mass folds into ``w_open``."""
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
        """
        active = [(w, h) for w, h in zip(self.weights, self.hypotheses) if h.active]
        if not active:
            return 1.0
        total_w = sum(w for w, _ in active)
        if total_w <= 0:
            return 1.0
        weighted_lik = sum(w * float(likelihood_fn(h, observation)) for w, h in active) / total_w
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
