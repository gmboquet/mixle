"""The reasoning front door: fuse modality evidence into a belief, query it with native UQ.

``reason(prior, evidence)`` folds a sequence of linear-Gaussian observations into a belief state
by exact Kalman assimilation, tracking how many nats of uncertainty each modality removed. The
returned :class:`ReasonedAnswer` is the posterior belief plus the tools a scientific answer needs:
credible intervals, per-modality attribution, and an epistemic/aleatoric split of any prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.belief import BeliefState, GaussianBelief
from mixle.inference.uncertainty import UncertaintyDecomposition


class Latent:
    """Factories for the shared latent's prior belief (the starting point of assimilation)."""

    @staticmethod
    def gaussian(mean: Any, cov: Any) -> GaussianBelief:
        """A Gaussian prior ``N(mean, cov)`` over the latent."""
        return GaussianBelief(mean, cov)

    @staticmethod
    def vector(dim: int, *, mean: float = 0.0, var: float = 1.0) -> GaussianBelief:
        """An isotropic Gaussian prior over a ``dim``-vector latent: ``N(mean*1, var*I)``."""
        return GaussianBelief(np.full(int(dim), float(mean)), np.eye(int(dim)) * float(var))


@dataclass(frozen=True)
class LinearGaussianEvidence:
    """One modality's evidence about the latent ``z``: ``y = H z + noise``, ``noise ~ N(0, R)``.

    ``H`` is the (possibly linearized) forward operator mapping the latent to this modality's
    measurement space, ``y`` the observed data, ``R`` its noise covariance (matrix, diagonal, or
    scalar). Application forward models (e.g. ``mixle_pde`` geophysics operators) produce these.
    """

    H: Any
    y: Any
    R: Any
    name: str = ""


#: Short alias -- ``Evidence(H, y, R, name)``.
Evidence = LinearGaussianEvidence


class ReasonedAnswer:
    """A posterior belief about a query, with the UQ a scientific answer needs.

    Beyond ``mean`` / ``interval`` / ``entropy`` (delegated to the belief), it exposes
    :meth:`attribution` -- the nats of uncertainty each modality removed -- and :meth:`predict`,
    which splits a *prediction's* uncertainty into epistemic (from latent uncertainty) and
    aleatoric (observation noise) via the law of total variance.
    """

    def __init__(self, belief: BeliefState, prior_entropy: float, contributions: dict[str, float]) -> None:
        self.belief = belief
        self._prior_entropy = float(prior_entropy)
        self._contributions = dict(contributions)

    @property
    def mean(self) -> np.ndarray:
        return self.belief.mean()

    def cov(self) -> np.ndarray:
        return self.belief.cov()

    def sd(self) -> np.ndarray:
        return self.belief.sd()

    def entropy(self) -> float:
        return self.belief.entropy()

    def interval(self, level: float = 0.9) -> np.ndarray:
        """Per-coordinate central credible interval at ``level`` (an ``(d, 2)`` array of ``[lo, hi]``)."""
        return self.belief.interval(level)

    def information_gain(self) -> float:
        """Total nats of uncertainty the evidence removed from the prior (``H[prior] - H[posterior]``)."""
        return self._prior_entropy - self.belief.entropy()

    def attribution(self, *, normalize: bool = False) -> dict[str, float]:
        """Per-modality information gain in nats -- which modality sharpened the belief, and by how much.

        With ``normalize=True``, values are the fraction of the total gain (they then sum to ~1).
        """
        contrib = dict(self._contributions)
        if normalize:
            total = sum(contrib.values())
            if total > 0:
                contrib = {k: v / total for k, v in contrib.items()}
        return contrib

    def predict(self, H: Any, R: Any = 0.0) -> UncertaintyDecomposition:
        """Split the uncertainty of a new prediction ``y* = H z + noise(R)`` (law of total variance).

        ``epistemic = diag(H P Hᵀ)`` (from the latent's remaining uncertainty, reducible by more
        data) and ``aleatoric = diag(R)`` (irreducible observation noise). Exact for the Gaussian
        belief.
        """
        P = np.atleast_2d(self.belief.cov())
        Hm = np.atleast_2d(np.asarray(H, dtype=float))
        if Hm.shape[1] != P.shape[0]:
            Hm = Hm.reshape(-1, P.shape[0])
        epistemic = np.diag(Hm @ P @ Hm.T).copy()
        Rm = np.asarray(R, dtype=float)
        if Rm.ndim == 0:
            aleatoric = np.full(epistemic.shape, float(Rm))
        elif Rm.ndim == 1:
            aleatoric = Rm.copy()
        else:
            aleatoric = np.diag(Rm).copy()
        total = epistemic + aleatoric
        return UncertaintyDecomposition(total, aleatoric, epistemic, "variance")

    def marginal(self, indices: Any) -> ReasonedAnswer:
        """Restrict the answer to a subset of latent coordinates (query a specific variable)."""
        sub = self.belief.marginal(indices)
        return ReasonedAnswer(sub, self._prior_entropy, self._contributions)

    def __repr__(self) -> str:
        return (
            f"ReasonedAnswer(dim={np.size(self.mean)}, "
            f"info_gain={self.information_gain():.3f} nats, "
            f"modalities={list(self._contributions)})"
        )


def _to_belief(prior: Any) -> GaussianBelief:
    if isinstance(prior, GaussianBelief):
        return prior
    if isinstance(prior, BeliefState):
        return GaussianBelief(prior.mean(), prior.cov())
    raise TypeError(f"prior must be a GaussianBelief (or BeliefState with cov), got {type(prior).__name__}")


def reason(prior: Any, evidence: Any, *, query: Any = None) -> ReasonedAnswer:
    """Fuse ``evidence`` into ``prior`` by exact Kalman assimilation; return the queried posterior.

    Args:
        prior: the latent's prior belief (:class:`GaussianBelief`; build one with :class:`Latent`).
        evidence: a sequence of :class:`LinearGaussianEvidence` -- one per modality / observation.
            They are folded in one at a time (order does not affect the result), and the nats each
            removes are recorded for :meth:`ReasonedAnswer.attribution`.
        query: optional latent coordinate indices to restrict the answer to.

    Returns:
        A :class:`ReasonedAnswer` -- the posterior belief plus attribution and prediction UQ.
    """
    belief = _to_belief(prior)
    prior_entropy = belief.entropy()
    contributions: dict[str, float] = {}
    for i, e in enumerate(evidence):
        name = e.name or f"evidence[{i}]"
        before = belief.entropy()
        belief = belief.update(e.H, e.y, e.R)
        gain = before - belief.entropy()
        contributions[name] = contributions.get(name, 0.0) + gain
    answer = ReasonedAnswer(belief, prior_entropy, contributions)
    if query is not None:
        answer = answer.marginal(query)
    return answer
