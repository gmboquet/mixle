"""Epistemic / aleatoric uncertainty decomposition for any predictive.

Current LLMs emit a point estimate, so their "confidence" cannot be split into *what is
irreducibly noisy* (aleatoric) versus *what more data would resolve* (epistemic). A model that
carries a posterior can. This module makes that split a first-class, model-agnostic operation --
generalizing :meth:`mixle.stats.graphs.knowledge_graph.KnowledgeGraphEnsemble.epistemic_tail_uncertainty`
(which did it only for knowledge-graph tails) to any predictive.

Two *exact* decompositions, matched to the two answer types:

* **discrete outcomes -> the entropy (BALD mutual-information) split**::

      total     = H( mean_m p_m )          predictive entropy (all uncertainty)
      aleatoric = mean_m H( p_m )          expected member entropy (genuine ambiguity)
      epistemic = total - aleatoric >= 0   mutual information (disagreement among members)

  The epistemic term is the Bayesian-active-learning-by-disagreement (BALD) score: it is zero
  when every posterior draw agrees, and large where they disagree -- i.e. where more data helps.

* **continuous outcomes -> the law-of-total-variance split**::

      aleatoric = mean_m Var_m             expected member variance (irreducible noise)
      epistemic = Var_m( mean_m )          variance of member means (model uncertainty)
      total     = aleatoric + epistemic    total predictive variance

"Members" are draws from ``q(theta | data)`` (parameter uncertainty, via
:class:`~mixle.inference.posterior.ParameterPosterior`) or an explicit ensemble of fitted models
(as in a deep ensemble / bagged fit). Both splits are exact given the members; the only
approximation is the finite number of members used to represent the posterior.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.utils.special import softmax as _softmax

__all__ = [
    "UncertaintyDecomposition",
    "decompose_entropy",
    "decompose_variance",
    "predictive_distribution",
    "posterior_ensemble",
    "decompose_uncertainty",
]


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


def _entropy_last(p: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) over the last axis, with the ``0 log 0 = 0`` guard."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0.0, p * np.log(p), 0.0), axis=-1)


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """A predictive uncertainty split into ``aleatoric`` + ``epistemic`` (summing to ``total``).

    ``kind`` is ``"entropy"`` (values in nats) or ``"variance"`` (values in the outcome's squared
    units). Each field is a scalar for a single query point, or an array over query points.
    """

    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    kind: str

    @property
    def fraction_epistemic(self) -> np.ndarray:
        """Share of the total uncertainty that is epistemic (reducible by more data), in ``[0, 1]``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(self.total > 0.0, self.epistemic / self.total, 0.0)
        return frac

    def item(self) -> UncertaintyDecomposition:
        """Collapse size-1 arrays to Python floats (convenience for single-point decompositions)."""
        if np.size(self.total) != 1:
            raise ValueError("item() only applies to a single-point decomposition")
        return UncertaintyDecomposition(
            float(np.reshape(self.total, -1)[0]),
            float(np.reshape(self.aleatoric, -1)[0]),
            float(np.reshape(self.epistemic, -1)[0]),
            self.kind,
        )


def decompose_entropy(member_probs: Any) -> UncertaintyDecomposition:
    """BALD entropy split of a discrete predictive.

    Args:
        member_probs: array ``(M, ..., K)`` -- ``M`` posterior draws / ensemble members, each a
            categorical predictive over ``K`` outcomes (optionally batched over query points in the
            middle axes). Rows need not be normalized; each is renormalized over the last axis.

    Returns:
        An :class:`UncertaintyDecomposition` with ``kind="entropy"`` (nats). ``epistemic`` is the
        mutual information ``H(mean) - mean H`` and is clamped to ``>= 0`` (it is non-negative in
        exact arithmetic; the clamp only removes tiny floating-point negatives).
    """
    p = np.asarray(member_probs, dtype=float)
    if p.ndim < 2:
        raise ValueError("member_probs must have shape (M, ..., K) with at least a member and outcome axis")
    if p.shape[0] < 2:
        raise ValueError("need at least two members (M >= 2) to estimate epistemic uncertainty")
    totals = p.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0.0, p / totals, 0.0)
    mean = p.mean(axis=0)  # (..., K)
    total = _entropy_last(mean)  # H(mean)  -> (...)
    aleatoric = _entropy_last(p).mean(axis=0)  # mean_m H(p_m) -> (...)
    epistemic = np.maximum(total - aleatoric, 0.0)
    return UncertaintyDecomposition(total, aleatoric, epistemic, "entropy")


def decompose_variance(member_means: Any, member_vars: Any = None) -> UncertaintyDecomposition:
    """Law-of-total-variance split of a continuous predictive.

    Args:
        member_means: array ``(M, ...)`` -- each member's predictive mean ``E[y | theta_m]``.
        member_vars: array ``(M, ...)`` -- each member's predictive variance ``Var[y | theta_m]``.
            If ``None``, aleatoric noise is taken as zero (members are point predictors) and the
            decomposition reports only the epistemic spread of the means.

    Returns:
        An :class:`UncertaintyDecomposition` with ``kind="variance"``:
        ``aleatoric = mean_m Var_m``, ``epistemic = Var_m(mean_m)``, ``total`` their sum.
    """
    mu = np.asarray(member_means, dtype=float)
    if mu.shape[0] < 2:
        raise ValueError("need at least two members (M >= 2) to estimate epistemic uncertainty")
    epistemic = mu.var(axis=0)  # Var_m E[y|m]
    if member_vars is None:
        aleatoric = np.zeros_like(epistemic)
    else:
        v = np.asarray(member_vars, dtype=float)
        if v.shape != mu.shape:
            raise ValueError(f"member_vars shape {v.shape} must match member_means shape {mu.shape}")
        aleatoric = v.mean(axis=0)  # mean_m Var[y|m]
    total = aleatoric + epistemic
    return UncertaintyDecomposition(total, aleatoric, epistemic, "variance")


def predictive_distribution(members: Iterable[Any], support: Sequence[Any]) -> np.ndarray:
    """Evaluate an iterable of fitted distributions over a discrete ``support`` -> ``(M, K)`` probs.

    Each member's ``log_density`` is evaluated at every point of ``support`` and softmax-normalized
    over the support, giving one categorical row per member. Feed the result to
    :func:`decompose_entropy`.
    """
    support = list(support)
    if len(support) < 2:
        raise ValueError("support must contain at least two outcomes")
    rows = []
    for m in members:
        logs = np.array([float(m.log_density(s)) for s in support], dtype=float)
        rows.append(_softmax(logs))
    out = np.asarray(rows, dtype=float)
    if out.shape[0] < 2:
        raise ValueError("need at least two members (M >= 2) to estimate epistemic uncertainty")
    return out


def posterior_ensemble(param_post: Any, build: Callable[[Any], Any], n: int = 200, rng: Any = None) -> list[Any]:
    """Materialize ``n`` models from a parameter posterior -- an ensemble representing ``q(theta|data)``.

    ``build`` maps one parameter draw (whatever :meth:`ParameterPosterior.sample` returns) to a
    fitted distribution, mirroring
    :meth:`~mixle.inference.posterior.PredictivePosterior.from_parameter_posterior`. The returned
    list is the "members" the decomposition integrates over -- so epistemic uncertainty here is
    genuine *parameter* uncertainty, not just ensemble disagreement.
    """
    r = _as_rng(rng)
    return [build(param_post.sample(r)) for _ in range(int(n))]


def decompose_uncertainty(
    *,
    probs: Any = None,
    means: Any = None,
    variances: Any = None,
) -> UncertaintyDecomposition:
    """Front door: decompose a predictive into aleatoric + epistemic uncertainty.

    Pass exactly one representation of the per-member predictive:

    * ``probs=(M, ..., K)`` -- categorical predictives -> BALD entropy split
      (:func:`decompose_entropy`);
    * ``means=(M, ...)`` (with optional ``variances=(M, ...)``) -- continuous predictives ->
      law-of-total-variance split (:func:`decompose_variance`).

    To decompose a *fitted* model's parameter uncertainty, build the members first with
    :func:`posterior_ensemble` (then :func:`predictive_distribution` for the discrete case).
    """
    if probs is not None and means is not None:
        raise ValueError("pass either probs= (discrete) or means= (continuous), not both")
    if probs is not None:
        return decompose_entropy(probs)
    if means is not None:
        return decompose_variance(means, variances)
    raise ValueError("provide probs= (discrete) or means= (continuous)")
