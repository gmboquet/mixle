"""Numeric cross-model belief fusion -- precision-weighted assimilation across MODELS, not modalities.

``mixle.reason.core.reason`` already folds a sequence of per-*modality* linear-Gaussian evidence
into a shared latent belief (a product of experts). This module reuses exactly that Kalman path to
fuse per-*model* claims about the same latent: several independently-run models (e.g. two physics
solvers, or a solver and a learned emulator) each hand back a linear-Gaussian view of the same
quantity, tagged with the model's identity and an operator-supplied reliability. ``fuse_models``
scales each claim's noise by its reliability (inverse-variance precision weighting), assimilates
all of them in one pass, and reports which model contributed how much (in nats, via
``ReasonedAnswer.attribution``) plus whether the models disagree badly enough that the fused
answer should not be trusted outright.

This is deliberately *not* new fusion math: the only new logic here is precision-scaling claim
noise and a pairwise disagreement check before/around the existing ``reason`` call. Merging
non-scalar structured knowledge (graphs/tables/images) across models is a different lifecycle
owned by the knowledge-conflict machinery, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.reason.core import GaussianBelief, LinearGaussianEvidence, ReasonedAnswer, reason


@dataclass(frozen=True)
class ModelClaim:
    """One model's linear-Gaussian claim about the shared latent, tagged with identity + trust.

    ``evidence`` is the model's raw ``(H, y, R)`` observation of the latent (its ``name`` is
    ignored -- :func:`fuse_models` stamps ``f"{model_id}@{version}"`` on it so attribution and
    disagreement reporting are keyed by model identity rather than whatever the caller set).
    ``reliability`` in ``(0, 1]`` is how much the claim's stated noise should be trusted: ``1.0``
    takes ``R`` at face value; smaller values inflate ``R`` (divide by ``reliability``), so a
    known-flaky model's claim is assimilated as effectively noisier and contributes less
    information to the fused belief.
    """

    evidence: LinearGaussianEvidence
    model_id: str
    version: str
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.reliability <= 1.0):
            raise ValueError(f"reliability must be in (0, 1]; got {self.reliability!r}")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")

    @property
    def name(self) -> str:
        """The stable ``model_id@version`` identity used as this claim's evidence/attribution key."""
        return f"{self.model_id}@{self.version}"


@dataclass
class ModelFusionResult:
    """The outcome of fusing several models' claims about one latent.

    ``answer`` is the fused :class:`ReasonedAnswer` (posterior belief + UQ). ``weights`` is each
    model's attribution -- nats of uncertainty it removed (``ReasonedAnswer.attribution()``),
    keyed by ``model_id@version``. ``disagreement`` reports the pairwise standardized differences
    between models' individual (solo) views of the query, in the same units as ``disagree_sigma``.
    ``abstain`` is ``True`` when any pair disagrees by more than ``disagree_sigma`` -- the fused
    number should not be trusted at face value; callers are expected to route that case to a
    verifier (IC-6) / an abstaining natural-language surface rather than presenting the fused mean
    as settled.
    """

    answer: ReasonedAnswer
    weights: dict[str, float]
    disagreement: dict[str, Any]
    abstain: bool


def _precision_scaled_evidence(claim: ModelClaim) -> LinearGaussianEvidence:
    """Return ``claim.evidence`` with its noise ``R`` divided by reliability and named by model identity."""
    e = claim.evidence
    scaled_R = np.asarray(e.R, dtype=float) / claim.reliability
    return LinearGaussianEvidence(H=e.H, y=e.y, R=scaled_R, name=claim.name)


def _pairwise_disagreement(
    prior: GaussianBelief,
    scaled: list[LinearGaussianEvidence],
    *,
    query: Any,
    disagree_sigma: float,
) -> dict[str, Any]:
    """Standardized pairwise distance between each model's solo posterior (folding only that model's
    evidence into ``prior``), restricted to ``query`` when given -- the same readout ``fuse_models``
    ultimately returns. A pair's sigma is the largest per-coordinate ``|mean_i - mean_j| / sqrt(var_i
    + var_j)`` across the queried coordinates -- a conservative (worst-coordinate) conflict test.
    """
    solo = {e.name: reason(prior, [e], query=query) for e in scaled}
    names = list(solo)
    pairwise_sigma: dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = solo[names[i]], solo[names[j]]
            diff = np.abs(a.mean - b.mean)
            var_sum = np.maximum(np.diag(np.atleast_2d(a.cov())) + np.diag(np.atleast_2d(b.cov())), 1e-300)
            sigma = float(np.max(diff / np.sqrt(var_sum))) if diff.size else 0.0
            pairwise_sigma[f"{names[i]}|{names[j]}"] = sigma
    max_sigma = max(pairwise_sigma.values()) if pairwise_sigma else 0.0
    flagged = [pair for pair, sigma in pairwise_sigma.items() if sigma > disagree_sigma]
    return {
        "pairwise_sigma": pairwise_sigma,
        "max_sigma": max_sigma,
        "disagree_sigma": float(disagree_sigma),
        "flagged_pairs": flagged,
    }


def fuse_models(
    prior: GaussianBelief,
    claims: list[ModelClaim],
    *,
    query: Any = None,
    disagree_sigma: float = 3.0,
) -> ModelFusionResult:
    """Fuse several models' claims about the same latent into one belief (precision-weighted Kalman).

    Each claim's noise is scaled by ``1 / reliability`` and named ``model_id@version`` (DR-ALG L5/M2
    precision weighting), then all claims are folded into ``prior`` in one pass via
    :func:`mixle.reason.core.reason` -- the existing multi-source assimilation path, with one
    evidence per *model* rather than per modality. Per-model attribution comes straight from
    ``ReasonedAnswer.attribution()``. Before returning, each pair of models' *individual* (solo)
    views of the query are compared; if any pair disagrees by more than ``disagree_sigma`` standard
    deviations, ``abstain`` is set so callers don't present the fused answer as settled without
    further verification.

    Args:
        prior: the latent's prior belief.
        claims: one :class:`ModelClaim` per model (must have distinct ``model_id@version`` identity).
        query: optional latent coordinate indices to restrict the fused answer (and the
            disagreement check) to.
        disagree_sigma: the standardized-distance threshold above which models are considered to
            be in conflict.

    Returns:
        A :class:`ModelFusionResult` with the fused answer, per-model weights, the disagreement
        report, and the abstain flag.
    """
    if not claims:
        raise ValueError("fuse_models requires at least one ModelClaim")
    scaled = [_precision_scaled_evidence(c) for c in claims]
    names = [e.name for e in scaled]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model claim identity (model_id@version): {names}")

    fused = reason(prior, scaled, query=query)
    weights = fused.attribution()
    disagreement = _pairwise_disagreement(prior, scaled, query=query, disagree_sigma=disagree_sigma)
    abstain = bool(disagreement["flagged_pairs"])
    return ModelFusionResult(answer=fused, weights=weights, disagreement=disagreement, abstain=abstain)
