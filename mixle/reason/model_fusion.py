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

Fusion reuses the existing Kalman path, while conflict detection stays in raw
measurement space so a shared prior cannot hide contradictory claims. The
pairwise test uses the full claim-difference covariance and a rank-aware
chi-square bound. Merging
non-scalar structured knowledge (graphs/tables/images) across models is a different lifecycle
owned by the knowledge-conflict machinery, not this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

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
        if not np.isfinite(self.reliability) or not (0.0 < self.reliability <= 1.0):
            raise ValueError(f"reliability must be in (0, 1]; got {self.reliability!r}")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not isinstance(self.version, str) or not self.version.strip():
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
    keyed by ``model_id@version``. ``disagreement`` reports covariance-aware
    pairwise tests of the raw claims in measurement space.
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
    scaled: list[LinearGaussianEvidence],
    *,
    query: Any,
    disagree_sigma: float,
    latent_dim: int,
    cross_covariances: Mapping[tuple[str, str], Any] | None,
) -> dict[str, Any]:
    """Covariance-aware pairwise conflict in the models' raw measurement space.

    This deliberately excludes the shared prior: comparing solo posteriors
    would correlate and pull both claims through that same prior. For matching
    measurement operators, the raw difference has covariance
    ``R_i + R_j - C_ij - C_ij.T``. A chi-square decision accounts for
    covariance geometry and effective rank.
    """
    query_idx = _query_indices(query, latent_dim)
    names = [e.name for e in scaled]
    pairwise_sigma: dict[str, float] = {}
    pairwise: dict[str, dict[str, Any]] = {}
    incomparable: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scaled[i], scaled[j]
            pair_name = f"{names[i]}|{names[j]}"
            h_a = np.asarray(a.H, dtype=float)
            h_b = np.asarray(b.H, dtype=float)
            if h_a.shape != h_b.shape or not np.allclose(h_a, h_b, rtol=1e-10, atol=1e-12):
                incomparable.append(pair_name)
                pairwise[pair_name] = {"status": "incomparable_measurement_operators"}
                continue
            relevant = np.any(np.abs(h_a[:, query_idx]) > 1e-12, axis=1)
            if not np.any(relevant):
                pairwise[pair_name] = {"status": "no_information_about_query"}
                pairwise_sigma[pair_name] = 0.0
                continue
            delta = np.asarray(a.y, dtype=float)[relevant] - np.asarray(b.y, dtype=float)[relevant]
            r_a = np.asarray(a.R, dtype=float)[np.ix_(relevant, relevant)]
            r_b = np.asarray(b.R, dtype=float)[np.ix_(relevant, relevant)]
            cross = _cross_covariance(cross_covariances, names[i], names[j], len(a.y))
            cross = cross[np.ix_(relevant, relevant)]
            covariance = r_a + r_b - cross - cross.T
            squared, dof = _mahalanobis_squared(delta, covariance)
            sigma = float(np.sqrt(squared))
            confidence = float(2.0 * norm.cdf(disagree_sigma) - 1.0)
            critical = float(chi2.ppf(confidence, dof)) if dof else 0.0
            p_value = float(chi2.sf(squared, dof)) if dof and np.isfinite(squared) else 0.0
            pairwise_sigma[pair_name] = sigma
            pairwise[pair_name] = {
                "status": "comparable",
                "mahalanobis_squared": squared,
                "degrees_of_freedom": dof,
                "critical_value": critical,
                "p_value": p_value,
                "flagged": bool(squared > critical),
            }
    max_sigma = max(pairwise_sigma.values()) if pairwise_sigma else 0.0
    flagged = [pair for pair, result in pairwise.items() if result.get("flagged", False)]
    return {
        "pairwise_sigma": pairwise_sigma,
        "pairwise": pairwise,
        "max_sigma": max_sigma,
        "disagree_sigma": float(disagree_sigma),
        "flagged_pairs": flagged,
        "incomparable_pairs": incomparable,
        "covariance_assumption": "independent unless pairwise cross-covariance is supplied",
    }


def fuse_models(
    prior: GaussianBelief,
    claims: list[ModelClaim],
    *,
    query: Any = None,
    disagree_sigma: float = 3.0,
    cross_covariances: Mapping[tuple[str, str], Any] | None = None,
) -> ModelFusionResult:
    """Fuse several models' claims about the same latent into one belief (precision-weighted Kalman).

    Each claim's noise is scaled by ``1 / reliability`` and named ``model_id@version`` (DR-ALG L5/M2
    precision weighting), then all claims are folded into ``prior`` in one pass via
    :func:`mixle.reason.core.reason` -- the existing multi-source assimilation path, with one
    evidence per *model* rather than per modality. Per-model attribution comes straight from
    ``ReasonedAnswer.attribution()``. Before returning, each pair of models' *individual* (solo)
    raw measurement claims are compared with a rank-aware Mahalanobis test
    whose chi-square confidence mass corresponds to ``disagree_sigma``. An
    incomparable pair also causes abstention rather than silently passing.

    Args:
        prior: the latent's prior belief.
        claims: one :class:`ModelClaim` per model (must have distinct ``model_id@version`` identity).
        query: optional latent coordinate indices to restrict the fused answer (and the
            disagreement check) to.
        disagree_sigma: the standardized-distance threshold above which models are considered to
            be in conflict.
        cross_covariances: optional pairwise covariance between model
            measurement errors, keyed by ``(model_name, model_name)``. Missing
            pairs are explicitly treated as independent.

    Returns:
        A :class:`ModelFusionResult` with the fused answer, per-model weights, the disagreement
        report, and the abstain flag.
    """
    if not claims:
        raise ValueError("fuse_models requires at least one ModelClaim")
    if not np.isfinite(disagree_sigma) or disagree_sigma < 0.0:
        raise ValueError("disagree_sigma must be finite and non-negative.")
    scaled = [_precision_scaled_evidence(c) for c in claims]
    names = [e.name for e in scaled]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model claim identity (model_id@version): {names}")

    fused = reason(prior, scaled, query=query)
    weights = fused.attribution()
    disagreement = _pairwise_disagreement(
        scaled,
        query=query,
        disagree_sigma=float(disagree_sigma),
        latent_dim=prior.dim,
        cross_covariances=cross_covariances,
    )
    abstain = bool(disagreement["flagged_pairs"] or disagreement["incomparable_pairs"])
    return ModelFusionResult(answer=fused, weights=weights, disagreement=disagreement, abstain=abstain)


def _query_indices(query: Any, latent_dim: int) -> np.ndarray:
    if query is None:
        return np.arange(latent_dim)
    idx = np.asarray(query)
    if idx.ndim == 0:
        idx = idx.reshape(1)
    if idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu":
        raise ValueError("query must be a non-empty one-dimensional integer index.")
    idx = idx.astype(int, copy=False)
    if np.any(idx < 0) or np.any(idx >= latent_dim) or len(np.unique(idx)) != len(idx):
        raise ValueError("query indices must be unique and within the latent dimension.")
    return idx


def _cross_covariance(
    covariances: Mapping[tuple[str, str], Any] | None,
    first: str,
    second: str,
    dimension: int,
) -> np.ndarray:
    if covariances is None:
        return np.zeros((dimension, dimension))
    forward = (first, second)
    reverse = (second, first)
    if forward in covariances and reverse in covariances:
        raise ValueError(f"supply only one orientation of cross-covariance for {first!r} and {second!r}.")
    if forward in covariances:
        value = np.asarray(covariances[forward], dtype=float)
    elif reverse in covariances:
        value = np.asarray(covariances[reverse], dtype=float).T
    else:
        return np.zeros((dimension, dimension))
    if value.shape != (dimension, dimension) or not np.isfinite(value).all():
        raise ValueError(
            f"cross-covariance for {first!r}, {second!r} must be finite with shape {(dimension, dimension)}."
        )
    return value


def _mahalanobis_squared(delta: np.ndarray, covariance: np.ndarray) -> tuple[float, int]:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all():
        raise ValueError("claim-difference covariance must be finite.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * np.finfo(float).eps * len(eigenvalues) * 100.0
    if np.any(eigenvalues < -tolerance):
        raise ValueError("claim-difference covariance must be positive semidefinite.")
    positive = eigenvalues > tolerance
    projected = eigenvectors.T @ np.asarray(delta, dtype=float)
    if np.any(np.abs(projected[~positive]) > np.sqrt(tolerance)):
        return float("inf"), int(np.sum(positive))
    if not np.any(positive):
        return 0.0, 0
    return float(np.sum(np.square(projected[positive]) / eigenvalues[positive])), int(np.sum(positive))
