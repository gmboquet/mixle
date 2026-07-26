"""Canonical non-finite evidence semantics for finite-mixture kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class ImpossibleEvidencePolicy(StrEnum):
    """How an E-step handles a row outside every component's support."""

    ZERO_RESPONSIBILITY = "zero_responsibility"


IMPOSSIBLE_EVIDENCE_POLICY = ImpossibleEvidencePolicy.ZERO_RESPONSIBILITY


class InvalidMixtureEvidenceError(ValueError):
    """Raised when component scores cannot define a mixture posterior."""

    def __init__(self, rows: Any, reason: str) -> None:
        self.rows = tuple(int(row) for row in np.asarray(rows, dtype=np.int64).reshape(-1))
        self.reason = reason
        super().__init__("invalid mixture evidence at rows %s: %s" % (self.rows, reason))


@dataclass(frozen=True)
class MixtureEvidence:
    """Normalized evidence and responsibilities under the canonical policy."""

    log_evidence: Any
    responsibilities: Any
    impossible: Any


def normalize_mixture_log_scores(weighted_log_scores: Any) -> MixtureEvidence:
    """Normalize a NumPy ``(rows, components)`` weighted log-score matrix.

    Impossible rows retain ``-inf`` evidence and receive all-zero
    responsibilities, so they cannot fabricate sufficient statistics. A
    unique ``+inf`` component is the exact winner. NaN scores and multiple
    ``+inf`` components are ambiguous and raise a typed error.
    """
    scores = np.asarray(weighted_log_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("mixture weighted log scores must have shape (rows, nonzero components).")

    nan_rows = np.flatnonzero(np.isnan(scores).any(axis=1))
    if nan_rows.size:
        raise InvalidMixtureEvidenceError(nan_rows, "component scores contain NaN")

    positive_infinite = np.isposinf(scores)
    positive_counts = positive_infinite.sum(axis=1)
    ambiguous_rows = np.flatnonzero(positive_counts > 1)
    if ambiguous_rows.size:
        raise InvalidMixtureEvidenceError(ambiguous_rows, "multiple components have +inf weighted score")

    unique_positive = positive_counts == 1
    impossible = np.isneginf(scores).all(axis=1)
    ordinary = ~(unique_positive | impossible)

    evidence = np.full(scores.shape[0], -np.inf, dtype=np.float64)
    responsibilities = np.zeros_like(scores)
    evidence[unique_positive] = np.inf
    responsibilities[unique_positive] = positive_infinite[unique_positive]

    if ordinary.any():
        row_scores = scores[ordinary]
        maxima = row_scores.max(axis=1, keepdims=True)
        shifted = np.exp(row_scores - maxima)
        totals = shifted.sum(axis=1, keepdims=True)
        evidence[ordinary] = maxima[:, 0] + np.log(totals[:, 0])
        responsibilities[ordinary] = shifted / totals

    return MixtureEvidence(evidence, responsibilities, impossible)


def normalize_engine_mixture_log_scores(weighted_log_scores: Any, engine: Any) -> MixtureEvidence:
    """Engine-preserving counterpart of :func:`normalize_mixture_log_scores`."""
    scores = weighted_log_scores
    shape = tuple(getattr(scores, "shape", ()))
    if len(shape) != 2 or shape[1] == 0:
        raise ValueError("mixture weighted log scores must have shape (rows, nonzero components).")

    nan_mask = engine.isnan(scores)
    nan_rows = np.flatnonzero(np.asarray(engine.to_numpy(engine.sum(nan_mask, axis=1))) > 0)
    if nan_rows.size:
        raise InvalidMixtureEvidenceError(nan_rows, "component scores contain NaN")

    positive_infinite = engine.isinf(scores) & (scores > engine.asarray(0.0))
    positive_counts = engine.sum(positive_infinite, axis=1)
    positive_counts_host = np.asarray(engine.to_numpy(positive_counts))
    ambiguous_rows = np.flatnonzero(positive_counts_host > 1)
    if ambiguous_rows.size:
        raise InvalidMixtureEvidenceError(ambiguous_rows, "multiple components have +inf weighted score")

    evidence = engine.logsumexp(scores, axis=1)
    unique_positive = positive_counts > engine.asarray(0)
    impossible = engine.isinf(evidence) & (evidence < engine.asarray(0.0))
    special = unique_positive | impossible
    safe_evidence = engine.where(special, engine.asarray(0.0), evidence)
    safe_scores = engine.where(special[:, None], engine.asarray(0.0), scores)
    ordinary = engine.exp(safe_scores - safe_evidence[:, None])
    one_hot = engine.where(positive_infinite, engine.asarray(1.0), engine.asarray(0.0))
    responsibilities = engine.where(unique_positive[:, None], one_hot, ordinary)
    responsibilities = engine.where(
        impossible[:, None],
        engine.zeros(shape),
        responsibilities,
    )
    return MixtureEvidence(evidence, responsibilities, impossible)


def raise_for_invalid_log_evidence(log_evidence: Any) -> None:
    """Raise the typed evidence error when a generated scorer reports NaN."""
    rows = np.flatnonzero(np.isnan(np.asarray(log_evidence, dtype=np.float64)))
    if rows.size:
        raise InvalidMixtureEvidenceError(rows, "component scores contain NaN or ambiguous +inf evidence")
