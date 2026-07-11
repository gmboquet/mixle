"""P16 (experimental) -- data-free spectral-health receipts for weight matrices.

Trained-weight spectra carry a training-quality signal that is readable WITHOUT any data.
Following heavy-tailed self-regularization theory, the empirical spectral density of a layer's
weights develops a power-law tail as it trains: an under-trained (near-random) layer has a
light, Marchenko-Pastur-like bulk and a large tail exponent ``alpha``; a well-trained layer
settles into ``alpha`` roughly in ``[2, 4]``; a memorizing / over-correlated layer pushes
``alpha`` below 2 (a very heavy tail) and grows spike outliers.

This module fits that spectral law per layer and returns a health receipt -- the same "fit +
goodness-of-fit" discipline mixle uses elsewhere, applied to a matrix's singular values instead
of data. It complements the moment lens (G1) and the quantile-profile lens (R1/G4) as a third,
data-free view of a weight tensor, and is meant to feed run-report health (F4) and
compressibility prediction (J3).

Exploratory ``mixle.experimental`` code (P16 card). The receipt is validated in
``spectral_health_test.py`` on matrices with known spectral laws: the metrics must order the
under-trained / well-trained / memorizing regimes as documented (the card's kill criterion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpectralReceipt:
    """Per-layer spectral-health receipt.

    ``alpha`` is the power-law tail exponent of the eigenvalue spectrum (of ``W W^T``); ``ks_d``
    is the Kolmogorov-Smirnov distance of that power-law fit (smaller = better fit). ``verdict``
    classifies the layer from ``alpha`` and the spike count.
    """

    spectral_norm: float
    frobenius_norm: float
    stable_rank: float
    effective_rank: float
    alpha: float
    ks_d: float
    lambda_min: float
    n_spikes: int
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "spectral_norm": self.spectral_norm,
            "frobenius_norm": self.frobenius_norm,
            "stable_rank": self.stable_rank,
            "effective_rank": self.effective_rank,
            "alpha": self.alpha,
            "ks_d": self.ks_d,
            "lambda_min": self.lambda_min,
            "n_spikes": self.n_spikes,
            "verdict": self.verdict,
        }


def singular_values(weight: Any) -> np.ndarray:
    """Descending singular values of a 2-D weight matrix (higher dims are flattened to 2-D)."""
    w = np.asarray(weight, dtype=float)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    elif w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    s = np.linalg.svd(w, compute_uv=False)
    return np.sort(s)[::-1]


def stable_rank(s: np.ndarray) -> float:
    """``||W||_F^2 / ||W||_2^2`` -- a soft, noise-robust rank in ``[1, min(shape)]``."""
    s2 = s**2
    top = s2[0] if s2.size else 0.0
    return float(s2.sum() / top) if top > 0 else 0.0


def effective_rank(s: np.ndarray) -> float:
    """Roy-Vetterli effective rank: ``exp`` of the Shannon entropy of the normalized spectrum."""
    if s.size == 0 or s.sum() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def _powerlaw_alpha(lambdas: np.ndarray, lambda_min: float) -> float:
    """Continuous power-law MLE exponent for eigenvalues ``>= lambda_min`` (Clauset et al.)."""
    tail = lambdas[lambdas >= lambda_min]
    if tail.size < 2 or lambda_min <= 0:
        return float("inf")
    return float(1.0 + tail.size / np.sum(np.log(tail / lambda_min)))


def _ks_distance(tail: np.ndarray, lambda_min: float, alpha: float) -> float:
    """KS distance between the empirical tail CDF and the fitted power-law CDF."""
    if tail.size < 2 or not np.isfinite(alpha) or alpha <= 1.0:
        return float("inf")
    x = np.sort(tail)
    emp = np.arange(1, x.size + 1) / x.size
    model = 1.0 - (x / lambda_min) ** (1.0 - alpha)  # CCDF complement -> CDF
    return float(np.max(np.abs(emp - model)))


def fit_tail_exponent(s: np.ndarray) -> tuple[float, float, float]:
    """Fit the eigenvalue-spectrum power-law tail, scanning ``lambda_min`` to minimize KS.

    Returns ``(alpha, ks_d, lambda_min)`` on the eigenvalues ``lambda = s^2``.
    """
    lambdas = np.sort(s**2)
    lambdas = lambdas[lambdas > 0]
    if lambdas.size < 8:
        return float("inf"), float("inf"), 0.0
    # Candidate lower bounds: distinct eigenvalues, but leave at least a handful in the tail.
    candidates = np.unique(lambdas)[:-4]
    best = (float("inf"), float("inf"), 0.0)  # (ks, alpha, lmin)
    for lmin in candidates:
        alpha = _powerlaw_alpha(lambdas, lmin)
        ks = _ks_distance(lambdas[lambdas >= lmin], lmin, alpha)
        if ks < best[0]:
            best = (ks, alpha, float(lmin))
    ks, alpha, lmin = best
    return alpha, ks, lmin


def _count_spikes(s: np.ndarray, shape: tuple[int, int]) -> int:
    """Informational count of eigenvalues far above the bulk (a rough correlation-trap signal).

    The noise floor is estimated from the LOWER half of the spectrum (the bulk), so a few large
    outliers do not inflate it; the Marchenko-Pastur edge for that noise scale sets the cutoff.
    This is only a meaningful "spike" count when the bulk is roughly random; it is reported for
    context, not used to classify (the tail exponent does that).
    """
    lambdas = s**2
    if lambdas.size < 8:
        return 0
    n, m = shape
    q = min(n, m) / max(n, m)
    bulk = lambdas[lambdas <= np.median(lambdas)]  # lower-half: the noise floor, spike-free
    sigma2 = float(np.mean(bulk)) / max(1e-12, (1.0 + np.sqrt(q)) ** 2 / 3.0) if bulk.size else 0.0
    edge = sigma2 * (1.0 + np.sqrt(q)) ** 2
    return int(np.sum(lambdas > 3.0 * edge))


def _verdict(alpha: float) -> str:
    """Classify a layer from its tail exponent (heavy-tailed self-regularization bands).

    ``alpha > 4`` -- light tail, near-random: under-trained. ``2 <= alpha <= 4`` -- the trained
    band: well-trained. ``alpha < 2`` -- a very heavy tail / correlation traps: memorizing.
    """
    if not np.isfinite(alpha) or alpha > 4.0:
        return "under-trained"
    if alpha < 2.0:
        return "memorizing"
    return "well-trained"


def spectral_health(weight: Any) -> SpectralReceipt:
    """Compute the data-free spectral-health receipt for one weight matrix."""
    w = np.asarray(weight, dtype=float)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    elif w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    s = singular_values(w)
    alpha, ks, lmin = fit_tail_exponent(s)
    n_spikes = _count_spikes(s, (w.shape[0], w.shape[1]))
    return SpectralReceipt(
        spectral_norm=float(s[0]) if s.size else 0.0,
        frobenius_norm=float(np.sqrt(np.sum(s**2))),
        stable_rank=stable_rank(s),
        effective_rank=effective_rank(s),
        alpha=alpha,
        ks_d=ks,
        lambda_min=lmin,
        n_spikes=n_spikes,
        verdict=_verdict(alpha),
    )


def model_spectral_report(named_weights: Any) -> dict[str, SpectralReceipt]:
    """Receipt per named 2-D weight matrix. ``named_weights``: mapping name -> array-like."""
    items = named_weights.items() if hasattr(named_weights, "items") else named_weights
    return {str(name): spectral_health(w) for name, w in items}
