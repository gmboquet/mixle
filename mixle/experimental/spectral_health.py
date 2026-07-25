"""Descriptive spectral receipts for real, finite weight tensors.

The singular spectrum, stable/effective rank, outlier count, and a power-law tail fit can describe
a weight matrix without evaluation data. They do **not**, by themselves, establish that a model is
under-trained, well-trained, memorizing, or otherwise healthy. This module therefore reports fit
quality, tail sample size, and a conditional bootstrap interval, and always abstains from scientific
diagnosis until a separately validated and calibrated diagnostic model is supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectralReceipt:
    """Descriptive statistics and an explicitly scoped tail-fit receipt."""

    matrix_shape: tuple[int, int]
    n_singular_values: int
    spectral_norm: float
    frobenius_norm: float
    stable_rank: float
    effective_rank: float
    alpha: float | None
    alpha_ci_low: float | None
    alpha_ci_high: float | None
    ks_d: float | None
    lambda_min: float | None
    tail_points: int
    bootstrap_samples: int
    tail_fit_status: str
    tail_fit_accepted: bool
    n_spikes: int
    verdict: None
    diagnostic_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weight_matrix(weight: Any) -> np.ndarray:
    raw = np.asarray(weight)
    if np.iscomplexobj(raw):
        raise TypeError("weight must be real; complex matrices require a separately defined spectral contract")
    if raw.ndim < 2:
        raise ValueError("weight must have at least two dimensions")
    if raw.size == 0 or any(size == 0 for size in raw.shape):
        raise ValueError("weight must have no empty dimensions")
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("weight must be coercible to a real floating-point matrix") from exc
    if not np.isfinite(matrix).all():
        raise ValueError("weight must contain only finite values")
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix


def _spectrum(s: Any) -> np.ndarray:
    values = np.asarray(s)
    if np.iscomplexobj(values):
        raise TypeError("singular values must be real")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular values must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("singular values must be finite and non-negative")
    return values


def singular_values(weight: Any) -> np.ndarray:
    """Descending singular values; dimensions after the first are flattened into columns."""
    w = _weight_matrix(weight)
    s = np.linalg.svd(w, compute_uv=False)
    return np.sort(s)[::-1]


def stable_rank(s: np.ndarray) -> float:
    """``||W||_F^2 / ||W||_2^2`` -- a soft, noise-robust rank in ``[1, min(shape)]``."""
    s = _spectrum(s)
    s2 = s**2
    top = s2[0] if s2.size else 0.0
    return float(s2.sum() / top) if top > 0 else 0.0


def effective_rank(s: np.ndarray) -> float:
    """Roy-Vetterli effective rank: ``exp`` of the Shannon entropy of the normalized spectrum."""
    s = _spectrum(s)
    if s.sum() == 0:
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
    denominator = float(np.sum(np.log(tail / lambda_min)))
    return float(1.0 + tail.size / denominator) if denominator > 0 else float("inf")


def _ks_distance(tail: np.ndarray, lambda_min: float, alpha: float) -> float:
    """KS distance between the empirical tail CDF and the fitted power-law CDF."""
    if tail.size < 2 or not np.isfinite(alpha) or alpha <= 1.0:
        return float("inf")
    x = np.sort(tail)
    emp = np.arange(1, x.size + 1) / x.size
    model = 1.0 - (x / lambda_min) ** (1.0 - alpha)  # CCDF complement -> CDF
    return float(np.max(np.abs(emp - model)))


def fit_tail_exponent(s: np.ndarray, *, min_tail_points: int = 50) -> tuple[float, float, float]:
    """Fit the eigenvalue-spectrum power-law tail, scanning ``lambda_min`` to minimize KS.

    Returns ``(alpha, ks_d, lambda_min)`` on the eigenvalues ``lambda = s^2``.
    """
    s = _spectrum(s)
    min_tail_points = _positive_integer(min_tail_points, "min_tail_points")
    lambdas = np.sort(s**2)
    lambdas = lambdas[lambdas > 0]
    if lambdas.size < min_tail_points:
        return float("inf"), float("inf"), 0.0
    candidates = [value for value in np.unique(lambdas) if np.count_nonzero(lambdas >= value) >= min_tail_points]
    best = (float("inf"), float("inf"), 0.0)  # (ks, alpha, lmin)
    for lmin in candidates:
        alpha = _powerlaw_alpha(lambdas, lmin)
        ks = _ks_distance(lambdas[lambdas >= lmin], lmin, alpha)
        if ks < best[0]:
            best = (ks, alpha, float(lmin))
    ks, alpha, lmin = best
    return alpha, ks, lmin


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive exact integer")
    return int(value)


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


def _bootstrap_alpha_interval(
    tail: np.ndarray,
    lambda_min: float,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Conditional percentile interval with the selected ``lambda_min`` held fixed."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        replicate = rng.choice(tail, size=tail.size, replace=True)
        estimates[index] = _powerlaw_alpha(replicate, lambda_min)
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size < max(20, samples // 2):
        return float("nan"), float("nan")
    tail_probability = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail_probability, 1.0 - tail_probability])
    return float(low), float(high)


def _count_spikes(s: np.ndarray, shape: tuple[int, int]) -> int:
    """Informational count of eigenvalues far above an estimated random-matrix bulk.

    The noise floor is estimated from the LOWER half of the spectrum (the bulk), so a few large
    outliers do not inflate it; the Marchenko-Pastur edge for that noise scale sets the cutoff.
    This is only a meaningful "spike" count when the bulk is roughly random; it is reported for
    context only and is not used to classify the matrix.
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


def spectral_health(
    weight: Any,
    *,
    min_tail_points: int = 50,
    max_ks: float = 0.1,
    bootstrap_samples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> SpectralReceipt:
    """Compute a descriptive receipt and abstain from uncalibrated health diagnosis."""
    min_tail_points = _positive_integer(min_tail_points, "min_tail_points")
    bootstrap_samples = _positive_integer(bootstrap_samples, "bootstrap_samples")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an exact integer")
    if isinstance(max_ks, (bool, np.bool_)) or not isinstance(max_ks, Real):
        raise TypeError("max_ks must be a real number")
    max_ks = float(max_ks)
    if not np.isfinite(max_ks) or not 0 < max_ks <= 1:
        raise ValueError("max_ks must lie in (0, 1]")
    confidence = _finite_probability(confidence, "confidence")

    w = _weight_matrix(weight)
    s = singular_values(w)
    alpha, ks, lmin = fit_tail_exponent(s, min_tail_points=min_tail_points)
    lambdas = np.sort(s**2)
    tail = lambdas[lambdas >= lmin] if lmin > 0 else np.empty(0, dtype=np.float64)
    tail_points = int(tail.size)
    if tail_points >= min_tail_points and np.isfinite(alpha):
        ci_low, ci_high = _bootstrap_alpha_interval(
            tail,
            lmin,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=int(seed),
        )
    else:
        ci_low = ci_high = float("nan")
    uncertainty_valid = np.isfinite(ci_low) and np.isfinite(ci_high)
    fit_accepted = bool(tail_points >= min_tail_points and np.isfinite(ks) and ks <= max_ks and uncertainty_valid)
    if tail_points < min_tail_points:
        fit_status = "insufficient-tail"
    elif not np.isfinite(ks) or ks > max_ks:
        fit_status = "poor-fit"
    elif not uncertainty_valid:
        fit_status = "bootstrap-failed"
    else:
        fit_status = "descriptive-fit"
    n_spikes = _count_spikes(s, (w.shape[0], w.shape[1]))
    return SpectralReceipt(
        matrix_shape=(int(w.shape[0]), int(w.shape[1])),
        n_singular_values=int(s.size),
        spectral_norm=float(s[0]) if s.size else 0.0,
        frobenius_norm=float(np.sqrt(np.sum(s**2))),
        stable_rank=stable_rank(s),
        effective_rank=effective_rank(s),
        alpha=float(alpha) if np.isfinite(alpha) else None,
        alpha_ci_low=float(ci_low) if np.isfinite(ci_low) else None,
        alpha_ci_high=float(ci_high) if np.isfinite(ci_high) else None,
        ks_d=float(ks) if np.isfinite(ks) else None,
        lambda_min=float(lmin) if lmin > 0 and np.isfinite(lmin) else None,
        tail_points=tail_points,
        bootstrap_samples=bootstrap_samples,
        tail_fit_status=fit_status,
        tail_fit_accepted=fit_accepted,
        n_spikes=n_spikes,
        verdict=None,
        diagnostic_reason=(
            "No calibrated diagnostic model links these descriptive weight-spectrum statistics "
            "to training quality, generalization, or memorization."
        ),
    )


def model_spectral_report(named_weights: Any) -> dict[str, SpectralReceipt]:
    """Receipt per named 2-D weight matrix. ``named_weights``: mapping name -> array-like."""
    items = named_weights.items() if hasattr(named_weights, "items") else named_weights
    return {str(name): spectral_health(w) for name, w in items}
