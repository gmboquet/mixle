"""P13 (experimental) -- usable-information (V-information) receipts on a model family.

Shannon mutual information ``I(X;Y)`` measures how much information about ``Y`` is *present* in
``X``; it says nothing about whether a given model family can *use* it. **V-information** does:
``I_V(X -> Y)`` is the reduction in the population-optimal predictive log-loss from letting the
family condition on ``X``,

    I_V(X -> Y) = H_V(Y) - H_V(Y | X),

where ``H_V(Y)`` is the population risk of the optimal marginal ``Y`` model in the family and
``H_V(Y | X)`` the corresponding risk of an optimal conditional ``Y | X`` model.
The **usability gap** ``gap = I(X;Y) - I_V`` is then a receipt on the *library*, not a model: it
says how much real dependence the current grammar cannot capture, and it closes when the missing
capability is added.

This module produces finite-sample held-out *estimates* with polynomial-Gaussian conditional
families (degree 1 = linear grammar, degree 2 = adds a quadratic feature), plus the population
closed-form Gaussian ``I(X;Y)`` reference when its correlation is known. A single split is only a
point estimate, not a population optimum. :func:`estimate_v_information` repeats the split and
reports split-assignment uncertainty; it still does not quantify dataset-sampling or model-selection
uncertainty.

Exploratory ``mixle.experimental`` code (P13 card).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class VInformationEstimate:
    """Repeated-holdout V-information estimate with explicitly scoped uncertainty."""

    estimate: float
    standard_error: float
    interval: tuple[float, float]
    confidence: float
    split_estimates: tuple[float, ...]
    degree: int
    holdout: float
    uncertainty_scope: str = "random_holdout_assignment_only"


def _validate_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    if array.size < 3:
        raise ValueError(f"{name} must contain at least three observations")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_degree(degree: Any) -> int:
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    return int(degree)


def _validate_holdout(holdout: Any) -> float:
    if (
        isinstance(holdout, bool)
        or not isinstance(holdout, (int, float, np.integer, np.floating))
        or not math.isfinite(holdout)
        or not 0.0 < holdout < 1.0
    ):
        raise ValueError("holdout must be finite and strictly between 0 and 1")
    return float(holdout)


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _gaussian_nll(y: np.ndarray, mu: np.ndarray, s2: float) -> float:
    """Mean per-point Gaussian negative log-likelihood in nats."""
    if y.shape != mu.shape or y.size == 0:
        raise ValueError("Gaussian NLL inputs must be non-empty arrays with matching shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(mu)):
        raise ValueError("Gaussian NLL inputs must be finite")
    if not math.isfinite(s2) or s2 < 0:
        raise ValueError("Gaussian residual variance must be finite and non-negative")
    s2 = max(float(s2), 1e-12)
    return float(np.mean(0.5 * (_LOG_2PI + np.log(s2) + (y - mu) ** 2 / s2)))


def _fit_poly(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Least-squares polynomial regression; return (coeffs, residual variance) on the train set."""
    if x.size <= degree + 1:
        raise ValueError(f"training split needs more observations than the {degree + 1} polynomial coefficients")
    feats = np.vander(x, N=degree + 1, increasing=True)  # [1, x, x^2, ...]
    coeffs, *_ = np.linalg.lstsq(feats, y, rcond=None)
    resid = y - feats @ coeffs
    return coeffs, float(np.mean(resid**2))


def _poly_predict(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return np.vander(x, N=len(coeffs), increasing=True) @ coeffs


def _split(x: np.ndarray, y: np.ndarray, holdout: float, seed: int) -> tuple[Any, Any, Any, Any]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(round(len(x) * (1.0 - holdout)))
    if cut == 0 or cut == len(x):
        raise ValueError("holdout produces an empty training or test split")
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]


def marginal_nll(y: np.ndarray, *, holdout: float = 0.3, seed: int = 0) -> float:
    """One-split held-out NLL estimate for a fitted marginal Gaussian ``Y`` model."""
    y = _validate_vector(y, "y")
    holdout = _validate_holdout(holdout)
    seed = _validate_seed(seed)
    _, y_tr, _, y_te = _split(np.zeros_like(y), y, holdout, seed)
    if y_tr.size < 2:
        raise ValueError("marginal Gaussian training split needs at least two observations")
    mu, s2 = float(np.mean(y_tr)), float(np.var(y_tr))
    return _gaussian_nll(y_te, np.full_like(y_te, mu), s2)


def conditional_nll(x: np.ndarray, y: np.ndarray, *, degree: int, holdout: float = 0.3, seed: int = 0) -> float:
    """One-split held-out NLL estimate for a fitted polynomial-Gaussian ``Y | X`` model."""
    x = _validate_vector(x, "x")
    y = _validate_vector(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    degree = _validate_degree(degree)
    holdout = _validate_holdout(holdout)
    seed = _validate_seed(seed)
    x_tr, y_tr, x_te, y_te = _split(x, y, holdout, seed)
    coeffs, s2 = _fit_poly(x_tr, y_tr, degree)
    return _gaussian_nll(y_te, _poly_predict(x_te, coeffs), s2)


def v_information(x: Any, y: Any, *, degree: int = 1, holdout: float = 0.3, seed: int = 0) -> float:
    """Return one held-out-split point estimate of ``I_V(X -> Y)`` in nats.

    Marginal and conditional models use the same train/test indices. The result is the fitted family's realized
    reduction in predictive log-loss on that test split. It is not the population-optimal V-information and
    carries no uncertainty estimate; use :func:`estimate_v_information` when split uncertainty matters.
    """
    x = _validate_vector(x, "x")
    y = _validate_vector(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    degree = _validate_degree(degree)
    holdout = _validate_holdout(holdout)
    seed = _validate_seed(seed)
    h_y = marginal_nll(y, holdout=holdout, seed=seed)
    h_y_given_x = conditional_nll(x, y, degree=degree, holdout=holdout, seed=seed)
    return float(h_y - h_y_given_x)


def estimate_v_information(
    x: Any,
    y: Any,
    *,
    degree: int = 1,
    holdout: float = 0.3,
    n_splits: int = 10,
    confidence: float = 0.95,
    seed: int = 0,
) -> VInformationEstimate:
    """Estimate V-information over repeated splits and report split-assignment uncertainty.

    The normal interval describes variability from random train/test assignment conditional on this fixed
    dataset. It is not a confidence interval for population V-information and does not include model-selection
    uncertainty.
    """
    x = _validate_vector(x, "x")
    y = _validate_vector(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    degree = _validate_degree(degree)
    holdout = _validate_holdout(holdout)
    seed = _validate_seed(seed)
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float, np.integer, np.floating))
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be finite and strictly between 0 and 1")

    split_estimates = tuple(
        v_information(x, y, degree=degree, holdout=holdout, seed=seed + split) for split in range(int(n_splits))
    )
    estimate = float(np.mean(split_estimates))
    standard_error = float(np.std(split_estimates, ddof=1) / np.sqrt(len(split_estimates)))
    critical_value = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    half_width = critical_value * standard_error
    return VInformationEstimate(
        estimate=estimate,
        standard_error=standard_error,
        interval=(estimate - half_width, estimate + half_width),
        confidence=float(confidence),
        split_estimates=split_estimates,
        degree=degree,
        holdout=holdout,
    )


def gaussian_mutual_information(rho: float) -> float:
    """Population ``I(X;Y)`` for a valid bivariate-Gaussian correlation ``rho`` (nats)."""
    if (
        isinstance(rho, bool)
        or not isinstance(rho, (int, float, np.integer, np.floating))
        or not math.isfinite(rho)
        or abs(rho) > 1.0
    ):
        raise ValueError("rho must be a finite correlation in [-1, 1]")
    rho = float(rho)
    if abs(rho) == 1.0:
        return math.inf
    return float(-0.5 * np.log(1.0 - rho**2))


def usability_gap(true_mi: float, i_v: float) -> float:
    """Difference between a population MI reference and a finite-sample V-information estimate."""
    if (
        isinstance(true_mi, bool)
        or not isinstance(true_mi, (int, float, np.integer, np.floating))
        or math.isnan(true_mi)
        or true_mi < 0
    ):
        raise ValueError("true_mi must be a non-negative numeric value")
    if isinstance(i_v, bool) or not isinstance(i_v, (int, float, np.integer, np.floating)) or not math.isfinite(i_v):
        raise ValueError("i_v must be a finite numeric estimate")
    return float(true_mi - i_v)
