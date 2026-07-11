"""P13 (experimental) -- usable-information (V-information) receipts on a model family.

Shannon mutual information ``I(X;Y)`` measures how much information about ``Y`` is *present* in
``X``; it says nothing about whether a given model family can *use* it. **V-information** does:
``I_V(X -> Y)`` is the reduction in the best achievable held-out predictive log-loss from letting
the family condition on ``X``,

    I_V(X -> Y) = H_V(Y) - H_V(Y | X),

where ``H_V(Y)`` is the best held-out negative log-likelihood of a marginal ``Y`` model in the
family and ``H_V(Y | X)`` the best held-out NLL of a conditional ``Y | X`` model in the family.
The **usability gap** ``gap = I(X;Y) - I_V`` is then a receipt on the *library*, not a model: it
says how much real dependence the current grammar cannot capture, and it closes when the missing
capability is added.

This module estimates ``I_V`` with polynomial-Gaussian conditional families (degree 1 = linear
grammar, degree 2 = adds a quadratic feature), plus the closed-form Gaussian ``I(X;Y)`` reference,
so the gap can be measured exactly on a synthetic task whose generative law sits just outside the
low-degree grammar.

Exploratory ``mixle.experimental`` code (P13 card).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


def _gaussian_nll(y: np.ndarray, mu: np.ndarray, s2: float) -> float:
    """Mean per-point Gaussian negative log-likelihood in nats."""
    s2 = max(float(s2), 1e-12)
    return float(np.mean(0.5 * (_LOG_2PI + np.log(s2) + (y - mu) ** 2 / s2)))


def _fit_poly(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Least-squares polynomial regression; return (coeffs, residual variance) on the train set."""
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
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]


def marginal_nll(y: np.ndarray, *, holdout: float = 0.3, seed: int = 0) -> float:
    """Best held-out NLL of a marginal Gaussian ``Y`` model -- ``H_V(Y)``."""
    _, y_tr, _, y_te = _split(np.zeros_like(y), y, holdout, seed)
    mu, s2 = float(np.mean(y_tr)), float(np.var(y_tr))
    return _gaussian_nll(y_te, np.full_like(y_te, mu), s2)


def conditional_nll(x: np.ndarray, y: np.ndarray, *, degree: int, holdout: float = 0.3, seed: int = 0) -> float:
    """Best held-out NLL of a degree-``degree`` polynomial-Gaussian ``Y | X`` model -- ``H_V(Y|X)``."""
    x_tr, y_tr, x_te, y_te = _split(x, y, holdout, seed)
    coeffs, s2 = _fit_poly(x_tr, y_tr, degree)
    return _gaussian_nll(y_te, _poly_predict(x_te, coeffs), s2)


def v_information(x: Any, y: Any, *, degree: int = 1, holdout: float = 0.3, seed: int = 0) -> float:
    """Estimate ``I_V(X -> Y)`` for the degree-``degree`` polynomial-Gaussian family (nats).

    ``I_V = H_V(Y) - H_V(Y | X)`` on the SAME held-out split, so it is the family's realized
    reduction in predictive log-loss from conditioning on ``X``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h_y = marginal_nll(y, holdout=holdout, seed=seed)
    h_y_given_x = conditional_nll(x, y, degree=degree, holdout=holdout, seed=seed)
    return float(h_y - h_y_given_x)


def gaussian_mutual_information(rho: float) -> float:
    """Closed-form ``I(X;Y)`` for a bivariate Gaussian with correlation ``rho`` (nats)."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(-0.5 * np.log(1.0 - rho**2))


def usability_gap(true_mi: float, i_v: float) -> float:
    """``gap = I(X;Y) - I_V`` -- the information the family cannot use (a receipt on the library)."""
    return float(true_mi - i_v)
