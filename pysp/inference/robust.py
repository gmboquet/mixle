"""Sandwich (robust) covariance estimators for M-estimators and regression.

A model's "model-based" covariance (e.g. ``sigma^2 (X'X)^{-1}`` for OLS) is only correct when the
model's variance assumptions hold. When they don't -- heteroscedasticity, within-cluster correlation,
serial correlation -- the point estimate is often still consistent but its *standard errors* are
wrong, usually too small. The sandwich estimator fixes the standard errors without changing the
estimate:

    Cov(theta_hat) = B  M  B'

where the **bread** ``B`` is the inverse sensitivity of the estimating equations (``(X'X)^{-1}`` for
OLS, the inverse negative Hessian / Fisher for a general M-estimator or GLM) and the **meat** ``M`` is
the empirical covariance of the per-observation score contributions, formed to respect the failure you
are guarding against:

  * :func:`ols_robust_covariance` -- heteroscedasticity-consistent ``HC0``--``HC3`` for OLS.
  * :func:`cluster_robust_covariance` -- one-way and multi-way (Cameron--Gelbach--Miller) clustering,
    for correlation within groups (the genus-clustered SE pattern, and any repeated-measures design).
  * :func:`newey_west_covariance` -- heteroscedasticity- and autocorrelation-consistent (HAC) for
    serially correlated series.
  * :func:`sandwich_covariance` -- the generic core: hand it the per-observation scores and the bread
    and it works for *any* M-estimator / GLM.

:func:`robust_standard_errors` reads the standard errors off any of these covariance matrices.
"""

from __future__ import annotations

import numpy as np


def robust_standard_errors(cov: np.ndarray) -> np.ndarray:
    """Standard errors ``sqrt(diag(cov))`` from a covariance matrix."""
    return np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, None))


def sandwich_covariance(
    scores: np.ndarray,
    bread: np.ndarray,
    *,
    clusters: np.ndarray | None = None,
    n_params: int | None = None,
    small_sample: bool = True,
) -> np.ndarray:
    """Generic sandwich covariance ``B M B'`` from per-observation scores and the bread.

    Args:
        scores: ``(n, p)`` per-observation contributions to the estimating equation (for OLS the
            ``i``-th row is ``x_i * e_i``; for a GLM the score of observation ``i``).
        bread: ``(p, p)`` inverse sensitivity ``B`` (e.g. ``(X'X)^{-1}`` for OLS, the inverse negative
            Hessian / inverse Fisher for an M-estimator).
        clusters: optional ``(n,)`` cluster labels; scores are summed within cluster before forming the
            meat (the robust-to-within-cluster-correlation meat). If None, observations are independent.
        n_params: number of estimated parameters for the small-sample correction (defaults to ``p``).
        small_sample: apply the usual finite-sample correction (``n/(n-p)`` for independent data,
            ``G/(G-1) * (n-1)/(n-p)`` for clustered).

    Returns:
        The ``(p, p)`` robust covariance matrix.
    """
    g = np.asarray(scores, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    n, p = g.shape
    b = np.asarray(bread, dtype=float)
    k = p if n_params is None else n_params

    if clusters is None:
        meat = g.T @ g
        corr = (n / (n - k)) if (small_sample and n > k) else 1.0
    else:
        labels = np.asarray(clusters)
        uniq = np.unique(labels)
        agg = np.zeros((len(uniq), p))
        for j, c in enumerate(uniq):
            agg[j] = g[labels == c].sum(axis=0)
        meat = agg.T @ agg
        g_n = len(uniq)
        corr = (g_n / (g_n - 1.0)) * ((n - 1.0) / (n - k)) if (small_sample and g_n > 1 and n > k) else 1.0
    return b @ (corr * meat) @ b.T


def _ols_pieces(x: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, e, (X'X)^{-1})`` with shape checks."""
    X = np.atleast_2d(np.asarray(x, dtype=float))
    e = np.asarray(residuals, dtype=float).ravel()
    if X.shape[0] != e.shape[0]:
        raise ValueError("X and residuals must share the first axis length.")
    xtx_inv = np.linalg.inv(X.T @ X)
    return X, e, xtx_inv


def ols_robust_covariance(x: np.ndarray, residuals: np.ndarray, *, hc: str = "hc1") -> np.ndarray:
    """Heteroscedasticity-consistent (White) covariance for OLS coefficients.

    ``Cov = (X'X)^{-1} ( sum_i w_i x_i x_i' ) (X'X)^{-1}`` where the per-observation weight ``w_i`` is
    the squared residual, optionally adjusted for leverage ``h_ii``:

      * ``hc0``: ``e_i^2`` (White's original; biased down in small samples).
      * ``hc1``: ``e_i^2 * n/(n-p)`` (degrees-of-freedom correction; the common default).
      * ``hc2``: ``e_i^2 / (1 - h_ii)``.
      * ``hc3``: ``e_i^2 / (1 - h_ii)^2`` (best small-sample behaviour; ~jackknife).

    Args:
        x: ``(n, p)`` design matrix (include an intercept column if wanted).
        residuals: ``(n,)`` OLS residuals ``y - X beta_hat``.
        hc: one of ``"hc0"``, ``"hc1"``, ``"hc2"``, ``"hc3"``.

    Returns:
        The ``(p, p)`` robust covariance matrix.
    """
    X, e, xtx_inv = _ols_pieces(x, residuals)
    n, p = X.shape
    hc = hc.lower()
    if hc in ("hc2", "hc3"):
        h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)  # leverage diag(H)
        if hc == "hc2":
            w = e**2 / (1.0 - h)
        else:
            w = e**2 / (1.0 - h) ** 2
    elif hc == "hc0":
        w = e**2
    elif hc == "hc1":
        w = e**2 * (n / (n - p))
    else:
        raise ValueError("hc must be 'hc0', 'hc1', 'hc2', or 'hc3'.")
    meat = (X * w[:, None]).T @ X
    return xtx_inv @ meat @ xtx_inv


def cluster_robust_covariance(
    x: np.ndarray, residuals: np.ndarray, clusters: np.ndarray | list, *, small_sample: bool = True
) -> np.ndarray:
    """Cluster-robust (one-way or multi-way) covariance for OLS coefficients.

    Allows arbitrary correlation *within* clusters while assuming independence *across* them. Pass one
    label array for one-way clustering, or a list/tuple of label arrays for multi-way clustering, which
    uses the Cameron--Gelbach--Miller inclusion--exclusion ``V_A + V_B - V_{A∩B}`` (two-way) and its
    higher-way generalisation.

    Args:
        x: ``(n, p)`` design matrix.
        residuals: ``(n,)`` OLS residuals.
        clusters: ``(n,)`` labels, or a list of such arrays for multi-way clustering.
        small_sample: apply the ``G/(G-1) * (n-1)/(n-p)`` finite-sample correction per term.

    Returns:
        The ``(p, p)`` cluster-robust covariance matrix.
    """
    X, e, xtx_inv = _ols_pieces(x, residuals)
    scores = X * e[:, None]
    if not isinstance(clusters, (list, tuple)):
        return sandwich_covariance(scores, xtx_inv, clusters=clusters, small_sample=small_sample)

    dims = [np.asarray(c) for c in clusters]
    cov = np.zeros((X.shape[1], X.shape[1]))
    # inclusion-exclusion over non-empty subsets of the clustering dimensions
    from itertools import combinations

    d = len(dims)
    for r in range(1, d + 1):
        sign = (-1.0) ** (r + 1)
        for combo in combinations(range(d), r):
            # intersection clustering: a unique label per distinct tuple of the chosen dimensions
            keys = np.stack([dims[i] for i in combo], axis=1)
            _, inter = np.unique(keys, axis=0, return_inverse=True)
            cov += sign * sandwich_covariance(scores, xtx_inv, clusters=inter, small_sample=small_sample)
    return cov


def newey_west_covariance(
    x: np.ndarray, residuals: np.ndarray, *, lags: int | None = None, small_sample: bool = True
) -> np.ndarray:
    """Newey--West HAC covariance for OLS coefficients (serially correlated errors).

    Heteroscedasticity- and autocorrelation-consistent: the meat is the long-run covariance of the
    scores ``s_t = x_t e_t``, estimated with Bartlett (triangular) weights so it stays positive
    semi-definite:

        S = Gamma_0 + sum_{l=1}^{L} (1 - l/(L+1)) (Gamma_l + Gamma_l'),  Gamma_l = sum_t s_t s_{t-l}'.

    Rows of ``x`` (and ``residuals``) are assumed to be in time order.

    Args:
        x: ``(n, p)`` design matrix, rows in time order.
        residuals: ``(n,)`` residuals in time order.
        lags: truncation lag ``L``; defaults to the Newey--West rule ``floor(4 (n/100)^{2/9})``.
        small_sample: apply the ``n/(n-p)`` degrees-of-freedom correction.

    Returns:
        The ``(p, p)`` HAC covariance matrix.
    """
    X, e, xtx_inv = _ols_pieces(x, residuals)
    n, p = X.shape
    s = X * e[:, None]
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))
    meat = s.T @ s
    for ell in range(1, lags + 1):
        w = 1.0 - ell / (lags + 1.0)
        gamma = s[ell:].T @ s[:-ell]
        meat = meat + w * (gamma + gamma.T)
    if small_sample and n > p:
        meat = meat * (n / (n - p))
    return xtx_inv @ meat @ xtx_inv


__all__ = [
    "robust_standard_errors",
    "sandwich_covariance",
    "ols_robust_covariance",
    "cluster_robust_covariance",
    "newey_west_covariance",
]
