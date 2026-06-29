"""Extreme-value analysis: tails, threshold exceedances, and support endpoints.

The mean of a sample tells you nothing reliable about its rare extremes -- floods, market crashes,
record temperatures live in the *tail*, where ordinary fitting has almost no data. Extreme-value theory
gives the limiting shapes that govern tails and provides estimators built only from the largest order
statistics:

  * :func:`gpd_fit` / :func:`peaks_over_threshold` -- the Generalized Pareto Distribution for threshold
    exceedances (the POT method), by maximum likelihood or probability-weighted moments, with
    :func:`return_level` for "the once-in-``m``-observations level".
  * :func:`hill_estimator` / :func:`moment_estimator` -- tail-index estimators from the top-``k`` order
    statistics (Hill for heavy tails; the Dekkers--Einmahl--de Haan moment estimator for any tail).
  * :func:`mean_residual_life` -- the mean-excess plot for choosing the POT threshold (linear in the
    threshold once the GPD regime is reached).
  * :func:`endpoint_estimator` -- the finite right endpoint of a bounded support (Hall-type / GPD),
    generic to frontier analysis, reliability limits, and image edges.
  * :func:`record_times` / :func:`n_records` -- running-maximum records and their count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class GPDFit:
    """Fitted Generalized Pareto Distribution for exceedances over a threshold.

    Attributes:
        shape: tail index ``xi`` (``> 0`` heavy/Pareto tail, ``= 0`` exponential, ``< 0`` bounded).
        scale: scale ``beta`` (> 0).
        threshold: the threshold ``u`` the exceedances were measured over.
        n_exceedances / n_total: exceedance and full-sample sizes (for return levels).
        method: ``"mle"`` or ``"pwm"``.
    """

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    n_total: int
    method: str

    @property
    def endpoint(self) -> float:
        """Right endpoint of the support (finite when ``shape < 0``, else ``inf``)."""
        return self.threshold - self.scale / self.shape if self.shape < 0 else float("inf")


def _gpd_nll(params: np.ndarray, z: np.ndarray) -> float:
    xi, beta = params
    if beta <= 0:
        return 1e10
    if abs(xi) < 1e-8:
        return float(z.shape[0] * np.log(beta) + np.sum(z) / beta)
    t = 1.0 + xi * z / beta
    if np.any(t <= 0):
        return 1e10
    return float(z.shape[0] * np.log(beta) + (1.0 + 1.0 / xi) * np.sum(np.log(t)))


def gpd_fit(
    exceedances: np.ndarray, *, threshold: float = 0.0, method: str = "mle", n_total: int | None = None
) -> GPDFit:
    """Fit a Generalized Pareto Distribution to threshold exceedances.

    Args:
        exceedances: the *excesses* ``x - u`` for observations above the threshold (all positive). If you
            have raw data, use :func:`peaks_over_threshold` instead.
        threshold: the threshold ``u`` (stored for return-level computation).
        method: ``"mle"`` (Newton/Nelder-Mead on the GPD likelihood) or ``"pwm"`` (probability-weighted
            moments, closed form, robust for ``xi < 0.5``).
        n_total: full sample size before thresholding (defaults to the number of exceedances).

    Returns:
        A :class:`GPDFit`.
    """
    z = np.asarray(exceedances, dtype=float).ravel()
    z = z[z > 0]
    n = z.shape[0]
    if n < 2:
        raise ValueError("need at least two positive exceedances.")
    if method == "pwm":
        zs = np.sort(z)
        b0 = zs.mean()
        p = (np.arange(1, n + 1) - 0.35) / n
        b1 = float(np.mean((1.0 - p) * zs))
        scale = 2.0 * b0 * b1 / (b0 - 2.0 * b1)
        shape = 2.0 - b0 / (b0 - 2.0 * b1)
    elif method == "mle":
        beta0 = z.mean()
        res = optimize.minimize(_gpd_nll, np.array([0.1, beta0]), args=(z,), method="Nelder-Mead")
        shape, scale = float(res.x[0]), float(res.x[1])
    else:
        raise ValueError("method must be 'mle' or 'pwm'.")
    return GPDFit(shape, scale, threshold, n, n_total or n, method)


def peaks_over_threshold(data: np.ndarray, threshold: float, *, method: str = "mle") -> GPDFit:
    """Peaks-over-threshold: select exceedances above ``threshold`` and fit a GPD to the excesses."""
    x = np.asarray(data, dtype=float).ravel()
    exc = x[x > threshold] - threshold
    return gpd_fit(exc, threshold=threshold, method=method, n_total=x.shape[0])


def return_level(fit: GPDFit, period: float) -> float:
    """POT return level: the level exceeded on average once per ``period`` observations.

    ``x_m = u + (beta/xi) [ (m zeta_u)^xi - 1 ]`` with ``zeta_u = n_exceed/n_total`` the exceedance
    rate (``period = m``). For ``xi = 0`` it reduces to ``u + beta log(m zeta_u)``.
    """
    zeta = fit.n_exceedances / fit.n_total
    m = period * zeta
    if abs(fit.shape) < 1e-8:
        return float(fit.threshold + fit.scale * np.log(m))
    return float(fit.threshold + (fit.scale / fit.shape) * (m**fit.shape - 1.0))


def hill_estimator(data: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index ``xi = 1/alpha`` from the top ``k`` order statistics.

    ``xi_hat = (1/k) sum_{i=1}^{k} log(X_(n-i+1) / X_(n-k))`` -- consistent for heavy (Pareto-type,
    ``xi > 0``) tails. For a Pareto tail with exponent ``alpha`` this estimates ``1/alpha``.
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    n = x.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be in [1, n-1].")
    if x[n - k - 1] <= 0:
        raise ValueError("Hill estimator needs positive upper order statistics.")
    top = x[n - k :]
    return float(np.mean(np.log(top) - np.log(x[n - k - 1])))


def moment_estimator(data: np.ndarray, k: int) -> float:
    """Dekkers--Einmahl--de Haan moment estimator of the extreme-value index (any tail sign).

    Generalises Hill to ``xi`` of either sign by combining the first two log-moments of the top ``k``
    exceedances; works for heavy, light, and bounded (``xi < 0``) tails.
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    n = x.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be in [1, n-1].")
    xnk = x[n - k - 1]
    if xnk <= 0:
        raise ValueError("moment estimator needs positive upper order statistics.")
    logs = np.log(x[n - k :]) - np.log(xnk)
    m1 = float(np.mean(logs))
    m2 = float(np.mean(logs**2))
    return float(m1 + 1.0 - 0.5 / (1.0 - m1**2 / m2))


def mean_residual_life(data: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Mean-excess (mean-residual-life) function for POT threshold selection.

    ``e(u) = mean(X - u | X > u)``. Over a range where the GPD fits, ``e(u)`` is approximately linear
    in ``u`` (slope ``xi/(1-xi)``); the lowest threshold from which the plot is linear is the choice.

    Returns:
        ``{'threshold', 'mean_excess', 'n_exceed'}``.
    """
    x = np.asarray(data, dtype=float).ravel()
    thresholds = np.asarray(thresholds, dtype=float)
    me = np.empty(thresholds.shape[0])
    ne = np.empty(thresholds.shape[0], dtype=int)
    for i, u in enumerate(thresholds):
        exc = x[x > u] - u
        ne[i] = exc.shape[0]
        me[i] = float(exc.mean()) if exc.size else np.nan
    return {"threshold": thresholds, "mean_excess": me, "n_exceed": ne}


def endpoint_estimator(data: np.ndarray, k: int, *, method: str = "gpd") -> float:
    """Estimate the finite right endpoint of a bounded support (frontier / boundary estimation).

    Fits a GPD to the top ``k`` exceedances over ``X_(n-k)``; when the tail index ``xi`` is negative the
    support is bounded and the endpoint is ``x+ = X_(n-k) - beta/xi`` (which, by the GPD support
    constraint, necessarily exceeds the observed maximum). Generic to econometric frontier analysis,
    reliability limits, and image-edge localisation. Returns ``inf`` if the estimated tail is unbounded
    (``xi >= 0``).

    Args:
        data: the sample.
        k: number of upper order statistics (exceedances) used.
        method: ``"gpd"`` -- GPD-MLE endpoint.

    Returns:
        The estimated right endpoint (``inf`` if unbounded).
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    n = x.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be in [1, n-1].")
    u = x[n - k - 1]
    fit = gpd_fit(x[n - k :] - u, threshold=u, method="mle", n_total=n)
    return fit.endpoint


def record_times(data: np.ndarray) -> np.ndarray:
    """Indices at which a new running maximum (upper record) occurs, including the first observation."""
    x = np.asarray(data, dtype=float).ravel()
    running = np.maximum.accumulate(x)
    is_record = np.empty(x.shape[0], dtype=bool)
    is_record[0] = True
    is_record[1:] = x[1:] > running[:-1]
    return np.nonzero(is_record)[0]


def n_records(data: np.ndarray) -> int:
    """Number of upper records. For an i.i.d. sequence of length ``n`` the expectation is ``H_n``."""
    return int(record_times(data).shape[0])


__all__ = [
    "GPDFit",
    "gpd_fit",
    "peaks_over_threshold",
    "return_level",
    "hill_estimator",
    "moment_estimator",
    "mean_residual_life",
    "endpoint_estimator",
    "record_times",
    "n_records",
]
