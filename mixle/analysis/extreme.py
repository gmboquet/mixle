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
        n_dropped_nonpositive: count of finite, non-positive (``<= 0``) input values excluded because
            they are not actually above the threshold (a receipt for :func:`gpd_fit`'s filtering; NaN
            and non-finite input are rejected outright rather than silently dropped, so they never
            contribute to this count).
    """

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    n_total: int
    method: str
    n_dropped_nonpositive: int = 0

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


def _validate_gpd_params(shape: float, scale: float, z: np.ndarray, method: str) -> None:
    """Enforce that ``(shape, scale)`` is a legitimate GPD fit for the exceedances ``z``.

    Checks finiteness, ``scale > 0``, and the GPD support constraint ``1 + shape*z/scale > 0`` against
    every exceedance actually used to fit it. PWM (probability-weighted moments) has no built-in
    safeguard against this: it is a closed-form moment match, not a constrained optimization, so on
    some samples it returns a ``(shape, scale)`` whose implied endpoint is below the sample's own
    maximum -- a self-contradictory fit asserting its own fitting data was impossible.
    """
    if not (np.isfinite(shape) and np.isfinite(scale)):
        raise ValueError(f"{method} fit produced a non-finite parameter (shape={shape}, scale={scale}).")
    if scale <= 0:
        raise ValueError(f"{method} fit produced a non-positive scale ({scale}); not a valid GPD.")
    if np.any(1.0 + shape * z / scale <= 0):
        raise ValueError(
            f"{method} fit violates the GPD support constraint (1 + shape*z/scale > 0) for some "
            "exceedances; degenerate data for this method."
        )


def gpd_fit(
    exceedances: np.ndarray, *, threshold: float = 0.0, method: str = "mle", n_total: int | None = None
) -> GPDFit:
    """Fit a Generalized Pareto Distribution to threshold exceedances.

    Args:
        exceedances: the *excesses* ``x - u`` for observations above the threshold (all positive).
            NaN or non-finite entries are invalid data, not sub-threshold observations, and raise.
            Finite entries ``<= 0`` are legitimately not above the threshold, so they are dropped and
            the count is receipted on the returned :class:`GPDFit` (``n_dropped_nonpositive``) rather
            than silently vanishing. If you have raw data, use :func:`peaks_over_threshold` instead.
        threshold: the threshold ``u`` (stored for return-level computation).
        method: ``"mle"`` (Nelder-Mead on the GPD likelihood) or ``"pwm"`` (probability-weighted
            moments, closed form, robust for ``xi < 0.5``).
        n_total: full sample size before thresholding (defaults to the number of valid, positive
            exceedances; if given explicitly it must be at least that many).

    Returns:
        A :class:`GPDFit`.

    Raises:
        ValueError: ``exceedances`` contains NaN/non-finite values; fewer than two positive
            exceedances remain after filtering; ``n_total`` is smaller than the exceedance count;
            ``method`` is not ``"mle"``/``"pwm"``; the MLE optimizer fails to converge; or the fitted
            ``(shape, scale)`` is not a valid GPD (see :func:`_validate_gpd_params`).
    """
    z_raw = np.asarray(exceedances, dtype=float).ravel()
    if not np.all(np.isfinite(z_raw)):
        raise ValueError("exceedances must be finite; NaN/inf are invalid data, not sub-threshold observations.")
    z = z_raw[z_raw > 0]
    n = z.shape[0]
    n_dropped_nonpositive = z_raw.shape[0] - n
    if n < 2:
        raise ValueError(
            f"need at least two positive exceedances (got {n} after dropping {n_dropped_nonpositive} "
            f"non-positive of {z_raw.shape[0]} total)."
        )
    if n_total is not None and n_total < n:
        raise ValueError(f"n_total ({n_total}) cannot be smaller than the number of exceedances ({n}).")
    if method == "pwm":
        zs = np.sort(z)
        b0 = float(zs.mean())
        p = (np.arange(1, n + 1) - 0.35) / n
        b1 = float(np.mean((1.0 - p) * zs))
        denom = b0 - 2.0 * b1
        if denom == 0.0:
            raise ValueError("PWM fit is degenerate for this sample (b0 == 2*b1); cannot estimate GPD parameters.")
        scale = 2.0 * b0 * b1 / denom
        shape = 2.0 - b0 / denom
    elif method == "mle":
        beta0 = z.mean()
        res = optimize.minimize(_gpd_nll, np.array([0.1, beta0]), args=(z,), method="Nelder-Mead")
        if not res.success:
            raise ValueError(f"GPD MLE fit did not converge: {res.message}")
        shape, scale = float(res.x[0]), float(res.x[1])
    else:
        raise ValueError("method must be 'mle' or 'pwm'.")
    _validate_gpd_params(shape, scale, z, method)
    return GPDFit(
        shape=shape,
        scale=scale,
        threshold=threshold,
        n_exceedances=n,
        n_total=n_total if n_total is not None else n,
        method=method,
        n_dropped_nonpositive=n_dropped_nonpositive,
    )


def peaks_over_threshold(data: np.ndarray, threshold: float, *, method: str = "mle") -> GPDFit:
    """Peaks-over-threshold: select exceedances above ``threshold`` and fit a GPD to the excesses."""
    x = np.asarray(data, dtype=float).ravel()
    exc = x[x > threshold] - threshold
    return gpd_fit(exc, threshold=threshold, method=method, n_total=x.shape[0])


def return_level(fit: GPDFit, period: float) -> float:
    """POT return level: the level exceeded on average once per ``period`` observations.

    ``x_m = u + (beta/xi) [ (m zeta_u)^xi - 1 ]`` with ``zeta_u = n_exceed/n_total`` the exceedance
    rate (``period = m``). For ``xi = 0`` it reduces to ``u + beta log(m zeta_u)``.

    Raises:
        ValueError: ``period`` is not positive (a non-positive return period is not meaningful --
            ``m = 0`` sends the level to +/-infinity and negative ``m`` is undefined for non-integer
            ``xi``).
    """
    if not period > 0:
        raise ValueError(f"period must be positive, got {period}.")
    zeta = fit.n_exceedances / fit.n_total
    m = period * zeta
    if abs(fit.shape) < 1e-8:
        return float(fit.threshold + fit.scale * np.log(m))
    return float(fit.threshold + (fit.scale / fit.shape) * (m**fit.shape - 1.0))


def _top_order_stats(x_sorted: np.ndarray, k: int, *, min_k: int = 1) -> tuple[np.ndarray, float]:
    """Validate ``k`` against ``x_sorted`` (ascending) and return the top-``k`` slice with ``X_(n-k)``.

    Shared range/finiteness check for the order-statistic tail estimators. ``numpy.sort`` places NaN at
    the very end of the array regardless of magnitude (+/-inf sort normally, by size, ahead of any
    NaN) -- so a NaN anywhere in ``data`` lands inside ``top`` no matter how small it "should" be. A
    boundary check of the form ``x[n-k-1] <= 0`` alone does not catch this: a single NaN elsewhere in
    ``data`` can silently poison ``top`` while the finite boundary value still passes. So this checks
    finiteness of both explicitly.

    Raises:
        ValueError: ``k`` is outside ``[min_k, n-1]``, or the boundary value / top-``k`` slice contains
            a non-finite entry.
    """
    n = x_sorted.shape[0]
    if not min_k <= k < n:
        raise ValueError(f"k must be in [{min_k}, {n - 1}].")
    top = x_sorted[n - k :]
    xnk = x_sorted[n - k - 1]
    if not np.isfinite(xnk) or not np.all(np.isfinite(top)):
        raise ValueError(
            "upper order statistics must be finite (NaN/inf sorts to the top and would silently corrupt the estimate)."
        )
    return top, xnk


def hill_estimator(data: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index ``xi = 1/alpha`` from the top ``k`` order statistics.

    ``xi_hat = (1/k) sum_{i=1}^{k} log(X_(n-i+1) / X_(n-k))`` -- consistent for heavy (Pareto-type,
    ``xi > 0``) tails. For a Pareto tail with exponent ``alpha`` this estimates ``1/alpha``.
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    top, xnk = _top_order_stats(x, k)
    if xnk <= 0:
        raise ValueError("Hill estimator needs positive upper order statistics.")
    return float(np.mean(np.log(top) - np.log(xnk)))


def moment_estimator(data: np.ndarray, k: int) -> float:
    """Dekkers--Einmahl--de Haan moment estimator of the extreme-value index (any tail sign).

    Generalises Hill to ``xi`` of either sign by combining the first two log-moments of the top ``k``
    exceedances; works for heavy, light, and bounded (``xi < 0``) tails. Needs ``k >= 2``: with a single
    log-spacing the second moment is identically the square of the first (zero variance), which sends
    the estimator's denominator to zero.
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    top, xnk = _top_order_stats(x, k, min_k=2)
    if xnk <= 0:
        raise ValueError("moment estimator needs positive upper order statistics.")
    logs = np.log(top) - np.log(xnk)
    m1 = float(np.mean(logs))
    m2 = float(np.mean(logs**2))
    denom = 1.0 - m1**2 / m2 if m2 > 0 else 0.0
    if denom <= 0:
        raise ValueError(
            "moment estimator is undefined for this k: the top-k log-spacings have zero variance "
            "(tied upper order statistics); choose a different k."
        )
    return float(m1 + 1.0 - 0.5 / denom)


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
        method: ``"gpd"`` -- GPD-MLE endpoint. No other method is implemented.

    Returns:
        The estimated right endpoint (``inf`` if unbounded).

    Raises:
        ValueError: ``method`` is not ``"gpd"``, ``k`` is out of range, or the upper order statistics
            used are non-finite.
    """
    if method != "gpd":
        raise ValueError(f"method must be 'gpd' (no other endpoint method is implemented), got {method!r}.")
    x = np.sort(np.asarray(data, dtype=float).ravel())
    top, u = _top_order_stats(x, k)
    fit = gpd_fit(top - u, threshold=u, method="mle", n_total=x.shape[0])
    return fit.endpoint


def record_times(data: np.ndarray) -> np.ndarray:
    """Indices at which a new running maximum (upper record) occurs, including the first observation.

    An empty ``data`` has no observations and so, vacuously, no records: returns an empty index array
    rather than raising.
    """
    x = np.asarray(data, dtype=float).ravel()
    if x.shape[0] == 0:
        return np.nonzero(np.array([], dtype=bool))[0]
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
