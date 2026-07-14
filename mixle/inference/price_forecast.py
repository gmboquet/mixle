"""``forecast_price`` -- commodity-price / cost forecasting with conformally-calibrated intervals.

Wraps the generic HMM front door (:func:`mixle.inference.forecast.forecast`) with a
recalibration pass so the reported ``[lo, hi]`` band achieves nominal coverage on *real* held-out
data, not just the model's own (possibly misspecified) predictive shape::

    pf = forecast_price(model, price_history, horizon=12, level=0.9)
    pf.mean, pf.lo, pf.hi   # (horizon,) point forecast + calibrated band
    pf.paths                # (horizon, n) Monte-Carlo scenario draws for downstream DCF (J2)

The forecast itself is exact where the model is exact (HMM state marginals ``p_T A^h``) and
Monte-Carlo only where it has to be (emission quantiles for arbitrary, possibly skewed or
multimodal, price/cost emission families) -- see :mod:`mixle.inference.forecast`. What this module
adds is the recalibration: reserve the most recent ``cal_frac`` of ``history`` and, within that
window, run a rolling-origin backtest of the SAME ``horizon``-step-ahead point forecast the caller
is about to receive, scoring each origin's forecast against what actually happened. That gives a
sample of real ``horizon``-step-ahead residuals at the depth that matters (rather than mixing in
easier short-horizon or harder long-horizon errors), which recalibrates the requested band via
split conformal (:func:`mixle.inference.conformal.split_conformal`). This is what makes the band
honest on real commodity series, where the emission family is only ever an approximation of the
true price process.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from mixle.inference.conformal import split_conformal
from mixle.inference.forecast import forecast


class PriceForecast(NamedTuple):
    """A calibrated price/cost forecast: point path, band, and the raw scenario draws."""

    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    paths: np.ndarray
    level: float


def forecast_price(
    model: Any,
    history: Any,
    horizon: int,
    *,
    level: float = 0.9,
    cal_frac: float = 0.3,
    seed: int = 0,
) -> PriceForecast:
    """Forecast ``horizon`` steps of a price/cost series with a conformally-calibrated band.

    Args:
        model: a fitted ``HiddenMarkovModelDistribution`` over the price/cost series (any scalar
            emission family with a sampler -- see :func:`mixle.inference.forecast.forecast`).
        history: the observed series to condition on (one sequence, oldest first).
        horizon: number of future steps to forecast.
        level: central-interval mass (``0.9`` -> the calibrated 5%..95% band).
        cal_frac: the fraction of ``history`` (most recent) reserved for calibration. Within that
            reserved window, a rolling-origin backtest of the same ``horizon``-step-ahead forecast
            is scored against the real outcomes to build the calibration residuals.
        seed: reproducibility for the calibration and requested-horizon Monte Carlo draws.

    Returns:
        A :class:`PriceForecast` with the calibrated ``(lo, hi)`` band, the point forecast
        ``mean``, the raw per-step predictive draws ``paths`` (for Monte-Carlo DCF scenario
        analysis downstream), and ``level``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f"cal_frac must be in (0.0, 1.0), got {cal_frac!r}.")

    hist = np.asarray(list(history), dtype=np.float64).ravel()
    n_hist = hist.shape[0]
    n_cal_window = max(int(round(n_hist * cal_frac)), horizon + 1)
    if n_cal_window >= n_hist:
        raise ValueError("history is too short to hold out a calibration window at this horizon")
    cal_start = n_hist - n_cal_window

    # Recalibration set: a rolling-origin backtest, within the reserved window, of the SAME
    # horizon-step-ahead point forecast the caller is about to receive -- so the residuals are at
    # the depth that matters, not a mix of easier (short-horizon) and harder (long-horizon) errors.
    cal_pred = []
    cal_y = []
    for origin in range(cal_start, n_hist - horizon):
        cf = forecast(model, hist[:origin].tolist(), horizon=horizon, level=level, seed=seed, keep_samples=False)
        cal_pred.append(float(np.asarray(cf.mean)[-1]))
        cal_y.append(float(hist[origin + horizon - 1]))
    cal_pred = np.asarray(cal_pred, dtype=np.float64)
    cal_y = np.asarray(cal_y, dtype=np.float64)

    # The forecast actually being delivered, drawn from the full history.
    f = forecast(model, hist.tolist(), horizon, level=level, keep_samples=True, seed=seed)
    test_pred = np.asarray(f.mean, dtype=np.float64)

    lo, hi = split_conformal(cal_pred, cal_y, test_pred, alpha=1.0 - level, side="two-sided")

    paths = f.samples if f.samples is not None else np.asarray([])
    return PriceForecast(mean=test_pred, lo=lo, hi=hi, paths=paths, level=level)


__all__ = ["PriceForecast", "forecast_price"]
