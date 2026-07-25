"""``forecast_price`` -- commodity-price / cost forecasting with horizon-matched conformal intervals.

Wraps the generic HMM front door (:func:`mixle.inference.forecast.forecast`) with a
recalibration pass so each reported ``[lo, hi]`` lead uses held-out residuals from that same lead::

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
eligible for marginal coverage under the usual held-out exchangeability assumption. Time-series
dependence or distribution shift can violate that assumption, so the result records the calibration
count and assumptions rather than claiming unconditional real-world coverage.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, NamedTuple

import numpy as np

from mixle.inference.conformal import split_conformal
from mixle.inference.forecast import forecast


class PriceForecast(NamedTuple):
    """A price/cost forecast with a horizon-matched interval calibration receipt."""

    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    paths: np.ndarray
    level: float
    calibration_count: int = 0
    interval_method: str = "horizon_matched_split_conformal"
    coverage_assumptions: tuple[str, ...] = ("held_out_exchangeability", "stable_data_generating_process")


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
    if isinstance(horizon, bool) or not isinstance(horizon, Integral):
        raise TypeError("horizon must be a positive integer")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(level, bool) or not isinstance(level, Real) or not math.isfinite(float(level)):
        raise TypeError("level must be a finite real number in (0, 1)")
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if isinstance(cal_frac, bool) or not isinstance(cal_frac, Real) or not math.isfinite(float(cal_frac)):
        raise TypeError("cal_frac must be a finite real number in (0, 1)")
    cal_frac = float(cal_frac)
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f"cal_frac must be in (0.0, 1.0), got {cal_frac!r}.")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    hist = np.asarray(list(history), dtype=np.float64)
    if hist.ndim != 1 or hist.size == 0 or not np.all(np.isfinite(hist)):
        raise ValueError("history must be a non-empty finite one-dimensional series")
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
        prediction = np.asarray(cf.mean, dtype=np.float64)
        if prediction.shape != (horizon,) or not np.all(np.isfinite(prediction)):
            raise RuntimeError("calibration forecast returned an invalid horizon path")
        cal_pred.append(prediction)
        cal_y.append(hist[origin : origin + horizon])
    cal_pred = np.asarray(cal_pred, dtype=np.float64)
    cal_y = np.asarray(cal_y, dtype=np.float64)
    if cal_pred.ndim != 2 or cal_pred.shape[0] == 0 or cal_pred.shape != cal_y.shape:
        raise ValueError("history does not provide a valid horizon-matched calibration set")

    # The forecast actually being delivered, drawn from the full history.
    f = forecast(model, hist.tolist(), horizon, level=level, keep_samples=True, seed=seed)
    test_pred = np.asarray(f.mean, dtype=np.float64)

    bounds = [
        split_conformal(
            cal_pred[:, lead],
            cal_y[:, lead],
            test_pred[lead : lead + 1],
            alpha=1.0 - level,
            side="two-sided",
        )
        for lead in range(horizon)
    ]
    lo = np.asarray([bound[0][0] for bound in bounds], dtype=np.float64)
    hi = np.asarray([bound[1][0] for bound in bounds], dtype=np.float64)

    paths = f.samples if f.samples is not None else np.asarray([])
    return PriceForecast(
        mean=test_pred,
        lo=lo,
        hi=hi,
        paths=paths,
        level=level,
        calibration_count=int(cal_pred.shape[0]),
    )


__all__ = ["PriceForecast", "forecast_price"]
