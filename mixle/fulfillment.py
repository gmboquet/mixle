"""mixle.fulfillment — demand forecasting + supply routing/dispatch (worklist H6).

Two steps, chained: forecast tomorrow's demand honestly, then route today's supply to meet it at
minimum cost. Renamed from ``mixle.distribution``: "distribution" collided with :mod:`mixle.dist`
(the object-namespace alias for :mod:`mixle.stats`) -- same word, unrelated meanings (probability
distribution vs. supply-chain distribution/routing).

:func:`forecast_demand` fits a small regime-switching (Gaussian-emission HMM) model directly from a
raw history series and calls the existing :func:`mixle.inference.forecast.forecast` front door for
the exact state-marginal / Monte-Carlo-emission predictive band, then *empirically* rescales that
band from the residuals on a held-out tail of the SAME history, which usually tracks the nominal
level better than the raw model band when the HMM is mis-specified (few states, short history,
non-Gaussian residuals, ...).

That rescaling borrows split-conformal machinery but **is not a split-conformal guarantee**, and the
band must not be described as one. Split conformal earns its finite-sample coverage from
exchangeable calibration scores; here the calibration cases are successive horizons from a single
held-out tail, so their residuals are ordered, dependent and generally non-exchangeable, and one
constant half-width is then transferred to every future horizon of a *different* model refit on the
full series. Neither residual invariance across horizon nor invariance across the refit is
established. :class:`DemandBandCalibration`, attached to every returned forecast, records exactly
what was done -- one calibration origin, no horizon-specific evidence, ``guarantee="empirical"`` --
so a caller can see the band is an empirical heuristic rather than a certified interval. A
time-series-valid procedure (rolling/block conformal over multiple forecast origins, with
horizon-specific calibration retained) is what a coverage claim here would require.

:func:`route_distribution` then turns that forecast into a network flow problem: the forecast mean
is the per-node demand, ``supply_nodes - demand`` is the net supply IC-9's :func:`min_cost_flow`
consumes directly (H1's already-solved minimum-cost network flow), giving the min-cost dispatch that
meets the forecast at once.

    >>> history = [50.0 + 10 * ((-1) ** (t // 12)) for t in range(60)]
    >>> f = forecast_demand(history, horizon=4, level=0.9, seed=0)
    >>> f.mean.shape
    (4,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.conformal import split_conformal
from mixle.inference.forecast import Forecast, forecast
from mixle.relations import Flow, min_cost_flow

__all__ = ["DemandBandCalibration", "DemandForecast", "forecast_demand", "route_distribution"]

_N_STATES = 2  # low/high demand regimes -- matches the two-state precedent in forecast()'s own tests
_CAL_FRAC = 0.35  # fraction of history reserved as the calibration tail
_MAX_ITS = 50

_BAND_ASSUMPTIONS = (
    "calibration residuals are successive horizons from one held-out tail: ordered, dependent, "
    "generally NOT exchangeable",
    "one constant half-width is transferred to every horizon; residual invariance across horizon is not established",
    "the half-width is measured on a model fit to history[:-k] and applied to a model refit on the "
    "full history; invariance across the refit is not established",
)


@dataclass(frozen=True)
class DemandBandCalibration:
    """How a demand band's width was obtained, and what it is (and is not) entitled to claim.

    ``guarantee`` is ``"empirical"``: the width was measured from residuals, and nothing here
    establishes the exchangeability split conformal needs, so no finite-sample coverage is asserted.
    """

    method: str
    guarantee: str
    n_calibration: int
    calibration_origins: int
    horizon_specific: bool
    half_width: float
    assumptions: tuple[str, ...] = _BAND_ASSUMPTIONS


@dataclass
class DemandForecast(Forecast):
    """A :class:`Forecast` carrying the :class:`DemandBandCalibration` receipt for its band."""

    calibration: DemandBandCalibration | None = None


def _fit_demand_hmm(values: np.ndarray, *, seed: int) -> Any:
    """Fit a small Gaussian-emission HMM to a univariate demand series (EM, quantile-seeded init).

    A cold/random EM init on a single continuous sequence is prone to a degenerate collapse (a state
    grabbing zero mass and its variance flooring to ~0); seeding from the calibration-set's own lower/
    upper quartiles as two well-separated regimes (with a mildly sticky transition prior) gives EM a
    basin that converges to a genuine two-regime fit instead.
    """
    from mixle.inference.estimation import optimize
    from mixle.stats import GaussianDistribution, GaussianEstimator, HiddenMarkovModelDistribution
    from mixle.stats.latent.hidden_markov import HiddenMarkovEstimator

    lo_q, hi_q = np.quantile(values, [0.25, 0.75])
    var0 = max(float(values.std()) ** 2, 1.0e-6)
    init = HiddenMarkovModelDistribution(
        [GaussianDistribution(float(lo_q), var0), GaussianDistribution(float(hi_q), var0)],
        [0.5, 0.5],
        [[0.8, 0.2], [0.2, 0.8]],
    )
    estimator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(_N_STATES)], pseudo_count=(1.0, 1.0))
    return optimize([values.tolist()], estimator, max_its=_MAX_ITS, prev_estimate=init, structure="off")


def forecast_demand(history: Any, horizon: int, *, level: float = 0.9, seed: int = 0) -> DemandForecast:
    """Forecast ``horizon`` steps of demand beyond ``history``, with an empirically rescaled band.

    Fits a 2-state Gaussian-emission HMM to ``history`` and calls :func:`mixle.inference.forecast.forecast`
    for the exact state-marginal / Monte-Carlo predictive mean and band; the band width is then
    rescaled using :func:`mixle.inference.conformal.split_conformal`'s quantile machinery: a model fit
    on everything but the last ``~35%`` of ``history`` forecasts that held-out tail, the
    (prediction, actual) pairs there give one constant additive half-width at ``level``, and that
    half-width is placed around the mean of a *second* model refit on the full ``history``.

    The band is empirical, **not** a split-conformal interval. Its calibration cases are successive
    horizons from a single held-out tail -- ordered, dependent, generally non-exchangeable -- so the
    finite-sample coverage split conformal offers is not established here, and the one half-width is
    reused across horizons and across a model refit without evidence of invariance. Empirically the
    band usually tracks ``level`` closely on well-behaved series, which is what it is for; treat it
    as a calibrated-looking heuristic and read ``.calibration`` for the terms.

    Args:
        history: past demand observations (one univariate series).
        horizon: number of future periods to forecast.
        level: central-interval mass the half-width targets (0.9 -> a ~90%-wide band).
        seed: reproducibility (both the HMM's EM/MC internals and the calibration split).

    Returns:
        A :class:`DemandForecast` (`.mean`, `.lo`, `.hi`, `.level`, `.state_probs`, `.calibration`).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim != 1:
        raise ValueError("history must be a 1-D demand series")
    n = hist.shape[0]
    if n < 8:
        raise ValueError("forecast_demand needs at least 8 history points to hold out a calibration tail")

    k = max(horizon, int(round(_CAL_FRAC * n)))
    k = min(k, n - 4)  # leave at least 4 points to fit the calibration-split model itself
    train, cal_actual = hist[:-k], hist[-k:]

    cal_model = _fit_demand_hmm(train, seed=seed)
    cal_forecast = forecast(cal_model, train.tolist(), horizon=k, level=level, seed=seed)
    cal_pred = np.asarray(cal_forecast.mean, dtype=np.float64)

    alpha = 1.0 - level
    _lo_adj, hi_adj = split_conformal(cal_pred, cal_actual, cal_pred, alpha=alpha, side="two-sided")
    half_width = float(np.mean(hi_adj - cal_pred))  # split_conformal's two-sided q is a single constant

    final_model = _fit_demand_hmm(hist, seed=seed)
    final_forecast = forecast(final_model, hist.tolist(), horizon=horizon, level=level, seed=seed)
    mean = np.asarray(final_forecast.mean, dtype=np.float64)

    return DemandForecast(
        mean=mean,
        lo=mean - half_width,
        hi=mean + half_width,
        level=level,
        state_probs=final_forecast.state_probs,
        samples=final_forecast.samples,
        calibration=DemandBandCalibration(
            method="single-origin split-conformal quantile on a held-out tail",
            guarantee="empirical",
            n_calibration=int(k),
            calibration_origins=1,
            horizon_specific=False,
            half_width=half_width,
        ),
    )


def route_distribution(supply_nodes: Any, demand_forecast: Forecast, cost: Any, cap: Any) -> Flow:
    """Route supply to meet forecast demand at minimum cost (H1/IC-9's :func:`min_cost_flow`).

    ``supply_nodes`` and ``demand_forecast.mean`` align one-to-one, per node (a plant/depot/customer
    per forecast horizon step); ``supply = supply_nodes - demand_forecast.mean`` is the net node
    supply :func:`min_cost_flow` consumes directly under the given ``(n, n)`` arc ``cap``/``cost``.

    Args:
        supply_nodes: length-``n`` available supply per node.
        demand_forecast: a :class:`Forecast` (from :func:`forecast_demand` or
            :func:`mixle.inference.forecast.forecast`) whose ``.mean`` is length ``n``.
        cost: ``(n, n)`` per-unit arc routing cost.
        cap: ``(n, n)`` arc capacity.

    Returns:
        The resolved :class:`mixle.relations.Flow` (``value`` = total routing cost, ``flow`` = arcs).
    """
    supply_nodes = np.asarray(supply_nodes, dtype=np.float64)
    demand = np.asarray(demand_forecast.mean, dtype=np.float64)
    if supply_nodes.shape != demand.shape:
        raise ValueError("supply_nodes and demand_forecast.mean must align, one entry per node")
    supply = supply_nodes - demand
    return min_cost_flow(cap, cost, supply)
