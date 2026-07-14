"""J1: forecast_price() delivers a conformally-calibrated commodity-price band.

DoD: on a fitted synthetic price HMM, the empirical out-of-sample coverage of ``[lo, hi]`` over
held-out steps is within 0.05 of ``level`` (fraction of held-out truths inside the band; no
reliance on ``coverage.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.inference.price_forecast import PriceForecast, forecast_price
from mixle.stats import GaussianDistribution, HiddenMarkovModelDistribution

_MEANS = [22.0, 38.0]  # e.g. a two-regime commodity price ($/unit)
_STDS = [2.0, 3.5]


def _price_hmm(stay: float = 0.95) -> HiddenMarkovModelDistribution:
    return HiddenMarkovModelDistribution(
        [GaussianDistribution(_MEANS[0], _STDS[0]), GaussianDistribution(_MEANS[1], _STDS[1])],
        [0.5, 0.5],
        [[stay, 1 - stay], [1 - stay, stay]],
    )


def _simulate_path(model: HiddenMarkovModelDistribution, n_steps: int, seed: int) -> np.ndarray:
    """A plain-numpy price path from the same regime-switching process the model represents."""
    rng = np.random.RandomState(seed)
    a = np.asarray(model.transitions)
    w = np.asarray(model.w)
    s = int(rng.choice(len(w), p=w))
    path = np.empty(n_steps)
    for t in range(n_steps):
        s = int(rng.choice(len(w), p=a[s]))
        path[t] = rng.normal(_MEANS[s], _STDS[s])
    return path


def test_out_of_sample_coverage_matches_nominal_level():
    model = _price_hmm(stay=0.95)
    level = 0.9
    window = 40  # fixed-size rolling history window for each backtest step
    full_path = _simulate_path(model, n_steps=window + 220, seed=7)

    hits = 0
    total = 0
    for t in range(window, len(full_path)):
        history = full_path[t - window : t]
        pf = forecast_price(model, history, horizon=1, level=level, cal_frac=0.3, seed=t)
        truth = full_path[t]
        hits += int(pf.lo[0] <= truth <= pf.hi[0])
        total += 1

    coverage = hits / total
    assert abs(coverage - level) < 0.05, f"empirical coverage {coverage:.3f} vs nominal {level}"


def test_paths_are_scenario_draws_for_downstream_monte_carlo_dcf():
    model = _price_hmm(stay=0.9)
    history = _simulate_path(model, n_steps=60, seed=1)
    pf = forecast_price(model, history, horizon=6, level=0.9, seed=0)

    assert isinstance(pf, PriceForecast)
    assert pf.mean.shape == (6,)
    assert pf.lo.shape == (6,)
    assert pf.hi.shape == (6,)
    assert pf.paths.ndim == 2
    assert pf.paths.shape[0] == 6
    # the band must actually bracket the point forecast
    assert np.all(pf.lo <= pf.mean) and np.all(pf.mean <= pf.hi)


def test_rejects_too_short_horizon_and_bad_cal_frac():
    model = _price_hmm()
    history = _simulate_path(model, n_steps=30, seed=2)
    with pytest.raises(ValueError):
        forecast_price(model, history, horizon=0)
    with pytest.raises(ValueError):
        forecast_price(model, history, horizon=3, cal_frac=0.0)
    with pytest.raises(ValueError):
        forecast_price(model, history, horizon=3, cal_frac=1.0)
