"""J2 DoD — project valuation under uncertainty, NPV/DCF Monte Carlo (notes/exec/workstream-J.md).

A toy 1-block/1-period project: grade is lognormal (an IC-1 `Posterior` stub), price is normal (a J1
``PriceForecast.paths``-shaped array), tonnage/capex/opex are fixed. ``monte_carlo_npv``'s ``mean`` and
``p50`` must match an independent, hand-written Monte-Carlo reference built from the exact same formula
using the exact same seed (``np.random.default_rng(0)``) — not a call into `monte_carlo_npv` itself.

Both the implementation and the reference draw grade with a *freshly seeded* ``np.random.default_rng(0)``
as their very first (and only) call on that generator, and the price scenarios are the same fixed array
passed to both, so the two computations consume bit-identical randomness and should agree far tighter
than ordinary Monte-Carlo sampling noise — the tolerance below is generous headroom, not the expected
error.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.valuation import NPVDistribution, monte_carlo_npv
from mixle.reason.posterior_protocol import Posterior

N = 20_000
DISCOUNT_RATE = 0.10
TONNAGE = 1_000_000.0  # tonnes, the project's single period
GRADE_MU = 0.0  # ln(grade); grade ~ lognormal(mu, sigma), median grade = 1.0 unit/t
GRADE_SIGMA = 0.25
PRICE_MEAN = 50.0  # $/unit
PRICE_STD = 6.0
OPEX_PER_TONNE = 8.0  # $/t
CAPEX = 5_000_000.0  # $, period-0 capital


class _LognormalGradePosterior:
    """A minimal IC-1 `Posterior` over a single project-life head grade, lognormally distributed."""

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size=(n, 1))

    @property
    def mean(self) -> np.ndarray:
        return np.array([np.exp(self.mu + self.sigma**2 / 2.0)])

    @property
    def cov(self) -> np.ndarray:
        return np.array([[np.expm1(self.sigma**2) * np.exp(2 * self.mu + self.sigma**2)]])

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        draws = self.samples(200_000, np.random.default_rng(0))
        return np.quantile(draws, a, axis=0), np.quantile(draws, 1 - a, axis=0)

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError("unused by this test")


def _cost_model(t: int, tonnage_t: float) -> float:
    assert t == 0
    return OPEX_PER_TONNE * tonnage_t


def _schedule():
    return {"tonnage": np.array([TONNAGE]), "capex": np.array([CAPEX])}


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_LognormalGradePosterior(GRADE_MU, GRADE_SIGMA), Posterior)


def test_monte_carlo_npv_matches_hand_rolled_reference():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    # Fixed price scenarios, exactly N rows so monte_carlo_npv's grade/price pairing needs no
    # resampling -- the only randomness either computation consumes is the grade draw below.
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))

    result = monte_carlo_npv(
        posterior,
        price_paths,
        _cost_model,
        _schedule(),
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(0),
    )

    assert isinstance(result, NPVDistribution)
    assert result.samples.shape == (N,)

    # Independent hand-written reference: same formula, same seeds, no call into monte_carlo_npv.
    grade_ref = np.random.default_rng(0).lognormal(GRADE_MU, GRADE_SIGMA, size=N)
    price_ref = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=N)
    cashflow_ref = TONNAGE * grade_ref * price_ref - OPEX_PER_TONNE * TONNAGE - CAPEX
    npv_ref = cashflow_ref / (1.0 + DISCOUNT_RATE) ** 0  # single period, t = 0: undiscounted

    assert result.mean == pytest.approx(npv_ref.mean(), rel=1e-9, abs=1e-6)
    assert result.p50 == pytest.approx(float(np.median(npv_ref)), rel=1e-9, abs=1e-6)

    # p10 < p50 < p90 and mean sits inside the P10-P90 band for this roughly-symmetric setup.
    assert result.p10 < result.p50 < result.p90
    assert result.p10 <= result.mean <= result.p90


def test_sensitivity_decomposes_variance_between_grade_and_price():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(1).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))

    result = monte_carlo_npv(
        posterior,
        price_paths,
        _cost_model,
        _schedule(),
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(1),
    )

    sens = result.sensitivity
    assert set(sens) >= {"grade", "price", "grade_variance", "price_variance", "total_variance"}
    assert 0.0 <= sens["grade"] <= 1.0
    assert 0.0 <= sens["price"] <= 1.0
    assert sens["total_variance"] > 0.0
    # Both grade and price are genuine uncertainty sources here, so each should explain a material
    # share of the variance (loosely bounded -- this is not a precise ANOVA claim).
    assert sens["grade"] > 0.05
    assert sens["price"] > 0.05


def test_monte_carlo_npv_resamples_mismatched_price_path_count():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    # Only 500 scenario paths for N draws -- monte_carlo_npv must align (resample) rather than error.
    price_paths = np.random.default_rng(2).normal(PRICE_MEAN, PRICE_STD, size=(500, 1))

    result = monte_carlo_npv(
        posterior,
        price_paths,
        _cost_model,
        _schedule(),
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(2),
    )
    assert result.samples.shape == (N,)
    assert np.isfinite(result.mean)


def test_monte_carlo_npv_rejects_period_dimension_mismatch():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(N, 3))  # 3 periods
    with pytest.raises(ValueError):
        monte_carlo_npv(
            posterior,
            price_paths,
            _cost_model,
            _schedule(),  # 1-period schedule -- shape mismatch against the 3-period price_paths
            discount_rate=DISCOUNT_RATE,
            n=N,
            rng=np.random.default_rng(0),
        )
