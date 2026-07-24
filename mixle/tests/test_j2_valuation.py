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


class _SingleRowGradePosterior:
    """A misbehaving IC-1 `Posterior` stub that always returns exactly ONE draw, no matter what ``n``
    is requested -- the MXR-080-0118 'one-row posterior' repro."""

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.array([[GRADE_MU + 1.0]])

    @property
    def mean(self) -> np.ndarray:
        return np.array([GRADE_MU + 1.0])

    @property
    def cov(self) -> np.ndarray:
        return np.array([[1.0]])

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([GRADE_MU]), np.array([GRADE_MU + 2.0])

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError("unused by this test")


class _ConstantGradePosterior:
    """An IC-1 `Posterior` stub that returns exactly ``n`` draws (a well-behaved draw COUNT), all equal
    to a fixed ``value`` -- lets tests pin the grade VALUE (e.g. negative, non-finite) independently of
    the draw-count contract."""

    def __init__(self, value: float) -> None:
        self._value = value

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full((n, 1), self._value)

    @property
    def mean(self) -> np.ndarray:
        return np.array([self._value])

    @property
    def cov(self) -> np.ndarray:
        return np.array([[0.0]])

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self._value]), np.array([self._value])

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


def test_monte_carlo_npv_accepts_a_one_arg_cost_model():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))

    def one_arg_cost_model(t: int) -> float:
        assert t == 0
        return OPEX_PER_TONNE * TONNAGE

    result = monte_carlo_npv(
        posterior,
        price_paths,
        one_arg_cost_model,
        _schedule(),
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(result.mean)


def test_monte_carlo_npv_does_not_swallow_an_unrelated_type_error_from_a_two_arg_cost_model():
    # A TypeError raised *inside* cost_model(t, tonnage_t) for a reason unrelated to arity used to be
    # misread as "cost_model only takes one argument" and silently retried as cost_model(t) --
    # invoking cost_model a second time (with different side effects) and masking the real error.
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))
    calls = []

    def buggy_cost_model(t: int, tonnage_t: float) -> float:
        calls.append(t)
        raise TypeError("boom: unrelated bug inside cost_model")

    with pytest.raises(TypeError):
        monte_carlo_npv(
            posterior,
            price_paths,
            buggy_cost_model,
            _schedule(),
            discount_rate=DISCOUNT_RATE,
            n=N,
            rng=np.random.default_rng(0),
        )
    assert calls == [0]  # called exactly once, not retried as a one-arg call


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


def test_monte_carlo_npv_accepts_price_forecast_paths_directly():
    # The documented J1 -> J2 workflow is `monte_carlo_npv(..., price_paths=pf.paths, ...)`, and
    # mixle.inference.price_forecast.PriceForecast.paths is (n_periods, m) -- time-major -- not the
    # (m, n_periods) scenario-major shape monte_carlo_npv's own docstring required. A 5-period,
    # non-square schedule makes the two orientations unambiguous (and would previously raise).
    n_periods = 5
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    m = 777  # != n_periods and != N, so both resampling AND the transpose are exercised together
    price_forecast_paths = np.random.default_rng(3).normal(PRICE_MEAN, PRICE_STD, size=(n_periods, m))

    def cost_model(t: int, tonnage_t: float) -> float:
        return OPEX_PER_TONNE * tonnage_t

    schedule = {"tonnage": np.full(n_periods, TONNAGE), "capex": np.array([CAPEX] + [0.0] * (n_periods - 1))}

    result = monte_carlo_npv(
        posterior,
        price_forecast_paths,  # (n_periods, m): PriceForecast.paths' own orientation, unmodified
        cost_model,
        schedule,
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(3),
    )
    assert isinstance(result, NPVDistribution)
    assert result.samples.shape == (N,)
    assert np.all(np.isfinite(result.samples))

    # Passing the manually pre-transposed (m, n_periods) form must give the identical distribution
    # (same seed): proves the auto-detected transpose is not just "doesn't crash" but scores the
    # correct axis as "period", not merely a differently-shaped one.
    result_pretransposed = monte_carlo_npv(
        posterior,
        price_forecast_paths.T,
        cost_model,
        schedule,
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(3),
    )
    np.testing.assert_array_equal(result.samples, result_pretransposed.samples)


# --- MXR-080-0118: Monte Carlo NPV does not enforce its sample/economic contract -------------------
#
# `n` used to not be checked as an exact integer, the posterior's returned draw count was never
# checked against the request (so a one-row posterior would silently numpy-broadcast across every
# price scenario, fabricating repeated draws), non-finite/negative grade/price/tonnage/capex/opex
# propagated straight through to the DCF, and `discount_rate <= -1` produced a division by zero or an
# alternating-sign discount factor. The tests below pin the fix.


def test_monte_carlo_npv_rejects_non_integer_n():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(10, 1))
    with pytest.raises(ValueError):
        monte_carlo_npv(
            posterior,
            price_paths,
            _cost_model,
            {"tonnage": np.array([TONNAGE])},
            discount_rate=DISCOUNT_RATE,
            n=10.5,  # not an int, even though it is a positive, integer-valued float
            rng=np.random.default_rng(0),
        )


def test_monte_carlo_npv_rejects_non_positive_n():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(10, 1))
    for bad_n in (0, -5):
        with pytest.raises(ValueError):
            monte_carlo_npv(
                posterior,
                price_paths,
                _cost_model,
                {"tonnage": np.array([TONNAGE])},
                discount_rate=DISCOUNT_RATE,
                n=bad_n,
                rng=np.random.default_rng(0),
            )


def test_monte_carlo_npv_rejects_posterior_that_returns_fewer_draws_than_requested():
    # The exact MXR-080-0118 repro: a posterior that always returns exactly one row used to silently
    # numpy-broadcast that single grade draw against all `n` requested price scenarios, fabricating `n`
    # "independent" NPV samples that in fact all shared the one real grade draw.
    posterior = _SingleRowGradePosterior()
    n_req = 500
    price_paths = np.random.default_rng(4).normal(PRICE_MEAN, PRICE_STD, size=(n_req, 1))
    with pytest.raises(ValueError, match="returned 1 draw"):
        monte_carlo_npv(
            posterior,
            price_paths,
            _cost_model,
            _schedule(),
            discount_rate=DISCOUNT_RATE,
            n=n_req,
            rng=np.random.default_rng(4),
        )


def test_monte_carlo_npv_rejects_non_finite_or_negative_grade_draws():
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(10, 1))
    for bad_value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            monte_carlo_npv(
                _ConstantGradePosterior(bad_value),
                price_paths,
                _cost_model,
                {"tonnage": np.array([TONNAGE])},
                discount_rate=DISCOUNT_RATE,
                n=10,
                rng=np.random.default_rng(0),
            )


def test_monte_carlo_npv_rejects_non_finite_or_negative_price_paths():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    for bad_value in (-50.0, float("nan"), float("inf")):
        price_paths = np.full((10, 1), bad_value)
        with pytest.raises(ValueError):
            monte_carlo_npv(
                posterior,
                price_paths,
                _cost_model,
                {"tonnage": np.array([TONNAGE])},
                discount_rate=DISCOUNT_RATE,
                n=10,
                rng=np.random.default_rng(0),
            )


def test_monte_carlo_npv_rejects_non_finite_or_negative_schedule_quantities():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(10, 1))
    for bad_schedule in (
        {"tonnage": np.array([-TONNAGE])},
        {"tonnage": np.array([float("nan")])},
        {"tonnage": np.array([TONNAGE]), "capex": np.array([-CAPEX])},
        {"tonnage": np.array([TONNAGE]), "capex": np.array([float("inf")])},
    ):
        with pytest.raises(ValueError):
            monte_carlo_npv(
                posterior,
                price_paths,
                _cost_model,
                bad_schedule,
                discount_rate=DISCOUNT_RATE,
                n=10,
                rng=np.random.default_rng(0),
            )


def test_monte_carlo_npv_rejects_negative_or_non_finite_opex_from_cost_model():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(10, 1))
    for bad_opex in (-1.0, float("nan"), float("inf")):

        def bad_cost_model(t: int, tonnage_t: float, _bad_opex=bad_opex) -> float:
            return _bad_opex

        with pytest.raises(ValueError):
            monte_carlo_npv(
                posterior,
                price_paths,
                bad_cost_model,
                {"tonnage": np.array([TONNAGE])},
                discount_rate=DISCOUNT_RATE,
                n=10,
                rng=np.random.default_rng(0),
            )


def test_monte_carlo_npv_rejects_discount_rate_at_or_below_negative_one():
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(0).normal(PRICE_MEAN, PRICE_STD, size=(10, 2))
    for bad_rate in (-1.0, -1.5, -2.0, float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            monte_carlo_npv(
                posterior,
                price_paths,
                _cost_model,
                {"tonnage": np.array([TONNAGE, TONNAGE])},
                discount_rate=bad_rate,
                n=10,
                rng=np.random.default_rng(0),
            )


def test_monte_carlo_npv_negative_control_normal_call_still_works():
    # Negative control: a normal, well-shaped Monte Carlo NPV calculation (positive integer n, a
    # conforming posterior, finite non-negative economics, a sane discount rate) is unaffected by the
    # MXR-080-0118 validation and still produces a finite, sensible distribution.
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
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
    assert result.samples.shape == (N,)
    assert np.all(np.isfinite(result.samples))
    assert result.p10 < result.p50 < result.p90
