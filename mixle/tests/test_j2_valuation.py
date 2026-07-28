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

from mixle.analysis.valuation import NPVDistribution, _sobol_first_order_share, monte_carlo_npv
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
    # Seeded independently of `rng` below (101, not 1): Generator.normal and Generator.lognormal, freshly
    # seeded identically, consume the SAME underlying standard-normal stream (lognormal is exp(normal)
    # internally) -- so a `price_paths` generator sharing `rng`'s own seed would make this test's grade
    # and price accidentally perfectly (monotonically) correlated in the primary sample, instead of the
    # independent uncertainty sources the assertions below assume. A different seed avoids that.
    price_paths = np.random.default_rng(101).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))

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
    # Both grade and price are genuine (independent) uncertainty sources here, so each should explain a
    # material share of the variance (loosely bounded -- this is not a precise ANOVA claim; see
    # test_sensitivity_recovers_known_analytic_first_order_indices for a tight quantitative check).
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


class _DeterministicGradePosterior:
    """An IC-1 `Posterior` stub that always returns the exact same ``k`` grade values, ignoring ``rng``
    entirely -- lets a test pin an EXACT (grade, price) pairing, as MXR-080-0117's repro needs."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        assert n == self._values.shape[0]
        return self._values[:, None]

    @property
    def mean(self) -> np.ndarray:
        return np.array([float(np.mean(self._values))])

    @property
    def cov(self) -> np.ndarray:
        return np.array([[float(np.var(self._values))]])

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([float(np.min(self._values))]), np.array([float(np.max(self._values))])

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError("unused by this test")


# --- MXR-080-0117: NPV sensitivity shares can exceed one by millions ------------------------------
#
# The previous sensitivity computation froze one factor at its mean, varied the other, and divided
# that frozen-factor variance by the variance of the actual paired joint sample. With negatively paired
# grade/price scenarios, the joint (paired) sample's variance can be driven nearly to zero by
# cancellation while each frozen-factor variance stays large, sending the ratio into the millions --
# despite the API documenting values in [0, 1]. Replaced with Sobol first-order variance-based
# sensitivity indices, estimated via a second sample independent of the caller's own (possibly
# adversarially paired) joint sample -- see _sobol_first_order_share. The tests below pin the fix.


def test_sensitivity_negatively_paired_repro_no_longer_explodes():
    # The audit's exact adversarial construction: two grade draws, two price scenarios, paired so that
    # grade*price products are nearly IDENTICAL (near-zero joint variance) while grade alone and price
    # alone (each frozen at the other's mean, under the old method) varied enormously. The old
    # ratio-of-variances method returned shares in the millions (grade ~2,251,500; price ~2,254,502) on
    # exactly this kind of input.
    grade_vals = np.array([1.0, 100.0])
    price_vals = np.array([100.0, 1.01])  # paired: grade[0]*price[0] = 100, grade[1]*price[1] = 101
    posterior = _DeterministicGradePosterior(grade_vals)
    price_paths = price_vals[:, None]

    result = monte_carlo_npv(
        posterior,
        price_paths,
        lambda t, tonnage_t: 0.0,
        {"tonnage": np.array([1.0]), "capex": np.array([0.0])},
        discount_rate=0.0,
        n=2,
        rng=np.random.default_rng(0),
    )
    assert 0.0 <= result.sensitivity["grade"] <= 1.0
    assert 0.0 <= result.sensitivity["price"] <= 1.0
    # The old method's failure mode was specifically shares in the *millions*; also pin a generous bound
    # far below that as a belt-and-suspenders check against any regression back toward it.
    assert result.sensitivity["grade"] < 10.0
    assert result.sensitivity["price"] < 10.0


def test_sensitivity_shares_always_bounded_in_unit_interval():
    # Broader stress sweep across many small, awkward (grade, price) pairings -- including several more
    # negatively-paired constructions in the spirit of the MXR-080-0117 repro, and some independent ones
    # -- never produces a share outside [0, 1].
    driver = np.random.default_rng(123)
    for _ in range(25):
        k = int(driver.integers(2, 12))
        grade_vals = driver.uniform(0.1, 100.0, size=k)
        if driver.uniform() < 0.5:
            price_vals = 500.0 / grade_vals  # anti-correlated by construction, like the 0117 repro
        else:
            price_vals = driver.uniform(1.0, 100.0, size=k)  # unrelated to grade

        posterior = _DeterministicGradePosterior(grade_vals)
        result = monte_carlo_npv(
            posterior,
            price_vals[:, None],
            lambda t, tonnage_t: 0.0,
            {"tonnage": np.array([1.0])},
            discount_rate=0.0,
            n=k,
            rng=np.random.default_rng(int(driver.integers(0, 2**31 - 1))),
        )
        assert 0.0 <= result.sensitivity["grade"] <= 1.0, (k, grade_vals, price_vals)
        assert 0.0 <= result.sensitivity["price"] <= 1.0, (k, grade_vals, price_vals)


def test_sobol_first_order_share_recovers_known_analytic_additive_case():
    # Isolate the ESTIMATOR from this module's mine-valuation-specific plumbing: for a purely additive
    # Y = a*X1 + b*X2 with independent X1, X2 ~ N(0, 1), the Sobol first-order index for each factor has
    # a clean closed form -- S1 = a**2 / (a**2 + b**2), S2 = b**2 / (a**2 + b**2) -- and, being purely
    # additive (no interaction term), S1 + S2 == 1 exactly in the population.
    rng = np.random.default_rng(42)
    n = 200_000
    a, b = 3.0, 1.0

    x1_a, x2_a = rng.normal(0.0, 1.0, size=n), rng.normal(0.0, 1.0, size=n)
    x1_b, x2_b = rng.normal(0.0, 1.0, size=n), rng.normal(0.0, 1.0, size=n)
    y_a = a * x1_a + b * x2_a
    y_b = a * x1_b + b * x2_b
    y_ab1 = a * x1_b + b * x2_a  # factor 1 (x1) swapped in from B, factor 2 stays A
    y_ab2 = a * x1_a + b * x2_b  # factor 2 (x2) swapped in from B, factor 1 stays A

    total_variance = float(np.var(y_a))
    variance_1, s1 = _sobol_first_order_share(y_a, y_b, y_ab1, total_variance)
    variance_2, s2 = _sobol_first_order_share(y_a, y_b, y_ab2, total_variance)

    assert variance_1 >= 0.0
    assert variance_2 >= 0.0
    s1_true = a**2 / (a**2 + b**2)
    s2_true = b**2 / (a**2 + b**2)
    assert s1 == pytest.approx(s1_true, abs=0.02)
    assert s2 == pytest.approx(s2_true, abs=0.02)
    assert (s1 + s2) == pytest.approx(1.0, abs=0.02)


def test_sensitivity_recovers_known_analytic_first_order_indices():
    # For monte_carlo_npv's actual (multiplicative) cashflow structure -- a single period, deterministic
    # tonnage/opex/capex, so NPV = tonnage*grade*price - opex - capex -- with independent grade ~
    # Lognormal(mu, sigma) and price ~ Normal(mean, std), the first-order Sobol indices have a closed
    # form via the variance of a product of independent random variables,
    # Var(grade*price) = Var(g)*Var(p) + Var(g)*E[p]**2 + E[g]**2*Var(p):
    #   S_grade = Var(g)*E[p]**2 / Var(g*p),  S_price = E[g]**2*Var(p) / Var(g*p).
    # NPV's constant tonnage scale and opex/capex shift don't affect a variance-based index (grade and
    # price are the only stochastic factors; see test_sobol_first_order_share_recovers_known_analytic_
    # additive_case for the estimator validated on a purely additive, interaction-free model instead).
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    # Seeded independently from `rng` (see test_sensitivity_decomposes_variance_between_grade_and_price
    # for why reusing the same seed value would accidentally correlate grade and price here).
    price_paths = np.random.default_rng(301).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))

    result = monte_carlo_npv(
        posterior,
        price_paths,
        _cost_model,
        _schedule(),
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(7),
    )

    e_g = np.exp(GRADE_MU + GRADE_SIGMA**2 / 2.0)
    var_g = np.expm1(GRADE_SIGMA**2) * np.exp(2 * GRADE_MU + GRADE_SIGMA**2)
    e_p, var_p = PRICE_MEAN, PRICE_STD**2
    var_gp = var_g * var_p + var_g * e_p**2 + e_g**2 * var_p
    s_grade_true = var_g * e_p**2 / var_gp
    s_price_true = e_g**2 * var_p / var_gp

    assert 0.0 <= result.sensitivity["grade"] <= 1.0
    assert 0.0 <= result.sensitivity["price"] <= 1.0
    assert result.sensitivity["grade"] == pytest.approx(s_grade_true, abs=0.08)
    assert result.sensitivity["price"] == pytest.approx(s_price_true, abs=0.05)


def test_sensitivity_independent_factors_negative_control():
    # Negative control: with grade and price independently drawn (no adversarial or accidental
    # pairing), both factors still decompose sensibly -- material, non-degenerate shares, comfortably
    # inside [0, 1], roughly consistent with (not pinned as tightly as) the closed-form values in
    # test_sensitivity_recovers_known_analytic_first_order_indices.
    posterior = _LognormalGradePosterior(GRADE_MU, GRADE_SIGMA)
    price_paths = np.random.default_rng(555).normal(PRICE_MEAN, PRICE_STD, size=(N, 1))

    result = monte_carlo_npv(
        posterior,
        price_paths,
        _cost_model,
        _schedule(),
        discount_rate=DISCOUNT_RATE,
        n=N,
        rng=np.random.default_rng(6),
    )
    sens = result.sensitivity
    assert 0.0 <= sens["grade"] <= 1.0
    assert 0.0 <= sens["price"] <= 1.0
    assert sens["grade"] > 0.5  # grade dominates NPV variance for these constants (matches the
    assert 0.05 < sens["price"] < 0.5  # closed-form ~0.81 / ~0.18 split, loosely bounded here)


# --- NPVDistribution construction-time validation: same samples-carrying-result-type guard applied ---
# --- to carcinogenic_risk.RiskQuantity, closing the same gap found in this sibling module. Since ---
# --- NPVDistribution is a NamedTuple (not a dataclass), the guard is a __new__ override on a genuine ---
# --- subclass rather than a __post_init__ (typing.NamedTuple forbids overriding __new__ directly). ---


def _npv_distribution_kwargs(**overrides):
    base = dict(samples=np.array([1.0, 2.0, 3.0]), mean=2.0, p10=1.2, p50=2.0, p90=2.8, sensitivity={})
    base.update(overrides)
    return base


def test_npv_distribution_rejects_empty_or_non_finite_samples():
    """`NPVDistribution` had no construction-time validation at all: empty or NaN/Inf samples were
    silently accepted. Defense-in-depth so invalid state can never flow downstream even if
    `monte_carlo_npv` itself (or some other caller) fails to validate its own inputs."""
    with pytest.raises(ValueError):
        NPVDistribution(**_npv_distribution_kwargs(samples=np.array([])))
    with pytest.raises(ValueError):
        NPVDistribution(**_npv_distribution_kwargs(samples=np.array([1.0, np.nan, 3.0])))
    with pytest.raises(ValueError):
        NPVDistribution(**_npv_distribution_kwargs(samples=np.array([1.0, np.inf, 3.0])))


def test_npv_distribution_rejects_empty_or_non_finite_samples_positionally():
    """The same validation applies to positional construction, not only keyword -- `__new__`
    intercepts every construction path, unlike a factory function a caller could bypass."""
    with pytest.raises(ValueError):
        NPVDistribution(np.array([]), 0.0, 0.0, 0.0, 0.0, {})
    with pytest.raises(ValueError):
        NPVDistribution(np.array([1.0, np.nan]), 0.0, 0.0, 0.0, 0.0, {})


def test_npv_distribution_accepts_valid_samples():
    """Negative control: a legitimate, non-empty, finite samples array still constructs cleanly and
    keeps behaving like a normal NamedTuple (indexing, unpacking, field access)."""
    result = NPVDistribution(**_npv_distribution_kwargs())
    assert isinstance(result, tuple)
    assert result.mean == 2.0
    assert result[0] is result.samples
    samples, mean, p10, p50, p90, sensitivity = result
    assert mean == 2.0 and p10 == 1.2 and p50 == 2.0 and p90 == 2.8 and sensitivity == {}
