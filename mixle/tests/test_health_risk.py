"""K3 DoD -- dose-response / health-risk models (notes/exec/workstream-K.md).

`DoseResponse.probability` must turn an exposure `Posterior` (IC-1) into an outcome-probability
`DerivedQuantity` whose credible interval is *calibrated*: a nominal 90% interval built from one draw
of the pushforward should cover the true dose-response probability -- computed from independent
fresh draws of the same exposure distribution -- close to 90% of the time, and that interval should
widen as the exposure posterior's own variance grows.

Named with the ``test_*.py`` prefix (rather than this repo's own ``*_test.py`` `python_files`
convention -- see ``pyproject.toml``) because this exact path + node id is the frozen DoD command in
``notes/exec/workstream-K.md``; explicit pytest node ids are collected regardless of the
``python_files`` glob, so this does not conflict with the repo's discovery config.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.health_risk import DoseResponse, cumulative_exposure
from mixle.reason.posterior_protocol import Posterior


def _lognormal_exposure_posterior(mu: float, sigma: float) -> Posterior:
    """A minimal IC-1 `Posterior` over a single-receptor exposure: dose ~ LogNormal(mu, sigma)."""

    class _ExposurePosterior:
        def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
            return np.exp(mu + sigma * rng.standard_normal(n))

        @property
        def mean(self) -> np.ndarray:
            return np.array([np.exp(mu + sigma**2 / 2.0)])

        @property
        def cov(self) -> np.ndarray:
            var = (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)
            return np.array([[var]])

        def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
            s = self.samples(200_000, np.random.default_rng(999))
            a = (1.0 - level) / 2.0
            return np.array([np.quantile(s, a)]), np.array([np.quantile(s, 1.0 - a)])

        def derived_quantity(self, fn, n: int, rng: np.random.Generator):
            draws = self.samples(n, rng)
            pushed = fn(draws)

            class _DQ:
                samples = pushed
                prior_dominated = False

                def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
                    a = (1.0 - level) / 2.0
                    return np.quantile(self.samples, a), np.quantile(self.samples, 1.0 - a)

            return _DQ()

    return _ExposurePosterior()


def test_dose_response_calibrated():
    mu, sigma = np.log(15.0), 0.4
    posterior = _lognormal_exposure_posterior(mu, sigma)
    assert isinstance(posterior, Posterior)

    dr = DoseResponse(model="loglinear", params={"beta": 0.05})
    dq = dr.probability(posterior, n=5000, rng=np.random.default_rng(1))
    assert dq.prior_dominated is False
    lo, hi = dq.credible_interval(0.9)

    # Empirical coverage: fresh, independent draws from the *same* generative exposure distribution,
    # pushed through the same response curve, should fall inside the nominal 90% interval close to
    # 90% of the time (loglinear is monotone in dose, so quantiles commute with the pushforward).
    check_rng = np.random.default_rng(42)
    true_doses = np.exp(mu + sigma * check_rng.standard_normal(5000))
    true_probs = dr.response_fn()(true_doses)
    coverage = float(np.mean((true_probs >= lo) & (true_probs <= hi)))
    assert coverage >= 0.88

    # The interval must widen as the exposure posterior's variance grows.
    wide_posterior = _lognormal_exposure_posterior(mu, sigma * 2.0)
    dq_wide = dr.probability(wide_posterior, n=5000, rng=np.random.default_rng(2))
    lo_wide, hi_wide = dq_wide.credible_interval(0.9)
    assert (hi_wide - lo_wide) > (hi - lo)


def test_dose_response_rejects_invalid_parameters():
    """MXR-080-0094: each model's coefficients are validated against their own domain at construction,
    since a negative/non-finite beta, an invalid Hill ec50/exponent/emax, or a non-finite logit/
    threshold parameter can otherwise produce a negative, above-one, or NaN "probability"."""
    invalid = [
        ("loglinear", {"beta": -0.1}),  # negative beta -> P < 0 for any dose > 0
        ("loglinear", {"beta": float("nan")}),
        ("loglinear", {"beta": float("inf")}),
        ("hill", {"ec50": 0.0}),  # ec50 == 0 -> 0/0 at dose == 0
        ("hill", {"ec50": -5.0}),  # negative ec50 with non-integer n -> complex, not just non-finite
        ("hill", {"ec50": 1.0, "n": -1.0}),  # negative Hill exponent -> divide by zero at dose == 0
        ("hill", {"ec50": 1.0, "n": 0.0}),  # zero exponent collapses to a dose-independent constant
        ("hill", {"ec50": 1.0, "emax": 1.5}),  # emax > 1 -> P > 1 well before infinite dose
        ("hill", {"ec50": 1.0, "emax": -0.1}),
        ("logit", {"a": float("nan")}),
        ("logit", {"b": float("inf")}),
        ("threshold_linear", {"slope": float("inf")}),
        ("threshold_linear", {"slope": 1.0, "threshold": float("nan")}),
    ]
    for model, params in invalid:
        with pytest.raises(ValueError):
            DoseResponse(model=model, params=params)


def test_dose_response_valid_parameters_still_produce_probabilities():
    """Negative control for MXR-080-0094: legitimate parameters for every model keep producing finite,
    in-range output -- the new per-model domain checks reject only genuinely invalid coefficients."""
    cases = [
        ("loglinear", {"beta": 0.05}),
        ("loglinear", {"beta": 0.0}),  # boundary: a flat zero-response curve is still valid
        ("logit", {"a": 1.0, "b": 0.0}),
        ("hill", {"ec50": 10.0, "emax": 1.0, "n": 2.0}),
        ("threshold_linear", {"slope": 0.1, "threshold": 5.0}),
        ("threshold_linear", {"slope": -0.1, "threshold": 5.0}),  # a negative slope is still finite
    ]
    doses = np.linspace(0.0, 50.0, 25)
    for model, params in cases:
        dr = DoseResponse(model=model, params=params)
        dq = dr.probability(doses, n=25, rng=np.random.default_rng(7))
        assert np.isfinite(dq.samples).all(), (model, params)
        assert (dq.samples >= 0.0).all() and (dq.samples <= 1.0).all(), (model, params)


def test_dose_response_output_gate_rejects_non_finite_dose():
    """MXR-080-0094: the final pushforward gate is defense-in-depth against a bad *dose*, independent
    of parameter validation -- valid coefficients cannot rescue a non-finite dose input."""
    dr = DoseResponse(model="loglinear", params={"beta": 0.05})
    with pytest.raises(ValueError):
        dr.probability(np.array([1.0, float("nan"), 3.0]), n=3, rng=np.random.default_rng(0))


@pytest.mark.parametrize("model, params", [
    ("loglinear", {"beta": 0.05}),
    ("logit", {"a": 1.0}),
    ("hill", {"ec50": 1.0}),
    ("threshold_linear", {"slope": 0.1}),
])
def test_dose_response_rejects_negative_dose_for_every_model(model, params):
    dr = DoseResponse(model=model, params=params)
    with pytest.raises(ValueError, match="nonnegative"):
        dr.probability(np.array([1.0, -1.0]), n=2, rng=np.random.default_rng(0))


def test_dose_response_rejects_implicit_bare_receptor_axis():
    dr = DoseResponse(model="logit", params={})
    with pytest.raises(ValueError, match="one-dimensional"):
        dr.probability(np.ones((2, 2)), n=2, rng=np.random.default_rng(0))


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_dose_response_rejects_invalid_draw_count_before_sampling(n):
    dr = DoseResponse(model="logit", params={})
    with pytest.raises(ValueError, match="n must"):
        dr.probability(1.0, n=n, rng=np.random.default_rng(0))


def test_dose_response_validates_posterior_draw_domain_and_axis():
    class InvalidExposurePosterior:
        @property
        def mean(self):
            return np.array([1.0])

        @property
        def cov(self):
            return np.array([[1.0]])

        def samples(self, n, rng):
            return np.full(n, -1.0)

        def credible_interval(self, level):
            return np.array([-1.0]), np.array([-1.0])

        def derived_quantity(self, fn, n, rng):
            pushed = fn(self.samples(n, rng))
            return type("Quantity", (), {"samples": pushed, "prior_dominated": False})()

    posterior = InvalidExposurePosterior()
    assert isinstance(posterior, Posterior)
    with pytest.raises(ValueError, match="nonnegative"):
        DoseResponse(model="logit", params={}).probability(
            posterior,
            n=4,
            rng=np.random.default_rng(0),
        )


def test_cumulative_exposure_rejects_invalid_time_step_decay_and_series():
    """MXR-080-0098: dt must be finite and positive, decay finite and non-negative, series finite."""
    series = np.array([1.0, 2.0, 3.0, 4.0])
    for bad_dt in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cumulative_exposure(series, bad_dt)
    for bad_decay in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cumulative_exposure(series, 1.0, decay=bad_decay)
    with pytest.raises(ValueError):
        cumulative_exposure(np.array([1.0, float("nan"), 3.0]), 1.0)
    with pytest.raises(ValueError):
        cumulative_exposure(np.array([1.0, float("inf"), 3.0]), 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        cumulative_exposure(np.array([-1.0, -2.0]), 1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        cumulative_exposure(np.ones((2, 2)), 1.0)


def test_cumulative_exposure_valid_inputs_unchanged():
    """Negative control for MXR-080-0098: legitimate dt/decay/series combinations still integrate
    correctly (also exercises the multi-element trapezoidal path end to end)."""
    series = np.array([1.0, 2.0, 3.0, 4.0])
    assert cumulative_exposure(series, 1.0) == pytest.approx(7.5)  # plain trapezoidal area
    assert cumulative_exposure(series, 1.0, decay=0.0) == pytest.approx(7.5)
    decayed = cumulative_exposure(series, 1.0, decay=0.5)
    assert np.isfinite(decayed)
    # decay discounts everything except the final timestep, so the decayed integral is strictly less
    # than the undiscounted one for a series that isn't degenerate.
    assert 0.0 < decayed < 7.5
    # boundary/degenerate sizes still work
    assert cumulative_exposure(np.array([]), 1.0) == 0.0
    assert cumulative_exposure(np.array([5.0]), 2.0) == pytest.approx(10.0)
