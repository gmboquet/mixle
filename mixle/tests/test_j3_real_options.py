"""J3 DoD -- real options & decision-under-uncertainty (notes/exec/workstream-J.md).

The Definition of Done asks for exactly two things about the deferral option:

1. Its value strictly exceeds the naive ``npv_dist.mean`` under high price volatility.
2. It collapses to ``~= max(npv_dist.mean, 0)`` as volatility -> 0.

Both are asserted below on the same underlying (positive-mean) project, plus a handful of supporting
checks on the rest of this task's public API (``OptionValue``'s shape, ``expand``/``abandon`` kinds,
and ``voi_dollars``' non-negativity) that aren't part of the DoD command itself but guard against
regressions in code this task also ships.

Repo-boundary note: J2's ``mixle.analysis.valuation.NPVDistribution`` had not landed on ``release/0.8.0``
as of this PR (see ``mixle/analysis/real_options.py``'s module docstring), so these tests build the
minimal duck-typed stand-in the real work order itself types as a forward reference (``"NPVDistribution"``)
rather than importing a class that doesn't exist yet on this branch.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pytest

from mixle.analysis.real_options import (
    GaussianObservationModel,
    OptionValue,
    VoiEstimate,
    real_option_value,
    voi_dollars,
    voi_estimate,
    voi_stopping_decision,
)


class _FakeNPVDistribution(NamedTuple):
    """Minimal duck-typed stand-in for J2's NPVDistribution -- only `.mean` is consumed."""

    samples: np.ndarray
    mean: float


def _npv_dist(mean: float, spread: float = 5.0, n: int = 2000, seed: int = 0) -> _FakeNPVDistribution:
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean, spread, size=n)
    return _FakeNPVDistribution(samples=samples, mean=float(mean))


def test_defer_option_exceeds_npv_under_high_volatility():
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=0.6, horizon=5, kind="defer", rate=0.05)
    assert isinstance(opt, OptionValue)
    assert opt.value > npv_dist.mean
    assert opt.premium_over_npv == pytest.approx(opt.value - npv_dist.mean)


def test_defer_option_collapses_to_naive_npv_as_volatility_to_zero():
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="defer", rate=0.05)
    assert opt.value == pytest.approx(max(npv_dist.mean, 0.0), abs=1e-6)


def test_defer_option_monotone_in_volatility():
    npv_dist = _npv_dist(mean=10.0)
    low = real_option_value(npv_dist, volatility=0.05, horizon=5, kind="defer", rate=0.05)
    high = real_option_value(npv_dist, volatility=0.6, horizon=5, kind="defer", rate=0.05)
    assert high.value > low.value


def test_defer_option_on_negative_mean_project_still_floors_at_zero_as_volatility_vanishes():
    npv_dist = _npv_dist(mean=-8.0)
    opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="defer", rate=0.05)
    assert opt.value == pytest.approx(0.0, abs=1e-6)
    # but with real volatility, the option to wait for an upswing is worth strictly more than 0
    opt_vol = real_option_value(npv_dist, volatility=0.6, horizon=5, kind="defer", rate=0.05)
    assert opt_vol.value > 0.0


def test_exercise_boundary_has_one_entry_per_lattice_step():
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=0.4, horizon=6, kind="defer", rate=0.05)
    assert opt.exercise_boundary.shape == (7,)


@pytest.mark.parametrize("kind", ["defer", "expand", "abandon"])
def test_all_kinds_run_and_return_option_value(kind):
    npv_dist = _npv_dist(mean=10.0)
    opt = real_option_value(npv_dist, volatility=0.3, horizon=4, kind=kind, rate=0.05)
    assert isinstance(opt, OptionValue)
    assert np.isfinite(opt.value)


def test_unknown_kind_raises():
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError):
        real_option_value(npv_dist, volatility=0.3, horizon=4, kind="bogus", rate=0.05)


def test_expand_option_collapses_to_its_own_intrinsic_at_zero_volatility():
    # At zero dispersion every kind must collapse to ITS OWN immediate-exercise payoff, not a
    # kind-independent max(mean, 0) -- "expand" pays mean + expand_fraction * max(mean, 0), strictly
    # more than the naive floor whenever mean > 0.
    npv_dist = _npv_dist(mean=100.0)
    opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="expand", rate=0.05, expand_fraction=0.3)
    expected = 100.0 + 0.3 * 100.0
    assert opt.value == pytest.approx(expected, abs=1e-6)
    assert opt.value != pytest.approx(max(npv_dist.mean, 0.0), abs=1e-3)


def test_expand_option_collapses_to_its_own_intrinsic_at_zero_horizon():
    # horizon=0 forces dt=0 -> h=0 through the SAME branch, independent of volatility.
    npv_dist = _npv_dist(mean=100.0)
    opt = real_option_value(npv_dist, volatility=0.6, horizon=0, kind="expand", rate=0.05, expand_fraction=0.3)
    assert opt.value == pytest.approx(130.0, abs=1e-6)


def test_defer_and_abandon_zero_volatility_intrinsic_is_unchanged():
    # Regression guard: the fix must not change the already-correct defer/abandon zero-vol value.
    npv_dist = _npv_dist(mean=10.0)
    for kind in ("defer", "abandon"):
        opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind=kind, rate=0.05)
        assert opt.value == pytest.approx(max(npv_dist.mean, 0.0), abs=1e-6)


@pytest.mark.parametrize("bad_n_steps", [0, -1, -5])
def test_non_positive_n_steps_raises_a_clear_error(bad_n_steps):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="n_steps"):
        real_option_value(npv_dist, volatility=0.3, horizon=5, kind="defer", rate=0.05, n_steps=bad_n_steps)


# --- MXR-080-0110: exact-finite controls + expansion/salvage exercise economics ---


@pytest.mark.parametrize("bad_rate", [float("nan"), float("inf"), float("-inf")])
def test_real_option_value_rejects_non_finite_rate(bad_rate):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="rate"):
        real_option_value(npv_dist, volatility=0.3, horizon=5, kind="defer", rate=bad_rate)


@pytest.mark.parametrize("bad_volatility", [float("nan"), float("inf")])
def test_real_option_value_rejects_non_finite_volatility(bad_volatility):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="volatility"):
        real_option_value(npv_dist, volatility=bad_volatility, horizon=5, kind="defer", rate=0.05)


@pytest.mark.parametrize("bad_horizon", [float("nan"), float("inf")])
def test_real_option_value_rejects_non_finite_horizon(bad_horizon):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="horizon"):
        real_option_value(npv_dist, volatility=0.3, horizon=bad_horizon, kind="defer", rate=0.05)


@pytest.mark.parametrize("bad_mean", [float("nan"), float("inf"), float("-inf")])
def test_real_option_value_rejects_non_finite_npv_mean(bad_mean):
    npv_dist = _FakeNPVDistribution(samples=np.array([]), mean=bad_mean)
    with pytest.raises(ValueError, match="npv_dist.mean"):
        real_option_value(npv_dist, volatility=0.3, horizon=5, kind="defer", rate=0.05)


@pytest.mark.parametrize("bad_expand_fraction", [float("nan"), float("inf")])
def test_real_option_value_rejects_non_finite_expand_fraction(bad_expand_fraction):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="expand_fraction"):
        real_option_value(
            npv_dist, volatility=0.3, horizon=5, kind="expand", rate=0.05, expand_fraction=bad_expand_fraction
        )


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf")])
def test_real_option_value_rejects_non_finite_expansion_cost(bad_cost):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="expansion_cost"):
        real_option_value(npv_dist, volatility=0.3, horizon=5, kind="expand", rate=0.05, expansion_cost=bad_cost)


@pytest.mark.parametrize("bad_salvage", [float("nan"), float("inf")])
def test_real_option_value_rejects_non_finite_salvage_value(bad_salvage):
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="salvage_value"):
        real_option_value(npv_dist, volatility=0.3, horizon=5, kind="abandon", rate=0.05, salvage_value=bad_salvage)


@pytest.mark.parametrize("bad_horizon", [5.5, 2.1])
def test_real_option_value_rejects_fractional_horizon(bad_horizon):
    """Fractional horizons used to be silently truncated via int(...); they must now be rejected --
    the caller has to decide whether they meant e.g. 5 or 6 periods, not have it picked for them."""
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="horizon must be an exact integer"):
        real_option_value(npv_dist, volatility=0.3, horizon=bad_horizon, kind="defer", rate=0.05)


def test_real_option_value_rejects_fractional_n_steps():
    npv_dist = _npv_dist(mean=10.0)
    with pytest.raises(ValueError, match="n_steps must be an exact integer"):
        real_option_value(npv_dist, volatility=0.3, horizon=5, kind="defer", rate=0.05, n_steps=5.5)


def test_real_option_value_accepts_a_whole_number_float_horizon():
    """Negative control: a whole-number float (e.g. 5.0) is not the silently-truncated fraction the fix
    targets, so it must still be accepted and price identically to the equivalent int."""
    npv_dist = _npv_dist(mean=10.0)
    as_int = real_option_value(npv_dist, volatility=0.3, horizon=5, kind="defer", rate=0.05)
    as_float = real_option_value(npv_dist, volatility=0.3, horizon=5.0, kind="defer", rate=0.05)
    assert as_int.value == as_float.value


def test_expansion_cost_prevents_the_capacity_bonus_when_it_exceeds_the_gain():
    """Without an expansion investment cost, a positive-NPV project always got the capacity bonus for
    free -- there was no actual exercise decision being priced. A cost that exceeds the bonus (0.3 * 100
    = 30) must suppress it entirely: the option collapses back to just the base project value."""
    npv_dist = _npv_dist(mean=100.0)
    free_bonus = real_option_value(
        npv_dist, volatility=1e-9, horizon=5, kind="expand", rate=0.05, expand_fraction=0.3, expansion_cost=0.0
    )
    suppressed = real_option_value(
        npv_dist, volatility=1e-9, horizon=5, kind="expand", rate=0.05, expand_fraction=0.3, expansion_cost=50.0
    )
    assert free_bonus.value == pytest.approx(130.0, abs=1e-6)
    assert suppressed.value == pytest.approx(100.0, abs=1e-6)
    assert suppressed.value < free_bonus.value


def test_expansion_cost_also_suppresses_the_bonus_under_real_volatility():
    """Same suppression, now through the actual lattice (not just the zero-volatility intrinsic
    shortcut) -- an expansion too expensive to ever be worth exercising must not trivially pay off
    through American-exercise backward induction either."""
    npv_dist = _npv_dist(mean=100.0)
    free_bonus = real_option_value(
        npv_dist, volatility=0.4, horizon=5, kind="expand", rate=0.05, expand_fraction=0.3, expansion_cost=0.0
    )
    suppressed = real_option_value(
        npv_dist, volatility=0.4, horizon=5, kind="expand", rate=0.05, expand_fraction=0.3, expansion_cost=1000.0
    )
    assert suppressed.value < free_bonus.value


def test_expansion_cost_defaults_to_zero_matching_previous_behavior():
    """Negative control: a legitimate finite setup left at the new parameters' defaults
    (expansion_cost=0.0) must still price exactly as it did before this fix existed."""
    npv_dist = _npv_dist(mean=100.0)
    opt = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="expand", rate=0.05, expand_fraction=0.3)
    assert opt.value == pytest.approx(130.0, abs=1e-6)


def test_salvage_value_raises_the_abandon_floor_above_the_previous_hardcoded_zero():
    """'abandon' previously hardcoded a total write-off (payoff exactly 0 when walking away from a
    negative-NPV project). A caller who can recover salvage should see the option value reflect that."""
    npv_dist = _npv_dist(mean=-40.0)
    write_off = real_option_value(npv_dist, volatility=1e-9, horizon=5, kind="abandon", rate=0.05, salvage_value=0.0)
    with_salvage = real_option_value(
        npv_dist, volatility=1e-9, horizon=5, kind="abandon", rate=0.05, salvage_value=15.0
    )
    assert write_off.value == pytest.approx(0.0, abs=1e-6)
    assert with_salvage.value == pytest.approx(15.0, abs=1e-6)


class _ToyPosterior:
    """A minimal IC-1-conforming posterior: an independent Gaussian belief over one grade parameter."""

    def __init__(self, mean: float, std: float):
        self._mean = mean
        self._std = std

    def samples(self, n, rng):
        return rng.normal(self._mean, self._std, size=(n, 1))

    @property
    def mean(self):
        return np.array([self._mean])

    @property
    def cov(self):
        return np.array([[self._std**2]])

    def credible_interval(self, level):
        z = self._std * 1.6448536269514722  # ~ inverse-CDF fudge, not exercised by this test
        return self.mean - z, self.mean + z

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError


def _decision_value(samples: np.ndarray) -> float:
    """Toy decision rule: a single risk-neutral go/no-go choice, priced at the belief's mean."""
    return float(max(np.mean(samples[:, 0]), 0.0))


def _heuristic_info(**values):
    return {"method": "variance_rescaling_heuristic", **values}


def test_voi_requires_an_observation_model_by_default():
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    with pytest.raises(ValueError, match="observation_model"):
        voi_dollars(posterior, _decision_value, {"variance_reduction": 0.7}, rng=np.random.default_rng(0))


def test_voi_dollars_is_non_negative():
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    rng = np.random.default_rng(0)
    voi = voi_dollars(posterior, _decision_value, _heuristic_info(variance_reduction=0.7), rng=rng)
    assert voi >= 0.0


def test_voi_dollars_grows_with_variance_reduction():
    rng = np.random.default_rng(0)
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    voi_small = voi_dollars(
        posterior, _decision_value, _heuristic_info(variance_reduction=0.1), rng=np.random.default_rng(1)
    )
    voi_large = voi_dollars(
        posterior, _decision_value, _heuristic_info(variance_reduction=0.9), rng=np.random.default_rng(1)
    )
    assert voi_large >= voi_small


def test_zero_information_voi_is_exactly_zero_across_seeds():
    """MXR-080-0108's exact repro: a standard-normal fake posterior and a go/no-go decision at
    ``variance_reduction=0``. Before the common-random-numbers fix, the no-info and with-info sides were
    estimated from INDEPENDENT Monte Carlo draws even though they should coincide exactly at zero
    information, so their noisy difference -- floored at zero -- spuriously reported a positive VOI for
    most seeds (including 0.118). With a shared base draw per replicate, the two sides are bit-for-bit
    identical at r=0, so this must hold EXACTLY (not just "close to zero") for every seed tried."""
    posterior = _ToyPosterior(mean=0.0, std=1.0)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        voi = voi_dollars(posterior, _decision_value, _heuristic_info(variance_reduction=0.0), rng=rng)
        assert voi == 0.0, f"seed={seed}: expected exactly 0.0 at zero information, got {voi!r}"


def test_voi_dollars_is_no_longer_upward_biased_near_zero_information():
    """Negative control proving the zero floor is really gone, not just short-circuited exactly at
    r=0: with a tiny (but nonzero) variance reduction, the true VOI signal is small relative to Monte
    Carlo noise, so an HONEST (unfloored) estimator should land on both sides of zero across seeds. If
    every seed still came back >= 0.0, that would mean some upward bias is still silently in effect."""
    posterior = _ToyPosterior(mean=0.0, std=1.0)
    values = [
        voi_dollars(
            posterior,
            _decision_value,
            _heuristic_info(variance_reduction=1e-6),
            rng=np.random.default_rng(seed),
        )
        for seed in range(40)
    ]
    assert any(v < 0.0 for v in values), f"expected at least one negative estimate, got {values!r}"
    assert any(v > 0.0 for v in values), f"expected at least one positive estimate, got {values!r}"


def test_voi_estimate_at_zero_information_has_zero_uncertainty_too():
    posterior = _ToyPosterior(mean=0.0, std=1.0)
    rng = np.random.default_rng(0)
    est = voi_estimate(posterior, _decision_value, _heuristic_info(variance_reduction=0.0), rng=rng)
    assert isinstance(est, VoiEstimate)
    assert (est.value, est.standard_error, est.ci_low, est.ci_high) == (0.0, 0.0, 0.0, 0.0)
    assert est.method == "variance_rescaling_heuristic"


def test_voi_estimate_reports_monte_carlo_uncertainty_away_from_zero_information():
    """Away from zero information, the estimate comes with a nonzero standard error and a CI that -- for
    a clearly-informative reduction on a wide, mean-away-from-zero posterior -- excludes zero. This is
    the honest replacement for the old zero floor: a caller can tell a real effect apart from noise
    instead of trusting a single (previously upward-biased) number."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    rng = np.random.default_rng(0)
    est = voi_estimate(posterior, _decision_value, _heuristic_info(variance_reduction=0.7), rng=rng)
    assert est.standard_error > 0.0
    assert est.ci_low == pytest.approx(est.value - 1.959963984540054 * est.standard_error)
    assert est.ci_high == pytest.approx(est.value + 1.959963984540054 * est.standard_error)
    assert est.ci_low < est.value < est.ci_high
    assert 0.0 < est.ci_low, "expected a clearly-informative reduction to exclude zero from the CI"
    assert est.method == "variance_rescaling_heuristic"


def test_voi_dollars_matches_voi_estimate_value():
    """voi_dollars must not silently compute something different from voi_estimate's point estimate."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    drill_info = _heuristic_info(variance_reduction=0.4)
    direct = voi_dollars(posterior, _decision_value, drill_info, rng=np.random.default_rng(3))
    est = voi_estimate(posterior, _decision_value, drill_info, rng=np.random.default_rng(3))
    assert direct == est.value


# --- MXR-080-0109: genuine sample-then-condition EVSI under a declared GaussianObservationModel ---


class _BimodalToyPosterior:
    """A genuinely bimodal (two well-separated Gaussian bumps) belief -- clearly not the Gaussian,
    unimodal regime the variance-rescaling heuristic (and the Gaussian-conjugate observation_model path)
    both assume; used to exercise the honest regime warning."""

    def __init__(self, separation: float = 12.0, bump_std: float = 1.0):
        self._separation = separation
        self._bump_std = bump_std

    def samples(self, n, rng):
        which = rng.integers(0, 2, size=n)
        half = self._separation / 2.0
        out = np.where(
            which == 0,
            rng.normal(-half, self._bump_std, size=n),
            rng.normal(half, self._bump_std, size=n),
        )
        return out[:, None]

    @property
    def mean(self):
        return np.array([0.0])

    @property
    def cov(self):
        return np.array([[(self._separation / 2.0) ** 2 + self._bump_std**2]])

    def credible_interval(self, level):
        raise NotImplementedError

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError


def test_observation_model_zero_information_is_exactly_zero_across_seeds():
    """The same zero-information exactness property as the heuristic path (MXR-080-0108's
    common-random-numbers fix), now for the genuine EVSI path: an all-zero obs_matrix declares a
    literally uninformative experiment, so VOI must come back exactly 0.0, not just approximately."""
    posterior = _ToyPosterior(mean=0.5, std=3.0)
    uninformative = GaussianObservationModel(obs_matrix=np.array([[0.0]]), obs_cov=np.array([[1.0]]))
    for seed in range(10):
        rng = np.random.default_rng(seed)
        voi = voi_dollars(posterior, _decision_value, {}, rng=rng, observation_model=uninformative)
        assert voi == 0.0, f"seed={seed}: expected exactly 0.0 for an uninformative observation, got {voi!r}"


def test_observation_model_reports_gaussian_conjugate_evsi_method():
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    om = GaussianObservationModel(obs_matrix=np.array([[1.0]]), obs_cov=np.array([[4.0]]))
    est = voi_estimate(posterior, _decision_value, {}, rng=np.random.default_rng(0), observation_model=om)
    assert est.method == "gaussian_conjugate_evsi"
    assert est.standard_error > 0.0


def test_observation_model_matches_closed_form_gaussian_conjugate_voi():
    """Verified against a known closed form: for a normal-normal conjugate pair (prior N(mu0, tau0^2),
    a single noisy observation of variance sigma^2) and a go/no-go decision "invest iff the posterior
    mean is positive", the expected with-info value is E[max(mu1(Y), 0)] for mu1(Y) ~ N(mu0, tau0^2 -
    tau1^2) (tau1^2 the posterior variance; Var(mu1(Y)) = tau0^2 - tau1^2 is the standard law-of-total-
    variance identity). E[max(X, 0)] for X ~ N(m, v) has the closed form m*Phi(m/s) + s*phi(m/s), s =
    sqrt(v) (the Bachelier/rectified-normal formula) -- so VOI = that minus max(mu0, 0), independent of
    this module's own Monte Carlo machinery entirely."""
    from scipy.stats import norm

    mu0, tau0, sigma = 0.5, 3.0, 2.0
    tau1_sq = 1.0 / (1.0 / tau0**2 + 1.0 / sigma**2)
    spread = float(np.sqrt(tau0**2 - tau1_sq))
    closed_form = (mu0 * norm.cdf(mu0 / spread) + spread * norm.pdf(mu0 / spread)) - max(mu0, 0.0)

    posterior = _ToyPosterior(mean=mu0, std=tau0)
    om = GaussianObservationModel(obs_matrix=np.array([[1.0]]), obs_cov=np.array([[sigma**2]]))
    est = voi_estimate(
        posterior, _decision_value, {}, rng=np.random.default_rng(42), n_outer=4000, n_inner=4000, observation_model=om
    )
    # Generous multiple of the reported standard error plus a small slack for decision_fn's own residual
    # finite-n_inner bias (it estimates max(posterior mean, 0) via a sample mean, not the exact mean).
    assert abs(est.value - closed_form) < 4.0 * est.standard_error + 0.02


def test_observation_model_rejects_mismatched_obs_matrix_shape():
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    bad = GaussianObservationModel(obs_matrix=np.array([[1.0, 1.0]]), obs_cov=np.array([[1.0]]))
    with pytest.raises(ValueError, match="obs_matrix"):
        voi_dollars(posterior, _decision_value, {}, rng=np.random.default_rng(0), observation_model=bad)


def test_observation_model_rejects_non_finite_obs_cov():
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    bad = GaussianObservationModel(obs_matrix=np.array([[1.0]]), obs_cov=np.array([[float("nan")]]))
    with pytest.raises(ValueError, match="finite"):
        voi_dollars(posterior, _decision_value, {}, rng=np.random.default_rng(0), observation_model=bad)


def test_observation_model_rejects_matrix_free_covariance():
    from scipy.sparse.linalg import LinearOperator

    class _LinOpPosterior(_ToyPosterior):
        @property
        def cov(self):
            dense = super().cov
            return LinearOperator(dense.shape, matvec=lambda x: dense @ x)

    posterior = _LinOpPosterior(mean=1.0, std=5.0)
    om = GaussianObservationModel(obs_matrix=np.array([[1.0]]), obs_cov=np.array([[1.0]]))
    with pytest.raises(TypeError, match="dense"):
        voi_dollars(posterior, _decision_value, {}, rng=np.random.default_rng(0), observation_model=om)


def test_heuristic_warns_on_a_clearly_non_gaussian_posterior():
    posterior = _BimodalToyPosterior()
    with pytest.warns(UserWarning, match="non-Gaussian"):
        voi_dollars(
            posterior, _decision_value, _heuristic_info(variance_reduction=0.5), rng=np.random.default_rng(0)
        )


def test_observation_model_warns_on_a_clearly_non_gaussian_posterior():
    posterior = _BimodalToyPosterior()
    om = GaussianObservationModel(obs_matrix=np.array([[1.0]]), obs_cov=np.array([[1.0]]))
    with pytest.warns(UserWarning, match="non-Gaussian"):
        voi_dollars(posterior, _decision_value, {}, rng=np.random.default_rng(0), observation_model=om)


def test_heuristic_does_not_warn_on_a_genuinely_gaussian_posterior():
    """Negative control: the regime screen must not cry wolf on the exact regime it is meant to allow --
    checked across several seeds since it is itself a (generous-threshold) statistical test."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    for seed in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            voi_dollars(
                posterior,
                _decision_value,
                _heuristic_info(variance_reduction=0.5),
                rng=np.random.default_rng(seed),
            )


def test_real_option_value_type_hints_are_resolvable():
    # npv_dist's annotation used to live behind a TYPE_CHECKING-only import, so typing.get_type_hints
    # (and any other runtime introspection) raised NameError even though NPVDistribution has been a
    # real, importable module for a while now.
    import typing

    from mixle.analysis.real_options import real_option_value
    from mixle.analysis.valuation import NPVDistribution

    hints = typing.get_type_hints(real_option_value)
    assert hints["npv_dist"] is NPVDistribution


def test_voi_stopping_decision_says_keep_sampling_when_voi_exceeds_a_cheap_cost():
    """A real decision-theoretic replacement for the arbitrary CI-width thresholds hand-picked in
    experiments/adaptive-groundwater-monitoring and experiments/adaptive-gravity-survey-design: a
    wide, uncertain posterior with an informative (high variance-reduction) next sample and a cheap
    sample cost should say to keep sampling."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    rng = np.random.default_rng(0)
    decision = voi_stopping_decision(
        posterior,
        _decision_value,
        _heuristic_info(variance_reduction=0.8),
        sample_cost=0.01,
        rng=rng,
    )
    assert decision.voi_dollars > 0.0
    assert decision.keep_sampling is True
    assert decision.net_value == pytest.approx(decision.voi_dollars - 0.01)


def test_voi_stopping_decision_says_stop_when_the_sample_costs_more_than_it_is_worth():
    """A tight, already-confident posterior (tiny std, small variance-reduction left to gain) against
    an expensive next sample should say to stop -- the mirror case of the test above."""
    posterior = _ToyPosterior(mean=1.0, std=0.05)
    rng = np.random.default_rng(0)
    decision = voi_stopping_decision(
        posterior,
        _decision_value,
        _heuristic_info(variance_reduction=0.05),
        sample_cost=1_000_000.0,
        rng=rng,
    )
    assert decision.keep_sampling is False
    assert decision.net_value < 0.0


def test_voi_stopping_decision_is_consistent_with_voi_dollars_directly():
    """The wrapper must not silently compute something different from voi_dollars itself."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    drill_info = _heuristic_info(variance_reduction=0.6)
    direct = voi_dollars(posterior, _decision_value, drill_info, rng=np.random.default_rng(7))
    decision = voi_stopping_decision(
        posterior, _decision_value, drill_info, sample_cost=0.0, rng=np.random.default_rng(7)
    )
    assert decision.voi_dollars == pytest.approx(direct)


@pytest.mark.parametrize("bad_sample_cost", [float("nan"), float("inf"), float("-inf")])
def test_voi_stopping_decision_rejects_non_finite_sample_cost(bad_sample_cost):
    """A NaN sample_cost previously compared as False against everything (NaN > x and NaN < x are both
    False), so keep_sampling silently came back False regardless of the actual VOI -- must be a clear
    error instead."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    with pytest.raises(ValueError, match="sample_cost"):
        voi_stopping_decision(
            posterior,
            _decision_value,
            _heuristic_info(variance_reduction=0.5),
            sample_cost=bad_sample_cost,
            rng=np.random.default_rng(0),
        )


def test_voi_stopping_decision_reports_standard_error():
    """Negative control alongside the rejection test above: a legitimate finite sample_cost still prices
    a real decision, now carrying the VOI estimate's own Monte Carlo standard error."""
    posterior = _ToyPosterior(mean=1.0, std=5.0)
    decision = voi_stopping_decision(
        posterior,
        _decision_value,
        _heuristic_info(variance_reduction=0.5),
        sample_cost=0.01,
        rng=np.random.default_rng(0),
    )
    assert decision.standard_error > 0.0
    assert np.isfinite(decision.voi_dollars)
