"""K9 DoD -- epidemiological cohort attribution (notes/exec/workstream-K.md).

The Definition of Done asks for one concrete scenario: on a simulated cohort with a known exposure ->
outcome hazard ratio (``HR_true``) and right-censoring, ``cohort_attribution`` recovers ``hazard_ratio``
within 15% of truth, ``hr_ci`` covers ``HR_true``, and ``af_ci`` covers the implied true attributable
fraction ``(HR_true - 1) / HR_true``. The remaining tests exercise the other DoD clauses: CI coverage
tracking the nominal rate across repeated seeds, competing-risks CIF validity, and the IC-1
``DerivedQuantity``-shaped bootstrap summary in ``provenance``.

Cohort sizes and ``n_boot`` are kept as small as the assertions tolerate -- each bootstrap draw is a
full `cox_ph` refit, so this file's total cost is ``n_boot`` Cox fits, not one; seeds below were checked
to give a comfortable margin against their tolerance (not cherry-picked to the edge) so the file stays
in the fast gate instead of ballooning it (see conftest.py's fast/slow triage policy).
"""

from __future__ import annotations

import numpy as np
import pytest

import mixle.analysis.epidemiology as epidemiology_module
from mixle.analysis.epidemiology import CohortAttribution, cohort_attribution
from mixle.reason.posterior_protocol import DerivedQuantity

HR_TRUE = 2.0


def _simulate_cohort(seed: int, *, n: int = 300, hr_true: float = HR_TRUE, p_exposed: float = 0.5):
    """A one-covariate (binary exposure) proportional-hazards cohort with exponential censoring.

    Same construction as `mixle/tests/survival_regression_test.py`'s `CoxTest._sim`: event time
    ``T = -log(U) / exp(x @ beta)`` gives exactly the proportional-hazards model `cox_ph` assumes, so
    `beta = log(hr_true)` on the exposure column is the ground truth `cohort_attribution` should recover.
    """
    rng = np.random.default_rng(seed)
    exposed = (rng.random(n) < p_exposed).astype(float)
    covariates = exposed.reshape(-1, 1)
    beta = np.log(hr_true)
    event_time = -np.log(rng.random(n)) / np.exp(covariates[:, 0] * beta)
    censor_time = rng.exponential(3.0, n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return covariates, time, event


def test_known_hazard_recovered():
    covariates, time, event = _simulate_cohort(seed=0)

    result = cohort_attribution(covariates, time, event, exposure_col=0, n_boot=100, rng=0)

    assert isinstance(result, CohortAttribution)

    # (1) hazard ratio within 15% of truth
    assert result.hazard_ratio == pytest.approx(HR_TRUE, rel=0.15)

    # (2) hr_ci covers HR_true
    hr_lo, hr_hi = result.hr_ci
    assert hr_lo <= HR_TRUE <= hr_hi

    # (3) af_ci covers the implied true attributable fraction
    af_true = (HR_TRUE - 1.0) / HR_TRUE
    af_lo, af_hi = result.af_ci
    assert af_lo <= af_true <= af_hi
    assert result.attributable_fraction == pytest.approx(af_true, rel=0.2)

    # no competing risks requested -> cif is the empty dict, not a stub
    assert result.cif == {}


def test_af_distribution_is_ic1_derived_quantity_shaped():
    covariates, time, event = _simulate_cohort(seed=1)
    result = cohort_attribution(covariates, time, event, n_boot=80, rng=1)

    dq = result.provenance["af_distribution"]
    assert isinstance(dq, DerivedQuantity)
    assert dq.prior_dominated is False
    lo, hi = dq.credible_interval(0.95)
    assert lo <= hi
    # `.samples` holds exactly the draws that converged -- not necessarily all `n_boot` of them (a clean
    # simulated cohort like this one should converge every time, but the *invariant* being tested is the
    # count matching `n_boot_valid`, not happening to equal the requested `n_boot`; see MXR-080-0089).
    assert np.all(np.isfinite(dq.samples))
    assert dq.samples.shape == (result.provenance["n_boot_valid"],)
    assert dq.samples.shape[0] <= 80

    # every number attributes to the fit + the RNG's full reproducible state (not just a seed -- see
    # MXR-080-0089: a bare seed cannot replay draws from a generator already advanced before this call)
    assert isinstance(result.provenance["rng_state"], dict)
    assert result.provenance["rng_state"]["bit_generator"] == "PCG64"
    assert result.provenance["n"] == covariates.shape[0]
    assert result.provenance["ties"] == "efron"


def test_ci_coverage_tracks_nominal_rate_across_seeds():
    # A light-weight coverage check: 15 independent cohorts, small bootstrap (speed), counting how
    # often the 95% hr_ci actually covers HR_true. With only 15 replicates the count is noisy, but it
    # should be solidly majority-covering, not degenerate.
    n_reps = 15
    covered = 0
    for seed in range(n_reps):
        covariates, time, event = _simulate_cohort(seed=100 + seed, n=250)
        result = cohort_attribution(covariates, time, event, n_boot=40, rng=seed)
        lo, hi = result.hr_ci
        covered += int(lo <= HR_TRUE <= hi)
    coverage_rate = covered / n_reps
    assert coverage_rate >= 0.6, f"95% CI coverage collapsed to {coverage_rate:.2f} over {n_reps} reps"


def test_competing_risks_cif_nondecreasing_and_bounded():
    rng = np.random.default_rng(2)
    n = 300
    exposed = (rng.random(n) < 0.5).astype(float)
    covariates = exposed.reshape(-1, 1)
    beta = np.log(HR_TRUE)
    t_cause1 = -np.log(rng.random(n)) / np.exp(covariates[:, 0] * beta)
    t_cause2 = rng.exponential(2.5, n)  # competing cause, unaffected by exposure
    censor = rng.exponential(3.0, n)

    time = np.minimum(np.minimum(t_cause1, t_cause2), censor)
    event = np.zeros(n, dtype=int)
    event[(t_cause1 <= t_cause2) & (t_cause1 <= censor)] = 1
    event[(t_cause2 < t_cause1) & (t_cause2 <= censor)] = 2

    result = cohort_attribution(covariates, time, event, competing=True, n_boot=40, rng=2)

    assert set(result.cif.keys()) == {1, 2}
    total = np.zeros_like(next(iter(result.cif.values())))
    for curve in result.cif.values():
        assert np.all(np.diff(curve) >= -1e-12), "CIF must be non-decreasing"
        assert np.all(curve >= 0.0)
        total = total + curve
    assert np.all(total <= 1.0 + 1e-9), "cause-specific CIFs must not sum past 1"


def test_latency_left_truncates_the_risk_set():
    covariates, time, event = _simulate_cohort(seed=3, n=350)
    result = cohort_attribution(covariates, time, event, latency=0.1, n_boot=30, rng=3)

    assert np.isfinite(result.hazard_ratio)
    assert result.provenance["latency"] == 0.1
    assert result.provenance["n"] == covariates.shape[0]
    # left-truncation drops subjects who never survive past the latency window
    assert result.provenance["n_fit_rows"] <= covariates.shape[0]
    assert np.any(time <= 0.1), "test setup should include some subjects truncated before latency"


# --------------------------------------------------------------------------- MXR-080-0088: cohort validation
# No cross-array shape, finiteness, time, event-code, latency, or exposure-column validation ran before
# fitting/bootstrapping. `_validate_cohort` now runs once, before any of that, and rejects each of these
# outright instead of silently corrupting a fit or crashing deep inside `cox_ph`.


def test_fractional_event_code_is_rejected():
    # 1.5 would truncate to 1 (the event of interest) under the old `.astype(int)` cast -- silently
    # recording a fractional/unknown code as a real outcome.
    covariates, time, event = _simulate_cohort(seed=4, n=50)
    event = event.copy()
    event[0] = 1.5
    with pytest.raises(ValueError, match="exact integer"):
        cohort_attribution(covariates, time, event, n_boot=10, rng=4)


def test_fractional_event_code_toward_competing_cause_is_rejected():
    # 2.5 would truncate to 2 (a competing cause) under the old cast -- a *different* silent mislabel
    # than the 1.5 case above: exactly the "fractional causes can become censoring or another cause"
    # failure mode the finding describes.
    covariates, time, event = _simulate_cohort(seed=4, n=50)
    event = event.copy()
    event[0] = 2.5
    with pytest.raises(ValueError, match="exact integer"):
        cohort_attribution(covariates, time, event, competing=True, n_boot=10, rng=4)


def test_negative_latency_is_rejected():
    # Previously: `_fit_lagged`'s `latency > 0` check is also False for negative latency, so it silently
    # fell through to the "no latency" branch instead of being rejected.
    covariates, time, event = _simulate_cohort(seed=5, n=50)
    with pytest.raises(ValueError, match="non-negative"):
        cohort_attribution(covariates, time, event, latency=-0.5, n_boot=10, rng=5)


def test_mismatched_array_lengths_are_rejected():
    covariates, time, event = _simulate_cohort(seed=6, n=50)
    with pytest.raises(ValueError, match="shape"):
        cohort_attribution(covariates, time[:-5], event, n_boot=10, rng=6)


def test_non_finite_covariates_are_rejected():
    covariates, time, event = _simulate_cohort(seed=7, n=50)
    covariates = covariates.copy()
    covariates[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cohort_attribution(covariates, time, event, n_boot=10, rng=7)


def test_non_finite_time_is_rejected():
    covariates, time, event = _simulate_cohort(seed=8, n=50)
    time = time.copy()
    time[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        cohort_attribution(covariates, time, event, n_boot=10, rng=8)


def test_negative_time_is_rejected():
    covariates, time, event = _simulate_cohort(seed=9, n=50)
    time = time.copy()
    time[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        cohort_attribution(covariates, time, event, n_boot=10, rng=9)


def test_out_of_vocabulary_event_code_without_competing_is_rejected():
    # 2 is a legitimate competing-risk code, but competing=False declares the binary {0, 1} contract; it
    # must not be silently accepted (or silently cause-1-censored).
    covariates, time, event = _simulate_cohort(seed=10, n=50)
    event = event.copy()
    event[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        cohort_attribution(covariates, time, event, competing=False, n_boot=10, rng=10)


def test_exposure_col_out_of_bounds_is_rejected():
    covariates, time, event = _simulate_cohort(seed=11, n=50)
    with pytest.raises(ValueError, match="exposure_col"):
        cohort_attribution(covariates, time, event, exposure_col=5, n_boot=10, rng=11)


def test_covariates_with_too_many_dimensions_are_rejected():
    covariates, time, event = _simulate_cohort(seed=13, n=20)
    covariates_3d = covariates.reshape(20, 1, 1)
    with pytest.raises(ValueError, match="2-D"):
        cohort_attribution(covariates_3d, time, event, n_boot=10, rng=13)


def test_well_formed_cohort_still_fits_normally_after_validation():
    # Negative control: boundary-legal values (time == 0, latency == 0, event in {0, 1}) must NOT be
    # rejected by the new validation -- it should reject exactly the malformed cases above, nothing more.
    covariates, time, event = _simulate_cohort(seed=12, n=200)
    time = time.copy()
    time[0] = 0.0
    result = cohort_attribution(covariates, time, event, latency=0.0, n_boot=50, rng=12)
    assert isinstance(result, CohortAttribution)
    assert np.isfinite(result.hazard_ratio)
    assert np.isfinite(result.attributable_fraction)


# --------------------------------------------------------------------------- MXR-080-0089: bootstrap evidence
# A failed draw stayed NaN in the published `_AFDistribution.samples`, one converged draw was enough to
# report a full interval, and the recorded "seed" could not actually reproduce the bootstrap.


def test_failed_bootstrap_draws_are_excluded_from_samples_not_nan_filled(monkeypatch):
    covariates, time, event = _simulate_cohort(seed=24, n=200)
    real_fit_lagged = epidemiology_module._fit_lagged
    state = {"calls": 0}

    def _partially_flaky_fit_lagged(x, t, e, latency):
        state["calls"] += 1
        # Call 1 is the main-cohort fit (must succeed, or `cohort_attribution` never reaches the
        # bootstrap loop); of the bootstrap draws that follow, every 3rd one fails -- a controlled,
        # deterministic ~33% failure rate that still clears the adequate-evidence threshold at
        # n_boot=100 (~67 valid draws expected), isolating NaN-exclusion (0089 part A) from the separate
        # insufficient-evidence threshold (0089 part B, exercised below).
        if state["calls"] > 1 and state["calls"] % 3 == 0:
            raise np.linalg.LinAlgError("simulated degenerate resample")
        return real_fit_lagged(x, t, e, latency)

    monkeypatch.setattr(epidemiology_module, "_fit_lagged", _partially_flaky_fit_lagged)

    result = cohort_attribution(covariates, time, event, n_boot=100, rng=24)

    dq = result.provenance["af_distribution"]
    assert isinstance(dq, DerivedQuantity), "partial failures should still clear the adequate-evidence threshold"
    assert np.all(np.isfinite(dq.samples)), "failed draws must be excluded, never carried through as NaN"
    assert 0 < dq.samples.shape[0] < result.provenance["n_boot"]
    assert dq.samples.shape[0] == result.provenance["n_boot_valid"]


def test_mostly_failing_bootstrap_returns_insufficient_evidence_not_a_spurious_interval(monkeypatch):
    covariates, time, event = _simulate_cohort(seed=21, n=200)
    real_fit_lagged = epidemiology_module._fit_lagged
    state = {"calls": 0}

    def _flaky_fit_lagged(x, t, e, latency):
        state["calls"] += 1
        # Call 1 (the main-cohort fit) and every 25th bootstrap draw after it succeed -- a ~4% bootstrap
        # convergence rate, deterministic regardless of the exact adequacy threshold chosen.
        if state["calls"] % 25 == 1:
            return real_fit_lagged(x, t, e, latency)
        raise np.linalg.LinAlgError("simulated degenerate resample")

    monkeypatch.setattr(epidemiology_module, "_fit_lagged", _flaky_fit_lagged)

    result = cohort_attribution(covariates, time, event, n_boot=100, rng=21)

    dq = result.provenance["af_distribution"]
    assert not isinstance(dq, DerivedQuantity), "an insufficient-evidence result must not pass as a real interval"
    assert dq.n_boot == 100
    assert dq.n_boot_valid < 40
    assert dq.reason
    assert all(np.isnan(v) for v in result.af_ci), "af_ci must not report a spuriously precise interval"

    # the point estimate (from the ONE always-succeeding main-cohort fit) is unaffected -- the HR fit and
    # the bootstrap evidence for its interval are separable failure modes.
    assert np.isfinite(result.hazard_ratio)


def test_rng_state_reproduces_the_exact_bootstrap_draws_even_from_a_pre_advanced_generator():
    # The bug this closes: recording only `bit_generator.seed_seq.entropy` reflects a generator's
    # ORIGINAL construction seed. If the caller had already advanced the generator before passing it in
    # (as here), entropy-based replay would silently restart from scratch and reproduce the WRONG
    # sequence. `rng_state`, captured the moment `cohort_attribution` receives the generator, must
    # reproduce the actual draws regardless.
    covariates, time, event = _simulate_cohort(seed=22, n=150)

    caller_rng = np.random.default_rng(999)
    _ = caller_rng.integers(0, 1_000_000, size=37)  # the caller already used this generator elsewhere

    result = cohort_attribution(covariates, time, event, n_boot=50, rng=caller_rng)
    rng_state = result.provenance["rng_state"]

    # A naive "just replay the original construction seed" approach would get this wrong.
    assert np.random.default_rng(999).bit_generator.state != rng_state

    # Reconstructing a fresh Generator purely from the recorded state and rerunning must reproduce the
    # exact same bootstrap AF draws -- the actual proof `rng_state` is usable, not just non-None.
    bit_generator_cls = getattr(np.random, rng_state["bit_generator"])
    replayed_bit_generator = bit_generator_cls()
    replayed_bit_generator.state = rng_state
    replayed = cohort_attribution(covariates, time, event, n_boot=50, rng=np.random.Generator(replayed_bit_generator))

    np.testing.assert_array_equal(
        result.provenance["af_distribution"].samples, replayed.provenance["af_distribution"].samples
    )
    assert result.af_ci == replayed.af_ci
    assert result.hazard_ratio == replayed.hazard_ratio
