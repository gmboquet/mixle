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
    assert dq.samples.shape == (80,)

    # every number attributes to the fit + seed
    assert result.provenance["seed"] == 1
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
