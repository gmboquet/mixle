import numpy as np
import pytest

from mixle.analysis.developmental_risk import benchmark_dose, rfd_exceedance


def test_bmdl_matches_reference():
    rng = np.random.default_rng(0)
    b_true, c_true = -3.0, 1.2
    background = 1.0 / (1.0 + np.exp(-b_true))
    target = background + 0.10 * (1.0 - background)

    def p_true(d):
        return 1.0 / (1.0 + np.exp(-(b_true + c_true * np.log(np.clip(d, 1e-9, None)))))

    lo, hi = 1e-6, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_true(mid) < target:
            lo = mid
        else:
            hi = mid
    bmd_true = 0.5 * (lo + hi)

    doses = np.array([0.001, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    n_total = np.full(doses.shape, 200.0)
    n_affected = rng.binomial(200, p_true(doses)).astype(float)

    result = benchmark_dose(doses, n_affected, n_total, bmr=0.10, model="loglogistic")

    assert abs(result.bmd - bmd_true) / bmd_true < 0.5
    assert result.bmdl < result.bmd
    assert result.bmdl > 0

    covered = 0
    trials = 30
    for seed in range(trials):
        rng_i = np.random.default_rng(seed + 1000)
        n_affected_i = rng_i.binomial(200, p_true(doses)).astype(float)
        exposure = np.abs(rng_i.normal(loc=result.bmdl / 100.0, scale=result.bmdl / 400.0, size=2000))
        r_i = benchmark_dose(doses, n_affected_i, n_total, bmr=0.10, model="loglogistic")
        dq = rfd_exceedance(exposure, r_i, uf=100.0, n=2000, rng=rng_i)
        p_exceed = float(np.mean(dq.samples))
        if 0.0 <= p_exceed <= 1.0:
            covered += 1
    assert covered / trials >= 0.88


# MXR-080-0081: benchmark_dose must reject impossible cohorts before any likelihood evaluation.


def test_rejects_affected_exceeding_total():
    # The audit's own repro: 2 subjects "affected" out of a total of 1 is impossible.
    with pytest.raises(ValueError, match="n_affected must not exceed n_total"):
        benchmark_dose(
            dose=np.array([1.0, 2.0]),
            n_affected=np.array([2.0, 3.0]),
            n_total=np.array([1.0, 5.0]),
            bmr=0.10,
        )


def test_rejects_mismatched_n_total_length():
    # 2 dose groups, n_total of length 3: not a scalar/length-1 broadcast and not a per-group
    # match either -- must be rejected outright, not silently misaligned by numpy broadcasting.
    with pytest.raises(ValueError, match="n_total must be"):
        benchmark_dose(
            dose=np.array([1.0, 2.0]),
            n_affected=np.array([5.0, 10.0]),
            n_total=np.array([50.0, 60.0, 70.0]),
            bmr=0.10,
        )


def test_length_one_n_total_is_an_explicit_documented_broadcast():
    # A length-1 n_total broadcasting across every dose group is a real, intentional feature
    # (shared total per group, e.g. a balanced design) -- it must behave IDENTICALLY to writing
    # the same total out per group explicitly, not just "happen to not crash".
    dose = np.array([1.0, 2.0, 4.0])
    n_affected = np.array([5.0, 12.0, 30.0])
    r_broadcast = benchmark_dose(dose, n_affected, np.array([50.0]), bmr=0.10)
    r_explicit = benchmark_dose(dose, n_affected, np.array([50.0, 50.0, 50.0]), bmr=0.10)
    assert r_broadcast.bmd == r_explicit.bmd
    assert r_broadcast.bmdl == r_explicit.bmdl


def test_rejects_single_dose_group():
    # A 2-parameter curve cannot be identified from a single dose group.
    with pytest.raises(ValueError, match="distinct dose"):
        benchmark_dose(dose=np.array([1.0]), n_affected=np.array([5.0]), n_total=np.array([50.0]))


def test_rejects_all_identical_doses():
    # 2 dose groups, but both at the same dose value -- still only 1 distinct design point.
    with pytest.raises(ValueError, match="distinct dose"):
        benchmark_dose(
            dose=np.array([1.0, 1.0]),
            n_affected=np.array([5.0, 6.0]),
            n_total=np.array([50.0, 50.0]),
        )


def test_rejects_non_integer_and_negative_counts():
    with pytest.raises(ValueError, match="integer subject counts"):
        benchmark_dose(np.array([1.0, 2.0]), np.array([5.5, 10.0]), np.array([50.0, 50.0]))
    with pytest.raises(ValueError, match="nonnegative"):
        benchmark_dose(np.array([1.0, 2.0]), np.array([-1.0, 10.0]), np.array([50.0, 50.0]))


def test_rejects_bad_bmr_and_ci_level():
    dose = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_total = np.full(5, 100.0)
    n_affected = np.array([5.0, 10.0, 25.0, 60.0, 90.0])
    with pytest.raises(ValueError, match="bmr"):
        benchmark_dose(dose, n_affected, n_total, bmr=0.0)
    with pytest.raises(ValueError, match="ci_level"):
        benchmark_dose(dose, n_affected, n_total, ci_level=0.4)


def test_valid_multidose_cohort_still_fits():
    # Negative control: a well-formed, genuinely dose-responsive cohort must still fit cleanly.
    dose = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_total = np.full(5, 100.0)
    n_affected = np.array([5.0, 10.0, 25.0, 60.0, 90.0])
    result = benchmark_dose(dose, n_affected, n_total, bmr=0.10)
    assert result.status == "ok"
    assert result.converged is True
    assert np.isfinite(result.bmd) and result.bmd > 0
    assert np.isfinite(result.bmdl) and 0 <= result.bmdl <= result.bmd


# MXR-080-0082: a curve that never reaches the benchmark target must report an explicit
# unidentifiable result, never a fabricated dose from an exhausted search loop.


def test_flat_curve_is_unidentifiable_not_fabricated():
    # The audit's own repro: a flat (no dose-response) cohort must not return an astronomical
    # "BMD" (the pre-fix behaviour returned dose_hi * 2**40 here).
    doses = np.array([1.0, 2.0, 4.0, 8.0])
    n_total = np.full(4, 100.0)
    n_affected = np.array([10.0, 11.0, 9.0, 10.0])  # ~10% at every dose: no trend
    result = benchmark_dose(doses, n_affected, n_total, bmr=0.10, model="loglogistic")
    assert result.status == "unidentifiable"
    assert result.converged is False
    assert np.isnan(result.bmd)
    assert np.isnan(result.bmdl)
    # In particular, never anything resembling the old exhausted-search boundary.
    assert not (np.isfinite(result.bmd) and result.bmd > 1e6)


def test_solve_bmd_never_returns_exhausted_search_boundary():
    from mixle.analysis.developmental_risk import _solve_bmd

    coef_flat = np.array([-2.0, 0.0])  # c == 0: p(dose) is constant, target is never reached
    background = 1.0 / (1.0 + np.exp(2.0))
    dose, converged = _solve_bmd("loglogistic", coef_flat, background, bmr=0.10, risk="extra", dose_hi=1.0)
    assert converged is False
    assert np.isnan(dose)


def test_dose_responsive_curve_still_finds_a_real_bmd():
    # Negative control for 0082: a genuinely dose-responsive curve must still converge normally.
    doses = np.array([0.001, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    n_total = np.full(doses.shape, 200.0)
    rng = np.random.default_rng(7)
    b_true, c_true = -3.0, 1.2

    def p_true(d):
        return 1.0 / (1.0 + np.exp(-(b_true + c_true * np.log(np.clip(d, 1e-9, None)))))

    n_affected = rng.binomial(200, p_true(doses)).astype(float)
    result = benchmark_dose(doses, n_affected, n_total, bmr=0.10, model="loglogistic")
    assert result.status == "ok"
    assert result.converged is True
    assert 0 < result.bmd < 100  # comfortably within the tested dose range, not a search artifact
    assert 0 <= result.bmdl <= result.bmd


def test_rfd_exceedance_monotone_in_uf():
    doses = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_total = np.full(doses.shape, 100.0)
    n_affected = np.array([5.0, 10.0, 25.0, 60.0, 90.0])
    result = benchmark_dose(doses, n_affected, n_total)
    rng = np.random.default_rng(1)
    exposure = np.full(2000, result.bmdl / 50.0)
    dq_strict = rfd_exceedance(exposure, result, uf=10.0, n=2000, rng=rng)
    dq_lenient = rfd_exceedance(exposure, result, uf=1000.0, n=2000, rng=rng)
    assert float(np.mean(dq_strict.samples)) <= float(np.mean(dq_lenient.samples))
