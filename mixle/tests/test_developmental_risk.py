import numpy as np
import pytest

from mixle.analysis.developmental_risk import BMDResult, benchmark_dose, rfd_exceedance


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


# MXR-080-0083: the reported BMDL must be a genuine, dimensionally-consistent one-sided
# confidence bound (delta method: implicit-differentiation gradient x observed-information
# covariance), with convergence diagnostics, not an arbitrary penalty term or a fallback that
# mixes dose-unit and bmr-unit quantities.


def test_bmdl_is_dimensionally_consistent_under_dose_rescaling():
    # A real confidence bound on a dose is the same physical quantity under any unit choice: BMD
    # and BMDL must both scale by exactly the dose-unit factor. The pre-fix fallback failed this
    # outright (see the audit): it multiplied a dose-unit-derived step by bmd a second time, so
    # the same underlying curve gave wildly different relative BMDL widths at different scales.
    doses = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_total = np.full(5, 100.0)
    n_affected = np.array([5.0, 10.0, 25.0, 60.0, 90.0])
    r1 = benchmark_dose(doses, n_affected, n_total, bmr=0.10, model="loglogistic")
    k = 1000.0
    r2 = benchmark_dose(doses * k, n_affected, n_total, bmr=0.10, model="loglogistic")
    assert r1.status == "ok" and r2.status == "ok"
    assert abs(r2.bmd / r1.bmd - k) / k < 1e-3
    assert abs(r2.bmdl / r1.bmdl - k) / k < 1e-3


def test_bmd_gradient_reports_none_when_ill_conditioned():
    # Convergence diagnostic: the gradient helper must refuse to divide by a near-zero dF/dd
    # rather than manufacture an arbitrarily large, meaningless derivative.
    from mixle.analysis.developmental_risk import _bmd_gradient

    coef = np.array([-1.0, 50.0])  # saturates almost immediately -- flat in dose past that point
    grad = _bmd_gradient("loglogistic", coef, dose_min_eff=1e-9, bmr=0.10, risk="extra", bmd=1e-9)
    assert grad is None


def test_bmd_identified_but_bmdl_unavailable_under_quasi_separation():
    # A near-perfect step function pushes the MLE toward extreme coefficients; the BMD root can
    # still be found and bracketed, but the delta-method covariance can fail to be well-posed --
    # BMDResult must distinguish this ("bmd real, bmdl not") from full unidentifiability.
    doses = np.array([1.0, 2.0, 3.0, 4.0])
    n_total = np.full(4, 50.0)
    n_affected = np.array([0.0, 0.0, 50.0, 50.0])
    result = benchmark_dose(doses, n_affected, n_total, bmr=0.10)
    assert result.status == "bmdl_unavailable"
    assert result.converged is False
    assert np.isfinite(result.bmd)
    assert np.isnan(result.bmdl)
    assert np.isnan(result.bmd_se)


def test_bmdl_delta_method_achieves_nominal_coverage():
    # The actual statistical validation the audit asked for: simulate from a KNOWN model with a
    # KNOWN true BMD many times, refit, and confirm the true BMD falls above the reported BMDL
    # roughly ci_level of the time. benchmark_dose's 2-parameter log-logistic has no free
    # background parameter -- "background" is the fitted response rate at the lowest TESTED dose
    # -- so the reference "true" BMD must be computed under that same convention (solving for the
    # true coefficients' own root), not a from-scratch closed-form zero-dose asymptote that the
    # code was never trying to estimate in the first place.
    from mixle.analysis.developmental_risk import _quantal_p, _solve_bmd

    b_true, c_true = -2.0, 1.5
    coef_true = np.array([b_true, c_true])
    doses = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    n_group = 100
    n_total = np.full(doses.shape, float(n_group))
    dose_hi = float(doses.max()) * 10.0
    dose_min_eff = doses.min()
    background_true = float(_quantal_p("loglogistic", np.array([dose_min_eff]), coef_true)[0])
    bmd_true, ok = _solve_bmd("loglogistic", coef_true, background_true, bmr=0.10, risk="extra", dose_hi=dose_hi)
    assert ok

    def p_true(d):
        return 1.0 / (1.0 + np.exp(-(b_true + c_true * np.log(np.clip(d, 1e-9, None)))))

    n_reps = 300
    covered = 0
    ok_count = 0
    for seed in range(n_reps):
        rng = np.random.default_rng(seed)
        n_affected = rng.binomial(n_group, p_true(doses)).astype(float)
        result = benchmark_dose(doses, n_affected, n_total, bmr=0.10, model="loglogistic", ci_level=0.95)
        if result.status != "ok":
            continue
        ok_count += 1
        if result.bmdl <= bmd_true:
            covered += 1

    assert ok_count >= 0.9 * n_reps  # this design should be well-identified almost every replication
    coverage = covered / ok_count
    # Nominal is 0.95; with n_reps=300 the Monte Carlo SE of the coverage estimate itself is
    # ~1.3pp, so this wide-but-meaningful one-sided band catches a badly broken interval (e.g.
    # ~50% coverage from a sign error or a mis-set confidence quantile) without flaking on
    # ordinary simulation noise.
    assert coverage >= 0.85, f"empirical coverage {coverage:.3f} over {ok_count} reps is far below the nominal 0.95"


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


# MXR-080-0084: rfd_exceedance must validate its control domains (uf, n, exposure draws) before
# any division or resampling, instead of letting bad input reach an opaque division/RNG error.


def _valid_bmd_result():
    doses = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_total = np.full(doses.shape, 100.0)
    n_affected = np.array([5.0, 10.0, 25.0, 60.0, 90.0])
    return benchmark_dose(doses, n_affected, n_total, bmr=0.10)


@pytest.mark.parametrize("uf", [True, 0.0, -5.0, float("nan"), float("inf"), np.array([100.0])])
def test_rejects_bad_uncertainty_factor(uf):
    result = _valid_bmd_result()
    with pytest.raises((TypeError, ValueError), match="uf"):
        rfd_exceedance(np.array([1.0, 2.0, 3.0]), result, uf=uf, n=3, rng=np.random.default_rng(0))


@pytest.mark.parametrize("n", [True, 0, -3, 2.5, 3.0, float("nan")])
def test_rejects_bad_sample_count(n):
    result = _valid_bmd_result()
    with pytest.raises(ValueError, match="n must be"):
        rfd_exceedance(np.array([1.0, 2.0, 3.0]), result, uf=100.0, n=n, rng=np.random.default_rng(0))


def test_rejects_empty_exposure_array():
    result = _valid_bmd_result()
    with pytest.raises(ValueError, match="empty"):
        rfd_exceedance(np.array([]), result, uf=100.0, n=5, rng=np.random.default_rng(0))


def test_rejects_non_finite_and_negative_exposure_draws():
    result = _valid_bmd_result()
    with pytest.raises(ValueError, match="finite"):
        rfd_exceedance(np.array([1.0, np.nan, 3.0]), result, uf=100.0, n=5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="nonnegative"):
        rfd_exceedance(np.array([1.0, -2.0, 3.0]), result, uf=100.0, n=5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="nonnegative"):
        rfd_exceedance(-1.0, result, uf=100.0, n=5, rng=np.random.default_rng(0))


def test_rejects_non_finite_bmdl():
    # A consequence of MXR-080-0082/0083: bmd.bmdl can now legitimately be nan (unidentifiable or
    # bmdl_unavailable). rfd_exceedance must refuse to divide by it rather than silently produce
    # rfd=nan and have `draws > nan` quietly evaluate to all-False.
    doses = np.array([1.0, 2.0, 4.0, 8.0])
    n_total = np.full(4, 100.0)
    n_affected = np.array([10.0, 11.0, 9.0, 10.0])  # flat curve -> unidentifiable
    flat_result = benchmark_dose(doses, n_affected, n_total, bmr=0.10)
    assert flat_result.status == "unidentifiable"
    with pytest.raises(ValueError, match="not 'ok'"):
        rfd_exceedance(np.array([1.0, 2.0, 3.0]), flat_result, uf=100.0, n=3, rng=np.random.default_rng(0))


def test_bmd_result_enforces_a_closed_immutable_state_machine():
    valid = BMDResult(
        bmd=2.0,
        bmdl=1.0,
        bmr=0.1,
        model="loglogistic",
        dof=3,
        status="ok",
        bmd_se=0.2,
    )
    assert valid.converged is True
    with pytest.raises((AttributeError, TypeError)):
        valid.status = "unidentifiable"
    with pytest.raises(ValueError):
        valid._coef[0] = 10.0

    invalid_states = [
        dict(bmd=2.0, bmdl=1.0, bmr=0.1, model="bogus", dof=3, status="ok", bmd_se=0.2),
        dict(bmd=2.0, bmdl=-1.0, bmr=0.1, model="loglogistic", dof=3, status="ok", bmd_se=0.2),
        dict(bmd=1.0, bmdl=2.0, bmr=0.1, model="loglogistic", dof=3, status="ok", bmd_se=0.2),
        dict(bmd=2.0, bmdl=1.0, bmr=1.1, model="loglogistic", dof=3, status="ok", bmd_se=0.2),
        dict(bmd=2.0, bmdl=1.0, bmr=0.1, model="loglogistic", dof=3, status="bogus", bmd_se=0.2),
        dict(
            bmd=2.0,
            bmdl=float("nan"),
            bmr=0.1,
            model="loglogistic",
            dof=3,
            status="unidentifiable",
        ),
        dict(
            bmd=2.0,
            bmdl=1.0,
            bmr=0.1,
            model="loglogistic",
            dof=3,
            status="bmdl_unavailable",
        ),
    ]
    for kwargs in invalid_states:
        with pytest.raises((TypeError, ValueError)):
            BMDResult(**kwargs)


def test_valid_rfd_exceedance_call_still_works():
    # Negative control: well-formed inputs must still work exactly as before.
    result = _valid_bmd_result()
    rng = np.random.default_rng(2)
    dq = rfd_exceedance(np.array([1.0, 2.0, 3.0, 100.0]), result, uf=100.0, n=4, rng=rng)
    assert dq.samples.shape == (4,)
    assert set(np.unique(dq.samples)).issubset({0.0, 1.0})

    dq_scalar = rfd_exceedance(5.0, result, uf=100.0, n=10, rng=rng)
    assert dq_scalar.samples.shape == (10,)


def test_rfd_exceedance_rejects_ambiguous_plain_exposure_axes():
    result = _valid_bmd_result()
    with pytest.raises(ValueError, match="one-dimensional"):
        rfd_exceedance(np.ones((3, 2)), result, n=3, rng=np.random.default_rng(0))


# Follow-up sweep (companion to MXR-080-0074/0075's fix in carcinogenic_risk.py): the same two
# defensive patterns applied there -- construction-time validation on a samples-carrying result
# type, and closing a silent-fallthrough gap in exposure/uncertainty consumption -- were found
# still open in this sibling module.


class _RawDrawDerivedQuantity:
    """Minimal IC-1 `DerivedQuantity`-conforming wrapper around a pushforward's raw output."""

    def __init__(self, samples: np.ndarray):
        self.samples = np.asarray(samples, dtype=float)
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


class _RawDrawPosterior:
    """A minimal IC-1 `Posterior` whose `samples()` returns caller-supplied draws verbatim, so a
    test can hand `rfd_exceedance` draws a real posterior should never produce (negative/NaN) --
    mirrors `test_carcinogenic_risk.py`'s `_RawDrawPosterior`."""

    def __init__(self, draws: np.ndarray):
        self._draws = np.asarray(draws, dtype=float)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._draws

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self._draws, axis=0)

    @property
    def cov(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self._draws, rowvar=False))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self._draws, a, axis=0), np.quantile(self._draws, 1.0 - a, axis=0)

    def derived_quantity(self, fn, n: int, rng: np.random.Generator) -> _RawDrawDerivedQuantity:
        return _RawDrawDerivedQuantity(fn(self.samples(n, rng)))


def test_sample_derived_quantity_rejects_empty_or_non_finite_samples():
    """`_SampleDerivedQuantity` had no construction-time validation at all: empty, NaN, or Inf
    samples were silently accepted. Defense-in-depth so invalid state can never flow downstream even
    if some upstream pushforward (here, `rfd_exceedance`) fails to validate its own inputs."""
    from mixle.analysis.developmental_risk import _SampleDerivedQuantity

    with pytest.raises(ValueError):
        _SampleDerivedQuantity(samples=np.array([]))
    with pytest.raises(ValueError):
        _SampleDerivedQuantity(samples=np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError):
        _SampleDerivedQuantity(samples=np.array([0.0, np.inf, 1.0]))
    with pytest.raises(ValueError):
        _SampleDerivedQuantity(samples=np.ones((2, 2)))


def test_sample_derived_quantity_accepts_valid_samples():
    """Negative control: a legitimate, non-empty, finite sample array still constructs cleanly."""
    from mixle.analysis.developmental_risk import _SampleDerivedQuantity

    dq = _SampleDerivedQuantity(samples=np.array([0.0, 1.0, 1.0, 0.0]))
    assert dq.credible_interval(0.5) is not None


def test_rfd_exceedance_posterior_draws_validated_same_as_plain_array():
    """`rfd_exceedance`'s array/scalar exposure path validates finite/nonnegative via
    `_as_dose_samples`, but a `Posterior`'s own draws were handed straight to `fn` with no check at
    all -- a mis-specified exposure posterior emitting a negative or NaN draw silently produced a
    confident-looking "not exceeding" (0.0) result instead of raising (NaN compared with `> rfd` is
    silently False). Both must now raise, exactly like the plain-array path already does."""
    result = _valid_bmd_result()

    negative_draws = np.array([[-5.0], [0.01], [0.02]])
    with pytest.raises(ValueError, match="exposure"):
        rfd_exceedance(_RawDrawPosterior(negative_draws), result, uf=100.0, n=3, rng=np.random.default_rng(0))

    nan_draws = np.array([[np.nan], [0.01], [0.02]])
    with pytest.raises(ValueError, match="exposure"):
        rfd_exceedance(_RawDrawPosterior(nan_draws), result, uf=100.0, n=3, rng=np.random.default_rng(0))


def test_rfd_exceedance_posterior_with_legitimate_draws_still_works():
    """Negative control: a well-behaved exposure `Posterior` (finite, nonnegative draws) still
    produces a working exceedance `DerivedQuantity` through the now-validated `fn` pushforward."""
    result = _valid_bmd_result()
    draws = np.array([[0.001], [0.01], [100.0]])
    dq = rfd_exceedance(_RawDrawPosterior(draws), result, uf=100.0, n=3, rng=np.random.default_rng(0))
    assert np.asarray(dq.samples).shape == (3,)
    assert set(np.unique(dq.samples)).issubset({0.0, 1.0})


@pytest.mark.parametrize("draws", [np.ones((3, 2)), np.ones((2, 1)), np.ones((3, 1, 1))])
def test_rfd_exceedance_rejects_posterior_draws_with_ambiguous_or_wrong_axes(draws):
    result = _valid_bmd_result()
    with pytest.raises(ValueError, match="exactly one scalar"):
        rfd_exceedance(_RawDrawPosterior(draws), result, uf=100.0, n=3, rng=np.random.default_rng(0))
