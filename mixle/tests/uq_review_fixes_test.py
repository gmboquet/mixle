"""Regression tests for the UQ/statistical-test review fixes (ledger U-1..U-10, I-4, I-5).

Each test pins a defect from audit/CODEBASE_REVIEW_LEDGER.md: inverted one-sided
brunner_munzel tails (U-1), missing Pratt zero corrections (U-2), dropped SIS weight
carryover in the particle filter (U-3), (n-1)-based jackknife+ endpoints (U-4), a
noise-accumulating ESS truncation (U-5), thinning that shrank the retained draw count
(U-6), an off-by-one cumulative-hazard lookup (U-7), the rank-normalize plotting
position (U-8), Breslow log-likelihood reported for Efron fits (U-9), mislabeled
canonical links (U-10), unrescaled m-out-of-n intervals (I-4), and the +1 correction
applied to fully enumerated permutation nulls (I-5).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps
from scipy.special import ndtri

from mixle.inference.conformal import jackknife_plus
from mixle.inference.diagnostics import _rank_normalize, ess
from mixle.inference.glm import _FAMILIES
from mixle.inference.mcmc.samplers import particle_filter
from mixle.inference.nonparametric import brunner_munzel, wilcoxon_signed_rank
from mixle.inference.resampling import bootstrap, permutation_test
from mixle.inference.survival import _cumhaz_at, cox_ph


# --------------------------------------------------------------------- U-1: one-sided direction
def test_brunner_munzel_one_sided_matches_scipy() -> None:
    x = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    for alternative in ("two-sided", "greater", "less"):
        ours = brunner_munzel(x, y, alternative=alternative)
        ref = sps.brunnermunzel(x, y, alternative=alternative)
        assert ours.pvalue == pytest.approx(ref.pvalue, rel=1e-9), alternative


# --------------------------------------------------------------------- U-2: Pratt zero corrections
def test_wilcoxon_pratt_matches_scipy() -> None:
    d = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    ours = wilcoxon_signed_rank(d, zero_method="pratt")
    ref = sps.wilcoxon(d, zero_method="pratt", correction=False, mode="approx")
    assert ours.pvalue == pytest.approx(ref.pvalue, rel=1e-9)

    rng = np.random.RandomState(seed=5)
    for _ in range(20):
        dd = rng.randint(-3, 4, size=12).astype(float)
        if np.all(dd == 0):
            continue
        ours = wilcoxon_signed_rank(dd, zero_method="pratt")
        ref = sps.wilcoxon(dd, zero_method="pratt", correction=False, mode="approx")
        assert ours.pvalue == pytest.approx(ref.pvalue, rel=1e-9)


# --------------------------------------------------------------------- U-3: SIS weight carryover
def test_particle_filter_without_resampling_matches_kalman_evidence() -> None:
    phi, q, r = 0.9, 0.3, 0.4
    rng = np.random.RandomState(seed=7)
    t_len = 12
    x_true = np.zeros(t_len)
    for t in range(1, t_len):
        x_true[t] = phi * x_true[t - 1] + rng.normal(scale=np.sqrt(q))
    ys = x_true + rng.normal(scale=np.sqrt(r), size=t_len)

    # exact scalar Kalman log-likelihood
    mean, var, ll_exact = 0.0, 1.0, 0.0
    for y in ys:
        pm, pv = phi * mean, phi * phi * var + q
        s = pv + r
        ll_exact += sps.norm.logpdf(y, loc=pm, scale=np.sqrt(s))
        gain = pv / s
        mean, var = pm + gain * (y - pm), (1.0 - gain) * pv

    def propagate(particles: np.ndarray, prng: np.random.RandomState) -> np.ndarray:
        return phi * particles + prng.normal(scale=np.sqrt(q), size=particles.shape)

    def log_likelihood(particles: np.ndarray, y: float) -> np.ndarray:
        return sps.norm.logpdf(y, loc=particles[:, 0], scale=np.sqrt(r))

    init = np.random.RandomState(seed=11).normal(size=(4000, 1))
    _, ll_sis = particle_filter(ys, propagate, log_likelihood, init, resample=False, rng=np.random.RandomState(seed=13))
    assert ll_sis == pytest.approx(ll_exact, abs=0.75)


# --------------------------------------------------------------------- U-4: J+ order statistics
def test_jackknife_plus_uses_finite_sample_order_statistics() -> None:
    def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        return np.full(len(x_eval), float(np.mean(y_train)))

    rng = np.random.RandomState(seed=3)

    # n=5, alpha=0.1: ceil(0.9*6)=6 > 5 and floor(0.1*6)=0 < 1 -> both endpoints unbounded
    x5, y5 = rng.normal(size=(5, 1)), rng.normal(size=5)
    lower, upper = jackknife_plus(x5, y5, fit_predict, x5[:2], alpha=0.1)
    assert np.all(np.isinf(lower)) and np.all(lower < 0)
    assert np.all(np.isinf(upper)) and np.all(upper > 0)

    # n=12, alpha=0.1: lower is the 1st smallest of mu_-i - R_i, upper the 12th smallest (max)
    x12, y12 = rng.normal(size=(12, 1)), rng.normal(size=12)
    x_test = rng.normal(size=(3, 1))
    lower, upper = jackknife_plus(x12, y12, fit_predict, x_test, alpha=0.1)
    lo_mat = np.empty((12, 3))
    hi_mat = np.empty((12, 3))
    for i in range(12):
        keep = np.arange(12) != i
        mu_loo = fit_predict(x12[keep], y12[keep], x_test)
        resid = abs(y12[i] - fit_predict(x12[keep], y12[keep], x12[i : i + 1])[0])
        lo_mat[i], hi_mat[i] = mu_loo - resid, mu_loo + resid
    np.testing.assert_allclose(lower, np.sort(lo_mat, axis=0)[0])
    np.testing.assert_allclose(upper, np.sort(hi_mat, axis=0)[11])


# --------------------------------------------------------------------- U-5: Geyer ESS truncation
def test_ess_iid_vector_and_ar1_chains() -> None:
    rng = np.random.RandomState(seed=17)
    iid = rng.normal(size=(1, 1500, 6))
    per_dim = ess(iid)
    assert np.all(per_dim >= 0.75 * 1500), per_dim

    rho, n = 0.9, 4000
    chain = np.empty(n)
    chain[0] = rng.normal()
    for t in range(1, n):
        chain[t] = rho * chain[t - 1] + rng.normal(scale=np.sqrt(1 - rho * rho))
    theory = n * (1 - rho) / (1 + rho)
    got = float(ess(chain[None, :, None])[0])
    assert got == pytest.approx(theory, rel=0.35), (got, theory)


# --------------------------------------------------------------------- U-6: nuts_numba thinning
@pytest.mark.numba
@pytest.mark.optional
def test_nuts_numba_thin_keeps_num_samples() -> None:
    numba = pytest.importorskip("numba")
    from mixle.inference.mcmc.nuts_numba import nuts_numba

    @numba.njit(cache=False)
    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        return -0.5 * np.sum(x * x), -x

    result = nuts_numba(value_and_grad, np.zeros(2), num_samples=50, warmup=50, thin=3)
    assert np.asarray(result.samples).shape[0] == 50


# --------------------------------------------------------------------- U-7: cumulative-hazard lookup
def test_cumhaz_step_function_is_right_continuous_from_zero() -> None:
    event_times = np.asarray([1.0, 2.0, 3.0])
    base = np.asarray([0.1, 0.3, 0.6])
    t = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(_cumhaz_at(event_times, base, t), [0.0, 0.1, 0.1, 0.3, 0.3])


# --------------------------------------------------------------------- U-9: Efron reported loglik
def test_cox_reported_loglik_matches_requested_ties_method() -> None:
    time = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.asarray([1, 1, 1, 1, 1, 0, 1, 0])
    x = np.asarray([[0.5], [-0.2], [1.0], [0.1], [-0.7], [0.3], [0.9], [-1.1]])
    res_b = cox_ph(x, time, event, ties="breslow")
    res_e = cox_ph(x, time, event, ties="efron")

    def efron_loglik(beta: np.ndarray) -> float:
        eta = x @ beta
        w = np.exp(eta)
        ll = 0.0
        for t in np.unique(time[event == 1]):
            tied = (time == t) & (event == 1)
            risk = time >= t
            d = int(tied.sum())
            s0, sd0 = w[risk].sum(), w[tied].sum()
            ll += float(eta[tied].sum()) - float(np.sum(np.log(s0 - np.arange(d) / d * sd0)))
        return ll

    assert res_e.loglik == pytest.approx(efron_loglik(res_e.coef), rel=1e-9)
    assert res_e.loglik != pytest.approx(res_b.loglik, rel=1e-12)


# --------------------------------------------------------------------- U-10: canonical link labels
def test_family_canonical_links_are_mathematically_correct() -> None:
    assert _FAMILIES["gamma"].canonical == "inverse"
    assert _FAMILIES["inverse_gaussian"].canonical == "inverse_squared"
    # the numerically-safe fitting default is preserved, separately from the label
    assert _FAMILIES["gamma"].default_link == "log"
    assert _FAMILIES["inverse_gaussian"].default_link == "log"


# --------------------------------------------------------------------- U-8: rank-normal position
def test_rank_normalize_uses_blom_plotting_position() -> None:
    chains = np.arange(24.0).reshape(2, 12, 1)
    got = _rank_normalize(chains)
    ranks = np.arange(1, 25, dtype=float)
    expected = ndtri((ranks - 0.375) / (24 + 0.25)).reshape(2, 12, 1)
    np.testing.assert_allclose(got, expected)


# --------------------------------------------------------------------- I-4: m-out-of-n rescaling
def test_m_out_of_n_bootstrap_is_at_full_sample_scale() -> None:
    rng = np.random.RandomState(seed=23)
    data = rng.normal(size=400)
    full = bootstrap(data, np.mean, n_boot=600, method="percentile", seed=1)
    sub = bootstrap(data, np.mean, n_boot=600, method="percentile", seed=2, m=40)
    width_full = float(full.ci_high - full.ci_low)
    width_sub = float(sub.ci_high - sub.ci_low)
    assert width_sub == pytest.approx(width_full, rel=0.30), (width_sub, width_full)


# --------------------------------------------------------------------- I-5: exact p-values
def test_exact_permutation_pvalue_has_no_plus_one_correction() -> None:
    res = permutation_test(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]), alternative="less")
    assert res.pvalue == pytest.approx(1.0 / 6.0)
