"""Regression tests for the 2026-07 codebase-review `mixle.ppl` fixes.

Each test pins a specific defect from audit/CODEBASE_REVIEW_LEDGER.md section II.B (P-1..P-11)
so it cannot silently regress. Test names carry the finding id.
"""

import importlib.util
import math

import numpy as np
import pytest

from mixle.ppl import Field, Group, Mix, Normal, free

HAS_TORCH = importlib.util.find_spec("torch") is not None


# --------------------------------------------------------------------------- P-1 / P-11
@pytest.mark.skipif(not HAS_TORCH, reason="the ADVI route under test needs torch")
def test_p1_vi_noncentered_returns_value_space_params():
    # The ADVI readout never applied the non-centered back-transform value = loc + scale*z:
    # fit(how="vi") on Normal(Normal(0, 10).noncentered(), 1) with data mean ~5 returned the
    # z-space latent (~0.5) as the parameter posterior and bound the model at z.
    rng = np.random.RandomState(0)
    data = list(rng.normal(5.0, 1.0, size=80))
    m = Normal(mean=Normal(mean=0.0, sd=10.0).noncentered(), sd=1.0).fit(
        data, how="vi", steps=300, samples=800, rng=np.random.RandomState(1)
    )
    assert abs(float(m.dist.mu) - 5.0) < 0.5  # bound in value space, not z-space
    assert abs(float(m.result.mean("arg0")) - 5.0) < 0.5  # posterior draws in value space


def test_p11_vi_accepts_seed_and_is_deterministic():
    # fit(how="vi", seed=...) raised TypeError (vi_fit had no `seed`); now threads it as the rng.
    rng = np.random.RandomState(2)
    data = list(rng.normal(1.0, 1.0, size=40))
    kw = {"how": "vi", "seed": 1, "steps": 60, "samples": 200, "max_iter": 200}
    a = Normal(mean=Normal(mean=0.0, sd=5.0), sd=1.0).fit(data, **kw)
    b = Normal(mean=Normal(mean=0.0, sd=5.0), sd=1.0).fit(data, **kw)
    assert float(a.dist.mu) == pytest.approx(float(b.dist.mu), abs=0.0)  # same seed -> same fit


# --------------------------------------------------------------------------- P-2 / P-3
def _reference_lmm_em(y, g, n_groups, iters):
    """Random-intercept EM (intercept-only fixed effect) run to convergence, as keyword-built
    reference: y_gi = beta + b_g + eps, b_g ~ N(0, tau^2), eps ~ N(0, sigma^2)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    beta = float(y.mean())
    var0 = max(float(np.var(y)), 1e-3)
    tau2, sigma2 = 0.5 * var0, 0.5 * var0
    bb = np.zeros(n_groups)
    for _ in range(iters):
        resid = y - beta
        ss, err2, tr = 0.0, 0.0, 0.0
        for gi in range(n_groups):
            idx = np.where(g == gi)[0]
            v = 1.0 / (1.0 / tau2 + idx.size / sigma2)
            m = v * float(resid[idx].sum()) / sigma2
            bb[gi] = m
            ss += m * m + v
            err2 += float(((resid[idx] - m) ** 2).sum())
            tr += idx.size * v
        tau2 = ss / n_groups
        sigma2 = max((err2 + tr) / n, 1e-8)
        beta = float((y - bb[g]).mean())
    return beta, math.sqrt(tau2), math.sqrt(sigma2)


def test_p2_p3_balanced_lmm_converges_variance_components_and_gls_se():
    # Balanced design: beta is an exact EM fixed point after one update, so the old beta-only
    # convergence test exited at iteration ~0 with sigma one step from its 0.5*var(y) init
    # (P-2), and the fixed-effect covariance inv(X'X/sigma^2) ignored the random effects (P-3).
    rng = np.random.RandomState(42)
    n_groups, n_per = 20, 5
    tau_true, sigma_true = 3.0, 1.0
    g = np.repeat(np.arange(n_groups), n_per)
    b = rng.normal(0.0, tau_true, n_groups)
    y = list(2.0 + b[g] + rng.normal(0.0, sigma_true, n_groups * n_per))

    m = Normal(mean=Group("g") + free, sd=free).fit(y, given={"g": list(g)})
    res = m.result
    _, tau_ref, sigma_ref = _reference_lmm_em(y=y, g=g, n_groups=n_groups, iters=1000)

    # P-2: sigma near truth and matching a reference EM run to convergence (not the init).
    assert abs(res.sigma - sigma_true) / sigma_true < 0.15
    assert res.sigma == pytest.approx(sigma_ref, abs=0.02)
    assert res.tau == pytest.approx(tau_ref, abs=0.05)

    # P-3: intercept SE from the GLS covariance (X'V^-1 X)^-1, V = tau^2 J + sigma^2 I per group.
    xtvx = 0.0
    for gi in range(n_groups):
        idx = np.where(g == gi)[0]
        v_g = tau_ref**2 * np.ones((idx.size, idx.size)) + sigma_ref**2 * np.eye(idx.size)
        ones = np.ones((idx.size, 1))
        xtvx += float((ones.T @ np.linalg.solve(v_g, ones))[0, 0])
    se_gls = math.sqrt(1.0 / xtvx)
    se_fit = float(math.sqrt(res.cov[0, 0]))
    assert abs(se_fit - se_gls) / se_gls < 0.20


# --------------------------------------------------------------------------- P-4
@pytest.mark.skipif(not HAS_TORCH, reason="the 1e-6 tolerance needs the analytic-gradient MAP")
def test_p4_map_with_flat_priors_is_the_mle():
    # how="map" used to maximize ll + prior + log|J| (the unconstrained-space density), so a
    # flat-prior MAP returned the sqrt(S/(n-1)) sd. The point estimate now drops the Jacobian:
    # MAP with flat priors == MLE, sd = sqrt(S/n).
    rng = np.random.RandomState(0)
    data = list(rng.normal(5.0, 1.0, size=80))
    m = Normal(mean=free, sd=free).fit(data, how="map")
    arr = np.asarray(data, dtype=float)
    s = float(((arr - arr.mean()) ** 2).sum())
    sd_mle = math.sqrt(s / arr.size)
    assert float(np.sqrt(m.dist.sigma2)) == pytest.approx(sd_mle, abs=1e-6)
    assert float(m.dist.mu) == pytest.approx(float(arr.mean()), abs=1e-6)


# --------------------------------------------------------------------------- P-5
def test_p5_mixture_unnamed_prior_slots_do_not_collide():
    # Unnamed prior slots were named arg{i} positionally WITHIN each component, so a 2-component
    # mixture produced two "arg0" slots and summary()/samples() silently dropped one component.
    from mixle.ppl.inference import _collect_composite

    mix = Mix(
        [
            Normal(mean=Normal(mean=-2.0, sd=3.0), sd=1.0),
            Normal(mean=Normal(mean=2.0, sd=3.0), sd=1.0),
        ],
        weights=np.array([0.5, 0.5]),
    )
    slots, _rebuild = _collect_composite(mix)
    names = [s.name for s in slots]
    assert len(set(names)) == len(names)  # unique
    assert names == ["comp0.arg0", "comp1.arg0"]

    rng = np.random.RandomState(3)
    data = list(np.concatenate([rng.normal(-2.0, 1.0, 25), rng.normal(2.0, 1.0, 25)]))
    m = mix.fit(data, how="mcmc", draws=200, burn=100, rng=np.random.RandomState(4))
    rows = {k: v for k, v in m.result.summary().items() if not k.startswith("_")}
    assert set(rows) == {"comp0.arg0", "comp1.arg0"}  # one row per slot, no collision
    a = m.result.samples("comp0.arg0")
    b = m.result.samples("comp1.arg0")
    assert a.shape == b.shape and not np.array_equal(a, b)


# --------------------------------------------------------------------------- P-6
def test_p6_lp_halfnormal_matches_scipy_on_a_grid():
    # The normalizing constant had the wrong sign: +0.22579 instead of log sqrt(2/pi) = -0.22579.
    from scipy.stats import halfnorm

    from mixle.ppl.inference import _lp_halfnormal

    xs = np.linspace(0.05, 6.0, 25)
    for s in (0.5, 1.0, 2.3):
        got = np.array([_lp_halfnormal(x, s, np) for x in xs])
        want = halfnorm.logpdf(xs, scale=s)
        np.testing.assert_allclose(got, want, atol=1e-12)


# --------------------------------------------------------------------------- P-7
def test_p7_gaussian_coefficient_prior_is_scaled_by_sigma2():
    # Normal coefficient priors entered IRLS as X'WX + P0 without the 1/sigma^2 likelihood
    # scaling -- ridge-at-sigma=1, not the posterior. With a fixed scale the fit now matches the
    # exact Gaussian posterior mean and covariance.
    rng = np.random.RandomState(7)
    xs = rng.normal(0.0, 1.0, 40)
    sigma = 3.0
    y = list(1.3 + 0.6 * xs + rng.normal(0.0, sigma, 40))
    m = Normal(mean=Normal(mean=0.0, sd=1.0) * Field("x") + Normal(mean=0.0, sd=1.0), sd=sigma).fit(
        y, given={"x": list(xs)}
    )
    design = np.column_stack([xs, np.ones_like(xs)])  # column order: slope, intercept
    prior_prec = np.eye(2)  # 1/sd^2 with sd=1
    a_exact = design.T @ design / sigma**2 + prior_prec
    beta_exact = np.linalg.solve(a_exact, design.T @ np.asarray(y) / sigma**2)
    np.testing.assert_allclose(m.result.beta, beta_exact, atol=1e-6)
    np.testing.assert_allclose(m.result.cov, np.linalg.inv(a_exact), atol=1e-6)


# --------------------------------------------------------------------------- P-9
def test_p9_kalman_em_loglik_is_nondecreasing_on_short_series():
    # The initial-state M-step assigned the smoothed time-0 posterior while the filter treats
    # (x0, P0) as pre-first-observation; on short series that broke EM monotonicity (LL drops
    # of ~0.6 nats on the first step). The M-step now smooths back to the pre-sample state.
    from mixle.ppl.statespace import _kalman_em

    for seed in (7, 14, 39):  # series that exposed LL decreases before the fix
        y = np.random.RandomState(seed).normal(0.0, 1.0, 4)
        for phi_free in (True, False):
            lls = [_kalman_em(y, phi_free, k, 0.0).loglik for k in range(1, 25)]
            increments = np.diff(np.asarray(lls))
            assert increments.min() > -1e-9, f"LL decreased (seed={seed}, phi_free={phi_free})"


# --------------------------------------------------------------------------- P-10
def test_p10_nig_sigma_posterior_mean_is_e_sigma_not_sqrt_e_sigma2():
    # The NIG summary labeled sqrt(E[sigma^2]) as sigma's "mean"; the inverse-gamma marginal has
    # E[sigma] = sqrt(b_n) * Gamma(a_n - 1/2) / Gamma(a_n).
    rng = np.random.RandomState(11)
    data = list(rng.normal(0.0, 2.0, size=60))
    m = Normal(mean=free, sd=free).fit(data, how="conjugate")
    arr = np.asarray(data, dtype=float)
    s = float(((arr - arr.mean()) ** 2).sum())
    a_n, b_n = 1.0 + arr.size / 2.0, 0.5 * s
    e_sigma = math.sqrt(b_n) * math.exp(math.lgamma(a_n - 0.5) - math.lgamma(a_n))
    assert m.result.summary()["sigma"]["mean"] == pytest.approx(e_sigma, rel=1e-12)
    # sanity: the Monte-Carlo mean of sigma draws agrees with the labeled mean
    draws = m.result.samples("sigma", n=4000, rng=np.random.RandomState(0))
    assert float(draws.mean()) == pytest.approx(e_sigma, rel=0.05)
