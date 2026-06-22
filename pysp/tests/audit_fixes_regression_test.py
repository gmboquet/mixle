"""Regression tests for the 2026-06 computation audit fixes.

Each test pins a specific bug found in the audit (see audit/COMPUTATION_AUDIT.md) so it cannot
silently regress. Test names carry the audit finding id (H1/H2/H3/M*/C*/E*).
"""

import warnings

import numpy as np
import pytest


# --------------------------------------------------------------------------- H3
def test_h3_beta_posterior_mode_is_a_over_a_plus_b():
    from pysp.stats.sets.bernoulli_set import _beta_posterior_mode

    # a == b case previously fell through to 1.0; b > a case used a wrong formula.
    assert _beta_posterior_mode(2, 2, 5, 10) == pytest.approx(0.5)  # a=b=6 -> 6/12
    # b > a: prior Beta(2,3), obs 3 of 10 -> a=4, b=9 -> 4/13
    assert _beta_posterior_mode(2, 3, 3, 10) == pytest.approx(4.0 / 13.0)
    # a > b branch unchanged and correct -> a=9, b=4 -> 9/13
    assert _beta_posterior_mode(3, 2, 7, 10) == pytest.approx(9.0 / 13.0)


# --------------------------------------------------------------------------- M4
def test_m4_multivariate_gaussian_zero_nobs_mean_is_finite():
    from pysp.stats.multivariate.multivariate_gaussian import MultivariateGaussianEstimator

    est = MultivariateGaussianEstimator(dim=2)
    acc = est.accumulator_factory().make()
    # Zero-responsibility component: empty/zero-weight suff stat.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any divide warning becomes a failure
        d = est.estimate(0.0, acc.value())
    assert np.all(np.isfinite(d.mu))


def test_m4_diagonal_gaussian_zero_nobs_mean_is_finite():
    from pysp.stats.multivariate.diagonal_gaussian import DiagonalGaussianEstimator

    est = DiagonalGaussianEstimator(dim=3)
    acc = est.accumulator_factory().make()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = est.estimate(0.0, acc.value())
    assert np.all(np.isfinite(d.mu))


# --------------------------------------------------------------------------- C1
def test_c1_integer_multinomial_out_of_support_zero_count_is_finite():
    from pysp.stats.leaf.integer_multinomial import IntegerMultinomialDistribution

    d = IntegerMultinomialDistribution(0, [0.5, 0.5])
    # value 5 is out of the 2-symbol support but carries count 0: (-inf)*0 must not be NaN.
    val = d.log_density([(5, 0)])
    assert np.isfinite(val)


# --------------------------------------------------------------------------- C2
def test_c2_categorical_estimate_does_not_mutate_suff_stat():
    from pysp.stats.leaf.categorical import CategoricalEstimator

    est = CategoricalEstimator(pseudo_count=1.0)
    suff = {"a": 3.0, "b": 1.0}
    snapshot = dict(suff)
    est.estimate(None, suff)
    assert suff == snapshot  # counts must not be overwritten with probabilities


# --------------------------------------------------------------------------- M2 (variance cancellation)
def test_m2_skew_normal_recovers_shape_under_large_offset():
    from pysp.stats.leaf.skew_normal import SkewNormalDistribution, SkewNormalEstimator

    est = SkewNormalEstimator()
    d = SkewNormalDistribution(1.0e6, 1.3, 4.0)
    x = np.asarray(d.sampler(3).sample(40000), dtype=float)
    acc = est.accumulator_factory().make()
    acc.seq_update(x, np.ones(len(x)), None)
    fit = est.estimate(None, acc.value())
    # Pre-fix this returned a wildly wrong negative shape (~-2000); now it is near the true 4.0.
    assert fit.shape == pytest.approx(4.0, abs=1.5)


def test_m2_skew_normal_combine_matches_single_batch():
    from pysp.stats.leaf.skew_normal import SkewNormalDistribution, SkewNormalEstimator

    est = SkewNormalEstimator()
    d = SkewNormalDistribution(1.0e6, 1.3, 4.0)
    x = np.asarray(d.sampler(4).sample(40000), dtype=float)
    a = est.accumulator_factory().make()
    b = est.accumulator_factory().make()
    full = est.accumulator_factory().make()
    a.seq_update(x[:15000], np.ones(15000), None)
    b.seq_update(x[15000:], np.ones(len(x) - 15000), None)
    a.combine(b.value())
    full.seq_update(x, np.ones(len(x)), None)
    # Parallel central-moment merge must equal the single-pass accumulation (up to FP reassociation).
    assert est.estimate(None, a.value()).shape == pytest.approx(est.estimate(None, full.value()).shape, rel=1e-6)


def test_m2_exgaussian_recovers_under_large_offset():
    from pysp.stats.leaf.exgaussian import (
        ExponentiallyModifiedGaussianDistribution as EMG,
    )
    from pysp.stats.leaf.exgaussian import (
        ExponentiallyModifiedGaussianEstimator as EMGEst,
    )

    est = EMGEst()
    d = EMG(1.0e6, 1.0, 0.5)  # mu, sigma, lam
    x = np.asarray(d.sampler(5).sample(60000), dtype=float)
    acc = est.accumulator_factory().make()
    acc.seq_update(x, np.ones(len(x)), None)
    fit = est.estimate(None, acc.value())
    assert fit.sigma == pytest.approx(1.0, abs=0.3)
    assert fit.lam == pytest.approx(0.5, abs=0.2)


# --------------------------------------------------------------------------- M8 (conjugate centered scatter)
def test_m8_gaussian_conjugate_scatter_is_nonnegative_offset_data():
    from pysp.stats.bayes.conjugate import _build_gaussian

    x = 1.0e8 + np.random.RandomState(0).randn(2000)
    post = _build_gaussian(None, x, np.ones(len(x)), {"m": 0.0, "kappa": 1e-3, "a": 1e-3, "b": 1e-3})
    # b_n must stay > b0 (the scatter contribution is a nonnegative centered sum of squares).
    assert post.b > 1e-3
    assert np.isfinite(post.b)


# --------------------------------------------------------------------------- M6 (survival tail)
def test_m6_survival_deep_tail_is_finite_not_neg_inf():
    from pysp.stats.combinator.survival import SurvivalDistribution
    from pysp.stats.leaf.gaussian import GaussianDistribution

    base = GaussianDistribution(0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # (t, event=0) is a right-censored observation -> scored by log S(t); the deep tail
        # must not collapse to -inf (true log S(40) ~ -804.6).
        ll = SurvivalDistribution(base).log_density((40.0, 0))
    assert np.isfinite(ll) and ll < 0.0


# --------------------------------------------------------------------------- E1/E3 (symbolic engine)
def test_e1_symbolic_digamma_evaluates():
    import scipy.special

    from pysp.engines import SYMBOLIC_ENGINE

    out = SYMBOLIC_ENGINE.digamma(SYMBOLIC_ENGINE.asarray(2.0))
    assert float(out.evaluate({})) == pytest.approx(scipy.special.digamma(2.0))


def test_e3_symbolic_logsumexp_is_overflow_safe():
    from pysp.engines import SYMBOLIC_ENGINE

    out = SYMBOLIC_ENGINE.logsumexp(SYMBOLIC_ENGINE.asarray([1000.0, 1000.0]))
    assert float(out.evaluate({})) == pytest.approx(1000.0 + np.log(2.0))


# --------------------------------------------------------------------------- geometric no-warn
def test_geometric_all_ones_does_not_warn():
    from pysp.stats.leaf.geometric import GeometricEstimator

    est = GeometricEstimator()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = est.estimate(None, (100.0, 100.0))
    assert np.isfinite(d.p)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
