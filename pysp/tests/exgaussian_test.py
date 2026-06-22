"""Tests for the exponentially-modified Gaussian (EMG) leaf and the stable log_erfcx helper."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfcx
from scipy.stats import exponnorm

from pysp.stats import (
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
)
from pysp.utils.special import log_erfcx


def _emg(mu=1.0, sigma2=0.49, lam=2.0):
    return ExponentiallyModifiedGaussianDistribution(mu, sigma2, lam)


def test_log_density_matches_scipy_exponnorm():
    d = _emg()
    sigma = math.sqrt(d.sigma2)
    K = 1.0 / (d.lam * sigma)
    xs = np.linspace(-3.0, 12.0, 50)
    ref = exponnorm.logpdf(xs, K, loc=d.mu, scale=sigma)
    mine = d.seq_log_density(xs)
    assert np.allclose(mine, ref, atol=1e-10)
    # scalar path matches too
    for x in (-1.0, 0.5, 3.3, 9.0):
        assert abs(d.log_density(x) - float(exponnorm.logpdf(x, K, loc=d.mu, scale=sigma))) < 1e-10


def test_density_integrates_to_one():
    d = _emg(mu=0.5, sigma2=0.25, lam=1.5)
    total, _ = quad(d.density, -20.0, 60.0, limit=200)
    assert abs(total - 1.0) < 1e-6


def test_log_erfcx_stability_at_large_x():
    # log_erfcx matches the direct log(erfcx) across the moderate range (both branches' overlaps).
    xs_mid = np.array([-20.0, -2.0, 0.0, 1.0, 5.0, 20.0])
    assert np.allclose(log_erfcx(xs_mid), np.log(erfcx(xs_mid)), atol=1e-9)

    # large positive x: erfcx -> 1/(x*sqrt(pi)) so log_erfcx ~ -log(x) - 0.5*log(pi).
    # The asymptotic branch matches the direct log(erfcx) on its overlap and stays finite/accurate
    # even past where x*x would overflow.
    for x in (1e2, 1e3, 1e5, 1e8, 1e30, 1e200):
        val = float(log_erfcx(x))
        assert np.isfinite(val)
        approx = -math.log(x) - 0.5 * math.log(math.pi)
        assert abs(val - approx) < 1e-3

    # large negative x: erfcx = exp(x^2)*erfc(x) overflows (erfc -> 2), naive log is +inf
    assert np.isposinf(erfcx(-50.0))
    val_neg = float(log_erfcx(-50.0))
    assert np.isfinite(val_neg)
    assert abs(val_neg - ((-50.0) ** 2 + math.log(2.0))) < 1e-6


def test_sampler_moments():
    mu, sigma2, lam = 2.0, 0.36, 1.0
    d = ExponentiallyModifiedGaussianDistribution(mu, sigma2, lam)
    samp = np.asarray(d.sampler(seed=17).sample(200_000))
    # mean = mu + 1/lam, var = sigma2 + 1/lam^2
    assert abs(samp.mean() - (mu + 1.0 / lam)) < 0.05
    assert abs(samp.var() - (sigma2 + 1.0 / lam**2)) < 0.05
    # positive skew
    assert ((samp - samp.mean()) ** 3).mean() > 0.0


def test_estimator_recovers_parameters():
    mu, sigma2, lam = 1.5, 0.25, 0.8
    d = ExponentiallyModifiedGaussianDistribution(mu, sigma2, lam)
    data = np.asarray(d.sampler(seed=3).sample(400_000))

    est = ExponentiallyModifiedGaussianEstimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(data, np.ones_like(data), None)
    fit = est.estimate(None, acc.value())

    # method of moments: loose but consistent
    assert abs(fit.mu - mu) < 0.1
    assert abs(fit.sigma2 - sigma2) < 0.1
    assert abs((1.0 / fit.lam) - (1.0 / lam)) < 0.1


def test_estimator_handles_symmetric_data():
    # near-symmetric (Gaussian-ish) data should still yield a valid EMG, not crash
    rng = np.random.RandomState(0)
    data = rng.normal(0.0, 1.0, size=10_000)
    est = ExponentiallyModifiedGaussianEstimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(data, np.ones_like(data), None)
    fit = est.estimate(None, acc.value())
    assert fit.sigma2 > 0.0
    assert fit.lam > 0.0
    assert np.isfinite(fit.mu)


def test_combine_and_value_roundtrip():
    est = ExponentiallyModifiedGaussianEstimator()
    a = est.accumulator_factory().make()
    b = est.accumulator_factory().make()
    a.seq_update(np.array([1.0, 2.0, 3.0]), np.ones(3), None)
    b.seq_update(np.array([4.0, 5.0]), np.ones(2), None)
    a.combine(b.value())
    # The accumulator stores weighted central moments (count, mean, M2, M3); combining the two
    # batches must reproduce the moments of the pooled data [1, 2, 3, 4, 5].
    count, mean, m2, m3 = a.value()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert count == 5.0
    assert mean == pytest.approx(data.mean())
    assert m2 == pytest.approx(np.sum((data - data.mean()) ** 2))
    assert m3 == pytest.approx(np.sum((data - data.mean()) ** 3))
