"""Tests for the CensoredDistribution combinator and the public TruncatedDistribution export."""

import math

import numpy as np
import pytest

from mixle.stats import (
    CensoredDistribution,
    GaussianDistribution,
    TruncatedDistribution,
)


def _base():
    return GaussianDistribution(0.0, 1.0)


def test_truncated_is_importable_and_works():
    # TruncatedDistribution must be reachable from the public mixle.stats API.
    base = GaussianDistribution(0.0, 1.0)
    t = TruncatedDistribution(base, allowed=None, forbidden=[])  # forbidding nothing keeps all mass
    # forbidding nothing => Z = 1 => log_density equals the base
    assert abs(t.log_density(0.3) - base.log_density(0.3)) < 1e-12


def test_interval_censoring_equals_cdf_difference():
    base = _base()
    d = CensoredDistribution(base)
    a, b = -0.5, 1.2
    expected = math.log(base.cdf(b) - base.cdf(a))
    assert abs(d.log_density((a, b)) - expected) < 1e-12


def test_right_censoring():
    base = _base()
    d = CensoredDistribution(base)
    a = 0.7
    # right censoring: known only that X >= a  -> P = 1 - F(a)
    expected = math.log(1.0 - base.cdf(a))
    assert abs(d.log_density((a, math.inf)) - expected) < 1e-12


def test_left_censoring():
    base = _base()
    d = CensoredDistribution(base)
    b = -0.3
    # left censoring: known only that X <= b -> P = F(b)
    expected = math.log(base.cdf(b))
    assert abs(d.log_density((-math.inf, b)) - expected) < 1e-12


def test_exact_observation_uses_base_density():
    base = _base()
    d = CensoredDistribution(base)
    assert abs(d.log_density(0.4) - base.log_density(0.4)) < 1e-12


def test_seq_log_density_mixed_batch():
    base = _base()
    d = CensoredDistribution(base)
    data = [0.4, (-0.5, 1.2), -1.0, (0.7, math.inf), (-math.inf, -0.3)]
    enc = d.dist_to_encoder().seq_encode(data)
    out = d.seq_log_density(enc)
    expected = np.array([d.log_density(v) for v in data])
    assert np.allclose(out, expected, atol=1e-12)


def test_zero_width_interval_is_minus_inf():
    base = _base()
    d = CensoredDistribution(base)
    # a degenerate interval has zero mass under a continuous base
    assert d.log_density((0.5, 0.5)) == -math.inf


def test_estimator_fits_base_on_exact_observations():
    rng = np.random.RandomState(0)
    exact = rng.normal(3.0, 2.0, size=5000)
    data = list(exact) + [(3.0, math.inf)] * 50  # a handful of right-censored points
    base = GaussianDistribution(0.0, 1.0)
    cens = CensoredDistribution(base)

    est = cens.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_initialize(cens.dist_to_encoder().seq_encode(data), np.ones(len(data)), rng)
    fit = est.estimate(None, acc.value())

    assert isinstance(fit, CensoredDistribution)
    # base recovered from the exact observations only
    assert abs(fit.base.mu - 3.0) < 0.2
    assert abs(fit.base.sigma2 - 4.0) < 0.5


def test_requires_cdf():
    class NoCDF:
        def log_density(self, x):
            return 0.0

    with pytest.raises(ValueError):
        CensoredDistribution(NoCDF())
