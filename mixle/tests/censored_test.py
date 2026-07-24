"""Tests for the CensoredDistribution combinator and the public TruncatedDistribution export."""

import math

import numpy as np
import pytest

from mixle.stats import (
    BernoulliDistribution,
    CensoredDistribution,
    GaussianDistribution,
    TruncatedDistribution,
    UniformDistribution,
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


def test_interval_entirely_outside_support_is_exact_zero():
    # Uniform(0, 1) has zero density anywhere outside [0, 1]; an interval entirely beyond the
    # support (both CDFs saturate to the same value, here 1.0) is a true zero, not an underflowed
    # tiny-but-positive float.
    d = CensoredDistribution(UniformDistribution(0.0, 1.0))
    assert d.density((2.0, 3.0)) == 0.0
    assert d.log_density((2.0, 3.0)) == -math.inf
    # same on the other side of the support
    assert d.density((-3.0, -2.0)) == 0.0
    assert d.log_density((-3.0, -2.0)) == -math.inf
    # negative control: an interval genuinely inside the support still scores as ordinary mass
    assert abs(d.density((0.2, 0.8)) - 0.6) < 1e-12


def test_far_tail_interval_without_logcdf_still_floors():
    # A base exposing only a linear `cdf` (no logcdf/logsf) can have a real, if tiny, representable
    # density at an interval whose CDF difference has itself underflowed to 0 in probability space
    # -- that must still floor to a tiny finite mass, not silently collapse to the exact-zero
    # reserved for intervals genuinely outside the support (previous test).
    class _SaturatingTail:
        # cdf saturates to 1.0 at x=1 while density stays representably nonzero past that point,
        # standing in for a real unbounded tail without needing extreme floating-point values.
        def cdf(self, x):
            return 1.0 if x >= 1.0 else 0.0

        def density(self, x):
            return 1e-300 if x >= 1.0 else 0.0

        def log_density(self, x):
            dv = self.density(x)
            return math.log(dv) if dv > 0.0 else -math.inf

    d = CensoredDistribution(_SaturatingTail())
    lm = d.log_density((2.0, 3.0))
    assert lm > -math.inf  # floored, not silently zeroed
    assert math.exp(lm) < 1e-300  # the floor is tiny, near the double-precision underflow limit


def test_closed_interval_on_discrete_base_includes_both_endpoints():
    # Bernoulli(p=0.25): P(0) = 0.75, P(1) = 0.25. The closed interval [0, 1] must include both
    # outcomes, i.e. the full probability mass.
    base = BernoulliDistribution(0.25)
    d = CensoredDistribution(base)
    assert abs(d.density((0, 1)) - 1.0) < 1e-12


def test_closed_single_point_interval_on_discrete_base_is_point_mass():
    # A single-point closed interval [0, 0] on a discrete base is exactly that outcome's mass, not
    # a degenerate (measure-zero) interval the way it would be for a continuous base.
    base = BernoulliDistribution(0.25)
    d = CensoredDistribution(base)
    assert abs(d.density((0, 0)) - 0.75) < 1e-12
    assert abs(d.density((1, 1)) - 0.25) < 1e-12


def test_closed_interval_on_discrete_base_excludes_non_atoms():
    # Negative control: closedness at `a` must only add back a real atom, not spuriously inflate an
    # interval that happens to touch no support point at all, or double-count an atom already
    # captured by the open-lower F(b) - F(a) term.
    base = BernoulliDistribution(0.25)
    d = CensoredDistribution(base)
    assert d.density((0.2, 0.8)) == 0.0  # no atom in [0.2, 0.8]
    assert abs(d.density((0.5, 1.0)) - 0.25) < 1e-12  # only atom 1 falls in [0.5, 1]


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
