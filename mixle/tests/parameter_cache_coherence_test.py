"""MXR-080-1192 -- a reassigned parameter must not leave a derived constant stale.

Most families precompute a log-normalizer from their scalar parameters in ``__init__`` and read it
in ``log_density``. Assigning a parameter afterwards used to leave that constant untouched, so the
scorer kept reporting the *previous* parameter's density with no error at all -- ``sigma2 = 100``
still scoring as ``sigma2 = 1``, and Poisson ``lam = -3`` returning ``log_density(1) == +3.0``, a
probability above one.

This is a whole defect class, not a handful of sites, so it is gated as one: every case below
perturbs a parameter and requires the score to move. A family that starts caching a new constant
without refreshing it fails here rather than silently mis-scoring.
"""

from __future__ import annotations

import inspect
import math

import pytest

import mixle.stats as st

# (class name, ctor args, parameter to perturb, perturbed value, a support point)
CASES = [
    ("GaussianDistribution", (0.0, 1.0), "sigma2", 4.0, 0.0),
    ("PoissonDistribution", (2.0,), "lam", 7.0, 1),
    ("BernoulliDistribution", (0.3,), "p", 0.8, 1),
    ("BinomialDistribution", (0.3, 10), "p", 0.8, 3),
    ("ExponentialDistribution", (2.0,), "beta", 5.0, 1.0),
    ("LaplaceDistribution", (0.0, 1.0), "b", 3.0, 0.5),
    ("UniformDistribution", (0.0, 1.0), "high", 4.0, 0.5),
    ("BetaDistribution", (2.0, 3.0), "a", 5.0, 0.5),
    ("InverseGammaDistribution", (2.0, 3.0), "beta", 6.0, 1.0),
    ("WeibullDistribution", (2.0, 1.0), "scale", 3.0, 1.0),
    ("ParetoDistribution", (2.0, 1.0), "alpha", 5.0, 2.0),
    ("LogisticDistribution", (0.0, 1.0), "scale", 3.0, 0.5),
    ("HalfNormalDistribution", (1.0,), "sigma", 3.0, 0.5),
    ("NegativeBinomialDistribution", (3.0, 0.4), "p", 0.7, 2),
    ("LogSeriesDistribution", (0.4,), "p", 0.7, 2),
    ("InverseGaussianDistribution", (1.0, 2.0), "lam", 5.0, 1.0),
    ("SkellamDistribution", (2.0, 3.0), "mu1", 5.0, 1),
    ("BetaBinomialDistribution", (10, 2.0, 3.0), "a", 6.0, 3),
    ("GeneralizedGaussianDistribution", (0.0, 1.0, 2.0), "alpha", 3.0, 0.5),
    ("NakagamiDistribution", (1.0, 1.0), "omega", 4.0, 1.0),
    ("GeneralizedExtremeValueDistribution", (0.0, 1.0, 0.1), "scale", 3.0, 0.5),
    ("GammaDistribution", (2.0, 1.0), "theta", 3.0, 1.5),
    ("StudentTDistribution", (5.0, 0.0, 1.0), "scale", 3.0, 1.0),
    ("WrappedCauchyDistribution", (0.0, 0.3), "rho", 0.8, 0.5),
    ("NakagamiDistribution", (1.0, 1.0), "m", 2.5, 1.0),
    ("SkellamDistribution", (2.0, 3.0), "mu2", 6.0, 1),
    ("NegativeBinomialDistribution", (3.0, 0.4), "r", 7.0, 2),
    ("WeibullDistribution", (2.0, 1.0), "shape", 4.0, 1.0),
]


@pytest.mark.parametrize("cls_name, args, attr, new_value, x", CASES, ids=[c[0] for c in CASES])
def test_reassigning_a_parameter_moves_the_score(cls_name, args, attr, new_value, x):
    """Perturb one parameter; the density must follow it rather than a cached constant."""
    cls = getattr(st, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} is not exported from mixle.stats")
    dist = cls(*args)
    assert hasattr(dist, attr), f"{cls_name} has no parameter {attr!r}; update this case"
    before = float(dist.log_density(x))
    setattr(dist, attr, new_value)
    after = float(dist.log_density(x))
    assert not math.isclose(before, after, rel_tol=0.0, abs_tol=1e-12), (
        f"{cls_name}.log_density({x}) stayed at {before!r} after {attr}={new_value}: a constant "
        f"derived from {attr!r} in __init__ was not refreshed, so scoring reports the old parameter."
    )
    assert math.isfinite(after), f"{cls_name} scored non-finite after a valid {attr}={new_value}"

    # "The score moved" is NOT sufficient, and relying on it is what let MXR-080-1192 survive a
    # first repair: when a class caches SEVERAL constants from one parameter and the refresh hook
    # updates only some of them, the score does move -- to a wrong number. Nakagami refreshed
    # `_log_const` but not `_m_over_omega`, Skellam refreshed `log_ratio_half` but neither
    # `sqrt_diff_sq` nor `two_sqrt_prod`, and NegativeBinomial refreshed `log_p` but not `log_1p`
    # (which scored a log-density of +0.089 -- a probability above one -- and passed the old check).
    # The only comparison that cannot be fooled by a forgotten field is against a fresh object.
    rebuilt_args = list(args)
    parameters = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    assert attr in parameters, f"{cls_name}.__init__ takes no {attr!r}; update this case"
    rebuilt_args[parameters.index(attr)] = new_value
    expected = float(cls(*rebuilt_args).log_density(x))
    assert math.isclose(after, expected, rel_tol=1e-12, abs_tol=1e-12), (
        f"{cls_name}.log_density({x}) is {after!r} after {attr}={new_value}, but a freshly built "
        f"{cls_name} with the same parameters scores {expected!r}. The score moved, so a partial "
        f"refresh ran, but at least one other constant derived from {attr!r} is still stale."
    )


def test_the_gate_would_notice_a_stale_constant():
    """Guard the guard: a class that caches without refreshing must fail the comparison above."""

    class StaleByConstruction:
        def __init__(self, scale):
            self.scale = float(scale)
            self._log_scale = math.log(self.scale)  # never refreshed

        def log_density(self, x):
            return -self._log_scale

    dist = StaleByConstruction(1.0)
    before = float(dist.log_density(0.0))
    dist.scale = 4.0
    after = float(dist.log_density(0.0))
    assert math.isclose(before, after, rel_tol=0.0, abs_tol=1e-12), (
        "the negative control must reproduce the stale-cache behaviour this gate detects"
    )
