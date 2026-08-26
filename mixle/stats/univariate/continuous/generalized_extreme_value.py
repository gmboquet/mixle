"""Generalized Extreme Value distribution (GEV): the limit law of block maxima.

By the Fisher-Tippett-Gnedenko theorem the normalized maximum of a large block of iid observations
converges to a GEV, the standard model for extremes (flood levels, wind speeds, record losses). With
location ``mu``, scale ``sigma > 0`` and shape ``xi`` (the EVT sign convention; ``scipy``'s
``genextreme`` uses ``c = -xi``), for ``z = (x - mu)/sigma``:

    log f = -log sigma - (1/xi + 1) log s - s ** (-1/xi),   s = 1 + xi z > 0   (xi != 0),
    log f = -log sigma - z - exp(-z)                                            (xi == 0, Gumbel).

``xi > 0`` is the heavy-tailed Frechet type (support ``x >= mu - sigma/xi``), ``xi = 0`` the Gumbel
type (all reals), ``xi < 0`` the bounded Weibull type (``x <= mu - sigma/xi``). All three parameters
are fit by method of moments: the shape is solved from the (monotone) skewness-vs-``xi`` relation,
then scale from the variance and location from the mean.


Reference: Coles, *An Introduction to Statistical Modeling of Extreme Values* (Springer, 2001).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gamma as _gamma

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.continuous._observation_contracts import (
    finite_observations,
    scored_observation,
)

_XI_TOL = 1.0e-8  # |xi| below this is treated as the Gumbel limit
_EULER = 0.5772156649015329
_GUMBEL_SKEW = 12.0 * math.sqrt(6.0) * 1.2020569031595943 / math.pi**3  # 12 sqrt(6) zeta(3) / pi^3
# How much worse (in nats) the covering-clamped shape may score the observation nearest the fitted
# support endpoint, relative to the Gumbel member with the same mean and variance, before the
# estimate abandons the clamped shape for that Gumbel limit. The clamp guarantees the endpoint lies
# outside the observed range, but with a Frechet-type shape the density decays like exp(-s^(-1/xi))
# at the endpoint, so "outside" can still mean a log-density of -1e28 for one real observation -- a
# fit that is numerically indistinguishable from declaring its own training data impossible
# (MXR-080 B12). Thirty nats is on the order of the tail penalty the Gumbel member itself assigns
# to a 3-4 sd extreme observation, so the clamped fit is kept whenever it remains comparable and
# dropped only when it is catastrophically worse.
_COVERING_FALLBACK_NATS = 30.0


def _gev_skewness(xi: float) -> float:
    """Theoretical skewness of a GEV with shape ``xi`` (monotone increasing; defined for ``xi < 1/3``)."""
    if abs(xi) < _XI_TOL:
        return _GUMBEL_SKEW
    g1, g2, g3 = _gamma(1.0 - xi), _gamma(1.0 - 2.0 * xi), _gamma(1.0 - 3.0 * xi)
    return float(np.sign(xi) * (g3 - 3.0 * g1 * g2 + 2.0 * g1**3) / (g2 - g1 * g1) ** 1.5)


def _xi_from_skewness(skew: float, xi_min: float, xi_max: float) -> float:
    """Invert the monotone skewness-vs-``xi`` relation by bisection."""
    if skew <= _gev_skewness(xi_min):
        return xi_min
    if skew >= _gev_skewness(xi_max):
        return xi_max
    lo, hi = xi_min, xi_max
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _gev_skewness(mid) < skew:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _endpoint_in_sds(xi: float) -> float:
    """Distance from the mean to the finite support endpoint, in standard deviations.

    Under moment matching the bounded end of a GEV sits a fixed number of standard deviations from
    the mean, independent of location and scale: substituting ``sigma = sd |xi| / sqrt(g2 - g1^2)``
    and ``mu = mean - sigma (g1 - 1)/xi`` into the endpoint ``mu - sigma/xi`` leaves
    ``mean +/- sd g1 / sqrt(g2 - g1^2)``. The value diverges as ``xi -> 0`` (Gumbel is unbounded)
    and shrinks monotonically as ``|xi|`` grows -- which is exactly why an unconstrained
    moment-matched shape can put the endpoint inside the sample it was fit on.
    """
    g1, g2 = _gamma(1.0 - xi), _gamma(1.0 - 2.0 * xi)
    denom = g2 - g1 * g1
    if denom <= 0.0 or not np.isfinite(denom):
        return np.inf
    return float(g1 / math.sqrt(denom))


def _shape_covering_range(xi: float, mean: float, sd: float, min_val: float, max_val: float) -> float:
    """Shrink ``|xi|`` until the moment-matched support contains the observed range.

    Method of moments matches mean, variance and skewness but says nothing about *support*: with
    ``xi < 0`` a GEV is bounded above, and a sample whose skew maps to, say, ``xi = -0.72`` gets an
    upper endpoint only 1.85 sd above the mean -- so the largest observations in the very sample
    being fit score ``-inf``. That is not a numerical accident; it is an estimate that declares its
    own training data impossible, and downstream EM cannot recover from it because the moments (and
    therefore the estimate) never change.

    Give up the third moment rather than the support: keep mean and variance matched exactly and
    move ``xi`` toward the Gumbel limit, which pushes the endpoint outward monotonically, until the
    observed range is strictly inside it. ``|xi|`` is only ever reduced, so a shape whose support
    already covers the data is returned untouched.
    """
    if abs(xi) < _XI_TOL or sd <= 0.0 or not (np.isfinite(min_val) and np.isfinite(max_val)):
        return xi
    needed = (max_val - mean) / sd if xi < 0.0 else (mean - min_val) / sd
    if not np.isfinite(needed):
        return xi
    margin = 1.0e-6 * max(1.0, abs(needed))
    target = needed + margin
    if _endpoint_in_sds(xi) >= target:
        return xi
    lo, hi = _XI_TOL, abs(xi)  # invariant: lo covers the range, hi does not
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _endpoint_in_sds(math.copysign(mid, xi)) >= target:
            lo = mid
        else:
            hi = mid
    return math.copysign(lo, xi)


class GeneralizedExtremeValueDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Extreme Value distribution with location ``loc``, scale ``> 0`` and shape ``xi``."""

    def __init__(
        self, loc: float, scale: float, shape: float, name: str | None = None, keys: str | None = None
    ) -> None:
        if scale <= 0.0 or not np.isfinite(scale) or not np.isfinite(loc) or not np.isfinite(shape):
            raise ValueError("GeneralizedExtremeValueDistribution requires finite parameters and scale > 0.")
        self.loc = float(loc)  # mu
        self.scale = float(scale)  # sigma
        self.shape = float(shape)  # xi
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep ``log_scale`` tied to the parameter(s) ``scale`` they derive from.

        Computed once in ``__init__`` and read by ``log_density``, so a later assignment used to
        leave them stale and the scorer kept reporting the *previous* parameters' density with no
        error at all (MXR-080-1192).

        Recompute rather than validate: callers legitimately install out-of-domain or non-finite
        parameters -- deserialized legacy states and NaN-propagation checks both do -- so a value
        outside the domain yields a NaN constant that propagates honestly instead of rejecting a
        state the library is expected to be able to hold.
        """
        object.__setattr__(self, name, value)
        if name not in ("scale",):
            return
        try:
            object.__setattr__(self, "log_scale", float(math.log(self.scale)))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, AttributeError, FloatingPointError):
            # AttributeError covers __init__, where the first parameter is assigned before the rest.
            object.__setattr__(self, "log_scale", float("nan"))

    def __str__(self) -> str:
        return "GeneralizedExtremeValueDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.loc),
            repr(self.scale),
            repr(self.shape),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation (``-inf`` outside the support)."""
        z = (scored_observation(x, label="GEV observations") - self.loc) / self.scale
        if abs(self.shape) < _XI_TOL:
            return -self.log_scale - z - math.exp(-z)
        s = 1.0 + self.shape * z
        if s <= 0.0:
            return -np.inf
        return -self.log_scale - (1.0 / self.shape + 1.0) * math.log(s) - s ** (-1.0 / self.shape)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (np.asarray(x, dtype=np.float64) - self.loc) / self.scale
        if abs(self.shape) < _XI_TOL:
            return -self.log_scale - z - np.exp(-z)
        s = 1.0 + self.shape * z
        with np.errstate(divide="ignore", invalid="ignore"):
            rv = -self.log_scale - (1.0 / self.shape + 1.0) * np.log(s) - np.power(s, -1.0 / self.shape)
        return np.where(s <= 0.0, -np.inf, rv)

    # --- compute-engine backend (numpy + torch/GPU): scoring + sufficient statistics in engine ops ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated GEV kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for GEV distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="generalized_extreme_value",
            distribution_type=cls,
            parameters=(ParameterSpec("loc"), ParameterSpec("scale", constraint="positive"), ParameterSpec("shape")),
            statistics=(
                StatisticSpec("sum"),
                StatisticSpec("sum2"),
                StatisticSpec("sum3"),
                StatisticSpec("count"),
                StatisticSpec("min_val", kind="support_bound", additive=False, scales=False),
                StatisticSpec("max_val", kind="support_bound", additive=False, scales=False),
            ),
            support="real",
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row GEV moment sums in accumulator order ``(sum, sum2, sum3, count)``.

        Deliberately not wired into the declaration as ``legacy_sufficient_statistics``. The
        generated statistic path reduces every declared statistic with a weighted *sum*, ignoring
        ``StatisticSpec.additive``, so it cannot produce the observed range the M-step needs to keep
        the fitted support around the data (see :func:`_shape_covering_range`) -- it would silently
        hand ``estimate`` a summed min and max. GEV therefore accumulates through the host
        ``seq_update``, which is three dot products for a univariate leaf; engine-resident *scoring*
        via ``backend_seq_log_density`` is unaffected. Reinstate this hook once the generated path
        honours non-additive support bounds.
        """
        xx = engine.asarray(x)
        x2 = xx * xx
        return xx, x2, x2 * xx, xx * 0.0 + engine.asarray(1.0)

    @staticmethod
    def backend_log_density_from_params(x: Any, loc: Any, scale: Any, shape: Any, engine: Any) -> Any:
        """Engine-neutral GEV log-density; the ``|xi| < tol`` Gumbel limit is selected per element.

        ``s^{-1/xi}`` is computed as ``exp(-log(s)/xi)`` so the whole expression stays on engine ops."""
        z = (x - loc) / scale
        neg_inf = engine.asarray(float("-inf"))
        is_limit = engine.abs(shape) < _XI_TOL
        xi_safe = engine.where(is_limit, engine.asarray(1.0), shape)
        s = 1.0 + xi_safe * z
        s_pos = engine.where(s > 0.0, s, engine.asarray(1.0))  # keep log/exp NaN-free off-support
        log_s = engine.log(s_pos)
        general = -engine.log(scale) - (1.0 / xi_safe + 1.0) * log_s - engine.exp(-log_s / xi_safe)
        general = engine.where(s > 0.0, general, neg_inf)
        limit = -engine.log(scale) - z - engine.exp(-z)
        return engine.where(is_limit, limit, general)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.loc),
            engine.asarray(self.scale),
            engine.asarray(self.shape),
            engine,
        )

    @classmethod
    def backend_stacked_params(
        cls, dists: Sequence["GeneralizedExtremeValueDistribution"], engine: Any
    ) -> dict[str, Any]:
        """Stacked GEV parameters for a homogeneous mixture kernel."""
        return {
            "loc": engine.asarray([d.loc for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
            "shape": engine.asarray([d.shape for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of GEV log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["loc"][None, :], params["scale"][None, :], params["shape"][None, :], engine
        )

    # No backend_stacked_sufficient_statistics here, deliberately. GEV declares six statistics --
    # the three moment sums, the count, and the observed min/max that keep its bounded support from
    # excluding a positive-weight training row. A stacked backend returning only the four moments
    # silently drops the two support bounds, and StackedMixtureKernel.has_resident_accumulate selects
    # a backend on mere presence, so the stale method won over the correct host accumulator and the
    # engine-resident path produced -inf on rows the host path fits (MXR-080-1846). Re-add this only
    # together with the support bounds, and only with the arity guard in stacked.accumulate green.

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact)."""
        from scipy.stats import genextreme as _sp

        return float(_sp.cdf(x, -self.shape, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import genextreme as _sp

        return float(_sp.ppf(q, -self.shape, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean: loc + scale*(Gamma(1-xi)-1)/xi (loc+scale*euler_gamma at xi=0); inf for xi>=1."""
        from scipy.special import gamma as _gamma

        xi = self.shape
        if abs(xi) < 1.0e-12:
            return float(self.loc + self.scale * np.euler_gamma)
        if xi < 1.0:
            return float(self.loc + self.scale * (_gamma(1.0 - xi) - 1.0) / xi)
        return float("inf")

    def variance(self) -> float:
        """Variance: scale^2 (Gamma(1-2xi)-Gamma(1-xi)^2)/xi^2 (scale^2 pi^2/6 at xi=0); inf for xi>=1/2."""
        import math

        from scipy.special import gamma as _gamma

        xi = self.shape
        if abs(xi) < 1.0e-12:
            return float(self.scale * self.scale * math.pi * math.pi / 6.0)
        if xi < 0.5:
            g1 = _gamma(1.0 - xi)
            g2 = _gamma(1.0 - 2.0 * xi)
            return float(self.scale * self.scale * (g2 - g1 * g1) / (xi * xi))
        return float("inf")

    def entropy(self) -> float:
        """Differential entropy log(scale) + euler_gamma * xi + euler_gamma + 1."""
        return float(self.log_scale + np.euler_gamma * self.shape + np.euler_gamma + 1.0)

    def sampler(self, seed: int | None = None) -> "GeneralizedExtremeValueSampler":
        """Return a sampler for drawing observations from this distribution."""
        return GeneralizedExtremeValueSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeneralizedExtremeValueEstimator":
        """Return a method-of-moments estimator for ``loc``, ``scale`` and ``shape``."""
        if pseudo_count is None or pseudo_count == 0.0:
            return GeneralizedExtremeValueEstimator(name=self.name, keys=self.keys)
        if not np.isfinite(pseudo_count) or pseudo_count < 0.0:
            raise ValueError("GEV pseudo_count must be finite and non-negative.")
        if self.shape >= 1.0 / 3.0:
            raise ValueError("GEV moment regularization requires a prior with a finite third moment (shape < 1/3).")
        # Convert this distribution's own (loc, scale, shape) into the raw first three moments, the
        # same space `estimate()` accumulates in, so pseudo_count can blend a prior pseudo-sample
        # toward them (mirrors GumbelEstimator / WeibullEstimator's suff_stat pattern).
        mean0 = self.mean()
        var0 = self.variance()
        skew0 = _gev_skewness(self.shape)
        second0 = var0 + mean0 * mean0
        # Invert estimate()'s own central-moment formula (m3 = E[X^3] - 3*mean*E[X^2] + 2*mean^3):
        # E[X^3] = skew*var^1.5 + 3*mean*E[X^2] - 2*mean^3.
        third0 = skew0 * (var0**1.5) + 3.0 * mean0 * second0 - 2.0 * mean0**3
        return GeneralizedExtremeValueEstimator(
            pseudo_count=pseudo_count, suff_stat=(mean0, second0, third0), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "GeneralizedExtremeValueDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return GeneralizedExtremeValueDataEncoder()


class GeneralizedExtremeValueSampler(DistributionSampler):
    """Draw iid GEV observations by inverse-CDF transform."""

    def __init__(self, dist: GeneralizedExtremeValueDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples by inverse CDF."""
        d = self.dist
        e = -np.log(self.rng.uniform(size=size))  # -log U ~ Exp(1) = the standard Gumbel core
        if abs(d.shape) < _XI_TOL:
            z = -np.log(e)
        else:
            z = (np.power(e, -d.shape) - 1.0) / d.shape
        return d.loc + d.scale * z


class GeneralizedExtremeValueAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first three moments and the observed range for GEV estimation.

    The range is not a moment: it is the support bound the estimate has to respect, since a GEV with
    ``xi != 0`` is bounded on one side and moment matching alone can place that bound inside the
    sample. See :func:`_shape_covering_range`.
    """

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.count = 0.0
        self.min_val = np.inf
        self.max_val = -np.inf
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: GeneralizedExtremeValueDistribution | None) -> None:
        """Accumulate weighted first three raw moments and the observed range for one observation."""
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.sum3 += x * x * x * weight
        self.count += weight
        if weight > 0.0:
            self.min_val = min(self.min_val, float(x))
            self.max_val = max(self.max_val, float(x))

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: np.ndarray, weights: np.ndarray, estimate: GeneralizedExtremeValueDistribution | None
    ) -> None:
        """Accumulate weighted first three raw moments and the observed range from encoded data."""
        xx = np.asarray(x, dtype=np.float64)
        self.sum += np.dot(xx, weights)
        self.sum2 += np.dot(xx * xx, weights)
        self.sum3 += np.dot(xx * xx * xx, weights)
        self.count += np.sum(weights, dtype=np.float64)
        mask = np.asarray(weights) > 0.0
        if np.any(mask):
            self.min_val = min(self.min_val, float(np.min(xx[mask])))
            self.max_val = max(self.max_val, float(np.max(xx[mask])))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: tuple[float, float, float, float, float, float]
    ) -> "GeneralizedExtremeValueAccumulator":
        """Merge another GEV sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        if len(suff_stat) > 5:
            self.min_val = min(self.min_val, float(suff_stat[4]))
            self.max_val = max(self.max_val, float(suff_stat[5]))
        return self

    def value(self) -> tuple[float, float, float, float, float, float]:
        """Return raw moment sums, observation count, and the observed range."""
        return self.sum, self.sum2, self.sum3, self.count, self.min_val, self.max_val

    def from_value(self, x: tuple[float, float, float, float, float, float]) -> "GeneralizedExtremeValueAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.sum3, self.count = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        # A four-entry tuple predates the range statistic and simply leaves the support unknown --
        # estimate() then skips the coverage clamp rather than inventing a bound.
        self.min_val = float(x[4]) if len(x) > 5 else np.inf
        self.max_val = float(x[5]) if len(x) > 5 else -np.inf
        return self

    def scale(self, c: float) -> "GeneralizedExtremeValueAccumulator":
        """Scale accumulated evidence while preserving the observed range."""
        self.sum *= c
        self.sum2 *= c
        self.sum3 *= c
        self.count *= c
        return self

    def acc_to_encoder(self) -> "GeneralizedExtremeValueDataEncoder":
        """Return the encoder used by this accumulator."""
        return GeneralizedExtremeValueDataEncoder()


class GeneralizedExtremeValueAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GeneralizedExtremeValueAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> GeneralizedExtremeValueAccumulator:
        """Create a fresh GEV accumulator."""
        return GeneralizedExtremeValueAccumulator(name=self.name, keys=self.keys)


class GeneralizedExtremeValueEstimator(ParameterEstimator):
    """Method-of-moments estimator for GEV location, scale and shape.

    This is a moment estimator, not an MLE: the shape is solved from the skewness-vs-``xi``
    relation, and the shape is additionally constrained so the fitted support covers the observed
    range (see :func:`_shape_covering_range`). Whenever that constraint rewrites the shape -- or the
    rewritten shape still scores the extreme observation catastrophically and the estimate falls
    back to the Gumbel limit -- the returned distribution is not the raw moment fit, and the change
    is recorded in ``numerical_repairs()`` so ``fit_provenance()`` carries it (MXR-080-1202).
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float, float] | None = None,
        min_scale: float = 1.0e-12,
        xi_max: float = 1.0 / 3.0 - 1.0e-4,  # third moment finite only for xi < 1/3
        xi_min: float = -1.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if pseudo_count is not None and (
            isinstance(pseudo_count, (bool, np.bool_)) or not np.isfinite(pseudo_count) or pseudo_count < 0.0
        ):
            raise ValueError("GEV pseudo_count must be finite and non-negative.")
        prior = None if suff_stat is None else tuple(float(value) for value in suff_stat)
        if prior is not None and (len(prior) != 3 or not all(np.isfinite(value) for value in prior)):
            raise ValueError("GEV prior moments must be a finite length-three tuple.")
        if pseudo_count not in (None, 0.0) and prior is None:
            raise ValueError("positive GEV pseudo_count requires prior moments.")
        if not np.isfinite(min_scale) or min_scale <= 0.0:
            raise ValueError("GEV min_scale must be finite and positive.")
        if not np.isfinite(xi_min) or not np.isfinite(xi_max) or xi_min >= xi_max or xi_max >= 1.0 / 3.0:
            raise ValueError("GEV shape bounds must be finite, ordered, and keep xi_max below 1/3.")
        self.pseudo_count = None if pseudo_count == 0.0 else pseudo_count
        self.suff_stat = prior
        self.min_scale = float(min_scale)
        self.xi_max = float(xi_max)
        self.xi_min = float(xi_min)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedExtremeValueAccumulatorFactory:
        """Return an accumulator factory for GEV raw-moment statistics."""
        return GeneralizedExtremeValueAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float, float]
    ) -> GeneralizedExtremeValueDistribution:
        """Estimate location, scale, and shape from weighted moments, keeping the data in support."""
        sum_x, sum_x2, sum_x3, count = suff_stat[:4]
        min_val, max_val = (float(suff_stat[4]), float(suff_stat[5])) if len(suff_stat) > 5 else (np.inf, -np.inf)
        if self.pseudo_count is not None and self.suff_stat is not None:
            mean0, second0, third0 = self.suff_stat
            sum_x += self.pseudo_count * mean0
            sum_x2 += self.pseudo_count * second0
            sum_x3 += self.pseudo_count * third0
            count += self.pseudo_count
        if count <= 0.0:
            return GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0, name=self.name, keys=self.keys)
        mean = sum_x / count
        var = sum_x2 / count - mean * mean
        if var <= 0.0:
            return GeneralizedExtremeValueDistribution(mean, self.min_scale, 0.0, name=self.name, keys=self.keys)
        m3 = sum_x3 / count - 3.0 * mean * (sum_x2 / count) + 2.0 * mean**3  # central third moment
        skew = m3 / var**1.5
        # A pseudo_count/suff_stat blend (real data + prior moments) can produce a skew estimate
        # that maps to a boundary xi where (g2 - g1*g1) is <= 0 -- an out-of-domain fractional
        # power (NaN) rather than a numerical accident; the un-blended path never reached this
        # region in practice. Fall back to the same safe Gumbel-shaped degenerate case the
        # var<=0.0 branch above already uses, rather than let a non-finite parameter reach the
        # constructor's validation as an opaque crash.
        gumbel_fallback = math.sqrt(6.0 * var) / math.pi, mean - math.sqrt(6.0 * var) / math.pi * _EULER, 0.0
        repairs: list[str] = []
        if not np.isfinite(skew):
            scale, loc, xi = gumbel_fallback
        else:
            xi_skew = _xi_from_skewness(skew, self.xi_min, self.xi_max)
            # Trade the third moment for support coverage before building the parameters: a shape
            # solved purely from skewness can put the bounded end of the GEV inside the very sample
            # it summarizes, which scores those observations -inf and leaves EM with a permanently
            # non-finite model (the moments never move, so the estimate never recovers).
            xi_cov = _shape_covering_range(xi_skew, mean, math.sqrt(var), min_val, max_val)
            xi = xi_cov
            if abs(xi) < _XI_TOL:  # Gumbel limit
                scale = math.sqrt(6.0 * var) / math.pi
                loc = mean - scale * _EULER
            else:
                g1, g2 = _gamma(1.0 - xi), _gamma(1.0 - 2.0 * xi)
                denom = g2 - g1 * g1
                if denom <= 0.0 or not np.isfinite(denom):
                    scale, loc, xi = gumbel_fallback
                else:
                    scale = math.sqrt(var) * abs(xi) / math.sqrt(denom)
                    loc = mean - scale * (g1 - 1.0) / xi
            if xi_cov != xi_skew and xi == xi_cov and np.isfinite(loc) and np.isfinite(scale):
                # The covering clamp rewrote the moment shape, so the returned parameters are not
                # the raw moment fit -- record the repair the same way the Gaussian records its
                # variance floor (MXR-080-1202). The clamp only guarantees the endpoint lies
                # *outside* the data; when the surviving margin is so thin that the extreme
                # observation still scores astronomically below what the Gumbel member would give
                # it, the clamped shape has effectively declared training data impossible after
                # all, and the estimate takes the (unbounded-support) Gumbel limit instead.
                extreme = min_val if xi > 0.0 else max_val
                candidate = GeneralizedExtremeValueDistribution(loc, max(scale, self.min_scale), xi)
                gumbel = GeneralizedExtremeValueDistribution(
                    gumbel_fallback[1], max(gumbel_fallback[0], self.min_scale), 0.0
                )
                if candidate.log_density(extreme) < gumbel.log_density(extreme) - _COVERING_FALLBACK_NATS:
                    scale, loc, xi = gumbel_fallback
                    repairs.append("shape-covering-fallback(%.6g -> gumbel)" % xi_skew)
                else:
                    repairs.append("shape-covering-clamped(%.6g -> %.6g)" % (xi_skew, xi))
        unfloored_scale = scale
        scale = max(scale, self.min_scale)
        if scale > unfloored_scale:
            repairs.append("scale-floored(%.3g -> %.3g)" % (unfloored_scale, self.min_scale))
        if not (np.isfinite(loc) and np.isfinite(scale) and np.isfinite(xi)):
            scale, loc, xi = gumbel_fallback
            scale = max(scale, self.min_scale)
        dist = GeneralizedExtremeValueDistribution(loc, scale, xi, name=self.name, keys=self.keys)
        if repairs:
            dist._numerical_repairs = tuple(repairs)
        return dist


class GeneralizedExtremeValueDataEncoder(DataSequenceEncoder):
    """Encode GEV observations as a float array."""

    def __str__(self) -> str:
        return "GeneralizedExtremeValueDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedExtremeValueDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return finite_observations(x, label="GEV observations")
