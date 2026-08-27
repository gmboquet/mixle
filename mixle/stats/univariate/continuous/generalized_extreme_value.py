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

The M-step is *shift-equivariant*: fitting ``x + c`` returns ``loc + c`` with an unchanged ``scale``
and ``shape``. That does not come for free from raw power sums -- the central third moment
differenced out of ``E[x^3]`` loses about ``3*log2(abs(mean)/sd)`` bits, and the variance about twice
that -- so the accumulator carries a conditioning-gated shift-anchored moment track alongside the raw
sums (see :class:`GeneralizedExtremeValueAccumulator`).

Reference: Coles, *An Introduction to Statistical Modeling of Extreme Values* (Springer, 2001).
"""

import math
import warnings
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
    warn_uncorrectable_raw_moments,
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

# Conditioning threshold for the anchored-moment gate. The M-step's highest-order reduced moment is
# the third, whose raw form ``E[x^3] - 3 m E[x^2] + 2 m^3`` loses about ``eps * (|mean|/sd)^3``
# relative accuracy; a (mean/sd)^2 up to 2.5e4 (ratio ~158) keeps that within ~1e-9, so the
# historical single-pass path is bit-preserved there and the anchored track takes over beyond it.
# Chunks pooled from gate-passing content stay well-conditioned as a pool (Cauchy-Schwarz:
# n*mean_pool^2 <= sum_i n_i*mean_i^2), so a pool built only from gate-passing chunks never needs the
# anchor retroactively.
_ANCHOR_CONDITION_RATIO = 2.5e4


def _needs_anchor(chunk_sum: float, chunk_sum2: float, w_sum: float) -> bool:
    """Whether a chunk's weighted moments are too ill-conditioned for the raw reduced-moment form.

    ``spread2`` computed here is itself the cancellation-prone estimate, but as a GATE it is
    reliable: when cancellation has corrupted it, the corruption is bounded by ``eps * m^2``, which
    still leaves ``m*m`` orders of magnitude above ``_ANCHOR_CONDITION_RATIO * spread2``.
    A non-positive computed spread activates the anchor outright (constant or near-constant data).
    """
    m = chunk_sum / w_sum
    spread2 = chunk_sum2 / w_sum - m * m
    return spread2 <= 0.0 or m * m > _ANCHOR_CONDITION_RATIO * spread2


def _shift_moments(m0: float, m1: float, m2: float, m3: float, d: float) -> tuple[float, float, float]:
    """Re-express weighted power sums accumulated about a point sitting ``d`` above the new one.

    Given ``m_k = sum_i w_i y_i^k`` (with ``m0 = sum_i w_i``), return the same sums for ``y_i + d``.
    Used both to convert raw sums onto an anchor (``d = -anchor``) and to fold a differently anchored
    partner in :meth:`GeneralizedExtremeValueAccumulator.combine`.
    """
    d2 = d * d
    return (
        m1 + d * m0,
        m2 + 2.0 * d * m1 + d2 * m0,
        m3 + 3.0 * d * m2 + 3.0 * d2 * m1 + d2 * d * m0,
    )


class GeneralizedExtremeValueSuffStat(tuple):
    """A ``(sum, sum2, sum3, count, min_val, max_val)`` statistic that also carries a side payload.

    Behaves exactly like the plain 6-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one); ``anchored`` is the shift-anchored payload
    ``(anchor, sum_i w_i (x_i - anchor)^k for k = 1..3)`` the accumulator maintains alongside the raw
    power sums so the M-step survives large-offset data. Code that doesn't know about the payload
    (generic ``scale_suff_stat``, serializers, ...) sees an ordinary tuple and the estimate falls
    back to the historical raw path.
    """

    def __new__(
        cls,
        sum_: float,
        sum2_: float,
        sum3_: float,
        count_: float,
        min_val: float,
        max_val: float,
        anchored: tuple[float, float, float, float] | None = None,
    ) -> "GeneralizedExtremeValueSuffStat":
        obj = super().__new__(cls, (sum_, sum2_, sum3_, count_, min_val, max_val))
        obj.anchored = anchored
        return obj

    def __reduce__(self):
        # A tuple subclass with a payload-bearing __new__ does not pickle by default, and the
        # Spark/multiprocessing reducers round-trip accumulator values through pickle.
        return (_rebuild_gev_suff_stat, (tuple(self), self.anchored))


def _rebuild_gev_suff_stat(values: tuple, anchored: tuple | None) -> GeneralizedExtremeValueSuffStat:
    """Unpickle helper for :class:`GeneralizedExtremeValueSuffStat` (module level so pickle can import it)."""
    return GeneralizedExtremeValueSuffStat(*values, anchored=anchored)


def _consistent_anchored_moments(suff_stat: Any, sum_x: float, count: float) -> tuple[float, ...] | None:
    """Return the anchored payload of ``suff_stat`` when it is usable, else ``None``.

    ``None`` falls back to the raw reduced-moment M-step, so a payload is only trusted when it is
    finite, has the right arity, carries a non-negative second order, and agrees with the raw first
    moment it claims to describe -- a hand-built :class:`GeneralizedExtremeValueSuffStat` whose
    payload contradicts its tuple must not silently change the estimate the tuple alone would have
    produced.
    """
    anchored = getattr(suff_stat, "anchored", None)
    if anchored is None or count <= 0.0:
        return None
    if len(anchored) != 4 or not all(np.isfinite(v) for v in anchored):
        return None
    anchor, a1, a2, _a3 = (float(v) for v in anchored)
    if a2 < 0.0:
        return None
    implied_sum = a1 + count * anchor
    tolerance = 1.0e-6 * max(abs(sum_x), abs(count * anchor), 1.0)
    if abs(implied_sum - sum_x) > tolerance:
        return None
    return tuple(float(v) for v in anchored)


def _prior_is_ill_conditioned(raw_prior: Any) -> bool:
    """Whether raw prior moments ``(E[X], E[X^2], E[X^3])`` have lost their spread to cancellation.

    A diagnostic, not a gate: it decides whether to *warn*, never whether to reject. The prior is a
    one-unit pseudo-sample, so the accumulator's own conditioning test applies with ``w_sum = 1``.
    """
    try:
        e1, e2 = float(raw_prior[0]), float(raw_prior[1])
    except (TypeError, ValueError, IndexError):
        return False
    if not (np.isfinite(e1) and np.isfinite(e2)):
        return False
    return _needs_anchor(e1, e2, 1.0)


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
    the mean, independent of location and scale: substituting ``sigma = sd * abs(xi) / sqrt(g2 - g1^2)``
    and ``mu = mean - sigma (g1 - 1)/xi`` into the endpoint ``mu - sigma/xi`` leaves
    ``mean +/- sd g1 / sqrt(g2 - g1^2)``. The value diverges as ``xi -> 0`` (Gumbel is unbounded)
    and shrinks monotonically as ``abs(xi)`` grows -- which is exactly why an unconstrained
    moment-matched shape can put the endpoint inside the sample it was fit on.
    """
    g1, g2 = _gamma(1.0 - xi), _gamma(1.0 - 2.0 * xi)
    denom = g2 - g1 * g1
    if denom <= 0.0 or not np.isfinite(denom):
        return np.inf
    return float(g1 / math.sqrt(denom))


def _shape_covering_range(xi: float, mean: float, sd: float, min_val: float, max_val: float) -> float:
    """Shrink ``abs(xi)`` until the moment-matched support contains the observed range.

    Method of moments matches mean, variance and skewness but says nothing about *support*: with
    ``xi < 0`` a GEV is bounded above, and a sample whose skew maps to, say, ``xi = -0.72`` gets an
    upper endpoint only 1.85 sd above the mean -- so the largest observations in the very sample
    being fit score ``-inf``. That is not a numerical accident; it is an estimate that declares its
    own training data impossible, and downstream EM cannot recover from it because the moments (and
    therefore the estimate) never change.

    Give up the third moment rather than the support: keep mean and variance matched exactly and
    move ``xi`` toward the Gumbel limit, which pushes the endpoint outward monotonically, until the
    observed range is strictly inside it. ``abs(xi)`` is only ever reduced, so a shape whose support
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
        """Engine-neutral GEV log-density; the ``abs(xi) < tol`` Gumbel limit is selected per element.

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
            pseudo_count=pseudo_count,
            suff_stat=(mean0, second0, third0),
            name=self.name,
            keys=self.keys,
            # The raw moments above are the release-pinned exchange form, but at a large |loc| they
            # no longer contain this distribution's spread at all (``var0 + mean0**2`` rounds
            # ``var0`` away once ``mean0**2`` exceeds ~1e16 times it). Carry the central restatement
            # alongside them so estimate() can place the prior on the data anchor exactly.
            prior_central=(mean0, var0, skew0 * (var0**1.5)),
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

    Alongside the raw sums the accumulator keeps a CONDITIONING-GATED shift-anchored track,
    ``sum_i w_i (x_i - anchor)^k`` for ``k = 1..3`` about a data anchor. The method of moments needs
    central moments, and differencing them out of raw power sums is the classic cancellation-prone
    form: at sd ~0.6 and offset 1.7e9 the raw variance is wrong by a factor of thousands and the raw
    third moment has no correct digits at all, which drives the shape onto its bound and the scale
    tens of times too large -- silently. Anchoring keeps every term of the scatter
    ``O(count * spread^3)``, making the M-step shift-equivariant.

    The gate keeps the historical path bit-identical for well-conditioned data: a chunk whose
    ``abs(mean)/spread`` ratio the raw form handles to ~1e-9 relative (see :func:`_needs_anchor`)
    accumulates exactly the way it always did, with no anchor and no second pass. The raw sums remain
    the exchange format, so the anchored track rides along as a payload on
    :class:`GeneralizedExtremeValueSuffStat`; a consumer that drops the payload simply gets the
    historical raw estimate back.
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
        self._anchor: float | None = None
        self._a1 = 0.0
        self._a2 = 0.0
        self._a3 = 0.0
        self._anchor_unrecoverable = False

    def _absorb_raw(self, count: float, sum_: float, sum2: float, sum3: float) -> None:
        """Fold raw power sums into the live anchored track, converting them about the anchor.

        The conversion is itself the cancellation-prone form, and it is safe on content the gate has
        already certified as well-conditioned (raw error ~1e-9 relative or better).

        It is NOT safe on ill-conditioned raw statistics arriving through ``from_value``/``combine``:
        power sums whose own ``abs(mean)/spread`` ratio has already erased the central moments cannot
        have them restored by any change of reference point, and converting them anyway seeds the
        anchored track with an error far larger than the spread it is supposed to measure -- which
        would make the pooled estimate *worse* than the historical raw one, not better. Such content
        marks the track unrecoverable: :meth:`value` then withholds the anchored payload, the estimate
        falls back to exactly the historical raw M-step, and the caller is told why.
        """
        if count == 0.0 and sum_ == 0.0 and sum2 == 0.0 and sum3 == 0.0:
            return
        if count > 0.0 and _needs_anchor(sum_, sum2, count):
            self._anchor_unrecoverable = True
            warnings.warn(
                "GeneralizedExtremeValueAccumulator merged raw power sums whose location dominates "
                "their spread into a shift-anchored pool. Raw sums at that conditioning no longer "
                "contain the central moments, and no change of reference point can restore them, so "
                "this pool falls back to the historical raw M-step and its scale/shape are "
                "unreliable at this offset. Accumulate through update()/seq_update(), or combine "
                "statistics that still carry their anchored payload, instead of restoring plain "
                "power sums.",
                RuntimeWarning,
                stacklevel=3,
            )
        a1, a2, a3 = _shift_moments(count, sum_, sum2, sum3, -self._anchor)
        self._a1 += a1
        self._a2 += max(a2, 0.0)
        self._a3 += a3

    def _activate_anchor(self, anchor: float) -> None:
        """Start the shift-anchored moment track at ``anchor``, converting any raw content onto it."""
        self._anchor = float(anchor)
        self._absorb_raw(self.count, self.sum, self.sum2, self.sum3)

    def update(self, x: float, weight: float, estimate: GeneralizedExtremeValueDistribution | None) -> None:
        """Accumulate weighted first three raw moments and the observed range for one observation."""
        # Scalar updates carry no chunk to assess conditioning from, so the anchor activates on the
        # first observation (O(1) bookkeeping on this path). Activation happens BEFORE the raw fold
        # so any pre-anchor content is converted from statistics the gate has already vouched for.
        if self._anchor is None:
            self._activate_anchor(float(x))
        dx = float(x) - self._anchor
        self._a1 += dx * weight
        self._a2 += dx * dx * weight
        self._a3 += dx * dx * dx * weight
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
        w_sum = float(np.sum(weights, dtype=np.float64))
        chunk_sum = float(np.dot(xx, weights))
        chunk_sum2 = float(np.dot(xx * xx, weights))
        # Conditioning gate: activate the anchored track only when this chunk's raw moments would
        # corrupt the reduced moments (or the anchor is already live). BEFORE the raw fold, so
        # activation converts only the content that preceded this chunk.
        if len(xx) > 0 and (self._anchor is not None or (w_sum > 0.0 and _needs_anchor(chunk_sum, chunk_sum2, w_sum))):
            if self._anchor is None:
                self._activate_anchor(float(xx[0]))
            dx = xx - self._anchor
            dx2 = dx * dx
            self._a1 += float(np.dot(dx, weights))
            self._a2 += float(np.dot(dx2, weights))
            self._a3 += float(np.dot(dx2 * dx, weights))
        self.sum += chunk_sum
        self.sum2 += chunk_sum2
        self.sum3 += float(np.dot(xx * xx * xx, weights))
        self.count += w_sum
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
        anchored = getattr(suff_stat, "anchored", None)
        if anchored is not None and len(anchored) == 4:
            # Re-express the incoming anchored moments about this accumulator's anchor. The anchor
            # gap ``d`` is between two data values, so every term stays O(count * spread^3).
            # Activation (when this side has no anchor yet) runs BEFORE the raw fold below so the
            # pre-existing raw content is converted exactly once.
            b_anchor, b1, b2, b3 = (float(v) for v in anchored)
            if self._anchor is None:
                self._activate_anchor(b_anchor)
            f1, f2, f3 = _shift_moments(float(suff_stat[3]), b1, b2, b3, b_anchor - self._anchor)
            self._a1 += f1
            self._a2 += max(f2, 0.0)
            self._a3 += f3
        elif self._anchor is not None:
            # Raw-only statistics joining an anchored pool -- the mirror of the case above. Same
            # conversion, same recoverability check: see :meth:`_absorb_raw`.
            self._absorb_raw(float(suff_stat[3]), float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2]))
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        if len(suff_stat) > 5:
            self.min_val = min(self.min_val, float(suff_stat[4]))
            self.max_val = max(self.max_val, float(suff_stat[5]))
        return self

    def value(self) -> tuple[float, float, float, float, float, float]:
        """Return raw moment sums, observation count, and the observed range.

        The returned object is a plain 6-tuple for every consumer that treats it as one; once the
        shift-anchored track is live it additionally carries the anchored moments in its
        ``.anchored`` attribute, so :meth:`combine` can fold them in and
        :meth:`GeneralizedExtremeValueEstimator.estimate` can use them for the reduced moments. The
        payload is withheld when the track was seeded from raw statistics that had already lost their
        central moments (see :meth:`_activate_anchor`), so a pool that cannot honestly claim
        shift-equivariance reports the historical raw statistics instead of a worse anchored guess.
        """
        if self._anchor is None or self._anchor_unrecoverable:
            return self.sum, self.sum2, self.sum3, self.count, self.min_val, self.max_val
        return GeneralizedExtremeValueSuffStat(
            self.sum,
            self.sum2,
            self.sum3,
            self.count,
            self.min_val,
            self.max_val,
            anchored=(self._anchor, self._a1, self._a2, self._a3),
        )

    def from_value(self, x: tuple[float, float, float, float, float, float]) -> "GeneralizedExtremeValueAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.sum3, self.count = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        # A four-entry tuple predates the range statistic and simply leaves the support unknown --
        # estimate() then skips the coverage clamp rather than inventing a bound.
        self.min_val = float(x[4]) if len(x) > 5 else np.inf
        self.max_val = float(x[5]) if len(x) > 5 else -np.inf
        anchored = getattr(x, "anchored", None)
        self._anchor_unrecoverable = False
        if anchored is not None and len(anchored) == 4:
            self._anchor, self._a1, self._a2, self._a3 = (float(v) for v in anchored)
        else:
            # Raw-only statistics replace the state: the anchored track restarts unactivated, and a
            # later activation (first update / anchored merge) converts this content then.
            self._anchor = None
            self._a1 = self._a2 = self._a3 = 0.0
        return self

    def scale(self, c: float) -> "GeneralizedExtremeValueAccumulator":
        """Scale accumulated evidence while preserving the observed range.

        The shift-anchored track scales with the raw sums; leaving it behind would turn a scaled
        accumulator back into the ill-conditioned raw path.
        """
        self.sum *= c
        self.sum2 *= c
        self.sum3 *= c
        self.count *= c
        self._a1 *= c
        self._a2 *= c
        self._a3 *= c
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


def _spread_is_resolvable(variance: float, magnitude: float) -> bool:
    """Whether a spread of ``sqrt(variance)`` is representable at all at scale ``magnitude``.

    Used only to decide what to DISCLOSE through ``numerical_repairs()``, never to clamp -- see the
    identical predicate on the Gaussian family
    (``mixle.stats.univariate.continuous.gaussian._spread_is_resolvable``) for the full rationale,
    which applies unchanged here.
    """
    if not np.isfinite(magnitude) or magnitude <= 0.0 or not np.isfinite(variance) or variance <= 0.0:
        return False
    half_ulp = 0.5 * float(np.spacing(magnitude))
    return variance > half_ulp * half_ulp


# Bound on how far the reported location can sit from the exact sample mean the anchored track
# knows: ~4-8 grid steps of ``|mean|``. Bounds a rounding residue of the mean, not a spread -- same
# constant the Gaussian family's own bound uses, so every degenerate payload that collapsed to
# exactly zero under the old whole-scatter clamp still does.
_MEAN_ROUNDING_BOUND = 8.8817841970012523e-16  # 4 * eps


def _anchored_central_moments(
    ref: float,
    n: float,
    a1: float,
    a2: float,
    a3: float,
    delta: float,
    pc: float,
    prior: tuple[float, float, float] | None,
    prior_central: tuple[float, float, float] | None,
) -> tuple[float, float, tuple[str, ...]]:
    """Second and third central moments about ``ref + delta``, from shift-anchored data moments.

    ``ref``/``n``/``a1..a3`` describe the DATA alone (the accumulator's anchored track, never
    touched by any prior blend); ``delta`` is the SMALL offset from ``ref`` to the location the
    estimator actually reports (``mean = ref + delta``), which can differ from the data's own mean
    when a pseudo-count prior pulls it -- ``delta`` is passed rather than ``mean`` itself because
    every displacement below is computed in this small, already-offset coordinate, never by
    differencing two separately-materialized ``O(magnitude)`` floats (``mean_data - mean``): that
    subtraction would reintroduce, in the DISPLACEMENT, exactly the ``ulp(magnitude)``-scale
    rounding the anchor track exists to keep out of the SPREAD, and it is invisible at the loose
    tolerances the clamp itself needs (an absolute ulp of ~1e-7 buried in a threshold check of
    ~1e-6) but not at the tight ones a prior blend must hit (see
    ``campaign3b_families_test.py``'s
    ``test_pseudo_count_prior_at_a_large_location_is_blended_exactly``, which pinned this down to
    7 decimal places). Mirrors
    :func:`mixle.stats.univariate.continuous.gaussian._anchored_pooled_variance`, extended to a
    third bracket for the third-order skewness numerator by phrasing it as "recenter each group's
    own central moments onto ``mean``, then pool":

    1. ``core2``/``core3`` are the DATA's own central moments about its own mean -- every term is
       ``O(spread^k)``, computed entirely at small magnitude, and this is where all the data's real
       spread lives. Only ``core2`` is gated, by the same RELATIVE 1e-12 cancellation clamp the
       Gaussian family uses (``core2`` is a difference of two ``O(spread^2)`` quantities); when it
       clamps to zero, ``core3`` clamps with it, since a truly-degenerate sample has every central
       moment exactly zero, not just its second.
    2. ``shift_data = delta_data - delta`` is the displacement of the reported location from the
       data's own mean, computed ENTIRELY in small-offset coordinates (both terms are ``O(spread)``
       unless a prior has pulled ``delta`` far from the data, in which case ``shift_data`` correctly
       comes out ``O(delta)`` with no cancellation either way). Below the mean's own rounding
       granularity it is pure rounding on the plain ML path (where ``delta`` is computed from the
       SAME anchored sums as ``delta_data`` and the two agree exactly unless a prior is blended in),
       so it alone gets the absolute ulp-scale clamp. Recentering the data's own central moments onto
       ``mean`` from its own mean is the single-group parallel-axis expansion,
       ``E[(Y+d)^3] = m3 + 3 d m2 + d^3`` (``E[Y]=0``, ``d=shift_data``): a polynomial evaluation,
       never a cancellation, so ``shift_data`` may legitimately be large (a real prior pull) without
       needing any further clamp once it has cleared the rounding check.
    3. When a pseudo-count prior is blended in, it contributes as a second "group" of weight ``pc``
       with its own central moments (from ``prior_central`` when available -- exact at any
       magnitude -- else recovered from the raw power-sum payload ``prior`` the same, already-warned,
       degraded way the raw M-step blend has always used) recentered onto ``mean`` the identical way
       (``shift_prior = prior[0] - delta``, ``prior[0]`` already being the prior's own small offset
       from ``ref`` that :meth:`GeneralizedExtremeValueEstimator._prior_about` computed), and the two
       groups are pooled by weight. No clamp applies to the prior's own recentering displacement:
       like the Gaussian family's ``prior_scatter`` term, it is an explicit additive contribution,
       not a cancellation, and reporting it in full (even when large) is correct -- mixing in a prior
       whose mean sits far from the data legitimately inflates the pooled spread.
    """
    delta_data = a1 / n
    r2d, r3d = a2 / n, a3 / n
    core2 = r2d - delta_data * delta_data
    core3 = r3d - 3.0 * delta_data * r2d + 2.0 * delta_data**3
    noise = 1.0e-12 * max(abs(r2d), delta_data * delta_data, 1.0e-300)
    repairs: tuple[str, ...] = ()
    if core2 < noise:
        if _spread_is_resolvable(core2, abs(ref)):
            repairs = ("spread-below-noise(%.3g of %.3g)" % (core2, noise),)
        core2 = 0.0
        core3 = 0.0
    core2 = max(core2, 0.0)
    shift_data = delta_data - delta
    mean = ref + delta  # materialized only for the ulp-scale threshold's magnitude, never differenced
    if abs(shift_data) <= _MEAN_ROUNDING_BOUND * max(abs(mean), abs(ref)):
        shift_data = 0.0
    data2 = core2 + shift_data * shift_data
    data3 = core3 + 3.0 * shift_data * core2 + shift_data**3
    if pc <= 0.0 or prior is None:
        return data2, data3, repairs
    if prior_central is not None:
        _, c2, c3 = prior_central
    else:
        # Degraded fallback (the caller has already warned): recover the prior's own central moments
        # by un-shifting its raw power-sum-about-ref payload. Cancellation-prone when the prior's own
        # location sits far from ref -- the same pre-existing limitation the warning discloses, no
        # worse than the un-split blend.
        _, c2, c3 = _shift_moments(1.0, prior[0], prior[1], prior[2], -prior[0])
        c2 = max(c2, 0.0)
    shift_prior = prior[0] - delta
    prior2 = c2 + shift_prior * shift_prior
    prior3 = c3 + 3.0 * shift_prior * c2 + shift_prior**3
    total = n + pc
    return (n * data2 + pc * prior2) / total, (n * data3 + pc * prior3) / total, repairs


class GeneralizedExtremeValueEstimator(ParameterEstimator):
    """Method-of-moments estimator for GEV location, scale and shape.

    This is a moment estimator, not an MLE: the shape is solved from the skewness-vs-``xi``
    relation, and the shape is additionally constrained so the fitted support covers the observed
    range (see :func:`_shape_covering_range`). Whenever that constraint rewrites the shape -- or the
    rewritten shape still scores the extreme observation catastrophically and the estimate falls
    back to the Gumbel limit -- the returned distribution is not the raw moment fit, and the change
    is recorded in ``numerical_repairs()`` so ``fit_provenance()`` carries it (MXR-080-1202).

    The reduced moments are formed about a reference point rather than about zero whenever the
    accumulated statistics carry a shift-anchored payload (see
    :class:`GeneralizedExtremeValueAccumulator`), which makes the fit shift-equivariant: ``estimate``
    on ``x + c`` returns ``loc + c`` with ``scale`` and ``shape`` unchanged. With a plain raw tuple --
    statistics restored from an older serialization, or a hand-built one -- the historical raw path
    is used unchanged.
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
        prior_central: tuple[float, float, float] | None = None,
    ) -> None:
        """Create a method-of-moments estimator.

        ``prior_central`` is an optional ``(mean, variance, third central moment)`` restatement of the
        raw ``suff_stat`` moments, supplied by
        :meth:`GeneralizedExtremeValueDistribution.estimator`. Raw prior moments at a large location
        are not recoverable in float64 (``E[X^2]`` at ``loc = 1.7e9`` has an ulp of 512, so a
        variance of 1 is simply not present in the number); when the data needs the anchored track,
        this payload lets the prior be placed on the anchor exactly instead. Without it a
        large-location prior is still blended the historical raw way, and ``estimate`` warns rather
        than pretending the blend was well-conditioned.
        """
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
        central = None if prior_central is None else tuple(float(value) for value in prior_central)
        if central is not None and (len(central) != 3 or not all(np.isfinite(value) for value in central)):
            raise ValueError("GEV prior_central moments must be a finite length-three tuple.")
        self.pseudo_count = None if pseudo_count == 0.0 else pseudo_count
        self.suff_stat = prior
        self.prior_central = central
        self.min_scale = float(min_scale)
        self.xi_max = float(xi_max)
        self.xi_min = float(xi_min)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedExtremeValueAccumulatorFactory:
        """Return an accumulator factory for GEV raw-moment statistics."""
        return GeneralizedExtremeValueAccumulatorFactory(name=self.name, keys=self.keys)

    def _prior_about(self, ref: float) -> tuple[float, float, float] | None:
        """Prior power sums for one unit of pseudo-count, expressed about ``ref``.

        Returns ``None`` when there is no prior. Uses the central payload when available (exact at
        any reference point); otherwise shifts the stored raw moments, which is only well-conditioned
        when ``ref`` is zero or the prior's own location is small.
        """
        if self.pseudo_count is None or self.suff_stat is None:
            return None
        e1, e2, e3 = (float(v) for v in self.suff_stat)
        if ref == 0.0:
            # The stored raw moments ARE the power sums about zero; using them verbatim keeps the
            # historical blend bit-identical even when a central payload is also available.
            return e1, e2, e3
        # getattr, not attribute access: an estimator unpickled from a release that predates this
        # field has no such attribute, and the right answer there is the historical raw shift.
        central = getattr(self, "prior_central", None)
        if central is not None:
            mean0, c2, c3 = central
            u = mean0 - ref
            return (u, c2 + u * u, c3 + 3.0 * u * c2 + u * u * u)
        return _shift_moments(1.0, e1, e2, e3, -ref)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float, float]
    ) -> GeneralizedExtremeValueDistribution:
        """Estimate location, scale, and shape from weighted moments, keeping the data in support."""
        sum_x, sum_x2, sum_x3, count = (float(v) for v in suff_stat[:4])
        min_val, max_val = (float(suff_stat[4]), float(suff_stat[5])) if len(suff_stat) > 5 else (np.inf, -np.inf)
        anchored = _consistent_anchored_moments(suff_stat, sum_x, count)
        # Everything below is the historical algebra with an explicit reference point: at ref = 0 the
        # power sums ARE the raw sums and every formula reduces to exactly what it was before.
        if anchored is None:
            # Raw-only statistics cannot be corrected here; before this the family was silent, and
            # sd ~2 data at offset 1.7e9 handed in as the declared raw tuple returned scale = 1e-12
            # (the min_scale floor) for a true 1.4903. The gate reads the second moment, which is
            # where the loss shows up first; a third-moment-only loss cannot occur without it.
            warn_uncorrectable_raw_moments(sum_x, sum_x2, count, family="generalized extreme value")
            ref, p1, p2, p3 = 0.0, sum_x, sum_x2, sum_x3
        else:
            ref, p1, p2, p3 = anchored
        n, a1, a2, a3 = count, p1, p2, p3  # data-only weight/moments, before any prior blend
        repairs: list[str] = []
        prior = self._prior_about(ref)
        pc = 0.0
        if prior is not None:
            if (
                anchored is not None
                and getattr(self, "prior_central", None) is None
                and _prior_is_ill_conditioned(self.suff_stat)
            ):
                warnings.warn(
                    "GeneralizedExtremeValueEstimator is blending raw prior moments whose own "
                    "location dominates their spread into data that needed shift-anchored "
                    "accumulation; the prior's central moments cannot be recovered from raw float64 "
                    "power sums, so the blended scale/shape are unreliable. Build the prior with "
                    "GeneralizedExtremeValueDistribution.estimator(pseudo_count=...) (which carries "
                    "the central moments), or pass prior_central=(mean, variance, third_central).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                repairs.append("prior-moments-ill-conditioned")
            pc = float(self.pseudo_count)
            p1 += pc * prior[0]
            p2 += pc * prior[1]
            p3 += pc * prior[2]
            count += pc
        if count <= 0.0:
            return GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0, name=self.name, keys=self.keys)
        delta = p1 / count  # the mean, measured from ref
        mean = ref + delta
        if anchored is not None:
            # Anchored path: the data's own moments (a1..a3) are already O(spread) about ref, so
            # split the second/third central moments about mean into a data-only "core" (well
            # conditioned, gated by the RELATIVE cancellation clamp) plus the displacement of mean
            # from the data's own mean (gated by the ulp-scale clamp) -- see
            # _anchored_central_moments. This never differences two O(magnitude^2) quantities, unlike
            # computing var/m3 directly from the prior-blended p2/p3, so genuine spread at extreme
            # magnitude survives.
            var, m3, moment_repairs = _anchored_central_moments(
                ref, n, a1, a2, a3, delta, pc, prior, getattr(self, "prior_central", None)
            )
            repairs.extend(moment_repairs)
        else:
            # Historical raw path (ref = 0): bit-identical to before the split.
            r2, r3 = p2 / count, p3 / count
            var = r2 - delta * delta
            m3 = r3 - 3.0 * delta * r2 + 2.0 * delta**3
        if var <= 0.0:
            return GeneralizedExtremeValueDistribution(mean, self.min_scale, 0.0, name=self.name, keys=self.keys)
        skew = m3 / var**1.5
        # A pseudo_count/suff_stat blend (real data + prior moments) can produce a skew estimate
        # that maps to a boundary xi where (g2 - g1*g1) is <= 0 -- an out-of-domain fractional
        # power (NaN) rather than a numerical accident; the un-blended path never reached this
        # region in practice. Fall back to the same safe Gumbel-shaped degenerate case the
        # var<=0.0 branch above already uses, rather than let a non-finite parameter reach the
        # constructor's validation as an opaque crash.
        gumbel_fallback = math.sqrt(6.0 * var) / math.pi, mean - math.sqrt(6.0 * var) / math.pi * _EULER, 0.0
        if not np.isfinite(skew):
            scale, loc, xi = gumbel_fallback
        else:
            xi_skew = _xi_from_skewness(skew, self.xi_min, self.xi_max)
            # Trade the third moment for support coverage before building the parameters: a shape
            # solved purely from skewness can put the bounded end of the GEV inside the very sample
            # it summarizes, which scores those observations -inf and leaves EM with a permanently
            # non-finite model (the moments never move, so the estimate never recovers).
            # In coordinates centered on the mean. ``_shape_covering_range`` only ever reads the two
            # gaps ``max_val - mean`` and ``mean - min_val``, so subtracting the reference point
            # first is algebraically identical -- and at ref = 0 it is the same float expression the
            # raw path always evaluated -- but on the anchored path it removes the last place where
            # the M-step differences two numbers of the offset's magnitude. That mattered: the
            # clamp's margin is 1e-6 relative, and the rounding of ``max_val - mean`` at offset 1.7e9
            # is ~3e-7 relative, so a shape sitting near the covering threshold could be clamped at
            # one offset and not at another.
            xi_cov = _shape_covering_range(xi_skew, 0.0, math.sqrt(var), min_val - ref - delta, max_val - ref - delta)
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
