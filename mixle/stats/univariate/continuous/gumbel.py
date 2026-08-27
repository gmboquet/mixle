"""Gumbel distributions for extreme-value maxima.

Observations are real-valued floats. A Gumbel distribution with location ``loc`` (``mu``) and scale
``beta > 0`` has log-density

        log(f(x; mu, beta)) = -log(beta) - z - exp(-z),   z = (x - mu) / beta,

on the whole real line. It models the distribution of maxima; the mean is mu + beta*gamma (gamma the
Euler-Mascheroni constant) and the variance is (pi^2 / 6) * beta^2, which the moment estimator inverts.

The per-row log-density lowers cleanly to the shared symbolic kernel, so the family gets generated
NumPy, Torch, and Numba scoring through ``backend_log_density_from_params``.


Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.continuous._observation_contracts import (
    AnchoredMomentTrack,
    anchored_pooled_variance,
    consistent_anchored_triple,
    scored_observation,
    warn_uncorrectable_raw_moments,
)

_EULER_GAMMA = 0.5772156649015328606
_PI_SQRT6 = math.pi / math.sqrt(6.0)


class GumbelSuffStat(tuple):
    """A ``(sum, sum2, count)`` sufficient statistic that also carries a shift-anchored side payload.

    Behaves exactly like the plain 3-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one); ``anchored`` is the shift-anchored moment payload
    ``(anchor, sum_i w_i*(x_i - anchor), sum_i w_i*(x_i - anchor)^2)`` that :class:`GumbelAccumulator`
    maintains alongside the raw moments so the M-step's variance survives large-offset data. Code
    that doesn't know about the payload (generic ``scale_suff_stat``, engine kernels, ...) sees an
    ordinary tuple and the estimate falls back to the historical raw path.
    """

    def __new__(cls, sum_: float, sum2_: float, count_: float, anchored: tuple[float, float, float] | None = None):
        obj = super().__new__(cls, (sum_, sum2_, count_))
        obj.anchored = anchored
        return obj

    def __reduce__(self):
        # A plain tuple subclass with a payload-bearing __new__ does not pickle by default; the
        # Spark/mp reducers round-trip accumulator values through pickle, so keep the payload.
        return (_rebuild_gumbel_suff_stat, (tuple(self), self.anchored))


def _rebuild_gumbel_suff_stat(values: tuple, anchored: tuple | None) -> "GumbelSuffStat":
    """Unpickle helper for :class:`GumbelSuffStat` (module-level so pickle can import it)."""
    return GumbelSuffStat(values[0], values[1], values[2], anchored=anchored)


class GumbelDistribution(SequenceEncodableProbabilityDistribution):
    """Gumbel (extreme value type I) distribution with location loc and scale beta > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Gumbel kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Gumbel distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="gumbel",
            distribution_type=cls,
            parameters=(ParameterSpec("loc"), ParameterSpec("scale", constraint="positive")),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Gumbel sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx, xx * xx, xx * 0.0 + engine.asarray(1.0)

    @staticmethod
    def backend_log_density_from_params(x: Any, loc: Any, scale: Any, engine: Any) -> Any:
        """Engine-neutral Gumbel log-density from explicit parameters."""
        z = (x - loc) / scale
        return -engine.log(scale) - z - engine.exp(-z)

    def __init__(self, loc: float = 0.0, scale: float = 1.0, name: str | None = None, keys: str | None = None) -> None:
        """GumbelDistribution for location loc and scale beta.

        Args:
            loc (float): Location parameter mu (real valued).
            scale (float): Positive scale parameter beta.
            name (Optional[str]): Assign a name to GumbelDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        """
        if not np.isfinite(loc) or scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("GumbelDistribution requires finite loc and scale > 0.")
        self.loc = float(loc)
        self.scale = float(scale)
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the Gumbel distribution."""
        return "GumbelDistribution(loc=%s, scale=%s, name=%s, keys=%s)" % (
            repr(self.loc),
            repr(self.scale),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation."""
        try:
            coerced = float(x)
        except (TypeError, ValueError):
            return -np.inf
        xx = scored_observation(coerced, label="GumbelDistribution")
        z = (xx - self.loc) / self.scale
        # exp(-z) overflows on the far-left tail (z -> -inf); math.exp raises OverflowError there, so
        # short-circuit to the correct -inf limit instead of crashing on a valid observation.
        if -z > 709.0:
            return -np.inf
        return -self.log_scale - z - math.exp(-z)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (x - self.loc) / self.scale
        # exp(-z) overflows to +inf on the far-left tail; the result correctly -> -inf, so silence the
        # benign overflow warning rather than let it surface to the caller.
        with np.errstate(over="ignore"):
            return -self.log_scale - z - np.exp(-z)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.loc), engine.asarray(self.scale), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["GumbelDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Gumbel parameters for a homogeneous mixture kernel."""
        return {
            "loc": engine.asarray([d.loc for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Gumbel log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["loc"][None, :], params["scale"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked Gumbel sufficient statistics using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        return (
            engine.sum(ww * xx[:, None], axis=0),
            engine.sum(ww * (xx * xx)[:, None], axis=0),
            engine.sum(ww, axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x)."""
        return math.exp(-math.exp(-(float(x) - self.loc) / self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        return float(self.loc - self.scale * math.log(-math.log(float(q))))

    def mean(self) -> float:
        """Mean E[X] = loc + scale * euler_gamma."""
        import numpy as np

        return float(self.loc + self.scale * np.euler_gamma)

    def variance(self) -> float:
        """Variance Var[X] = pi^2 scale^2 / 6."""
        import math

        return float(math.pi * math.pi * self.scale * self.scale / 6.0)

    def entropy(self) -> float:
        """Differential entropy log(scale) + euler_gamma + 1."""
        import math

        import numpy as np

        return float(math.log(self.scale) + np.euler_gamma + 1.0)

    def skewness(self) -> float:
        """Skewness 12*sqrt(6)*zeta(3)/pi^3."""
        import math

        return float(12.0 * math.sqrt(6.0) * 1.2020569031595942 / math.pi**3)

    def kurtosis(self) -> float:
        """Excess kurtosis (12/5)."""
        return 2.4

    def mode(self) -> float:
        """Mode (= the location loc)."""
        return float(self.loc)

    def sampler(self, seed: int | None = None) -> "GumbelSampler":
        """Return a sampler for drawing observations from this distribution."""
        return GumbelSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GumbelEstimator":
        """Return a moment estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return GumbelEstimator(name=self.name, keys=self.keys)
        return GumbelEstimator(
            pseudo_count=pseudo_count, suff_stat=(self.loc, self.scale), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "GumbelDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return GumbelDataEncoder()


class GumbelSampler(DistributionSampler):
    """Draw iid Gumbel observations."""

    def __init__(self, dist: GumbelDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw ``size`` iid observations (a float when ``size`` is None)."""
        return self.rng.gumbel(self.dist.loc, self.dist.scale, size=size)


class GumbelAccumulator(AnchoredMomentTrack, SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for Gumbel estimation.

    Alongside the raw ``(sum, sum2, count)`` the accumulator keeps the conditioning-gated
    shift-anchored moment track of :class:`AnchoredMomentTrack`: the naive ``E[x^2]-E[x]^2`` form
    the moment inversion needs loses ~``2*log2(abs(mean)/sd)`` bits to cancellation, so Gumbel data
    with sd ~2 fitted a scale 8.8x too large at offset 1.7e9 and collapsed onto ``min_scale`` at
    2**31 -- silently, with an empty ``numerical_repairs()``. Well-conditioned data never activates
    the track and accumulates bit-identically to the historical single-pass path.
    """

    def __init__(self, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.keys = keys
        self._init_anchor()

    def update(self, x: float, weight: float, estimate: GumbelDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        xx = float(x)
        self._anchor_scalar(xx, weight)
        self.sum += xx * weight
        self.sum2 += xx * xx * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GumbelDistribution | None) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        self._anchor_fold_chunk(x, weights)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "GumbelAccumulator":
        """Merge another Gumbel sufficient-statistic tuple."""
        self._anchor_absorb(suff_stat)
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count.

        A plain 3-tuple for every consumer that treats it as one; once the shift-anchored moment
        track is live it is a :class:`GumbelSuffStat` additionally carrying the anchored moments in
        its ``.anchored`` attribute, so :meth:`combine` can fold them in and
        :meth:`GumbelEstimator.estimate` can invert a shift-invariant variance.
        """
        anchored = self._anchor_payload()
        if anchored is None:
            return self.sum, self.sum2, self.count
        return GumbelSuffStat(self.sum, self.sum2, self.count, anchored=anchored)

    def from_value(self, x: tuple[float, float, float]) -> "GumbelAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.count = x[0], x[1], x[2]
        self._anchor_restore(x)
        return self

    def scale(self, c: float) -> "GumbelAccumulator":
        """Scale the accumulated statistics in place by ``c``, anchored track included."""
        self._anchor_scale(c)
        self.sum *= c
        self.sum2 *= c
        self.count *= c
        return self

    def acc_to_encoder(self) -> "GumbelDataEncoder":
        """Return the encoder used by this accumulator."""
        return GumbelDataEncoder()


class GumbelAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GumbelAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> GumbelAccumulator:
        """Create a fresh Gumbel accumulator."""
        return GumbelAccumulator(keys=self.keys)


class GumbelEstimator(ParameterEstimator):
    """Moment estimator for the Gumbel location and scale.

    Inverts the Gumbel moments: ``beta = sqrt(6 * var) / pi`` and ``loc = mean - beta * gamma`` where
    gamma is the Euler-Mascheroni constant.

    The variance moment is formed from shift-anchored statistics whenever the accumulated
    sufficient statistics carry that payload (see :class:`GumbelAccumulator`), which makes the fit
    SHIFT-EQUIVARIANT: estimating on ``x + c`` returns ``loc + c`` with ``scale`` unchanged, for any
    constant ``c`` the data can carry -- epoch seconds (~1.7e9) and 2**31 included. With a plain raw
    tuple -- statistics the conditioning gate never needed to anchor, or ones that arrived already
    reduced from an engine kernel or an older serialization -- the historical raw path is used
    bit-identically, and ``estimate`` warns rather than returning a scale it cannot stand behind.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_scale: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GumbelAccumulatorFactory:
        """Return an accumulator factory for Gumbel moment statistics."""
        return GumbelAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> GumbelDistribution:
        """Estimate location and scale from weighted moments."""
        sum_x, sum_x2, count = float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2])
        # The anchored payload (when present) describes the RAW data only -- captured before any
        # prior blend below -- so it is read off the untouched (sum_x, count) pair.
        anchored = consistent_anchored_triple(suff_stat, sum_x, count)
        raw_count = count

        prior_mean: float | None = None
        prior_variance: float | None = None
        pc = self.pseudo_count
        if pc is not None and self.suff_stat is not None:
            loc0, scale0 = self.suff_stat
            prior_mean = loc0 + scale0 * _EULER_GAMMA
            prior_variance = (math.pi * math.pi / 6.0) * scale0 * scale0
            sum_x += pc * prior_mean
            sum_x2 += pc * (prior_variance + prior_mean * prior_mean)
            count += pc

        if count <= 0.0:
            return GumbelDistribution(name=self.name, keys=self.keys)

        mean = sum_x / count
        notes: tuple[str, ...] = ()
        if anchored is not None:
            # Only the variance loses to cancellation at large magnitude; the mean is a plain
            # same-sign sum and is computed from the raw (possibly prior-blended) moments above,
            # unchanged. The prior is folded into the variance separately, never through the
            # anchored sums (which only ever describe the observed data).
            var, notes = anchored_pooled_variance(
                anchored[0], anchored[1], anchored[2], raw_count, mean, pc, prior_mean, prior_variance
            )
        else:
            warn_uncorrectable_raw_moments(sum_x, sum_x2, count, family="Gumbel")
            var = max(sum_x2 / count - mean * mean, 0.0)
        scale = max(math.sqrt(max(var, 0.0)) / _PI_SQRT6, self.min_scale)
        loc = mean - scale * _EULER_GAMMA
        dist = GumbelDistribution(loc=loc, scale=scale, name=self.name, keys=self.keys)
        if notes:
            dist._numerical_repairs = notes
        return dist


class GumbelDataEncoder(DataSequenceEncoder):
    """Encode Gumbel observations as a float array."""

    def __str__(self) -> str:
        return "GumbelDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GumbelDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        rv = np.asarray(x, dtype=np.float64)
        # "real-valued" excludes the infinities: the vectorized tail formula evaluates
        # ``inf - inf`` at ``x = -inf`` and would hand back NaN for an observation the
        # scalar scorer answers as the -inf limit. Refuse it on both paths instead.
        if rv.size and np.any(~np.isfinite(rv)):
            raise ValueError("GumbelDistribution requires real-valued observations.")
        return rv
