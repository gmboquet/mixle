"""Location-scale logistic distributions over real values.

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
    consistent_anchored_triple,
    scale_anchored_triple,
    scored_observation,
)


class LogisticDistribution(SequenceEncodableProbabilityDistribution):
    """Logistic distribution with location loc and scale > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated logistic kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for logistic distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="logistic",
            distribution_type=cls,
            parameters=(ParameterSpec("loc"), ParameterSpec("scale", constraint="positive")),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Logistic sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx, xx * xx, xx * 0.0 + engine.asarray(1.0)

    def __init__(self, loc: float = 0.0, scale: float = 1.0, name: str | None = None, keys: str | None = None) -> None:
        if not np.isfinite(loc) or scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("LogisticDistribution requires finite loc and scale > 0.")
        self.loc = float(loc)
        self.scale = float(scale)
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
        return "LogisticDistribution(loc=%s, scale=%s, name=%s, keys=%s)" % (
            repr(self.loc),
            repr(self.scale),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        xx = scored_observation(x, label="LogisticDistribution", allow_infinite=True)
        # |z| keeps the algebraically identical branch that stays finite in both tails:
        # -z - 2*log1p(exp(-z)) evaluates to inf - inf at z = -inf, which is the NaN the
        # engine-neutral kernel already avoids by branching on the sign of z.
        a = abs((xx - self.loc) / self.scale)
        return -self.log_scale - a - 2.0 * float(np.logaddexp(0.0, -a))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        a = np.abs((x - self.loc) / self.scale)
        return -self.log_scale - a - 2.0 * np.logaddexp(0.0, -a)

    @staticmethod
    def backend_log_density_from_params(x: Any, loc: Any, scale: Any, engine: Any) -> Any:
        """Engine-neutral logistic log-density from explicit parameters."""
        z = (x - loc) / scale
        log_scale = engine.log(scale)
        pos = -log_scale - z - engine.asarray(2.0) * engine.log(engine.asarray(1.0) + engine.exp(-z))
        neg = -log_scale + z - engine.asarray(2.0) * engine.log(engine.asarray(1.0) + engine.exp(z))
        return engine.where(z >= 0.0, pos, neg)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.loc), engine.asarray(self.scale), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["LogisticDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked logistic parameters for a homogeneous mixture kernel."""
        return {
            "loc": engine.asarray([d.loc for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of logistic log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["loc"][None, :], params["scale"][None, :], engine
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import logistic as _sp

        return float(_sp.cdf(x, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import logistic as _sp

        return float(_sp.ppf(q, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.loc)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float((np.pi**2 / 3.0) * self.scale * self.scale)

    def entropy(self) -> float:
        """Differential entropy log(scale) + 2."""
        import math

        return float(math.log(self.scale) + 2.0)

    def skewness(self) -> float:
        """Skewness (0)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis (6/5)."""
        return 1.2

    def mode(self) -> float:
        """Mode (= the location loc)."""
        return float(self.loc)

    def sampler(self, seed: int | None = None) -> "LogisticSampler":
        """Return a sampler for drawing observations from this distribution."""
        return LogisticSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LogisticEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return LogisticEstimator(name=self.name, keys=self.keys)
        return LogisticEstimator(
            pseudo_count=pseudo_count, suff_stat=(self.loc, self.scale), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "LogisticDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return LogisticDataEncoder()


class LogisticSampler(DistributionSampler):
    """Draw iid logistic observations."""

    def __init__(self, dist: LogisticDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.logistic(loc=self.dist.loc, scale=self.dist.scale, size=size)


class LogisticSuffStat(tuple):
    """A ``(sum, sum2, count)`` sufficient statistic that also carries a shift-anchored side payload.

    Behaves exactly like the plain 3-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one); ``anchored`` is the shift-anchored moment payload
    ``(anchor, sum_i w_i*(x_i - anchor), sum_i w_i*(x_i - anchor)^2)`` that :class:`LogisticAccumulator`
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
        return (_rebuild_logistic_suff_stat, (tuple(self), self.anchored))


def _rebuild_logistic_suff_stat(values: tuple, anchored: tuple | None) -> "LogisticSuffStat":
    """Unpickle helper for :class:`LogisticSuffStat` (module-level so pickle can import it)."""
    stat = LogisticSuffStat(values[0], values[1], values[2])
    stat.anchored = anchored
    return stat


# Conditioning threshold for the anchored-moment gate: the raw ``E[x^2]-E[x]^2`` variance the M-step
# needs loses about ``eps * (mean/sd)^2`` relative accuracy, so a (mean/sd)^2 up to 4e6 (ratio ~2000)
# keeps the raw form within ~1e-9 relative error -- the historical single-pass path is bit-preserved
# there. Beyond it the anchored track takes over. Chunks pooled from gate-passing content stay
# well-conditioned as a pool (Cauchy-Schwarz: n*mean_pool^2 <= sum_i n_i*mean_i^2), so a pool built
# only from gate-passing chunks never needs the anchor retroactively. Same constant, same rationale,
# as the Gaussian family's own gate (mixle.stats.univariate.continuous.gaussian) -- the logistic
# M-step needs only the same second moment.
_ANCHOR_CONDITION_RATIO = 4.0e6


def _needs_anchor(chunk_sum: float, chunk_sum2: float, w_sum: float) -> bool:
    """Whether a chunk's weighted moments are too ill-conditioned for the raw variance form.

    ``spread2`` computed here is itself the cancellation-prone estimate, but as a GATE it is
    reliable: when cancellation has corrupted it, the corruption is bounded by ``eps * m^2``, which
    still leaves ``m*m`` orders of magnitude above ``_ANCHOR_CONDITION_RATIO * spread2``.
    A non-positive computed spread activates the anchor outright (constant or near-constant data).
    """
    m = chunk_sum / w_sum
    spread2 = chunk_sum2 / w_sum - m * m
    return spread2 <= 0.0 or m * m > _ANCHOR_CONDITION_RATIO * spread2


class LogisticAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for logistic estimation.

    Alongside the raw ``(sum, sum2, count)`` the accumulator keeps a CONDITIONING-GATED
    shift-anchored moment track. The naive ``E[x^2]-E[x]^2`` form the M-step needs loses
    ~2*log2(|mean|/sd) bits to cancellation, so data with sd ~0.99 at offset 1.7e9 fits a scale
    12.6x too large with no warning. Anchoring at a data value keeps every term of the scatter
    O(count * spread^2), making the M-step variance shift-invariant.

    The track is CONDITIONING-GATED: a chunk whose ``|mean|/spread`` ratio the raw form handles to
    ~1e-9 relative error (see :func:`_needs_anchor`) accumulates exactly the historical single-pass
    way -- bit-identical statistics, no second pass -- and the anchor activates only when a chunk (or
    a scalar ``update``) would corrupt the variance. The raw moments remain the exchange format --
    ``(sum, sum2, count)`` is the declared ``StatisticSpec`` tuple consumed by engine kernels -- so
    the anchored track rides along as a payload on :class:`LogisticSuffStat`.
    """

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys
        self._anchor: float | None = None
        self._anchored_sum = 0.0
        self._anchored_sum2 = 0.0

    def _activate_anchor(self, anchor: float) -> None:
        """Start the shift-anchored moment track at ``anchor``.

        Any content already accumulated raw-only is converted about the new anchor. The conversion
        is the cancellation-prone form, but it is only ever applied to content that accumulated
        WITHOUT activating the gate -- i.e. content the gate certified as well-conditioned -- or to
        pre-existing raw statistics restored through ``from_value``/``combine``, where the
        conversion is no less accurate than the raw-only estimate those statistics supported before.
        """
        self._anchor = float(anchor)
        if self.sum != 0.0 or self.sum2 != 0.0 or self.count != 0.0:
            self._anchored_sum += self.sum - self._anchor * self.count
            self._anchored_sum2 += max(
                self.sum2 - 2.0 * self._anchor * self.sum + self._anchor * self._anchor * self.count, 0.0
            )

    def update(self, x: float, weight: float, estimate: LogisticDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        # Scalar updates carry no chunk to assess conditioning from, so the anchor activates on the
        # first observation (a zero-cost O(1) bookkeeping track on this path). Activation happens
        # BEFORE the raw fold so any pre-anchor content is converted from statistics the
        # conditioning gate has already vouched for.
        if self._anchor is None:
            self._activate_anchor(x)
        dx = x - self._anchor
        self._anchored_sum += dx * weight
        self._anchored_sum2 += dx * dx * weight
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: LogisticDistribution | None) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        chunk_sum = np.dot(x, weights)
        chunk_sum2 = np.dot(x * x, weights)
        w_sum = np.sum(weights, dtype=np.float64)
        # Conditioning gate: activate the anchored track only when this chunk's raw moments would
        # corrupt the variance (or the anchor is already live). BEFORE the raw fold, so activation
        # converts only pre-chunk content -- content the gate has already passed as well-conditioned.
        if len(x) > 0 and (self._anchor is not None or (w_sum > 0.0 and _needs_anchor(chunk_sum, chunk_sum2, w_sum))):
            if self._anchor is None:
                self._activate_anchor(float(x[0]))
            dx = x - self._anchor
            wdx = dx * weights
            self._anchored_sum += float(np.sum(wdx))
            self._anchored_sum2 += float(np.dot(wdx, dx))
        self.sum += chunk_sum
        self.sum2 += chunk_sum2
        self.count += w_sum

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "LogisticAccumulator":
        """Merge another logistic sufficient-statistic tuple."""
        anchored = getattr(suff_stat, "anchored", None)
        if anchored is not None:
            # Chan's parallel-merge: re-express the incoming anchored moments about this
            # accumulator's anchor. The anchor gap ``d`` is between two data values, so every term
            # stays O(count * spread^2) -- no large-offset cancellation is reintroduced. Activation
            # (when this side has no anchor yet) runs BEFORE the raw fold below so it converts only
            # this side's pre-existing content.
            b_anchor, b_asum, b_asum2 = anchored
            b_count = float(suff_stat[2])
            if self._anchor is None:
                self._activate_anchor(b_anchor)
            d = b_anchor - self._anchor
            self._anchored_sum += b_asum + b_count * d
            self._anchored_sum2 += b_asum2 + 2.0 * d * b_asum + b_count * d * d
        elif self._anchor is not None and (suff_stat[0] != 0.0 or suff_stat[1] != 0.0 or suff_stat[2] != 0.0):
            # Raw-only statistics (an engine kernel, a hand-built tuple, a gate-passing peer) joining
            # an anchored pool: convert about our anchor. See _activate_anchor for why the
            # cancellation-prone conversion is acceptable exactly here.
            a = self._anchor
            self._anchored_sum += suff_stat[0] - a * float(suff_stat[2])
            self._anchored_sum2 += max(suff_stat[1] - 2.0 * a * suff_stat[0] + a * a * float(suff_stat[2]), 0.0)

        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count.

        The returned object is a plain 3-tuple for every consumer that treats it as one; once the
        shift-anchored moment track is live it is a :class:`LogisticSuffStat` that additionally
        carries the anchored moments in its ``.anchored`` attribute, so :meth:`combine` can fold
        them in and :meth:`LogisticEstimator.estimate` can compute a shift-invariant variance.
        """
        if self._anchor is None:
            return self.sum, self.sum2, self.count
        stat = LogisticSuffStat(self.sum, self.sum2, self.count)
        stat.anchored = (self._anchor, self._anchored_sum, self._anchored_sum2)
        return stat

    def from_value(self, x: tuple[float, float, float]) -> "LogisticAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        anchored = getattr(x, "anchored", None)
        if anchored is not None:
            self._anchor, self._anchored_sum, self._anchored_sum2 = anchored
        else:
            # Raw-only statistics replace the state: the anchored track restarts unactivated, and a
            # later activation (first update / anchored merge) converts this content then.
            self._anchor = None
            self._anchored_sum = 0.0
            self._anchored_sum2 = 0.0
        return self

    def scale(self, c: float) -> "LogisticAccumulator":
        """Scale the accumulated statistics in place by ``c``, anchored track included.

        The structural default round-trips through ``value()`` and ``from_value()``, and
        ``scale_suff_stat`` rebuilds the payload as a PLAIN tuple -- which drops the ``anchored``
        attribute, so ``from_value`` sees raw-only statistics and restarts the track unactivated.
        Uniform weight scaling is exactly linear in both anchored moments and leaves the anchor
        (a data value, not a statistic) alone, so the track scales as the raw moments do.

        The moment arithmetic lives in ``scale_anchored_triple``; this method is pure attribute
        wiring, and its body is byte-identical to the Logistic/Gaussian sibling accumulator's --
        flagged by the duplicate-body scanner and accepted in the manifest deliberately, because
        de-duplicating the wiring itself would mean a mixin coupling both classes' private
        attribute names, a bigger structural change than the release stage warrants for zero
        remaining bug-risk (the shared math is what could drift; this cannot).
        """
        anchor, anchored_sum, anchored_sum2 = scale_anchored_triple(
            self._anchor, self._anchored_sum, self._anchored_sum2, c
        )
        super().scale(c)
        self._anchor = anchor
        self._anchored_sum = anchored_sum
        self._anchored_sum2 = anchored_sum2
        return self

    def acc_to_encoder(self) -> "LogisticDataEncoder":
        """Return the encoder used by this accumulator."""
        return LogisticDataEncoder()


class LogisticAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LogisticAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> LogisticAccumulator:
        """Create a fresh logistic accumulator."""
        return LogisticAccumulator(name=self.name, keys=self.keys)


def _spread_is_resolvable(variance: float, magnitude: float) -> bool:
    """Whether a spread of ``sqrt(variance)`` is representable at all at scale ``magnitude``.

    float64 values near ``magnitude`` lie on a grid of spacing ``u = ulp(magnitude)``, so deviations
    from the mean below about ``u/2`` cannot be carried by any sample there. Used only to decide what
    to DISCLOSE through ``numerical_repairs()``, never to clamp -- see the identical predicate on the
    Gaussian family (``mixle.stats.univariate.continuous.gaussian._spread_is_resolvable``) for the
    full rationale, which applies unchanged here.
    """
    if not np.isfinite(magnitude) or magnitude <= 0.0 or not np.isfinite(variance) or variance <= 0.0:
        return False
    half_ulp = 0.5 * float(np.spacing(magnitude))
    return variance > half_ulp * half_ulp


# Bound on how far the reported mean can sit from the exact sample mean the anchored track knows:
# ~4-8 grid steps of ``|mean|``. It bounds a rounding residue of the mean, not a spread, so a
# multiple of the ulp is the right shape here -- and it is deliberately the same constant the
# previous whole-scatter clamp used, so every degenerate payload that collapsed to exactly zero
# before still does. Same constant as the Gaussian family's own bound.
_MEAN_ROUNDING_BOUND = 8.8817841970012523e-16  # 4 * eps


def _anchored_pooled_variance(
    anchor: float,
    a_sum: float,
    a_sum2: float,
    count: float,
    mean: float,
    pseudo_count: float | None,
    prior_mean: float | None,
    prior_variance: float | None,
) -> tuple[float, tuple[str, ...]]:
    """Population variance of ``x`` computed from shift-anchored moments, plus any repairs to disclose.

    ``count``, ``mean`` are the RAW data count and the final (possibly prior-blended) location; the
    prior itself is folded in afterward through ``pseudo_count``/``prior_mean``/``prior_variance``,
    never through the anchored sums (which only ever describe the observed data).

    The scatter is SPLIT rather than accumulated in one sum, which is what lets the noise clamp stay
    off the data. Writing ``anchored_mean = a_sum/count`` for the sample's own mean in
    anchor-relative coordinates,

        scatter(mean) = [a_sum2 - a_sum * anchored_mean] + count * (mean - anchor - anchored_mean)**2

    and the two brackets have completely different error characters. The first (``core``) is the
    scatter about the sample's OWN mean: both terms are O(count * spread^2), computed entirely at
    small magnitude, and it carries all the data. The second is the displacement of the mean
    actually reported (``mean``) from that sample mean -- genuine when a pseudo-count prior pulls
    the location, but on the plain maximum-likelihood path pure rounding of ``sum_x / count`` at
    data magnitude, and the ONLY place the large magnitude enters. Clamping the rounding term alone
    leaves the data untouched; a single combined sum could only clamp the total, so its ulp-scale
    threshold would have to be crossed by the spread as well, and any spread below
    ~``4 eps |mean|`` per observation would read as constant -- exactly the defect this split fixes
    (see mixle.stats.univariate.continuous.gaussian._anchored_pooled_variance for the reference
    repair this mirrors).
    """
    if count <= 0.0:
        observed_scatter = 0.0
        repairs: tuple[str, ...] = ()
    else:
        anchored_mean = a_sum / count
        # Scatter about the sample's own mean -- the whole of the data, computed at spread scale.
        core = a_sum2 - a_sum * anchored_mean
        # Mathematically >= 0; only last-ulp rounding of the two O(count * spread^2) terms can
        # undershoot -- or overshoot, and a degenerate component's scatter must come out EXACTLY
        # zero on every algebraically equivalent path, or the scale-relative variance floor reads
        # the +O(eps) residue as a genuine spread and two equivalent fits disagree.
        noise_scale = max(abs(a_sum2), abs(a_sum * anchored_mean), 1.0e-300)
        repairs = ()
        if core < 1.0e-12 * noise_scale:
            # Reporting zero for something whose apparent scatter was positive is a repair, not a
            # measurement -- but only worth saying when the spread it stood for was one this
            # magnitude could have represented.
            if _spread_is_resolvable(core / count, max(abs(mean), abs(anchor))):
                repairs = ("spread-below-noise(%.3g of %.3g)" % (core / count, noise_scale / count),)
            core = 0.0
        core = max(core, 0.0)
        # Displacement of the reported mean from the sample mean. Below the mean's own rounding
        # granularity it is not a displacement at all, just which order the large-magnitude sum was
        # accumulated in, and squaring it would turn that into variance.
        shift = (mean - anchor) - anchored_mean
        if abs(shift) <= _MEAN_ROUNDING_BOUND * max(abs(mean), abs(anchor)):
            shift = 0.0
        observed_scatter = core + count * shift * shift
    if pseudo_count not in (None, 0.0) and prior_variance is not None:
        offset = 0.0 if prior_mean is None else (prior_mean - mean) ** 2
        prior_scatter = pseudo_count * (prior_variance + offset)
        return (observed_scatter + prior_scatter) / (count + pseudo_count), repairs
    if count == 0.0:
        return 0.0, repairs
    return observed_scatter / count, repairs


class LogisticEstimator(ParameterEstimator):
    """Moment estimator for logistic location and scale.

    The likelihood MLE has no closed-form M-step. The EM estimator uses the
    identities mean=loc and var=pi^2 scale^2 / 3; torch gradient MLE can refine
    both parameters when exact likelihood optimization is desired.

    The variance moment is formed from shift-anchored statistics whenever the accumulated
    sufficient statistics carry that payload (see :class:`LogisticAccumulator`), which makes the
    fit shift-equivariant: estimating on ``x + c`` returns ``loc + c`` with ``scale`` unchanged.
    With a plain raw tuple -- statistics the conditioning gate never needed to anchor, or ones
    restored from an older serialization -- the historical raw path is used, bit-identically.
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

    def accumulator_factory(self) -> LogisticAccumulatorFactory:
        """Return an accumulator factory for logistic moment statistics."""
        return LogisticAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> LogisticDistribution:
        """Estimate location and scale from weighted moments."""
        sum_x, sum_x2, count = float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2])
        # The anchored payload (when present) describes the RAW data only -- captured before any
        # prior blend below -- so it is read off the untouched (sum_x, count) pair.
        anchored = consistent_anchored_triple(suff_stat, sum_x, count)

        prior_mean: float | None = None
        prior_variance: float | None = None
        pc = self.pseudo_count
        if pc is not None and self.suff_stat is not None:
            prior_mean, scale0 = self.suff_stat
            prior_variance = (math.pi * math.pi / 3.0) * scale0 * scale0

        # Blended raw sums, kept byte-for-byte identical to the historical pre-anchor computation
        # for the fallback (non-anchored) path below -- ordinary well-scaled data must not move.
        blended_sum_x, blended_sum_x2, blended_count = sum_x, sum_x2, count
        if prior_mean is not None:
            blended_sum_x = sum_x + pc * prior_mean
            blended_sum_x2 = sum_x2 + pc * (prior_variance + prior_mean * prior_mean)
            blended_count = count + pc

        if blended_count <= 0.0:
            return LogisticDistribution(name=self.name, keys=self.keys)

        loc = blended_sum_x / blended_count

        notes: tuple[str, ...] = ()
        if anchored is not None:
            # Only the variance loses to cancellation at large magnitude; the mean is a plain
            # same-sign sum and is computed from the raw (possibly prior-blended) moments above,
            # unchanged. The prior is folded into the variance separately, never through the
            # anchored sums (which only ever describe the observed data) -- see
            # _anchored_pooled_variance.
            var, notes = _anchored_pooled_variance(
                anchored[0], anchored[1], anchored[2], count, loc, pc, prior_mean, prior_variance
            )
        else:
            var = max(blended_sum_x2 / blended_count - loc * loc, 0.0)

        scale = math.sqrt(max(3.0 * var / (math.pi * math.pi), self.min_scale * self.min_scale))
        dist = LogisticDistribution(loc=loc, scale=scale, name=self.name, keys=self.keys)
        if notes:
            dist._numerical_repairs = notes
        return dist


class LogisticDataEncoder(DataSequenceEncoder):
    """Encode logistic observations as a float array."""

    def __str__(self) -> str:
        return "LogisticDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogisticDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError("LogisticDistribution requires finite or infinite real-valued observations.")
        return rv
