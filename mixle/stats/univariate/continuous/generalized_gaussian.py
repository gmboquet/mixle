"""Generalized Gaussian (exponential-power) distribution.

A symmetric location-scale family with a tunable tail/peakedness shape ``beta`` that interpolates the
Laplace (``beta = 1``), Gaussian (``beta = 2``), and uniform (``beta -> inf``) laws. With location
``mu``, scale ``alpha > 0`` and shape ``beta > 0``,

    ``f(x; mu, alpha, beta) = beta / (2 alpha Gamma(1/beta)) * exp(-(abs(x - mu) / alpha)^beta)``.

The normalizer is closed form (a Gamma function), so density/CDF/quantile/moments/entropy are all exact;
it samples exactly via a Gamma draw with a random sign. Parameters are fit by the method of moments:
``mu`` is the mean, the excess kurtosis ``Gamma(5/beta)Gamma(1/beta)/Gamma(3/beta)^2 - 3`` pins ``beta``
(monotone, solved by a bracketed root find), and ``alpha`` follows from the variance
``alpha^2 Gamma(3/beta)/Gamma(1/beta)``.

The M-step is *shift-equivariant*: fitting ``x + c`` returns ``mu + c`` with an unchanged ``alpha``
and ``beta``. That does not come for free from raw power sums -- the central fourth moment differenced
out of ``E[x^4]`` loses about ``4*log2(abs(mean)/sd)`` bits -- so the accumulator carries a
conditioning-gated shift-anchored moment track alongside the raw sums (see
:class:`GeneralizedGaussianAccumulator`).

References:
  - Nadarajah, "A generalized normal distribution", *J. Applied Statistics* 32 (2005).
  - Subbotin (1923), the original exponential-power family.
"""

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gamma, gammainc, gammaincinv, gammaln

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


def _excess_kurtosis(beta: float) -> float:
    """Excess kurtosis of the exponential-power law as a function of the shape ``beta``."""
    return float(gamma(5.0 / beta) * gamma(1.0 / beta) / gamma(3.0 / beta) ** 2 - 3.0)


# Conditioning threshold for the anchored-moment gate. The M-step's highest-order reduced moment is
# the fourth, whose raw form ``E[x^4] - 4 m E[x^3] + 6 m^2 E[x^2] - 3 m^4`` loses about
# ``eps * (|mean|/sd)^4`` relative accuracy; a (mean/sd)^2 up to 2e3 (ratio ~45) keeps that within
# ~1e-9, so the historical single-pass path is bit-preserved there and the anchored track takes over
# beyond it. This is a much tighter gate than the Gaussian's 4e6, and deliberately so: the Gaussian
# only has to protect a second moment. Chunks pooled from gate-passing content stay well-conditioned
# as a pool (Cauchy-Schwarz), so a pool built only from gate-passing chunks never needs the anchor
# retroactively.
_ANCHOR_CONDITION_RATIO = 2.0e3


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


def _shift_moments(
    m0: float, m1: float, m2: float, m3: float, m4: float, d: float
) -> tuple[float, float, float, float]:
    """Re-express weighted power sums accumulated about a point sitting ``d`` above the new one.

    Given ``m_k = sum_i w_i y_i^k`` (with ``m0 = sum_i w_i``), return the same sums for
    ``y_i + d``. Used both to convert raw sums (``d = -anchor``) onto an anchor and to fold a
    differently anchored partner in :meth:`GeneralizedGaussianAccumulator.combine`.
    """
    d2 = d * d
    return (
        m1 + d * m0,
        m2 + 2.0 * d * m1 + d2 * m0,
        m3 + 3.0 * d * m2 + 3.0 * d2 * m1 + d2 * d * m0,
        m4 + 4.0 * d * m3 + 6.0 * d2 * m2 + 4.0 * d2 * d * m1 + d2 * d2 * m0,
    )


class GeneralizedGaussianSuffStat(tuple):
    """A ``(count, s1, s2, s3, s4)`` sufficient statistic that also carries a side payload.

    Behaves exactly like the plain 5-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one); ``anchored`` is the shift-anchored payload
    ``(anchor, sum_i w_i (x_i - anchor)^k for k = 1..4)`` the accumulator maintains alongside the raw
    power sums so the M-step survives large-offset data. Code that doesn't know about the payload
    (generic ``scale_suff_stat``, serializers, ...) sees an ordinary tuple and the estimate falls
    back to the historical raw path.
    """

    def __new__(
        cls,
        count: float,
        s1: float,
        s2: float,
        s3: float,
        s4: float,
        anchored: tuple[float, float, float, float, float] | None = None,
    ) -> "GeneralizedGaussianSuffStat":
        obj = super().__new__(cls, (count, s1, s2, s3, s4))
        obj.anchored = anchored
        return obj

    def __reduce__(self):
        # A tuple subclass with a payload-bearing __new__ does not pickle by default, and the
        # Spark/multiprocessing reducers round-trip accumulator values through pickle.
        return (_rebuild_generalized_gaussian_suff_stat, (tuple(self), self.anchored))


def _rebuild_generalized_gaussian_suff_stat(values: tuple, anchored: tuple | None) -> GeneralizedGaussianSuffStat:
    """Unpickle helper for :class:`GeneralizedGaussianSuffStat` (module level so pickle can import it)."""
    return GeneralizedGaussianSuffStat(*values, anchored=anchored)


def _consistent_anchored_moments(suff_stat: Any, sum_x: float, count: float) -> tuple[float, ...] | None:
    """Return the anchored payload of ``suff_stat`` when it is usable, else ``None``.

    ``None`` falls back to the raw reduced-moment M-step, so a payload is only trusted when it is
    finite, has the right arity, carries non-negative even orders, and agrees with the raw first
    moment it claims to describe -- a hand-built :class:`GeneralizedGaussianSuffStat` whose payload
    contradicts its tuple must not silently change the estimate the tuple alone would have produced.
    """
    anchored = getattr(suff_stat, "anchored", None)
    if anchored is None or count <= 0.0:
        return None
    if len(anchored) != 5 or not all(np.isfinite(v) for v in anchored):
        return None
    anchor, a1, a2, _a3, a4 = (float(v) for v in anchored)
    if a2 < 0.0 or a4 < 0.0:
        return None
    implied_sum = a1 + count * anchor
    tolerance = 1.0e-6 * max(abs(sum_x), abs(count * anchor), 1.0)
    if abs(implied_sum - sum_x) > tolerance:
        return None
    return tuple(float(v) for v in anchored)


def _prior_is_ill_conditioned(raw_prior: Any) -> bool:
    """Whether raw prior moments ``(E[X], E[X^2], ...)`` have lost their own spread to cancellation.

    A diagnostic, not a gate: it decides whether to *warn*, never whether to reject. The prior is a
    one-unit pseudo-sample, so the same conditioning test the accumulator applies to a chunk applies
    here with ``w_sum = 1``.
    """
    try:
        e1, e2 = float(raw_prior[0]), float(raw_prior[1])
    except (TypeError, ValueError, IndexError):
        return False
    if not (np.isfinite(e1) and np.isfinite(e2)):
        return False
    return _needs_anchor(e1, e2, 1.0)


class GeneralizedGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Gaussian (exponential power) with location ``mu``, scale ``alpha`` and shape ``beta``."""

    def __init__(self, mu: float, alpha: float, beta: float, name: str | None = None, keys: str | None = None) -> None:
        if not np.isfinite(mu):
            raise ValueError("GeneralizedGaussianDistribution requires finite mu.")
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError("GeneralizedGaussianDistribution requires finite alpha > 0.")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("GeneralizedGaussianDistribution requires finite beta > 0.")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name
        self.keys = keys
        self._log_norm = math.log(self.beta) - math.log(2.0 * self.alpha) - gammaln(1.0 / self.beta)

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep ``_log_norm`` tied to the parameter(s) ``alpha``/``beta`` they derive from.

        Computed once in ``__init__`` and read by ``log_density``, so a later assignment used to
        leave them stale and the scorer kept reporting the *previous* parameters' density with no
        error at all (MXR-080-1192).

        Recompute rather than validate: callers legitimately install out-of-domain or non-finite
        parameters -- deserialized legacy states and NaN-propagation checks both do -- so a value
        outside the domain yields a NaN constant that propagates honestly instead of rejecting a
        state the library is expected to be able to hold.
        """
        object.__setattr__(self, name, value)
        if name not in ("alpha", "beta"):
            return
        try:
            object.__setattr__(
                self, "_log_norm", float(math.log(self.beta) - math.log(2.0 * self.alpha) - gammaln(1.0 / self.beta))
            )
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, AttributeError, FloatingPointError):
            # AttributeError covers __init__, where the first parameter is assigned before the rest.
            object.__setattr__(self, "_log_norm", float("nan"))

    def __str__(self) -> str:
        return "GeneralizedGaussianDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.alpha),
            repr(self.beta),
            repr(self.name),
            repr(self.keys),
        )

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for generalized Gaussian distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        # declaring the engine-neutral density lets the symbolic->numba lowering compile a scalar kernel
        # for this non-exponential-family leaf (parity with Laplace/Logistic/Weibull/...).
        return DistributionDeclaration(
            name="generalized_gaussian",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu"),
                ParameterSpec("alpha", constraint="positive"),
                ParameterSpec("beta", constraint="positive"),
            ),
            statistics=(
                StatisticSpec("values", kind="raw_observations", scales=False),
                StatisticSpec("weights", kind="weights"),
            ),
            support="real",
        )

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, alpha: Any, beta: Any, engine: Any) -> Any:
        """Engine-neutral generalized-Gaussian log-density: ``log_norm - (abs(x-mu)/alpha)**beta``."""
        log_norm = (
            engine.log(beta) - engine.log(engine.asarray(2.0) * alpha) - engine.gammaln(engine.asarray(1.0) / beta)
        )
        return log_norm - (engine.abs(x - mu) / alpha) ** beta

    def density(self, x: float) -> float:
        """Return the probability density at ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at ``x``."""
        xx = scored_observation(x, label="generalized-Gaussian observations")
        return self._log_norm - (abs(xx - self.mu) / self.alpha) ** self.beta

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        z = np.abs(np.asarray(x, dtype=np.float64) - self.mu) / self.alpha
        return self._log_norm - z**self.beta

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x)."""
        xv = float(x) - self.mu
        z = (abs(xv) / self.alpha) ** self.beta
        return float(0.5 + math.copysign(0.5 * gammainc(1.0 / self.beta, z), xv))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        qv = float(q) - 0.5
        z = gammaincinv(1.0 / self.beta, 2.0 * abs(qv))
        return float(self.mu + math.copysign(self.alpha * z ** (1.0 / self.beta), qv))

    def mean(self) -> float:
        """Mean (the location ``mu``)."""
        return self.mu

    def variance(self) -> float:
        """Variance alpha^2 Gamma(3/beta) / Gamma(1/beta)."""
        return float(self.alpha * self.alpha * gamma(3.0 / self.beta) / gamma(1.0 / self.beta))

    def skewness(self) -> float:
        """Skewness (0 -- the law is symmetric)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis Gamma(5/beta)Gamma(1/beta)/Gamma(3/beta)^2 - 3."""
        return _excess_kurtosis(self.beta)

    def entropy(self) -> float:
        """Differential entropy 1/beta - log(beta / (2 alpha Gamma(1/beta)))."""
        return float(1.0 / self.beta - self._log_norm)

    def sampler(self, seed: int | None = None) -> "GeneralizedGaussianSampler":
        """Return a sampler (Gamma magnitude with a random sign)."""
        return GeneralizedGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeneralizedGaussianEstimator":
        """Return a method-of-moments estimator for ``mu``, ``alpha``, ``beta``."""
        if pseudo_count is None:
            return GeneralizedGaussianEstimator(name=self.name, keys=self.keys)
        # Convert this distribution's own (mu, alpha, beta) into the raw first four moments -- the
        # space estimate() accumulates in (s1..s4) -- so pseudo_count can blend a prior pseudo-sample
        # toward them (mirrors GumbelEstimator / WeibullEstimator's suff_stat pattern). The law is
        # symmetric about mu, so its odd central moments vanish, giving closed-form raw moments from
        # mean()/variance()/kurtosis() alone:
        #   E[X] = mu, E[X^2] = var + mu^2, E[X^3] = 3*mu*var + mu^3,
        #   E[X^4] = (kurt+3)*var^2 + 6*mu^2*var + mu^4.
        mu0 = self.mean()
        var0 = self.variance()
        kurt0 = self.kurtosis()
        e1 = mu0
        e2 = var0 + mu0 * mu0
        e3 = 3.0 * mu0 * var0 + mu0**3
        e4 = (kurt0 + 3.0) * var0 * var0 + 6.0 * mu0 * mu0 * var0 + mu0**4
        return GeneralizedGaussianEstimator(
            pseudo_count=pseudo_count,
            suff_stat=(e1, e2, e3, e4),
            name=self.name,
            keys=self.keys,
            # The raw moments above are the release-pinned exchange form, but at a large |mu| they no
            # longer contain this distribution's spread at all (``var0 + mu0**2`` rounds ``var0``
            # away once ``mu0**2`` exceeds ~1e16 times it). Carry the central restatement alongside
            # them so estimate() can place the prior on the data anchor exactly; the law is symmetric
            # about mu, so its odd central moments vanish.
            prior_central=(mu0, var0, 0.0, (kurt0 + 3.0) * var0 * var0),
        )

    def dist_to_encoder(self) -> "GeneralizedGaussianDataEncoder":
        """Return the data encoder used by this distribution (the raw value)."""
        return GeneralizedGaussianDataEncoder()


class GeneralizedGaussianSampler(DistributionSampler):
    """Draw ``x = mu + sign * alpha * Gamma(1/beta)**(1/beta)``."""

    def __init__(self, dist: GeneralizedGaussianDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        d = self.dist
        n = 1 if size is None else int(size)
        g = self.rng.gamma(1.0 / d.beta, 1.0, size=n)
        sign = self.rng.randint(0, 2, size=n) * 2 - 1
        x = d.mu + sign * d.alpha * g ** (1.0 / d.beta)
        return float(x[0]) if size is None else x


class GeneralizedGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x, sum x^2, sum x^3, sum x^4)`` for the MoM.

    Alongside the raw sums the accumulator keeps a CONDITIONING-GATED shift-anchored track,
    ``sum_i w_i (x_i - anchor)^k`` for ``k = 1..4`` about a data anchor. The method of moments needs
    central moments, and differencing them out of raw power sums is the classic cancellation-prone
    form: the fourth reduced moment loses roughly ``4*log2(abs(mean)/sd)`` bits, so data with sd ~0.7 at
    offset 1.7e9 has *no* correct digits left in ``m4`` and the fit collapses onto the shape bound
    with a scale two orders of magnitude too small -- silently. Anchoring keeps every term of the
    scatter ``O(count * spread^4)``, making the M-step shift-equivariant.

    The gate keeps the historical path bit-identical for well-conditioned data: a chunk whose
    ``abs(mean)/spread`` ratio the raw form handles to ~1e-9 relative (see :func:`_needs_anchor`)
    accumulates exactly the way it always did, with no anchor and no second pass. The raw sums remain
    the exchange format, so the anchored track rides along as a payload on
    :class:`GeneralizedGaussianSuffStat`; a consumer that drops the payload simply gets the
    historical raw estimate back.
    """

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.name = name
        self.keys = keys
        self._anchor: float | None = None
        self._a1 = 0.0
        self._a2 = 0.0
        self._a3 = 0.0
        self._a4 = 0.0
        self._anchor_unrecoverable = False

    def _absorb_raw(self, count: float, s1: float, s2: float, s3: float, s4: float) -> None:
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
        if count == 0.0 and s1 == 0.0 and s2 == 0.0 and s3 == 0.0 and s4 == 0.0:
            return
        if count > 0.0 and _needs_anchor(s1, s2, count):
            self._anchor_unrecoverable = True
            warnings.warn(
                "GeneralizedGaussianAccumulator merged raw power sums whose location dominates their "
                "spread into a shift-anchored pool. Raw sums at that conditioning no longer contain "
                "the central moments, and no change of reference point can restore them, so this "
                "pool falls back to the historical raw M-step and its alpha/beta are unreliable at "
                "this offset. Accumulate through update()/seq_update(), or combine statistics that "
                "still carry their anchored payload, instead of restoring plain power sums.",
                RuntimeWarning,
                stacklevel=3,
            )
        a1, a2, a3, a4 = _shift_moments(count, s1, s2, s3, s4, -self._anchor)
        self._a1 += a1
        self._a2 += max(a2, 0.0)
        self._a3 += a3
        self._a4 += max(a4, 0.0)

    def _activate_anchor(self, anchor: float) -> None:
        """Start the shift-anchored moment track at ``anchor``, converting any raw content onto it."""
        self._anchor = float(anchor)
        self._absorb_raw(self.count, self.s1, self.s2, self.s3, self.s4)

    def update(self, x: float, weight: float, estimate: GeneralizedGaussianDistribution | None) -> None:
        """Accumulate weighted raw moments up to order four for one observation."""
        xv = float(x)
        # Scalar updates carry no chunk to assess conditioning from, so the anchor activates on the
        # first observation THAT CARRIES POSITIVE WEIGHT (O(1) bookkeeping on this path). A weight of
        # exactly 0.0 -- an EM component's responsibility for a point it does not own, ordinary usage
        # of this calling convention -- contributes nothing to any of the four anchored moments
        # regardless of the anchor, so it must never be allowed to SET the anchor: an
        # extreme-magnitude zero-weight observation would otherwise become the permanent reference
        # point every later, fully-weighted observation is differenced against, reintroducing exactly
        # the cancellation this track exists to avoid. Activation happens BEFORE the raw fold so any
        # pre-anchor content is converted from statistics the gate has already vouched for.
        if self._anchor is None and weight > 0.0:
            self._activate_anchor(xv)
        if self._anchor is not None:
            dx = xv - self._anchor
            # A weight of exactly 0.0 must contribute exactly zero to any of the four anchored
            # moments regardless of dx's magnitude, but squaring dx BEFORE weighting can overflow
            # for an ordinary finite dx, and inf * 0.0 is nan. Masking dx to 0.0 here keeps a
            # positively-weighted call bit-identical while making a zero-weight call's contribution
            # exactly zero at any magnitude.
            safe_dx = dx if weight != 0.0 else 0.0
            safe_dx2 = safe_dx * safe_dx
            self._a1 += weight * dx
            self._a2 += weight * safe_dx2
            self._a3 += weight * safe_dx2 * safe_dx
            self._a4 += weight * safe_dx2 * safe_dx2
        self.count += weight
        # Same hazard as above for the raw moments: xv**2/3/4 can overflow before weight is ever
        # applied, poisoning s2/s3/s4 for good once weight is exactly 0.0.
        safe_xv = xv if weight != 0.0 else 0.0
        self.s1 += weight * xv
        self.s2 += weight * safe_xv**2
        self.s3 += weight * safe_xv**3
        self.s4 += weight * safe_xv**4

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted raw moments up to order four from encoded data."""
        xv = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w_sum = float(w.sum())
        chunk_sum = float(np.dot(w, xv))
        xv2 = xv * xv
        chunk_sum2 = float(np.dot(w, xv2))
        chunk_sum3 = float(np.dot(w, xv2 * xv))
        chunk_sum4 = float(np.dot(w, xv2 * xv2))
        if not (np.isfinite(chunk_sum2) and np.isfinite(chunk_sum3) and np.isfinite(chunk_sum4)):
            # A weight of exactly 0.0 must contribute exactly zero to the raw moments regardless of
            # xv's magnitude, but squaring/cubing/etc. xv BEFORE weighting can overflow for an
            # ordinary finite xv, and inf * 0.0 is nan -- silently poisoning the raw moments for
            # every other, fully-weighted observation folded in the same chunk. Recomputed with xv
            # masked to 0.0 wherever its own weight is exactly zero, only on this rare,
            # already-broken path (checked via isfinite rather than masked unconditionally, to keep
            # the ordinary path at its historical cost).
            safe_xv = np.where(w != 0.0, xv, 0.0)
            xv2 = safe_xv * safe_xv
            chunk_sum2 = float(np.dot(w, xv2))
            chunk_sum3 = float(np.dot(w, xv2 * safe_xv))
            chunk_sum4 = float(np.dot(w, xv2 * xv2))
        # Conditioning gate: activate the anchored track only when this chunk's raw moments would
        # corrupt the reduced moments (or the anchor is already live). BEFORE the raw fold, so
        # activation converts only the content that preceded this chunk.
        if len(xv) > 0 and (self._anchor is not None or (w_sum > 0.0 and _needs_anchor(chunk_sum, chunk_sum2, w_sum))):
            if self._anchor is None:
                # w_sum > 0.0 is guaranteed by the branch above, so this chunk carries at least one
                # positively-weighted element; anchor at the FIRST one of those rather than at xv[0]
                # positionally, so a zero-weight (or negative-weight) observation at any magnitude --
                # an EM responsibility of exactly 0.0 for the first point in a batch is ordinary
                # usage, not misuse -- can never seed the anchor. See the identical gate in
                # AnchoredMomentTrack._anchor_chunk.
                self._activate_anchor(float(xv[np.argmax(w > 0.0)]))
            dx = xv - self._anchor
            dx2 = dx * dx
            a1 = float(np.dot(w, dx))
            a2 = float(np.dot(w, dx2))
            a3 = float(np.dot(w, dx2 * dx))
            a4 = float(np.dot(w, dx2 * dx2))
            if not (np.isfinite(a2) and np.isfinite(a3) and np.isfinite(a4)):
                # Same hazard as chunk_sum2/3/4 above, applied to the anchor-relative deltas.
                safe_dx = np.where(w != 0.0, dx, 0.0)
                dx2 = safe_dx * safe_dx
                a2 = float(np.dot(w, dx2))
                a3 = float(np.dot(w, dx2 * safe_dx))
                a4 = float(np.dot(w, dx2 * dx2))
            self._a1 += a1
            self._a2 += a2
            self._a3 += a3
            self._a4 += a4
        self.count += w_sum
        self.s1 += chunk_sum
        self.s2 += chunk_sum2
        self.s3 += chunk_sum3
        self.s4 += chunk_sum4

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float, float, float]) -> "GeneralizedGaussianAccumulator":
        """Merge another generalized-Gaussian sufficient-statistic tuple."""
        anchored = getattr(suff_stat, "anchored", None)
        if anchored is not None and len(anchored) == 5:
            # Re-express the incoming anchored moments about this accumulator's anchor. The anchor
            # gap ``d`` is between two data values, so every term stays O(count * spread^4).
            # Activation (when this side has no anchor yet) runs BEFORE the raw fold below so the
            # pre-existing raw content is converted exactly once.
            b_anchor, b1, b2, b3, b4 = (float(v) for v in anchored)
            b_count = float(suff_stat[0])
            if self._anchor is None:
                self._activate_anchor(b_anchor)
            f1, f2, f3, f4 = _shift_moments(b_count, b1, b2, b3, b4, b_anchor - self._anchor)
            self._a1 += f1
            self._a2 += max(f2, 0.0)
            self._a3 += f3
            self._a4 += max(f4, 0.0)
        elif self._anchor is not None:
            # Raw-only statistics joining an anchored pool -- the mirror of the case above. Same
            # conversion, same recoverability check: see :meth:`_absorb_raw`.
            self._absorb_raw(*(float(v) for v in suff_stat[:5]))
        self.count += suff_stat[0]
        self.s1 += suff_stat[1]
        self.s2 += suff_stat[2]
        self.s3 += suff_stat[3]
        self.s4 += suff_stat[4]
        return self

    def value(self) -> tuple[float, float, float, float, float]:
        """Return count and raw moment sums through order four.

        The returned object is a plain 5-tuple for every consumer that treats it as one; once the
        shift-anchored track is live it additionally carries the anchored moments in its
        ``.anchored`` attribute, so :meth:`combine` can fold them in and
        :meth:`GeneralizedGaussianEstimator.estimate` can use them for the reduced moments. The
        payload is withheld when the track was seeded from raw statistics that had already lost their
        central moments (see :meth:`_activate_anchor`), so a pool that cannot honestly claim
        shift-equivariance reports the historical raw statistics instead of a worse anchored guess.
        """
        if self._anchor is None or self._anchor_unrecoverable:
            return self.count, self.s1, self.s2, self.s3, self.s4
        return GeneralizedGaussianSuffStat(
            self.count,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
            anchored=(self._anchor, self._a1, self._a2, self._a3, self._a4),
        )

    def from_value(self, x: tuple[float, float, float, float, float]) -> "GeneralizedGaussianAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.s1, self.s2, self.s3, self.s4 = (float(v) for v in x[:5])
        anchored = getattr(x, "anchored", None)
        self._anchor_unrecoverable = False
        if anchored is not None and len(anchored) == 5:
            self._anchor, self._a1, self._a2, self._a3, self._a4 = (float(v) for v in anchored)
        else:
            # Raw-only statistics replace the state: the anchored track restarts unactivated, and a
            # later activation (first update / anchored merge) converts this content then.
            self._anchor = None
            self._a1 = self._a2 = self._a3 = self._a4 = 0.0
        return self

    def scale(self, c: float) -> "GeneralizedGaussianAccumulator":
        """Scale accumulated evidence, keeping the shift-anchored track in step with the raw sums.

        The generic base implementation round-trips through ``scale_suff_stat``/``from_value``, which
        sees only a plain tuple and would silently drop the anchored payload -- turning a scaled
        accumulator back into the ill-conditioned raw path.
        """
        self.count *= c
        self.s1 *= c
        self.s2 *= c
        self.s3 *= c
        self.s4 *= c
        self._a1 *= c
        self._a2 *= c
        self._a3 *= c
        self._a4 *= c
        return self

    def acc_to_encoder(self) -> "GeneralizedGaussianDataEncoder":
        """Return the encoder used by this accumulator."""
        return GeneralizedGaussianDataEncoder()


class GeneralizedGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GeneralizedGaussianAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> GeneralizedGaussianAccumulator:
        """Create a fresh generalized-Gaussian accumulator."""
        return GeneralizedGaussianAccumulator(name=self.name, keys=self.keys)


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
    a4: float,
    delta: float,
    pc: float,
    prior: tuple[float, float, float, float] | None,
    prior_central: tuple[float, float, float, float] | None,
) -> tuple[float, float, tuple[str, ...]]:
    """Second and fourth central moments about ``ref + delta``, from shift-anchored data moments.

    ``ref``/``n``/``a1..a4`` describe the DATA alone (the accumulator's anchored track, never
    touched by any prior blend); ``delta`` is the SMALL offset from ``ref`` to the location the
    estimator actually reports (``mu = ref + delta``), which can differ from the data's own mean
    when a pseudo-count prior pulls it -- ``delta`` is passed rather than ``mu`` itself because every
    displacement below is computed in this small, already-offset coordinate, never by differencing
    two separately-materialized ``O(magnitude)`` floats (``mu_data - mu``): that subtraction would
    reintroduce, in the DISPLACEMENT, exactly the ``ulp(magnitude)``-scale rounding the anchor track
    exists to keep out of the SPREAD, and it is invisible at the loose tolerances the clamp itself
    needs (an absolute ulp of ~1e-7 buried in a threshold check of ~1e-6) but not at the tight ones a
    prior blend must hit (see ``campaign3b_families_test.py``'s
    ``test_pseudo_count_prior_at_a_large_location_is_blended_exactly``, which pinned this down to
    7 decimal places). Order four needs more than the Gaussian's two brackets, but the same reasoning
    extends cleanly once it is phrased as "recenter each group's own central moments onto ``mu``,
    then pool":

    1. ``core2/core3/core4`` are the DATA's own central moments about its own mean -- every term is
       ``O(spread^k)``, computed entirely at small magnitude, and this is where all the data's real
       spread lives. Only ``core2`` is gated, by the same RELATIVE 1e-12 cancellation clamp the
       Gaussian family uses (``core2`` is a difference of two ``O(spread^2)`` quantities); when it
       clamps to zero, ``core3``/``core4`` clamp with it, since a truly-degenerate sample has every
       central moment exactly zero, not just its second.
    2. ``shift_data = delta_data - delta`` is the displacement of the reported location from the
       data's own mean, computed ENTIRELY in small-offset coordinates (both terms are ``O(spread)``
       unless a prior has pulled ``delta`` far from the data, in which case ``shift_data`` correctly
       comes out ``O(delta)`` with no cancellation either way). Below the mean's own rounding
       granularity it is pure rounding on the plain ML path (where ``delta`` is computed from the
       SAME anchored sums as ``delta_data`` and the two agree exactly unless a prior is blended in),
       so it alone gets the absolute ulp-scale clamp. Recentering the data's own central moments onto
       ``mu`` from its own mean is the single-group parallel-axis expansion,
       ``E[(Y+d)^4] = m4 + 4 d m3 + 6 d^2 m2 + d^4`` (``E[Y]=0``, ``d=shift_data``): a polynomial
       evaluation, never a cancellation, so ``shift_data`` may legitimately be large (a real prior
       pull) without needing any further clamp once it has cleared the rounding check.
    3. When a pseudo-count prior is blended in, it contributes as a second "group" of weight ``pc``
       with its own central moments (from ``prior_central`` when available -- exact at any
       magnitude -- else recovered from the raw power-sum payload ``prior`` the same, already-warned,
       degraded way the raw M-step blend has always used) recentered onto ``mu`` the identical way
       (``shift_prior = prior[0] - delta``, ``prior[0]`` already being the prior's own small offset
       from ``ref`` that :meth:`GeneralizedGaussianEstimator._prior_about` computed), and the two
       groups are pooled by weight. No clamp applies to the prior's own recentering displacement:
       like the Gaussian family's ``prior_scatter`` term, it is an explicit additive contribution,
       not a cancellation, and reporting it in full (even when large) is correct -- mixing in a prior
       whose mean sits far from the data legitimately inflates the pooled spread.
    """
    delta_data = a1 / n
    r2d, r3d, r4d = a2 / n, a3 / n, a4 / n
    core2 = r2d - delta_data * delta_data
    core3 = r3d - 3.0 * delta_data * r2d + 2.0 * delta_data**3
    core4 = r4d - 4.0 * delta_data * r3d + 6.0 * delta_data * delta_data * r2d - 3.0 * delta_data**4
    noise = 1.0e-12 * max(abs(r2d), delta_data * delta_data, 1.0e-300)
    repairs: tuple[str, ...] = ()
    if core2 < noise:
        if _spread_is_resolvable(core2, abs(ref)):
            repairs = ("spread-below-noise(%.3g of %.3g)" % (core2, noise),)
        core2 = 0.0
        core3 = 0.0
        core4 = 0.0
    core2 = max(core2, 0.0)
    shift_data = delta_data - delta
    mu = ref + delta  # materialized only for the ulp-scale threshold's magnitude, never differenced
    if abs(shift_data) <= _MEAN_ROUNDING_BOUND * max(abs(mu), abs(ref)):
        shift_data = 0.0
    data2 = core2 + shift_data * shift_data
    data4 = core4 + 4.0 * shift_data * core3 + 6.0 * shift_data * shift_data * core2 + shift_data**4
    if pc <= 0.0 or prior is None:
        return data2, data4, repairs
    if prior_central is not None:
        _, c2, c3, c4 = prior_central
    else:
        # Degraded fallback (the caller has already warned): recover the prior's own central moments
        # by un-shifting its raw power-sum-about-ref payload. Cancellation-prone when the prior's own
        # location sits far from ref -- the same pre-existing limitation the warning discloses, no
        # worse than the un-split blend.
        _, c2, c3, c4 = _shift_moments(1.0, prior[0], prior[1], prior[2], prior[3], -prior[0])
        c2 = max(c2, 0.0)
    shift_prior = prior[0] - delta
    prior2 = c2 + shift_prior * shift_prior
    prior4 = c4 + 4.0 * shift_prior * c3 + 6.0 * shift_prior * shift_prior * c2 + shift_prior**4
    total = n + pc
    return (n * data2 + pc * prior2) / total, (n * data4 + pc * prior4) / total, repairs


class GeneralizedGaussianEstimator(ParameterEstimator):
    """Method-of-moments estimator: ``mu`` = mean, ``beta`` from excess kurtosis, ``alpha`` from variance.

    The reduced moments are formed about a reference point rather than about zero whenever the
    accumulated statistics carry a shift-anchored payload (see :class:`GeneralizedGaussianAccumulator`),
    which makes the fit shift-equivariant: ``estimate`` on ``x + c`` returns ``mu + c`` with ``alpha``
    and ``beta`` unchanged. With a plain raw tuple -- statistics restored from an older serialization,
    or a hand-built one -- the historical raw path is used unchanged.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float, float, float] | None = None,
        beta_bounds: tuple[float, float] = (0.25, 50.0),
        name: str | None = None,
        keys: str | None = None,
        prior_central: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Create a method-of-moments estimator.

        Args:
            pseudo_count: Weight of the prior pseudo-sample described by ``suff_stat``.
            suff_stat: Prior RAW moments ``(E[X], E[X^2], E[X^3], E[X^4])``.
            beta_bounds: Bracket for the shape root find.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior_central: Optional ``(mean, m2, m3, m4)`` CENTRAL restatement of ``suff_stat``,
                supplied by :meth:`GeneralizedGaussianDistribution.estimator`. Raw prior moments at a
                large location are not recoverable in float64 (``E[X^2]`` at ``mu = 1.7e9`` has an
                ulp of 512, so a variance of 1 is simply not present in the number); when the data
                needs the anchored track, this payload lets the prior be placed on the anchor exactly
                instead. Without it a large-location prior is still blended the historical raw way,
                and ``estimate`` warns rather than pretending the blend was well-conditioned.
        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.beta_bounds = beta_bounds
        self.name = name
        self.keys = keys
        self.prior_central = None if prior_central is None else tuple(float(v) for v in prior_central)

    def accumulator_factory(self) -> GeneralizedGaussianAccumulatorFactory:
        """Return an accumulator factory for generalized-Gaussian raw moments."""
        return GeneralizedGaussianAccumulatorFactory(name=self.name, keys=self.keys)

    def _prior_about(self, ref: float) -> tuple[float, float, float, float] | None:
        """Prior power sums for one unit of pseudo-count, expressed about ``ref``.

        Returns ``None`` when there is no prior. Uses the central payload when available (exact at
        any reference point); otherwise shifts the stored raw moments, which is only well-conditioned
        when ``ref`` is zero or the prior's own location is small.
        """
        if self.pseudo_count is None or self.suff_stat is None:
            return None
        e1, e2, e3, e4 = (float(v) for v in self.suff_stat)
        if ref == 0.0:
            # The stored raw moments ARE the power sums about zero; using them verbatim keeps the
            # historical blend bit-identical even when a central payload is also available.
            return e1, e2, e3, e4
        # getattr, not attribute access: an estimator unpickled from a release that predates this
        # field has no such attribute, and the right answer there is the historical raw shift.
        central = getattr(self, "prior_central", None)
        if central is not None:
            mean0, c2, c3, c4 = central
            u = mean0 - ref
            u2 = u * u
            return (u, c2 + u2, c3 + 3.0 * u * c2 + u2 * u, c4 + 4.0 * u * c3 + 6.0 * u2 * c2 + u2 * u2)
        return _shift_moments(1.0, e1, e2, e3, e4, -ref)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float]
    ) -> GeneralizedGaussianDistribution:
        """Estimate location, scale, and shape from weighted raw moments."""
        from scipy.optimize import brentq

        count, s1, s2, s3, s4 = (float(v) for v in suff_stat[:5])
        anchored = _consistent_anchored_moments(suff_stat, s1, count)
        # Everything below is the historical algebra with an explicit reference point: at ref = 0 the
        # power sums ARE the raw sums and every formula reduces to exactly what it was before.
        if anchored is None:
            # Raw-only statistics cannot be corrected here; before this the family was silent, and
            # sd ~2 data at offset 1.7e9 handed in as the declared raw tuple returned alpha = 1e-6
            # (the degenerate branch below) for a true 2.7127. The gate reads the second moment,
            # which is where the loss shows up first; the fourth cannot survive it.
            warn_uncorrectable_raw_moments(s1, s2, count, family="generalized Gaussian")
            ref, p1, p2, p3, p4 = 0.0, s1, s2, s3, s4
        else:
            ref, p1, p2, p3, p4 = anchored
        n, a1, a2, a3, a4 = count, p1, p2, p3, p4  # data-only weight/moments, before any prior blend
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
                    "GeneralizedGaussianEstimator is blending raw prior moments whose own location "
                    "dominates their spread into data that needed shift-anchored accumulation; the "
                    "prior's central moments cannot be recovered from raw float64 power sums, so the "
                    "blended alpha/beta are unreliable. Build the prior with "
                    "GeneralizedGaussianDistribution.estimator(pseudo_count=...) (which carries the "
                    "central moments), or pass prior_central=(mean, m2, m3, m4).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                repairs.append("prior-moments-ill-conditioned")
            pc = float(self.pseudo_count)
            p1 += pc * prior[0]
            p2 += pc * prior[1]
            p3 += pc * prior[2]
            p4 += pc * prior[3]
            count += pc
        if count <= 0.0:
            return GeneralizedGaussianDistribution(0.0, 1.0, 2.0, name=self.name, keys=self.keys)
        delta = p1 / count  # the mean, measured from ref
        mu = ref + delta
        if anchored is not None:
            # Anchored path: the data's own moments (a1..a4) are already O(spread) about ref, so
            # split the second/fourth central moments about mu into a data-only "core" (well
            # conditioned, gated by the RELATIVE cancellation clamp) plus the displacement of mu from
            # the data's own mean (gated by the ulp-scale clamp) -- see _anchored_central_moments.
            # This never differences two O(magnitude^2) quantities, unlike computing m2/m4 directly
            # from the prior-blended p2..p4, so genuine spread at extreme magnitude survives.
            m2, m4, moment_repairs = _anchored_central_moments(
                ref, n, a1, a2, a3, a4, delta, pc, prior, getattr(self, "prior_central", None)
            )
            repairs.extend(moment_repairs)
        else:
            # Historical raw path (ref = 0): bit-identical to before the split.
            r2, r3, r4 = p2 / count, p3 / count, p4 / count
            m2 = r2 - delta * delta
            m4 = r4 - 4.0 * delta * r3 + 6.0 * delta * delta * r2 - 3.0 * delta**4
        if m2 <= 0.0:
            return GeneralizedGaussianDistribution(mu, 1.0e-6, 2.0, name=self.name, keys=self.keys)
        k = m4 / (m2 * m2) - 3.0  # sample excess kurtosis
        lo, hi = self.beta_bounds
        k_lo, k_hi = _excess_kurtosis(lo), _excess_kurtosis(hi)  # k decreases as beta grows
        if k >= k_lo:
            beta = lo
        elif k <= k_hi:
            beta = hi
        else:
            beta = float(brentq(lambda b: _excess_kurtosis(b) - k, lo, hi, xtol=1.0e-8))
        alpha = math.sqrt(m2 * gamma(1.0 / beta) / gamma(3.0 / beta))
        dist = GeneralizedGaussianDistribution(mu, alpha, beta, name=self.name, keys=self.keys)
        if repairs:
            dist._numerical_repairs = tuple(repairs)
        return dist


class GeneralizedGaussianDataEncoder(DataSequenceEncoder):
    """Encode observations as a float array."""

    def __str__(self) -> str:
        return "GeneralizedGaussianDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedGaussianDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return finite_observations(x, label="generalized-Gaussian observations")
