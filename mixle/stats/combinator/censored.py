"""Censoring likelihood combinator with explicitly typed exact and interval evidence.

``CensoredDistribution`` wraps a base distribution and scores **censored** observations -- ones
that are not observed exactly but are known only to fall in some interval ``[a, b]``.  Such an
observation contributes its interval probability mass

    P(a <= X <= b) = F(b) - F(a)

to the likelihood, where ``F`` is the base distribution's CDF (for a discrete base, whose CDF has a
jump at ``a``, the point mass ``P(X = a)`` is added back in -- see ``Data type`` below).  This is
distinct from truncation:
truncation *renormalizes* the density over a restricted support (the observation is exact but the
support is limited), whereas censoring keeps the original distribution and only coarsens what was
observed.  Survival-analysis right/left/interval censoring (and Tobit-style bounds) are all this
combinator.

Data type: each observation is one of

    * ``ExactObservation(x)`` or a raw child value ``x`` -- exact evidence;
    * ``CensoredInterval(a, b)`` -- interval evidence over the closed interval ``[a, b]``.

Raw two-element tuples are exact child observations, so bivariate child values remain representable.
Censored evidence is a likelihood contribution, not an outcome sampled by the base model: an observation
process must be modeled separately when interval selection itself is random.

The base distribution must expose ``cdf``.  Estimation of the base parameters under censoring has no
generic closed form (the censored MLE couples the bounds with the base parameters), so the supplied
estimator fits the base on the **exact** observations only and re-wraps; document and prefer a
dedicated censored MLE when the censored fraction is large.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.random import RandomState

from mixle.stats.combinator._base import MaskedBaseEncoder
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import logsubexp

# Tiny finite log-mass floor for an interval that underflowed to 0 in probability space when the base
# exposes only a linear ``cdf`` (~log of the smallest positive normal double); avoids silently zeroing.
_LOG_MASS_FLOOR = math.log(np.finfo(np.float64).tiny)


@dataclass(frozen=True)
class ExactObservation:
    """Explicit exact evidence; useful when the child value could otherwise look like metadata."""

    value: Any


@dataclass(frozen=True)
class CensoredInterval:
    """Closed interval likelihood evidence with validated ordered bounds."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        try:
            lower = float(self.lower)
            upper = float(self.upper)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("censoring bounds must be real numeric scalars.") from exc
        if math.isnan(lower) or math.isnan(upper):
            raise ValueError("censoring bounds cannot be NaN.")
        if lower > upper:
            raise ValueError("censoring interval lower bound cannot exceed its upper bound.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _is_interval(x: Any) -> bool:
    return isinstance(x, CensoredInterval)


def _exact_value(x: Any) -> Any:
    return x.value if isinstance(x, ExactObservation) else x


def _is_discrete_base(base: Any) -> bool:
    """Return whether ``base`` has enumerable (discrete/atomic) support.

    Probes :meth:`~mixle.stats.compute.pdist.SequenceEncodableProbabilityDistribution.enumerator`
    the same way combinator enumerators already do (e.g. ``TruncatedEnumerator``): a distribution
    with atomic support implements it, while a continuous one raises :class:`EnumerationError`. A
    ``base`` that does not even expose ``enumerator`` (e.g. a minimal duck-typed stub that only
    implements the handful of methods a particular caller needs) is likewise treated as continuous,
    matching the base class's own default. This is how closed-interval scoring tells apart a
    ``base.log_density(v)`` that is itself a probability (an atom, for a discrete base) from one
    that is a density (zero-measure at any single point, for a continuous base).
    """
    if not hasattr(base, "enumerator"):
        return False
    try:
        base.enumerator()
    except EnumerationError:
        return False
    return True


class CensoredDistribution(SequenceEncodableProbabilityDistribution):
    """A base distribution whose observations may be interval/left/right censored."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
        *,
        fit_receipt: CensoredExactOnlyFitReceipt | None = None,
    ) -> None:
        """Create a censored wrapper around ``base``.

        Args:
            base: The base distribution. Must expose ``cdf`` for interval scoring.
            name, keys: Optional instance name / parameter key.
        """
        if not hasattr(base, "cdf"):
            raise ValueError("CensoredDistribution requires the base distribution to expose `cdf`.")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        if keys is not None and not isinstance(keys, str):
            raise ValueError("keys must be a string or None.")
        self.base = base
        self.name = name
        self.keys = keys
        self.fit_receipt = fit_receipt
        # Cached once (not re-probed per observation): whether a single point of `base` carries a
        # true point-mass probability (discrete/atomic support) rather than a zero-measure density
        # value (continuous support). See `_is_discrete_base`.
        self._discrete_base = _is_discrete_base(base)

    def __str__(self) -> str:
        return "CensoredDistribution(%s, name=%s, keys=%s)" % (str(self.base), repr(self.name), repr(self.keys))

    def _interval_log_mass(self, a: float, b: float) -> float:
        """Return the log-mass of the *closed* censoring interval ``[a, b]``, in log space.

        For a continuous base a single point has zero measure, so this is just
        ``log(F(b) - F(a))``. For a discrete base the closed interval also includes the point mass
        at ``a`` itself -- ``F(a) = P(X <= a)`` already nets ``a`` out of ``F(b) - F(a)``, which is
        really ``P(a < X <= b)`` -- so that open-lower mass has ``P(X = a)`` added back on. See
        :meth:`_open_lower_log_mass` and :func:`_is_discrete_base`.

        Tail censoring is the normal use case, so the open-lower mass routinely underflows to ``0``
        in probability space (``log(0) = -inf``) even when the true interval log-mass is a perfectly
        finite large-negative number. When the base distribution exposes ``logcdf``/``logsf`` the mass
        is formed by a stable :func:`mixle.utils.special.logsubexp`; otherwise the linear
        ``F(b) - F(a)`` is used but the underflow is guarded (for a continuous base only) so a real
        far-tail interval is not silently zeroed.
        """
        if math.isnan(a) or math.isnan(b):
            raise ValueError("censoring bounds cannot be NaN.")
        if b < a:
            raise ValueError("censoring interval lower bound cannot exceed its upper bound.")
        discrete = self._discrete_base
        if a == b:
            if discrete:
                # A closed single-point interval on a discrete base is exactly that outcome's mass
                # (itself ``-inf`` if ``a`` is not in the support).
                point_log_mass = float(self.base.log_density(a))
                if math.isnan(point_log_mass) or point_log_mass > 0.0:
                    raise ValueError("atomic base returned an invalid point log-probability.")
                return point_log_mass
            # A degenerate (zero-width) interval has genuinely zero mass under a continuous base; this
            # is a true ``-inf``, not an underflow, so return it before any underflow floor kicks in.
            return -math.inf
        log_mass = self._open_lower_log_mass(a, b)
        if discrete and a != -math.inf:
            # Closed-below: add back X == a, which the open-lower F(b) - F(a) formula nets out.
            log_pa = float(self.base.log_density(a))
            if math.isnan(log_pa) or log_pa > 0.0:
                raise ValueError("atomic base returned an invalid point log-probability.")
            if log_pa > -math.inf:
                log_mass = float(np.logaddexp(log_mass, log_pa))
        return log_mass

    def _open_lower_log_mass(self, a: float, b: float) -> float:
        """Return ``log P(a < X <= b)``, the open-lower/closed-upper mass that
        :meth:`_interval_log_mass` builds on (closedness at ``a`` for a discrete base is layered on
        by the caller).
        """
        has_logsf = hasattr(self.base, "logsf")
        has_logcdf = hasattr(self.base, "logcdf")
        if has_logsf or has_logcdf:
            # log(F(b) - F(a)) = logsubexp(log F(b), log F(a)); in the upper tail prefer the survival
            # function: F(b) - F(a) = S(a) - S(b) = logsubexp(log S(a), log S(b)).
            if has_logsf:
                log_sa = 0.0 if a == -math.inf else float(self.base.logsf(a))
                log_sb = -math.inf if b == math.inf else float(self.base.logsf(b))
                if math.isnan(log_sa) or math.isnan(log_sb) or log_sa > 0.0 or log_sb > 0.0 or log_sb > log_sa:
                    raise ValueError("base log-survival values are invalid or non-monotone.")
                result = logsubexp(log_sa, log_sb)
                if math.isnan(result):
                    raise ValueError("base log-survival values produced an undefined interval mass.")
                return result
            log_fa = -math.inf if a == -math.inf else float(self.base.logcdf(a))
            log_fb = 0.0 if b == math.inf else float(self.base.logcdf(b))
            if math.isnan(log_fa) or math.isnan(log_fb) or log_fa > 0.0 or log_fb > 0.0 or log_fa > log_fb:
                raise ValueError("base log-CDF values are invalid or non-monotone.")
            result = logsubexp(log_fb, log_fa)
            if math.isnan(result):
                raise ValueError("base log-CDF values produced an undefined interval mass.")
            return result
        fa = 0.0 if a == -math.inf else float(self.base.cdf(a))
        fb = 1.0 if b == math.inf else float(self.base.cdf(b))
        if (
            not np.isfinite(fa)
            or not np.isfinite(fb)
            or fa < 0.0
            or fa > 1.0
            or fb < 0.0
            or fb > 1.0
            or fb < fa
        ):
            raise ValueError("base CDF values must be finite, in [0, 1], and monotone.")
        mass = fb - fa
        if mass > 0.0:
            return math.log(mass)
        if self._discrete_base:
            # An exact discrete CDF difference has no float-cancellation underflow the way a
            # continuous tail integral does (its operands are exact sums of atomic masses, not an
            # asymptotically-saturating tail integral); `mass <= 0` here means no atom lies in (a, b].
            return -math.inf
        # The interval mass underflowed to 0 in probability space (both CDFs rounded to the same
        # value). This is a genuine zero -- not an underflow -- when the base has no density at
        # either endpoint either: a real far-tail interval still has a representable (if tiny) density
        # at its boundary even after its CDF has saturated to 0/1 in float64 (the only way to do
        # better here is a base that exposes logcdf/logsf, handled above). Clamp to a tiny finite
        # floor in that case so a real far-tail interval is not silently zeroed; otherwise -- e.g. an
        # interval entirely outside a bounded base's support -- report the true, exact zero.
        da = 0.0 if math.isinf(a) else float(self.base.density(a))
        db = 0.0 if math.isinf(b) else float(self.base.density(b))
        if not np.isfinite(da) or not np.isfinite(db) or da < 0.0 or db < 0.0:
            raise ValueError("base density returned an invalid endpoint value.")
        if da == 0.0 and db == 0.0:
            return -math.inf
        return _LOG_MASS_FLOOR

    def density(self, x: Any) -> float:
        """Return the contribution of ``x`` (density for exact, interval mass for censored)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Log-likelihood contribution of ``x``.

        Exact observation ``x`` -> ``log p_base(x)``; interval ``(a, b)`` -> ``log P(a <= X <= b)``
        (see :meth:`_interval_log_mass`).
        """
        if _is_interval(x):
            return self._interval_log_mass(x.lower, x.upper)
        return float(self.base.log_density(_exact_value(x)))

    def seq_log_density(self, x: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Per-row censored log-densities for an encoded batch."""
        exact_enc, exact_idx, lows, highs, cens_idx = _validated_encoded_batch(x)
        n = len(exact_idx) + len(cens_idx)
        rv = np.empty(n, dtype=np.float64)

        if len(exact_idx) > 0:
            rv[exact_idx] = np.asarray(self.base.seq_log_density(exact_enc), dtype=np.float64)

        for k in range(len(cens_idx)):
            rv[cens_idx[k]] = self._interval_log_mass(float(lows[k]), float(highs[k]))

        return rv

    def sampler(self, seed: int | None = None) -> CensoredSampler:
        """Return exact evidence from the base; censoring requires a separate observation process."""
        return CensoredSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ExactOnlyCensoredEstimator:
        """Return the explicit exact-only projection estimator with an effective-count receipt.

        This is not a censored MLE. Use :meth:`likelihood_estimator` with a base-family-specific fitting
        callback when interval evidence must influence the fitted parameters.
        """
        return ExactOnlyCensoredEstimator(
            self.base.estimator(pseudo_count=pseudo_count),
            name=self.name,
            keys=self.keys,
        )

    def likelihood_estimator(
        self,
        fit: Callable[[tuple[Any, ...], np.ndarray, Any], SequenceEncodableProbabilityDistribution],
    ) -> CensoredLikelihoodEstimator:
        """Return a likelihood-aware estimator backed by a declared base-family fitting callback.

        ``fit(observations, weights, initial_base)`` receives validated exact/interval evidence and must
        optimize the full censored objective, returning a fitted base distribution.
        """
        if not callable(fit):
            raise ValueError("fit must be callable.")
        return CensoredLikelihoodEstimator(self.base, fit, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> CensoredDataEncoder:
        """Return the data encoder (exact observations + censoring intervals split out)."""
        return CensoredDataEncoder(self)


class CensoredSampler(DistributionSampler):
    """Draw exact values from the base distribution (the sampler is not censored)."""

    def __init__(self, dist: CensoredDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.rng = RandomState(seed)
        self.base_sampler = dist.base.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw exact value(s) from the base distribution."""
        sample = self.base_sampler.sample(size=size)
        if size is None:
            return ExactObservation(sample)
        return [ExactObservation(value) for value in sample]


class CensoredExactOnlyFitReceipt(NamedTuple):
    exact_weight: float
    censored_weight: float
    censored_fraction: float
    likelihood_aware: bool


class CensoredExactOnlyStatistics(NamedTuple):
    schema_version: int
    child: Any
    exact_weight: float
    censored_weight: float


def _finite_weight(value: Any, *, name: str = "weight") -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _validate_exact_only_statistics(value: Any) -> CensoredExactOnlyStatistics:
    if not isinstance(value, CensoredExactOnlyStatistics) or value.schema_version != 1:
        raise ValueError("exact-only censored statistics must use schema version 1.")
    exact_weight = _finite_weight(value.exact_weight, name="exact_weight")
    censored_weight = _finite_weight(value.censored_weight, name="censored_weight")
    return CensoredExactOnlyStatistics(1, value.child, exact_weight, censored_weight)


def _validated_encoded_batch(
    x: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(x, tuple) or len(x) != 5:
        raise ValueError("encoded censored evidence must have five components.")
    exact_enc, exact_idx, lows, highs, cens_idx = x
    exact_idx = np.asarray(exact_idx)
    cens_idx = np.asarray(cens_idx)
    lows = np.asarray(lows, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    if (
        exact_idx.ndim != 1
        or cens_idx.ndim != 1
        or not np.issubdtype(exact_idx.dtype, np.integer)
        or not np.issubdtype(cens_idx.dtype, np.integer)
        or lows.shape != cens_idx.shape
        or highs.shape != cens_idx.shape
    ):
        raise ValueError("encoded censored evidence has incompatible index or bound shapes.")
    nrows = len(exact_idx) + len(cens_idx)
    all_indices = np.concatenate([exact_idx.astype(np.int64), cens_idx.astype(np.int64)])
    if not np.array_equal(np.sort(all_indices), np.arange(nrows)):
        raise ValueError("exact and censored indices must partition all encoded rows exactly once.")
    for lower, upper in zip(lows, highs):
        CensoredInterval(float(lower), float(upper))
    return exact_enc, exact_idx.astype(np.int64), lows, highs, cens_idx.astype(np.int64)


class CensoredAccumulator(SequenceEncodableStatisticAccumulator):
    """Explicit exact-only projection statistics with auditable discarded interval weight."""

    def __init__(self, base_accumulator: SequenceEncodableStatisticAccumulator, keys: str | None = None) -> None:
        self.base_accumulator = base_accumulator
        self.keys = keys
        self.exact_weight = 0.0
        self.censored_weight = 0.0

    def update(self, x: Any, weight: float, estimate: CensoredDistribution | None) -> None:
        checked_weight = _finite_weight(weight)
        if _is_interval(x):
            self.censored_weight += checked_weight
            return
        self.base_accumulator.update(
            _exact_value(x),
            checked_weight,
            None if estimate is None else estimate.base,
        )
        self.exact_weight += checked_weight

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        checked_weight = _finite_weight(weight)
        if _is_interval(x):
            self.censored_weight += checked_weight
            return
        self.base_accumulator.initialize(_exact_value(x), checked_weight, rng)
        self.exact_weight += checked_weight

    def seq_update(
        self,
        x: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: CensoredDistribution | None,
    ) -> None:
        exact_enc, exact_idx, _lows, _highs, cens_idx = _validated_encoded_batch(x)
        checked_weights = np.asarray(weights, dtype=np.float64)
        nrows = len(exact_idx) + len(cens_idx)
        if checked_weights.shape != (nrows,) or np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite, non-negative, and aligned with encoded censored rows.")
        if len(exact_idx):
            exact_weights = checked_weights[exact_idx]
            self.base_accumulator.seq_update(
                exact_enc,
                exact_weights,
                None if estimate is None else estimate.base,
            )
            self.exact_weight += float(exact_weights.sum())
        self.censored_weight += float(checked_weights[cens_idx].sum())

    def seq_initialize(
        self,
        x: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        rng: RandomState | None,
    ) -> None:
        exact_enc, exact_idx, _lows, _highs, cens_idx = _validated_encoded_batch(x)
        checked_weights = np.asarray(weights, dtype=np.float64)
        nrows = len(exact_idx) + len(cens_idx)
        if checked_weights.shape != (nrows,) or np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite, non-negative, and aligned with encoded censored rows.")
        if len(exact_idx):
            exact_weights = checked_weights[exact_idx]
            self.base_accumulator.seq_initialize(exact_enc, exact_weights, rng)
            self.exact_weight += float(exact_weights.sum())
        self.censored_weight += float(checked_weights[cens_idx].sum())

    def combine(self, suff_stat: CensoredExactOnlyStatistics) -> CensoredAccumulator:
        checked = _validate_exact_only_statistics(suff_stat)
        self.base_accumulator.combine(checked.child)
        self.exact_weight += checked.exact_weight
        self.censored_weight += checked.censored_weight
        return self

    def value(self) -> CensoredExactOnlyStatistics:
        return CensoredExactOnlyStatistics(
            1,
            self.base_accumulator.value(),
            self.exact_weight,
            self.censored_weight,
        )

    def from_value(self, x: CensoredExactOnlyStatistics) -> CensoredAccumulator:
        checked = _validate_exact_only_statistics(x)
        self.base_accumulator.from_value(checked.child)
        self.exact_weight = checked.exact_weight
        self.censored_weight = checked.censored_weight
        return self

    def scale(self, c: float) -> CensoredAccumulator:
        checked = _finite_weight(c, name="scale")
        self.base_accumulator.scale(checked)
        self.exact_weight *= checked
        self.censored_weight *= checked
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is None:
            self.base_accumulator.key_merge(stats_dict)
            return
        if self.keys in stats_dict:
            self.combine(stats_dict[self.keys])
        stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is None:
            self.base_accumulator.key_replace(stats_dict)
        elif self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> CensoredDataEncoder:
        return CensoredDataEncoder.from_base_encoder(self.base_accumulator.acc_to_encoder())


class CensoredAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`CensoredAccumulator`."""

    def __init__(self, base_factory: StatisticAccumulatorFactory, keys: str | None = None) -> None:
        self.base_factory = base_factory
        self.keys = keys

    def make(self) -> CensoredAccumulator:
        """Create an empty censored-data accumulator."""
        return CensoredAccumulator(self.base_factory.make(), keys=self.keys)


class ExactOnlyCensoredEstimator(ParameterEstimator):
    """Projection estimator that deliberately excludes interval evidence and reports its weight."""

    def __init__(
        self,
        base_estimator: ParameterEstimator,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> CensoredAccumulatorFactory:
        """Return a factory for censored-data sufficient-statistic accumulators."""
        return CensoredAccumulatorFactory(self.base_estimator.accumulator_factory(), keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: Any) -> CensoredDistribution:
        checked = _validate_exact_only_statistics(suff_stat)
        if checked.exact_weight <= 0.0:
            raise ValueError("exact-only censored estimation requires positive exact-observation weight.")
        base = self.base_estimator.estimate(checked.exact_weight, checked.child)
        total = checked.exact_weight + checked.censored_weight
        receipt = CensoredExactOnlyFitReceipt(
            checked.exact_weight,
            checked.censored_weight,
            checked.censored_weight / total if total > 0.0 else 0.0,
            False,
        )
        return CensoredDistribution(
            base,
            name=self.name,
            keys=self.keys,
            fit_receipt=receipt,
        )


CensoredEstimator = ExactOnlyCensoredEstimator


class CensoredLikelihoodStatistics(NamedTuple):
    schema_version: int
    observations: tuple[Any, ...]
    weights: np.ndarray


class CensoredEvidenceEncoder(DataSequenceEncoder):
    """Identity encoder that validates explicit exact/interval evidence for likelihood fitting."""

    def seq_encode(self, x: Sequence[Any]) -> tuple[Any, ...]:
        observations = tuple(x)
        for observation in observations:
            if _is_interval(observation):
                CensoredInterval(observation.lower, observation.upper)
        return observations

    def row_count(self, x: Any) -> int:
        if not isinstance(x, tuple):
            raise ValueError("encoded censored likelihood evidence must be a tuple.")
        return len(x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CensoredEvidenceEncoder)


class CensoredLikelihoodAccumulator(SequenceEncodableStatisticAccumulator):
    """Retain validated evidence for a base-family-specific full censored-likelihood fit."""

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.weights: list[float] = []

    def update(self, x: Any, weight: float, estimate: Any | None) -> None:
        checked_weight = _finite_weight(weight)
        if _is_interval(x):
            CensoredInterval(x.lower, x.upper)
        self.observations.append(x)
        self.weights.append(checked_weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: tuple[Any, ...], weights: np.ndarray, estimate: Any | None) -> None:
        observations = CensoredEvidenceEncoder().seq_encode(x)
        checked_weights = np.asarray(weights, dtype=np.float64)
        if (
            checked_weights.shape != (len(observations),)
            or np.any(~np.isfinite(checked_weights))
            or np.any(checked_weights < 0.0)
        ):
            raise ValueError("weights must be finite, non-negative, and aligned with censored evidence.")
        self.observations.extend(observations)
        self.weights.extend(checked_weights.tolist())

    def seq_initialize(self, x: tuple[Any, ...], weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: CensoredLikelihoodStatistics) -> CensoredLikelihoodAccumulator:
        if not isinstance(suff_stat, CensoredLikelihoodStatistics) or suff_stat.schema_version != 1:
            raise ValueError("censored likelihood statistics must use schema version 1.")
        observations = CensoredEvidenceEncoder().seq_encode(suff_stat.observations)
        weights = np.asarray(suff_stat.weights, dtype=np.float64)
        if weights.shape != (len(observations),) or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("censored likelihood statistic weights are invalid.")
        self.observations.extend(observations)
        self.weights.extend(weights.tolist())
        return self

    def value(self) -> CensoredLikelihoodStatistics:
        return CensoredLikelihoodStatistics(
            1,
            tuple(self.observations),
            np.asarray(self.weights, dtype=np.float64),
        )

    def from_value(self, x: CensoredLikelihoodStatistics) -> CensoredLikelihoodAccumulator:
        self.observations = []
        self.weights = []
        return self.combine(x)

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        pass

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> CensoredEvidenceEncoder:
        return CensoredEvidenceEncoder()


class CensoredLikelihoodAccumulatorFactory(StatisticAccumulatorFactory):
    def make(self) -> CensoredLikelihoodAccumulator:
        return CensoredLikelihoodAccumulator()


class CensoredLikelihoodEstimator(ParameterEstimator):
    """Delegate full censored-objective optimization to a declared base-family callback."""

    def __init__(
        self,
        initial_base: Any,
        fit: Callable[[tuple[Any, ...], np.ndarray, Any], SequenceEncodableProbabilityDistribution],
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.initial_base = initial_base
        self.fit = fit
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> CensoredLikelihoodAccumulatorFactory:
        return CensoredLikelihoodAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: CensoredLikelihoodStatistics) -> CensoredDistribution:
        accumulator = CensoredLikelihoodAccumulator().from_value(suff_stat)
        weights = np.asarray(accumulator.weights, dtype=np.float64)
        if len(accumulator.observations) == 0 or float(weights.sum()) <= 0.0:
            raise ValueError("censored likelihood estimation requires positive evidence weight.")
        base = self.fit(tuple(accumulator.observations), weights, self.initial_base)
        if not isinstance(base, SequenceEncodableProbabilityDistribution) or not hasattr(base, "cdf"):
            raise ValueError("censored likelihood fit callback must return a CDF-capable probability distribution.")
        censored_weight = sum(
            weight
            for observation, weight in zip(accumulator.observations, weights)
            if _is_interval(observation)
        )
        total = float(weights.sum())
        receipt = CensoredExactOnlyFitReceipt(total - censored_weight, censored_weight, censored_weight / total, True)
        return CensoredDistribution(base, name=self.name, keys=self.keys, fit_receipt=receipt)


class CensoredDataEncoder(MaskedBaseEncoder):
    """Split a batch into exact observations (base-encoded) and censoring intervals.

    Encoded form is ``(exact_enc, exact_idx, lows, highs, cens_idx)`` where ``exact_idx`` /
    ``cens_idx`` are the original positions of the exact / censored rows so ``seq_log_density``
    can scatter results back into the right order.
    """

    def __init__(
        self, dist: CensoredDistribution | None = None, base_encoder: DataSequenceEncoder | None = None
    ) -> None:
        if base_encoder is not None:
            self.base_encoder = base_encoder
        elif dist is not None:
            self.base_encoder = dist.base.dist_to_encoder()
        else:
            raise ValueError("CensoredDataEncoder needs a distribution or a base encoder.")

    @classmethod
    def from_base_encoder(cls, base_encoder: DataSequenceEncoder) -> CensoredDataEncoder:
        """Create a censored encoder from an already configured base encoder."""
        return cls(base_encoder=base_encoder)

    def seq_encode(self, x: Sequence[Any]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode a batch into exact values, interval bounds, and row indices."""
        exact_vals: list[Any] = []
        exact_idx: list[int] = []
        lows: list[float] = []
        highs: list[float] = []
        cens_idx: list[int] = []
        for i, v in enumerate(x):
            if _is_interval(v):
                interval = CensoredInterval(v.lower, v.upper)
                lows.append(interval.lower)
                highs.append(interval.upper)
                cens_idx.append(i)
            else:
                exact_vals.append(_exact_value(v))
                exact_idx.append(i)
        exact_enc = self.base_encoder.seq_encode(exact_vals) if exact_vals else self.base_encoder.seq_encode([])
        return (
            exact_enc,
            np.asarray(exact_idx, dtype=np.int64),
            np.asarray(lows, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(cens_idx, dtype=np.int64),
        )

    def row_count(self, x: Any) -> int:
        """Return the number of exact plus interval evidence rows."""
        _exact_enc, exact_idx, _lows, _highs, cens_idx = _validated_encoded_batch(x)
        return len(exact_idx) + len(cens_idx)
