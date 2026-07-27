"""Truncated-support combinator: restrict a base distribution to an allowed set, renormalized.

``TruncatedDistribution`` wraps a base distribution and conditions it on a support restriction:

    p(x) = p_base(x) / Z   for x in the allowed support,   else 0,

where ``Z = sum_{y allowed} p_base(y)`` is the retained mass.  The restriction is given either as a
finite ``forbidden`` set to exclude (``Z = 1 - sum_f p_base(f)`` -- works for an infinite base) or a
finite ``allowed`` set to keep (``Z = sum_a p_base(a)``).  It pairs with the Phase-1c support tools:
the renormalizer is exactly the truncated tail/total the enumeration bounds reason about.

``allowed``/``forbidden`` are deduplicated (by value, not by position), so a repeated entry cannot
double-count its mass.  The point-mass sums above assume a *discrete* (enumerable) base, where
``p_base(v)`` at a single value ``v`` is itself a probability.  For a *continuous* base, any single
point has zero measure, so a finite point list can only ever remove or retain zero mass: forbidding
one is therefore a no-op (``Z = 1``) and allowing one necessarily retains nothing (``Z = 0``, which
raises -- see below).
"""

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.capability import Discrete, supports
from mixle.enumeration.algorithms import freeze
from mixle.stats.combinator._base import MaskedBaseEncoder
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import log1mexp

_REJECTION_BUDGET = 1_000_000  # max base draws before a rejection sampler gives up


def _is_discrete_base(base: Any) -> bool:
    """Return whether ``base`` declares discrete/atomic density semantics."""
    return supports(base, Discrete)


def _dedupe(values: Sequence[Any]) -> list[Any]:
    """Deduplicate values by stable identity plus scalar observation equality.

    Keeps the first occurrence and the input order, so a repeated ``allowed``/``forbidden`` entry
    (e.g. ``[0, 0]``) cannot be counted twice into the normalizing constant. Numeric aliases that
    the child accepts as the same observation (such as Bernoulli ``False`` and integer ``0``) are
    likewise one support point.
    """
    out: list[Any] = []
    for value in values:
        if not any(_same_support_value(value, prior) for prior in out):
            out.append(value)
    return out


def _same_support_value(left: Any, right: Any) -> bool:
    """Return scalar support equality, including equivalent numeric representations."""
    try:
        if freeze(left) == freeze(right):
            return True
    except TypeError:
        pass
    try:
        equal = left == right
        if isinstance(equal, np.ndarray):
            return bool(np.all(equal))
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _finite_weight(value: Any, *, name: str) -> float:
    """Return one finite non-negative statistic weight."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a finite non-negative real number." % name) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be a finite non-negative real number." % name)
    return result


def _restriction_keys(
    allowed: Sequence[Any] | None,
    forbidden: Sequence[Any] | None,
) -> tuple[set[Any] | None, set[Any] | None]:
    if (allowed is None) == (forbidden is None):
        raise ValueError("Provide exactly one of `allowed` or `forbidden`.")
    return (
        None if allowed is None else set(map(freeze, allowed)),
        None if forbidden is None else set(map(freeze, forbidden)),
    )


def _is_allowed(
    value: Any,
    allowed_keys: set[Any] | None,
    forbidden_keys: set[Any] | None,
    allowed_values: Sequence[Any] | None,
    forbidden_values: Sequence[Any] | None,
) -> bool:
    try:
        key = freeze(value)
    except TypeError:
        return False
    if allowed_keys is not None:
        return key in allowed_keys or any(
            _same_support_value(value, allowed) for allowed in allowed_values
        )
    return key not in forbidden_keys and not any(
        _same_support_value(value, forbidden) for forbidden in forbidden_values
    )


def _atomic_log_probability(base: Any, value: Any) -> float:
    try:
        result = float(base.log_density(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("atomic base scores must be scalar log-probabilities.") from exc
    if math.isnan(result) or result > 0.0:
        raise ValueError("atomic base log-probabilities must be in [-inf, 0].")
    return result


class TruncatedProjectionFitReceipt(NamedTuple):
    """Evidence accounting for the explicit untruncated-base projection fit."""

    accepted_weight: float
    rejected_weight: float
    rejected_fraction: float
    likelihood_aware: bool


class TruncatedStatistics(NamedTuple):
    """Versioned sufficient statistics for support-filtered projection fitting."""

    schema_version: int
    child: Any
    accepted_weight: float
    rejected_weight: float


def _validate_statistics(value: Any) -> TruncatedStatistics:
    if not isinstance(value, TruncatedStatistics) or value.schema_version != 1:
        raise ValueError("truncated statistics must use schema version 1.")
    accepted = _finite_weight(value.accepted_weight, name="accepted_weight")
    rejected = _finite_weight(value.rejected_weight, name="rejected_weight")
    return TruncatedStatistics(1, value.child, accepted, rejected)


class TruncatedDistribution(SequenceEncodableProbabilityDistribution):
    """A base distribution restricted to an allowed support and renormalized."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        allowed: Sequence[Any] | None = None,
        forbidden: Sequence[Any] | None = None,
        name: str | None = None,
        keys: str | None = None,
        fit_receipt: TruncatedProjectionFitReceipt | None = None,
    ) -> None:
        """Create a truncated distribution.

        Args:
            base: The base distribution to restrict.
            allowed: Finite set of permitted values (keep only these). Mutually exclusive with ``forbidden``.
            forbidden: Finite set of excluded values (keep everything else). Works for an infinite base.
            name, keys: Optional instance name / parameter key.
        """
        if (allowed is None) == (forbidden is None):
            raise ValueError("Provide exactly one of `allowed` or `forbidden`.")
        self.base = base
        self.name = name
        self.keys = keys
        self.fit_receipt = fit_receipt
        # Deduplicated up front (by `freeze` key): a repeated entry (e.g. `allowed=[0, 0]`) must not
        # be counted twice into the normalizing constant below, nor into `support_size`.
        self._allowed_values = None if allowed is None else _dedupe(list(allowed))
        self._forbidden_values = None if forbidden is None else _dedupe(list(forbidden))
        self._allowed_keys, self._forbidden_keys = _restriction_keys(
            self._allowed_values,
            self._forbidden_values,
        )
        # The retained log-mass ``log Z`` is formed in log space so it survives the tail-censoring
        # regime: an ``allowed`` set whose atoms are individually tiny is summed by ``logsumexp``, and
        # a ``forbidden`` set whose mass is ~1 uses a stable ``log(1 - p_forbidden)`` instead of the
        # catastrophically cancelling ``1 - sum(exp(...))``. That point-mass accounting is only valid
        # for a discrete base, where `log_density(v)` at a single value is itself a log-probability;
        # for a continuous base a finite point list has zero measure, so `forbidden` is a mass-1 no-op
        # and `allowed` necessarily retains nothing (handled by the `log_z > -inf` check below).
        discrete = _is_discrete_base(base)
        if self._allowed_values is not None:
            if discrete:
                log_probs = [_atomic_log_probability(base, v) for v in self._allowed_values]
                finite = [lp for lp in log_probs if lp > -math.inf]
                log_z = float(logsumexp(finite)) if finite else -math.inf
            else:
                log_z = -math.inf
        else:
            if discrete:
                log_forbidden = [_atomic_log_probability(base, v) for v in self._forbidden_values]
                finite = [lp for lp in log_forbidden if lp > -math.inf]
                log_p_forbidden = float(logsumexp(finite)) if finite else -math.inf
                if log_p_forbidden > 0.0:
                    raise ValueError("Forbidden atomic probability mass exceeds one.")
                log_z = log1mexp(log_p_forbidden)
            else:
                log_z = 0.0
        if not (log_z > -math.inf):
            raise ValueError("Truncation retains no probability mass.")
        self.log_z = log_z

    def __str__(self) -> str:
        sel = (
            "allowed=%s" % repr(self._allowed_values)
            if self._allowed_values is not None
            else "forbidden=%s" % repr(self._forbidden_values)
        )
        return "TruncatedDistribution(%s, %s, name=%s, keys=%s)" % (
            str(self.base),
            sel,
            repr(self.name),
            repr(self.keys),
        )

    def _allowed(self, x: Any) -> bool:
        return _is_allowed(
            x,
            self._allowed_keys,
            self._forbidden_keys,
            self._allowed_values,
            self._forbidden_values,
        )

    def density(self, x: Any) -> float:
        """Return the renormalized probability/density at ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return ``log p_base(x) - log Z`` for allowed ``x``, else ``-inf``."""
        if not self._allowed(x):
            return -np.inf
        return float(self.base.log_density(x)) - self.log_z

    def seq_log_density(self, x: tuple[Any, np.ndarray]) -> np.ndarray:
        """Return per-row truncated log-densities for an encoded batch."""
        base_enc, allowed_mask = x
        rv = np.asarray(self.base.seq_log_density(base_enc), dtype=np.float64) - self.log_z
        rv[~allowed_mask] = -np.inf
        return rv

    def sampler(self, seed: int | None = None) -> "TruncatedSampler":
        """Return a rejection sampler over the allowed support."""
        return TruncatedSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "TruncatedProjectionEstimator":
        """Return the explicit projection estimator, not a truncated-likelihood MLE.

        The projection fits the child to retained observations and then reapplies the support
        restriction. It deliberately omits the parameter-dependent ``-n log Z(theta)`` term, so the
        returned model carries a receipt with ``likelihood_aware=False``.
        """
        return TruncatedProjectionEstimator(
            self.base.estimator(pseudo_count=pseudo_count),
            allowed=self._allowed_values,
            forbidden=self._forbidden_values,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "TruncatedDataEncoder":
        """Return the data encoder (base encoding + an allowed-membership mask)."""
        return TruncatedDataEncoder(self)

    def support_size(self) -> int | None:
        """Cardinality of the retained support (``None`` if infinite)."""
        if self._allowed_values is not None:
            return sum(
                _atomic_log_probability(self.base, value) > -math.inf
                for value in self._allowed_values
            )
        base_n = self.base.support_size()
        if base_n is None:
            return None
        forbidden_in = sum(
            _atomic_log_probability(self.base, value) > -math.inf
            for value in self._forbidden_values
        )
        return max(0, int(base_n) - forbidden_in)

    def enumerator(self) -> "TruncatedEnumerator":
        """Enumerate the allowed support in descending (renormalized) probability order."""
        return TruncatedEnumerator(self)


class TruncatedEnumerator(DistributionEnumerator):
    """Filter the base enumeration to the allowed support, renormalized by ``-log Z``."""

    def __init__(self, dist: TruncatedDistribution) -> None:
        super().__init__(dist)
        try:
            self._base_iter = iter(dist.base.enumerator())
        except EnumerationError as e:
            raise EnumerationError(dist, reason="truncation requires an enumerable base: %s" % e.reason) from None
        self._dist = dist

    def __next__(self) -> tuple[Any, float]:
        for value, lp in self._base_iter:
            if self._dist._allowed(value):
                return value, float(lp) - self._dist.log_z
        raise StopIteration


class TruncatedSampler(DistributionSampler):
    """Rejection sampler: draw from the base, keep only allowed values."""

    def __init__(self, dist: TruncatedDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.rng = RandomState(seed)
        self.base_sampler = dist.base.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one allowed value (or a list of ``size``) by rejection.

        With ``batched=True`` (default) base draws are taken in blocks sized from the running accept
        rate, instead of one per draw -- far faster when the retained mass is small, with a clear
        diagnostic (accept rate, attempts) if the budget is exhausted. Set ``batched=False`` for the
        per-draw reference path. Batched draws differ in RNG-call order from the per-draw loop.
        """
        if size is None:
            for _ in range(_REJECTION_BUDGET):
                v = self.base_sampler.sample()
                if self.dist._allowed(v):
                    return v
            raise RuntimeError("TruncatedSampler exceeded the rejection budget; retained mass may be tiny.")
        if not batched:
            return [self.sample() for _ in range(size)]
        # Precompute a numeric allowed/forbidden array so the accept test can vectorize via np.isin
        # (the per-element membership test, not base sampling, is the bottleneck for low retained mass).
        allowed_arr = forbidden_arr = None
        if self.dist._allowed_values is not None:
            a = np.asarray(self.dist._allowed_values)
            allowed_arr = a if a.dtype.kind in "iuf" else None
        elif self.dist._forbidden_values is not None:
            f = np.asarray(self.dist._forbidden_values)
            forbidden_arr = f if f.dtype.kind in "iuf" else None
        out: list[Any] = []
        attempts = 0
        budget = max(_REJECTION_BUDGET, int(size) * 1000)
        block = max(int(size), 256)
        while len(out) < size and attempts < budget:
            draws = self.base_sampler.sample(size=block)
            arr = np.asarray(draws)
            if (allowed_arr is not None or forbidden_arr is not None) and arr.dtype.kind in "iuf":
                mask = np.isin(arr, allowed_arr) if allowed_arr is not None else ~np.isin(arr, forbidden_arr)
                out.extend(arr[mask][: size - len(out)].tolist())
                attempts += arr.shape[0]
            else:
                for v in draws:
                    attempts += 1
                    if self.dist._allowed(v):
                        out.append(v)
                        if len(out) >= size:
                            break
            rate = len(out) / attempts if attempts else 0.0
            block = max(int((size - len(out)) / rate) + 16, 64) if rate > 0.0 else block * 2
        if len(out) < size:
            rate = len(out) / attempts if attempts else 0.0
            raise RuntimeError(
                "TruncatedSampler exceeded the rejection budget over %d attempts (accept rate %.2g); "
                "retained mass may be tiny." % (attempts, rate)
            )
        return out[:size]


class TruncatedAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate accepted child statistics and account explicitly for rejected evidence."""

    def __init__(
        self,
        base_accumulator: SequenceEncodableStatisticAccumulator,
        *,
        allowed: Sequence[Any] | None,
        forbidden: Sequence[Any] | None,
        keys: str | None = None,
    ) -> None:
        self.base_accumulator = base_accumulator
        self.keys = keys
        self._allowed_values = None if allowed is None else _dedupe(list(allowed))
        self._forbidden_values = None if forbidden is None else _dedupe(list(forbidden))
        self._allowed_keys, self._forbidden_keys = _restriction_keys(
            self._allowed_values,
            self._forbidden_values,
        )
        self.accepted_weight = 0.0
        self.rejected_weight = 0.0

    def _allowed(self, value: Any) -> bool:
        return _is_allowed(
            value,
            self._allowed_keys,
            self._forbidden_keys,
            self._allowed_values,
            self._forbidden_values,
        )

    def update(self, x: Any, weight: float, estimate: TruncatedDistribution | None) -> None:
        """Accumulate one observation under the same support rule as batch accumulation."""
        checked = _finite_weight(weight, name="weight")
        if self._allowed(x):
            self.base_accumulator.update(x, checked, None if estimate is None else estimate.base)
            self.accepted_weight += checked
        else:
            self.rejected_weight += checked

    def seq_update(
        self, x: tuple[Any, np.ndarray], weights: np.ndarray, estimate: TruncatedDistribution | None
    ) -> None:
        """Accumulate encoded observations, zeroing weights outside the retained support."""
        base_enc, allowed_mask = x
        allowed_mask = np.asarray(allowed_mask)
        checked = np.asarray(weights, dtype=np.float64)
        if (
            allowed_mask.dtype != np.bool_
            or allowed_mask.ndim != 1
            or checked.shape != allowed_mask.shape
            or np.any(~np.isfinite(checked))
            or np.any(checked < 0.0)
        ):
            raise ValueError("truncated mask and weights must be aligned, finite, and non-negative.")
        accepted = checked * allowed_mask
        self.base_accumulator.seq_update(
            base_enc,
            accepted,
            None if estimate is None else estimate.base,
        )
        self.accepted_weight += float(accepted.sum())
        self.rejected_weight += float(checked[~allowed_mask].sum())

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize under the same support rule as batch initialization."""
        checked = _finite_weight(weight, name="weight")
        if self._allowed(x):
            self.base_accumulator.initialize(x, checked, rng)
            self.accepted_weight += checked
        else:
            self.rejected_weight += checked

    def seq_initialize(self, x: tuple[Any, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize from encoded observations, zeroing weights outside the retained support."""
        base_enc, allowed_mask = x
        allowed_mask = np.asarray(allowed_mask)
        checked = np.asarray(weights, dtype=np.float64)
        if (
            allowed_mask.dtype != np.bool_
            or allowed_mask.ndim != 1
            or checked.shape != allowed_mask.shape
            or np.any(~np.isfinite(checked))
            or np.any(checked < 0.0)
        ):
            raise ValueError("truncated mask and weights must be aligned, finite, and non-negative.")
        accepted = checked * allowed_mask
        self.base_accumulator.seq_initialize(base_enc, accepted, rng)
        self.accepted_weight += float(accepted.sum())
        self.rejected_weight += float(checked[~allowed_mask].sum())

    def combine(self, suff_stat: TruncatedStatistics) -> "TruncatedAccumulator":
        checked = _validate_statistics(suff_stat)
        self.base_accumulator.combine(checked.child)
        self.accepted_weight += checked.accepted_weight
        self.rejected_weight += checked.rejected_weight
        return self

    def value(self) -> TruncatedStatistics:
        return TruncatedStatistics(
            1,
            self.base_accumulator.value(),
            self.accepted_weight,
            self.rejected_weight,
        )

    def from_value(self, x: TruncatedStatistics) -> "TruncatedAccumulator":
        checked = _validate_statistics(x)
        self.base_accumulator.from_value(checked.child)
        self.accepted_weight = checked.accepted_weight
        self.rejected_weight = checked.rejected_weight
        return self

    def scale(self, c: float) -> "TruncatedAccumulator":
        checked = _finite_weight(c, name="scale")
        self.base_accumulator.scale(checked)
        self.accepted_weight *= checked
        self.rejected_weight *= checked
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

    def acc_to_encoder(self) -> "TruncatedDataEncoder":
        """Return an encoder carrying the accumulator's exact support restriction."""
        return TruncatedDataEncoder.from_base_encoder(
            self.base_accumulator.acc_to_encoder(),
            allowed=self._allowed_values,
            forbidden=self._forbidden_values,
        )


class TruncatedAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`TruncatedAccumulator`."""

    def __init__(
        self,
        base_factory: StatisticAccumulatorFactory,
        *,
        allowed: Sequence[Any] | None,
        forbidden: Sequence[Any] | None,
        keys: str | None = None,
    ) -> None:
        self.base_factory = base_factory
        self.allowed = None if allowed is None else list(allowed)
        self.forbidden = None if forbidden is None else list(forbidden)
        self.keys = keys
        _restriction_keys(self.allowed, self.forbidden)

    def make(self) -> TruncatedAccumulator:
        """Create an empty truncated-data accumulator."""
        return TruncatedAccumulator(
            self.base_factory.make(),
            allowed=self.allowed,
            forbidden=self.forbidden,
            keys=self.keys,
        )


class TruncatedProjectionEstimator(ParameterEstimator):
    """Approximate projection that fits the base only to retained observations.

    This is intentionally not called a truncated MLE: it omits the parameter-dependent normalizer
    from the objective and marks every fitted model with ``likelihood_aware=False``.
    """

    def __init__(
        self,
        base_estimator: ParameterEstimator,
        allowed: Sequence[Any] | None = None,
        forbidden: Sequence[Any] | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.allowed = None if allowed is None else list(allowed)
        self.forbidden = None if forbidden is None else list(forbidden)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> TruncatedAccumulatorFactory:
        """Return a factory for truncated-data sufficient-statistic accumulators."""
        return TruncatedAccumulatorFactory(
            self.base_estimator.accumulator_factory(),
            allowed=self.allowed,
            forbidden=self.forbidden,
            keys=self.keys,
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> TruncatedDistribution:
        """Project retained evidence onto the base and report excluded evidence."""
        checked = _validate_statistics(suff_stat)
        if checked.accepted_weight <= 0.0:
            raise ValueError("truncated projection estimation requires positive accepted weight.")
        base = self.base_estimator.estimate(checked.accepted_weight, checked.child)
        total = checked.accepted_weight + checked.rejected_weight
        receipt = TruncatedProjectionFitReceipt(
            checked.accepted_weight,
            checked.rejected_weight,
            checked.rejected_weight / total if total else 0.0,
            False,
        )
        return TruncatedDistribution(
            base,
            allowed=self.allowed,
            forbidden=self.forbidden,
            name=self.name,
            keys=self.keys,
            fit_receipt=receipt,
        )


TruncatedEstimator = TruncatedProjectionEstimator


class TruncatedDataEncoder(MaskedBaseEncoder):
    """Encode observations via the base encoder, plus a boolean allowed-membership mask."""

    def __init__(self, dist: TruncatedDistribution) -> None:
        self.base_encoder = dist.base.dist_to_encoder()
        self._allowed_values = dist._allowed_values
        self._forbidden_values = dist._forbidden_values
        self._allowed_keys, self._forbidden_keys = _restriction_keys(
            self._allowed_values,
            self._forbidden_values,
        )

    @classmethod
    def from_base_encoder(
        cls,
        base_encoder: DataSequenceEncoder,
        *,
        allowed: Sequence[Any] | None,
        forbidden: Sequence[Any] | None,
    ) -> "TruncatedDataEncoder":
        encoder = cls.__new__(cls)
        encoder.base_encoder = base_encoder
        encoder._allowed_values = None if allowed is None else _dedupe(list(allowed))
        encoder._forbidden_values = None if forbidden is None else _dedupe(list(forbidden))
        encoder._allowed_keys, encoder._forbidden_keys = _restriction_keys(
            encoder._allowed_values,
            encoder._forbidden_values,
        )
        return encoder

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TruncatedDataEncoder)
            and self.base_encoder == other.base_encoder
            and self._allowed_keys == other._allowed_keys
            and self._forbidden_keys == other._forbidden_keys
        )

    def _extra_columns(self, x: Sequence[Any]) -> tuple[np.ndarray]:
        return (
            np.asarray(
                [
                    _is_allowed(
                        v,
                        self._allowed_keys,
                        self._forbidden_keys,
                        self._allowed_values,
                        self._forbidden_values,
                    )
                    for v in x
                ],
                dtype=bool,
            ),
        )
