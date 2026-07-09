"""Truncated-support combinator: restrict a base distribution to an allowed set, renormalized.

``TruncatedDistribution`` wraps a base distribution and conditions it on a support restriction:

    p(x) = p_base(x) / Z   for x in the allowed support,   else 0,

where ``Z = sum_{y allowed} p_base(y)`` is the retained mass.  The restriction is given either as a
finite ``forbidden`` set to exclude (``Z = 1 - sum_f p_base(f)`` -- works for an infinite base) or a
finite ``allowed`` set to keep (``Z = sum_a p_base(a)``).  It pairs with the Phase-1c support tools:
the renormalizer is exactly the truncated tail/total the enumeration bounds reason about.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.enumeration.algorithms import freeze
from mixle.stats.combinator._base import MaskedBaseEncoder, SingleChildAccumulator
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


class TruncatedDistribution(SequenceEncodableProbabilityDistribution):
    """A base distribution restricted to an allowed support and renormalized."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        allowed: Sequence[Any] | None = None,
        forbidden: Sequence[Any] | None = None,
        name: str | None = None,
        keys: str | None = None,
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
        self._allowed_values = None if allowed is None else list(allowed)
        self._forbidden_values = None if forbidden is None else list(forbidden)
        self._allowed_keys = None if allowed is None else {freeze(v) for v in allowed}
        self._forbidden_keys = None if forbidden is None else {freeze(v) for v in forbidden}
        # The retained log-mass ``log Z`` is formed in log space so it survives the tail-censoring
        # regime: an ``allowed`` set whose atoms are individually tiny is summed by ``logsumexp``, and
        # a ``forbidden`` set whose mass is ~1 uses a stable ``log(1 - p_forbidden)`` instead of the
        # catastrophically cancelling ``1 - sum(exp(...))``.
        if self._allowed_values is not None:
            log_probs = [float(base.log_density(v)) for v in self._allowed_values]
            finite = [lp for lp in log_probs if lp > -math.inf]
            log_z = float(logsumexp(finite)) if finite else -math.inf
        else:
            log_forbidden = [float(base.log_density(v)) for v in self._forbidden_values]
            finite = [lp for lp in log_forbidden if lp > -math.inf]
            log_p_forbidden = float(logsumexp(finite)) if finite else -math.inf
            log_z = log1mexp(log_p_forbidden)
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
        try:
            k = freeze(x)
        except TypeError:
            return False
        if self._allowed_keys is not None:
            return k in self._allowed_keys
        return k not in self._forbidden_keys

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

    def estimator(self, pseudo_count: float | None = None) -> "TruncatedEstimator":
        """Return an estimator that re-fits the base on the (in-support) data, keeping the truncation.

        This is the fixed-truncation estimator: it maximizes the base likelihood over the observed
        (already in-support) data and re-wraps with the same support restriction. It does not solve the
        full truncated MLE (whose normalizer Z depends on the base parameters); use it when the
        truncation set is fixed and known, which is the typical censored/restricted-support case.
        """
        return TruncatedEstimator(
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
            return len({freeze(v) for v in self._allowed_values})
        base_n = self.base.support_size()
        if base_n is None:
            return None
        forbidden_in = sum(1 for v in self._forbidden_values if self.base.log_density(v) > -np.inf)
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


class TruncatedAccumulator(SingleChildAccumulator):
    """Delegate accumulation to the base accumulator over in-support observations."""

    def __init__(self, base_accumulator: SequenceEncodableStatisticAccumulator, keys: str | None = None) -> None:
        self.base_accumulator = base_accumulator
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: TruncatedDistribution | None) -> None:
        """Accumulate one in-support observation through the base accumulator."""
        self.base_accumulator.update(x, weight, None if estimate is None else estimate.base)

    def seq_update(
        self, x: tuple[Any, np.ndarray], weights: np.ndarray, estimate: TruncatedDistribution | None
    ) -> None:
        """Accumulate encoded observations, zeroing weights outside the retained support."""
        base_enc, allowed_mask = x
        w = np.asarray(weights, dtype=np.float64) * allowed_mask.astype(np.float64)
        self.base_accumulator.seq_update(base_enc, w, None if estimate is None else estimate.base)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the base accumulator from one retained observation."""
        self.base_accumulator.initialize(x, weight, rng)

    def seq_initialize(self, x: tuple[Any, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize from encoded observations, zeroing weights outside the retained support."""
        base_enc, allowed_mask = x
        self.base_accumulator.seq_initialize(base_enc, np.asarray(weights, dtype=np.float64) * allowed_mask, rng)

    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return the base encoder used by the delegated accumulator."""
        return self.base_accumulator.acc_to_encoder()


class TruncatedAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`TruncatedAccumulator`."""

    def __init__(self, base_factory: StatisticAccumulatorFactory, keys: str | None = None) -> None:
        self.base_factory = base_factory
        self.keys = keys

    def make(self) -> TruncatedAccumulator:
        """Create an empty truncated-data accumulator."""
        return TruncatedAccumulator(self.base_factory.make(), keys=self.keys)


class TruncatedEstimator(ParameterEstimator):
    """Fixed-truncation estimator: fit the base on in-support data, re-wrap with the truncation."""

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
        return TruncatedAccumulatorFactory(self.base_estimator.accumulator_factory(), keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: Any) -> TruncatedDistribution:
        """Estimate the base distribution and re-apply the fixed truncation."""
        base = self.base_estimator.estimate(nobs, suff_stat)
        return TruncatedDistribution(
            base, allowed=self.allowed, forbidden=self.forbidden, name=self.name, keys=self.keys
        )


class TruncatedDataEncoder(MaskedBaseEncoder):
    """Encode observations via the base encoder, plus a boolean allowed-membership mask."""

    def __init__(self, dist: TruncatedDistribution) -> None:
        self.base_encoder = dist.base.dist_to_encoder()
        self._dist = dist

    def _extra_columns(self, x: Sequence[Any]) -> tuple[np.ndarray]:
        return (np.asarray([self._dist._allowed(v) for v in x], dtype=bool),)
