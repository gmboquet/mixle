"""Censoring combinator: a censored observation is known only to lie in an interval.

``CensoredDistribution`` wraps a base distribution and scores **censored** observations -- ones
that are not observed exactly but are known only to fall in some interval ``[a, b]``.  Such an
observation contributes its interval probability mass

    P(a <= X <= b) = F(b) - F(a)

to the likelihood, where ``F`` is the base distribution's CDF.  This is distinct from truncation:
truncation *renormalizes* the density over a restricted support (the observation is exact but the
support is limited), whereas censoring keeps the original distribution and only coarsens what was
observed.  Survival-analysis right/left/interval censoring (and Tobit-style bounds) are all this
combinator.

Data type: each observation is one of

    * a scalar ``x``                  -- an exact (uncensored) observation, scored by the base density;
    * a pair ``(a, b)``               -- interval-censored, scored by ``log(F(b) - F(a))``;
                                         use ``a = -inf`` for left censoring, ``b = +inf`` for right.

The base distribution must expose ``cdf``.  Estimation of the base parameters under censoring has no
generic closed form (the censored MLE couples the bounds with the base parameters), so the supplied
estimator fits the base on the **exact** observations only and re-wraps; document and prefer a
dedicated censored MLE when the censored fraction is large.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.combinator._base import MaskedBaseEncoder, SingleChildAccumulator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import logsubexp

# Tiny finite log-mass floor for an interval that underflowed to 0 in probability space when the base
# exposes only a linear ``cdf`` (~log of the smallest positive normal double); avoids silently zeroing.
_LOG_MASS_FLOOR = math.log(np.finfo(np.float64).tiny)


def _is_interval(x: Any) -> bool:
    """An observation is a censoring interval if it is a length-2 sequence of bounds."""
    return isinstance(x, (tuple, list)) and len(x) == 2


class CensoredDistribution(SequenceEncodableProbabilityDistribution):
    """A base distribution whose observations may be interval/left/right censored."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a censored wrapper around ``base``.

        Args:
            base: The base distribution. Must expose ``cdf`` for interval scoring.
            name, keys: Optional instance name / parameter key.
        """
        if not hasattr(base, "cdf"):
            raise ValueError("CensoredDistribution requires the base distribution to expose `cdf`.")
        self.base = base
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "CensoredDistribution(%s, name=%s, keys=%s)" % (str(self.base), repr(self.name), repr(self.keys))

    def _interval_log_mass(self, a: float, b: float) -> float:
        """Return ``log(F(b) - F(a))`` for a censoring interval ``[a, b]``, computed in log space.

        Tail censoring is the normal use case, so ``F(b) - F(a)`` routinely underflows to ``0`` in
        probability space (``log(0) = -inf``) even when the true interval log-mass is a perfectly
        finite large-negative number. When the base distribution exposes ``logcdf``/``logsf`` the mass
        is formed by a stable :func:`mixle.utils.special.logsubexp`; otherwise the linear ``F(b) - F(a)`` is used but the
        underflow is guarded so a real far-tail interval is not silently zeroed.
        """
        if b < a:
            a, b = b, a
        if a == b:
            # A degenerate (zero-width) interval has genuinely zero mass under a continuous base; this
            # is a true ``-inf``, not an underflow, so return it before any underflow floor kicks in.
            return -math.inf
        has_logsf = hasattr(self.base, "logsf")
        has_logcdf = hasattr(self.base, "logcdf")
        if has_logsf or has_logcdf:
            # log(F(b) - F(a)) = logsubexp(log F(b), log F(a)); in the upper tail prefer the survival
            # function: F(b) - F(a) = S(a) - S(b) = logsubexp(log S(a), log S(b)).
            if has_logsf:
                log_sa = 0.0 if a == -math.inf else float(self.base.logsf(a))
                log_sb = -math.inf if b == math.inf else float(self.base.logsf(b))
                return logsubexp(log_sa, log_sb)
            log_fa = -math.inf if a == -math.inf else float(self.base.logcdf(a))
            log_fb = 0.0 if b == math.inf else float(self.base.logcdf(b))
            return logsubexp(log_fb, log_fa)
        fa = 0.0 if a == -math.inf else float(self.base.cdf(a))
        fb = 1.0 if b == math.inf else float(self.base.cdf(b))
        mass = fb - fa
        if mass > 0.0:
            return math.log(mass)
        # The interval mass underflowed to 0 in probability space (both CDFs rounded to the same
        # value). Clamp to a tiny finite floor so a real far-tail interval is not silently zeroed
        # (-inf) -- the only way to do better is a base that exposes logcdf/logsf (handled above).
        return _LOG_MASS_FLOOR

    def density(self, x: Any) -> float:
        """Return the contribution of ``x`` (density for exact, interval mass for censored)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Log-likelihood contribution of ``x``.

        Exact observation ``x`` -> ``log p_base(x)``; interval ``(a, b)`` -> ``log(F(b) - F(a))``.
        """
        if _is_interval(x):
            return self._interval_log_mass(float(x[0]), float(x[1]))
        return float(self.base.log_density(x))

    def seq_log_density(self, x: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Per-row censored log-densities for an encoded batch."""
        exact_enc, exact_idx, lows, highs, cens_idx = x
        n = len(exact_idx) + len(cens_idx)
        rv = np.empty(n, dtype=np.float64)

        if len(exact_idx) > 0:
            rv[exact_idx] = np.asarray(self.base.seq_log_density(exact_enc), dtype=np.float64)

        for k in range(len(cens_idx)):
            rv[cens_idx[k]] = self._interval_log_mass(float(lows[k]), float(highs[k]))

        return rv

    def sampler(self, seed: int | None = None) -> "CensoredSampler":
        """Return a sampler that draws exact values from the base (sampling is uncensored)."""
        return CensoredSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "CensoredEstimator":
        """Return an estimator that fits the base on the exact observations, re-wrapping with censoring.

        The censored MLE has no generic closed form (the bounds couple with the base parameters), so
        this fits the base distribution on the **exact** (uncensored) observations and re-wraps. Use
        it when the censored fraction is modest; prefer a dedicated censored MLE otherwise.
        """
        return CensoredEstimator(self.base.estimator(pseudo_count=pseudo_count), name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "CensoredDataEncoder":
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
        return self.base_sampler.sample(size=size)


class CensoredAccumulator(SingleChildAccumulator):
    """Accumulate base sufficient statistics over the exact (uncensored) observations only."""

    def __init__(self, base_accumulator: SequenceEncodableStatisticAccumulator, keys: str | None = None) -> None:
        self.base_accumulator = base_accumulator
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: CensoredDistribution | None) -> None:
        """Accumulate one exact observation and ignore interval-censored observations."""
        if not _is_interval(x):
            self.base_accumulator.update(x, weight, None if estimate is None else estimate.base)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize from one exact observation and ignore censoring intervals."""
        if not _is_interval(x):
            self.base_accumulator.initialize(x, weight, rng)

    def seq_update(
        self,
        x: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: CensoredDistribution | None,
    ) -> None:
        """Accumulate exact rows from an encoded censored batch."""
        exact_enc, exact_idx, _lows, _highs, _cens_idx = x
        if len(exact_idx) > 0:
            w = np.asarray(weights, dtype=np.float64)[exact_idx]
            self.base_accumulator.seq_update(exact_enc, w, None if estimate is None else estimate.base)

    def seq_initialize(
        self,
        x: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        rng: RandomState | None,
    ) -> None:
        """Initialize from exact rows in an encoded censored batch."""
        exact_enc, exact_idx, _lows, _highs, _cens_idx = x
        if len(exact_idx) > 0:
            w = np.asarray(weights, dtype=np.float64)[exact_idx]
            self.base_accumulator.seq_initialize(exact_enc, w, rng)

    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return an encoder that separates exact observations from intervals."""
        return CensoredDataEncoder.from_base_encoder(self.base_accumulator.acc_to_encoder())


class CensoredAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`CensoredAccumulator`."""

    def __init__(self, base_factory: StatisticAccumulatorFactory, keys: str | None = None) -> None:
        self.base_factory = base_factory
        self.keys = keys

    def make(self) -> CensoredAccumulator:
        """Create an empty censored-data accumulator."""
        return CensoredAccumulator(self.base_factory.make(), keys=self.keys)


class CensoredEstimator(ParameterEstimator):
    """Fit the base on the exact observations, re-wrap with censoring."""

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
        """Estimate the base distribution from exact-observation statistics."""
        base = self.base_estimator.estimate(nobs, suff_stat)
        return CensoredDistribution(base, name=self.name, keys=self.keys)


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
    def from_base_encoder(cls, base_encoder: DataSequenceEncoder) -> "CensoredDataEncoder":
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
                lows.append(float(v[0]))
                highs.append(float(v[1]))
                cens_idx.append(i)
            else:
                exact_vals.append(v)
                exact_idx.append(i)
        exact_enc = self.base_encoder.seq_encode(exact_vals) if exact_vals else self.base_encoder.seq_encode([])
        return (
            exact_enc,
            np.asarray(exact_idx, dtype=np.int64),
            np.asarray(lows, dtype=np.float64),
            np.asarray(highs, dtype=np.float64),
            np.asarray(cens_idx, dtype=np.int64),
        )
