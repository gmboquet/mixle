"""Survival model with right-censoring: a bundled event-time + censoring-indicator likelihood.

Survival/reliability data is ``(t, event)``: either the event was observed at time ``t`` (``event=1``)
or the unit was still alive when observation stopped, a *right-censored* time (``event=0``, the true
event is only known to be ``> t``). The likelihood bundles the density for observed events with the
survival function for censored ones:

    log L = sum_{event}  log f(t)  +  sum_{censored}  log S(t),    S(t) = 1 - F(t).

``SurvivalDistribution`` wraps any event-time ``base`` that exposes ``cdf`` and ``quantile`` (Weibull,
Exponential, GeneralizedPareto, GeneralizedExtremeValue, ...). Censoring is exogenous, so the only
parameters are the base's; fitting them under censoring is the standard EM that *imputes the unobserved
event times*: a censored time ``c`` contributes the base's mass beyond ``c``, approximated
deterministically by a grid of conditional quantiles ``F^{-1}(F(c) + q (1 - F(c)))``. Each mixle EM
iteration re-imputes with the improved base, converging to the right-censored MLE -- which, unlike
fitting on the observed events alone, is unbiased.
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
    scale_suff_stat,
)
from mixle.utils.special import log1mexp

# Tiny finite log-survival floor when the base exposes only a linear ``cdf`` and ``1 - F(t)`` underflows
# to 0 in the right tail (~log of the smallest positive normal double); avoids a -inf log-likelihood.
_LOG_SURVIVAL_FLOOR = math.log(np.finfo(np.float64).tiny)


def _log_survival(base: Any, t: float) -> float:
    """Return ``log S(t) = log(1 - F(t))`` for a right-censored time, computed stably.

    Right censoring is the normal use case, so ``1 - F(t)`` routinely underflows to ``0`` in the
    right tail (``log1p(-1) = -inf``) even when the true log-survival is finite and large-negative.
    Routes through the base's ``logsf`` when available; otherwise forms ``1 - F(t)`` via the shared
    :func:`mixle.utils.special.log1mexp` and guards a fully-saturated ``F(t) = 1`` with a tiny finite
    floor rather than silently zeroing the survival contribution.
    """
    if hasattr(base, "logsf"):
        return float(base.logsf(t))
    cdf = float(base.cdf(t))
    if cdf < 1.0:
        # log(1 - F(t)) = log(1 - exp(log F(t))); log1mexp(-inf) correctly yields 0 when F(t) = 0.
        return log1mexp(math.log(cdf)) if cdf > 0.0 else 0.0
    return _LOG_SURVIVAL_FLOOR


class SurvivalDistribution(SequenceEncodableProbabilityDistribution):
    """Right-censored survival likelihood over a base event-time distribution."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if not (hasattr(base, "cdf") and hasattr(base, "quantile")):
            raise ValueError("SurvivalDistribution base must expose cdf() and quantile() (the survival function).")
        self.base = base
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "SurvivalDistribution(%s, name=%s, keys=%s)" % (str(self.base), repr(self.name), repr(self.keys))

    def density(self, x: tuple[float, int]) -> float:
        """Return the likelihood contribution of ``(t, event)``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[float, int]) -> float:
        """Return ``log f(t)`` for an observed event, else ``log S(t)`` for a right-censored time."""
        t, event = x
        if event:
            return float(self.base.log_density(t))
        return _log_survival(self.base, float(t))

    def seq_log_density(self, x: tuple[Any, np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized likelihood: base density for events, log survival for censored rows."""
        base_enc, times, event_mask = x
        rv = np.asarray(self.base.seq_log_density(base_enc), dtype=np.float64)
        cens = ~event_mask
        if np.any(cens):
            surv = np.array([_log_survival(self.base, float(t)) for t in times[cens]])
            rv[cens] = surv
        return rv

    def sampler(self, seed: int | None = None) -> "SurvivalSampler":
        """Return a sampler (draws uncensored event times; censoring is exogenous)."""
        return SurvivalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SurvivalEstimator":
        """Return an estimator that fits the base under right-censoring by conditional-quantile EM."""
        return SurvivalEstimator(self.base.estimator(pseudo_count=pseudo_count), name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "SurvivalDataEncoder":
        """Return the data encoder (base encoding of the times + the event mask)."""
        return SurvivalDataEncoder(self.base.dist_to_encoder())


class SurvivalSampler(DistributionSampler):
    """Draw uncensored event times ``(t, 1)`` from the base (censoring is applied externally)."""

    def __init__(self, dist: SurvivalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.base_sampler = dist.base.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one uncensored event-time pair or a list of pairs."""
        if size is None:
            return (self.base_sampler.sample(), 1)
        return [(t, 1) for t in self.base_sampler.sample(size=int(size))]


class SurvivalAccumulator(SingleChildAccumulator):
    """Feed observed events and conditional-quantile imputations of censored times to the base accumulator."""

    def __init__(
        self,
        base_accumulator: SequenceEncodableStatisticAccumulator,
        base_encoder: DataSequenceEncoder,
        n_impute: int = 16,
        keys: str | None = None,
    ) -> None:
        self.base_accumulator = base_accumulator
        self.base_encoder = base_encoder
        self.n_impute = int(n_impute)
        self.keys = keys
        # midpoint-rule conditional-quantile grid q in (0,1): integrates the suff stats over the tail
        self._qgrid = (np.arange(self.n_impute) + 0.5) / self.n_impute

    def _impute(self, times: np.ndarray, weights: np.ndarray, base: Any) -> tuple[list[float], list[float]]:
        # each censored (c, w) -> n_impute points at conditional quantiles F^{-1}(F(c)+q(1-F(c))), weight w/K
        aug_t: list[float] = []
        aug_w: list[float] = []
        for c, w in zip(times, weights):
            fc = base.cdf(float(c))
            for q in self._qgrid:
                aug_t.append(float(base.quantile(fc + q * (1.0 - fc))))
                aug_w.append(w / self.n_impute)
        return aug_t, aug_w

    def _accumulate(self, x, weights, estimate, initialize: bool, rng) -> None:
        base_enc, times, event_mask = x
        w = np.asarray(weights, dtype=np.float64)
        ev = event_mask
        # observed events: straight to the base accumulator
        aug_t = list(times[ev])
        aug_w = list(w[ev])
        cens_t, cens_w = times[~ev], w[~ev]
        if len(cens_t):
            if estimate is not None:
                it, iw = self._impute(cens_t, cens_w, estimate.base)  # impute from the current base
            else:
                it, iw = list(cens_t), list(cens_w)  # no model yet: seed the base with the censored times as-is
            aug_t.extend(it)
            aug_w.extend(iw)
        if not aug_t:
            return
        enc = self.base_encoder.seq_encode(aug_t)
        aug_w_arr = np.asarray(aug_w, dtype=np.float64)
        base_est = None if estimate is None else estimate.base
        if initialize:
            self.base_accumulator.seq_initialize(enc, aug_w_arr, rng)
        else:
            self.base_accumulator.seq_update(enc, aug_w_arr, base_est)

    def update(self, x: tuple[float, int], weight: float, estimate: SurvivalDistribution | None) -> None:
        """Accumulate one observed or right-censored event-time record."""
        t, event = x
        enc = (None, np.array([t], dtype=np.float64), np.array([bool(event)]))
        self._accumulate(enc, np.array([weight]), estimate, initialize=False, rng=None)

    def initialize(self, x: tuple[float, int], weight: float, rng: RandomState | None) -> None:
        """Initialize the base sufficient statistics from one survival record."""
        t, event = x
        enc = (None, np.array([t], dtype=np.float64), np.array([bool(event)]))
        self._accumulate(enc, np.array([weight]), None, initialize=True, rng=rng)

    def seq_update(self, x, weights: np.ndarray, estimate: SurvivalDistribution) -> None:
        """Accumulate encoded survival records using conditional-tail imputation."""
        self._accumulate(x, weights, estimate, initialize=False, rng=None)

    def seq_initialize(self, x, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize from encoded survival records without a current estimate."""
        self._accumulate(x, weights, None, initialize=True, rng=rng)

    def scale(self, c: float) -> "SurvivalAccumulator":
        """Scale the delegated base sufficient statistics by a constant."""
        # Structural default over the bare child value (this accumulator carries no scalable
        # statistics of its own; n_impute / _qgrid are configuration, not sufficient statistics).
        self.from_value(scale_suff_stat(self.value(), c))
        return self

    def acc_to_encoder(self) -> "SurvivalDataEncoder":
        """Return an encoder that records event times and censoring indicators."""
        return SurvivalDataEncoder(self.base_accumulator.acc_to_encoder())


class SurvivalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`SurvivalAccumulator`."""

    def __init__(
        self,
        base_factory: StatisticAccumulatorFactory,
        base_encoder: DataSequenceEncoder,
        n_impute: int,
        keys: str | None,
    ) -> None:
        self.base_factory = base_factory
        self.base_encoder = base_encoder
        self.n_impute = n_impute
        self.keys = keys

    def make(self) -> SurvivalAccumulator:
        """Create an empty survival accumulator."""
        return SurvivalAccumulator(self.base_factory.make(), self.base_encoder, self.n_impute, keys=self.keys)


class SurvivalEstimator(ParameterEstimator):
    """Fit the base event-time distribution under right-censoring (conditional-quantile imputation EM)."""

    def __init__(
        self,
        base_estimator: ParameterEstimator,
        n_impute: int = 16,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_impute = int(n_impute)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> SurvivalAccumulatorFactory:
        """Return a factory for right-censored survival sufficient-statistic accumulators."""
        # the base encoder is needed to encode the imputed times; get it from a throwaway base estimate
        base_enc = self.base_estimator.estimate(None, self.base_estimator.accumulator_factory().make().value())
        return SurvivalAccumulatorFactory(
            self.base_estimator.accumulator_factory(), base_enc.dist_to_encoder(), self.n_impute, self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> SurvivalDistribution:
        """Estimate the base event-time distribution from imputed survival statistics."""
        return SurvivalDistribution(self.base_estimator.estimate(nobs, suff_stat), name=self.name, keys=self.keys)


class SurvivalDataEncoder(MaskedBaseEncoder):
    """Encode ``(t, event)`` data as the base encoding of the times plus the boolean event mask."""

    def seq_encode(self, x: Sequence[tuple[float, int]]) -> tuple[Any, np.ndarray, np.ndarray]:
        """Encode survival records as base times plus a boolean event mask."""
        times = np.asarray([t for t, _ in x], dtype=np.float64)
        event_mask = np.asarray([bool(e) for _, e in x], dtype=bool)
        return self.base_encoder.seq_encode(list(times)), times, event_mask
