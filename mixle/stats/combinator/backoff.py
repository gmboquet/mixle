"""Backoff: reserve a small share of mass for outcomes a fitted support cannot represent.

A distribution fitted on observed data can only score what it observed. A bounded family is the sharp
case -- ``IntegerCategoricalDistribution`` fitted on a training split gives ``-inf`` to any integer
outside (or inside a hole of) that split's support, and one such row drives a whole held-out mean
log-density to ``-inf``.

``BackoffDistribution`` is the composable answer: a two-component mixture of a ``base`` (the sharp
fitted model) and a ``fallback`` whose support covers the outcomes ``base`` cannot reach, with the
mixing weight *pinned small* rather than freely estimated::

    log p(x) = logaddexp( log(1 - w) + log p_base(x),  log w + log p_fallback(x) )

The pin is the whole point, and it is what separates this from ``MixtureEstimator`` over the same two
components. Free EM on a categorical-plus-count pair is model selection, not smoothing: on 400 Poisson
draws it settles near ``[0.46, 0.54]`` and the fallback swallows the detail the base was there to
capture. ``max_escape_weight`` bounds that, so the base keeps explaining the data it can explain and
the fallback only covers the tail.

Two things the caller owns:

* **The fallback sets the tail.** It must have support wherever ``base`` does not. A
  ``PoissonDistribution`` fallback scores a far-out integer at roughly ``-74`` because its tail decays
  factorially; ``NegativeBinomialDistribution`` or a broad bounded-uniform is usually the better
  choice. Whatever you pass is what the tail behaves like.
* **This needs a fallback family to exist.** That holds for integers. It does not hold for arbitrary
  string labels -- there is no natural distribution over unseen strings -- so
  ``CategoricalDistribution.default_value`` (and its integer-categorical counterpart) remains the
  right tool there, and this combinator is not a replacement for it.
"""

from __future__ import annotations

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

DEFAULT_ESCAPE_WEIGHT = 0.01
DEFAULT_MAX_ESCAPE_WEIGHT = 0.05


def _validated_weight(value: Any, *, name: str, upper: float = 1.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"backoff {name} must be a real number, not {type(value).__name__}.")
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0 or weight > upper:
        raise ValueError(f"backoff {name} must be finite and within [0, {upper:g}], not {weight!r}.")
    return weight


class BackoffDistribution(SequenceEncodableProbabilityDistribution):
    """A sharp ``base`` mixed with a broad ``fallback`` under a pinned escape weight."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        fallback: SequenceEncodableProbabilityDistribution,
        escape_weight: float = DEFAULT_ESCAPE_WEIGHT,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a backoff mixture.

        Args:
            base: The sharp model. Carries ``1 - escape_weight`` of the mass.
            fallback: Covers outcomes ``base`` cannot score. Carries ``escape_weight``.
            escape_weight: Mass reserved for the fallback, in ``[0, 1]``. At ``0.0`` this scores
                exactly as ``base`` alone, which makes the combinator inert rather than wrong.
            name: Optional distribution name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            base, fallback: The two components.
            escape_weight: Mass on ``fallback``.
            log_escape, log_retain: ``log(escape_weight)`` and ``log(1 - escape_weight)``.
        """
        for label, child in (("base", base), ("fallback", fallback)):
            if not hasattr(child, "log_density") or not hasattr(child, "seq_log_density"):
                raise TypeError(f"backoff {label} must be a distribution with log_density/seq_log_density.")
        self.base = base
        self.fallback = fallback
        self.escape_weight = _validated_weight(escape_weight, name="escape_weight")
        with np.errstate(divide="ignore"):
            self.log_escape = float(np.log(self.escape_weight))
            self.log_retain = float(np.log1p(-self.escape_weight))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the backoff mixture."""
        return "BackoffDistribution(%s, %s, escape_weight=%s, name=%s, keys=%s)" % (
            str(self.base),
            str(self.fallback),
            repr(self.escape_weight),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the mixture density at ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return the mixture log-density at ``x``.

        Finite whenever *either* component can score ``x``, which is the reason this class exists: the
        base may return ``-inf`` for an unobserved outcome and the total stays finite.
        """
        if self.escape_weight == 0.0:
            return float(self.base.log_density(x))
        if self.escape_weight == 1.0:
            return float(self.fallback.log_density(x))
        return float(
            np.logaddexp(
                self.log_retain + float(self.base.log_density(x)),
                self.log_escape + float(self.fallback.log_density(x)),
            )
        )

    def component_log_densities(self, x: tuple[Any, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return the two weighted component log-density vectors for encoded data ``x``."""
        base_enc, fallback_enc = x
        base = np.asarray(self.base.seq_log_density(base_enc), dtype=np.float64) + self.log_retain
        fallback = np.asarray(self.fallback.seq_log_density(fallback_enc), dtype=np.float64) + self.log_escape
        return base, fallback

    def seq_log_density(self, x: tuple[Any, Any]) -> np.ndarray:
        """Vectorized mixture log-density over sequence-encoded observations."""
        if self.escape_weight == 0.0:
            return np.asarray(self.base.seq_log_density(x[0]), dtype=np.float64)
        if self.escape_weight == 1.0:
            return np.asarray(self.fallback.seq_log_density(x[1]), dtype=np.float64)
        base, fallback = self.component_log_densities(x)
        return np.logaddexp(base, fallback)

    def sampler(self, seed: int | None = None) -> BackoffSampler:
        """Return a sampler that draws the component first, then the value."""
        return BackoffSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> BackoffEstimator:
        """Return an estimator for this backoff structure, reusing the children's own estimators."""
        return BackoffEstimator(
            self.base.estimator(pseudo_count=pseudo_count) if pseudo_count is not None else self.base.estimator(),
            self.fallback.estimator(pseudo_count=pseudo_count)
            if pseudo_count is not None
            else self.fallback.estimator(),
            escape_weight=self.escape_weight,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> BackoffDataEncoder:
        """Return an encoder that encodes each observation for both components."""
        return BackoffDataEncoder(self.base.dist_to_encoder(), self.fallback.dist_to_encoder())


class BackoffSampler(DistributionSampler):
    """Sampler for :class:`BackoffDistribution`."""

    def __init__(self, dist: BackoffDistribution, seed: int | None = None) -> None:
        """Seed the component choice and both child samplers from one stream."""
        self.dist = dist
        self.rng = RandomState(seed)
        self.base_sampler = dist.base.sampler(seed=self.rng.randint(0, 2**31 - 1))
        self.fallback_sampler = dist.fallback.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw ``size`` observations (or one when ``size`` is None)."""
        if size is None:
            escaped = self.rng.rand() < self.dist.escape_weight
            return self.fallback_sampler.sample() if escaped else self.base_sampler.sample()
        return [self.sample() for _ in range(int(size))]


class BackoffAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates child statistics weighted by EM responsibilities, plus the escape counts."""

    def __init__(
        self,
        base_accumulator: SequenceEncodableStatisticAccumulator,
        fallback_accumulator: SequenceEncodableStatisticAccumulator,
        keys: str | None = None,
    ) -> None:
        """Hold one accumulator per component and the running escape/total mass."""
        self.base_accumulator = base_accumulator
        self.fallback_accumulator = fallback_accumulator
        self.escape_count = 0.0
        self.total_count = 0.0
        self.keys = keys

    @staticmethod
    def _responsibilities(base_ll: np.ndarray, fallback_ll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the posterior component shares from already-weighted log densities.

        Both components are ``-inf`` only when neither can explain the observation; the responsibility
        is then split evenly rather than left as NaN, so a single unscorable row cannot poison the
        whole fit with silent NaNs.
        """
        total = np.logaddexp(base_ll, fallback_ll)
        dead = ~np.isfinite(total)
        with np.errstate(invalid="ignore"):
            base_share = np.where(dead, 0.5, np.exp(base_ll - total))
        return base_share, 1.0 - base_share

    def update(self, x: Any, weight: float, estimate: BackoffDistribution | None) -> None:
        """Accumulate one raw observation."""
        if estimate is None:
            self.base_accumulator.update(x, weight, None)
            self.fallback_accumulator.update(x, weight, None)
            self.total_count += weight
            return
        base_ll = estimate.log_retain + float(estimate.base.log_density(x))
        fallback_ll = estimate.log_escape + float(estimate.fallback.log_density(x))
        base_share, fallback_share = self._responsibilities(
            np.asarray([base_ll], dtype=np.float64), np.asarray([fallback_ll], dtype=np.float64)
        )
        self.base_accumulator.update(x, weight * float(base_share[0]), estimate.base)
        self.fallback_accumulator.update(x, weight * float(fallback_share[0]), estimate.fallback)
        self.escape_count += weight * float(fallback_share[0])
        self.total_count += weight

    def seq_update(self, x: tuple[Any, Any], weights: np.ndarray, estimate: BackoffDistribution) -> None:
        """Accumulate a sequence-encoded batch."""
        base_enc, fallback_enc = x
        ww = np.asarray(weights, dtype=np.float64)
        base_ll, fallback_ll = estimate.component_log_densities(x)
        base_share, fallback_share = self._responsibilities(base_ll, fallback_ll)
        self.base_accumulator.seq_update(base_enc, ww * base_share, estimate.base)
        self.fallback_accumulator.seq_update(fallback_enc, ww * fallback_share, estimate.fallback)
        self.escape_count += float(np.sum(ww * fallback_share))
        self.total_count += float(np.sum(ww))

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Cold start: give both components the full observation, and no escape evidence yet.

        The escape weight starts at the estimator's pin rather than being inferred from an
        initialization that has no responsibilities to work with.
        """
        self.base_accumulator.initialize(x, weight, rng)
        self.fallback_accumulator.initialize(x, weight, rng)
        self.total_count += weight

    def seq_initialize(self, x: tuple[Any, Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized :meth:`initialize`."""
        ww = np.asarray(weights, dtype=np.float64)
        self.base_accumulator.seq_initialize(x[0], ww, rng)
        self.fallback_accumulator.seq_initialize(x[1], ww, rng)
        self.total_count += float(np.sum(ww))

    def combine(self, suff_stat: tuple[Any, Any, float, float]) -> BackoffAccumulator:
        """Merge another accumulator's value into this one."""
        base_stat, fallback_stat, escape_count, total_count = suff_stat
        self.base_accumulator.combine(base_stat)
        self.fallback_accumulator.combine(fallback_stat)
        self.escape_count += escape_count
        self.total_count += total_count
        return self

    def value(self) -> tuple[Any, Any, float, float]:
        """Return the merged sufficient statistic."""
        return (
            self.base_accumulator.value(),
            self.fallback_accumulator.value(),
            self.escape_count,
            self.total_count,
        )

    def from_value(self, x: tuple[Any, Any, float, float]) -> BackoffAccumulator:
        """Replace this accumulator's state with ``x``."""
        base_stat, fallback_stat, escape_count, total_count = x
        self.base_accumulator.from_value(base_stat)
        self.fallback_accumulator.from_value(fallback_stat)
        self.escape_count = float(escape_count)
        self.total_count = float(total_count)
        return self

    def scale(self, c: float) -> BackoffAccumulator:
        """Scale every accumulated statistic by ``c``."""
        self.base_accumulator.scale(c)
        self.fallback_accumulator.scale(c)
        self.escape_count *= c
        self.total_count *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed statistics, delegating to both children."""
        if self.keys is not None:
            if self.keys in stats_dict:
                escape_count, total_count = stats_dict[self.keys]
                self.escape_count += escape_count
                self.total_count += total_count
            stats_dict[self.keys] = (self.escape_count, self.total_count)
        self.base_accumulator.key_merge(stats_dict)
        self.fallback_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Adopt keyed statistics, delegating to both children."""
        if self.keys is not None and self.keys in stats_dict:
            escape_count, total_count = stats_dict[self.keys]
            self.escape_count = escape_count
            self.total_count = total_count
        self.base_accumulator.key_replace(stats_dict)
        self.fallback_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> BackoffDataEncoder:
        """Return the encoder matching this accumulator's children."""
        return BackoffDataEncoder(self.base_accumulator.acc_to_encoder(), self.fallback_accumulator.acc_to_encoder())


class BackoffAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`BackoffAccumulator`."""

    def __init__(
        self,
        base_factory: StatisticAccumulatorFactory,
        fallback_factory: StatisticAccumulatorFactory,
        keys: str | None = None,
    ) -> None:
        """Hold the two child factories."""
        self.base_factory = base_factory
        self.fallback_factory = fallback_factory
        self.keys = keys

    def make(self) -> BackoffAccumulator:
        """Return a fresh backoff accumulator."""
        return BackoffAccumulator(self.base_factory.make(), self.fallback_factory.make(), self.keys)


class BackoffEstimator(ParameterEstimator):
    """Estimator for :class:`BackoffDistribution` with a bounded escape weight."""

    def __init__(
        self,
        base_estimator: ParameterEstimator,
        fallback_estimator: ParameterEstimator,
        escape_weight: float = DEFAULT_ESCAPE_WEIGHT,
        max_escape_weight: float | None = DEFAULT_MAX_ESCAPE_WEIGHT,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a backoff estimator.

        Args:
            base_estimator: Fits the sharp component.
            fallback_estimator: Fits the broad component.
            escape_weight: The reserved escape mass. This is a *floor*, not merely a starting point:
                EM may raise the weight up to ``max_escape_weight`` but never below this, because zero
                is an absorbing state. At ``w == 0`` the fallback's log weight is ``-inf``, so its
                responsibility is zero for every row, so ``escape_count`` stays zero and the weight can
                never recover -- EM collapses the escape branch on its first step and the fitted model
                scores every unseen outcome ``-inf`` again, silently undoing the whole point. Same
                shape as the Fisher floor in ``mixle.models.continual.ewc``: without a floor the
                optimizer satisfies its objective by discarding the capability.
            max_escape_weight: Ceiling on the estimated escape weight. ``None`` lets EM choose freely
                up to 1, which is model selection between the two components rather than smoothing --
                the behaviour the floor-and-ceiling pin exists to avoid. Set it equal to
                ``escape_weight`` to freeze the weight exactly.
            name: Optional name for the fitted distribution.
            keys: Optional merge key.
        """
        self.base_estimator = base_estimator
        self.fallback_estimator = fallback_estimator
        self.escape_weight = _validated_weight(escape_weight, name="escape_weight")
        if max_escape_weight is None:
            self.max_escape_weight: float | None = None
        else:
            self.max_escape_weight = _validated_weight(max_escape_weight, name="max_escape_weight")
            if self.escape_weight > self.max_escape_weight:
                raise ValueError(
                    f"backoff escape_weight {self.escape_weight!r} exceeds max_escape_weight "
                    f"{self.max_escape_weight!r}; the starting weight cannot sit outside its own bound."
                )
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BackoffAccumulatorFactory:
        """Return the matching accumulator factory."""
        return BackoffAccumulatorFactory(
            self.base_estimator.accumulator_factory(), self.fallback_estimator.accumulator_factory(), self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[Any, Any, float, float]) -> BackoffDistribution:
        """Fit both components and the (bounded) escape weight."""
        base_stat, fallback_stat, escape_count, total_count = suff_stat
        base = self.base_estimator.estimate(None, base_stat)
        fallback = self.fallback_estimator.estimate(None, fallback_stat)

        if self.max_escape_weight is not None and self.max_escape_weight == self.escape_weight:
            weight = self.escape_weight
        elif total_count <= 0.0:
            # No evidence to move the weight off its floor, and dividing by the count would be a
            # fabricated estimate rather than a measured one.
            weight = self.escape_weight
        else:
            weight = float(escape_count) / float(total_count)
            if not np.isfinite(weight):
                weight = self.escape_weight
            # The floor is what keeps EM off the absorbing zero -- see __init__. Measured above the
            # floor, the estimate is used as-is; measured below it (which includes the first step, where
            # initialize() has recorded no escape responsibility at all), the floor holds.
            weight = max(weight, self.escape_weight)
            if self.max_escape_weight is not None:
                weight = min(weight, self.max_escape_weight)
            weight = min(max(weight, 0.0), 1.0)
        return BackoffDistribution(base, fallback, escape_weight=weight, name=self.name, keys=self.keys)


class BackoffDataEncoder(DataSequenceEncoder):
    """Encodes each observation for both components."""

    def __init__(self, base_encoder: DataSequenceEncoder, fallback_encoder: DataSequenceEncoder) -> None:
        """Hold the two child encoders."""
        self.base_encoder = base_encoder
        self.fallback_encoder = fallback_encoder

    def __str__(self) -> str:
        """Return the encoder's display name."""
        return f"BackoffDataEncoder({self.base_encoder},{self.fallback_encoder})"

    def __eq__(self, other: object) -> bool:
        """Two backoff encoders match when both of their children do."""
        if not isinstance(other, BackoffDataEncoder):
            return False
        return self.base_encoder == other.base_encoder and self.fallback_encoder == other.fallback_encoder

    def seq_encode(self, x: Sequence[Any]) -> tuple[Any, Any]:
        """Encode ``x`` once per component.

        Both components see every observation -- unlike the zero-inflated/hurdle encoders, there is no
        mask here, because a backoff does not partition the data. Which component explains a row is a
        posterior quantity settled at fit time, not a property of the row.
        """
        return self.base_encoder.seq_encode(x), self.fallback_encoder.seq_encode(x)
