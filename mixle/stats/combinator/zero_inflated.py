"""Zero-inflated count models: a point mass at 0 mixed with any base count distribution.

A zero-inflated distribution adds a separate source of zeros to a base count distribution: with
probability ``pi`` the observation is a structural zero, and with probability ``1 - pi`` it is drawn
from ``base``. So

    P(X = 0) = pi + (1 - pi) * P_base(0),
    P(X = k) = (1 - pi) * P_base(k)   for k > 0.

This is the standard remedy for *excess zeros* (ecology, insurance, defaults). Because the base can
itself produce zeros, an observed 0 is ambiguous -- it may be structural or a base zero -- so the
model is fit by EM (a two-component mixture whose inflation component is a fixed ``PointMass(0)``).
Wrapping *any* base gives the whole family at once: a Poisson base is the zero-inflated Poisson
(ZIP), a NegativeBinomial base is ZINB, a Binomial base is zero-inflated binomial, and so on.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.combinator._base import MaskedBaseEncoder
from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class ZeroInflatedDistribution(SequenceEncodableProbabilityDistribution):
    """A base count distribution with an extra point mass of probability ``pi`` at zero."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        pi: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a zero-inflated distribution.

        Args:
            base: The base count distribution. Its support must include 0 (e.g. Poisson,
                NegativeBinomial, Binomial, IntegerCategorical) so that 0 is shared between the
                inflation and the base; a base whose support excludes 0 (e.g. mixle's Geometric on
                {1, 2, ...}) is not a zero-inflation base -- compose a two-component MixtureDistribution
                of ``PointMass(0)`` and the base instead.
            pi: Structural-zero (inflation) probability, ``0 <= pi < 1``.
            name, keys: Optional instance name / parameter key (for the inflation probability).
        """
        if not (0.0 <= pi < 1.0):
            raise ValueError("zero-inflation probability pi must be in [0, 1).")
        self.base = base
        self.pi = float(pi)
        self.name = name
        self.keys = keys
        self._log_pi = math.log(self.pi) if self.pi > 0.0 else -np.inf
        self._log1mpi = math.log1p(-self.pi)

    def __str__(self) -> str:
        """Return a constructor-style representation of the zero-inflated distribution."""
        return "ZeroInflatedDistribution(%s, %s, name=%s, keys=%s)" % (
            str(self.base),
            repr(self.pi),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the zero-inflated probability at ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return ``log[(1-pi) p_base(x)]`` for ``x > 0``, mixed with ``pi`` at ``x == 0``."""
        lb = float(self.base.log_density(x))
        if x == 0:
            return float(np.logaddexp(self._log_pi, self._log1mpi + lb))
        return self._log1mpi + lb

    def seq_log_density(self, x: tuple[Any, np.ndarray]) -> np.ndarray:
        """Vectorized zero-inflated log-density for an encoded batch."""
        base_enc, zero_mask = x
        lb = np.asarray(self.base.seq_log_density(base_enc), dtype=np.float64)
        rv = self._log1mpi + lb
        if np.any(zero_mask):
            rv[zero_mask] = np.logaddexp(self._log_pi, self._log1mpi + lb[zero_mask])
        return rv

    def sampler(self, seed: int | None = None) -> "ZeroInflatedSampler":
        """Return a ZeroInflatedSampler for this distribution."""
        return ZeroInflatedSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ZeroInflatedEstimator":
        """Return an estimator that fits ``pi`` and the base by EM over the latent zero source."""
        return ZeroInflatedEstimator(
            self.base.estimator(pseudo_count=pseudo_count), pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "ZeroInflatedDataEncoder":
        """Return the data encoder (base encoding + a boolean is-zero mask)."""
        return ZeroInflatedDataEncoder(self.base.dist_to_encoder())


class ZeroInflatedSampler(DistributionSampler):
    """Draw a structural zero with probability ``pi``, otherwise draw from the base."""

    def __init__(self, dist: ZeroInflatedDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.base_sampler = dist.base.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None):
        """Draw one observation or a list of ``size`` observations."""
        if size is None:
            return 0 if self.rng.uniform() < self.dist.pi else self.base_sampler.sample()
        inflate = self.rng.uniform(size=int(size)) < self.dist.pi
        n_base = int((~inflate).sum())
        base_draws = list(self.base_sampler.sample(n_base)) if n_base else []
        out, bi = [], 0
        for inf in inflate:
            if inf:
                out.append(0)
            else:
                out.append(base_draws[bi])
                bi += 1
        return out


class ZeroInflatedAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the base sufficient statistics plus the expected structural-zero count."""

    def __init__(self, base_accumulator: SequenceEncodableStatisticAccumulator, keys: str | None = None) -> None:
        self.base_accumulator = base_accumulator
        self.inflation_count = 0.0  # sum of responsibilities that a zero is structural
        self.total = 0.0  # total weight
        self.keys = keys

    def _responsibility(self, estimate: ZeroInflatedDistribution) -> float:
        # P(structural | x == 0) = pi / (pi + (1-pi) P_base(0)); the same for every observed zero.
        p0 = float(np.exp(estimate.base.log_density(0)))
        denom = estimate.pi + (1.0 - estimate.pi) * p0
        return estimate.pi / denom if denom > 0.0 else 0.0

    def update(self, x: Any, weight: float, estimate: ZeroInflatedDistribution | None) -> None:
        """Accumulate one observation with EM responsibility for structural zeros."""
        if estimate is None or x != 0:
            self.base_accumulator.update(x, weight, None if estimate is None else estimate.base)
        else:
            r = self._responsibility(estimate)
            self.base_accumulator.update(x, weight * (1.0 - r), estimate.base)
            self.inflation_count += weight * r
        self.total += weight

    def seq_update(self, x: tuple[Any, np.ndarray], weights: np.ndarray, estimate: ZeroInflatedDistribution) -> None:
        """Accumulate encoded observations with zero-source responsibilities."""
        base_enc, zero_mask = x
        w = np.asarray(weights, dtype=np.float64)
        r = self._responsibility(estimate)
        base_w = w.copy()
        base_w[zero_mask] *= 1.0 - r  # a zero contributes (1-r) of its weight to the base
        self.base_accumulator.seq_update(base_enc, base_w, estimate.base)
        self.inflation_count += r * float(np.sum(w[zero_mask]))
        self.total += float(np.sum(w))

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the accumulator using a half-structural responsibility for zeros."""
        self.base_accumulator.initialize(x, weight, rng)
        if x == 0:
            self.inflation_count += 0.5 * weight  # heuristic: half the zeros start as structural
        self.total += weight

    def seq_initialize(self, x: tuple[Any, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize encoded observations using half-structural zero responsibilities."""
        base_enc, zero_mask = x
        w = np.asarray(weights, dtype=np.float64)
        self.base_accumulator.seq_initialize(base_enc, w, rng)
        self.inflation_count += 0.5 * float(np.sum(w[zero_mask]))
        self.total += float(np.sum(w))

    def combine(self, suff_stat: tuple[Any, float, float]) -> "ZeroInflatedAccumulator":
        """Merge serialized base statistics and inflation counts into this accumulator."""
        base_ss, ic, t = suff_stat
        self.base_accumulator.combine(base_ss)
        self.inflation_count += ic
        self.total += t
        return self

    def value(self) -> tuple[Any, float, float]:
        """Return base statistics, expected structural-zero weight, and total weight."""
        return self.base_accumulator.value(), self.inflation_count, self.total

    def from_value(self, x: tuple[Any, float, float]) -> "ZeroInflatedAccumulator":
        """Restore the accumulator from serialized zero-inflated statistics."""
        base_ss, ic, t = x
        self.base_accumulator.from_value(base_ss)
        self.inflation_count = float(ic)
        self.total = float(t)
        return self

    def scale(self, c: float) -> "ZeroInflatedAccumulator":
        """Scale base statistics and inflation counts by a constant."""
        self.base_accumulator.scale(c)
        self.inflation_count *= c
        self.total *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge base and inflation statistics into a keyed statistics dictionary."""
        self.base_accumulator.key_merge(stats_dict)
        if self.keys is not None:
            if self.keys in stats_dict:
                ic, t = stats_dict[self.keys]
                self.inflation_count += ic
                self.total += t
                # write the POOL back: without this, the dict keeps the FIRST site's stats and
                # key_replace hands every tied site that truncated pool -- later sites' data was
                # silently discarded (order-dependent wrong fits; found by the compiler review's
                # keyed-tying probe, present in 8 families vs the combine-into-dict families)
                stats_dict[self.keys] = (self.inflation_count, self.total)
            else:
                stats_dict[self.keys] = (self.inflation_count, self.total)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace base and inflation statistics from a keyed statistics dictionary."""
        self.base_accumulator.key_replace(stats_dict)
        if self.keys is not None and self.keys in stats_dict:
            self.inflation_count, self.total = stats_dict[self.keys]

    def acc_to_encoder(self) -> "ZeroInflatedDataEncoder":
        """Return an encoder that augments the base encoding with a zero mask."""
        return ZeroInflatedDataEncoder(self.base_accumulator.acc_to_encoder())


class ZeroInflatedAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`ZeroInflatedAccumulator`."""

    def __init__(self, base_factory: StatisticAccumulatorFactory, keys: str | None = None) -> None:
        self.base_factory = base_factory
        self.keys = keys

    def make(self) -> ZeroInflatedAccumulator:
        """Create an empty zero-inflated accumulator."""
        return ZeroInflatedAccumulator(self.base_factory.make(), keys=self.keys)


class ZeroInflatedEstimator(ParameterEstimator):
    """EM estimator: ``pi = (expected structural zeros) / N``, base re-fit on the down-weighted data."""

    def __init__(
        self,
        base_estimator: ParameterEstimator,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ZeroInflatedAccumulatorFactory:
        """Return a factory for zero-inflated sufficient-statistic accumulators."""
        return ZeroInflatedAccumulatorFactory(self.base_estimator.accumulator_factory(), keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[Any, float, float]) -> ZeroInflatedDistribution:
        """Estimate the base distribution and structural-zero probability."""
        base_ss, inflation_count, total = suff_stat
        base = self.base_estimator.estimate(nobs, base_ss)
        if self.pseudo_count is not None:
            pi = (inflation_count + 0.0) / (total + self.pseudo_count) if total + self.pseudo_count > 0 else 0.0
        else:
            pi = inflation_count / total if total > 0 else 0.0
        pi = min(max(pi, 0.0), 1.0 - 1.0e-12)
        return ZeroInflatedDistribution(base, pi, name=self.name, keys=self.keys)


class ZeroInflatedDataEncoder(MaskedBaseEncoder):
    """Encode observations via the base encoder, plus a boolean ``x == 0`` mask."""

    def _extra_columns(self, x: Sequence[Any]) -> tuple[np.ndarray]:
        return (np.asarray([v == 0 for v in x], dtype=bool),)
