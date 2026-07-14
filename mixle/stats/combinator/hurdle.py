"""Hurdle count models: a two-part model that decouples *whether* a count is zero from *how big* it is.

A hurdle model splits the count into a binary "hurdle" and a zero-truncated count part:

    P(X = 0) = pi,
    P(X = k) = (1 - pi) * P_base(k) / (1 - P_base(0))   for k > 0.

With probability ``pi`` the observation fails to cross the hurdle and is zero; otherwise it is a
*positive* count drawn from ``base`` **conditioned on being > 0** (the base zero is truncated away and
the remaining mass renormalized). Contrast with :class:`~mixle.stats.combinator.zero_inflated.
ZeroInflatedDistribution`, where a zero can come from *either* the inflation *or* the base, so an
observed zero is latent-ambiguous and needs EM. In a hurdle model every zero is structural and every
positive is a (truncated) base draw -- there is no latent variable, so the two parts are estimated
**independently and in closed form**: ``pi`` is just the zero rate, and the base is fit to the
positive observations. Wrapping any count base gives the whole family: a Poisson base is the hurdle
Poisson, a NegativeBinomial base the hurdle NB, and so on.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.combinator._base import MaskedBaseEncoder
from mixle.stats.combinator.truncated import TruncatedDistribution
from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class HurdleDistribution(SequenceEncodableProbabilityDistribution):
    """A zero hurdle (probability ``pi``) followed by a zero-truncated base for the positive counts."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        pi: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a hurdle distribution.

        Args:
            base: The base count distribution for the positive part. Its support may include 0 (the
                zero is truncated and the mass renormalized); a base with no mass at 0 leaves the
                positive part unchanged (the renormalizer is 1).
            pi: Zero (hurdle) probability, ``0 <= pi < 1``.
            name, keys: Optional instance name / parameter key (for the hurdle probability).
        """
        if not (0.0 <= pi < 1.0):
            raise ValueError("hurdle probability pi must be in [0, 1).")
        self.base = base
        self.pi = float(pi)
        self.name = name
        self.keys = keys
        self._log_pi = math.log(self.pi) if self.pi > 0.0 else -np.inf
        self._log1mpi = math.log1p(-self.pi)
        p0 = float(np.exp(base.log_density(0)))  # base mass at 0 that the truncation removes
        self._log_renorm = math.log1p(-p0) if p0 > 0.0 else 0.0  # log(1 - P_base(0))

    def __str__(self) -> str:
        """Return a constructor-style representation of the hurdle distribution."""
        return "HurdleDistribution(%s, %s, name=%s, keys=%s)" % (
            str(self.base),
            repr(self.pi),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the hurdle probability at ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return ``log pi`` at ``x == 0``, else ``log[(1-pi) p_base(x) / (1 - p_base(0))]``."""
        if x == 0:
            return self._log_pi
        return self._log1mpi + float(self.base.log_density(x)) - self._log_renorm

    def seq_log_density(self, x: tuple[Any, np.ndarray]) -> np.ndarray:
        """Vectorized hurdle log-density for an encoded batch."""
        base_enc, zero_mask = x
        lb = np.asarray(self.base.seq_log_density(base_enc), dtype=np.float64)
        rv = self._log1mpi + lb - self._log_renorm
        if np.any(zero_mask):
            rv[zero_mask] = self._log_pi
        return rv

    def sampler(self, seed: int | None = None) -> "HurdleSampler":
        """Return a HurdleSampler for this distribution."""
        return HurdleSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HurdleEstimator":
        """Return an estimator that fits ``pi`` (the zero rate) and the base on the positives -- closed form."""
        return HurdleEstimator(
            self.base.estimator(pseudo_count=pseudo_count), pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "HurdleDataEncoder":
        """Return the data encoder (base encoding + a boolean is-zero mask)."""
        return HurdleDataEncoder(self.base.dist_to_encoder())


class HurdleSampler(DistributionSampler):
    """Draw a zero with probability ``pi``, otherwise draw from the zero-truncated base."""

    def __init__(self, dist: HurdleDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        # the positive part is exactly a zero-truncated base; reuse the (batched-rejection) truncated sampler
        self._positive = TruncatedDistribution(dist.base, forbidden=[0]).sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None):
        """Draw one observation or a list of ``size`` observations."""
        if size is None:
            return 0 if self.rng.uniform() < self.dist.pi else self._positive.sample()
        cross = self.rng.uniform(size=int(size)) >= self.dist.pi  # crossed the hurdle -> a positive count
        n_pos = int(cross.sum())
        pos_draws = self._positive.sample(n_pos) if n_pos else []
        out, pi = [], 0
        for c in cross:
            if c:
                out.append(pos_draws[pi])
                pi += 1
            else:
                out.append(0)
        return out


class HurdleAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the zero count (for ``pi``) and the base sufficient statistics over the positives only."""

    def __init__(self, base_accumulator: SequenceEncodableStatisticAccumulator, keys: str | None = None) -> None:
        self.base_accumulator = base_accumulator
        self.zero_count = 0.0  # weighted number of zeros (= hurdle failures)
        self.total = 0.0
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: HurdleDistribution | None) -> None:
        """Accumulate one observation, sending only nonzeros to the base."""
        if x == 0:
            self.zero_count += weight
        else:
            self.base_accumulator.update(x, weight, None if estimate is None else estimate.base)
        self.total += weight

    def seq_update(self, x: tuple[Any, np.ndarray], weights: np.ndarray, estimate: HurdleDistribution) -> None:
        """Accumulate encoded observations, masking zeros out of the base update."""
        base_enc, zero_mask = x
        w = np.asarray(weights, dtype=np.float64)
        base_w = w.copy()
        base_w[zero_mask] = 0.0  # zeros never inform the (zero-truncated) base
        self.base_accumulator.seq_update(base_enc, base_w, estimate.base)
        self.zero_count += float(np.sum(w[zero_mask]))
        self.total += float(np.sum(w))

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the accumulator with one weighted observation."""
        if x == 0:
            self.zero_count += weight
        else:
            self.base_accumulator.initialize(x, weight, rng)
        self.total += weight

    def seq_initialize(self, x: tuple[Any, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize encoded observations, excluding zeros from the base."""
        base_enc, zero_mask = x
        w = np.asarray(weights, dtype=np.float64)
        base_w = w.copy()
        base_w[zero_mask] = 0.0
        self.base_accumulator.seq_initialize(base_enc, base_w, rng)
        self.zero_count += float(np.sum(w[zero_mask]))
        self.total += float(np.sum(w))

    def combine(self, suff_stat: tuple[Any, float, float]) -> "HurdleAccumulator":
        """Merge serialized base statistics and zero counts into this accumulator."""
        base_ss, zc, t = suff_stat
        self.base_accumulator.combine(base_ss)
        self.zero_count += zc
        self.total += t
        return self

    def value(self) -> tuple[Any, float, float]:
        """Return base statistics, weighted zero count, and total weight."""
        return self.base_accumulator.value(), self.zero_count, self.total

    def from_value(self, x: tuple[Any, float, float]) -> "HurdleAccumulator":
        """Restore the accumulator from serialized hurdle statistics."""
        base_ss, zc, t = x
        self.base_accumulator.from_value(base_ss)
        self.zero_count = float(zc)
        self.total = float(t)
        return self

    def scale(self, c: float) -> "HurdleAccumulator":
        """Scale base statistics and hurdle counts by a constant."""
        self.base_accumulator.scale(c)
        self.zero_count *= c
        self.total *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge base and hurdle statistics into a keyed statistics dictionary."""
        self.base_accumulator.key_merge(stats_dict)
        if self.keys is not None:
            if self.keys in stats_dict:
                zc, t = stats_dict[self.keys]
                self.zero_count += zc
                self.total += t
                # write the POOL back: without this, the dict keeps the FIRST site's stats and
                # key_replace hands every tied site that truncated pool -- later sites' data was
                # silently discarded (order-dependent wrong fits; found by the compiler review's
                # keyed-tying probe, present in 8 families vs the combine-into-dict families)
                stats_dict[self.keys] = (self.zero_count, self.total)
            else:
                stats_dict[self.keys] = (self.zero_count, self.total)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace base and hurdle statistics from a keyed statistics dictionary."""
        self.base_accumulator.key_replace(stats_dict)
        if self.keys is not None and self.keys in stats_dict:
            self.zero_count, self.total = stats_dict[self.keys]

    def acc_to_encoder(self) -> "HurdleDataEncoder":
        """Return an encoder that augments the base encoding with a zero mask."""
        return HurdleDataEncoder(self.base_accumulator.acc_to_encoder())


class HurdleAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`HurdleAccumulator`."""

    def __init__(self, base_factory: StatisticAccumulatorFactory, keys: str | None = None) -> None:
        self.base_factory = base_factory
        self.keys = keys

    def make(self) -> HurdleAccumulator:
        """Create an empty hurdle accumulator."""
        return HurdleAccumulator(self.base_factory.make(), keys=self.keys)


class HurdleEstimator(ParameterEstimator):
    """Closed-form ``pi`` (the zero rate) plus the *zero-truncated MLE* of the base over the positives.

    The two parts are independent. ``pi`` is the observed zero rate. The count part is the base fit by
    maximum likelihood under the zero truncation -- NOT the base fit naively to the positives, which
    is biased (it recovers the truncated mean, so the fitted model would not match the data). The
    truncated MLE is obtained by a short EM that treats the removed zeros as missing data: given the
    current base, the positives imply ``N_missing = N_pos * P0/(1-P0)`` hypothetical base zeros; refit
    the base to the positives plus those pseudo-zeros and iterate. (If the base has no mass at 0 there
    is no truncation and the positives are fit directly.)
    """

    def __init__(
        self,
        base_estimator: ParameterEstimator,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        trunc_max_iter: int = 100,
        trunc_threshold: float = 1.0e-10,
    ) -> None:
        self.base_estimator = base_estimator
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys
        self.trunc_max_iter = int(trunc_max_iter)
        self.trunc_threshold = float(trunc_threshold)

    def accumulator_factory(self) -> HurdleAccumulatorFactory:
        """Return a factory for hurdle sufficient-statistic accumulators."""
        return HurdleAccumulatorFactory(self.base_estimator.accumulator_factory(), keys=self.keys)

    def _truncated_mle(self, n_pos: float, base_ss: Any) -> SequenceEncodableProbabilityDistribution:
        # EM for the zero-truncated MLE: re-impute the hypothetical missing zeros each iteration.
        base = self.base_estimator.estimate(n_pos, base_ss)
        if n_pos <= 0:
            return base
        prev_p0 = -1.0
        for _ in range(self.trunc_max_iter):
            p0 = float(np.exp(base.log_density(0)))
            if not (0.0 < p0 < 1.0) or abs(p0 - prev_p0) < self.trunc_threshold:
                break  # no zero mass (no truncation) or converged
            prev_p0 = p0
            n_missing = n_pos * p0 / (1.0 - p0)
            acc = self.base_estimator.accumulator_factory().make()
            acc.from_value(base_ss)  # the positives' sufficient statistics ...
            acc.update(0, n_missing, base)  # ... plus the imputed hypothetical zeros at 0
            base = self.base_estimator.estimate(n_pos + n_missing, acc.value())
        return base

    def estimate(self, nobs: float | None, suff_stat: tuple[Any, float, float]) -> HurdleDistribution:
        """Estimate the hurdle probability and zero-truncated base distribution."""
        base_ss, zero_count, total = suff_stat
        base = self._truncated_mle(total - zero_count, base_ss)
        if self.pseudo_count is not None:
            pi = zero_count / (total + self.pseudo_count) if total + self.pseudo_count > 0 else 0.0
        else:
            pi = zero_count / total if total > 0 else 0.0
        pi = min(max(pi, 0.0), 1.0 - 1.0e-12)
        return HurdleDistribution(base, pi, name=self.name, keys=self.keys)


class HurdleDataEncoder(MaskedBaseEncoder):
    """Encode observations via the base encoder, plus a boolean ``x == 0`` mask."""

    def _extra_columns(self, x: Sequence[Any]) -> tuple[np.ndarray]:
        return (np.asarray([v == 0 for v in x], dtype=bool),)
