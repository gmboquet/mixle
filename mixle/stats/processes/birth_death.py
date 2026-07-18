"""Evaluate, estimate, and sample from a general birth-death-sampling process.

Defines BirthDeathSamplingDistribution, BirthDeathSamplingSampler,
BirthDeathSamplingAccumulatorFactory, BirthDeathSamplingAccumulator,
BirthDeathSamplingEstimator, and BirthDeathSamplingDataEncoder.

A continuous-time linear birth-death process on a population ``n(t)``: each individual independently
gives birth at rate ``birth_rate``, dies at rate ``death_rate``, and is *sampled* (observed through
time, without removal) at rate ``sampling_rate``. This is a general population/epidemic model; the
**fossilized birth-death** model is the special case where ``sampling_rate`` is the fossilization
rate. Pure birth-death is ``sampling_rate = 0``.

Data type: one fully-observed trajectory ``(n0, T, events)`` -- initial count ``n0``, observation
window length ``T``, and a time-ordered list of ``(time, type)`` events with ``type`` in
``{0: birth, 1: death, 2: sampling}``. The log-likelihood is

    sum_events log n_i  +  n_b log(birth) + n_d log(death) + n_s log(sampling)  -  (birth+death+sampling) * I,

where ``n_i`` is the population just before event ``i`` and ``I = integral_0^T n(t) dt`` (``n`` is
piecewise constant between events). The MLE is closed-form: each rate is its event count divided by
``I`` (summed over trajectories).


Reference: Feller, *An Introduction to Probability Theory and Its Applications*, Vol. 1 (Wiley).
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

_MIN_RATE = 1.0e-12
_BIRTH, _DEATH, _SAMPLING = 0, 1, 2


def _trajectory_stats(traj: Any) -> tuple[float, float, float, float, float]:
    """Replay a trajectory ``(n0, T, events)`` -> (n_births, n_deaths, n_samplings, integral_n, sum_log_n)."""
    n0, horizon, events = traj
    n = int(n0)
    if n < 0:
        raise ValueError("BirthDeathSampling trajectory requires a non-negative initial population.")
    t_prev = 0.0
    integral = 0.0
    sum_log_n = 0.0
    counts = [0.0, 0.0, 0.0]
    for time, etype in events:
        time = float(time)
        etype = int(etype)
        if time < t_prev or time > float(horizon):
            raise ValueError("BirthDeathSampling events must be time-ordered within [0, T].")
        if n <= 0:
            raise ValueError("BirthDeathSampling event occurred at zero population.")
        integral += n * (time - t_prev)
        sum_log_n += math.log(n)
        counts[etype] += 1.0
        if etype == _BIRTH:
            n += 1
        elif etype == _DEATH:
            n -= 1
        elif etype != _SAMPLING:
            raise ValueError("BirthDeathSampling event type must be 0 (birth), 1 (death), or 2 (sampling).")
        t_prev = time
    integral += n * (float(horizon) - t_prev)
    return counts[_BIRTH], counts[_DEATH], counts[_SAMPLING], integral, sum_log_n


class BirthDeathSamplingDistribution(SequenceEncodableProbabilityDistribution):
    """General linear birth-death-sampling process (fossilized birth-death is the ``sampling_rate>0`` case)."""

    def __init__(
        self,
        birth_rate: float,
        death_rate: float,
        sampling_rate: float = 0.0,
        initial_population: int = 1,
        horizon: float = 10.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a birth-death-sampling process with per-capita rates.

        Args:
            birth_rate (float): Per-capita birth rate ``>= 0``.
            death_rate (float): Per-capita death rate ``>= 0``.
            sampling_rate (float): Per-capita sampling/fossilization rate ``>= 0`` (no removal).
            initial_population (int): Initial count used when sampling trajectories.
            horizon (float): Observation window ``[0, horizon]`` used when sampling.
            name, keys: optional object name / parameter key.
        """
        for label, value in (("birth_rate", birth_rate), ("death_rate", death_rate), ("sampling_rate", sampling_rate)):
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"BirthDeathSamplingDistribution requires finite {label} >= 0.")
        self.birth_rate = float(birth_rate)
        self.death_rate = float(death_rate)
        self.sampling_rate = float(sampling_rate)
        self.initial_population = int(initial_population)
        self.horizon = float(horizon)
        self.name = name
        self.keys = keys
        with np.errstate(divide="ignore"):
            self._log_rates = np.log(np.array([self.birth_rate, self.death_rate, self.sampling_rate], dtype=np.float64))
        self._total_rate = self.birth_rate + self.death_rate + self.sampling_rate

    def __str__(self) -> str:
        """Return a constructor-style representation of the birth-death sampling distribution."""
        return "BirthDeathSamplingDistribution(%s, %s, %s, initial_population=%s, horizon=%s, name=%s, keys=%s)" % (
            repr(self.birth_rate),
            repr(self.death_rate),
            repr(self.sampling_rate),
            repr(self.initial_population),
            repr(self.horizon),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Probability density of one trajectory (see ``log_density``)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Log-likelihood of one fully-observed trajectory ``(n0, T, events)``."""
        nb, nd, ns, integral, sum_log_n = _trajectory_stats(x)
        return self._stats_log_density(np.array([[nb, nd, ns, integral, sum_log_n]], dtype=np.float64))[0]

    def _stats_log_density(self, rows: np.ndarray) -> np.ndarray:
        counts = rows[:, :3]
        integral = rows[:, 3]
        sum_log_n = rows[:, 4]
        # n_type * log(rate) per channel, with 0 contribution where both count and rate are 0
        # (errstate silences the discarded 0 * -inf when a channel rate is 0 and its count is 0).
        with np.errstate(invalid="ignore"):
            emitted = np.where(counts > 0.0, counts * self._log_rates[None, :], 0.0)
        rv = sum_log_n + np.sum(emitted, axis=1) - self._total_rate * integral
        rv[np.any(~np.isfinite(emitted), axis=1)] = -np.inf  # an event of a zero-rate channel
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-likelihood for an ``(N, 5)`` array of per-trajectory sufficient statistics."""
        return self._stats_log_density(np.asarray(x, dtype=np.float64))

    def sampler(self, seed: int | None = None) -> "BirthDeathSamplingSampler":
        """Return a BirthDeathSamplingSampler for this distribution."""
        return BirthDeathSamplingSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BirthDeathSamplingEstimator":
        """Return a BirthDeathSamplingEstimator (closed-form rate MLE)."""
        return BirthDeathSamplingEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "BirthDeathSamplingDataEncoder":
        """Return the encoder for birth-death samples."""
        return BirthDeathSamplingDataEncoder()


class BirthDeathSamplingSampler(DistributionSampler):
    """Exact Gillespie simulation of birth-death-sampling trajectories on ``[0, horizon]``."""

    def __init__(self, dist: BirthDeathSamplingDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> tuple[int, float, list[tuple[float, int]]]:
        d = self.dist
        n = d.initial_population
        t = 0.0
        events: list[tuple[float, int]] = []
        per_capita = d.birth_rate + d.death_rate + d.sampling_rate
        while n > 0 and per_capita > 0.0:
            total = n * per_capita
            t += self.rng.exponential(1.0 / total)
            if t >= d.horizon:
                break
            u = self.rng.uniform() * per_capita
            if u < d.birth_rate:
                events.append((t, _BIRTH))
                n += 1
            elif u < d.birth_rate + d.death_rate:
                events.append((t, _DEATH))
                n -= 1
            else:
                events.append((t, _SAMPLING))
        return d.initial_population, d.horizon, events

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one trajectory ``(n0, T, events)`` or a list of ``size`` trajectories."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(int(size))]


class BirthDeathSamplingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted event counts, the time-integral of the population, and trajectory count."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.births = 0.0
        self.deaths = 0.0
        self.samplings = 0.0
        self.integral = 0.0
        self.count = 0.0
        self.horizon_sum = 0.0
        self.name = name
        self.keys = keys

    def _add(self, traj: Any, weight: float) -> None:
        nb, nd, ns, integral, _ = _trajectory_stats(traj)
        self.births += weight * nb
        self.deaths += weight * nd
        self.samplings += weight * ns
        self.integral += weight * integral
        self.horizon_sum += weight * float(traj[1])
        self.count += weight

    def update(self, x: Any, weight: float, estimate: BirthDeathSamplingDistribution | None) -> None:
        """Accumulate weighted event counts and exposure for one trajectory."""
        self._add(x, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted trajectory."""
        self._add(x, weight)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted event counts and exposure from encoded trajectories."""
        rows = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        self.births += float(np.dot(rows[:, 0], ww))
        self.deaths += float(np.dot(rows[:, 1], ww))
        self.samplings += float(np.dot(rows[:, 2], ww))
        self.integral += float(np.dot(rows[:, 3], ww))
        self.horizon_sum += float(np.dot(rows[:, 5], ww)) if rows.shape[1] > 5 else 0.0
        self.count += float(ww.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded trajectories."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float, float, float, float]) -> "BirthDeathSamplingAccumulator":
        """Merge serialized birth-death-sampling statistics into this accumulator."""
        self.births += suff_stat[0]
        self.deaths += suff_stat[1]
        self.samplings += suff_stat[2]
        self.integral += suff_stat[3]
        self.count += suff_stat[4]
        self.horizon_sum += suff_stat[5]
        return self

    def value(self) -> tuple[float, float, float, float, float, float]:
        """Return event counts, exposure, trajectory count, and horizon total."""
        return self.births, self.deaths, self.samplings, self.integral, self.count, self.horizon_sum

    def from_value(self, x: tuple[float, float, float, float, float, float]) -> "BirthDeathSamplingAccumulator":
        """Restore the accumulator from serialized birth-death-sampling statistics."""
        self.births, self.deaths, self.samplings, self.integral, self.count, self.horizon_sum = x
        return self

    def scale(self, c: float) -> "BirthDeathSamplingAccumulator":
        """Scale accumulated sufficient statistics by a constant."""
        self.births *= c
        self.deaths *= c
        self.samplings *= c
        self.integral *= c
        self.count *= c
        self.horizon_sum *= c
        return self

    def acc_to_encoder(self) -> "BirthDeathSamplingDataEncoder":
        """Return an encoder for trajectory sufficient statistics."""
        return BirthDeathSamplingDataEncoder()


class BirthDeathSamplingAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BirthDeathSamplingAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> "BirthDeathSamplingAccumulator":
        """Create an empty birth-death-sampling accumulator."""
        return BirthDeathSamplingAccumulator(name=self.name, keys=self.keys)


class BirthDeathSamplingEstimator(ParameterEstimator):
    """Closed-form rate MLE: ``rate = (total events of that type) / integral_n``."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "BirthDeathSamplingAccumulatorFactory":
        """Return a factory for birth-death-sampling sufficient-statistic accumulators."""
        return BirthDeathSamplingAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float, float]
    ) -> "BirthDeathSamplingDistribution":
        """Estimate per-capita rates from accumulated event counts and population time-integral."""
        births, deaths, samplings, integral, count, horizon_sum = suff_stat
        denom = max(integral, _MIN_RATE)
        horizon = horizon_sum / count if count > 0.0 else 10.0
        return BirthDeathSamplingDistribution(
            births / denom,
            deaths / denom,
            samplings / denom,
            initial_population=1,
            horizon=horizon,
            name=self.name,
            keys=self.keys,
        )


class BirthDeathSamplingDataEncoder(DataSequenceEncoder):
    """Encode trajectories ``(n0, T, events)`` into an ``(N, 6)`` array of sufficient statistics + T."""

    def __str__(self) -> str:
        return "BirthDeathSamplingDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BirthDeathSamplingDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Encode trajectories as sufficient-statistic rows."""
        rows = []
        for traj in x:
            nb, nd, ns, integral, sum_log_n = _trajectory_stats(traj)
            rows.append((nb, nd, ns, integral, sum_log_n, float(traj[1])))
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 6), dtype=np.float64)
