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
import operator
from collections.abc import Sequence
from typing import Any, NamedTuple

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

_BIRTH, _DEATH, _SAMPLING = 0, 1, 2


class BirthDeathSamplingStatistics(NamedTuple):
    """Versioned, tuple-compatible aggregate birth-death statistics."""

    births: float
    deaths: float
    samplings: float
    population_exposure: float
    trajectory_weight: float
    horizon_weight: float

    @property
    def schema_version(self) -> int:
        """Return the serialized statistic schema version."""
        return 1


def _exact_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an exact integer") from exc


def _nonnegative_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _initial_population(value: Any, *, label: str) -> int:
    result = _exact_integer(value, label=label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _validated_rows(value: Any) -> np.ndarray:
    try:
        rows = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("BirthDeathSampling encoded rows must be numeric") from exc
    if rows.shape == (0,):
        rows = rows.reshape((0, 6))
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(
            "BirthDeathSampling encoded data must have exact shape (N, 6)"
        )
    if np.any(~np.isfinite(rows)) or np.any(rows < 0.0):
        raise ValueError(
            "BirthDeathSampling encoded statistics must be finite and non-negative"
        )
    counts = rows[:, :3]
    if np.any(counts != np.floor(counts)):
        raise ValueError(
            "BirthDeathSampling per-trajectory event counts must be exact integers"
        )
    if np.any((counts.sum(axis=1) > 0.0) & (rows[:, 3] == 0.0)):
        raise ValueError(
            "BirthDeathSampling positive event counts require population exposure"
        )
    return rows.copy()


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("BirthDeathSampling weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(
            f"BirthDeathSampling weights must have exact shape ({rows},)"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(
            "BirthDeathSampling weights must be finite and non-negative"
        )
    return weights


def _validated_statistics(value: Any) -> BirthDeathSamplingStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError(
            "BirthDeathSampling sufficient statistics must contain six fields"
        )
    fields = tuple(
        _nonnegative_scalar(
            item,
            label=f"BirthDeathSampling statistic field {index}",
        )
        for index, item in enumerate(value)
    )
    births, deaths, samplings, integral, count, horizon_sum = fields
    if count == 0.0 and any(
        item != 0.0
        for item in (births, deaths, samplings, integral, horizon_sum)
    ):
        raise ValueError(
            "Zero BirthDeathSampling trajectory weight requires zero statistics"
        )
    if births + deaths + samplings > 0.0 and integral == 0.0:
        raise ValueError(
            "BirthDeathSampling positive events require population exposure"
        )
    return BirthDeathSamplingStatistics(*fields)


def _prior_vector(
    value: Any,
    *,
    label: str,
    minimum: float,
) -> np.ndarray:
    if np.ndim(value) == 0:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"BirthDeathSampling {label} must be real-valued")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"BirthDeathSampling {label} must be real-valued"
            ) from exc
        result = np.full(3, scalar, dtype=np.float64)
    else:
        try:
            result = np.asarray(value, dtype=np.float64).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BirthDeathSampling {label} must be numeric"
            ) from exc
        if result.shape != (3,):
            raise ValueError(
                f"BirthDeathSampling {label} must have shape (3,)"
            )
    if np.any(~np.isfinite(result)) or np.any(result < minimum):
        raise ValueError(
            f"BirthDeathSampling {label} must be finite and at least {minimum}"
        )
    result.setflags(write=False)
    return result


def _trajectory_row(traj: Any) -> np.ndarray:
    """Validate and encode one ``(n0, T, events)`` trajectory."""
    try:
        fields = tuple(traj)
    except TypeError as exc:
        raise TypeError(
            "BirthDeathSampling trajectory must be an iterable record"
        ) from exc
    if len(fields) != 3:
        raise ValueError(
            "BirthDeathSampling trajectory must be (n0, horizon, events)"
        )
    n = _initial_population(
        fields[0],
        label="BirthDeathSampling trajectory initial population",
    )
    horizon = _nonnegative_scalar(
        fields[1],
        label="BirthDeathSampling trajectory horizon",
    )
    try:
        events = tuple(fields[2])
    except TypeError as exc:
        raise TypeError(
            "BirthDeathSampling trajectory events must be iterable"
        ) from exc
    t_prev = 0.0
    integral = 0.0
    sum_log_n = 0.0
    counts = [0.0, 0.0, 0.0]
    for index, event in enumerate(events):
        try:
            event_fields = tuple(event)
        except TypeError as exc:
            raise TypeError(
                f"BirthDeathSampling event {index} must be a two-field record"
            ) from exc
        if len(event_fields) != 2:
            raise ValueError(
                f"BirthDeathSampling event {index} must contain time and type"
            )
        time = _nonnegative_scalar(
            event_fields[0],
            label=f"BirthDeathSampling event {index} time",
        )
        etype = _exact_integer(
            event_fields[1],
            label=f"BirthDeathSampling event {index} type",
        )
        if etype not in (_BIRTH, _DEATH, _SAMPLING):
            raise ValueError(
                "BirthDeathSampling event type must be 0 (birth), "
                "1 (death), or 2 (sampling)"
            )
        if time <= t_prev or time > horizon:
            raise ValueError(
                "BirthDeathSampling events must have strictly increasing "
                "times in (0, T]"
            )
        if n <= 0:
            raise ValueError("BirthDeathSampling event occurred at zero population.")
        integral += n * (time - t_prev)
        sum_log_n += math.log(n)
        counts[etype] += 1.0
        if etype == _BIRTH:
            n += 1
        elif etype == _DEATH:
            n -= 1
        t_prev = time
    integral += n * (horizon - t_prev)
    return np.asarray(
        (
            counts[_BIRTH],
            counts[_DEATH],
            counts[_SAMPLING],
            integral,
            sum_log_n,
            horizon,
        ),
        dtype=np.float64,
    )


def _trajectory_stats(traj: Any) -> tuple[float, float, float, float, float]:
    """Return the five likelihood statistics for one validated trajectory."""
    row = _trajectory_row(traj)
    return tuple(float(value) for value in row[:5])


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
        prior_shape: Any = 1.0,
        prior_rate: Any = 0.0,
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
        self.birth_rate = _nonnegative_scalar(
            birth_rate,
            label="BirthDeathSampling birth rate",
        )
        self.death_rate = _nonnegative_scalar(
            death_rate,
            label="BirthDeathSampling death rate",
        )
        self.sampling_rate = _nonnegative_scalar(
            sampling_rate,
            label="BirthDeathSampling sampling rate",
        )
        self.initial_population = _initial_population(
            initial_population,
            label="BirthDeathSampling initial population",
        )
        self.horizon = _nonnegative_scalar(
            horizon,
            label="BirthDeathSampling sampling horizon",
        )
        self.name = name
        self.keys = keys
        self.prior_shape = _prior_vector(
            prior_shape,
            label="Gamma prior shape",
            minimum=1.0,
        )
        self.prior_rate = _prior_vector(
            prior_rate,
            label="Gamma prior rate",
            minimum=0.0,
        )
        with np.errstate(divide="ignore"):
            self._log_rates = np.log(np.array([self.birth_rate, self.death_rate, self.sampling_rate], dtype=np.float64))
        self._log_rates.setflags(write=False)
        self._total_rate = self.birth_rate + self.death_rate + self.sampling_rate

    def __str__(self) -> str:
        """Return a constructor-style representation of the birth-death sampling distribution."""
        return "BirthDeathSamplingDistribution(%s, %s, %s, initial_population=%s, horizon=%s, name=%s, keys=%s, prior_shape=%s, prior_rate=%s)" % (
            repr(self.birth_rate),
            repr(self.death_rate),
            repr(self.sampling_rate),
            repr(self.initial_population),
            repr(self.horizon),
            repr(self.name),
            repr(self.keys),
            repr(self.prior_shape.tolist()),
            repr(self.prior_rate.tolist()),
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
        """Vectorized log-likelihood for validated ``(N, 6)`` encoded rows."""
        return self._stats_log_density(_validated_rows(x))

    def sampler(self, seed: int | None = None) -> "BirthDeathSamplingSampler":
        """Return a BirthDeathSamplingSampler for this distribution."""
        return BirthDeathSamplingSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BirthDeathSamplingEstimator":
        """Return a fixed-configuration closed-form MLE or Gamma MAP estimator."""
        if pseudo_count is None:
            return BirthDeathSamplingEstimator(
                name=self.name,
                keys=self.keys,
                initial_population=self.initial_population,
                horizon=self.horizon,
                prior_shape=self.prior_shape,
                prior_rate=self.prior_rate,
            )
        return BirthDeathSamplingEstimator(
            name=self.name,
            keys=self.keys,
            pseudo_count=pseudo_count,
            initial_population=self.initial_population,
            horizon=self.horizon,
        )

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
        checked_size = _exact_integer(
            size,
            label="BirthDeathSampling sample size",
        )
        if checked_size < 0:
            raise ValueError(
                "BirthDeathSampling sample size must be non-negative"
            )
        return [self._sample_one() for _ in range(checked_size)]


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
        row = _trajectory_row(traj)
        checked_weight = _nonnegative_scalar(
            weight,
            label="BirthDeathSampling weight",
        )
        self.births += checked_weight * row[0]
        self.deaths += checked_weight * row[1]
        self.samplings += checked_weight * row[2]
        self.integral += checked_weight * row[3]
        self.horizon_sum += checked_weight * row[5]
        self.count += checked_weight

    def update(self, x: Any, weight: float, estimate: BirthDeathSamplingDistribution | None) -> None:
        """Accumulate weighted event counts and exposure for one trajectory."""
        self._add(x, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted trajectory."""
        self._add(x, weight)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted event counts and exposure from encoded trajectories."""
        rows = _validated_rows(x)
        ww = _validated_weights(weights, rows.shape[0])
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
        checked = _validated_statistics(suff_stat)
        self.births += checked.births
        self.deaths += checked.deaths
        self.samplings += checked.samplings
        self.integral += checked.population_exposure
        self.count += checked.trajectory_weight
        self.horizon_sum += checked.horizon_weight
        return self

    def value(self) -> tuple[float, float, float, float, float, float]:
        """Return event counts, exposure, trajectory count, and horizon total."""
        return BirthDeathSamplingStatistics(
            self.births,
            self.deaths,
            self.samplings,
            self.integral,
            self.count,
            self.horizon_sum,
        )

    def from_value(self, x: tuple[float, float, float, float, float, float]) -> "BirthDeathSamplingAccumulator":
        """Restore the accumulator from serialized birth-death-sampling statistics."""
        checked = _validated_statistics(x)
        (
            self.births,
            self.deaths,
            self.samplings,
            self.integral,
            self.count,
            self.horizon_sum,
        ) = checked
        return self

    def scale(self, c: float) -> "BirthDeathSamplingAccumulator":
        """Scale accumulated sufficient statistics by a constant."""
        checked_scale = _nonnegative_scalar(
            c,
            label="BirthDeathSampling scale",
        )
        self.births *= checked_scale
        self.deaths *= checked_scale
        self.samplings *= checked_scale
        self.integral *= checked_scale
        self.count *= checked_scale
        self.horizon_sum *= checked_scale
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into its configured shared statistic."""
        if self.keys is None:
            return
        if self.keys in stats_dict:
            stats_dict[self.keys].combine(self.value())
        else:
            stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator with its configured shared statistic."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

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
    """Closed-form per-channel MLE or Gamma-prior MAP estimator."""

    def __init__(
        self,
        name: str | None = None,
        keys: str | None = None,
        pseudo_count: float | None = None,
        initial_population: int = 1,
        horizon: float = 10.0,
        prior_shape: Any | None = None,
        prior_rate: Any | None = None,
    ) -> None:
        if pseudo_count is not None and (
            prior_shape is not None or prior_rate is not None
        ):
            raise ValueError(
                "BirthDeathSampling pseudo_count cannot be combined with "
                "explicit Gamma priors"
            )
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _nonnegative_scalar(
                pseudo_count,
                label="BirthDeathSampling pseudo-count",
            )
        )
        if self.pseudo_count is not None:
            shape_value: Any = 1.0 + self.pseudo_count
            rate_value: Any = self.pseudo_count
        else:
            shape_value = 1.0 if prior_shape is None else prior_shape
            rate_value = 0.0 if prior_rate is None else prior_rate
        self.prior_shape = _prior_vector(
            shape_value,
            label="Gamma prior shape",
            minimum=1.0,
        )
        self.prior_rate = _prior_vector(
            rate_value,
            label="Gamma prior rate",
            minimum=0.0,
        )
        self.initial_population = _initial_population(
            initial_population,
            label="BirthDeathSampling estimator initial population",
        )
        self.horizon = _nonnegative_scalar(
            horizon,
            label="BirthDeathSampling estimator sampling horizon",
        )
        self.name = name
        self.keys = keys

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Serialize the fixed configuration and effective explicit prior."""
        return {
            "name": self.name,
            "keys": self.keys,
            "initial_population": self.initial_population,
            "horizon": self.horizon,
            "prior_shape": self.prior_shape,
            "prior_rate": self.prior_rate,
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Validate and restore fixed estimator configuration."""
        required = {
            "name",
            "keys",
            "initial_population",
            "horizon",
            "prior_shape",
            "prior_rate",
        }
        if set(state) != required:
            raise ValueError(
                "invalid BirthDeathSampling estimator state fields: "
                "expected %r, got %r" % (sorted(required), sorted(state))
            )
        self.__init__(
            name=state["name"],
            keys=state["keys"],
            initial_population=state["initial_population"],
            horizon=state["horizon"],
            prior_shape=state["prior_shape"],
            prior_rate=state["prior_rate"],
        )

    def accumulator_factory(self) -> "BirthDeathSamplingAccumulatorFactory":
        """Return a factory for birth-death-sampling sufficient-statistic accumulators."""
        return BirthDeathSamplingAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float, float]
    ) -> "BirthDeathSamplingDistribution":
        """Estimate per-capita rates while preserving fixed sampling configuration."""
        checked = _validated_statistics(suff_stat)
        if checked.trajectory_weight <= 0.0:
            raise ValueError(
                "Cannot estimate BirthDeathSampling without positive trajectory weight"
            )
        if nobs is not None:
            _nonnegative_scalar(
                nobs,
                label="BirthDeathSampling observation count",
            )
        counts = np.asarray(
            (checked.births, checked.deaths, checked.samplings),
            dtype=np.float64,
        )
        numerator = counts + self.prior_shape - 1.0
        denominator = checked.population_exposure + self.prior_rate
        if np.any((numerator > 0.0) & (denominator == 0.0)):
            raise ValueError(
                "BirthDeathSampling positive event evidence or prior shape "
                "requires population exposure or prior rate"
            )
        rates = np.zeros(3, dtype=np.float64)
        np.divide(
            numerator,
            denominator,
            out=rates,
            where=denominator > 0.0,
        )
        return BirthDeathSamplingDistribution(
            rates[0],
            rates[1],
            rates[2],
            initial_population=self.initial_population,
            horizon=self.horizon,
            name=self.name,
            keys=self.keys,
            prior_shape=self.prior_shape,
            prior_rate=self.prior_rate,
        )


class BirthDeathSamplingDataEncoder(DataSequenceEncoder):
    """Encode trajectories ``(n0, T, events)`` into an ``(N, 6)`` array of sufficient statistics + T."""

    def __str__(self) -> str:
        return "BirthDeathSamplingDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BirthDeathSamplingDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Encode trajectories as sufficient-statistic rows."""
        rows = [_trajectory_row(traj) for traj in x]
        return (
            np.asarray(rows, dtype=np.float64)
            if rows
            else np.zeros((0, 6), dtype=np.float64)
        )

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of validated encoded trajectory rows."""
        return int(_validated_rows(x).shape[0])
