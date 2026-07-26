"""Continuous-time Markov chain (CTMC) over fully observed trajectories.

A CTMC on ``K`` states is governed by a generator matrix ``Q``: off-diagonal ``q_ij >= 0`` is the rate
of jumping ``i -> j``, and ``q_ii = -sum_{j!=i} q_ij``. A fully-observed trajectory is the initial state,
the total observation horizon, and the sequence of ``(dwell_time, next_state)`` jumps up to that
horizon; its log-likelihood is

    log L = sum_{i!=j} n_ij * log q_ij  -  sum_i q_i * T_i,      q_i = -q_ii = sum_{j!=i} q_ij,

where ``n_ij`` is the number of observed ``i->j`` transitions and ``T_i`` the total time spent in ``i``
*including the final, right-censored dwell* -- from the last jump (or from time 0, if there were no
jumps at all) out to the observation horizon. That final interval is real exposure with no observed
exit, exactly like the tail of a birth-death or renewal-process trajectory censored at its own window
(see ``mixle.stats.processes.birth_death``): a chain that dwells in a state for the whole horizon without
jumping still contributes ``-q_i * horizon`` to the log-likelihood, it just contributes no transition
count. Omitting it (as an earlier version of this module did) understates every ``T_i`` by the length of
each trajectory's final interval and biases the closed-form rate MLE upward.

This is a collection of independent Poisson-rate likelihoods, so the MLE is closed form and unique:
``q_ij = n_ij / T_i``. The estimator therefore certifies ``GLOBAL_UNIQUE`` (see
:func:`mixle.inference.certify`, which classifies this family). Data type: ``(s0, horizon, [(dt, s1),
(dt, s2), ...])`` -- the initial state, the total observation window length, and the observed jumps
(each ``dt > 0``, and the jumps' dwell times may not sum to more than ``horizon``). The horizon travels
with each trajectory rather than living only on the distribution, matching ``BirthDeathSamplingDistribution``'s
``(n0, T, events)`` convention: different trajectories in one dataset are free to have been observed for
different lengths of time. The distribution's own ``initial_state``/``horizon`` constructor parameters
are sampling-time defaults (where ``sample()`` starts a fresh chain, and for how long) and are round-
tripped through ``.estimator()``; they are not constraints on what a trajectory being *scored* is allowed
to contain -- ``log_density`` scores whatever ``s0`` and ``horizon`` the trajectory itself declares.

The family follows the standard Mixle distribution contract (Distribution / Sampler / Accumulator / Factory /
Estimator / DataEncoder) so it composes with ``optimize`` / ``seq_log_density`` / the PPL surface like
every other family.
"""

from __future__ import annotations

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


class ContinuousTimeMarkovChainStatistics(NamedTuple):
    """Versioned, tuple-compatible aggregate CTMC statistics."""

    transition_counts: np.ndarray
    dwell_times: np.ndarray

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


def _positive_integer(value: Any, *, label: str) -> int:
    result = _exact_integer(value, label=label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


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


def _state(value: Any, *, num_states: int, label: str) -> int:
    result = _exact_integer(value, label=label)
    if not 0 <= result < num_states:
        raise ValueError(
            f"{label} {result} is out of range for {num_states} states"
        )
    return result


def _validated_statistics(
    value: Any,
    *,
    num_states: int,
) -> ContinuousTimeMarkovChainStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "CTMC sufficient statistics must be (transition_counts, dwell_times)"
        )
    try:
        counts = np.asarray(value[0], dtype=np.float64)
        dwell = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("CTMC sufficient statistics must be numeric") from exc
    if counts.shape != (num_states, num_states):
        raise ValueError(
            f"CTMC transition counts must have shape ({num_states}, {num_states})"
        )
    if dwell.shape != (num_states,):
        raise ValueError(f"CTMC dwell times must have shape ({num_states},)")
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts < 0.0)
        or np.any(~np.isfinite(dwell))
        or np.any(dwell < 0.0)
    ):
        raise ValueError(
            "CTMC counts and dwell times must be finite and non-negative"
        )
    if np.any(np.diag(counts) != 0.0):
        raise ValueError("CTMC transition-count diagonal must be zero")
    outgoing = counts.sum(axis=1)
    if np.any((outgoing > 0.0) & (dwell == 0.0)):
        raise ValueError(
            "CTMC positive transition counts require positive source-state dwell"
        )
    checked_counts = counts.copy()
    np.fill_diagonal(checked_counts, 0.0)
    return ContinuousTimeMarkovChainStatistics(
        checked_counts,
        dwell.copy(),
    )


def _validated_encoded_rows(
    value: Any,
    *,
    num_states: int,
) -> tuple[ContinuousTimeMarkovChainStatistics, ...]:
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise TypeError("CTMC encoded data must be iterable") from exc
    return tuple(
        _validated_statistics(row, num_states=num_states) for row in rows
    )


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("CTMC weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"CTMC weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("CTMC weights must be finite and non-negative")
    return weights


def _prior_matrix(
    value: Any,
    *,
    num_states: int,
    label: str,
    minimum: float,
    diagonal: float,
) -> np.ndarray:
    if np.ndim(value) == 0:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"CTMC {label} must be real-valued")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"CTMC {label} must be real-valued") from exc
        matrix = np.full((num_states, num_states), scalar, dtype=np.float64)
        np.fill_diagonal(matrix, diagonal)
    else:
        try:
            matrix = np.asarray(value, dtype=np.float64).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CTMC {label} must be numeric") from exc
        if matrix.shape != (num_states, num_states):
            raise ValueError(
                f"CTMC {label} must have shape ({num_states}, {num_states})"
            )
    if np.any(~np.isfinite(matrix)) or np.any(matrix < minimum):
        raise ValueError(
            f"CTMC {label} must be finite and at least {minimum}"
        )
    if not np.allclose(np.diag(matrix), diagonal, rtol=0.0, atol=0.0):
        raise ValueError(f"CTMC {label} diagonal must equal {diagonal}")
    matrix.setflags(write=False)
    return matrix


def _trajectory_stats(traj: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_ij transition-count matrix, T_i dwell-time vector) for one ``(s0, horizon, [(dt, s1), ...])``
    trajectory, including the final right-censored dwell from the last jump (or from time 0, if there
    are no jumps) out to ``horizon``. Raises ``ValueError`` on a malformed trajectory: wrong shape, a
    non-finite/negative horizon, a non-finite/negative dwell time, dwell times summing past the horizon,
    or a state index outside ``[0, k)``.
    """
    try:
        trajectory = tuple(traj)
    except TypeError as exc:
        raise TypeError(
            "CTMC trajectory must be an iterable three-field record"
        ) from exc
    if len(trajectory) != 3:
        raise ValueError(
            "CTMC trajectory must be (initial_state, horizon, jumps); got a "
            f"{len(trajectory)}-element sequence. The 2-element (s0, jumps) format is no longer accepted: it "
            "has no field for the trajectory's observation horizon, so the final censored dwell time "
            "(the exposure between the last jump and when observation stopped) cannot be recovered."
        )
    s0, horizon, jumps = trajectory
    horizon = _nonnegative_scalar(
        horizon,
        label="CTMC trajectory horizon",
    )
    cur = _state(
        s0,
        num_states=k,
        label="CTMC trajectory initial state",
    )
    try:
        checked_jumps = tuple(jumps)
    except TypeError as exc:
        raise TypeError("CTMC trajectory jumps must be iterable") from exc

    counts = np.zeros((k, k), dtype=np.float64)
    dwell = np.zeros(k, dtype=np.float64)
    t = 0.0
    for index, jump in enumerate(checked_jumps):
        try:
            fields = tuple(jump)
        except TypeError as exc:
            raise TypeError(
                f"CTMC jump {index} must be a two-field record"
            ) from exc
        if len(fields) != 2:
            raise ValueError(
                f"CTMC jump {index} must contain dwell time and next state"
            )
        dt = _nonnegative_scalar(
            fields[0],
            label=f"CTMC jump {index} dwell time",
        )
        if dt <= 0.0:
            raise ValueError("CTMC jump dwell times must be strictly positive")
        t += dt
        if t > horizon:
            raise ValueError(f"CTMC trajectory dwell times sum to {t!r}, which exceeds its horizon {horizon!r}.")
        s_next = _state(
            fields[1],
            num_states=k,
            label=f"CTMC jump {index} target",
        )
        if s_next == cur:
            raise ValueError("CTMC jumps must change state")
        dwell[cur] += dt
        counts[cur, s_next] += 1.0
        cur = s_next
    dwell[cur] += horizon - t  # final censored dwell: last jump (or t=0) out to the observation horizon
    return counts, dwell


class ContinuousTimeMarkovChainDistribution(SequenceEncodableProbabilityDistribution):
    """CTMC on ``K`` states with generator ``Q`` (off-diagonal rates); MLE is closed form (GLOBAL_UNIQUE)."""

    def __init__(
        self,
        rates: np.ndarray,
        initial_state: int = 0,
        horizon: float = 10.0,
        name: str | None = None,
        keys: str | None = None,
        prior_shape: Any = 1.0,
        prior_rate: Any = 0.0,
    ) -> None:
        rates = np.asarray(rates, dtype=np.float64)
        if (
            rates.ndim != 2
            or rates.shape[0] != rates.shape[1]
            or rates.shape[0] == 0
        ):
            raise ValueError("rates must be a square (K, K) matrix of off-diagonal jump rates")
        if np.any(rates < 0.0) or not np.all(np.isfinite(rates)):
            raise ValueError("CTMC rates must be finite and >= 0")
        if np.any(np.diag(rates) != 0.0):
            raise ValueError("CTMC rate-matrix diagonal must be zero")
        self.rates = rates.copy()
        self.num_states = rates.shape[0]
        self.initial_state = _state(
            initial_state,
            num_states=self.num_states,
            label="CTMC initial state",
        )
        self.horizon = _nonnegative_scalar(
            horizon,
            label="CTMC sampling horizon",
        )
        self.name = name
        self.keys = keys
        self.prior_shape = _prior_matrix(
            prior_shape,
            num_states=self.num_states,
            label="Gamma prior shape",
            minimum=1.0,
            diagonal=1.0,
        )
        self.prior_rate = _prior_matrix(
            prior_rate,
            num_states=self.num_states,
            label="Gamma prior rate",
            minimum=0.0,
            diagonal=0.0,
        )
        self._exit = self.rates.sum(axis=1)  # q_i = total exit rate from state i
        with np.errstate(divide="ignore"):
            self._log_rates = np.log(self.rates)
        self.rates.setflags(write=False)
        self._exit.setflags(write=False)
        self._log_rates.setflags(write=False)

    def __str__(self) -> str:
        return (
            f"ContinuousTimeMarkovChainDistribution({self.rates.tolist()!r}, "
            f"initial_state={self.initial_state}, horizon={self.horizon}, "
            f"name={self.name!r}, keys={self.keys!r}, "
            f"prior_shape={self.prior_shape.tolist()!r}, "
            f"prior_rate={self.prior_rate.tolist()!r})"
        )

    @property
    def generator(self) -> np.ndarray:
        """The generator matrix ``Q`` (off-diagonal rates, diagonal = -exit rate)."""
        q = self.rates.copy()
        np.fill_diagonal(q, -self._exit)
        return q

    def density(self, x: Any) -> float:
        """Return the probability density of one fully observed trajectory."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the log-likelihood of one fully observed CTMC trajectory."""
        counts, dwell = _trajectory_stats(x, self.num_states)
        return self._stats_log_density(counts, dwell)

    def _stats_log_density(self, counts: np.ndarray, dwell: np.ndarray) -> float:
        with np.errstate(invalid="ignore"):
            emitted = np.where(counts > 0.0, counts * self._log_rates, 0.0)
        if np.any(~np.isfinite(emitted)):  # a transition on a zero-rate edge -> impossible
            return -np.inf
        return float(np.sum(emitted) - np.dot(self._exit, dwell))

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return vectorized log-likelihoods for encoded trajectory statistics."""
        rows = _validated_encoded_rows(x, num_states=self.num_states)
        return np.asarray(
            [self._stats_log_density(c, d) for c, d in rows],
            dtype=np.float64,
        )

    def sampler(self, seed: int | None = None) -> ContinuousTimeMarkovChainSampler:
        """Return a Gillespie sampler for trajectories from this CTMC."""
        return ContinuousTimeMarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ContinuousTimeMarkovChainEstimator:
        """Return the closed-form rate estimator for this state space.

        ``pseudo_count=c`` is the compatibility shorthand for independent
        off-diagonal ``Gamma(shape=1+c, rate=c)`` priors. With no override,
        the distribution's explicit edgewise priors are preserved.
        """
        if pseudo_count is None:
            return ContinuousTimeMarkovChainEstimator(
                self.num_states,
                initial_state=self.initial_state,
                horizon=self.horizon,
                name=self.name,
                keys=self.keys,
                prior_shape=self.prior_shape,
                prior_rate=self.prior_rate,
            )
        return ContinuousTimeMarkovChainEstimator(
            self.num_states,
            pseudo_count=pseudo_count,
            initial_state=self.initial_state,
            horizon=self.horizon,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> ContinuousTimeMarkovChainDataEncoder:
        """Return the trajectory-statistics encoder used by vectorized methods."""
        return ContinuousTimeMarkovChainDataEncoder(self.num_states)


class ContinuousTimeMarkovChainSampler(DistributionSampler):
    """Exact Gillespie simulation of CTMC trajectories on ``[0, horizon]``."""

    def __init__(self, dist: ContinuousTimeMarkovChainDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> tuple[int, float, list[tuple[float, int]]]:
        d = self.dist
        cur = d.initial_state
        t = 0.0
        jumps: list[tuple[float, int]] = []
        while True:
            exit_rate = d._exit[cur]
            if exit_rate <= 0.0:
                break  # absorbing state
            dt = self.rng.exponential(1.0 / exit_rate)
            if t + dt > d.horizon:
                break
            probs = d.rates[cur] / exit_rate
            nxt = int(self.rng.choice(d.num_states, p=probs))
            jumps.append((float(dt), nxt))
            t += dt
            cur = nxt
        return d.initial_state, d.horizon, jumps

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one ``(s0, horizon, jumps)`` trajectory or a list of ``size`` trajectories, all observed
        over the distribution's configured ``horizon``."""
        if size is None:
            return self._sample_one()
        checked_size = _exact_integer(size, label="CTMC sample size")
        if checked_size < 0:
            raise ValueError("CTMC sample size must be non-negative")
        return [self._sample_one() for _ in range(checked_size)]


class ContinuousTimeMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the CTMC sufficient statistics: transition counts ``n_ij`` and dwell times ``T_i``."""

    def __init__(self, num_states: int, name: str | None = None, keys: str | None = None) -> None:
        self.num_states = _positive_integer(
            num_states,
            label="CTMC accumulator state count",
        )
        self.counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.dwell = np.zeros(self.num_states, dtype=np.float64)
        self.name = name
        self.keys = keys

    def _add(self, traj: Any, weight: float) -> None:
        c, d = _trajectory_stats(traj, self.num_states)
        checked_weight = _nonnegative_scalar(weight, label="CTMC weight")
        self.counts += checked_weight * c
        self.dwell += checked_weight * d

    def update(self, x: Any, weight: float, estimate: Any | None) -> None:
        """Update transition-count and dwell-time statistics from one trajectory."""
        self._add(x, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one trajectory."""
        self._add(x, weight)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: Any | None) -> None:
        """Update sufficient statistics from encoded trajectory statistics."""
        rows = _validated_encoded_rows(x, num_states=self.num_states)
        checked_weights = _validated_weights(weights, len(rows))
        for (c, d), w in zip(rows, checked_weights):
            self.counts += w * c
            self.dwell += w * d

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from encoded trajectory statistics."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray]) -> ContinuousTimeMarkovChainAccumulator:
        """Merge transition-count and dwell-time sufficient statistics."""
        counts, dwell = _validated_statistics(
            suff_stat,
            num_states=self.num_states,
        )
        self.counts += counts
        self.dwell += dwell
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of transition counts and dwell times."""
        return ContinuousTimeMarkovChainStatistics(
            self.counts.copy(),
            self.dwell.copy(),
        )

    def from_value(self, x: tuple[np.ndarray, np.ndarray]) -> ContinuousTimeMarkovChainAccumulator:
        """Restore transition-count and dwell-time sufficient statistics."""
        counts, dwell = _validated_statistics(
            x,
            num_states=self.num_states,
        )
        self.counts = counts.copy()
        self.dwell = dwell.copy()
        return self

    def scale(self, c: float) -> ContinuousTimeMarkovChainAccumulator:
        """Scale accumulated sufficient statistics by a constant."""
        checked_scale = _nonnegative_scalar(c, label="CTMC scale")
        self.counts *= checked_scale
        self.dwell *= checked_scale
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

    def acc_to_encoder(self) -> ContinuousTimeMarkovChainDataEncoder:
        """Return the encoder compatible with CTMC sufficient statistics."""
        return ContinuousTimeMarkovChainDataEncoder(self.num_states)


class ContinuousTimeMarkovChainAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for CTMC transition-count and dwell-time statistics."""

    def __init__(self, num_states: int, name: str | None = None, keys: str | None = None) -> None:
        self.num_states = _positive_integer(
            num_states,
            label="CTMC accumulator-factory state count",
        )
        self.name = name
        self.keys = keys

    def make(self) -> ContinuousTimeMarkovChainAccumulator:
        """Create an empty CTMC accumulator."""
        return ContinuousTimeMarkovChainAccumulator(self.num_states, name=self.name, keys=self.keys)


class ContinuousTimeMarkovChainEstimator(ParameterEstimator):
    """Closed-form independent-edge CTMC MLE or Gamma-prior MAP estimator.

    For edge ``i -> j``, an explicit ``Gamma(shape_ij, rate_ij)`` prior gives
    ``q_ij = (n_ij + shape_ij - 1) / (T_i + rate_ij)``. Shapes must be at
    least one so the finite non-negative MAP is well defined. The default
    ``shape=1, rate=0`` recovers the MLE. The legacy ``pseudo_count=c``
    spelling maps coherently to ``shape=1+c, rate=c`` on every off-diagonal
    edge; it cannot be mixed with explicit priors.
    """

    def __init__(
        self,
        num_states: int,
        pseudo_count: float | None = None,
        initial_state: int = 0,
        horizon: float = 10.0,
        name: str | None = None,
        keys: str | None = None,
        prior_shape: Any | None = None,
        prior_rate: Any | None = None,
    ) -> None:
        self.num_states = _positive_integer(
            num_states,
            label="CTMC estimator state count",
        )
        if pseudo_count is not None and (
            prior_shape is not None or prior_rate is not None
        ):
            raise ValueError(
                "CTMC pseudo_count cannot be combined with explicit Gamma priors"
            )
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _nonnegative_scalar(
                pseudo_count,
                label="CTMC pseudo-count",
            )
        )
        if self.pseudo_count is not None:
            shape_value: Any = 1.0 + self.pseudo_count
            rate_value: Any = self.pseudo_count
        else:
            shape_value = 1.0 if prior_shape is None else prior_shape
            rate_value = 0.0 if prior_rate is None else prior_rate
        self.prior_shape = _prior_matrix(
            shape_value,
            num_states=self.num_states,
            label="Gamma prior shape",
            minimum=1.0,
            diagonal=1.0,
        )
        self.prior_rate = _prior_matrix(
            rate_value,
            num_states=self.num_states,
            label="Gamma prior rate",
            minimum=0.0,
            diagonal=0.0,
        )
        self.initial_state = _state(
            initial_state,
            num_states=self.num_states,
            label="CTMC estimator initial state",
        )
        self.horizon = _nonnegative_scalar(
            horizon,
            label="CTMC estimator sampling horizon",
        )
        self.name = name
        self.keys = keys

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Serialize the effective explicit prior, not its legacy shorthand."""
        return {
            "num_states": self.num_states,
            "initial_state": self.initial_state,
            "horizon": self.horizon,
            "name": self.name,
            "keys": self.keys,
            "prior_shape": self.prior_shape,
            "prior_rate": self.prior_rate,
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Validate and restore an estimator from its effective prior."""
        required = {
            "num_states",
            "initial_state",
            "horizon",
            "name",
            "keys",
            "prior_shape",
            "prior_rate",
        }
        if set(state) != required:
            raise ValueError(
                "invalid CTMC estimator state fields: expected %r, got %r"
                % (sorted(required), sorted(state))
            )
        self.__init__(
            state["num_states"],
            initial_state=state["initial_state"],
            horizon=state["horizon"],
            name=state["name"],
            keys=state["keys"],
            prior_shape=state["prior_shape"],
            prior_rate=state["prior_rate"],
        )

    def accumulator_factory(self) -> ContinuousTimeMarkovChainAccumulatorFactory:
        """Return a factory for CTMC sufficient-statistic accumulators."""
        return ContinuousTimeMarkovChainAccumulatorFactory(self.num_states, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]
    ) -> ContinuousTimeMarkovChainDistribution:
        """Estimate off-diagonal rates with the documented Gamma-prior MAP."""
        counts, dwell = _validated_statistics(
            suff_stat,
            num_states=self.num_states,
        )
        if nobs is not None:
            _nonnegative_scalar(nobs, label="CTMC observation count")
        numerator = counts + self.prior_shape - 1.0
        denominator = dwell[:, None] + self.prior_rate
        if np.any((numerator > 0.0) & (denominator == 0.0)):
            raise ValueError(
                "CTMC positive transition evidence or prior shape requires "
                "positive dwell exposure or prior rate"
            )
        rates = np.zeros_like(numerator)
        np.divide(
            numerator,
            denominator,
            out=rates,
            where=denominator > 0.0,
        )
        np.fill_diagonal(rates, 0.0)
        return ContinuousTimeMarkovChainDistribution(
            rates,
            initial_state=self.initial_state,
            horizon=self.horizon,
            name=self.name,
            keys=self.keys,
            prior_shape=self.prior_shape,
            prior_rate=self.prior_rate,
        )


class ContinuousTimeMarkovChainDataEncoder(DataSequenceEncoder):
    """Encode ``(s0, horizon, jumps)`` trajectories into per-trajectory ``(counts, dwell)`` sufficient
    statistics, ``dwell`` including each trajectory's final right-censored interval out to its horizon."""

    def __init__(self, num_states: int) -> None:
        self.num_states = _positive_integer(
            num_states,
            label="CTMC encoder state count",
        )

    def __str__(self) -> str:
        return f"ContinuousTimeMarkovChainDataEncoder({self.num_states})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContinuousTimeMarkovChainDataEncoder) and other.num_states == self.num_states

    def seq_encode(self, x: Sequence[Any]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode trajectories as per-trajectory transition counts and dwell times."""
        return [_trajectory_stats(traj, self.num_states) for traj in x]

    def row_count(self, x: Sequence[Any]) -> int:
        """Return the number of validated encoded trajectory rows."""
        return len(_validated_encoded_rows(x, num_states=self.num_states))
