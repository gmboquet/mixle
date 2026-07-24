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
(each ``dt >= 0``, and the jumps' dwell times may not sum to more than ``horizon``). The horizon travels
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

_MIN_TIME = 1e-12


def _trajectory_stats(traj: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_ij transition-count matrix, T_i dwell-time vector) for one ``(s0, horizon, [(dt, s1), ...])``
    trajectory, including the final right-censored dwell from the last jump (or from time 0, if there
    are no jumps) out to ``horizon``. Raises ``ValueError`` on a malformed trajectory: wrong shape, a
    non-finite/negative horizon, a non-finite/negative dwell time, dwell times summing past the horizon,
    or a state index outside ``[0, k)``.
    """
    if len(traj) != 3:
        raise ValueError(
            "CTMC trajectory must be (initial_state, horizon, jumps); got a "
            f"{len(traj)}-element sequence. The 2-element (s0, jumps) format is no longer accepted: it "
            "has no field for the trajectory's observation horizon, so the final censored dwell time "
            "(the exposure between the last jump and when observation stopped) cannot be recovered."
        )
    s0, horizon, jumps = traj[0], traj[1], traj[2]
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon < 0.0:
        raise ValueError(f"CTMC trajectory horizon must be finite and >= 0, got {horizon!r}.")
    cur = int(s0)
    if not (0 <= cur < k):
        raise ValueError(f"CTMC trajectory initial state {cur} is out of range for {k} states.")

    counts = np.zeros((k, k), dtype=np.float64)
    dwell = np.zeros(k, dtype=np.float64)
    t = 0.0
    for dt, s_next in jumps:
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError(f"CTMC trajectory dwell times must be finite and >= 0, got {dt!r}.")
        t += dt
        if t > horizon:
            raise ValueError(f"CTMC trajectory dwell times sum to {t!r}, which exceeds its horizon {horizon!r}.")
        s_next = int(s_next)
        if not (0 <= s_next < k):
            raise ValueError(f"CTMC trajectory jump target {s_next} is out of range for {k} states.")
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
    ) -> None:
        rates = np.asarray(rates, dtype=np.float64)
        if rates.ndim != 2 or rates.shape[0] != rates.shape[1]:
            raise ValueError("rates must be a square (K, K) matrix of off-diagonal jump rates")
        if np.any(rates < 0.0) or not np.all(np.isfinite(rates)):
            raise ValueError("CTMC rates must be finite and >= 0")
        self.rates = rates.copy()
        np.fill_diagonal(self.rates, 0.0)  # diagonal is derived, not a free rate
        self.num_states = rates.shape[0]
        self.initial_state = int(initial_state)
        self.horizon = float(horizon)
        self.name = name
        self.keys = keys
        self._exit = self.rates.sum(axis=1)  # q_i = total exit rate from state i
        with np.errstate(divide="ignore"):
            self._log_rates = np.log(self.rates)

    def __str__(self) -> str:
        return (
            f"ContinuousTimeMarkovChainDistribution({self.rates.tolist()!r}, "
            f"initial_state={self.initial_state}, horizon={self.horizon}, name={self.name!r}, keys={self.keys!r})"
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
        return np.asarray([self._stats_log_density(c, d) for c, d in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> ContinuousTimeMarkovChainSampler:
        """Return a Gillespie sampler for trajectories from this CTMC."""
        return ContinuousTimeMarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ContinuousTimeMarkovChainEstimator:
        """Return the closed-form rate estimator for this state space."""
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
        return [self._sample_one() for _ in range(size)]


class ContinuousTimeMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the CTMC sufficient statistics: transition counts ``n_ij`` and dwell times ``T_i``."""

    def __init__(self, num_states: int, name: str | None = None, keys: str | None = None) -> None:
        self.num_states = int(num_states)
        self.counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.dwell = np.zeros(self.num_states, dtype=np.float64)
        self.name = name
        self.keys = keys

    def _add(self, traj: Any, weight: float) -> None:
        c, d = _trajectory_stats(traj, self.num_states)
        self.counts += weight * c
        self.dwell += weight * d

    def update(self, x: Any, weight: float, estimate: Any | None) -> None:
        """Update transition-count and dwell-time statistics from one trajectory."""
        self._add(x, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one trajectory."""
        self._add(x, weight)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: Any | None) -> None:
        """Update sufficient statistics from encoded trajectory statistics."""
        for (c, d), w in zip(x, weights):
            self.counts += w * c
            self.dwell += w * d

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from encoded trajectory statistics."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray]) -> ContinuousTimeMarkovChainAccumulator:
        """Merge transition-count and dwell-time sufficient statistics."""
        self.counts += suff_stat[0]
        self.dwell += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of transition counts and dwell times."""
        return self.counts.copy(), self.dwell.copy()

    def from_value(self, x: tuple[np.ndarray, np.ndarray]) -> ContinuousTimeMarkovChainAccumulator:
        """Restore transition-count and dwell-time sufficient statistics."""
        self.counts = np.asarray(x[0], dtype=np.float64).copy()
        self.dwell = np.asarray(x[1], dtype=np.float64).copy()
        self.num_states = self.dwell.shape[0]
        return self

    def scale(self, c: float) -> ContinuousTimeMarkovChainAccumulator:
        """Scale accumulated sufficient statistics by a constant."""
        self.counts *= c
        self.dwell *= c
        return self

    def acc_to_encoder(self) -> ContinuousTimeMarkovChainDataEncoder:
        """Return the encoder compatible with CTMC sufficient statistics."""
        return ContinuousTimeMarkovChainDataEncoder(self.num_states)


class ContinuousTimeMarkovChainAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for CTMC transition-count and dwell-time statistics."""

    def __init__(self, num_states: int, name: str | None = None, keys: str | None = None) -> None:
        self.num_states = int(num_states)
        self.name = name
        self.keys = keys

    def make(self) -> ContinuousTimeMarkovChainAccumulator:
        """Create an empty CTMC accumulator."""
        return ContinuousTimeMarkovChainAccumulator(self.num_states, name=self.name, keys=self.keys)


class ContinuousTimeMarkovChainEstimator(ParameterEstimator):
    """Closed-form rate MLE: ``q_ij = n_ij / T_i`` (independent Poisson rates, unique global optimum)."""

    def __init__(
        self,
        num_states: int,
        pseudo_count: float | None = None,
        initial_state: int = 0,
        horizon: float = 10.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.num_states = int(num_states)
        self.pseudo_count = pseudo_count
        self.initial_state = int(initial_state)
        self.horizon = float(horizon)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ContinuousTimeMarkovChainAccumulatorFactory:
        """Return a factory for CTMC sufficient-statistic accumulators."""
        return ContinuousTimeMarkovChainAccumulatorFactory(self.num_states, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]
    ) -> ContinuousTimeMarkovChainDistribution:
        """Estimate off-diagonal generator rates from transition counts and dwell times."""
        counts, dwell = suff_stat
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        num = counts + pc
        denom = np.maximum(dwell + pc * self.num_states, _MIN_TIME)[:, None]
        rates = num / denom
        np.fill_diagonal(rates, 0.0)
        return ContinuousTimeMarkovChainDistribution(
            rates, initial_state=self.initial_state, horizon=self.horizon, name=self.name, keys=self.keys
        )


class ContinuousTimeMarkovChainDataEncoder(DataSequenceEncoder):
    """Encode ``(s0, horizon, jumps)`` trajectories into per-trajectory ``(counts, dwell)`` sufficient
    statistics, ``dwell`` including each trajectory's final right-censored interval out to its horizon."""

    def __init__(self, num_states: int) -> None:
        self.num_states = int(num_states)

    def __str__(self) -> str:
        return f"ContinuousTimeMarkovChainDataEncoder({self.num_states})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContinuousTimeMarkovChainDataEncoder) and other.num_states == self.num_states

    def seq_encode(self, x: Sequence[Any]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode trajectories as per-trajectory transition counts and dwell times."""
        return [_trajectory_stats(traj, self.num_states) for traj in x]
