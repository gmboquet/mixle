"""Continuous-time Markov chain (CTMC) over fully observed trajectories.

A CTMC on ``K`` states is governed by a generator matrix ``Q``: off-diagonal ``q_ij >= 0`` is the rate
of jumping ``i -> j``, and ``q_ii = -sum_{j!=i} q_ij``. A fully-observed trajectory is the initial state
plus the sequence of ``(dwell_time, next_state)`` jumps; its log-likelihood is

    log L = sum_{i!=j} n_ij * log q_ij  -  sum_i q_i * T_i,      q_i = -q_ii = sum_{j!=i} q_ij,

where ``n_ij`` is the number of observed ``i->j`` transitions and ``T_i`` the total time spent in ``i``.
This is a collection of independent Poisson-rate likelihoods, so the MLE is closed form and
unique: ``q_ij = n_ij / T_i``. The estimator therefore certifies ``GLOBAL_UNIQUE`` (see
:func:`mixle.inference.certify`, which classifies this family). Data type: ``(s0, [(dt, s1), (dt, s2),
...])`` -- the initial state and the observed jumps.

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
    """(n_ij transition-count matrix, T_i dwell-time vector) for one ``(s0, [(dt, s1), ...])`` trajectory."""
    counts = np.zeros((k, k), dtype=np.float64)
    dwell = np.zeros(k, dtype=np.float64)
    s0, jumps = traj[0], traj[1]
    cur = int(s0)
    for dt, s_next in jumps:
        dwell[cur] += float(dt)
        counts[cur, int(s_next)] += 1.0
        cur = int(s_next)
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
            self.num_states, pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> ContinuousTimeMarkovChainDataEncoder:
        """Return the trajectory-statistics encoder used by vectorized methods."""
        return ContinuousTimeMarkovChainDataEncoder(self.num_states)


class ContinuousTimeMarkovChainSampler(DistributionSampler):
    """Exact Gillespie simulation of CTMC trajectories on ``[0, horizon]``."""

    def __init__(self, dist: ContinuousTimeMarkovChainDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> tuple[int, list[tuple[float, int]]]:
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
        return d.initial_state, jumps

    def sample(self, size: int | None = None) -> Any:
        """Draw one trajectory or a list of trajectories over the configured horizon."""
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

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed transition and dwell statistics into this accumulator."""
        if self.keys is not None and self.keys in stats_dict:
            c, d = stats_dict[self.keys]
            self.counts += c
            self.dwell += d

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.counts, self.dwell = (np.asarray(v, dtype=np.float64).copy() for v in stats_dict[self.keys])

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
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.num_states = int(num_states)
        self.pseudo_count = pseudo_count
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
        return ContinuousTimeMarkovChainDistribution(rates, name=self.name, keys=self.keys)


class ContinuousTimeMarkovChainDataEncoder(DataSequenceEncoder):
    """Encode ``(s0, jumps)`` trajectories into per-trajectory ``(counts, dwell)`` sufficient statistics."""

    def __init__(self, num_states: int) -> None:
        self.num_states = int(num_states)

    def __str__(self) -> str:
        return f"ContinuousTimeMarkovChainDataEncoder({self.num_states})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContinuousTimeMarkovChainDataEncoder) and other.num_states == self.num_states

    def seq_encode(self, x: Sequence[Any]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode trajectories as per-trajectory transition counts and dwell times."""
        return [_trajectory_stats(traj, self.num_states) for traj in x]
