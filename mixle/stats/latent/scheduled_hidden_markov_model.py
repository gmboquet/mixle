"""A length- and position-conditional ("scheduled") hidden Markov model.

A standard HMM is time-homogeneous: the same initial distribution, transition matrix, and emissions apply at
every position, and (with a ``len_dist``) the length is drawn independently of the path -- it only sets the
count. This family makes the dynamics depend on **where you are in the sequence and how long it is**, through a
serializable ``PhaseSchedule`` ``phi(t, L)`` that maps position ``t`` in a length-``L`` sequence to a *phase*.
Each phase has its own initial / transition / emission parameters; EM pools sufficient statistics by phase.

One mechanism covers every reasonable "length-conditional" model:

- :class:`Homogeneous` -- ``phi(t, L) = 0`` -- the ordinary HMM (one phase).
- :class:`ByLength` -- ``phi(t, L) = bucket(L)`` -- a **length-conditional** HMM: short and long sequences use
  different dynamics (constant within a sequence).
- :class:`ByRelativePosition` -- ``phi(t, L) = floor(B * t / L)`` -- **relative position**: the chain knows how
  far through the sequence it is (e.g. winds down toward the end), regardless of absolute length.
- :class:`ByPosition` -- ``phi(t, L) = min(t, cap-1)`` -- **absolute position** (non-homogeneous in time).

The length itself is still drawn from ``len_dist`` (it remains a random variable); the schedule adds the
*conditioning* of the content on length/position that the homogeneous model lacks. Emissions are per-phase too,
so length/position can shape emissions, not just transitions.

This is a deliberately lean, numpy-only implementation (no numba / enumeration / terminal-state integration --
those live on :class:`~mixle.stats.latent.hidden_markov.HiddenMarkovModelDistribution`). It reuses the emission
families' own estimators for the M-step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_NEG_INF = -np.inf


# ---------------------------------------------------------------------------------------------------------
# Phase schedules: serializable phi(t, L) -> phase index in [0, n_phases).
# ---------------------------------------------------------------------------------------------------------
class PhaseSchedule:
    """Maps a position ``t`` in a length-``L`` sequence to a phase index in ``[0, n_phases)``."""

    n_phases: int = 1

    def phase(self, t: int, length: int) -> int:  # pragma: no cover - overridden
        """Return the phase index for position ``t`` in a sequence of ``length``."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schedule to a JSON-compatible dictionary."""
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PhaseSchedule:
        """Deserialize a schedule produced by :meth:`to_dict`."""
        kind = d["kind"]
        for cls in (Homogeneous, ByPosition, ByRelativePosition, ByLength):
            if cls.__name__ == kind:
                return cls._from_dict(d)
        raise ValueError("unknown PhaseSchedule kind %r" % kind)


class Homogeneous(PhaseSchedule):
    """One phase for everything -- the ordinary time-homogeneous HMM."""

    n_phases = 1

    def phase(self, t: int, length: int) -> int:
        """Return the single homogeneous phase."""
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the homogeneous schedule."""
        return {"kind": "Homogeneous"}

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Homogeneous:
        return cls()


class ByPosition(PhaseSchedule):
    """Absolute position: ``phi(t, L) = min(t, cap - 1)`` (positions past ``cap-1`` share the last phase)."""

    def __init__(self, cap: int) -> None:
        if cap < 1:
            raise ValueError("cap must be >= 1")
        self.cap = int(cap)
        self.n_phases = self.cap

    def phase(self, t: int, length: int) -> int:
        """Return the absolute-position phase capped at the final phase."""
        return min(int(t), self.cap - 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the absolute-position schedule."""
        return {"kind": "ByPosition", "cap": self.cap}

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ByPosition:
        return cls(d["cap"])


class ByRelativePosition(PhaseSchedule):
    """Relative position: ``phi(t, L) = min(bins - 1, floor(bins * t / L))`` -- progress through the sequence."""

    def __init__(self, bins: int) -> None:
        if bins < 1:
            raise ValueError("bins must be >= 1")
        self.bins = int(bins)
        self.n_phases = self.bins

    def phase(self, t: int, length: int) -> int:
        """Return the relative-position phase for ``t / length``."""
        if length <= 0:
            return 0
        return min(self.bins - 1, (int(t) * self.bins) // int(length))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the relative-position schedule."""
        return {"kind": "ByRelativePosition", "bins": self.bins}

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ByRelativePosition:
        return cls(d["bins"])


class ByLength(PhaseSchedule):
    """Length-conditional: phase is the bucket of ``L`` against sorted ``boundaries`` (constant within a seq).

    With ``boundaries = [5, 10]`` there are three phases: ``L <= 5``, ``5 < L <= 10``, ``L > 10``.
    """

    def __init__(self, boundaries: Sequence[int]) -> None:
        self.boundaries = [int(b) for b in boundaries]
        if any(self.boundaries[i] >= self.boundaries[i + 1] for i in range(len(self.boundaries) - 1)):
            raise ValueError("boundaries must be strictly increasing")
        self.n_phases = len(self.boundaries) + 1

    def phase(self, t: int, length: int) -> int:
        """Return the length-bucket phase for the sequence length."""
        return int(sum(1 for b in self.boundaries if length > b))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the length-bucket schedule."""
        return {"kind": "ByLength", "boundaries": list(self.boundaries)}

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ByLength:
        return cls(d["boundaries"])


# ---------------------------------------------------------------------------------------------------------
# The per-sequence forward / forward-backward over a phase-indexed trellis.
# ---------------------------------------------------------------------------------------------------------
def _log_b(emissions: list[list[Any]], schedule: PhaseSchedule, x: list[Any]) -> np.ndarray:
    """Per-position, per-state emission log-density ``(L, K)`` using each position's phase emissions."""
    length = len(x)
    k = len(emissions[0])
    log_b = np.empty((length, k))
    for t in range(length):
        em = emissions[schedule.phase(t, length)]
        for j in range(k):
            log_b[t, j] = em[j].log_density(x[t])
    return log_b


def _forward(log_inits: np.ndarray, log_trans: np.ndarray, log_b: np.ndarray, schedule: PhaseSchedule) -> float:
    length = log_b.shape[0]
    la = log_inits[schedule.phase(0, length)] + log_b[0]
    for t in range(1, length):
        a = log_trans[schedule.phase(t - 1, length)]  # transition leaving position t-1
        la = log_b[t] + logsumexp(la[:, None] + a, axis=0)
    return float(logsumexp(la))


def _forward_backward(
    log_inits: np.ndarray, log_trans: np.ndarray, log_b: np.ndarray, schedule: PhaseSchedule
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(loglik, gamma (L,K), xi (L-1,K,K))`` -- state and transition posteriors."""
    length, k = log_b.shape
    la = np.empty((length, k))
    la[0] = log_inits[schedule.phase(0, length)] + log_b[0]
    for t in range(1, length):
        a = log_trans[schedule.phase(t - 1, length)]
        la[t] = log_b[t] + logsumexp(la[t - 1][:, None] + a, axis=0)
    loglik = float(logsumexp(la[length - 1]))
    lb = np.empty((length, k))
    lb[length - 1] = 0.0
    for t in range(length - 2, -1, -1):
        a = log_trans[schedule.phase(t, length)]
        lb[t] = logsumexp(a + (log_b[t + 1] + lb[t + 1])[None, :], axis=1)
    gamma = np.exp(la + lb - loglik)
    xi = np.empty((max(length - 1, 0), k, k))
    for t in range(length - 1):
        a = log_trans[schedule.phase(t, length)]
        m = la[t][:, None] + a + (log_b[t + 1] + lb[t + 1])[None, :] - loglik
        xi[t] = np.exp(m)
    return loglik, gamma, xi


# ---------------------------------------------------------------------------------------------------------
# Distribution.
# ---------------------------------------------------------------------------------------------------------
class ScheduledHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """Phase-indexed (length-/position-conditional) HMM. See the module docstring for the modeling story."""

    def __init__(
        self,
        inits: np.ndarray,
        transitions: np.ndarray,
        emissions: list[list[Any]],
        schedule: PhaseSchedule,
        len_dist: Any = None,
        name: str | None = None,
    ) -> None:
        self.inits = np.asarray(inits, dtype=float)  # (P, K)
        self.transitions = np.asarray(transitions, dtype=float)  # (P, K, K)
        self.emissions = [list(row) for row in emissions]  # P x K grid of emission distributions
        self.schedule = schedule
        self.len_dist = len_dist
        self.name = name
        self.n_phases = schedule.n_phases
        self.n_states = self.inits.shape[1]
        if self.inits.shape != (self.n_phases, self.n_states):
            raise ValueError("inits must be (n_phases, n_states)")
        if self.transitions.shape != (self.n_phases, self.n_states, self.n_states):
            raise ValueError("transitions must be (n_phases, n_states, n_states)")
        if len(self.emissions) != self.n_phases or any(len(r) != self.n_states for r in self.emissions):
            raise ValueError("emissions must be an n_phases x n_states grid")
        with np.errstate(divide="ignore"):
            self._log_inits = np.log(self.inits)
            self._log_trans = np.log(self.transitions)

    def __str__(self) -> str:
        return "ScheduledHiddenMarkovModelDistribution(n_phases=%d, n_states=%d, schedule=%s)" % (
            self.n_phases,
            self.n_states,
            self.schedule.to_dict(),
        )

    def log_density(self, x: list[Any]) -> float:
        """Return the log likelihood of one scheduled HMM sequence."""
        length = len(x)
        if length == 0:
            return self.len_dist.log_density(0) if self.len_dist is not None else _NEG_INF
        lp = _forward(self._log_inits, self._log_trans, _log_b(self.emissions, self.schedule, x), self.schedule)
        if self.len_dist is not None:
            lp += self.len_dist.log_density(length)
        return lp

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Score a batch of scheduled HMM sequences."""
        return np.array([self.log_density(seq) for seq in x], dtype=float)

    def sampler(self, seed: int | None = None) -> ScheduledHMMSampler:
        """Return a sampler for scheduled HMM sequences."""
        return ScheduledHMMSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ScheduledHMMEstimator:
        """Raise because emission and length estimators must be supplied explicitly."""
        raise NotImplementedError("supply emission/len estimators via ScheduledHMMEstimator(...) directly")

    def dist_to_encoder(self) -> ScheduledHMMDataEncoder:
        """Return the pass-through scheduled HMM encoder."""
        return ScheduledHMMDataEncoder()


# ---------------------------------------------------------------------------------------------------------
# Sampler.
# ---------------------------------------------------------------------------------------------------------
class ScheduledHMMSampler(DistributionSampler):
    """Sampler for scheduled HMM sequences."""

    def __init__(self, dist: ScheduledHiddenMarkovModelDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.len_sampler = None if dist.len_dist is None else dist.len_dist.sampler(self.rng.randint(2**31))
        self.emit_seed = self.rng.randint(2**31)

    def _sample_length(self) -> int:
        if self.len_sampler is None:
            raise ValueError("a len_dist is required to sample (the length is a random variable).")
        return int(self.len_sampler.sample())

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one sequence or a list of sequences."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        d = self.dist
        length = self._sample_length()
        if length <= 0:
            return []
        out: list[Any] = []
        p0 = d.schedule.phase(0, length)
        z = int(self.rng.choice(d.n_states, p=d.inits[p0]))
        out.append(d.emissions[p0][z].sampler(self.rng.randint(2**31)).sample())
        for t in range(1, length):
            a = d.transitions[d.schedule.phase(t - 1, length)]
            z = int(self.rng.choice(d.n_states, p=a[z]))
            pt = d.schedule.phase(t, length)
            out.append(d.emissions[pt][z].sampler(self.rng.randint(2**31)).sample())
        return out


# ---------------------------------------------------------------------------------------------------------
# Encoder (lean pass-through; the seq_* methods loop over raw sequences).
# ---------------------------------------------------------------------------------------------------------
class ScheduledHMMDataEncoder(DataSequenceEncoder):
    """Pass-through encoder for scheduled HMM sequence observations."""

    def __str__(self) -> str:
        return "ScheduledHMMDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScheduledHMMDataEncoder)

    def seq_encode(self, x: list[list[Any]]) -> list[list[Any]]:
        """Encode scheduled HMM records as sequence lists."""
        return [list(seq) for seq in x]


# ---------------------------------------------------------------------------------------------------------
# EM: phase-pooled forward-backward. Emissions and len_dist are re-estimated by their own estimators.
# ---------------------------------------------------------------------------------------------------------
class ScheduledHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for phase-pooled scheduled HMM EM sufficient statistics."""

    def __init__(self, n_states: int, schedule: PhaseSchedule, emission_factory: Any, len_factory: Any = None) -> None:
        self.n_states = int(n_states)
        self.schedule = schedule
        self.n_phases = schedule.n_phases
        self.emission_factory = emission_factory
        self.len_factory = len_factory
        self.init_counts = np.zeros((self.n_phases, self.n_states))
        self.trans_counts = np.zeros((self.n_phases, self.n_states, self.n_states))
        self.emission_acc = [[emission_factory.make() for _ in range(self.n_states)] for _ in range(self.n_phases)]
        self.len_acc = None if len_factory is None else len_factory.make()

    def _accumulate(self, x: list[Any], weight: float, gamma: np.ndarray, xi: np.ndarray, estimate: Any) -> None:
        length = len(x)
        if self.len_acc is not None:
            self.len_acc.update(length, weight, None if estimate is None else estimate.len_dist)
        if length == 0:
            return
        self.init_counts[self.schedule.phase(0, length)] += weight * gamma[0]
        for t in range(length - 1):
            self.trans_counts[self.schedule.phase(t, length)] += weight * xi[t]
        for t in range(length):
            p = self.schedule.phase(t, length)
            for j in range(self.n_states):
                prev = None if estimate is None else estimate.emissions[p][j]
                self.emission_acc[p][j].update(x[t], weight * gamma[t, j], prev)

    def update(self, x: list[Any], weight: float, estimate: ScheduledHiddenMarkovModelDistribution) -> None:
        """Accumulate sufficient statistics from one weighted sequence."""
        if len(x) == 0:
            self._accumulate(
                x, weight, np.zeros((0, self.n_states)), np.zeros((0, self.n_states, self.n_states)), estimate
            )
            return
        log_b = _log_b(estimate.emissions, estimate.schedule, x)
        _, gamma, xi = _forward_backward(estimate._log_inits, estimate._log_trans, log_b, estimate.schedule)
        self._accumulate(x, weight, gamma, xi, estimate)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: ScheduledHiddenMarkovModelDistribution) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        for seq, w in zip(x, weights):
            self.update(seq, float(w), estimate)

    def initialize(self, x: list[Any], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with random soft state responsibilities."""
        length = len(x)
        if length == 0:
            self._accumulate(x, weight, np.zeros((0, self.n_states)), np.zeros((0, self.n_states, self.n_states)), None)
            return
        gamma = rng.dirichlet(np.ones(self.n_states), size=length)  # random soft responsibilities to seed EM
        xi = np.array([np.outer(gamma[t], gamma[t + 1]) for t in range(length - 1)]).reshape(
            (length - 1, self.n_states, self.n_states)
        )
        self._accumulate(x, weight, gamma, xi, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        for seq, w in zip(x, weights):
            self.initialize(seq, float(w), rng)

    def combine(self, other: Any) -> ScheduledHMMAccumulator:
        """Merge serialized scheduled HMM sufficient statistics."""
        ic, tc, em, lv = other
        self.init_counts += ic
        self.trans_counts += tc
        for p in range(self.n_phases):
            for j in range(self.n_states):
                self.emission_acc[p][j].combine(em[p][j])
        if self.len_acc is not None and lv is not None:
            self.len_acc.combine(lv)
        return self

    def value(self) -> tuple:
        """Return serialized scheduled HMM sufficient statistics."""
        em = [[self.emission_acc[p][j].value() for j in range(self.n_states)] for p in range(self.n_phases)]
        return (
            self.init_counts.copy(),
            self.trans_counts.copy(),
            em,
            None if self.len_acc is None else self.len_acc.value(),
        )

    def from_value(self, value: tuple) -> ScheduledHMMAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        ic, tc, em, lv = value
        self.init_counts = np.array(ic, dtype=float)
        self.trans_counts = np.array(tc, dtype=float)
        for p in range(self.n_phases):
            for j in range(self.n_states):
                self.emission_acc[p][j].from_value(em[p][j])
        if self.len_acc is not None and lv is not None:
            self.len_acc.from_value(lv)
        return self

    def acc_to_encoder(self) -> ScheduledHMMDataEncoder:
        """Return the encoder associated with this accumulator."""
        return ScheduledHMMDataEncoder()


class ScheduledHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for scheduled HMM accumulators."""

    def __init__(
        self, n_states: int, schedule: PhaseSchedule, emission_estimator: Any, len_estimator: Any = None
    ) -> None:
        self.n_states = n_states
        self.schedule = schedule
        self.emission_estimator = emission_estimator
        self.len_estimator = len_estimator

    def make(self) -> ScheduledHMMAccumulator:
        """Create a fresh scheduled HMM accumulator."""
        len_factory = None if self.len_estimator is None else self.len_estimator.accumulator_factory()
        return ScheduledHMMAccumulator(
            self.n_states, self.schedule, self.emission_estimator.accumulator_factory(), len_factory
        )


class ScheduledHMMEstimator(ParameterEstimator):
    """EM estimator for a :class:`ScheduledHiddenMarkovModelDistribution` with a fixed schedule.

    ``emission_estimator`` is the estimator for ONE emission distribution (reused for every phase x state);
    ``len_estimator`` (optional) estimates the length distribution. The schedule is fixed (it defines the
    parameter sharing); only the per-phase parameters are learned.
    """

    def __init__(
        self,
        n_states: int,
        schedule: PhaseSchedule,
        emission_estimator: Any,
        len_estimator: Any = None,
        pseudo_count: float = 1e-8,
        name: str | None = None,
    ) -> None:
        self.n_states = int(n_states)
        self.schedule = schedule
        self.emission_estimator = emission_estimator
        self.len_estimator = len_estimator
        self.pseudo_count = float(pseudo_count)
        self.name = name

    def accumulator_factory(self) -> ScheduledHMMAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return ScheduledHMMAccumulatorFactory(self.n_states, self.schedule, self.emission_estimator, self.len_estimator)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> ScheduledHiddenMarkovModelDistribution:
        """Estimate phase-indexed initial, transition, emission, and length models."""
        ic, tc, em_vals, lv = suff_stat
        pc = self.pseudo_count
        inits = ic + pc
        inits = inits / inits.sum(axis=1, keepdims=True)
        trans = tc + pc
        rsum = trans.sum(axis=2, keepdims=True)
        rsum[rsum == 0] = 1.0
        trans = trans / rsum
        emissions = [
            [self.emission_estimator.estimate(None, em_vals[p][j]) for j in range(self.n_states)]
            for p in range(self.schedule.n_phases)
        ]
        len_dist = None if self.len_estimator is None else self.len_estimator.estimate(None, lv)
        return ScheduledHiddenMarkovModelDistribution(inits, trans, emissions, self.schedule, len_dist, self.name)
