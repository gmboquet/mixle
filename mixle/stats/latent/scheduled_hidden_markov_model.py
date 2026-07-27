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
from numbers import Real
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.compute.mixture_evidence import validated_row_probability_matrix
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_observation_weights,
    validated_statistic_tuple,
)
from mixle.utils.vector import require_possible_log_evidence

_NEG_INF = -np.inf


def _exact_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    """Return an exact integer, optionally bounded below."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _schedule_phase(schedule: PhaseSchedule, t: int, length: int) -> int:
    """Evaluate a schedule and enforce its phase-index contract."""
    phase = schedule.phase(t, length)
    phase = _exact_integer(phase, f"{type(schedule).__name__}.phase result", minimum=0)
    if phase >= schedule.n_phases:
        raise ValueError(f"{type(schedule).__name__}.phase returned {phase}, outside [0, {schedule.n_phases}).")
    return phase


def _validated_schedule(schedule: Any) -> PhaseSchedule:
    """Require a schedule with an exact positive phase count."""
    if not isinstance(schedule, PhaseSchedule):
        raise TypeError("schedule must be a PhaseSchedule.")
    _exact_integer(schedule.n_phases, "schedule n_phases", minimum=1)
    return schedule


def _owned_emission_grid(values: Any, n_phases: int, n_states: int, label: str) -> list[list[Any]]:
    """Return an owned exact phase-by-state grid."""
    try:
        rows = [list(row) for row in values]
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of emission rows.") from exc
    if len(rows) != n_phases or any(len(row) != n_states for row in rows):
        raise ValueError(f"{label} must be an exact {n_phases} x {n_states} grid.")
    return rows


def _validated_pseudo_count(value: Any) -> float:
    """Return a finite non-negative scheduled-HMM pseudo-count."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("ScheduledHMMEstimator pseudo_count must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("ScheduledHMMEstimator pseudo_count must be finite and non-negative.")
    return result


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
        self.cap = _exact_integer(cap, "ByPosition cap", minimum=1)
        self.n_phases = self.cap

    def phase(self, t: int, length: int) -> int:
        """Return the absolute-position phase capped at the final phase."""
        position = _exact_integer(t, "ByPosition position", minimum=0)
        _exact_integer(length, "ByPosition length", minimum=0)
        return min(position, self.cap - 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the absolute-position schedule."""
        return {"kind": "ByPosition", "cap": self.cap}

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ByPosition:
        return cls(d["cap"])


class ByRelativePosition(PhaseSchedule):
    """Relative position: ``phi(t, L) = min(bins - 1, floor(bins * t / L))`` -- progress through the sequence."""

    def __init__(self, bins: int) -> None:
        self.bins = _exact_integer(bins, "ByRelativePosition bins", minimum=1)
        self.n_phases = self.bins

    def phase(self, t: int, length: int) -> int:
        """Return the relative-position phase for ``t / length``."""
        position = _exact_integer(t, "ByRelativePosition position", minimum=0)
        sequence_length = _exact_integer(length, "ByRelativePosition length", minimum=0)
        if sequence_length == 0:
            return 0
        return min(self.bins - 1, (position * self.bins) // sequence_length)

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
        self.boundaries = [_exact_integer(boundary, "ByLength boundary", minimum=0) for boundary in boundaries]
        if any(self.boundaries[i] >= self.boundaries[i + 1] for i in range(len(self.boundaries) - 1)):
            raise ValueError("boundaries must be strictly increasing")
        self.n_phases = len(self.boundaries) + 1

    def phase(self, t: int, length: int) -> int:
        """Return the length-bucket phase for the sequence length."""
        _exact_integer(t, "ByLength position", minimum=0)
        sequence_length = _exact_integer(length, "ByLength length", minimum=0)
        return sum(1 for boundary in self.boundaries if sequence_length > boundary)

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
        em = emissions[_schedule_phase(schedule, t, length)]
        for j in range(k):
            log_b[t, j] = em[j].log_density(x[t])
    return log_b


def _forward(log_inits: np.ndarray, log_trans: np.ndarray, log_b: np.ndarray, schedule: PhaseSchedule) -> float:
    length = log_b.shape[0]
    la = log_inits[_schedule_phase(schedule, 0, length)] + log_b[0]
    for t in range(1, length):
        a = log_trans[_schedule_phase(schedule, t - 1, length)]  # transition leaving position t-1
        la = log_b[t] + logsumexp(la[:, None] + a, axis=0)
    return float(logsumexp(la))


def _forward_backward(
    log_inits: np.ndarray, log_trans: np.ndarray, log_b: np.ndarray, schedule: PhaseSchedule
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(loglik, gamma (L,K), xi (L-1,K,K))`` -- state and transition posteriors."""
    length, k = log_b.shape
    la = np.empty((length, k))
    la[0] = log_inits[_schedule_phase(schedule, 0, length)] + log_b[0]
    for t in range(1, length):
        a = log_trans[_schedule_phase(schedule, t - 1, length)]
        la[t] = log_b[t] + logsumexp(la[t - 1][:, None] + a, axis=0)
    loglik = float(logsumexp(la[length - 1]))
    lb = np.empty((length, k))
    lb[length - 1] = 0.0
    for t in range(length - 2, -1, -1):
        a = log_trans[_schedule_phase(schedule, t, length)]
        lb[t] = logsumexp(a + (log_b[t + 1] + lb[t + 1])[None, :], axis=1)
    gamma = np.exp(la + lb - loglik)
    xi = np.empty((max(length - 1, 0), k, k))
    for t in range(length - 1):
        a = log_trans[_schedule_phase(schedule, t, length)]
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
        self.schedule = _validated_schedule(schedule)
        self.n_phases = self.schedule.n_phases
        try:
            raw_inits = np.asarray(inits, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("scheduled HMM inits must be a numeric phase-by-state matrix.") from exc
        if raw_inits.ndim != 2 or raw_inits.shape[0] != self.n_phases or raw_inits.shape[1] == 0:
            raise ValueError(f"scheduled HMM inits must have shape ({self.n_phases}, n_states) with n_states positive.")
        self.n_states = raw_inits.shape[1]
        self.inits = validated_row_probability_matrix(
            raw_inits,
            "scheduled HMM initial probabilities",
            shape=(self.n_phases, self.n_states),
        )
        expected_transitions = (self.n_phases, self.n_states, self.n_states)
        try:
            raw_transitions = np.asarray(transitions, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("scheduled HMM transitions must be a numeric phase-by-state tensor.") from exc
        if raw_transitions.shape != expected_transitions:
            raise ValueError(f"scheduled HMM transitions must have shape {expected_transitions}.")
        self.transitions = validated_row_probability_matrix(
            raw_transitions.reshape(self.n_phases * self.n_states, self.n_states),
            "scheduled HMM transition probabilities",
            shape=(self.n_phases * self.n_states, self.n_states),
        ).reshape(expected_transitions)
        self.emissions = _owned_emission_grid(
            emissions,
            self.n_phases,
            self.n_states,
            "scheduled HMM emissions",
        )
        self.len_dist = len_dist
        self.name = name
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
        """Create the matching estimator from this distribution's own components.

        Every phase/state cell retains its own estimator, so heterogeneous
        emission families survive fitting. The length estimator mirrors
        ``len_dist`` when one is modeled.
        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        emission_estimators = [
            [component.estimator(pseudo_count=pseudo_count) for component in row] for row in self.emissions
        ]
        arguments = dict(
            n_states=self.n_states,
            schedule=self.schedule,
            emission_estimator=None,
            len_estimator=len_est,
            name=self.name,
            emission_estimators=emission_estimators,
        )
        if pseudo_count is not None:
            arguments["pseudo_count"] = pseudo_count
        return ScheduledHMMEstimator(**arguments)

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
        return _exact_integer(self.len_sampler.sample(), "scheduled HMM sampled length", minimum=0)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one sequence or a list of sequences."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        d = self.dist
        length = self._sample_length()
        if length <= 0:
            return []
        out: list[Any] = []
        p0 = _schedule_phase(d.schedule, 0, length)
        z = int(self.rng.choice(d.n_states, p=d.inits[p0]))
        out.append(d.emissions[p0][z].sampler(self.rng.randint(2**31)).sample())
        for t in range(1, length):
            a = d.transitions[_schedule_phase(d.schedule, t - 1, length)]
            z = int(self.rng.choice(d.n_states, p=a[z]))
            pt = _schedule_phase(d.schedule, t, length)
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

    def row_count(self, x: Any) -> int:
        """Return the number of pass-through scheduled-HMM records."""
        if not isinstance(x, list):
            raise ValueError("scheduled HMM encoding must be a list of sequence records")
        return len(x)


# ---------------------------------------------------------------------------------------------------------
# EM: phase-pooled forward-backward. Emissions and len_dist are re-estimated by their own estimators.
# ---------------------------------------------------------------------------------------------------------
class ScheduledHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for phase-pooled scheduled HMM EM sufficient statistics."""

    def __init__(
        self,
        n_states: int,
        schedule: PhaseSchedule,
        emission_factories: Any,
        len_factory: Any = None,
    ) -> None:
        self.n_states = _exact_integer(n_states, "ScheduledHMMAccumulator n_states", minimum=1)
        self.schedule = _validated_schedule(schedule)
        self.n_phases = self.schedule.n_phases
        self.emission_factories = (
            [[emission_factories for _ in range(self.n_states)] for _ in range(self.n_phases)]
            if callable(getattr(emission_factories, "make", None))
            else _owned_emission_grid(
                emission_factories,
                self.n_phases,
                self.n_states,
                "ScheduledHMMAccumulator emission factories",
            )
        )
        self.len_factory = len_factory
        self.init_counts = np.zeros((self.n_phases, self.n_states))
        self.trans_counts = np.zeros((self.n_phases, self.n_states, self.n_states))
        self.emission_counts = np.zeros((self.n_phases, self.n_states))
        self.emission_acc = [
            [self.emission_factories[p][j].make() for j in range(self.n_states)] for p in range(self.n_phases)
        ]
        self.len_acc = None if len_factory is None else len_factory.make()

    def _accumulate(self, x: list[Any], weight: float, gamma: np.ndarray, xi: np.ndarray, estimate: Any) -> None:
        length = len(x)
        if self.len_acc is not None:
            self.len_acc.update(length, weight, None if estimate is None else estimate.len_dist)
        if length == 0:
            return
        self.init_counts[_schedule_phase(self.schedule, 0, length)] += weight * gamma[0]
        for t in range(length - 1):
            self.trans_counts[_schedule_phase(self.schedule, t, length)] += weight * xi[t]
        for t in range(length):
            p = _schedule_phase(self.schedule, t, length)
            self.emission_counts[p] += weight * gamma[t]
            for j in range(self.n_states):
                prev = None if estimate is None else estimate.emissions[p][j]
                self.emission_acc[p][j].update(x[t], weight * gamma[t, j], prev)

    def update(self, x: list[Any], weight: float, estimate: ScheduledHiddenMarkovModelDistribution) -> None:
        """Accumulate sufficient statistics from one weighted sequence."""
        weight = validated_observation_weight(weight, "scheduled-HMM observation weight")
        require_possible_log_evidence(
            estimate.log_density(x),
            context="ScheduledHMMAccumulator.update",
        )
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
        weights = validated_observation_weights(weights, len(x), "scheduled-HMM observation weights")
        require_possible_log_evidence(
            estimate.seq_log_density(x),
            context="ScheduledHMMAccumulator.seq_update",
        )
        for seq, w in zip(x, weights):
            if len(seq) == 0:
                self._accumulate(
                    seq,
                    float(w),
                    np.zeros((0, self.n_states)),
                    np.zeros((0, self.n_states, self.n_states)),
                    estimate,
                )
                continue
            log_b = _log_b(estimate.emissions, estimate.schedule, seq)
            _, gamma, xi = _forward_backward(estimate._log_inits, estimate._log_trans, log_b, estimate.schedule)
            self._accumulate(seq, float(w), gamma, xi, estimate)

    def initialize(self, x: list[Any], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with random soft state responsibilities."""
        weight = validated_observation_weight(weight, "scheduled-HMM initialization weight")
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
        weights = validated_observation_weights(weights, len(x), "scheduled-HMM initialization weights")
        for seq, w in zip(x, weights):
            self.initialize(seq, float(w), rng)

    def combine(self, other: Any) -> ScheduledHMMAccumulator:
        """Merge serialized scheduled HMM sufficient statistics."""
        ic, tc, ec, em, lv = validated_statistic_tuple(other, 5, "scheduled-HMM sufficient statistics")
        ic = validated_count_array(ic, self.init_counts.shape, "scheduled-HMM initial counts")
        tc = validated_count_array(tc, self.trans_counts.shape, "scheduled-HMM transition counts")
        ec = validated_count_array(ec, self.emission_counts.shape, "scheduled-HMM emission counts")
        if not np.isclose(float(ec.sum()), float(ic.sum() + tc.sum()), rtol=1.0e-9, atol=1.0e-9):
            raise ValueError("scheduled-HMM emission counts must equal initial plus transition mass")
        em = _owned_emission_grid(
            em,
            self.n_phases,
            self.n_states,
            "ScheduledHMMAccumulator emission statistics",
        )
        self.init_counts += ic
        self.trans_counts += tc
        self.emission_counts += ec
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
            self.emission_counts.copy(),
            em,
            None if self.len_acc is None else self.len_acc.value(),
        )

    def from_value(self, value: tuple) -> ScheduledHMMAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        ic, tc, ec, em, lv = validated_statistic_tuple(value, 5, "scheduled-HMM sufficient statistics")
        self.init_counts = validated_count_array(ic, self.init_counts.shape, "scheduled-HMM initial counts")
        self.trans_counts = validated_count_array(tc, self.trans_counts.shape, "scheduled-HMM transition counts")
        self.emission_counts = validated_count_array(ec, self.emission_counts.shape, "scheduled-HMM emission counts")
        if not np.isclose(
            float(self.emission_counts.sum()),
            float(self.init_counts.sum() + self.trans_counts.sum()),
            rtol=1.0e-9,
            atol=1.0e-9,
        ):
            raise ValueError("scheduled-HMM emission counts must equal initial plus transition mass")
        em = _owned_emission_grid(
            em,
            self.n_phases,
            self.n_states,
            "ScheduledHMMAccumulator emission statistics",
        )
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
        self,
        n_states: int,
        schedule: PhaseSchedule,
        emission_estimators: Any,
        len_estimator: Any = None,
    ) -> None:
        self.n_states = _exact_integer(n_states, "ScheduledHMMAccumulatorFactory n_states", minimum=1)
        self.schedule = _validated_schedule(schedule)
        self.emission_estimators = (
            [[emission_estimators for _ in range(self.n_states)] for _ in range(self.schedule.n_phases)]
            if callable(getattr(emission_estimators, "accumulator_factory", None))
            else _owned_emission_grid(
                emission_estimators,
                self.schedule.n_phases,
                self.n_states,
                "ScheduledHMMAccumulatorFactory emission estimators",
            )
        )
        self.len_estimator = len_estimator

    def make(self) -> ScheduledHMMAccumulator:
        """Create a fresh scheduled HMM accumulator."""
        len_factory = None if self.len_estimator is None else self.len_estimator.accumulator_factory()
        emission_factories = [
            [self.emission_estimators[p][j].accumulator_factory() for j in range(self.n_states)]
            for p in range(self.schedule.n_phases)
        ]
        return ScheduledHMMAccumulator(
            self.n_states,
            self.schedule,
            emission_factories,
            len_factory,
        )


class ScheduledHMMEstimator(ParameterEstimator):
    """EM estimator for a :class:`ScheduledHiddenMarkovModelDistribution` with a fixed schedule.

    ``emission_estimator`` is the backward-compatible homogeneous estimator
    prototype. ``emission_estimators`` may instead provide an exact
    phase-by-state grid, preserving heterogeneous families.
    """

    def __init__(
        self,
        n_states: int,
        schedule: PhaseSchedule,
        emission_estimator: Any,
        len_estimator: Any = None,
        pseudo_count: float = 1e-8,
        name: str | None = None,
        *,
        emission_estimators: Sequence[Sequence[Any]] | None = None,
    ) -> None:
        self.n_states = _exact_integer(n_states, "ScheduledHMMEstimator n_states", minimum=1)
        self.schedule = _validated_schedule(schedule)
        if emission_estimators is None:
            if emission_estimator is None:
                raise ValueError("ScheduledHMMEstimator requires an emission estimator or estimator grid.")
            self.emission_estimators = [
                [emission_estimator for _ in range(self.n_states)] for _ in range(self.schedule.n_phases)
            ]
        else:
            if emission_estimator is not None:
                raise ValueError("provide either emission_estimator or emission_estimators, not both.")
            self.emission_estimators = _owned_emission_grid(
                emission_estimators,
                self.schedule.n_phases,
                self.n_states,
                "ScheduledHMMEstimator emission estimators",
            )
        # Retain the historical attribute for callers that introspect a homogeneous estimator.
        self.emission_estimator = emission_estimator
        self.len_estimator = len_estimator
        self.pseudo_count = _validated_pseudo_count(pseudo_count)
        self.name = name

    def accumulator_factory(self) -> ScheduledHMMAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return ScheduledHMMAccumulatorFactory(
            self.n_states,
            self.schedule,
            self.emission_estimators,
            self.len_estimator,
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> ScheduledHiddenMarkovModelDistribution:
        """Estimate phase-indexed initial, transition, emission, and length models."""
        ic, tc, ec, em_vals, lv = validated_statistic_tuple(
            suff_stat,
            5,
            "scheduled-HMM sufficient statistics",
        )
        expected_initial = (self.schedule.n_phases, self.n_states)
        expected_transition = (self.schedule.n_phases, self.n_states, self.n_states)
        expected_emission = expected_initial
        ic = validated_count_array(ic, expected_initial, "scheduled-HMM initial counts")
        tc = validated_count_array(tc, expected_transition, "scheduled-HMM transition counts")
        ec = validated_count_array(ec, expected_emission, "scheduled-HMM emission counts")
        if not np.isclose(float(ec.sum()), float(ic.sum() + tc.sum()), rtol=1.0e-9, atol=1.0e-9):
            raise ValueError("scheduled-HMM emission counts must equal initial plus transition mass")
        validate_effective_sample_mass(
            nobs,
            float(ic.sum()),
            label="scheduled-HMM effective sample",
            allow_unassigned=True,
        )
        em_vals = _owned_emission_grid(
            em_vals,
            self.schedule.n_phases,
            self.n_states,
            "ScheduledHMMEstimator emission sufficient statistics",
        )
        pc = self.pseudo_count
        inits = ic + pc
        initial_sums = inits.sum(axis=1, keepdims=True)
        empty_initial = initial_sums[:, 0] == 0.0
        inits = np.divide(inits, initial_sums, out=np.zeros_like(inits), where=initial_sums > 0.0)
        inits[empty_initial, :] = 1.0 / self.n_states
        trans = tc + pc
        rsum = trans.sum(axis=2, keepdims=True)
        empty_transition = rsum[:, :, 0] == 0.0
        trans = np.divide(trans, rsum, out=np.zeros_like(trans), where=rsum > 0.0)
        for phase, state in np.argwhere(empty_transition):
            trans[phase, state, state] = 1.0
        emissions = [
            [self.emission_estimators[p][j].estimate(float(ec[p, j]), em_vals[p][j]) for j in range(self.n_states)]
            for p in range(self.schedule.n_phases)
        ]
        len_dist = None if self.len_estimator is None else self.len_estimator.estimate(nobs, lv)
        return ScheduledHiddenMarkovModelDistribution(inits, trans, emissions, self.schedule, len_dist, self.name)
