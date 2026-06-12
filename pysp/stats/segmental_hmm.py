"""Hidden Markov model over arbitrary emission segments.

Each hidden state emits one *segment* object. The segment can be any data type
accepted by that state's emission distribution: a scalar, tuple, set, sequence,
or another composable pysp.stats object. To model variable-length emissions,
use a ``SequenceDistribution`` (or any other distribution over sequences) as an
emission distribution.

Unlike ``HiddenMarkovModelDistribution``, each state keeps its own encoder, so
emission distributions may use different distribution classes as long as they
can all score the same raw segment observations.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from pysp.arithmetic import maxrandint
from pysp.stats.markovchain import MarkovChainDistribution
from pysp.stats.null_dist import NullAccumulator, NullAccumulatorFactory, NullDataEncoder, \
    NullDistribution, NullEstimator
from pysp.stats.pdist import DataSequenceEncoder, DistributionSampler, ParameterEstimator, \
    SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    StatisticAccumulatorFactory


def _forward_log(log_w: np.ndarray, log_a: np.ndarray, log_emit: np.ndarray) -> float:
    """Return log p(emissions) for one segment sequence."""
    n = log_emit.shape[0]
    if n == 0:
        return 0.0
    alpha = log_w + log_emit[0]
    for t in range(1, n):
        alpha = log_emit[t] + logsumexp(alpha[:, None] + log_a, axis=0)
    return float(logsumexp(alpha))


def _forward_backward(log_w: np.ndarray, log_a: np.ndarray,
                      log_emit: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Forward/backward posterior probabilities for one segment sequence.

    Returns (log_likelihood, gamma, xi_sum), where gamma[t, k] is p(z_t=k|x)
    and xi_sum[i, j] is sum_t p(z_t=i, z_{t+1}=j|x).
    """
    n, k = log_emit.shape
    gamma = np.zeros((n, k), dtype=np.float64)
    xi_sum = np.zeros((k, k), dtype=np.float64)
    if n == 0:
        return 0.0, gamma, xi_sum

    alpha = np.empty((n, k), dtype=np.float64)
    beta = np.zeros((n, k), dtype=np.float64)
    alpha[0] = log_w + log_emit[0]
    for t in range(1, n):
        alpha[t] = log_emit[t] + logsumexp(alpha[t - 1][:, None] + log_a, axis=0)

    ll = float(logsumexp(alpha[-1]))
    if not np.isfinite(ll):
        gamma.fill(1.0 / float(k))
        if n > 1:
            xi_sum.fill(float(n - 1) / float(k * k))
        return ll, gamma, xi_sum

    for t in range(n - 2, -1, -1):
        beta[t] = logsumexp(log_a + log_emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)

    gamma = np.exp(alpha + beta - ll)
    for t in range(n - 1):
        xi = alpha[t][:, None] + log_a + log_emit[t + 1][None, :] + beta[t + 1][None, :] - ll
        xi_sum += np.exp(xi)

    return ll, gamma, xi_sum


class SegmentalHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """HMM whose states emit arbitrary segment-valued distributions.

    Observations are lists of segment objects. For example, with
    ``SequenceDistribution(GaussianDistribution(...), len_dist=...)`` as an
    emission, each state emits a variable-length list of real values.
    """

    def __init__(self,
                 emissions: Sequence[SequenceEncodableProbabilityDistribution],
                 w: Union[Sequence[float], np.ndarray],
                 transitions: Union[Sequence[Sequence[float]], np.ndarray],
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None) -> None:
        self.emissions = list(emissions)
        self.n_states = len(self.emissions)
        self.w = np.asarray(w, dtype=np.float64)
        if self.w.shape[0] != self.n_states:
            raise ValueError('initial probability vector length must match number of emissions.')
        self.w = self.w / self.w.sum()
        self.transitions = np.asarray(transitions, dtype=np.float64).reshape((self.n_states, self.n_states))
        row_sum = self.transitions.sum(axis=1, keepdims=True)
        bad = row_sum.flatten() <= 0.0
        if np.any(bad):
            self.transitions[bad, :] = 1.0 / float(self.n_states)
            row_sum = self.transitions.sum(axis=1, keepdims=True)
        self.transitions = self.transitions / row_sum
        with np.errstate(divide='ignore'):
            self.log_w = np.log(self.w)
            self.log_transitions = np.log(self.transitions)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.null_len_dist = isinstance(self.len_dist, NullDistribution)
        self.name = name

    def __str__(self) -> str:
        s1 = ','.join(str(u) for u in self.emissions)
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        return 'SegmentalHiddenMarkovModelDistribution([%s], %s, %s, len_dist=%s, name=%s)' % (
            s1, s2, s3, str(self.len_dist), repr(self.name))

    @property
    def topics(self) -> List[SequenceEncodableProbabilityDistribution]:
        """Compatibility alias with HiddenMarkovModelDistribution terminology."""
        return self.emissions

    def density(self, x: Sequence[Any]) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Sequence[Any]) -> float:
        """Return the log-density or log-mass at a single observation."""
        n = len(x)
        if n == 0:
            return self.len_dist.log_density(0) if not self.null_len_dist else 0.0
        log_emit = np.empty((n, self.n_states), dtype=np.float64)
        for k, dist in enumerate(self.emissions):
            log_emit[:, k] = np.asarray([dist.log_density(xx) for xx in x], dtype=np.float64)
        rv = _forward_log(self.log_w, self.log_transitions, log_emit)
        if not self.null_len_dist:
            rv += self.len_dist.log_density(n)
        return rv

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray, Tuple[Any, ...], Optional[Any]]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        idx, sz, enc_by_state, len_enc = x
        nseq = len(sz)
        rv = np.zeros(nseq, dtype=np.float64)
        total = len(idx)
        log_emit = np.empty((total, self.n_states), dtype=np.float64)
        for k, dist in enumerate(self.emissions):
            log_emit[:, k] = dist.seq_log_density(enc_by_state[k])

        offsets = np.concatenate([[0], np.cumsum(sz)]).astype(int)
        for i in range(nseq):
            rv[i] = _forward_log(self.log_w, self.log_transitions, log_emit[offsets[i]:offsets[i + 1]])

        if not self.null_len_dist and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'SegmentalHiddenMarkovSampler':
        """Return a sampler for drawing observations from this distribution."""
        if self.null_len_dist:
            raise ValueError('SegmentalHiddenMarkovSampler requires a non-null len_dist.')
        return SegmentalHiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SegmentalHiddenMarkovEstimator':
        """Return an estimator for fitting this distribution from data."""
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        ests = [u.estimator(pseudo_count=pseudo_count) for u in self.emissions]
        return SegmentalHiddenMarkovEstimator(
            ests, len_estimator=len_est, pseudo_count=(pseudo_count, pseudo_count), name=self.name)

    def dist_to_encoder(self) -> 'SegmentalHiddenMarkovDataEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        return SegmentalHiddenMarkovDataEncoder(
            [d.dist_to_encoder() for d in self.emissions], self.len_dist.dist_to_encoder())


class SegmentalHiddenMarkovSampler(DistributionSampler):
    """Draw iid segmental-HMM observations."""

    def __init__(self, dist: SegmentalHiddenMarkovModelDistribution, seed: Optional[int] = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.obs_samplers = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in dist.emissions]
        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        p_map = {i: dist.w[i] for i in range(dist.n_states)}
        t_map = {i: {j: dist.transitions[i, j] for j in range(dist.n_states)}
                 for i in range(dist.n_states)}
        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> Union[List[Any], List[List[Any]]]:
        if size is not None:
            return [self.sample() for _ in range(size)]
        n = self.len_sampler.sample()
        states = self.state_sampler.sample_seq(n)
        return [self.obs_samplers[s].sample() for s in states]


class SegmentalHiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):
    """Baum-Welch accumulator for segmental HMMs."""

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        self.accumulators = list(accumulators)
        self.num_states = len(self.accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.init_key, self.trans_key, self.state_key = keys if keys is not None else (None, None, None)
        self.name = name
        self._init_rng = False
        self._state_rng: Optional[RandomState] = None
        self._len_rng: Optional[RandomState] = None
        self._acc_rng: Optional[List[RandomState]] = None

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(maxrandint, size=2 + self.num_states)
        self._state_rng = RandomState(seeds[0])
        self._len_rng = RandomState(seeds[1])
        self._acc_rng = [RandomState(seeds[i + 2]) for i in range(self.num_states)]
        self._init_rng = True

    def update(self, x: Sequence[Any], weight: float, estimate: SegmentalHiddenMarkovModelDistribution) -> None:
        self.seq_update(estimate.dist_to_encoder().seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)
        enc = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(enc, np.asarray([weight]), rng)

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray, Tuple[Any, ...], Optional[Any]],
                       weights: np.ndarray, rng: RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)
        idx, sz, enc_by_state, len_enc = x
        total = len(idx)
        states = self._state_rng.choice(self.num_states, size=total)
        offsets = np.concatenate([[0], np.cumsum(sz)]).astype(int)
        weighted_state = np.zeros((total, self.num_states), dtype=np.float64)

        for i, n in enumerate(sz):
            if n == 0:
                continue
            start, stop = offsets[i], offsets[i + 1]
            weighted_state[start:stop, :] = 0.0
            weighted_state[np.arange(start, stop), states[start:stop]] = weights[i]
            self.init_counts[states[start]] += weights[i]
            self.state_counts += np.bincount(states[start:stop], weights=np.full(n, weights[i]),
                                             minlength=self.num_states)
            for t in range(start, stop - 1):
                self.trans_counts[states[t], states[t + 1]] += weights[i]

        for k in range(self.num_states):
            self.accumulators[k].seq_initialize(enc_by_state[k], weighted_state[:, k], self._acc_rng[k])
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray, Tuple[Any, ...], Optional[Any]],
                   weights: np.ndarray, estimate: SegmentalHiddenMarkovModelDistribution) -> None:
        idx, sz, enc_by_state, len_enc = x
        total = len(idx)
        log_emit = np.empty((total, self.num_states), dtype=np.float64)
        for k, dist in enumerate(estimate.emissions):
            log_emit[:, k] = dist.seq_log_density(enc_by_state[k])

        offsets = np.concatenate([[0], np.cumsum(sz)]).astype(int)
        gamma_all = np.zeros((total, self.num_states), dtype=np.float64)
        for i, n in enumerate(sz):
            if n == 0:
                continue
            start, stop = offsets[i], offsets[i + 1]
            _, gamma, xi_sum = _forward_backward(
                estimate.log_w, estimate.log_transitions, log_emit[start:stop])
            w = weights[i]
            gamma_w = gamma * w
            gamma_all[start:stop, :] = gamma_w
            self.init_counts += gamma_w[0]
            self.state_counts += gamma_w.sum(axis=0)
            self.trans_counts += xi_sum * w

        for k in range(self.num_states):
            self.accumulators[k].seq_update(enc_by_state[k], gamma_all[:, k], estimate.emissions[k])
        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(self, suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Optional[Any]]) \
            -> 'SegmentalHiddenMarkovAccumulator':
        _, init_counts, state_counts, trans_counts, acc_values, len_value = suff_stat
        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts
        for k, value in enumerate(acc_values):
            self.accumulators[k].combine(value)
        if len_value is not None:
            self.len_accumulator.combine(len_value)
        return self

    def value(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Tuple[Any, ...], Optional[Any]]:
        return (self.num_states, self.init_counts, self.state_counts, self.trans_counts,
                tuple(a.value() for a in self.accumulators), self.len_accumulator.value())

    def from_value(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Optional[Any]]) \
            -> 'SegmentalHiddenMarkovAccumulator':
        num_states, init_counts, state_counts, trans_counts, acc_values, len_value = x
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts
        for k, value in enumerate(acc_values):
            self.accumulators[k].from_value(value)
        if len_value is not None:
            self.len_accumulator.from_value(len_value)
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.init_key is not None:
            stats_dict[self.init_key] = stats_dict.get(self.init_key, 0.0) + self.init_counts
        if self.trans_key is not None:
            stats_dict[self.trans_key] = stats_dict.get(self.trans_key, 0.0) + self.trans_counts
        if self.state_key is not None:
            if self.state_key in stats_dict:
                for k, acc in enumerate(stats_dict[self.state_key]):
                    acc.combine(self.accumulators[k].value())
            else:
                stats_dict[self.state_key] = self.accumulators
        for acc in self.accumulators:
            acc.key_merge(stats_dict)
        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.init_key is not None and self.init_key in stats_dict:
            self.init_counts = stats_dict[self.init_key]
        if self.trans_key is not None and self.trans_key in stats_dict:
            self.trans_counts = stats_dict[self.trans_key]
        if self.state_key is not None and self.state_key in stats_dict:
            self.accumulators = stats_dict[self.state_key]
        for acc in self.accumulators:
            acc.key_replace(stats_dict)
        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'SegmentalHiddenMarkovDataEncoder':
        return SegmentalHiddenMarkovDataEncoder(
            [a.acc_to_encoder() for a in self.accumulators], self.len_accumulator.acc_to_encoder())


class SegmentalHiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for SegmentalHiddenMarkovAccumulator."""

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory],
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        self.factories = list(factories)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys
        self.name = name

    def make(self) -> SegmentalHiddenMarkovAccumulator:
        return SegmentalHiddenMarkovAccumulator(
            [f.make() for f in self.factories], self.len_factory.make(), keys=self.keys, name=self.name)


class SegmentalHiddenMarkovEstimator(ParameterEstimator):
    """Baum-Welch estimator for SegmentalHiddenMarkovModelDistribution."""

    def __init__(self, estimators: Sequence[ParameterEstimator],
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)) -> None:
        self.estimators = list(estimators)
        self.num_states = len(self.estimators)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> SegmentalHiddenMarkovAccumulatorFactory:
        return SegmentalHiddenMarkovAccumulatorFactory(
            [e.accumulator_factory() for e in self.estimators],
            self.len_estimator.accumulator_factory(), keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Optional[Any]]) \
            -> SegmentalHiddenMarkovModelDistribution:
        num_states, init_counts, state_counts, trans_counts, emission_ss, len_ss = suff_stat
        emissions = [self.estimators[k].estimate(state_counts[k], emission_ss[k]) for k in range(num_states)]
        len_dist = self.len_estimator.estimate(nobs, len_ss)

        if self.pseudo_count[0] is not None:
            w = init_counts + self.pseudo_count[0] / float(num_states)
        else:
            w = init_counts.copy()
        w = np.ones(num_states) / float(num_states) if w.sum() <= 0.0 else w / w.sum()

        if self.pseudo_count[1] is not None:
            transitions = trans_counts + self.pseudo_count[1] / float(num_states * num_states)
        else:
            transitions = trans_counts.copy()
        row_sum = transitions.sum(axis=1, keepdims=True)
        bad = row_sum.flatten() <= 0.0
        if np.any(bad):
            transitions[bad, :] = 1.0
            row_sum = transitions.sum(axis=1, keepdims=True)
        transitions = transitions / row_sum

        return SegmentalHiddenMarkovModelDistribution(
            emissions, w, transitions, len_dist=len_dist, name=self.name)


class SegmentalHiddenMarkovDataEncoder(DataSequenceEncoder):
    """Encode a batch of segment sequences for a segmental HMM."""

    def __init__(self, emission_encoders: Sequence[DataSequenceEncoder],
                 len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()) -> None:
        self.emission_encoders = list(emission_encoders)
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        return 'SegmentalHiddenMarkovDataEncoder([%s], len_encoder=%s)' % (
            ','.join(str(e) for e in self.emission_encoders), str(self.len_encoder))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, SegmentalHiddenMarkovDataEncoder)
                and self.emission_encoders == other.emission_encoders
                and self.len_encoder == other.len_encoder)

    def seq_encode(self, x: Sequence[Sequence[Any]]) \
            -> Tuple[np.ndarray, np.ndarray, Tuple[Any, ...], Optional[Any]]:
        lengths = np.asarray([len(seq) for seq in x], dtype=np.int32)
        idx = np.repeat(np.arange(len(x), dtype=np.int32), lengths)
        flat: List[Any] = []
        for seq in x:
            flat.extend(seq)
        enc_by_state = tuple(enc.seq_encode(flat) for enc in self.emission_encoders)
        len_enc = None if isinstance(self.len_encoder, NullDataEncoder) else self.len_encoder.seq_encode(lengths)
        return idx, lengths, enc_by_state, len_enc


SegmentalHiddenMarkovDistribution = SegmentalHiddenMarkovModelDistribution
