"""Hidden Markov model over arbitrary emission segments.

Each hidden state emits one *segment* object. The segment can be any data type
accepted by that state's emission distribution: a scalar, tuple, set, sequence,
or another composable mixle.stats object. To model variable-length emissions,
use a ``SequenceDistribution`` (or any other distribution over sequences) as an
emission distribution.

Unlike ``HiddenMarkovModelDistribution``, each state keeps its own encoder, so
emission distributions may use different distribution classes as long as they
can all score the same raw segment observations.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.sequences.markov_chain import MarkovChainDistribution
from mixle.utils.aliasing import MISSING, broadcast_pseudo_count, coalesce_alias, require


def _forward_log(log_w: np.ndarray, log_a: np.ndarray, log_emit: np.ndarray) -> float:
    """Return log p(emissions) for one segment sequence."""
    n = log_emit.shape[0]
    if n == 0:
        return 0.0
    alpha = log_w + log_emit[0]
    for t in range(1, n):
        alpha = log_emit[t] + logsumexp(alpha[:, None] + log_a, axis=0)
    return float(logsumexp(alpha))


def _forward_backward(
    log_w: np.ndarray, log_a: np.ndarray, log_emit: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
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

    def compute_capabilities(self):
        """Describe backend support shared by emissions and optional length model."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.emissions)
        if not self.null_len_dist:
            children = children + (self.len_dist,)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def compute_declaration(self):
        """Return a composite compute declaration for the segmental HMM."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        emission_children = tuple(declaration_for(emission) for emission in self.emissions)
        length = None if self.null_len_dist else declaration_for(self.len_dist)
        children = tuple(
            child for child in emission_children + ((length,) if length is not None else ()) if child is not None
        )
        roles = tuple("state_%d_emission" % i for i, child in enumerate(emission_children) if child is not None)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="segmental_hidden_markov",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("w", constraint="simplex_vector"),
                ParameterSpec("transitions", constraint="row_simplex_matrix"),
            ),
            statistics=(
                StatisticSpec("num_states", kind="metadata", additive=False, scales=False),
                StatisticSpec("initial_counts"),
                StatisticSpec("state_counts"),
                StatisticSpec("transition_counts"),
                StatisticSpec("emissions", kind="tuple"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="segmental_hidden_state_sequence",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __init__(
        self,
        emissions: Sequence[SequenceEncodableProbabilityDistribution],
        w: Sequence[float] | np.ndarray = MISSING,
        transitions: Sequence[Sequence[float]] | np.ndarray = MISSING,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        weights: Sequence[float] | np.ndarray = MISSING,
        terminal_states: set[int] | Sequence[int] | None = None,
    ) -> None:
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        transitions = require("transitions", transitions, default=MISSING)
        self.emissions = list(emissions)
        self.n_states = len(self.emissions)
        self.w = np.asarray(w, dtype=np.float64)
        if self.w.shape[0] != self.n_states:
            raise ValueError("initial probability vector length must match number of emissions.")
        self.w = self.w / self.w.sum()
        self.transitions = np.asarray(transitions, dtype=np.float64).reshape((self.n_states, self.n_states))
        row_sum = self.transitions.sum(axis=1, keepdims=True)
        bad = row_sum.flatten() <= 0.0
        if np.any(bad):
            self.transitions[bad, :] = 1.0 / float(self.n_states)
            row_sum = self.transitions.sum(axis=1, keepdims=True)
        self.transitions = self.transitions / row_sum
        with np.errstate(divide="ignore"):
            self.log_w = np.log(self.w)
            self.log_transitions = np.log(self.transitions)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.null_len_dist = supports(self.len_dist, Neutral)
        self.name = name
        self.terminal_states = None if terminal_states is None else set(int(s) for s in terminal_states)
        if self.terminal_states is not None:
            self._terminal_mask = np.zeros(self.n_states, dtype=bool)
            self._terminal_mask[list(self.terminal_states)] = True

    def __str__(self) -> str:
        s1 = ",".join(str(u) for u in self.emissions)
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        return "SegmentalHiddenMarkovModelDistribution([%s], %s, %s, len_dist=%s, name=%s)" % (
            s1,
            s2,
            s3,
            str(self.len_dist),
            repr(self.name),
        )

    @property
    def topics(self) -> list[SequenceEncodableProbabilityDistribution]:
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
        if self.terminal_states is not None:
            from mixle.stats.latent.hidden_markov import terminal_forward_loglik

            return terminal_forward_loglik(self.log_w, self.log_transitions, log_emit, self._terminal_mask)
        rv = _forward_log(self.log_w, self.log_transitions, log_emit)
        if not self.null_len_dist:
            rv += self.len_dist.log_density(n)
        return rv

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, tuple[Any, ...], Any | None]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        idx, sz, enc_by_state, len_enc = x
        nseq = len(sz)
        rv = np.zeros(nseq, dtype=np.float64)
        total = len(idx)
        log_emit = np.empty((total, self.n_states), dtype=np.float64)
        for k, dist in enumerate(self.emissions):
            log_emit[:, k] = dist.seq_log_density(enc_by_state[k])

        offsets = np.concatenate([[0], np.cumsum(sz)]).astype(int)
        if self.terminal_states is not None:
            from mixle.stats.latent.hidden_markov import terminal_forward_loglik

            for i in range(nseq):
                rv[i] = terminal_forward_loglik(
                    self.log_w, self.log_transitions, log_emit[offsets[i] : offsets[i + 1]], self._terminal_mask
                )
            return rv
        for i in range(nseq):
            rv[i] = _forward_log(self.log_w, self.log_transitions, log_emit[offsets[i] : offsets[i + 1]])

        if not self.null_len_dist and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)
        return rv

    def backend_seq_log_density(
        self, x: tuple[np.ndarray, np.ndarray, tuple[Any, ...], Any | None], engine: Any
    ) -> Any:
        """Engine-neutral segmental-HMM forward scoring for encoded segment sequences."""
        from mixle.stats.compute.backend import backend_seq_log_density

        idx, sz, enc_by_state, len_enc = x
        nseq = len(sz)
        total = len(idx)
        rv = engine.zeros(nseq)

        if total > 0:
            emission_scores = [
                backend_seq_log_density(dist, enc_by_state[k], engine) for k, dist in enumerate(self.emissions)
            ]
            log_emit = engine.stack(emission_scores, axis=1)
            log_w = engine.asarray(self.log_w)
            log_a = engine.asarray(self.log_transitions)
            offsets = np.concatenate([[0], np.cumsum(sz)]).astype(int)

            for i, n in enumerate(sz):
                if n == 0:
                    continue
                start = offsets[i]
                alpha = log_w + log_emit[start]
                for t in range(1, int(n)):
                    alpha = log_emit[start + t] + engine.logsumexp(alpha[:, None] + log_a, axis=0)
                rv[i] = engine.logsumexp(alpha, axis=0)

        if not self.null_len_dist and len_enc is not None:
            rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)
        return rv

    def sampler(self, seed: int | None = None) -> "SegmentalHiddenMarkovSampler":
        """Return a sampler for drawing observations from this distribution."""
        if self.null_len_dist and self.terminal_states is None:
            raise ValueError("SegmentalHiddenMarkovSampler requires a non-null len_dist or terminal_states.")
        return SegmentalHiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SegmentalHiddenMarkovEstimator":
        """Return an estimator for fitting this distribution from data."""
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        ests = [u.estimator(pseudo_count=pseudo_count) for u in self.emissions]
        return SegmentalHiddenMarkovEstimator(
            ests,
            len_estimator=len_est,
            pseudo_count=(pseudo_count, pseudo_count),
            name=self.name,
            terminal_states=self.terminal_states,
        )

    def dist_to_encoder(self) -> "SegmentalHiddenMarkovDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return SegmentalHiddenMarkovDataEncoder(
            [d.dist_to_encoder() for d in self.emissions], self.len_dist.dist_to_encoder()
        )

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate segment sequences in descending marginal probability order.

        The segmental HMM has the *standard* HMM forward semantics -- each position emits one segment
        from its state's distribution, scored independently -- so it reuses
        :class:`HiddenMarkovModelEnumerator` directly via its per-state emission (``topics``),
        ``log_w``, ``log_transitions``, and ``len_dist``. Each segment is drawn from the union of the
        per-state emission supports, so every emission distribution must itself support enumeration
        (and a length distribution must be modeled).
        """
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelEnumerator

        return HiddenMarkovModelEnumerator(
            self,
            topics=self.emissions,
            log_w=self.log_w,
            log_transitions=self.log_transitions,
            len_dist=self.len_dist,
            path_root="SegmentalHiddenMarkovModelDistribution",
        )


class SegmentalHiddenMarkovSampler(DistributionSampler):
    """Draw iid segmental-HMM observations."""

    def __init__(self, dist: SegmentalHiddenMarkovModelDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.obs_samplers = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in dist.emissions]
        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        p_map = {i: dist.w[i] for i in range(dist.n_states)}
        t_map = {i: {j: dist.transitions[i, j] for j in range(dist.n_states)} for i in range(dist.n_states)}
        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None) -> list[Any] | list[list[Any]]:
        """Draw one segment sequence, or ``size`` iid segment sequences."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        if self.dist.terminal_states is not None:
            z = int(self.state_sampler.sample_seq())
            states = [z]
            while z not in self.dist.terminal_states and len(states) < 1_000_000:
                z = int(self.state_sampler.sample_seq(v0=z))
                states.append(z)
            return [self.obs_samplers[s].sample() for s in states]
        n = self.len_sampler.sample()
        states = self.state_sampler.sample_seq(n)
        return [self.obs_samplers[s].sample() for s in states]


class SegmentalHiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):
    """Baum-Welch accumulator for segmental HMMs."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        self.accumulators = list(accumulators)
        self.num_states = len(self.accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.init_key, self.trans_key, self.state_key = keys if keys is not None else (None, None, None)
        self.name = name

        # When _track_ll is enabled, seq_update accumulates the per-sequence data
        # log-likelihood into _seq_ll. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); default path is unchanged and zero-cost.
        self._track_ll = False
        self._seq_ll = 0.0

        self._init_rng = False
        self._state_rng: RandomState | None = None
        self._len_rng: RandomState | None = None
        self._acc_rng: list[RandomState] | None = None

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(maxrandint, size=2 + self.num_states)
        self._state_rng = RandomState(seeds[0])
        self._len_rng = RandomState(seeds[1])
        self._acc_rng = [RandomState(seeds[i + 2]) for i in range(self.num_states)]
        self._init_rng = True

    def update(self, x: Sequence[Any], weight: float, estimate: SegmentalHiddenMarkovModelDistribution) -> None:
        """Update sufficient statistics for one observed segment sequence."""
        self.seq_update(estimate.dist_to_encoder().seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState) -> None:
        """Randomly initialize sufficient statistics for one segment sequence."""
        if not self._init_rng:
            self._rng_initialize(rng)
        enc = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(enc, np.asarray([weight]), rng)

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, tuple[Any, ...], Any | None], weights: np.ndarray, rng: RandomState
    ) -> None:
        """Randomly initialize state and emission statistics for encoded sequences."""
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
            self.state_counts += np.bincount(
                states[start:stop], weights=np.full(n, weights[i]), minlength=self.num_states
            )
            for t in range(start, stop - 1):
                self.trans_counts[states[t], states[t + 1]] += weights[i]

        for k in range(self.num_states):
            self.accumulators[k].seq_initialize(enc_by_state[k], weighted_state[:, k], self._acc_rng[k])
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, tuple[Any, ...], Any | None],
        weights: np.ndarray,
        estimate: SegmentalHiddenMarkovModelDistribution,
    ) -> None:
        """Update encoded-sequence statistics with Baum-Welch posteriors."""
        idx, sz, enc_by_state, len_enc = x
        total = len(idx)
        log_emit = np.empty((total, self.num_states), dtype=np.float64)
        for k, dist in enumerate(estimate.emissions):
            log_emit[:, k] = dist.seq_log_density(enc_by_state[k])

        # When the fused-EM fast path requests it, accumulate the per-sequence data
        # log-likelihood from the existing forward pass (ll returned by _forward_backward,
        # which equals seq_log_density's _forward_log term exactly). The standard path skips it.
        track_ll = self._track_ll
        ll_ret = np.zeros(len(sz), dtype=np.float64) if track_ll else None

        terminal = estimate.terminal_states is not None
        if terminal:
            from mixle.stats.latent.hidden_markov import terminal_forward_backward

        offsets = np.concatenate([[0], np.cumsum(sz)]).astype(int)
        gamma_all = np.zeros((total, self.num_states), dtype=np.float64)
        for i, n in enumerate(sz):
            if n == 0:
                continue
            start, stop = offsets[i], offsets[i + 1]
            if terminal:
                ll_i, gamma, xi = terminal_forward_backward(
                    estimate.log_w, estimate.log_transitions, log_emit[start:stop], estimate._terminal_mask
                )
                if gamma is None:
                    continue
                xi_sum = xi.sum(axis=0)
            else:
                ll_i, gamma, xi_sum = _forward_backward(estimate.log_w, estimate.log_transitions, log_emit[start:stop])
            if track_ll:
                ll_ret[i] = ll_i
            w = weights[i]
            gamma_w = gamma * w
            gamma_all[start:stop, :] = gamma_w
            self.init_counts += gamma_w[0]
            self.state_counts += gamma_w.sum(axis=0)
            self.trans_counts += xi_sum * w

        if track_ll:
            if not estimate.null_len_dist and len_enc is not None:
                ll_ret = ll_ret + estimate.len_dist.seq_log_density(len_enc)
            self._seq_ll += float(np.dot(weights, ll_ret))

        for k in range(self.num_states):
            self.accumulators[k].seq_update(enc_by_state[k], gamma_all[:, k], estimate.emissions[k])
        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident Baum-Welch E-step (numpy or torch).

        Reuses hmm_engine_forward_backward over the sequence-contiguous segment encoding, producing
        the same initial/state/transition/emission statistics as the host seq_update.
        """
        from mixle.stats.latent.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

        idx, sz, enc_by_state, len_enc = x
        sz = np.asarray(sz)
        total = len(idx)
        num_states = self.num_states

        log_emit_flat = np.empty((total, num_states), dtype=np.float64)
        for k, dist in enumerate(estimate.emissions):
            log_emit_flat[:, k] = dist.seq_log_density(enc_by_state[k])

        padded, mask, offsets = hmm_pad_log_emissions(log_emit_flat, sz)
        with np.errstate(divide="ignore"):
            log_w = estimate.log_w
            log_a = estimate.log_transitions
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        _, gamma, xi_sum, pi = hmm_engine_forward_backward(engine, padded, log_w, log_a, mask, weights=weights_np)
        gamma = np.asarray(engine.to_numpy(gamma))
        xi_sum = np.asarray(engine.to_numpy(xi_sum))
        pi = np.asarray(engine.to_numpy(pi))

        gamma_flat = np.zeros((total, num_states), dtype=np.float64)
        for i in range(len(sz)):
            n = int(sz[i])
            if n > 0:
                gamma_flat[offsets[i] : offsets[i + 1], :] = gamma[i, :n, :]

        self.init_counts += pi.sum(axis=0)
        self.state_counts += gamma_flat.sum(axis=0)
        self.trans_counts += xi_sum
        for k in range(num_states):
            self.accumulators[k].seq_update(enc_by_state[k], gamma_flat[:, k], estimate.emissions[k])
        self.len_accumulator.seq_update(len_enc, weights_np, estimate.len_dist)

    def combine(
        self, suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Any | None]
    ) -> "SegmentalHiddenMarkovAccumulator":
        """Merge another segmental-HMM sufficient-statistic value."""
        _, init_counts, state_counts, trans_counts, acc_values, len_value = suff_stat
        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts
        for k, value in enumerate(acc_values):
            self.accumulators[k].combine(value)
        if len_value is not None:
            self.len_accumulator.combine(len_value)
        return self

    def value(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...], Any | None]:
        """Return transition, emission, and sequence-length sufficient statistics."""
        return (
            self.num_states,
            self.init_counts,
            self.state_counts,
            self.trans_counts,
            tuple(a.value() for a in self.accumulators),
            self.len_accumulator.value(),
        )

    def from_value(
        self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Any | None]
    ) -> "SegmentalHiddenMarkovAccumulator":
        """Replace this accumulator from serialized sufficient statistics."""
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

    def scale(self, c: float) -> "SegmentalHiddenMarkovAccumulator":
        """Scale all weight-linear sufficient statistics by ``c``."""
        self.init_counts *= c
        self.state_counts *= c
        self.trans_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed initial, transition, emission, and length statistics."""
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

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed initial, transition, emission, and length statistics."""
        if self.init_key is not None and self.init_key in stats_dict:
            self.init_counts = stats_dict[self.init_key]
        if self.trans_key is not None and self.trans_key in stats_dict:
            self.trans_counts = stats_dict[self.trans_key]
        if self.state_key is not None and self.state_key in stats_dict:
            self.accumulators = stats_dict[self.state_key]
        for acc in self.accumulators:
            acc.key_replace(stats_dict)
        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "SegmentalHiddenMarkovDataEncoder":
        """Return an encoder compatible with the emission and length accumulators."""
        return SegmentalHiddenMarkovDataEncoder(
            [a.acc_to_encoder() for a in self.accumulators], self.len_accumulator.acc_to_encoder()
        )


class SegmentalHiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for SegmentalHiddenMarkovAccumulator."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        self.factories = list(factories)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys
        self.name = name

    def make(self) -> SegmentalHiddenMarkovAccumulator:
        """Create a fresh segmental-HMM accumulator."""
        return SegmentalHiddenMarkovAccumulator(
            [f.make() for f in self.factories], self.len_factory.make(), keys=self.keys, name=self.name
        )


class SegmentalHiddenMarkovEstimator(ParameterEstimator):
    """Baum-Welch estimator for SegmentalHiddenMarkovModelDistribution."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        len_estimator: ParameterEstimator | None = NullEstimator(),
        pseudo_count: float | tuple[float | None, float | None] | None = (None, None),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        terminal_states: set[int] | Sequence[int] | None = None,
    ) -> None:
        self.estimators = list(estimators)
        self.num_states = len(self.estimators)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys
        self.name = name
        self.terminal_states = terminal_states

    def accumulator_factory(self) -> SegmentalHiddenMarkovAccumulatorFactory:
        """Return an accumulator factory for Baum-Welch sufficient statistics."""
        return SegmentalHiddenMarkovAccumulatorFactory(
            [e.accumulator_factory() for e in self.estimators],
            self.len_estimator.accumulator_factory(),
            keys=self.keys,
            name=self.name,
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Any | None],
    ) -> SegmentalHiddenMarkovModelDistribution:
        """Estimate initial, transition, emission, and length distributions."""
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
            emissions, w, transitions, len_dist=len_dist, name=self.name, terminal_states=self.terminal_states
        )


class SegmentalHiddenMarkovDataEncoder(DataSequenceEncoder):
    """Encode a batch of segment sequences for a segmental HMM."""

    def __init__(
        self,
        emission_encoders: Sequence[DataSequenceEncoder],
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
    ) -> None:
        self.emission_encoders = list(emission_encoders)
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        return "SegmentalHiddenMarkovDataEncoder([%s], len_encoder=%s)" % (
            ",".join(str(e) for e in self.emission_encoders),
            str(self.len_encoder),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SegmentalHiddenMarkovDataEncoder)
            and self.emission_encoders == other.emission_encoders
            and self.len_encoder == other.len_encoder
        )

    def seq_encode(self, x: Sequence[Sequence[Any]]) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...], Any | None]:
        """Flatten segment sequences and encode them for every state emission."""
        lengths = np.asarray([len(seq) for seq in x], dtype=np.int32)
        idx = np.repeat(np.arange(len(x), dtype=np.int32), lengths)
        flat: list[Any] = []
        for seq in x:
            flat.extend(seq)
        enc_by_state = tuple(enc.seq_encode(flat) for enc in self.emission_encoders)
        len_enc = None if supports(self.len_encoder, Neutral) else self.len_encoder.seq_encode(lengths)
        return idx, lengths, enc_by_state, len_enc


SegmentalHiddenMarkovDistribution = SegmentalHiddenMarkovModelDistribution

# --- Backward-compatible API naming aliases ---
SegmentalHiddenMarkovModelAccumulator = SegmentalHiddenMarkovAccumulator
SegmentalHiddenMarkovModelAccumulatorFactory = SegmentalHiddenMarkovAccumulatorFactory
SegmentalHiddenMarkovModelDataEncoder = SegmentalHiddenMarkovDataEncoder
SegmentalHiddenMarkovModelEstimator = SegmentalHiddenMarkovEstimator
SegmentalHiddenMarkovModelSampler = SegmentalHiddenMarkovSampler


def _register_segmental_engine_kernel():
    """Register the engine-resident segmental-HMM kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class SegmentalHiddenMarkovModelKernel(GenericKernel):
        """Segmental-HMM kernel whose E-step runs the forward-backward on the active engine."""

        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("SegmentalHiddenMarkovModelKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class SegmentalHiddenMarkovModelKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return SegmentalHiddenMarkovModelKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(SegmentalHiddenMarkovModelDistribution, SegmentalHiddenMarkovModelKernelFactory())


_register_segmental_engine_kernel()
