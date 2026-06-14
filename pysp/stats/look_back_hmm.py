"""Evaluate, estimate, and sample from a lookback hidden Markov model (typed rewrite).

Defines the LookbackHiddenMarkovDistribution, LookbackHiddenMarkovSampler,
LookbackHiddenMarkovEstimatorAccumulator, LookbackHiddenMarkovEstimatorAccumulatorFactory,
LookbackHiddenMarkovEstimator, and the LookbackHiddenMarkovDataEncoder classes for use with pysparkplug.

A lookback hidden Markov model is a hidden Markov model whose emission distributions condition on the
previous ``lag`` observations: with hidden states Z(t) following a Markov chain with initial state
probabilities w and transition matrix A,

    P(X(1),...,X(n)) = sum_z P(X(1:lag) | Z(1)=z_1) * w[z_1]
                       * prod_{t=lag+1}^{n} P(X(t) | X(t-lag:t-1), Z(t)=z_t) * A[z_{t-1}, z_t],

where the per-state ``topics`` distributions model windows x[t-lag:t+1] of length lag+1 (e.g.
IntegerMarkovChainDistribution), and the per-state ``init_dist`` distributions model the first ``lag``
observations. An optional length distribution models the number of hidden positions: len(x) - lag + 1
(initial segment plus emission windows) when lag > 0, and len(x) when lag == 0.

With lag == 0 the model reduces to an ordinary hidden Markov model: there is no initial segment,
``init_dist`` is never evaluated, the first state is drawn from w and emits the window x[0:1], and each
subsequent state emits x[t:t+1].

Data type: Sequence[T] - each observation is a sequence (e.g. a list) whose length-(lag+1) sliding
windows have the data type accepted by the topic distributions, and whose first ``lag`` entries have
the data type accepted by the initial distributions.

Note: This is the typed rewrite of the sibling module pysp.stats.lookback_hmm, which is the original
implementation kept stable for the example scripts and external users. The math is identical, but the two
modules differ slightly in their handling of optional arguments: this module substitutes Null*
objects (NullDistribution, NullEstimator, NullDataEncoder, ...) for an absent len_dist/init_dist,
while the sibling uses None (and omits the length term from densities). The
LookbackHiddenMarkovDataEncoder constructor signatures also differ (here: encoder first with an
``encoder`` attribute; sibling: lag first with a ``topic_encoder`` attribute).
"""

import math
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.arithmetic import maxrandint
from pysp.stats.markov_chain import MarkovChainDistribution
from pysp.stats.null_dist import (
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.aliasing import MISSING, coalesce_alias, require
from pysp.utils.optional_deps import numba

T = TypeVar("T")
E0 = TypeVar("E0")
E1 = TypeVar("E1")


class LookbackHiddenMarkovDistribution(SequenceEncodableProbabilityDistribution):
    """Hidden Markov model whose state emissions condition on the previous ``lag`` observations."""

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        w: np.ndarray = MISSING,
        transitions=MISSING,
        lag: int = 0,
        init_dist: Sequence[SequenceEncodableProbabilityDistribution] | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        weights: np.ndarray = MISSING,
    ) -> None:
        """LookbackHiddenMarkovDistribution object for sequences with lagged emission dependence.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Per-state emission
                distributions over windows x[t-lag:t+1] of length lag+1.
            w (np.ndarray): Initial state probabilities (sums to 1.0).
            transitions (Union[Sequence[Sequence[float]], np.ndarray]): Row-stochastic state
                transition matrix with shape (num_states, num_states).
            lag (int): Number of preceding observations each emission conditions on.
            init_dist (Optional[Sequence[SequenceEncodableProbabilityDistribution]]): Per-state
                distributions for the first ``lag`` observations x[:lag]. Defaults to a list of
                NullDistribution objects when None. Never evaluated when lag == 0.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the
                number of hidden positions (len(x) - lag + 1 when lag > 0, len(x) when lag == 0).
                Defaults to NullDistribution.
            name (Optional[str]): Assign string name to object instance.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Per-state emission distributions.
            init_dist (Sequence[SequenceEncodableProbabilityDistribution]): Per-state initial distributions.
            lag (int): Number of preceding observations each emission conditions on.
            num_topics (int): Number of topic distributions.
            num_states (int): Number of hidden states (length of w).
            w (np.ndarray): Initial state probabilities.
            log_w (np.ndarray): Log of w.
            transitions (np.ndarray): Transition matrix with shape (num_states, num_states).
            len_dist (SequenceEncodableProbabilityDistribution): Length distribution.
            name (Optional[str]): Name of object instance.

        """
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        transitions = require("transitions", transitions, default=MISSING)
        with np.errstate(divide="ignore"):
            self.topics = topics
            self.init_dist = init_dist if init_dist is not None else [NullDistribution()] * len(w)
            self.lag = lag
            self.num_topics = len(topics)
            self.num_states = len(w)
            self.w = vec.make(w)
            self.log_w = log(self.w)
            self.transitions = np.reshape(transitions, (self.num_states, self.num_states))
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.name = name

    def __str__(self) -> str:
        """Returns string representation of LookbackHiddenMarkovDistribution object."""
        s1 = ",".join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        s4 = repr(self.lag)
        s5 = ",".join(map(str, self.init_dist))
        s6 = str(self.len_dist)
        s7 = repr(self.name)

        return "LookbackHiddenMarkovDistribution([%s], %s, %s, lag=%s, init_dist=[%s], len_dist=%s, name=%s)" % (
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
        )

    def density(self, x):
        """Evaluate the density of the distribution at sequence x.

        Args:
            x (Sequence[T]): Observed sequence.

        Returns:
            float: Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x):
        """Evaluate the log-density of the distribution at sequence x.

        Marginalizes the hidden state path with a scaled forward pass. The initial segment x[:lag] is
        scored by init_dist, each window x[t-lag:t+1] by the topic distributions, and the number of
        hidden positions by len_dist. When lag == 0 there is no initial segment: the first state emits
        the window x[0:1] directly (ordinary HMM).

        Args:
            x (Sequence[T]): Observed sequence with len(x) >= lag.

        Returns:
            float: Log-density at x.

        """
        if x is None or len(x) == 0:
            if self.len_dist is not None:
                return self.len_dist.log_density(0)
            else:
                return 0.0

        log_w = self.log_w
        num_states = self.num_states
        comps = self.topics
        lag = self.lag
        init_comps = self.init_dist

        obs_log_likelihood = np.zeros(num_states, dtype=np.float64)
        obs_log_likelihood += log_w
        if lag > 0:
            for i in range(num_states):
                obs_log_likelihood[i] += init_comps[i].log_density(x[:lag])
        else:
            for i in range(num_states):
                obs_log_likelihood[i] += comps[i].log_density(x[0:1])

        if np.max(obs_log_likelihood) == -np.inf:
            return -np.inf

        max_ll = obs_log_likelihood.max()
        obs_log_likelihood -= max_ll
        np.exp(obs_log_likelihood, out=obs_log_likelihood)
        sum_ll = np.sum(obs_log_likelihood)
        retval = np.log(sum_ll) + max_ll

        for k in range(max(lag, 1), len(x)):
            #  P(Z(t+1) | Z(t) = i) P(Z(t) = i | X(t), X(t-1), ...)
            np.dot(self.transitions.T, obs_log_likelihood, out=obs_log_likelihood)
            obs_log_likelihood /= obs_log_likelihood.sum()

            # log P(Z(t+1) | X(t), X(t-1), ...)
            np.log(obs_log_likelihood, out=obs_log_likelihood)

            # log P(X(t+1) | X(t), ..., Z(t+1)=i) + log P(Z(t+1)=i | X(t), X(t-1), ...)
            for i in range(num_states):
                obs_log_likelihood[i] += comps[i].log_density(x[(k - lag) : (k + 1)])

            # P(X(t+1) | X(t), X(t-1), ...)  [prevent underflow]
            max_ll = obs_log_likelihood.max()
            obs_log_likelihood -= max_ll
            np.exp(obs_log_likelihood, out=obs_log_likelihood)
            sum_ll = np.sum(obs_log_likelihood)

            # P(X(t+1), X(t), ...)
            retval += np.log(sum_ll) + max_ll

        if self.len_dist is not None:
            retval += self.len_dist.log_density(len(x) - lag + 1 if lag > 0 else len(x))

        return retval

    def viterbi_sequence(self, x):
        """Compute the most likely hidden state sequence for observed sequence x.

        Args:
            x (Sequence[T]): Observed sequence with len(x) >= lag.

        Returns:
            np.ndarray: Integer array of len(x) - lag + 1 (len(x) when lag == 0) most likely hidden
                state indices.

        """
        obs_cnt = len(x) - self.lag + 1 if self.lag > 0 else len(x)
        log_w = self.log_w
        log_t = np.log(self.transitions)
        num_states = self.num_states
        comps = self.topics
        lag = self.lag
        init_comps = self.init_dist

        rv = np.zeros(obs_cnt, dtype=int)
        max_mat = np.zeros((num_states, obs_cnt), dtype=int)
        obs_mat = np.zeros((num_states, obs_cnt), dtype=float)

        obs_mat[:, 0] += log_w
        if lag > 0:
            for i in range(num_states):
                obs_mat[i, 0] += init_comps[i].log_density(x[:lag])
        else:
            for i in range(num_states):
                obs_mat[i, 0] += comps[i].log_density(x[0:1])

        for idx, k in enumerate(range(max(lag, 1), len(x))):
            for i in range(num_states):
                obs_ll = comps[i].log_density(x[(k - lag) : (k + 1)])
                temp_ll = obs_mat[:, idx] + log_t[:, i] + obs_ll
                max_idx = np.argmax(temp_ll)
                max_mat[i, idx + 1] = max_idx
                obs_mat[i, idx + 1] = temp_ll[max_idx]

        rv[obs_cnt - 1] = np.argmax(obs_mat[:, obs_cnt - 1])
        for idx in range(obs_cnt - 1, 0, -1):
            rv[idx - 1] = max_mat[rv[idx], idx]

        return rv

    def seq_log_density(self, x):
        """Vectorized evaluation of the log-density at encoded sequences x.

        Args:
            x: Encoded sequence data produced by seq_encode() / dist_to_encoder().

        Returns:
            np.ndarray: Log-density value for each encoded sequence.

        """
        num_states = self.num_states

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        w = self.w
        A = self.transitions
        tot_cnt = len(ids) + len(idi)
        num_seq = len(sz)

        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        ll_ret = np.zeros(num_seq, dtype=np.float64)
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            if self.lag > 0:
                pr_obs[imi, i] = self.init_dist[i].seq_log_density(enc_idata).astype(np.float64)
            pr_obs[ims, i] = self.topics[i].seq_log_density(enc_sdata).astype(np.float64)

        pr_max0 = pr_obs.max(axis=1)
        pr_obs -= pr_max0[:, None]
        np.exp(pr_obs, out=pr_obs)

        alpha_buff = np.zeros((num_seq, num_states), dtype=np.float64)
        next_alpha = np.zeros((num_seq, num_states), dtype=np.float64)

        numba_seq_log_density(num_states, tz, pr_obs, w, A, pr_max0, next_alpha, alpha_buff, ll_ret)

        ll_ret += self.len_dist.seq_log_density(len_enc)

        return ll_ret

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete lookback-HMM instance."""
        from pysp.stats.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.topics)
        if self.lag > 0 and self.init_dist is not None:
            children = children + tuple(self.init_dist)
        if self.len_dist is not None:
            children = children + (self.len_dist,)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def backend_seq_log_density(self, x, engine):
        """Engine-neutral lookback-HMM scoring via the shared HMM forward pass."""
        from pysp.stats.backend import backend_seq_log_density
        from pysp.stats.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x
        num_states = self.num_states
        tot_cnt = len(ids) + len(idi)
        log_pr = np.zeros((tot_cnt, num_states), dtype=np.float64)
        for i in range(num_states):
            if self.lag > 0:
                log_pr[imi, i] = np.asarray(
                    engine.to_numpy(backend_seq_log_density(self.init_dist[i], enc_idata, engine))
                )
            log_pr[ims, i] = np.asarray(engine.to_numpy(backend_seq_log_density(self.topics[i], enc_sdata, engine)))

        padded, mask, offsets = hmm_pad_log_emissions(log_pr, np.asarray(sz))
        with np.errstate(divide="ignore"):
            log_w = np.log(self.w)
            log_a = np.log(self.transitions)
        ll, _, _, _ = hmm_engine_forward_backward(engine, padded, log_w, log_a, mask)
        if self.len_dist is not None:
            ll = ll + backend_seq_log_density(self.len_dist, len_enc, engine)
        return ll

    def seq_posterior(self, x):
        """Compute posterior hidden state probabilities for encoded sequences x.

        Args:
            x: Encoded sequence data produced by seq_encode() / dist_to_encoder().

        Returns:
            List[np.ndarray]: For each sequence, an array of per-position posterior state
                probabilities with shape (num_windows, num_states).

        """
        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        tot_cnt = len(ids) + len(idi)
        seq_cnt = len(sz)
        num_states = self.num_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        weights = np.ones(seq_cnt, dtype=np.float64)

        max_len = sz.max()
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = self.w
        tran_mat = self.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            if self.lag > 0:
                pr_obs[imi, i] = self.init_dist[i].seq_log_density(enc_idata)
            pr_obs[ims, i] = self.topics[i].seq_log_density(enc_sdata)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch_alphas(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)

        return [alphas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

    def sampler(self, seed: int | None = None) -> "LookbackHiddenMarkovSampler":
        """Create a LookbackHiddenMarkovSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for random number generator.

        Returns:
            LookbackHiddenMarkovSampler: Sampler object (requires a non-null len_dist).

        """
        return LookbackHiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LookbackHiddenMarkovEstimator":
        """Create a LookbackHiddenMarkovEstimator from this distribution.

        Args:
            pseudo_count (Optional[float]): Regularize the initial-state and transition estimates.

        Returns:
            LookbackHiddenMarkovEstimator: Estimator built from the topic, initial-segment, and
                length distributions, preserving the lag.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        init_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.init_dist]
        return LookbackHiddenMarkovEstimator(
            comp_ests,
            lag=self.lag,
            init_estimators=init_ests,
            len_estimator=len_est,
            pseudo_count=(pseudo_count, pseudo_count),
            name=self.name,
        )

    def seq_encode(self, x: Sequence[Sequence[T]]):
        """Encode a sequence of observed sequences for vectorized 'seq_' calls.

        Args:
            x (Sequence[Sequence[T]]): Sequence of iid observed sequences.

        Returns:
            Encoded data consistent with seq_log_density(), seq_posterior(), and seq_update().

        """
        return self.dist_to_encoder().seq_encode(x)

    def dist_to_encoder(self) -> "LookbackHiddenMarkovDataEncoder":
        """Return a LookbackHiddenMarkovDataEncoder for encoding sequences of iid observations.

        Returns:
            LookbackHiddenMarkovDataEncoder: Encoder built from the topic, initial, and length
                distributions of this instance.

        """
        encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        init_encoder = self.init_dist[0].dist_to_encoder()

        return LookbackHiddenMarkovDataEncoder(
            encoder=encoder, len_encoder=len_encoder, init_encoder=init_encoder, lag=self.lag
        )


class LookbackHiddenMarkovSampler(DistributionSampler):
    """Sampler for LookbackHiddenMarkovDistribution. Requires non-null init_dist and len_dist."""

    def __init__(self, dist: LookbackHiddenMarkovDistribution, seed: int | None = None) -> None:
        """LookbackHiddenMarkovSampler object.

        Args:
            dist (LookbackHiddenMarkovDistribution): Distribution to sample from (init_dist and
                len_dist must be set, and topics must support sample_given()).
            seed (Optional[int]): Seed for random number generator.

        """
        self.num_states = dist.num_states
        self.dist = dist
        self.rng = RandomState(seed)

        self.init_samplers = [
            dist.init_dist[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in range(dist.num_states)
        ]
        self.obs_samplers = [
            dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in range(dist.num_states)
        ]
        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

        t_map = {i: {k: dist.transitions[i, k] for k in range(dist.num_states)} for i in range(dist.num_states)}
        p_map = {i: dist.w[i] for i in range(dist.num_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None):
        """Draw iid sequences from the lookback hidden Markov distribution.

        Args:
            size (Optional[int]): Number of sequences to draw. If None, a single sequence is returned.

        Returns:
            Union[List[T], List[List[T]]]: One sampled sequence if size is None, else a list of
                ``size`` sampled sequences.

        """
        if size is None:
            lag = self.dist.lag
            n = self.len_sampler.sample()
            state_seq = self.state_sampler.sample_seq(n)

            if lag == 0:
                # ordinary HMM: each of the n states emits one observation given an empty history
                return [self.obs_samplers[state_seq[i]].sample_given([]) for i in range(n)]

            rv = list(self.init_samplers[state_seq[0]].sample())  # [v_1, ..., v_lag]
            for i in range(1, n):
                rv.append(self.obs_samplers[state_seq[i]].sample_given(rv[-lag:]))
            return rv
        else:
            return [self.sample() for i in range(size)]


class LookbackHiddenMarkovEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of a lookback hidden Markov model."""

    def __init__(self, seq_accumulators, init_accumulators=None, lag=0, len_accumulator=None, keys=(None, None, None)):
        """LookbackHiddenMarkovEstimatorAccumulator object.

        Args:
            seq_accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Per-state accumulators
                for the emission window distributions.
            init_accumulators (Optional[Sequence[SequenceEncodableStatisticAccumulator]]): Per-state
                accumulators for the initial-segment distributions.
            lag (int): Number of preceding observations each emission conditions on.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the
                window-count distribution.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial-state counts,
                transition counts, and state accumulators.

        """
        self.seq_accumulators = seq_accumulators
        self.init_accumulators = init_accumulators
        self.num_states = len(seq_accumulators)
        self.init_counts = vec.zeros(self.num_states)
        self.trans_counts = vec.zeros((self.num_states, self.num_states))
        self.state_counts = vec.zeros(self.num_states)
        self.len_accumulator = len_accumulator
        self.lag = lag

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        # When _track_ll is enabled, seq_update accumulates the per-sequence data
        # log-likelihood into _seq_ll. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); default path is unchanged and zero-cost.
        self._track_ll = False
        self._seq_ll = 0.0

    def update(self, x, weight, estimate):
        """Update sufficient statistics with one observed sequence and weight.

        Args:
            x (Sequence[T]): Observed sequence.
            weight (float): Weight for the observation.
            estimate (LookbackHiddenMarkovDistribution): Current estimate used for the E-step.

        """
        self.seq_update(estimate.seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x, weight, rng):
        """Initialize sufficient statistics with one observed sequence using random state weights.

        Args:
            x (Sequence[T]): Observed sequence.
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator for the random state assignment.

        """
        lag = self.lag
        n = len(x) - lag + 1 if lag > 0 else len(x)

        if self.len_accumulator is not None:
            self.len_accumulator.initialize(n, weight, rng)

        if n > 0:
            w = rng.dirichlet(np.ones(self.num_states) / (self.num_states**2), size=n) * weight

            self.init_counts += w[0, :]
            self.state_counts += w.sum(axis=0)

            if lag > 0:
                for j in range(self.num_states):
                    self.init_accumulators[j].initialize(x[:lag], w[0, j], rng)

                for k, i in enumerate(range(lag, len(x))):
                    self.trans_counts += np.outer(w[k, :], w[k + 1, :])

                    for j in range(self.num_states):
                        self.seq_accumulators[j].initialize(x[(i - lag) : (i + 1)], w[k + 1, j], rng)
            else:
                for k in range(len(x)):
                    if k > 0:
                        self.trans_counts += np.outer(w[k - 1, :], w[k, :])

                    for j in range(self.num_states):
                        self.seq_accumulators[j].initialize(x[k : (k + 1)], w[k, j], rng)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization of sufficient statistics with encoded sequences.

        Args:
            x: Encoded sequence data produced by acc_to_encoder() (or a matching dist encoder).
            weights (np.ndarray): Weight for each encoded sequence.
            rng (np.random.RandomState): Random number generator for the random state assignment.

        """
        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        num_states = self.num_states
        tot_cnt = len(ids) + len(idi)

        # per-position sparse soft state assignment, mirroring initialize()
        ww = rng.dirichlet(np.ones(num_states) / (num_states**2), size=tot_cnt)

        w_init = ww[imi, :] * weights[idi][:, None]
        w_seq = ww[ims, :] * weights[ids][:, None]

        tz = np.concatenate([[0], sz]).cumsum().astype(np.int32)

        if self.lag > 0:
            self.init_counts += w_init.sum(axis=0)
            self.state_counts += w_init.sum(axis=0) + w_seq.sum(axis=0)
        else:
            # lag == 0: the first emission window of each sequence is the initial position
            nz = sz > 0
            self.init_counts += (ww[tz[:-1][nz], :] * weights[nz][:, None]).sum(axis=0)
            self.state_counts += w_seq.sum(axis=0)

        # transitions between consecutive positions within each sequence
        prev_mask = np.ones(tot_cnt, dtype=bool)
        prev_mask[tz[1:] - 1] = False
        prev_idx = np.flatnonzero(prev_mask)
        next_idx = prev_idx + 1
        seq_of_pos = np.repeat(np.arange(len(sz)), sz)
        w_pos = weights[seq_of_pos[prev_idx]]
        self.trans_counts += np.einsum("n,ni,nj->ij", w_pos, ww[prev_idx, :], ww[next_idx, :])

        for j in range(num_states):
            if self.lag > 0:
                self.init_accumulators[j].seq_initialize(enc_idata, w_init[:, j], rng)
            self.seq_accumulators[j].seq_initialize(enc_sdata, w_seq[:, j], rng)

        if self.len_accumulator is not None and len_enc is not None:
            self.len_accumulator.seq_initialize(len_enc, weights, rng)

    def acc_to_encoder(self) -> "LookbackHiddenMarkovDataEncoder":
        """Return a LookbackHiddenMarkovDataEncoder consistent with this accumulator.

        Returns:
            LookbackHiddenMarkovDataEncoder: Encoder built from the member accumulators.

        """
        encoder = self.seq_accumulators[0].acc_to_encoder()
        init_encoder = self.init_accumulators[0].acc_to_encoder() if self.init_accumulators else NullDataEncoder()
        len_encoder = self.len_accumulator.acc_to_encoder() if self.len_accumulator is not None else NullDataEncoder()

        return LookbackHiddenMarkovDataEncoder(
            encoder=encoder, lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder
        )

    def seq_update(self, x, weights, estimate):
        """Vectorized Baum-Welch update of sufficient statistics with encoded sequences.

        Args:
            x: Encoded sequence data produced by acc_to_encoder() (or a matching dist encoder).
            weights (np.ndarray): Weight for each encoded sequence.
            estimate (LookbackHiddenMarkovDistribution): Current estimate used for the E-step.

        """
        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        tot_cnt = len(ids) + len(idi)
        seq_cnt = len(sz)
        num_states = estimate.num_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

        max_len = sz.max()
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = estimate.w
        tran_mat = estimate.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            if self.lag > 0:
                pr_obs[imi, i] = estimate.init_dist[i].seq_log_density(enc_idata)
            pr_obs[ims, i] = estimate.topics[i].seq_log_density(enc_sdata)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        # When the fused-EM fast path requests it, compute the per-sequence data
        # log-likelihood from the already-scored emissions via the (read-only)
        # forward kernel, reusing pr_obs so no emissions are re-scored. Done before
        # Baum-Welch (which may overwrite pr_obs). Matches seq_log_density exactly.
        if self._track_ll:
            ll_ret = np.zeros(seq_cnt, dtype=np.float64)
            nb_next = np.zeros((seq_cnt, num_states), dtype=np.float64)
            nb_buff = np.zeros((seq_cnt, num_states), dtype=np.float64)
            pr_max_1d = np.ascontiguousarray(pr_max[:, 0])
            numba_seq_log_density(num_states, tz, pr_obs, init_pvec, tran_mat, pr_max_1d, nb_next, nb_buff, ll_ret)
            if estimate.len_dist is not None and len_enc is not None:
                ll_ret = ll_ret + estimate.len_dist.seq_log_density(len_enc)
            self._seq_ll += float(np.dot(weights, ll_ret))

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch2(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)
        self.init_counts += pi_acc.sum(axis=0)
        self.trans_counts += xi_acc.sum(axis=0)

        # numba_baum_welch2.parallel_diagnostics(level=4)

        for i in range(num_states):
            if self.lag > 0:
                self.init_accumulators[i].seq_update(enc_idata, alphas[imi, i], estimate.init_dist[i])
            self.seq_accumulators[i].seq_update(enc_sdata, alphas[ims, i], estimate.topics[i])

        self.state_counts += alphas.sum(axis=0)

        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident Baum-Welch E-step via the shared HMM forward-backward (numpy or torch).

        Emissions (init segment + windowed topics) are scored on the active engine, the
        forward-backward runs on the engine, and the resulting posteriors are routed to the init /
        topic / length accumulators. Mirrors seq_update.
        """
        from pysp.stats.backend import backend_seq_log_density
        from pysp.stats.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x
        num_states = estimate.num_states
        tot_cnt = len(ids) + len(idi)
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)

        log_pr = np.zeros((tot_cnt, num_states), dtype=np.float64)
        for i in range(num_states):
            if self.lag > 0:
                log_pr[imi, i] = np.asarray(
                    engine.to_numpy(backend_seq_log_density(estimate.init_dist[i], enc_idata, engine))
                )
            log_pr[ims, i] = np.asarray(engine.to_numpy(backend_seq_log_density(estimate.topics[i], enc_sdata, engine)))

        sz_np = np.asarray(sz)
        padded, mask, offsets = hmm_pad_log_emissions(log_pr, sz_np)
        with np.errstate(divide="ignore"):
            log_w = np.log(estimate.w)
            log_a = np.log(estimate.transitions)
        _, gamma, xi_sum, pi = hmm_engine_forward_backward(engine, padded, log_w, log_a, mask, weights=weights_np)
        gamma = np.asarray(engine.to_numpy(gamma))
        xi_sum = np.asarray(engine.to_numpy(xi_sum))
        pi = np.asarray(engine.to_numpy(pi))

        gamma_flat = np.zeros((tot_cnt, num_states), dtype=np.float64)
        for i in range(len(sz_np)):
            n = int(sz_np[i])
            if n > 0:
                gamma_flat[offsets[i] : offsets[i + 1], :] = gamma[i, :n, :]

        self.init_counts += pi.sum(axis=0)
        self.trans_counts += xi_sum
        self.state_counts += gamma_flat.sum(axis=0)
        for i in range(num_states):
            if self.lag > 0:
                self.init_accumulators[i].seq_update(enc_idata, gamma_flat[imi, i], estimate.init_dist[i])
            self.seq_accumulators[i].seq_update(enc_sdata, gamma_flat[ims, i], estimate.topics[i])

        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(len_enc, weights_np, estimate.len_dist)

    def combine(self, suff_stat):
        """Aggregate sufficient statistics from suff_stat (a value() tuple) into this accumulator.

        Args:
            suff_stat (Tuple): Sufficient statistics in the format returned by value().

        Returns:
            LookbackHiddenMarkovEstimatorAccumulator: This accumulator after aggregation.

        """
        lag, num_states, init_counts, state_counts, trans_counts, seq_accumulators, init_accumulators, len_acc = (
            suff_stat
        )

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.init_accumulators[i].combine(init_accumulators[i])
            self.seq_accumulators[i].combine(seq_accumulators[i])

        if self.len_accumulator is not None and len_acc is not None:
            self.len_accumulator.combine(len_acc)

        return self

    def value(self):
        """Return the sufficient statistics of this accumulator.

        Returns:
            Tuple: (lag, num_states, init_counts, state_counts, trans_counts, seq_acc_values,
                init_acc_values, len_acc_value).

        """
        if self.len_accumulator is not None:
            len_val = self.len_accumulator.value()
        else:
            len_val = None

        return (
            self.lag,
            self.num_states,
            self.init_counts,
            self.state_counts,
            self.trans_counts,
            tuple([u.value() for u in self.seq_accumulators]),
            tuple([u.value() for u in self.init_accumulators]),
            len_val,
        )

    def from_value(self, x):
        """Set the sufficient statistics of this accumulator from a value() tuple.

        Args:
            x (Tuple): Sufficient statistics in the format returned by value().

        Returns:
            LookbackHiddenMarkovEstimatorAccumulator: This accumulator after assignment.

        """
        lag, num_states, init_counts, state_counts, trans_counts, seq_accumulators, init_accumulators, len_acc = x

        self.lag = lag
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts

        for i, v in enumerate(init_accumulators):
            self.init_accumulators[i].from_value(v)

        for i, v in enumerate(seq_accumulators):
            self.seq_accumulators[i].from_value(v)

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(len_acc)

        return self

    def scale(self, c: float) -> "LookbackHiddenMarkovEstimatorAccumulator":
        self.init_counts *= c
        self.state_counts *= c
        self.trans_counts *= c
        for acc in self.init_accumulators:
            acc.scale(c)
        for acc in self.seq_accumulators:
            acc.scale(c)
        if self.len_accumulator is not None:
            self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict):
        """Merge keyed sufficient statistics of this accumulator into stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to merged sufficient statistics.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_counts
            else:
                stats_dict[self.init_key] = self.init_counts

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_counts
            else:
                stats_dict[self.trans_key] = self.trans_counts

        if self.state_key is not None:
            if self.state_key in stats_dict:
                acc = stats_dict[self.state_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.seq_accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.seq_accumulators

        for u in self.init_accumulators:
            u.key_merge(stats_dict)

        for u in self.seq_accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace keyed sufficient statistics of this accumulator with values from stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to merged sufficient statistics.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.seq_accumulators = stats_dict[self.state_key]

        for u in self.init_accumulators:
            u.key_replace(stats_dict)

        for u in self.seq_accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)


class LookbackHiddenMarkovEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating LookbackHiddenMarkovEstimatorAccumulator objects."""

    def __init__(
        self,
        lag: int,
        seq_factories: Sequence[StatisticAccumulatorFactory],
        init_factories: Sequence[StatisticAccumulatorFactory] | None = None,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
    ):
        """LookbackHiddenMarkovEstimatorAccumulatorFactory object.

        Args:
            lag (int): Number of preceding observations each emission conditions on.
            seq_factories (Sequence[StatisticAccumulatorFactory]): Per-state factories for the
                emission window accumulators.
            init_factories (Optional[Sequence[StatisticAccumulatorFactory]]): Per-state factories for
                the initial-segment accumulators. Defaults to NullAccumulatorFactory per state.
            len_factory (Optional[StatisticAccumulatorFactory]): Factory for the window-count
                accumulator. Defaults to NullAccumulatorFactory.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys for
                initial-state counts, transition counts, and state accumulators.

        """
        self.seq_factories = seq_factories
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.lag = lag

        if init_factories is None:
            self.init_factories = [NullAccumulatorFactory() for j in range(len(seq_factories))]
        else:
            self.init_factories = init_factories

    def make(self) -> "LookbackHiddenMarkovEstimatorAccumulator":
        """Create a new LookbackHiddenMarkovEstimatorAccumulator from the member factories.

        Returns:
            LookbackHiddenMarkovEstimatorAccumulator: Accumulator with zeroed sufficient statistics.

        """
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        seq_acc = [self.seq_factories[i].make() for i in range(len(self.seq_factories))]
        init_acc = [self.init_factories[i].make() for i in range(len(self.init_factories))]
        return LookbackHiddenMarkovEstimatorAccumulator(
            seq_acc, lag=self.lag, init_accumulators=init_acc, len_accumulator=len_acc, keys=self.keys
        )


class LookbackHiddenMarkovEstimator(ParameterEstimator):
    """Estimator for a lookback hidden Markov model from aggregated sufficient statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        lag: int = 0,
        init_estimators: Sequence[ParameterEstimator] | None = None,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        suff_stat=None,
        pseudo_count: tuple[float | None, float | None] | None = (None, None),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
    ):
        """LookbackHiddenMarkovEstimator object.

        Args:
            estimators (Sequence[ParameterEstimator]): Per-state estimators for the emission window
                distributions (one per hidden state).
            lag (int): Number of preceding observations each emission conditions on.
            init_estimators (Optional[Sequence[ParameterEstimator]]): Per-state estimators for the
                initial-segment distributions. Defaults to NullEstimator per state (the sibling
                module pysp.stats.lookback_hmm requires these to be passed explicitly).
            len_estimator (Optional[ParameterEstimator]): Estimator for the window-count distribution.
                Defaults to NullEstimator.
            suff_stat (Optional[Tuple]): Kept for interface consistency (unused).
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize the
                initial-state probabilities and the transition matrix respectively.
            name (Optional[str]): Assign string name to the estimated distribution.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys for
                initial-state counts, transition counts, and state accumulators.

        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.lag = lag

        if init_estimators is None:
            self.init_estimators = [NullEstimator() for xx in range(self.num_states)]
        else:
            self.init_estimators = init_estimators

    def accumulator_factory(self):
        """Create a LookbackHiddenMarkovEstimatorAccumulatorFactory from the member estimators.

        Returns:
            LookbackHiddenMarkovEstimatorAccumulatorFactory: Factory for accumulators consistent
                with this estimator.

        """
        est_factories = [u.accumulator_factory() for u in self.estimators]
        iest_factories = [u.accumulator_factory() for u in self.init_estimators]

        len_factory = self.len_estimator.accumulator_factory()
        return LookbackHiddenMarkovEstimatorAccumulatorFactory(
            self.lag, est_factories, iest_factories, len_factory, self.keys
        )

    def estimate(self, nobs: float | None, suff_stat):
        """Estimate a LookbackHiddenMarkovDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations (passed to the length estimator).
            suff_stat (Tuple): Sufficient statistics in the format returned by
                LookbackHiddenMarkovEstimatorAccumulator.value().

        Returns:
            LookbackHiddenMarkovDistribution: M-step estimate of the distribution.

        """
        lag, num_states, init_counts, state_counts, trans_counts, topic_ss, init_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss)

        topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]
        init_dist = [self.init_estimators[i].estimate(init_counts[i], init_ss[i]) for i in range(num_states)]

        if self.pseudo_count[0] is not None:
            p1 = self.pseudo_count[0] / float(num_states)
            w = init_counts + p1
            w /= w.sum()
        else:
            w = init_counts / init_counts.sum()

        if self.pseudo_count[1] is not None:
            p2 = self.pseudo_count[1] / float(num_states * num_states)
            transitions = trans_counts + p2
            row_sum = transitions.sum(axis=1, keepdims=True)
            transitions /= row_sum
        else:
            row_sum = trans_counts.sum(axis=1, keepdims=True)
            bad_rows = row_sum.flatten() == 0.0

            if np.any(bad_rows):
                good_rows = ~bad_rows
                transitions = np.zeros_like(trans_counts, dtype=np.float64)
                transitions[good_rows, :] += trans_counts[good_rows, :] / row_sum[good_rows]
            else:
                transitions = trans_counts / row_sum

        return LookbackHiddenMarkovDistribution(
            topics, w, transitions, lag=lag, init_dist=init_dist, len_dist=len_dist, name=self.name
        )


class LookbackHiddenMarkovDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid lookback-HMM observations (each a Sequence[T])."""

    def __init__(
        self,
        encoder: DataSequenceEncoder,
        lag: int,
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
        init_encoder: DataSequenceEncoder | None = NullDataEncoder(),
    ) -> None:
        """LookbackHiddenMarkovDataEncoder object.

        Args:
            encoder (DataSequenceEncoder): Encoder for length-(lag+1) emission windows.
            lag (int): Number of preceding observations each emission conditions on.
            len_encoder (Optional[DataSequenceEncoder]): Encoder for window counts. Defaults to
                NullDataEncoder.
            init_encoder (Optional[DataSequenceEncoder]): Encoder for the initial x[:lag] segments.
                Defaults to NullDataEncoder.

        """
        self.encoder = encoder
        self.lag = lag
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.init_encoder = init_encoder if init_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Returns string representation of LookbackHiddenMarkovDataEncoder object."""
        s = "LookbackHiddenMarkovDataEncoder(encoder=" + str(self.encoder) + ",lag=" + str(self.lag)
        s += ",len_encoder=" + str(self.len_encoder) + ",init_encoder=" + str(self.init_encoder) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        """Checks if other is an equivalent LookbackHiddenMarkovDataEncoder (same lag and member encoders).

        Args:
            other (object): Object to compare.

        Returns:
            True if other is a LookbackHiddenMarkovDataEncoder with equal lag and member encoders.

        """
        if isinstance(other, LookbackHiddenMarkovDataEncoder):
            c0 = self.len_encoder == other.len_encoder
            c1 = self.init_encoder == other.init_encoder
            c2 = self.lag == other.lag
            c3 = self.encoder == other.encoder

            return c0 and c1 and c2 and c3

        else:
            return False

    def seq_encode(self, x):
        """Encode a sequence of iid observed sequences for vectorized processing.

        Each sequence x[i] is split into its initial segment x[i][:lag] and its sliding windows
        x[i][j-lag:j+1] for j in [lag, len(x[i])); index arrays track which flattened position belongs
        to which sequence and which positions are initial segments vs emission windows. When lag == 0
        there are no initial segments: every position is an emission window (ordinary HMM).

        Args:
            x (Sequence[Sequence[T]]): Sequence of iid observed sequences.

        Returns:
            Tuple: ((ids, idi, ims, imi, sz, enc_windows, enc_inits), len_enc) where ids/idi map
                windows/initial segments to sequence indices, ims/imi give their flattened positions,
                sz holds per-sequence position counts, enc_windows/enc_inits are the encoded windows and
                initial segments (enc_inits is None when lag == 0), and len_enc is the encoded position
                counts.

        """
        ids = []
        idi = []
        xss = []
        sz = []
        xsi = []
        imi = []
        ims = []

        lag = self.lag
        cnt = 0
        for i in range(len(x)):
            xxs = [x[i][(j - lag) : (j + 1)] for j in range(lag, len(x[i]))]
            ids.extend([i] * len(xxs))
            xss.extend(xxs)

            if lag > 0:
                xsi.append(x[i][:lag])
                idi.append(i)
                sz.append(len(x[i]) - lag + 1)

                imi.append(cnt)
                ims.extend(range(cnt + 1, cnt + 1 + (len(x[i]) - lag)))
                cnt += len(x[i]) - lag + 1
            else:
                sz.append(len(x[i]))

                ims.extend(range(cnt, cnt + len(x[i])))
                cnt += len(x[i])

        len_enc = self.len_encoder.seq_encode(sz)

        ids = np.asarray(ids, dtype=np.int32)
        idi = np.asarray(idi, dtype=np.int32)
        ims = np.asarray(ims, dtype=np.int32)
        imi = np.asarray(imi, dtype=np.int32)
        sz = np.asarray(sz, dtype=np.int32)
        xss = self.encoder.seq_encode(xss)
        xsi = self.init_encoder.seq_encode(xsi) if lag > 0 else None

        return (ids, idi, ims, imi, sz, xss, xsi), len_enc


@numba.njit(
    "void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:])",
    parallel=True,
    fastmath=True,
    cache=True,
)
def numba_seq_log_density(num_states, tz, prob_mat, init_pvec, tran_mat, max_ll, next_alpha_mat, alpha_buff_mat, out):
    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            out[n] = 0
            continue

        next_alpha = next_alpha_mat[n, :]
        alpha_buff = alpha_buff_mat[n, :]

        llsum = 0
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            next_alpha[i] = temp
            alpha_sum += temp

        llsum += math.log(alpha_sum)
        llsum += max_ll[s0]

        for s in range(s0 + 1, s1):
            for i in range(num_states):
                alpha_buff[i] = next_alpha[i] / alpha_sum

            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_buff[j]
                temp *= prob_mat[s, i]
                next_alpha[i] = temp
                alpha_sum += temp

            llsum += math.log(alpha_sum)
            llsum += max_ll[s]

        out[n] = llsum


@numba.njit(
    "void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:,:])",
    cache=True,
)
def numba_baum_welch(
    num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc, beta_buff, xi_buff
):
    for n in range(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):
            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
                # alpha_sum = temp if temp > alpha_sum else alpha_sum

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        # beta_sum = 1/num_states
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):
            sp1 = s + 1

            for j in range(num_states):
                beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

            xi_buff_sum = 0
            gamma_buff = 0
            beta_sum = 0
            for i in range(num_states):
                temp_beta = 0
                for j in range(num_states):
                    temp = tran_mat[i, j] * beta_buff[j]
                    temp_beta += temp
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta
                # beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[i] += alpha_loc[s0, i]


@numba.njit(
    "void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])",
    parallel=True,
    fastmath=True,
    cache=True,
)
def numba_baum_welch2(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):
    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):
            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
                # alpha_sum = temp if temp > alpha_sum else alpha_sum

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        # beta_sum = 1/num_states
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):
            sp1 = s + 1

            for j in range(num_states):
                beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

            xi_buff_sum = 0
            gamma_buff = 0
            beta_sum = 0
            for i in range(num_states):
                temp_beta = 0
                for j in range(num_states):
                    temp = tran_mat[i, j] * beta_buff[j]
                    temp_beta += temp
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta
                # beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_loc[s0, i]


@numba.njit(
    "void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])",
    parallel=True,
    fastmath=True,
    cache=True,
)
def numba_baum_welch_alphas(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):
    for n in numba.prange(len(tz) - 1):
        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):
            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
                # alpha_sum = temp if temp > alpha_sum else alpha_sum

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
LookbackHiddenMarkovAccumulator = LookbackHiddenMarkovEstimatorAccumulator
LookbackHiddenMarkovAccumulatorFactory = LookbackHiddenMarkovEstimatorAccumulatorFactory
