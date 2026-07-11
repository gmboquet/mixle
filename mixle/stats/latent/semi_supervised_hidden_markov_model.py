"""Semi-supervised hidden Markov model: each observation may carry a per-position state prior.

A SemiSupervisedHiddenMarkovModelDistribution is an HMM with shared emissions and transitions in which every
observation can carry soft evidence (a prior) over the hidden state at *each* position of the sequence -- not
only the initial state. An observation is a pair ``(emission_seq, state_prior)``:

    - ``emission_seq``: a length-T sequence of emissions (data type of the emission distributions).
    - ``state_prior``: an optional ``T``-by-``S`` array of non-negative weights. Row t is a prior / soft label
      over the S hidden states at position t; it multiplies the hidden-state distribution there. ``None`` (or an
      all-ones row) imposes no constraint. There is no separate learned initial distribution -- the prior at
      position 0 plays that role (uniform when absent).

The prior folds into the forward-backward as an extra multiplicative factor on the emission likelihood at every
position, so it shapes both scoring (``log_density``) and the EM E-step. Only the transitions and emissions (and
an optional length distribution) are learned; the priors are given side information. With every prior ``None``
the model is an ordinary HMM with a uniform initial state distribution.

Defines SemiSupervisedHiddenMarkovModelDistribution, SemiSupervisedHiddenMarkovSampler,
SemiSupervisedHiddenMarkovEstimatorAccumulator, SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory,
SemiSupervisedHiddenMarkovEstimator, and SemiSupervisedHiddenMarkovDataEncoder.
"""

import numpy as np
from numpy.random import RandomState

from mixle.capability import Neutral, supports
from mixle.stats.combinator.null_dist import NullAccumulator, NullDataEncoder, NullDistribution, NullEstimator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_LOG_ZERO = -np.inf


def _as_prior(prior, length: int, num_states: int) -> np.ndarray | None:
    """Validate/normalize an observation's state prior to a (length, num_states) float array, or None."""
    if prior is None:
        return None
    p = np.asarray(prior, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape == (length, num_states):
        pass
    elif p.shape == (1, num_states) and length > 0:
        p = np.repeat(p, length, axis=0)  # a single shared prior row broadcast over the sequence
    else:
        raise ValueError("state_prior must be shape (T=%d, S=%d) or (1, S); got %s" % (length, num_states, p.shape))
    if np.any(p < 0.0):
        raise ValueError("state_prior must be non-negative.")
    return p


class SemiSupervisedHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """HMM with shared emissions/transitions where each observation may carry a per-position state prior."""

    def compute_capabilities(self):
        """Declare the legacy NumPy execution path for semi-supervised HMM inference."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")

    def __init__(self, topics, transitions, len_dist=None, name=None, keys=None, use_numba=None, terminal_states=None):
        """SemiSupervisedHiddenMarkovModelDistribution.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): the S emission distributions.
            transitions (Union[Sequence[Sequence[float]], np.ndarray]): the S-by-S row-stochastic transition matrix.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): optional sequence-length distribution.
            name (Optional[str]): optional name.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): optional (transition, emission) keys for tying.
            use_numba (Optional[bool]): accepted for backward compatibility and ignored; this model is numpy-only.
        """
        self.topics = list(topics)
        self.nStates = len(self.topics)
        self.transitions = np.reshape(np.asarray(transitions, dtype=float), (self.nStates, self.nStates))
        with np.errstate(divide="ignore"):
            self.logTransitions = np.log(self.transitions)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.name = name
        if keys is None:
            keys = (None, None)
        self.keys = keys
        # Deliberately hard False (not a tunable default): the semi-supervised forward/backward is a
        # custom per-sequence implementation with no numba kernel, and its encoder must emit the
        # matching per-sequence layout. Mirrors the terminal-state force-off in hidden_markov.py.
        self.use_numba = False
        self.terminal_states = None if terminal_states is None else set(int(s) for s in terminal_states)
        if self.terminal_states is not None:
            self._terminal_mask = np.zeros(self.nStates, dtype=bool)
            self._terminal_mask[list(self.terminal_states)] = True

    def _terminal_log_b(self, emissions, prior) -> np.ndarray:
        """Per-position log emission+prior potentials ``(T, S)`` for the terminal forward."""
        enc = self.topics[0].dist_to_encoder().seq_encode(list(emissions))
        log_b = np.empty((len(emissions), self.nStates))
        for s in range(self.nStates):
            log_b[:, s] = np.asarray(self.topics[s].seq_log_density(enc), dtype=float)
        p = _as_prior(prior, len(emissions), self.nStates)
        if p is not None:
            with np.errstate(divide="ignore"):
                log_b = log_b + np.log(p)
        return log_b

    def _terminal_forward_loglik(self, emissions, prior) -> float:
        """Stopping-time likelihood (uniform initial weight; shared terminal forward over phi=emission*prior)."""
        from mixle.stats.latent.hidden_markov import terminal_forward_loglik

        if len(emissions) == 0:
            return _LOG_ZERO
        return terminal_forward_loglik(
            np.zeros(self.nStates), self.logTransitions, self._terminal_log_b(emissions, prior), self._terminal_mask
        )

    def __str__(self) -> str:
        s1 = ",".join(map(str, self.topics))
        s2 = repr([list(map(float, row)) for row in self.transitions])
        return "SemiSupervisedHiddenMarkovModelDistribution([%s], %s, len_dist=%s, name=%s, keys=%s)" % (
            s1,
            s2,
            str(self.len_dist),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x) -> float:
        """Return the probability of one semi-supervised HMM observation."""
        return float(np.exp(self.log_density(x)))

    def _emission_potential(self, emissions, prior) -> tuple[np.ndarray, np.ndarray]:
        """Return the (T, S) emission likelihood times the state prior (probability space)."""
        n = len(emissions)
        if n == 0:
            return np.zeros((0, self.nStates)), np.zeros(0)
        enc = self.topics[0].dist_to_encoder().seq_encode(list(emissions))  # score emissions vectorized over T
        b = np.empty((n, self.nStates))
        for s in range(self.nStates):
            b[:, s] = np.asarray(self.topics[s].seq_log_density(enc), dtype=float)
        mx = b.max(axis=1, keepdims=True)
        mx[~np.isfinite(mx)] = 0.0
        phi = np.exp(b - mx)  # scaled emission likelihood; the per-row offset mx is added back via the loglik
        p = _as_prior(prior, n, self.nStates)
        if p is not None:
            phi = phi * p
        return phi, mx[:, 0]

    def _forward_loglik(self, emissions, prior) -> float:
        """Scaled forward; returns the log joint evidence log sum_paths prod phi * prod A."""
        if self.terminal_states is not None:
            return self._terminal_forward_loglik(emissions, prior)
        n = len(emissions)
        if n == 0:
            return 0.0
        phi, offset = self._emission_potential(emissions, prior)
        a = phi[0]
        c = a.sum()
        if c <= 0.0:
            return _LOG_ZERO
        ll = np.log(c) + offset[0]
        a = a / c
        for t in range(1, n):
            pred = a @ self.transitions
            u = phi[t] * pred
            c = u.sum()
            if c <= 0.0:
                return _LOG_ZERO
            ll += np.log(c) + offset[t]
            a = u / c
        return float(ll)

    def log_density(self, x) -> float:
        """Return the log-likelihood of one ``(emissions, state_prior)`` observation."""
        emissions, prior = x
        ll = self._forward_loglik(emissions, prior)
        if not supports(self.len_dist, Neutral):
            ll += self.len_dist.log_density(len(emissions))
        return ll

    def seq_log_density(self, x) -> np.ndarray:
        """Return vectorized log-likelihoods for encoded semi-supervised HMM observations."""
        emissions_list, priors, len_enc, _ = x
        out = np.empty(len(emissions_list))
        for i, emissions in enumerate(emissions_list):
            out[i] = self._forward_loglik(emissions, priors[i])
        if len_enc is not None and not supports(self.len_dist, Neutral):
            out = out + self.len_dist.seq_log_density(len_enc)
        return out

    def density_semantics(self):
        """Return the joined density semantics of emission and optional length distributions."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.topics) + ([] if self.len_dist is None else [self.len_dist])
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed=None):
        """Return a sampler for emission sequences with no external state priors."""
        return SemiSupervisedHiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count=None):
        """Return a Baum-Welch estimator for transitions, emissions, and optional length."""
        len_est = None if supports(self.len_dist, Neutral) else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return SemiSupervisedHiddenMarkovEstimator(
            comp_ests, len_estimator=len_est, pseudo_count=pseudo_count, terminal_states=self.terminal_states
        )

    def dist_to_encoder(self):
        """Return the encoder for emission sequences, priors, and optional lengths."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder() if not supports(self.len_dist, Neutral) else NullDataEncoder()
        return SemiSupervisedHiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder)


class SemiSupervisedHiddenMarkovSampler(DistributionSampler):
    """Sample emission sequences from the HMM with a uniform initial state distribution.

    Priors are external side information, so sampled observations carry ``None`` as their prior.
    """

    def __init__(self, dist: SemiSupervisedHiddenMarkovModelDistribution, seed=None):
        self.dist = dist
        self.rng = RandomState(seed)
        self.state_samplers = [t.sampler(seed=self.rng.randint(0, 2**31 - 1)) for t in dist.topics]
        if not supports(dist.len_dist, Neutral):
            self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, 2**31 - 1))
        else:
            self.len_sampler = None

    def _sample_one(self):
        n = self.len_sampler.sample() if self.len_sampler is not None else 1
        n = int(n)
        s = self.dist.nStates
        emissions = []
        z = self.rng.choice(s)  # uniform initial state
        for t in range(n):
            if t > 0:
                z = self.rng.choice(s, p=self.dist.transitions[z])
            emissions.append(self.state_samplers[z].sample())
        return (emissions, None)

    def _sample_terminal(self, cap=1_000_000):
        """Run the chain (uniform initial) until the first terminal state; emit one observation per state."""
        s = self.dist.nStates
        z = int(self.rng.choice(s))
        states = [z]
        while z not in self.dist.terminal_states and len(states) < cap:
            z = int(self.rng.choice(s, p=self.dist.transitions[z]))
            states.append(z)
        return ([self.state_samplers[st].sample() for st in states], None)

    def sample(self, size=None):
        """Draw one observation or a list of observations with ``None`` state priors."""
        if self.dist.terminal_states is not None:
            return self._sample_terminal() if size is None else [self._sample_terminal() for _ in range(size)]
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class SemiSupervisedHiddenMarkovEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Baum-Welch sufficient statistics for the semi-supervised HMM (transition + emission counts, length)."""

    def __init__(self, accumulators, len_accumulator=None, keys=(None, None)):
        # INTENTIONAL DIVERGENCE from the base HiddenMarkovAccumulator. This model has no learned initial
        # state distribution (the per-position state prior at t=0 plays that role -- see the module docstring),
        # so the merge key is a 2-tuple (trans_key, state_key) -- no init_key -- and value()/combine() carry a
        # 3-tuple (trans_counts, emission_values, len_value) rather than the base 6-tuple (num_states,
        # init_counts, state_counts, trans_counts, emission_values, len_value). There are no init/state counts
        # to track, and SemiSupervisedHiddenMarkovEstimator.estimate consumes exactly this 3-tuple. These
        # accumulators never interoperate with HiddenMarkovAccumulator (the factory makes this type and key
        # tying is within-class), so the shapes are self-consistent; aligning to the base tuples would require
        # fabricating statistics the estimator ignores. Keep this shape.
        self.accumulators = list(accumulators)
        self.num_states = len(self.accumulators)
        self.trans_counts = np.zeros((self.num_states, self.num_states))
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.trans_key = keys[0]
        self.state_key = keys[1]

    # --- forward-backward posteriors with the prior folded in ---
    def _posteriors(self, dist, emissions, prior):
        n = len(emissions)
        s = dist.nStates
        if n == 0:
            return np.zeros((0, s)), np.zeros((s, s))
        phi, _ = dist._emission_potential(emissions, prior)
        A = dist.transitions
        alpha = np.empty((n, s))
        scale = np.empty(n)
        a = phi[0].copy()
        c = a.sum()
        c = c if c > 0 else 1.0
        alpha[0] = a / c
        scale[0] = c
        for t in range(1, n):
            u = phi[t] * (alpha[t - 1] @ A)
            c = u.sum()
            c = c if c > 0 else 1.0
            alpha[t] = u / c
            scale[t] = c
        beta = np.zeros((n, s))
        beta[n - 1] = 1.0
        gamma = np.empty((n, s))
        gamma[n - 1] = alpha[n - 1]
        xi = np.zeros((s, s))
        for t in range(n - 2, -1, -1):
            b_next = phi[t + 1] * beta[t + 1]
            beta[t] = (A @ b_next) / scale[t + 1]
            gamma[t] = alpha[t] * beta[t]
            g = gamma[t].sum()
            gamma[t] = gamma[t] / (g if g > 0 else 1.0)
            xi_t = (alpha[t][:, None] * A) * b_next[None, :] / scale[t + 1]
            xt = xi_t.sum()
            xi += xi_t / (xt if xt > 0 else 1.0)
        # normalize gamma at the last position too
        g = gamma[n - 1].sum()
        gamma[n - 1] = gamma[n - 1] / (g if g > 0 else 1.0)
        return gamma, xi

    def _terminal_posteriors(self, dist, emissions, prior):
        """Terminal-state forward-backward responsibilities (uniform initial; phi=emission*prior)."""
        from mixle.stats.latent.hidden_markov import terminal_forward_backward

        log_b = dist._terminal_log_b(emissions, prior)
        _, gamma, xi = terminal_forward_backward(
            np.zeros(dist.nStates), dist.logTransitions, log_b, dist._terminal_mask
        )
        return (None, None) if gamma is None else (gamma, xi.sum(axis=0))

    def _accumulate(self, dist, emissions, prior, weight):
        n = len(emissions)
        if n > 0:
            if dist is not None and getattr(dist, "terminal_states", None) is not None:
                gamma, xi = self._terminal_posteriors(dist, emissions, prior)
                if gamma is None:
                    return  # zero-probability sequence under the terminal model
            else:
                gamma, xi = self._posteriors(dist, emissions, prior)
            self.trans_counts += weight * xi
            # accumulate emissions vectorized over T: one weighted seq_update per state instead of T*S calls
            enc = self.accumulators[0].acc_to_encoder().seq_encode(list(emissions))
            for s in range(self.num_states):
                self.accumulators[s].seq_update(enc, weight * gamma[:, s], None if dist is None else dist.topics[s])
        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.update(n, weight, None if dist is None else dist.len_dist)

    def update(self, x, weight, estimate):
        """Update Baum-Welch sufficient statistics from one weighted observation."""
        emissions, prior = x
        self._accumulate(estimate, emissions, prior, weight)

    def initialize(self, x, weight, rng):
        """Initialize emission and transition statistics with random soft state assignments."""
        emissions, prior = x
        n = len(emissions)
        # random soft responsibilities (respecting the prior's zeros) to break symmetry
        if n > 0:
            p = _as_prior(prior, n, self.num_states)
            gamma = rng.dirichlet(np.ones(self.num_states), size=n)
            if p is not None:
                gamma = gamma * (p > 0)
                gamma = gamma / np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)
            for t in range(n):
                for s in range(self.num_states):
                    self.accumulators[s].initialize(emissions[t], weight * gamma[t, s], rng)
                if t > 0:
                    self.trans_counts += weight * np.outer(gamma[t - 1], gamma[t])
        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.initialize(n, weight, rng)

    def seq_update(self, x, weights, estimate):
        """Update sufficient statistics from encoded observations and weights."""
        emissions_list, priors, _, _ = x
        for i, emissions in enumerate(emissions_list):
            self._accumulate(estimate, emissions, priors[i], float(weights[i]))

    def seq_initialize(self, x, weights, rng):
        """Initialize sufficient statistics from encoded observations and weights."""
        emissions_list, priors, _, _ = x
        for i, emissions in enumerate(emissions_list):
            self.initialize((emissions, priors[i]), float(weights[i]), rng)

    def combine(self, suff_stat):
        """Merge transition, emission, and length sufficient statistics."""
        trans, emissions, length = suff_stat
        self.trans_counts += trans
        for s in range(self.num_states):
            self.accumulators[s].combine(emissions[s])
        self.len_accumulator.combine(length)
        return self

    def value(self):
        """Return transition counts, per-state emission stats, and length stats."""
        return (
            self.trans_counts,
            tuple(acc.value() for acc in self.accumulators),
            self.len_accumulator.value(),
        )

    def from_value(self, x):
        """Restore transition counts, per-state emission stats, and length stats."""
        trans, emissions, length = x
        self.trans_counts = trans
        for s in range(self.num_states):
            self.accumulators[s].from_value(emissions[s])
        self.len_accumulator.from_value(length)
        return self

    def key_merge(self, stats_dict):
        """Merge transition, state, and child statistics into ``stats_dict``."""
        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] = stats_dict[self.trans_key] + self.trans_counts
            else:
                stats_dict[self.trans_key] = self.trans_counts
        if self.state_key is not None:
            if self.state_key in stats_dict:
                acc = stats_dict[self.state_key]
                for i in range(self.num_states):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators
        for acc in self.accumulators:
            acc.key_merge(stats_dict)
        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace transition, state, and child statistics from keyed entries when present."""
        if self.trans_key is not None and self.trans_key in stats_dict:
            self.trans_counts = stats_dict[self.trans_key]
        if self.state_key is not None and self.state_key in stats_dict:
            self.accumulators = stats_dict[self.state_key]
        for acc in self.accumulators:
            acc.key_replace(stats_dict)
        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self):
        """Return the encoder compatible with this accumulator."""
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        return SemiSupervisedHiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder)


class SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for semi-supervised HMM Baum-Welch statistics."""

    def __init__(self, factories, len_factory=None, keys=(None, None)):
        self.factories = factories
        self.len_factory = len_factory
        self.keys = keys

    def make(self):
        """Create an empty semi-supervised HMM accumulator."""
        len_acc = self.len_factory.make() if self.len_factory is not None else NullAccumulator()
        return SemiSupervisedHiddenMarkovEstimatorAccumulator(
            [f.make() for f in self.factories], len_accumulator=len_acc, keys=self.keys
        )


class SemiSupervisedHiddenMarkovEstimator(ParameterEstimator):
    """Estimate transitions, emissions, and optional length from semi-supervised HMM statistics."""

    def __init__(
        self, estimators, len_estimator=None, pseudo_count=None, name=None, keys=(None, None), terminal_states=None
    ):
        self.estimators = list(estimators)
        self.num_states = len(self.estimators)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.name = name
        if keys is None:
            keys = (None, None)
        self.keys = keys
        self.terminal_states = terminal_states

    def accumulator_factory(self):
        """Return a factory for semi-supervised HMM sufficient-statistic accumulators."""
        len_factory = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        )
        return SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory(
            [e.accumulator_factory() for e in self.estimators], len_factory=len_factory, keys=self.keys
        )

    def estimate(self, nobs, suff_stat):
        """Estimate the HMM transition matrix and child distributions from accumulated statistics."""
        trans_counts, emission_stats, length_stat = suff_stat
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        row = trans_counts + pc / self.num_states
        denom = row.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        transitions = row / denom
        topics = [self.estimators[s].estimate(None, emission_stats[s]) for s in range(self.num_states)]
        len_dist = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.estimate(None, length_stat)
        )
        return SemiSupervisedHiddenMarkovModelDistribution(
            topics, transitions, len_dist=len_dist, name=self.name, keys=self.keys, terminal_states=self.terminal_states
        )


class SemiSupervisedHiddenMarkovDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(emission_seq, state_prior)`` observations for the semi-supervised HMM."""

    def __init__(self, emission_encoder, len_encoder=None):
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        return "SemiSupervisedHiddenMarkovDataEncoder(%s, %s)" % (str(self.emission_encoder), str(self.len_encoder))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SemiSupervisedHiddenMarkovDataEncoder)
            and self.emission_encoder == other.emission_encoder
            and self.len_encoder == other.len_encoder
        )

    def seq_encode(self, x):
        """Encode observations as emission lists, priors, optional lengths, and raw lengths."""
        emissions_list = [list(obs[0]) for obs in x]
        priors = [obs[1] for obs in x]
        lengths = np.asarray([len(e) for e in emissions_list], dtype=int)
        len_enc = self.len_encoder.seq_encode(lengths.tolist()) if not supports(self.len_encoder, Neutral) else None
        # emissions are scored per-sequence in the forward; keep the raw lists (the emission encoder is used
        # by the per-state emission distributions through their own log_density)
        return (emissions_list, priors, len_enc, lengths)


# Standard-spelling aliases for the role classes.
SemiSupervisedHiddenMarkovModelDistribution = SemiSupervisedHiddenMarkovModelDistribution
SemiSupervisedHiddenMarkovModelSampler = SemiSupervisedHiddenMarkovSampler
SemiSupervisedHiddenMarkovModelEstimator = SemiSupervisedHiddenMarkovEstimator
SemiSupervisedHiddenMarkovModelDataEncoder = SemiSupervisedHiddenMarkovDataEncoder
SemiSupervisedHiddenMarkovModelAccumulator = SemiSupervisedHiddenMarkovEstimatorAccumulator
SemiSupervisedHiddenMarkovModelAccumulatorFactory = SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory
SemiSupervisedHiddenMarkovAccumulator = SemiSupervisedHiddenMarkovEstimatorAccumulator
SemiSupervisedHiddenMarkovAccumulatorFactory = SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory
