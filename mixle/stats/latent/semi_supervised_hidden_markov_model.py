"""Semi-supervised hidden Markov model: each observation may carry a per-position state prior.

A SemiSupervisedHiddenMarkovModelDistribution is an HMM with shared emissions and transitions in which every
observation can carry soft evidence (a prior) over the hidden state at *each* position of the sequence -- not
only the initial state. An observation is a pair ``(emission_seq, state_prior)``:

    - ``emission_seq``: a length-T sequence of emissions (data type of the emission distributions).
    - ``state_prior``: an optional ``T``-by-``S`` array of finite non-negative likelihood potentials. Row t is
      soft evidence over the S hidden states at position t. Rows are normalized to unit maximum, so their
      absolute scale is irrelevant and an all-ones row imposes no constraint. The HMM's initial state law is
      uniform and is not replaced by the evidence at position 0.

The prior folds into the forward-backward as an extra multiplicative factor on the emission likelihood at every
position, so it shapes both scoring (``log_density``) and the EM E-step. Only the transitions and emissions (and
an optional length distribution) are learned; the priors are given side information. With every prior ``None``
the model is an ordinary HMM with a uniform initial state distribution.

Defines SemiSupervisedHiddenMarkovModelDistribution, SemiSupervisedHiddenMarkovSampler,
SemiSupervisedHiddenMarkovEstimatorAccumulator, SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory,
SemiSupervisedHiddenMarkovEstimator, and SemiSupervisedHiddenMarkovDataEncoder.
"""

import copy
from typing import Any

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
from mixle.stats.latent.effective_sample import (
    heal_pooled_statistics,
    require_finite_count_totals,
    restore_accumulator_statistics,
    snapshot_accumulator_statistics,
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_observation_weights,
    validated_statistic_tuple,
)
from mixle.stats.latent.markov_stopping import (
    DEFAULT_TERMINAL_STEP_CAP,
    require_terminal_reached,
    validate_terminal_reachability,
    validated_terminal_states,
    validated_terminal_step_cap,
)
from mixle.utils.vector import require_possible_log_evidence

_LOG_ZERO = -np.inf


def _as_prior(prior, length: int, num_states: int) -> np.ndarray | None:
    """Return owned, unit-maximum state likelihood potentials, or ``None``."""
    if prior is None:
        return None
    try:
        p = np.asarray(prior, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("state_prior must be a numeric likelihood-potential array.") from exc
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape == (length, num_states):
        pass
    elif p.shape == (1, num_states) and length > 0:
        p = np.repeat(p, length, axis=0)  # a single shared prior row broadcast over the sequence
    else:
        raise ValueError("state_prior must be shape (T=%d, S=%d) or (1, S); got %s" % (length, num_states, p.shape))
    if np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("state_prior likelihood potentials must be finite and non-negative.")
    row_max = p.max(axis=1, keepdims=True)
    if np.any(row_max <= 0.0):
        bad_rows = np.flatnonzero(row_max[:, 0] <= 0.0).tolist()
        raise ValueError(f"state_prior likelihood-potential rows must contain positive evidence; bad rows {bad_rows}.")
    return (p / row_max).copy()


def _validated_semi_supervised_transitions(values: Any, n_states: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return an owned row-stochastic transition matrix and its explicit zero rows.

    The constructor previously did ``np.reshape(np.asarray(values, dtype=float), ...)`` and nothing
    else, so a matrix with NaN entries, negative entries, or rows that do not sum to one was
    accepted and silently turned into ``logTransitions`` full of NaN or of positive log-values
    (MXR-080-1252). Forward/backward then produced a number rather than an error, and nothing in the
    result distinguished it from a real likelihood.

    Zero rows are allowed for the same reason the segmental model allows them: a state with no
    outgoing mass is a *declared terminal* state, not a malformed row. That qualifier is the whole
    contract, and it was previously only asserted here, never enforced -- so an undeclared zero row
    let mass leak out of the model. The zero rows are therefore returned rather than discarded, and
    the constructor checks them against ``terminal_states`` once both are known (MXR-080-1856), which
    is what ``_validated_segmental_transitions`` has always done.
    """
    try:
        transitions = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("semi-supervised HMM transitions must be a numeric matrix.") from exc
    expected = (n_states, n_states)
    if transitions.shape != expected:
        raise ValueError(f"semi-supervised HMM transitions must have shape {expected}, got {transitions.shape}.")
    if np.any(~np.isfinite(transitions)) or np.any(transitions < 0.0):
        raise ValueError("semi-supervised HMM transitions must contain finite non-negative probabilities.")
    row_sums = transitions.sum(axis=1)
    zero_rows = np.flatnonzero(row_sums == 0.0)
    nonzero = row_sums != 0.0
    if not np.allclose(row_sums[nonzero], 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("semi-supervised HMM transition rows must sum to one.")
    return transitions.copy(), tuple(int(index) for index in zero_rows)


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
        self.transitions, zero_transition_rows = _validated_semi_supervised_transitions(transitions, self.nStates)
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
        self.terminal_states = validated_terminal_states(
            terminal_states,
            self.nStates,
            context="SemiSupervisedHiddenMarkovModelDistribution",
        )
        # A zero transition row means "this state has no successor". That is coherent only when the
        # state is declared terminal, because the terminal machinery is what accounts for the mass
        # that stops there. Undeclared, the mass simply vanishes: two identical Bernoulli emissions
        # with transitions [[0, 0], [0, 1]] and no terminal state assigned total length-two mass 0.5
        # while every individual call returned an ordinary finite log-likelihood, so nothing in the
        # result revealed that it was scoring a sub-probability model (MXR-080-1856).
        #
        # This model's initial state law is uniform, so every state carries positive initial mass and
        # every zero row is reachable -- no reachability filter is needed here, unlike the segmental
        # model whose initial weights may be zero.
        undeclared_zero_rows = tuple(
            index for index in zero_transition_rows if self.terminal_states is None or index not in self.terminal_states
        )
        if undeclared_zero_rows:
            raise ValueError(
                "SemiSupervisedHiddenMarkovModelDistribution states with a zero transition row must be declared "
                f"terminal, else their mass leaves the model: undeclared {undeclared_zero_rows}. Pass "
                "terminal_states=... for genuinely absorbing states, or give each row unit mass."
            )
        if self.terminal_states is not None:
            self._terminal_mask = np.zeros(self.nStates, dtype=bool)
            self._terminal_mask[list(self.terminal_states)] = True
            validate_terminal_reachability(
                np.ones(self.nStates, dtype=np.float64),
                self.transitions,
                self.terminal_states,
                context="SemiSupervisedHiddenMarkovModelDistribution",
            )

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
            np.full(self.nStates, -np.log(self.nStates)),
            self.logTransitions,
            self._terminal_log_b(emissions, prior),
            self._terminal_mask,
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
        a = phi[0] / self.nStates
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
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics") and not supports(c, Neutral)]
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

    def _sample_terminal(self, max_steps=DEFAULT_TERMINAL_STEP_CAP):
        """Run the chain (uniform initial) until the first terminal state; emit one observation per state."""
        max_steps = validated_terminal_step_cap(max_steps)
        s = self.dist.nStates
        z = int(self.rng.choice(s))
        states = [z]
        while z not in self.dist.terminal_states and len(states) < max_steps:
            z = int(self.rng.choice(s, p=self.dist.transitions[z]))
            states.append(z)
        require_terminal_reached(
            z in self.dist.terminal_states,
            mode="semi-supervised terminal-state",
            max_steps=max_steps,
            last_state=z,
        )
        return ([self.state_samplers[st].sample() for st in states], None)

    def sample(self, size=None, *, batched: bool = True):
        """Draw one observation or a list of observations with ``None`` state priors."""
        if self.dist.terminal_states is not None:
            return self._sample_terminal() if size is None else [self._sample_terminal() for _ in range(size)]
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class SemiSupervisedHiddenMarkovEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Baum-Welch sufficient statistics for the semi-supervised HMM (transition + emission counts, length)."""

    def __init__(self, accumulators, len_accumulator=None, keys=(None, None)):
        # This model has no learned initial-state distribution, so it intentionally
        # retains a two-key merge contract. State-occupancy counts are nevertheless
        # required to tell each emission estimator its assigned posterior mass.
        self.accumulators = list(accumulators)
        self.num_states = len(self.accumulators)
        self.trans_counts = np.zeros((self.num_states, self.num_states))
        self.state_counts = np.zeros(self.num_states)
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
        a = phi[0] / s
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
            np.full(dist.nStates, -np.log(dist.nStates)),
            dist.logTransitions,
            log_b,
            dist._terminal_mask,
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
            self.state_counts += weight * gamma.sum(axis=0)
            # accumulate emissions vectorized over T: one weighted seq_update per state instead of T*S calls
            enc = self.accumulators[0].acc_to_encoder().seq_encode(list(emissions))
            for s in range(self.num_states):
                self.accumulators[s].seq_update(enc, weight * gamma[:, s], None if dist is None else dist.topics[s])
        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.update(n, weight, None if dist is None else dist.len_dist)

    def update(self, x, weight, estimate):
        """Update Baum-Welch sufficient statistics from one weighted observation."""
        weight = validated_observation_weight(weight)
        emissions, prior = x
        require_possible_log_evidence(
            estimate.log_density(x),
            context="SemiSupervisedHiddenMarkovEstimatorAccumulator.update",
        )
        self._accumulate(estimate, emissions, prior, weight)

    def initialize(self, x, weight, rng):
        """Initialize emission and transition statistics with random soft state assignments."""
        weight = validated_observation_weight(weight)
        emissions, prior = x
        n = len(emissions)
        # random soft responsibilities (respecting the prior's zeros) to break symmetry
        if n > 0:
            p = _as_prior(prior, n, self.num_states)
            gamma = rng.dirichlet(np.ones(self.num_states), size=n)
            if p is not None:
                gamma = gamma * (p > 0)
                gamma = gamma / np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)
            self.state_counts += weight * gamma.sum(axis=0)
            for t in range(n):
                for s in range(self.num_states):
                    self.accumulators[s].initialize(emissions[t], weight * gamma[t, s], rng)
                if t > 0:
                    self.trans_counts += weight * np.outer(gamma[t - 1], gamma[t])
        if not supports(self.len_accumulator, Neutral):
            self.len_accumulator.initialize(n, weight, rng)

    def seq_update(self, x, weights, estimate):
        """Update sufficient statistics from encoded observations and weights."""
        require_possible_log_evidence(
            estimate.seq_log_density(x),
            context="SemiSupervisedHiddenMarkovEstimatorAccumulator.seq_update",
        )
        emissions_list, priors, _, _ = x
        weights = validated_observation_weights(weights, len(emissions_list))
        for i, emissions in enumerate(emissions_list):
            self._accumulate(estimate, emissions, priors[i], weights[i])

    def seq_initialize(self, x, weights, rng):
        """Initialize sufficient statistics from encoded observations and weights."""
        emissions_list, priors, _, _ = x
        weights = validated_observation_weights(weights, len(emissions_list))
        for i, emissions in enumerate(emissions_list):
            self.initialize((emissions, priors[i]), weights[i], rng)

    def combine(self, suff_stat):
        """Merge transition, emission, and length sufficient statistics."""
        trans, state_counts, emissions, length = validated_statistic_tuple(
            suff_stat, 4, "semi-supervised HMM sufficient statistics"
        )
        trans = validated_count_array(trans, (self.num_states, self.num_states), "transition counts")
        state_counts = validated_count_array(state_counts, (self.num_states,), "state counts")
        if len(emissions) != self.num_states:
            raise ValueError("emission statistics must have one item per hidden state")
        validate_effective_sample_mass(
            state_counts.sum(),
            trans.sum(),
            label="semi-supervised HMM transition mass",
            allow_unassigned=True,
        )
        # Transactional with a finiteness postcondition: a child rejecting its part mid-loop
        # used to leave the counts and earlier children merged, and individually valid count
        # arrays can sum to an infinite aggregate (measured in the latent-family mutator audit;
        # STAT-RR8-1/RR9-1 classes).
        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("trans_counts", "state_counts"),
            child_attrs=("accumulators",),
            single_child_attrs=("len_accumulator",),
        )
        self.trans_counts += trans
        self.state_counts += state_counts
        try:
            require_finite_count_totals(
                (("transition counts", self.trans_counts), ("state counts", self.state_counts)),
                label="combined semi-supervised HMM",
            )
            for s in range(self.num_states):
                self.accumulators[s].combine(emissions[s])
            self.len_accumulator.combine(length)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def value(self):
        """Return transition counts, per-state emission stats, and length stats."""
        return (
            self.trans_counts,
            self.state_counts,
            tuple(acc.value() for acc in self.accumulators),
            self.len_accumulator.value(),
        )

    def from_value(self, x):
        """Restore transition counts, per-state emission stats, and length stats."""
        trans, state_counts, emissions, length = validated_statistic_tuple(
            x, 4, "semi-supervised HMM sufficient statistics"
        )
        # Validate EVERYTHING before mutating ANYTHING, then restore transactionally: the
        # previous order assigned the count arrays first, so a rejected restoration or a child
        # failing mid-loop left the accumulator half-replaced (measured; STAT-RR9-1 class).
        candidate_trans = validated_count_array(trans, (self.num_states, self.num_states), "transition counts")
        candidate_state = validated_count_array(state_counts, (self.num_states,), "state counts")
        if len(emissions) != self.num_states:
            raise ValueError("emission statistics must have one item per hidden state")
        validate_effective_sample_mass(
            candidate_state.sum(),
            candidate_trans.sum(),
            label="semi-supervised HMM transition mass",
            allow_unassigned=True,
        )
        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("trans_counts", "state_counts"),
            child_attrs=("accumulators",),
            single_child_attrs=("len_accumulator",),
        )
        self.trans_counts, self.state_counts = candidate_trans, candidate_state
        try:
            for s in range(self.num_states):
                self.accumulators[s].from_value(emissions[s])
            self.len_accumulator.from_value(length)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def key_merge(self, stats_dict):
        """Merge transition, state, and child statistics into ``stats_dict``."""
        # Transactional against the mapping, healed in place on failure: a later pool failing
        # used to leave the transition pool merged, and a failed state-pool merge left the
        # dict-held (shared) emission accumulators combined (measured; STAT-RR9-1/RR10-1
        # classes). Pooling reaches overflow by addition, so merged pools are validated too.
        _snapshot = copy.deepcopy(stats_dict)
        try:
            if self.trans_key is not None:
                if self.trans_key in stats_dict:
                    pooled_trans = stats_dict[self.trans_key] + self.trans_counts
                    require_finite_count_totals(
                        (("pooled transition counts", pooled_trans),),
                        label="semi-supervised HMM key merge",
                    )
                    stats_dict[self.trans_key] = pooled_trans
                else:
                    # Copy on adoption: stats_dict must never alias this accumulator's own live
                    # array. The "already present" branch above is safe (`+` always allocates a
                    # new array), but without this copy a second tied accumulator's key_replace
                    # would still leave both accumulators pointing at this accumulator's own
                    # original array.
                    stats_dict[self.trans_key] = self.trans_counts.copy()
            if self.state_key is not None:
                if self.state_key in stats_dict:
                    counts, acc = stats_dict[self.state_key]
                    pooled_state = counts + self.state_counts
                    require_finite_count_totals(
                        (("pooled state counts", pooled_state),),
                        label="semi-supervised HMM key merge",
                    )
                    stats_dict[self.state_key] = (pooled_state, acc)
                    for i in range(self.num_states):
                        acc[i] = acc[i].combine(self.accumulators[i].value())
                else:
                    stats_dict[self.state_key] = (self.state_counts.copy(), self.accumulators)
            for acc in self.accumulators:
                acc.key_merge(stats_dict)
            self.len_accumulator.key_merge(stats_dict)
        except Exception:
            heal_pooled_statistics(stats_dict, _snapshot)
            raise

    def key_replace(self, stats_dict):
        """Replace transition, state, and child statistics from keyed entries when present."""
        # BOTH candidates validated before EITHER is assigned (replacements used to land with
        # no shape or finiteness checks at all), and the whole replace rolls back on any later
        # failure (measured; STAT-RR8-1/RR9-1 classes).
        candidate_trans = None
        if self.trans_key is not None and self.trans_key in stats_dict:
            candidate_trans = validated_count_array(
                stats_dict[self.trans_key],
                np.shape(self.trans_counts),
                "semi-supervised HMM replacement transition counts",
            )
            require_finite_count_totals(
                (("transition counts", candidate_trans),), label="semi-supervised HMM key replace"
            )
        candidate_state = None
        if self.state_key is not None and self.state_key in stats_dict:
            counts, _pooled_accumulators = stats_dict[self.state_key]
            candidate_state = validated_count_array(
                counts,
                np.shape(self.state_counts),
                "semi-supervised HMM replacement state counts",
            )
            require_finite_count_totals((("state counts", candidate_state),), label="semi-supervised HMM key replace")

        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("trans_counts", "state_counts"),
            child_attrs=("accumulators",),
            single_child_attrs=("len_accumulator",),
        )
        if candidate_trans is not None:
            # Copy on replace too: without it, every tied accumulator ends up pointing at the
            # SAME array object, so any one of them later accumulating new local data would
            # silently corrupt every other tied accumulator's counts.
            self.trans_counts = candidate_trans.copy()
        try:
            if candidate_state is not None:
                _counts, accumulators = stats_dict[self.state_key]
                self.state_counts = candidate_state.copy()
                for index, accumulator in enumerate(self.accumulators):
                    accumulator.from_value(accumulators[index].value())
            for acc in self.accumulators:
                acc.key_replace(stats_dict)
            self.len_accumulator.key_replace(stats_dict)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise

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
        trans_counts, state_counts, emission_stats, length_stat = validated_statistic_tuple(
            suff_stat, 4, "semi-supervised HMM sufficient statistics"
        )
        trans_counts = validated_count_array(trans_counts, (self.num_states, self.num_states), "transition counts")
        state_counts = validated_count_array(state_counts, (self.num_states,), "state counts")
        if len(emission_stats) != self.num_states:
            raise ValueError("emission statistics must have one item per hidden state")
        transition_mass = trans_counts.sum()
        state_mass = state_counts.sum()
        validate_effective_sample_mass(
            state_mass,
            transition_mass,
            label="semi-supervised HMM transition mass",
            allow_unassigned=True,
        )
        if nobs is not None:
            nobs = validated_observation_weight(nobs, "semi-supervised HMM observation mass")
            nonempty_sequence_mass = state_mass - transition_mass
            validate_effective_sample_mass(
                nobs,
                nonempty_sequence_mass,
                label="semi-supervised HMM nonempty-sequence mass",
                allow_unassigned=True,
            )
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        row = trans_counts + pc / self.num_states
        denom = row.sum(axis=1, keepdims=True)
        # A state that won no transition mass this iteration has an undefined M-step row. Setting the
        # denominator to one left that row at ZERO, which is not a distribution: the estimator emitted
        # a sub-probability model whose scores looked ordinary and finite while mass silently left it
        # (MXR-080-1856, estimator half). An unvisited state falls back to uniform -- the standard EM
        # convention, and the only choice here that keeps the matrix row-stochastic. A state declared
        # terminal keeps its zero row, because termination is exactly where its mass is supposed to go.
        empty = denom[:, 0] == 0.0
        if self.terminal_states is not None:
            empty[list(self.terminal_states)] = False
        row[empty, :] = 1.0 / self.num_states
        denom[empty, 0] = 1.0
        denom[denom == 0.0] = 1.0  # remaining zeros are declared-terminal rows; leave them at zero
        transitions = row / denom
        topics = [self.estimators[s].estimate(state_counts[s], emission_stats[s]) for s in range(self.num_states)]
        len_dist = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.estimate(nobs, length_stat)
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

    def row_count(self, x) -> int:
        """Return and validate the number of encoded semi-supervised sequences."""
        if not isinstance(x, tuple) or len(x) != 4:
            raise ValueError("semi-supervised HMM encoding must be a four-slot payload")
        emissions_list, priors, len_enc, lengths = x
        lengths = np.asarray(lengths)
        if lengths.ndim != 1:
            raise ValueError("semi-supervised HMM lengths must be one-dimensional")
        count = len(emissions_list)
        if len(priors) != count or len(lengths) != count:
            raise ValueError("semi-supervised HMM encoded rows, priors, and lengths must align")
        if any(len(emissions) != int(lengths[index]) for index, emissions in enumerate(emissions_list)):
            raise ValueError("semi-supervised HMM encoded lengths do not match emission rows")
        if len_enc is not None and self.len_encoder.row_count(len_enc) != count:
            raise ValueError("semi-supervised HMM encoded length distribution rows do not align")
        return count


# Standard-spelling aliases for the role classes.
SemiSupervisedHiddenMarkovModelDistribution = SemiSupervisedHiddenMarkovModelDistribution
SemiSupervisedHiddenMarkovModelSampler = SemiSupervisedHiddenMarkovSampler
SemiSupervisedHiddenMarkovModelEstimator = SemiSupervisedHiddenMarkovEstimator
SemiSupervisedHiddenMarkovModelDataEncoder = SemiSupervisedHiddenMarkovDataEncoder
SemiSupervisedHiddenMarkovModelAccumulator = SemiSupervisedHiddenMarkovEstimatorAccumulator
SemiSupervisedHiddenMarkovModelAccumulatorFactory = SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory
SemiSupervisedHiddenMarkovAccumulator = SemiSupervisedHiddenMarkovEstimatorAccumulator
SemiSupervisedHiddenMarkovAccumulatorFactory = SemiSupervisedHiddenMarkovEstimatorAccumulatorFactory
