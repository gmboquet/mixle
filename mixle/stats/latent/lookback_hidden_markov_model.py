"""Lookback hidden Markov models with emissions conditioned on recent observations.

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

Note: This is the typed rewrite of the sibling module mixle.stats.lookback_hmm, which is the original
implementation kept stable for the example scripts and external users. The math is identical, but the two
modules differ slightly in their handling of optional arguments: this module substitutes Null*
objects (NullDistribution, NullEstimator, NullDataEncoder, ...) for an absent len_dist/init_dist,
while the sibling uses None (and omits the length term from densities). The
LookbackHiddenMarkovModelDataEncoder constructor signatures also differ (here: encoder first with an
``encoder`` attribute; sibling: lag first with a ``topic_encoder`` attribute).
"""

import copy
from collections.abc import Sequence
from numbers import Real
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.stats.combinator.null_dist import (
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.compute.mixture_evidence import validated_probability_vector
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.latent._hidden_markov_numba_kernels import (
    numba_baum_welch2,
    numba_baum_welch_alphas,
    numba_seq_log_density,
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
)
from mixle.stats.latent.markov_stopping import (
    DEFAULT_TERMINAL_STEP_CAP,
    require_terminal_reached,
    validate_terminal_reachability,
    validated_terminal_states,
    validated_terminal_step_cap,
)
from mixle.stats.sequences.markov_chain import MarkovChainDistribution
from mixle.utils.aliasing import MISSING, broadcast_pseudo_count, coalesce_alias, require

T = TypeVar("T")
E0 = TypeVar("E0")
E1 = TypeVar("E1")


def _validated_lookback_lag(value: Any, *, context: str) -> int:
    """Return an exact non-negative integer lookback lag."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{context} lag must be an integer.")
    lag = int(value)
    if lag < 0:
        raise ValueError(f"{context} lag must be non-negative.")
    return lag


def _owned_sequence(
    values: Any,
    label: str,
    *,
    size: int | None = None,
    minimum: int = 0,
) -> list[Any]:
    """Return an owned list with explicit arity requirements."""
    try:
        owned = list(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable.") from exc
    if len(owned) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} item(s).")
    if size is not None and len(owned) != size:
        raise ValueError(f"{label} must contain exactly {size} item(s), got {len(owned)}.")
    return owned


def _validated_lookback_transition_matrix(values: Any, num_states: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return an owned stochastic transition matrix plus its zero-row indices."""
    try:
        transitions = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("LookbackHiddenMarkovModelDistribution transitions must be a numeric matrix.") from exc
    expected = (num_states, num_states)
    if transitions.shape != expected:
        raise ValueError(
            f"LookbackHiddenMarkovModelDistribution transitions must have shape {expected}, got {transitions.shape}."
        )
    if np.any(~np.isfinite(transitions)) or np.any(transitions < 0.0):
        raise ValueError(
            "LookbackHiddenMarkovModelDistribution transitions must contain finite non-negative probabilities."
        )
    row_sums = transitions.sum(axis=1)
    zero_rows = np.flatnonzero(row_sums == 0.0)
    nonzero_rows = row_sums != 0.0
    if not np.allclose(row_sums[nonzero_rows], 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("LookbackHiddenMarkovModelDistribution transition rows must sum to one.")
    return transitions.copy(), tuple(int(index) for index in zero_rows)


def _reachable_lookback_states(initial: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    """Return the states graph-reachable from positive initial mass."""
    reachable = initial > 0.0
    frontier = list(np.flatnonzero(reachable))
    while frontier:
        state = frontier.pop()
        for child in np.flatnonzero(transitions[state] > 0.0):
            if not reachable[child]:
                reachable[child] = True
                frontier.append(int(child))
    return reachable


def _validated_lookback_keys(values: Any, *, context: str) -> tuple[str | None, str | None, str | None]:
    """Return the three optional accumulator-sharing keys."""
    if values is None:
        return (None, None, None)
    keys = tuple(values)
    if len(keys) != 3:
        raise ValueError(f"{context} keys must contain exactly three entries.")
    if any(value is not None and not isinstance(value, str) for value in keys):
        raise TypeError(f"{context} keys must be strings or None.")
    return keys


def _validated_lookback_pseudo_count(value: Any) -> tuple[float | None, float | None]:
    """Return finite non-negative initial/transition pseudo-count controls."""
    value = broadcast_pseudo_count(value, 2)
    if value is None:
        return (None, None)
    try:
        controls = tuple(value)
    except TypeError as exc:
        raise TypeError("LookbackHiddenMarkovModelEstimator pseudo_count must be a scalar or a pair.") from exc
    if len(controls) != 2:
        raise ValueError("LookbackHiddenMarkovModelEstimator pseudo_count must contain exactly two entries.")
    result: list[float | None] = []
    for index, control in enumerate(controls):
        if control is None:
            result.append(None)
            continue
        if isinstance(control, bool) or not isinstance(control, Real):
            raise TypeError(f"LookbackHiddenMarkovModelEstimator pseudo_count[{index}] must be a real number or None.")
        numeric = float(control)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"LookbackHiddenMarkovModelEstimator pseudo_count[{index}] must be finite and non-negative."
            )
        result.append(numeric)
    return result[0], result[1]


def _validated_lookback_count_array(values: Any, label: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return an owned finite non-negative sufficient-statistic count array."""
    try:
        counts = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric array.") from exc
    if counts.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {counts.shape}.")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError(f"{label} must contain finite non-negative values.")
    return counts.copy()


def _validate_lookback_dirichlet_geometry(prior: Any, label: str, num_states: int) -> None:
    """Require a conjugate chain prior to cover exactly the hidden-state simplex."""
    try:
        parameters = np.asarray(prior.get_parameters(), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must expose a numeric concentration vector.") from exc
    if parameters.shape != (num_states,):
        raise ValueError(f"{label} must contain exactly {num_states} concentrations.")


def _validated_lookback_sufficient_statistics(
    suff_stat: Any,
    *,
    lag: int,
    num_states: int,
    context: str,
    init_key: str | None = None,
    trans_key: str | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, list[Any], list[Any], Any]:
    """Validate the complete sufficient-statistic schema before mutation or estimation."""
    try:
        values = tuple(suff_stat)
    except TypeError as exc:
        raise TypeError(f"{context} sufficient statistics must be an eight-item tuple.") from exc
    if len(values) != 8:
        raise ValueError(f"{context} sufficient statistics must contain exactly eight items.")
    observed_lag = _validated_lookback_lag(values[0], context=f"{context} sufficient-statistic")
    if observed_lag != lag:
        raise ValueError(f"{context} sufficient-statistic lag {observed_lag} does not match estimator lag {lag}.")
    observed_states = values[1]
    if isinstance(observed_states, bool) or not isinstance(observed_states, (int, np.integer)):
        raise TypeError(f"{context} sufficient-statistic state count must be an integer.")
    if int(observed_states) != num_states:
        raise ValueError(f"{context} sufficient-statistic state count {observed_states} does not match {num_states}.")
    init_counts = _validated_lookback_count_array(values[2], f"{context} initial-state counts", (num_states,))
    state_counts = _validated_lookback_count_array(values[3], f"{context} state counts", (num_states,))
    trans_counts = _validated_lookback_count_array(values[4], f"{context} transition counts", (num_states, num_states))
    # Mode-appropriate mass relation, shared with the chain HMM: equality when the initial and
    # transition parts are unkeyed, the pooled upper bound otherwise -- a KEYED accumulator's
    # own value() carries pooled parts, and the previous unconditional equality rejected its
    # own round-trip and every keyed fit's M-step (measured by the latent-family mutator
    # audit; the chain twin learned this in STAT-RR5-2).
    from mixle.stats.latent.hidden_markov import _validate_state_mass

    _validate_state_mass(
        init_counts,
        state_counts,
        trans_counts,
        init_key=init_key,
        trans_key=trans_key,
        label=f"{context} hidden-state mass",
    )
    topic_ss = _owned_sequence(values[5], f"{context} topic sufficient statistics", size=num_states)
    init_ss = _owned_sequence(values[6], f"{context} initial sufficient statistics", size=num_states)
    return lag, num_states, init_counts, state_counts, trans_counts, topic_ss, init_ss, values[7]


class LookbackHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
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
        terminal_states: set[int] | Sequence[int] | None = None,
        prior=None,
    ) -> None:
        """Distribution for sequences with lagged hidden-state emission dependence.

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
            name (Optional[str]): Optional distribution name.
            prior: Optional conjugate chain prior over w/transitions, ``(init_prior, row_priors)``
                -- mirrors :class:`~mixle.stats.latent.hidden_markov.HiddenMarkovModelDistribution`.

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
            name (Optional[str]): Optional distribution name.

        """
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        transitions = require("transitions", transitions, default=MISSING)
        context = "LookbackHiddenMarkovModelDistribution"
        self.lag = _validated_lookback_lag(lag, context=context)
        self.topics = _owned_sequence(topics, f"{context} topics", minimum=1)
        self.w = validated_probability_vector(w, f"{context} initial weights")
        self.num_states = len(self.w)
        self.num_topics = len(self.topics)
        if self.num_topics != self.num_states:
            raise ValueError(
                f"{context} requires one topic per state: got {self.num_topics} topics and {self.num_states} states."
            )
        if init_dist is None:
            if self.lag > 0:
                raise ValueError(f"{context} requires one initial distribution per state when lag is positive.")
            self.init_dist = [NullDistribution() for _ in range(self.num_states)]
        else:
            self.init_dist = _owned_sequence(
                init_dist,
                f"{context} initial distributions",
                size=self.num_states,
            )
        self.terminal_states = validated_terminal_states(
            terminal_states,
            self.num_states,
            context=context,
        )
        self.transitions, zero_transition_rows = _validated_lookback_transition_matrix(
            transitions,
            self.num_states,
        )
        reachable = _reachable_lookback_states(self.w, self.transitions)
        reachable_zero_rows = tuple(
            index
            for index in zero_transition_rows
            if reachable[index] and (self.terminal_states is None or index not in self.terminal_states)
        )
        if reachable_zero_rows:
            raise ValueError(
                f"{context} reachable non-terminal states cannot have zero transition rows: {reachable_zero_rows}."
            )
        self.unreachable_transition_rows = zero_transition_rows
        for index in zero_transition_rows:
            self.transitions[index, index] = 1.0
        with np.errstate(divide="ignore"):
            self.log_w = np.log(self.w)
            self.log_transitions = np.log(self.transitions)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.name = name
        if self.terminal_states is not None:
            self._terminal_mask = np.zeros(self.num_states, dtype=bool)
            self._terminal_mask[list(self.terminal_states)] = True
        validate_terminal_reachability(
            self.w,
            self.transitions,
            self.terminal_states,
            context=context,
        )
        self.set_prior(prior)

    def get_prior(self):
        """Returns the chain conjugate prior in ``(init_prior, row_priors)`` form (or None).

        Per-state emission/init component priors are owned by those distributions themselves --
        mirrors :meth:`~mixle.stats.latent.hidden_markov.HiddenMarkovModelDistribution.get_prior`.
        """
        if not self.has_conj_prior:
            return None
        return (self.init_prior, list(self.row_priors))

    def set_prior(self, prior) -> None:
        """Set the conjugate Dirichlet chain prior over w/transitions and flag ``has_conj_prior``.

        Mirrors :meth:`~mixle.stats.latent.hidden_markov.HiddenMarkovModelDistribution.set_prior`
        exactly (no digamma-expectation caching: this class has no ``expected_log_density``
        counterpart to feed). ``prior=None`` leaves the distribution a plain point model.

        Args:
            prior: ``(init_prior, row_priors)`` tuple or None.
        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.latent.hidden_markov import _unpack_hmm_chain_prior

        if prior is None:
            self.prior = None
            self.init_prior = None
            self.row_priors = None
            self.has_conj_prior = False
            return

        try:
            prior_values = tuple(prior)
        except TypeError as exc:
            raise TypeError(
                "LookbackHiddenMarkovModelDistribution prior must be an initial prior and row-prior sequence."
            ) from exc
        if len(prior_values) != 2:
            raise ValueError("LookbackHiddenMarkovModelDistribution prior must contain exactly two entries.")
        init_prior, row_priors = _unpack_hmm_chain_prior(prior_values)
        row_priors = _owned_sequence(
            row_priors,
            "LookbackHiddenMarkovModelDistribution transition-row priors",
            size=self.num_states,
        )
        self.prior = prior
        self.init_prior = init_prior
        self.row_priors = row_priors
        self.has_conj_prior = isinstance(init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in row_priors
        )
        if self.has_conj_prior:
            _validate_lookback_dirichlet_geometry(
                init_prior,
                "LookbackHiddenMarkovModelDistribution initial prior",
                self.num_states,
            )
            for index, row_prior in enumerate(row_priors):
                _validate_lookback_dirichlet_geometry(
                    row_prior,
                    f"LookbackHiddenMarkovModelDistribution row prior {index}",
                    self.num_states,
                )
        self.prior = (init_prior, tuple(row_priors))

    def _windowed_log_b(self, x: Sequence[Any]) -> np.ndarray:
        """Per-position, per-state emission log-densities ``(obs_cnt, num_states)`` for the lookback windows."""
        lag, ns = self.lag, self.num_states
        obs_cnt = len(x) - lag + 1 if lag > 0 else len(x)
        log_b = np.empty((obs_cnt, ns))
        for i in range(ns):
            log_b[0, i] = self.init_dist[i].log_density(x[:lag]) if lag > 0 else self.topics[i].log_density(x[0:1])
        for idx, k in enumerate(range(max(lag, 1), len(x))):
            for i in range(ns):
                log_b[idx + 1, i] = self.topics[i].log_density(x[(k - lag) : (k + 1)])
        return log_b

    def _terminal_states_log_density(self, x: Sequence[Any]) -> float:
        """Stopping-time likelihood for the lookback HMM (shared terminal forward over windowed emissions)."""
        from mixle.stats.latent.hidden_markov import terminal_forward_loglik

        if len(x) < max(self.lag, 1):
            return -np.inf
        return terminal_forward_loglik(self.log_w, self.log_transitions, self._windowed_log_b(x), self._terminal_mask)

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = ",".join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        s4 = repr(self.lag)
        s5 = ",".join(map(str, self.init_dist))
        s6 = str(self.len_dist)
        s7 = repr(self.name)

        return "LookbackHiddenMarkovModelDistribution([%s], %s, %s, lag=%s, init_dist=[%s], len_dist=%s, name=%s)" % (
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
        if self.terminal_states is not None:
            return self._terminal_states_log_density(x)

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
        if self.terminal_states is not None:
            # terminal-state lookback HMMs encode raw sequences (passthrough); score per sequence
            return np.array([self._terminal_states_log_density(s) for s in x], dtype=np.float64)

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
        with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> NaN; zeroed below
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)
        pr_obs[np.isnan(pr_obs).any(axis=1), :] = 0.0  # impossible observation -> zero emission row

        alpha_buff = np.zeros((num_seq, num_states), dtype=np.float64)
        next_alpha = np.zeros((num_seq, num_states), dtype=np.float64)

        numba_seq_log_density(num_states, tz, pr_obs, w, A, pr_max0, next_alpha, alpha_buff, ll_ret)

        ll_ret += self.len_dist.seq_log_density(len_enc)

        return ll_ret

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete lookback-HMM instance."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.topics)
        if self.lag > 0 and self.init_dist is not None:
            children = children + tuple(self.init_dist)
        if self.len_dist is not None:
            children = children + (self.len_dist,)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def backend_seq_log_density(self, x, engine):
        """Engine-neutral lookback-HMM scoring via the shared HMM forward pass."""
        from mixle.stats.compute.backend import backend_seq_log_density
        from mixle.stats.latent.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

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

        # See the identical removal (and its comment) in seq_update: `max_len` was never read, and
        # `sz.max()` crashes on a zero-sequence corpus.
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = self.w
        tran_mat = self.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            if self.lag > 0:
                pr_obs[imi, i] = self.init_dist[i].seq_log_density(enc_idata)
            pr_obs[ims, i] = self.topics[i].seq_log_density(enc_sdata)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> NaN; zeroed below
            pr_obs -= pr_max
            np.exp(pr_obs, out=pr_obs)
        pr_obs[np.isnan(pr_obs).any(axis=1), :] = 0.0  # impossible observation -> zero emission row

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch_alphas(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)

        return [alphas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from child models."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.topics) + ([] if self.len_dist is None else [self.len_dist])
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics") and not supports(c, Neutral)]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> "LookbackHiddenMarkovModelSampler":
        """Create a LookbackHiddenMarkovModelSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for random number generator.

        Returns:
            LookbackHiddenMarkovModelSampler: Sampler object (requires a non-null len_dist).

        """
        return LookbackHiddenMarkovModelSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Not supported: lookback emissions condition on the previous ``lag`` observations.

        Unlike the standard / segmental HMM, each emission depends on the preceding ``lag`` symbols,
        so the effective state is ``(hidden_state, last lag observations)``. Best-first enumeration
        over that augmented (and, for large/continuous alphabets, unbounded) state space is not
        implemented; use :meth:`sampler` or the exact ``log_density`` / ``viterbi_sequence`` instead.
        """
        raise EnumerationError(
            self,
            reason="lookback emissions condition on the previous lag observations; enumeration over the "
            "augmented (state, observation-history) space is not supported",
        )

    def estimator(self, pseudo_count: float | None = None) -> "LookbackHiddenMarkovModelEstimator":
        """Create a LookbackHiddenMarkovModelEstimator from this distribution.

        Args:
            pseudo_count (Optional[float]): Regularize the initial-state and transition estimates.

        Returns:
            LookbackHiddenMarkovModelEstimator: Estimator built from the topic, initial-segment, and
                length distributions, preserving the lag.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        init_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.init_dist]
        return LookbackHiddenMarkovModelEstimator(
            comp_ests,
            lag=self.lag,
            init_estimators=init_ests,
            len_estimator=len_est,
            pseudo_count=(pseudo_count, pseudo_count),
            name=self.name,
            terminal_states=self.terminal_states,
        )

    def seq_encode(self, x: Sequence[Sequence[T]]):
        """Encode a sequence of observed sequences for vectorized ``seq_`` calls.

        Args:
            x (Sequence[Sequence[T]]): Sequence of iid observed sequences.

        Returns:
            Encoded data consistent with seq_log_density(), seq_posterior(), and seq_update().

        """
        return self.dist_to_encoder().seq_encode(x)

    def dist_to_encoder(self) -> "LookbackHiddenMarkovModelDataEncoder":
        """Return a LookbackHiddenMarkovModelDataEncoder for encoding sequences of iid observations.

        Returns:
            LookbackHiddenMarkovModelDataEncoder: Encoder built from the topic, initial, and length
                distributions of this instance.

        """
        if self.terminal_states is not None:
            return LookbackTerminalDataEncoder()

        encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        init_encoder = self.init_dist[0].dist_to_encoder()

        return LookbackHiddenMarkovModelDataEncoder(
            encoder=encoder, len_encoder=len_encoder, init_encoder=init_encoder, lag=self.lag
        )


class LookbackTerminalDataEncoder(DataSequenceEncoder):
    """Passthrough encoder for terminal-state lookback HMMs: keeps raw sequences (scored per sequence)."""

    def __str__(self) -> str:
        return "LookbackTerminalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LookbackTerminalDataEncoder)

    def seq_encode(self, x):
        """Encode terminal-aware lookback HMM sequences as lists."""
        return [list(s) for s in x]


class LookbackHiddenMarkovModelSampler(DistributionSampler):
    """Sampler for LookbackHiddenMarkovModelDistribution. Requires non-null init_dist and len_dist."""

    def __init__(self, dist: LookbackHiddenMarkovModelDistribution, seed: int | None = None) -> None:
        """Create a sampler for a lookback hidden Markov model.

        Args:
            dist (LookbackHiddenMarkovModelDistribution): Distribution to sample from (init_dist and
                len_dist must be set, and topics must support sample_given()).
            seed (Optional[int]): Seed for random number generator.

        """
        self.num_states = dist.num_states
        self.dist = dist
        self.rng = RandomState(seed)

        init_seeds = [self.rng.randint(0, maxrandint) for _ in range(dist.num_states)]
        self.init_samplers = (
            [dist.init_dist[i].sampler(seed=init_seeds[i]) for i in range(dist.num_states)] if dist.lag > 0 else []
        )
        self.obs_samplers = [
            getattr(dist.topics[i], "transition_sampler", dist.topics[i].sampler)(seed=self.rng.randint(0, maxrandint))
            for i in range(dist.num_states)
        ]
        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

        t_map = {i: {k: dist.transitions[i, k] for k in range(dist.num_states)} for i in range(dist.num_states)}
        p_map = {i: dist.w[i] for i in range(dist.num_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).path_sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw iid sequences from the lookback hidden Markov distribution.

        Args:
            size (Optional[int]): Number of sequences to draw. If None, a single sequence is returned.

        Returns:
            Union[List[T], List[List[T]]]: One sampled sequence if size is None, else a list of
                ``size`` sampled sequences.

        """
        if self.dist.terminal_states is not None:
            if size is None:
                return self._sample_terminal()
            return [self._sample_terminal() for _ in range(size)]

        if size is None:
            lag = self.dist.lag
            n = self.len_sampler.sample()
            state_seq = self.state_sampler.sample_seq(n)

            if lag == 0:
                # ordinary HMM: each of the n states emits one observation given an empty history
                return [_emit_unconditioned(self.obs_samplers[state_seq[i]]) for i in range(n)]

            rv = list(self.init_samplers[state_seq[0]].sample())  # [v_1, ..., v_lag]
            for i in range(1, n):
                rv.append(self.obs_samplers[state_seq[i]].sample_given(rv[-lag:]))
            return rv
        else:
            return [self.sample() for i in range(size)]

    def _sample_terminal(self, max_steps: int = DEFAULT_TERMINAL_STEP_CAP):
        """Run the chain until the first terminal (absorbing) state, emitting the lookback windows."""
        max_steps = validated_terminal_step_cap(max_steps)
        lag = self.dist.lag
        z = int(self.state_sampler.sample_seq())
        states = [z]
        while z not in self.dist.terminal_states and len(states) < max_steps:
            z = int(self.state_sampler.sample_seq(v0=z))
            states.append(z)
        require_terminal_reached(
            z in self.dist.terminal_states,
            mode="lookback terminal-state",
            max_steps=max_steps,
            last_state=z,
        )
        if lag == 0:
            return [_emit_unconditioned(self.obs_samplers[s]) for s in states]
        rv = list(self.init_samplers[states[0]].sample())
        for s in states[1:]:
            rv.append(self.obs_samplers[s].sample_given(rv[-lag:]))
        return rv


def _emit_unconditioned(sampler):
    """One draw from an emission sampler with no history to condition on (the lag == 0 case).

    At lag 0 the model is an ordinary HMM: there is no lookback window, so "emit given an empty
    history" is just an unconditional draw. Calling sample_given([]) unconditionally demanded a
    conditional sampler interface that the emission does not need and mostly does not have --
    GaussianSampler and SequenceSampler have no sample_given at all, and the conditional laws that
    do have it reject an empty path when their own lag is >= 1. That left the lag == 0 sampling
    path with no usable emission type, including the SequenceDistribution emissions this model's
    own engine test builds (that test only fits and scores, so it never reached this line).
    """
    given = getattr(sampler, "sample_given", None)
    return given([]) if callable(given) else sampler.sample()


class LookbackHiddenMarkovModelEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for sufficient statistics of a lookback hidden Markov model."""

    def __init__(
        self,
        seq_accumulators,
        init_accumulators=None,
        lag=0,
        len_accumulator=None,
        keys=(None, None, None),
        terminal_states=None,
    ):
        """Create an accumulator for lookback-HMM sufficient statistics.

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
        context = "LookbackHiddenMarkovModelEstimatorAccumulator"
        self.lag = _validated_lookback_lag(lag, context=context)
        self.seq_accumulators = _owned_sequence(
            seq_accumulators,
            f"{context} sequence accumulators",
            minimum=1,
        )
        self.num_states = len(self.seq_accumulators)
        if init_accumulators is None:
            if self.lag > 0:
                raise ValueError(f"{context} requires one initial accumulator per state when lag is positive.")
            self.init_accumulators = [NullAccumulatorFactory().make() for _ in range(self.num_states)]
        else:
            self.init_accumulators = _owned_sequence(
                init_accumulators,
                f"{context} initial accumulators",
                size=self.num_states,
            )
        self.init_counts = vec.zeros(self.num_states)
        self.trans_counts = vec.zeros((self.num_states, self.num_states))
        self.state_counts = vec.zeros(self.num_states)
        self.len_accumulator = len_accumulator
        self.terminal_states = validated_terminal_states(terminal_states, self.num_states, context=context)

        self.init_key, self.trans_key, self.state_key = _validated_lookback_keys(keys, context=context)

        # When _track_ll is enabled, seq_update accumulates the per-sequence data
        # log-likelihood into _seq_ll. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); default path is unchanged and zero-cost.
        self._track_ll = False
        self._seq_ll = 0.0

    def _terminal_update(self, x, weight, estimate):
        """Terminal-state Baum-Welch E-step for one raw lookback sequence (windowed forward-backward)."""
        from mixle.stats.latent.hidden_markov import terminal_forward_backward

        if len(x) < max(self.lag, 1):
            return
        log_b = estimate._windowed_log_b(x)
        _, gamma, xi = terminal_forward_backward(
            estimate.log_w, estimate.log_transitions, log_b, estimate._terminal_mask
        )
        if gamma is None:
            return
        lag = self.lag
        w = weight * gamma  # per-position state responsibilities (obs_cnt, num_states)
        self.init_counts += w[0]
        self.state_counts += w.sum(axis=0)
        self.trans_counts += weight * xi.sum(axis=0)
        if lag > 0:
            for j in range(self.num_states):
                self.init_accumulators[j].update(x[:lag], w[0, j], estimate.init_dist[j])
            for k, i in enumerate(range(lag, len(x))):
                for j in range(self.num_states):
                    self.seq_accumulators[j].update(x[(i - lag) : (i + 1)], w[k + 1, j], estimate.topics[j])
        else:
            for k in range(len(x)):
                for j in range(self.num_states):
                    self.seq_accumulators[j].update(x[k : (k + 1)], w[k, j], estimate.topics[j])

    def update(self, x, weight, estimate):
        """Update sufficient statistics with one observed sequence and weight.

        Args:
            x (Sequence[T]): Observed sequence.
            weight (float): Weight for the observation.
            estimate (LookbackHiddenMarkovModelDistribution): Current estimate used for the E-step.

        """
        weight = validated_observation_weight(weight)
        if estimate.terminal_states is not None:
            self._terminal_update(x, weight, estimate)
            return
        self.seq_update(estimate.seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x, weight, rng):
        """Initialize sufficient statistics with one observed sequence using random state weights.

        Args:
            x (Sequence[T]): Observed sequence.
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator for the random state assignment.

        """
        weight = validated_observation_weight(weight)
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
        if self.terminal_states is not None:
            weights = validated_observation_weights(weights, len(x))
            for s, wt in zip(x, weights):
                self.initialize(s, wt, rng)
            return

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x
        weights = validated_observation_weights(weights, len(sz))

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
        # An all-empty corpus (every sequence length 0) makes tot_cnt 0, so tz[1:] - 1 is all -1 and
        # there is no "last position" for it to name -- prev_mask is already empty and correct as-is.
        if tot_cnt > 0:
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

    def acc_to_encoder(self) -> "LookbackHiddenMarkovModelDataEncoder":
        """Return a LookbackHiddenMarkovModelDataEncoder consistent with this accumulator.

        Returns:
            LookbackHiddenMarkovModelDataEncoder: Encoder built from the member accumulators.

        """
        if self.terminal_states is not None:
            return LookbackTerminalDataEncoder()

        encoder = self.seq_accumulators[0].acc_to_encoder()
        init_encoder = self.init_accumulators[0].acc_to_encoder() if self.init_accumulators else NullDataEncoder()
        len_encoder = self.len_accumulator.acc_to_encoder() if self.len_accumulator is not None else NullDataEncoder()

        return LookbackHiddenMarkovModelDataEncoder(
            encoder=encoder, lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder
        )

    def seq_update(self, x, weights, estimate):
        """Vectorized Baum-Welch update of sufficient statistics with encoded sequences.

        Args:
            x: Encoded sequence data produced by acc_to_encoder() (or a matching dist encoder).
            weights (np.ndarray): Weight for each encoded sequence.
            estimate (LookbackHiddenMarkovModelDistribution): Current estimate used for the E-step.

        """
        if estimate.terminal_states is not None:
            weights = validated_observation_weights(weights, len(x))
            for s, wt in zip(x, weights):
                self._terminal_update(s, wt, estimate)
            return

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x
        weights = validated_observation_weights(weights, len(sz))

        tot_cnt = len(ids) + len(idi)
        seq_cnt = len(sz)
        num_states = estimate.num_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

        # `max_len` was computed here but never read (dead since introduction) -- and `sz.max()`
        # raises `ValueError: zero-size array to reduction operation maximum which has no
        # identity` on a zero-sequence corpus (data=[], distinct from a corpus of individually-
        # empty sequences like [[],[],[]], which seq_initialize already tolerates). Removing the
        # dead computation is a true no-op for every other input and closes this crash.
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = estimate.w
        tran_mat = estimate.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            if self.lag > 0:
                pr_obs[imi, i] = estimate.init_dist[i].seq_log_density(enc_idata)
            pr_obs[ims, i] = estimate.topics[i].seq_log_density(enc_sdata)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> NaN; zeroed below
            pr_obs -= pr_max
            np.exp(pr_obs, out=pr_obs)
        pr_obs[np.isnan(pr_obs).any(axis=1), :] = 0.0  # impossible observation -> zero emission row

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
        from mixle.stats.compute.backend import backend_seq_log_density
        from mixle.stats.latent.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x
        num_states = estimate.num_states
        tot_cnt = len(ids) + len(idi)
        weights_np = validated_observation_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            len(sz),
        )

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
            LookbackHiddenMarkovModelEstimatorAccumulator: This accumulator after aggregation.

        """
        (
            _,
            _,
            init_counts,
            state_counts,
            trans_counts,
            seq_accumulators,
            init_accumulators,
            len_acc,
        ) = _validated_lookback_sufficient_statistics(
            suff_stat,
            lag=self.lag,
            num_states=self.num_states,
            context="LookbackHiddenMarkovModelEstimatorAccumulator.combine",
        )

        # The ENTIRE combine is transactional with a finiteness postcondition: a child
        # rejecting its part mid-loop used to leave the counts and earlier children merged,
        # and individually valid count arrays can sum to an infinite aggregate (measured in
        # the latent-family mutator audit; STAT-RR8-1/RR9-1 classes).
        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("init_counts", "state_counts", "trans_counts"),
            child_attrs=("seq_accumulators", "init_accumulators"),
            single_child_attrs=("len_accumulator",),
        )
        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts
        try:
            require_finite_count_totals(
                (
                    ("initial counts", self.init_counts),
                    ("state counts", self.state_counts),
                    ("transition counts", self.trans_counts),
                ),
                label="combined lookback-HMM",
            )
            for i in range(self.num_states):
                self.init_accumulators[i].combine(init_accumulators[i])
                self.seq_accumulators[i].combine(seq_accumulators[i])

            if self.len_accumulator is not None and len_acc is not None:
                self.len_accumulator.combine(len_acc)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise

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
            LookbackHiddenMarkovModelEstimatorAccumulator: This accumulator after assignment.

        """
        # Candidates validated BEFORE any assignment, with the mode-appropriate mass relation
        # (a keyed accumulator's own value() carries pooled parts; the previous unconditional
        # equality rejected its own round-trip -- measured), and assignment plus child
        # restoration run as one transaction (STAT-RR9-1 class).
        (
            _,
            _,
            candidate_init,
            candidate_state,
            candidate_trans,
            seq_accumulators,
            init_accumulators,
            len_acc,
        ) = _validated_lookback_sufficient_statistics(
            x,
            lag=self.lag,
            num_states=self.num_states,
            context="LookbackHiddenMarkovModelEstimatorAccumulator.from_value",
            init_key=self.init_key,
            trans_key=self.trans_key,
        )

        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("init_counts", "state_counts", "trans_counts"),
            child_attrs=("seq_accumulators", "init_accumulators"),
            single_child_attrs=("len_accumulator",),
        )
        self.init_counts, self.state_counts, self.trans_counts = candidate_init, candidate_state, candidate_trans
        try:
            for i, v in enumerate(init_accumulators):
                self.init_accumulators[i].from_value(v)

            for i, v in enumerate(seq_accumulators):
                self.seq_accumulators[i].from_value(v)

            if self.len_accumulator is not None:
                self.len_accumulator.from_value(len_acc)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise

        return self

    def scale(self, c: float) -> "LookbackHiddenMarkovModelEstimatorAccumulator":
        """Scale all accumulated lookback-HMM sufficient statistics in place."""
        # The factor is validated like every sibling family's (an infinite or negative factor
        # used to be applied silently), and parent counts plus every child scale as ONE
        # transaction with the scaled result validated as a postcondition (measured;
        # STAT-RR8-1/RR10-1 classes).
        c = validated_observation_weight(c, "lookback-HMM statistic scale")
        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("init_counts", "state_counts", "trans_counts"),
            child_attrs=("seq_accumulators", "init_accumulators"),
            single_child_attrs=("len_accumulator",),
        )
        self.init_counts *= c
        self.state_counts *= c
        self.trans_counts *= c
        try:
            require_finite_count_totals(
                (
                    ("initial counts", self.init_counts),
                    ("state counts", self.state_counts),
                    ("transition counts", self.trans_counts),
                ),
                label="scaled lookback-HMM",
            )
            for acc in self.init_accumulators:
                acc.scale(c)
            for acc in self.seq_accumulators:
                acc.scale(c)
            if self.len_accumulator is not None:
                self.len_accumulator.scale(c)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            raise
        return self

    def key_merge(self, stats_dict):
        """Merge keyed sufficient statistics of this accumulator into stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to merged sufficient statistics.

        """
        # The WHOLE merge is transactional against the mapping, healed in place on failure: a
        # later pool failing used to leave the initial pool merged, visible through any caller-
        # held alias, and pooling reaches overflow by addition exactly as combine() does
        # (measured; STAT-RR8-1/RR9-1/RR10-1 classes). The snapshot deep-copies the dict on
        # entry because the child recursions below touch keys this level cannot enumerate.
        _snapshot = copy.deepcopy(stats_dict)
        try:
            if self.init_key is not None:
                if self.init_key in stats_dict:
                    stats_dict[self.init_key] += self.init_counts
                    require_finite_count_totals(
                        (("pooled initial counts", stats_dict[self.init_key]),),
                        label="lookback-HMM key merge",
                    )
                else:
                    # Copy on adoption: stats_dict must never alias this accumulator's own live
                    # array, or a later tied accumulator's in-place += above would silently mutate
                    # this accumulator's private init_counts as a side effect of merging.
                    stats_dict[self.init_key] = self.init_counts.copy()

            if self.trans_key is not None:
                if self.trans_key in stats_dict:
                    stats_dict[self.trans_key] += self.trans_counts
                    require_finite_count_totals(
                        (("pooled transition counts", stats_dict[self.trans_key]),),
                        label="lookback-HMM key merge",
                    )
                else:
                    # Same aliasing hazard as init_key above, for the transition-count matrix.
                    stats_dict[self.trans_key] = self.trans_counts.copy()

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
        except Exception:
            heal_pooled_statistics(stats_dict, _snapshot)
            raise

    def key_replace(self, stats_dict):
        """Replace keyed sufficient statistics of this accumulator with values from stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to merged sufficient statistics.

        """
        # BOTH count candidates are validated before EITHER is assigned (replacements used to
        # land with no shape or finiteness checks at all), and the whole replace -- including
        # the mapping's own objects, which the recursion below mutates through ADOPTED children
        # -- rolls back on any later failure (measured; STAT-RR8-1/RR9-1/RR10-1 classes).
        candidate_init = None
        if self.init_key is not None and self.init_key in stats_dict:
            candidate_init = validated_count_array(
                stats_dict[self.init_key],
                np.shape(self.init_counts),
                "lookback-HMM replacement initial counts",
            )
            require_finite_count_totals((("initial counts", candidate_init),), label="lookback-HMM key replace")
        candidate_trans = None
        if self.trans_key is not None and self.trans_key in stats_dict:
            candidate_trans = validated_count_array(
                stats_dict[self.trans_key],
                np.shape(self.trans_counts),
                "lookback-HMM replacement transition counts",
            )
            require_finite_count_totals((("transition counts", candidate_trans),), label="lookback-HMM key replace")

        _snapshot = snapshot_accumulator_statistics(
            self,
            count_attrs=("init_counts", "state_counts", "trans_counts"),
            child_attrs=("seq_accumulators", "init_accumulators"),
            single_child_attrs=("len_accumulator",),
        )
        _dict_snapshot = copy.deepcopy(stats_dict)
        if candidate_init is not None:
            # Copy on replace too: without it, every tied accumulator ends up pointing at
            # the SAME array object, so any one of them later accumulating new local data
            # would silently corrupt every other tied accumulator's counts.
            self.init_counts = candidate_init.copy()
        if candidate_trans is not None:
            self.trans_counts = candidate_trans.copy()
        if self.state_key is not None and self.state_key in stats_dict:
            self.seq_accumulators = stats_dict[self.state_key]
        try:
            for u in self.init_accumulators:
                u.key_replace(stats_dict)

            for u in self.seq_accumulators:
                u.key_replace(stats_dict)

            if self.len_accumulator is not None:
                self.len_accumulator.key_replace(stats_dict)
        except Exception:
            restore_accumulator_statistics(self, _snapshot)
            heal_pooled_statistics(stats_dict, _dict_snapshot)
            raise


class LookbackHiddenMarkovModelEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating LookbackHiddenMarkovModelEstimatorAccumulator objects."""

    def __init__(
        self,
        lag: int,
        seq_factories: Sequence[StatisticAccumulatorFactory],
        init_factories: Sequence[StatisticAccumulatorFactory] | None = None,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        terminal_states=None,
    ):
        """Create a factory for lookback-HMM accumulators.

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
        context = "LookbackHiddenMarkovModelEstimatorAccumulatorFactory"
        self.lag = _validated_lookback_lag(lag, context=context)
        self.seq_factories = _owned_sequence(
            seq_factories,
            f"{context} sequence factories",
            minimum=1,
        )
        self.keys = _validated_lookback_keys(keys, context=context)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.terminal_states = validated_terminal_states(terminal_states, len(self.seq_factories), context=context)

        if init_factories is None:
            if self.lag > 0:
                raise ValueError(f"{context} requires one initial factory per state when lag is positive.")
            self.init_factories = [NullAccumulatorFactory() for _ in range(len(self.seq_factories))]
        else:
            self.init_factories = _owned_sequence(
                init_factories,
                f"{context} initial factories",
                size=len(self.seq_factories),
            )

    def make(self) -> "LookbackHiddenMarkovModelEstimatorAccumulator":
        """Create a new LookbackHiddenMarkovModelEstimatorAccumulator from the member factories.

        Returns:
            LookbackHiddenMarkovModelEstimatorAccumulator: Accumulator with zeroed sufficient statistics.

        """
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        seq_acc = [self.seq_factories[i].make() for i in range(len(self.seq_factories))]
        init_acc = [self.init_factories[i].make() for i in range(len(self.init_factories))]
        return LookbackHiddenMarkovModelEstimatorAccumulator(
            seq_acc,
            lag=self.lag,
            init_accumulators=init_acc,
            len_accumulator=len_acc,
            keys=self.keys,
            terminal_states=self.terminal_states,
        )


class LookbackHiddenMarkovModelEstimator(ParameterEstimator):
    """Estimator for a lookback hidden Markov model from aggregated sufficient statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        lag: int = 0,
        init_estimators: Sequence[ParameterEstimator] | None = None,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        suff_stat=None,
        pseudo_count: float | tuple[float | None, float | None] | None = (None, None),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        terminal_states=None,
        prior=None,
    ):
        """Create an estimator for a lookback hidden Markov model.

        Args:
            estimators (Sequence[ParameterEstimator]): Per-state estimators for the emission window
                distributions (one per hidden state).
            lag (int): Number of preceding observations each emission conditions on.
            init_estimators (Optional[Sequence[ParameterEstimator]]): Per-state estimators for the
                initial-segment distributions. Defaults to NullEstimator per state (the sibling
                module mixle.stats.lookback_hmm requires these to be passed explicitly).
            len_estimator (Optional[ParameterEstimator]): Estimator for the window-count distribution.
                Defaults to NullEstimator.
            suff_stat (Optional[Tuple]): Kept for interface consistency (unused).
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Regularize the
                initial-state probabilities and the transition matrix respectively.
            name (Optional[str]): Assign string name to the estimated distribution.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Keys for
                initial-state counts, transition counts, and state accumulators.
            prior: Optional conjugate chain prior over w/transitions, ``(init_prior, row_priors)``
                -- mirrors :class:`~mixle.stats.latent.hidden_markov.HiddenMarkovEstimator`. When
                set, ``estimate`` uses the true Dirichlet MAP update instead of ``pseudo_count``
                smoothing, and ``get_prior``/``model_log_density`` let
                :func:`mixle.inference.estimation.optimize` auto-detect the ``'map'`` objective.

        """
        context = "LookbackHiddenMarkovModelEstimator"
        self.lag = _validated_lookback_lag(lag, context=context)
        self.estimators = _owned_sequence(
            estimators,
            f"{context} topic estimators",
            minimum=1,
        )
        self.num_states = len(self.estimators)
        self.pseudo_count = _validated_lookback_pseudo_count(pseudo_count)
        self.suff_stat = suff_stat
        self.keys = _validated_lookback_keys(keys, context=context)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.terminal_states = validated_terminal_states(terminal_states, self.num_states, context=context)

        if init_estimators is None:
            if self.lag > 0:
                raise ValueError(f"{context} requires one initial estimator per state when lag is positive.")
            self.init_estimators = [NullEstimator() for _ in range(self.num_states)]
        else:
            self.init_estimators = _owned_sequence(
                init_estimators,
                f"{context} initial estimators",
                size=self.num_states,
            )
        self.set_prior(prior)

    def get_prior(self):
        """Returns the chain conjugate prior in ``(init_prior, row_priors)`` form (or None).

        Mirrors :meth:`~mixle.stats.latent.hidden_markov.HiddenMarkovEstimator.get_prior`.
        """
        if not self.has_conj_prior:
            return None
        return (self.init_prior, list(self.row_priors))

    def set_prior(self, prior) -> None:
        """Set the conjugate Dirichlet chain prior and flag whether it admits the conjugate update.

        Mirrors :meth:`~mixle.stats.latent.hidden_markov.HiddenMarkovEstimator.set_prior` exactly.

        Args:
            prior: ``(init_prior, row_priors)`` tuple or None; has_conj_prior is set when both the
                initial-state prior and all row priors are Dirichlet.
        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution

        if prior is None:
            self.prior = None
            self.init_prior = None
            self.row_priors = None
            self.has_conj_prior = False
            return

        try:
            prior_values = tuple(prior)
        except TypeError as exc:
            raise TypeError(
                "LookbackHiddenMarkovModelEstimator prior must be an initial prior and row-prior sequence."
            ) from exc
        if len(prior_values) != 2:
            raise ValueError("LookbackHiddenMarkovModelEstimator prior must contain exactly two entries.")
        init_prior = prior_values[0]
        row_priors = _owned_sequence(
            prior_values[1],
            "LookbackHiddenMarkovModelEstimator transition-row priors",
            size=self.num_states,
        )
        self.prior = prior
        self.init_prior = init_prior
        self.row_priors = row_priors
        self.has_conj_prior = isinstance(init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in row_priors
        )
        if self.has_conj_prior:
            _validate_lookback_dirichlet_geometry(
                init_prior,
                "LookbackHiddenMarkovModelEstimator initial prior",
                self.num_states,
            )
            for index, row_prior in enumerate(row_priors):
                _validate_lookback_dirichlet_geometry(
                    row_prior,
                    f"LookbackHiddenMarkovModelEstimator row prior {index}",
                    self.num_states,
                )
        self.prior = (init_prior, tuple(row_priors))

    def model_log_density(self, model: "LookbackHiddenMarkovModelDistribution") -> float:
        """Log-density of the model parameters under the priors (ELBO global term).

        Sums the Dirichlet log-densities of the initial-state and transition probabilities (when a
        conjugate chain prior is set) plus each topic/init-segment estimator's own
        ``model_log_density`` of its fitted distribution. Mirrors
        :meth:`~mixle.stats.latent.hidden_markov.HiddenMarkovEstimator.model_log_density` (which
        does not include its own ``len_estimator`` either -- kept at parity here rather than
        widening scope beyond the sibling class).

        Args:
            model (LookbackHiddenMarkovModelDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters.
        """
        rv = 0.0
        if self.has_conj_prior:
            tiny = 1.0e-300
            rv += float(self.init_prior.log_density(np.maximum(model.w, tiny)))
            for i, row_prior in enumerate(self.row_priors):
                rv += float(row_prior.log_density(np.maximum(model.transitions[i, :], tiny)))
        for est, topic in zip(self.estimators, model.topics):
            fn = getattr(est, "model_log_density", None)
            if callable(fn):
                rv += float(fn(topic))
        for est, init in zip(self.init_estimators, model.init_dist):
            fn = getattr(est, "model_log_density", None)
            if callable(fn):
                rv += float(fn(init))
        return rv

    def accumulator_factory(self):
        """Create a LookbackHiddenMarkovModelEstimatorAccumulatorFactory from the member estimators.

        Returns:
            LookbackHiddenMarkovModelEstimatorAccumulatorFactory: Factory for accumulators consistent
                with this estimator.

        """
        est_factories = [u.accumulator_factory() for u in self.estimators]
        iest_factories = [u.accumulator_factory() for u in self.init_estimators]

        len_factory = self.len_estimator.accumulator_factory()
        return LookbackHiddenMarkovModelEstimatorAccumulatorFactory(
            self.lag, est_factories, iest_factories, len_factory, self.keys, terminal_states=self.terminal_states
        )

    def estimate(self, nobs: float | None, suff_stat):
        """Estimate a LookbackHiddenMarkovModelDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations (passed to the length estimator).
            suff_stat (Tuple): Sufficient statistics in the format returned by
                LookbackHiddenMarkovModelEstimatorAccumulator.value().

        Returns:
            LookbackHiddenMarkovModelDistribution: M-step estimate of the distribution.

        """
        _est_keys = self.keys if isinstance(self.keys, (tuple, list)) and len(self.keys) == 3 else (None, None, None)
        # the M-step receives post-key_merge statistics, so this is where pooling actually
        # lands: equality when initial/transition are unkeyed, the pooled upper bound otherwise
        # (the previous unconditional equality rejected every keyed fit's M-step -- measured)
        (
            lag,
            num_states,
            init_counts,
            state_counts,
            trans_counts,
            topic_ss,
            init_ss,
            len_ss,
        ) = _validated_lookback_sufficient_statistics(
            suff_stat,
            lag=self.lag,
            num_states=self.num_states,
            context="LookbackHiddenMarkovModelEstimator.estimate",
            init_key=_est_keys[0],
            trans_key=_est_keys[1],
        )
        if _est_keys[0] is None:
            # a POOLED initial-count vector carries the mass of every site sharing the key, so
            # comparing it against THIS site's observation count stops being a corruption
            # check; an unkeyed one is still bounded by the observations that produced it
            validate_effective_sample_mass(
                nobs,
                init_counts.sum(),
                label="lookback HMM effective sample",
                allow_unassigned=True,
            )

        len_dist = self.len_estimator.estimate(nobs, len_ss)

        topic_nobs = [getattr(topic_ss[i], "length_nobs", state_counts[i]) for i in range(num_states)]
        initial_nobs = [getattr(init_ss[i], "length_nobs", init_counts[i]) for i in range(num_states)]
        topics = [self.estimators[i].estimate(topic_nobs[i], topic_ss[i]) for i in range(num_states)]
        init_dist = [self.init_estimators[i].estimate(initial_nobs[i], init_ss[i]) for i in range(num_states)]

        if self.has_conj_prior:
            from mixle.stats.bayes.dirichlet import DirichletDistribution
            from mixle.stats.latent.hidden_markov import _hmm_map_probs

            a0 = np.asarray(self.init_prior.get_parameters(), dtype=float)
            w = _hmm_map_probs(init_counts, a0)
            init_posterior = DirichletDistribution(init_counts + a0)

            transitions = np.zeros((num_states, num_states), dtype=np.float64)
            row_posteriors = []
            for i in range(num_states):
                ai = np.asarray(self.row_priors[i].get_parameters(), dtype=float)
                transitions[i, :] = _hmm_map_probs(trans_counts[i, :], ai)
                row_posteriors.append(DirichletDistribution(trans_counts[i, :] + ai))

            return LookbackHiddenMarkovModelDistribution(
                topics,
                w,
                transitions,
                lag=lag,
                init_dist=init_dist,
                len_dist=len_dist,
                name=self.name,
                terminal_states=self.terminal_states,
                prior=(init_posterior, row_posteriors),
            )

        if self.pseudo_count[0] is not None:
            p1 = self.pseudo_count[0] / float(num_states)
            w = init_counts + p1
            w /= w.sum()
        else:
            init_sum = init_counts.sum()
            w = np.full(num_states, 1.0 / num_states) if init_sum <= 0.0 else init_counts / init_sum

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
                bad_indices = np.flatnonzero(bad_rows)
                transitions[bad_indices, bad_indices] = 1.0
            else:
                transitions = trans_counts / row_sum

        return LookbackHiddenMarkovModelDistribution(
            topics,
            w,
            transitions,
            lag=lag,
            init_dist=init_dist,
            len_dist=len_dist,
            name=self.name,
            terminal_states=self.terminal_states,
        )


class LookbackHiddenMarkovModelDataEncoder(DataSequenceEncoder):
    """Encoder for sequences of iid lookback-HMM observations (each a Sequence[T])."""

    def __init__(
        self,
        encoder: DataSequenceEncoder,
        lag: int,
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
        init_encoder: DataSequenceEncoder | None = NullDataEncoder(),
    ) -> None:
        """Create an encoder for lookback-HMM sequences.

        Args:
            encoder (DataSequenceEncoder): Encoder for length-(lag+1) emission windows.
            lag (int): Number of preceding observations each emission conditions on.
            len_encoder (Optional[DataSequenceEncoder]): Encoder for window counts. Defaults to
                NullDataEncoder.
            init_encoder (Optional[DataSequenceEncoder]): Encoder for the initial x[:lag] segments.
                Defaults to NullDataEncoder.

        """
        self.encoder = encoder
        self.lag = _validated_lookback_lag(lag, context="LookbackHiddenMarkovModelDataEncoder")
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.init_encoder = init_encoder if init_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        s = "LookbackHiddenMarkovModelDataEncoder(encoder=" + str(self.encoder) + ",lag=" + str(self.lag)
        s += ",len_encoder=" + str(self.len_encoder) + ",init_encoder=" + str(self.init_encoder) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        """Checks if other is an equivalent LookbackHiddenMarkovModelDataEncoder (same lag and member encoders).

        Args:
            other (object): Object to compare.

        Returns:
            True if other is a LookbackHiddenMarkovModelDataEncoder with equal lag and member encoders.

        """
        if isinstance(other, LookbackHiddenMarkovModelDataEncoder):
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
            if len(x[i]) < lag:
                raise ValueError("lookback-HMM observations must contain at least lag values.")
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

    def row_count(self, x: Any) -> int:
        """Return and validate the number of encoded lookback-HMM sequences."""
        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError("lookback-HMM encoded data must be a two-slot tuple.")
        payload = x[0]
        if not isinstance(payload, tuple) or len(payload) != 7:
            raise ValueError("lookback-HMM encoded sequence payload must have seven slots.")
        ids, idi, ims, imi, sizes, _, _ = payload

        def index_vector(value: Any, label: str) -> np.ndarray:
            raw = np.asarray(value)
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError("%s must be a one-dimensional integer array." % label)
            return np.asarray(raw, dtype=np.intp)

        ids = index_vector(ids, "lookback-HMM window row indices")
        idi = index_vector(idi, "lookback-HMM initial row indices")
        ims = index_vector(ims, "lookback-HMM window position indices")
        imi = index_vector(imi, "lookback-HMM initial position indices")
        sizes = index_vector(sizes, "lookback-HMM sequence sizes")
        n_rows = len(sizes)
        if np.any(sizes < (1 if self.lag > 0 else 0)):
            raise ValueError("lookback-HMM encoded sequence sizes are incompatible with lag.")
        windows_per_row = sizes - 1 if self.lag > 0 else sizes
        expected_windows = int(windows_per_row.sum())
        if len(ids) != expected_windows or len(ims) != expected_windows:
            raise ValueError("lookback-HMM window indices disagree with encoded sequence sizes.")
        if len(ids) and (np.any(ids < 0) or np.any(ids >= n_rows)):
            raise ValueError("lookback-HMM window row indices are outside the encoded layout.")
        if not np.array_equal(np.bincount(ids, minlength=n_rows), windows_per_row):
            raise ValueError("lookback-HMM window counts disagree with encoded sequence sizes.")
        if self.lag > 0:
            if not np.array_equal(idi, np.arange(n_rows)) or len(imi) != n_rows:
                raise ValueError("lookback-HMM initial indices must cover every encoded row once.")
        elif len(idi) or len(imi):
            raise ValueError("lag-zero lookback-HMM encodings cannot contain initial indices.")
        return n_rows


# --- Backward-compatible API naming aliases ---
LookbackHiddenMarkovModelAccumulator = LookbackHiddenMarkovModelEstimatorAccumulator
LookbackHiddenMarkovModelAccumulatorFactory = LookbackHiddenMarkovModelEstimatorAccumulatorFactory
