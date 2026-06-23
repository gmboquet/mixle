""" "Create, estimate, and sample from a hidden markov model with K emission distributions (i.e. K states).

Defines the HierarchicalMixtureDistribution, HierarchicalMixtureSampler, HierarchicalMixtureEstimatorAccumulatorFactory,
HierarchicalMixtureEstimatorAccumulator, HierarchicalMixtureEstimator, and the HierarchicalMixtureDataEncoder classes
for use with pysparkplug.

Data type: Sequence[T] (determined by emission distributions).

Consider an observation x = (x_1, x_2, ..., x_T) where x_i is of data type T. Assume Z = (Z_1, ..., Z_T) is an
unobserved sequence of hidden states taking on values {1,2,..,K}. A K state hidden markov model can be written as
hierarchical model as follows:

For t = 1,2,..,T, the emission distributions are given by
    (1) P_1(X_t = x_t | Z_t = k), for k = {1,2,...,K}.

The state transitions are given by the K by K matrix formed from
    (2) p_mat(Z_t = i | Z_{t-1} = j), for i, j = {2,3,..,K}.

The initial state distribution is given by weights
    (3) p_mat(Z_1=k) = pi_k, for k = {1,2,...,K}, where sum_k pi_k = 1.0

If included, the length of the hidden markov model sequences is modeled through
    (4) P_len(T), where P_len() is a distribution with support on non-negative integers.

Note that P_1() in (1) must be a distribution compatible with type T data. p_mat() in (2) is a 2-d numpy array of 2-d
list of floats where the rows sum to 1.0. (3) is represented by a numpy array of list of floats that sum to 1.

"""

import heapq
import itertools
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

import pysp.utils.vector as vec
from pysp.capability import Neutral, supports
from pysp.engines.arithmetic import *
from pysp.engines.arithmetic import maxrandint
from pysp.enumeration.algorithms import BufferedStream, LengthFrontierMerge, best_first_union_max
from pysp.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from pysp.stats.compute.posterior import MarkovChainLatentPosterior
from pysp.stats.latent._hidden_markov_numba_kernels import (
    numba_baum_welch2,
    numba_baum_welch_alphas,
    numba_seq_log_density,
)
from pysp.stats.latent.mixture import MixtureDistribution
from pysp.stats.sequences.markov_chain import MarkovChainDistribution, stationary_distribution
from pysp.utils.aliasing import MISSING, coalesce_alias, require
from pysp.utils.optional_deps import HAS_NUMBA, numba

T = TypeVar("T")
T1 = TypeVar("T1")  # Emission suff-stat type
T2 = TypeVar("T2")  # Len suff-stat type
E1 = tuple[
    tuple[int, list[tuple[int, int]], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Any], Any, Any | None
]
E2 = tuple[tuple[np.ndarray, np.ndarray, np.ndarray], Any | None]


# --- Conjugate Dirichlet prior machinery (folded from pysp.bstats.hidden_markov) ---
#
# Hidden states are the fixed integers 0..S-1 (as in pysp.bstats), so the chain prior is a
# Dirichlet on the initial-state probabilities plus an independent Dirichlet on each transition
# row, carried as ``prior = (init_prior, row_priors)``.  Per-state emission ("topic") component
# priors are owned by the emission distributions/estimators themselves (the unified leaf-family
# Bayesian protocol), so the HMM only adds the chain-level prior and delegates emission terms to
# the topic estimators.  ``prior=None`` (the default) preserves the existing MLE / pseudo-count
# path byte-identically.


def hmm_dirichlet_default_prior(num_states: int):
    """Returns the default ``(init_prior, row_priors)`` pair of unit-parameter Dirichlets.

    Args:
        num_states (int): Number of hidden states S.

    Returns:
        Tuple ``(DirichletDistribution, list of S DirichletDistribution)``.

    """
    from pysp.stats.bayes.dirichlet import DirichletDistribution

    return (
        DirichletDistribution(np.ones(num_states)),
        [DirichletDistribution(np.ones(num_states)) for _ in range(num_states)],
    )


def _unpack_hmm_chain_prior(prior):
    """Normalize the chain prior into ``(init_prior, row_priors)``."""
    init_prior, row_priors = prior[0], list(prior[1])
    return init_prior, row_priors


def _hmm_map_probs(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Dirichlet MAP with boundary clamp; posterior mean when degenerate.

    Mirrors pysp.bstats.markov_chain._map_probs exactly.
    """
    num = np.maximum(counts + alpha - 1.0, 0.0)
    tot = num.sum()
    if tot > 0:
        return num / tot
    cpp = counts + alpha
    return cpp / cpp.sum()


def _hmm_forward_ll(log_b: np.ndarray, log_init: np.ndarray, log_trans: np.ndarray) -> float:
    """Scaled forward recursion returning a single sequence log-likelihood.

    Mirrors pysp.bstats.hidden_markov.HiddenMarkovModelDistribution._forward_ll exactly so that
    expected_log_density (which feeds digamma-expected init/transition log-probs and topic
    expected emissions) matches the bstats reference.
    """
    b_max = log_b.max(axis=1, keepdims=True)
    b = np.exp(log_b - b_max)
    a_mat = np.exp(log_trans)

    alpha = np.exp(log_init) * b[0, :]
    c = alpha.sum()
    ll = np.log(c) if c > 0 else -np.inf
    alpha = alpha / c if c > 0 else alpha

    for t in range(1, log_b.shape[0]):
        alpha = np.dot(alpha, a_mat) * b[t, :]
        c = alpha.sum()
        if c <= 0:
            return -np.inf
        ll += np.log(c)
        alpha /= c

    return float(ll + b_max.sum())


def terminal_forward_loglik(log_w: np.ndarray, log_a: np.ndarray, log_b: np.ndarray, term_mask: np.ndarray) -> float:
    """Log-likelihood of a terminal-state HMM sequence (length is a stopping time at the first terminal state).

    ``log_b`` is the per-position, per-state emission log-density ``(L, K)``. The forward only transitions
    *from* non-terminal states and the likelihood sums the final position over terminal states. Shared by
    every HMM variant whose forward is a linear state trellis (base, lookback, ...).
    """
    from scipy.special import logsumexp

    nonterm = ~term_mask
    la = log_w + log_b[0]
    for t in range(1, log_b.shape[0]):
        prev = np.where(nonterm, la, -np.inf)
        la = log_b[t] + logsumexp(prev[:, None] + log_a, axis=0)
    tf = la[term_mask]
    return float(logsumexp(tf)) if tf.size else -np.inf


def terminal_forward_backward(
    log_w: np.ndarray, log_a: np.ndarray, log_b: np.ndarray, term_mask: np.ndarray
) -> tuple[float, "np.ndarray | None", "np.ndarray | None"]:
    """Terminal-state forward-backward; returns ``(loglik, gamma (L,K), xi (L-1,K,K))`` (gamma/xi None if 0-prob).

    The backward pass mirrors the forward: only the final position may be terminal, and only non-terminal
    states have a future. Responsibilities are normalized by the sequence likelihood.
    """
    from scipy.special import logsumexp

    length, k = log_b.shape
    nonterm = ~term_mask
    la = np.empty((length, k))
    la[0] = log_w + log_b[0]
    for t in range(1, length):
        prev = np.where(nonterm, la[t - 1], -np.inf)
        la[t] = log_b[t] + logsumexp(prev[:, None] + log_a, axis=0)
    log_p = float(logsumexp(la[length - 1][term_mask])) if term_mask.any() else -np.inf
    if not np.isfinite(log_p):
        return log_p, None, None
    lb = np.full((length, k), -np.inf)
    lb[length - 1] = np.where(term_mask, 0.0, -np.inf)
    for t in range(length - 2, -1, -1):
        future = log_b[t + 1] + lb[t + 1]
        lb[t] = np.where(nonterm, logsumexp(log_a + future[None, :], axis=1), -np.inf)
    gamma = np.exp(la + lb - log_p)
    xi = np.zeros((max(length - 1, 0), k, k))
    for t in range(length - 1):
        log_xi = la[t][:, None] + log_a + (log_b[t + 1] + lb[t + 1])[None, :] - log_p
        xi[t] = np.where(nonterm[:, None], np.exp(log_xi), 0.0)
    return log_p, gamma, xi


class HiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):
    """Hidden Markov model distribution for variable-length observation sequences."""

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        w: Sequence[float] | np.ndarray = MISSING,
        transitions: list[list[float]] | np.ndarray = MISSING,
        taus: list[list[float]] | np.ndarray | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        terminal_values: set[T] | None = None,
        use_numba: bool = False,
        weights: Sequence[float] | np.ndarray = MISSING,
        prior=None,
        terminal_states: set[int] | Sequence[int] | None = None,
    ) -> None:
        """HiddenMarkovModelDistribution object defining HMM compatible with data type T.

        Defines an HMM with emission distributions in 'topics' (all must have the same data type T). If a length
        distribution for the length of HMM sequence is included, it must have data type int with support of non-negative
        integers.


        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions all having type T.
            w (Union[Sequence[float], np.ndarray]): Initial state probabilities.
            transitions (Union[List[List[float]], np.ndarray]): 2-d array of hidden state transition probabilities.
            taus (Optional[Union[Sequence[float], np.ndarray]]): Emission distributions are a Mixture over topics.
                Hidden states govern transitions between mixture weights.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]):
            name (Optional[str]): Set name to object instance.
            terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
            use_numba (bool): If True, use numba package for encoding and vectorized operations.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions all having type T.
            n_topics (int): Number of emission distributions.
            n_states (int): Number of hidden states.
            w (np.ndarray): Initial state probabilities.
            log_w (np.ndarray): Initial state log-probabilities.
            transitions (np.ndarray): 2-d Numpy array of hidden state transition probabilities. (n_states by n_states).
            log_transitions (np.ndarray): Log of above.
            taus (Optional[np.ndarray]): Emission distributions are a Mixture over topics. Hidden states govern
                transitions between mixture weights.
            log_taus (Optional[np.ndarray]): Log probabilties of taus above.
            has_topics (bool): True if taus is passed.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]):
            name (Optional[str]): Set name to object instance.
            terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
            use_numba (bool): If True, use numba package for encoding and vectorized operations.

        """
        w = coalesce_alias("w", w, "weights", weights, default=MISSING)
        transitions = require("transitions", transitions, default=MISSING)
        self.use_numba = use_numba

        with np.errstate(divide="ignore"):
            self.topics = topics
            self.n_topics = len(topics)
            self.n_states = len(w)
            self.w = vec.make(w)
            self.log_w = np.log(self.w)

            if not isinstance(transitions, np.ndarray):
                transitions = np.asarray(transitions, dtype=float)

            self.transitions = np.reshape(transitions, (self.n_states, self.n_states))
            self.log_transitions = np.log(self.transitions)
            self.terminal_values = terminal_values
            self.name = name
            self.len_dist = len_dist if len_dist is not None else NullDistribution()

            # Absorbing hidden states: the sequence ends exactly when one is first entered, so the
            # length is a stopping time (not governed by len_dist). Stored as a boolean state mask.
            self.terminal_states = None if terminal_states is None else set(int(s) for s in terminal_states)
            if self.terminal_states is not None:
                self._terminal_mask = np.zeros(self.n_states, dtype=bool)
                self._terminal_mask[list(self.terminal_states)] = True
                self.use_numba = False  # the terminal-state forward uses the non-numba per-sequence layout

        if taus is not None:
            self.taus = vec.make(taus)
            self.log_taus = log(self.taus)
            self.has_topics = True
        else:
            self.taus = None
            self.has_topics = False

        self.set_prior(prior)

    def get_prior(self):
        """Returns the chain conjugate prior in ``(init_prior, row_priors)`` form (or None).

        Per-state emission component priors are owned by the emission distributions themselves.
        """
        if not self.has_conj_prior:
            return None
        return (self.init_prior, list(self.row_priors))

    def set_prior(self, prior) -> None:
        """Set the conjugate Dirichlet chain prior and precompute its digamma expectations.

        With Dirichlet ``init_prior`` and Dirichlet ``row_priors`` (over the fixed hidden states
        0..S-1) this caches the digamma expectations E[ln p_k] = psi(alpha_k) - psi(sum alpha) used
        by expected_log_density and sets ``has_conj_prior`` accordingly. ``prior=None`` leaves the
        distribution a plain point model.

        Args:
            prior: ``(init_prior, row_priors)`` tuple or None.

        """
        from pysp.stats.bayes.dirichlet import DirichletDistribution

        if prior is None:
            self.prior = None
            self.init_prior = None
            self.row_priors = None
            self.e_log_init = None
            self.e_log_trans = None
            self.has_conj_prior = False
            return

        init_prior, row_priors = _unpack_hmm_chain_prior(prior)
        self.prior = prior
        self.init_prior = init_prior
        self.row_priors = row_priors

        if isinstance(init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in row_priors
        ):
            a0 = np.asarray(init_prior.get_parameters(), dtype=float)
            self.e_log_init = digamma(a0) - digamma(a0.sum())
            self.e_log_trans = np.zeros((self.n_states, self.n_states))
            for i, row_prior in enumerate(row_priors):
                ai = np.asarray(row_prior.get_parameters(), dtype=float)
                self.e_log_trans[i, :] = digamma(ai) - digamma(ai.sum())
            self.has_conj_prior = True
        else:
            self.e_log_init = None
            self.e_log_trans = None
            self.has_conj_prior = False

    def expected_log_density(self, x: list[T]) -> float:
        """Forward log-likelihood with digamma-expected initial/transition log-probabilities and
        the topics' expected_log_density emissions.

        Falls back to the plug-in log_density(x) when no conjugate prior is set. Not supported for
        the taus/topic-mixture parameterization (falls back to log_density there).

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Expected log-density of the observed HMM sequence x.

        """
        if not self.has_conj_prior or self.has_topics:
            return self.log_density(x)
        if x is None or len(x) == 0:
            return self.len_dist.log_density(0)

        log_b = np.asarray([[topic.expected_log_density(u) for topic in self.topics] for u in x])
        rv = _hmm_forward_ll(log_b, self.e_log_init, self.e_log_trans)
        rv += self.len_dist.log_density(len(x))
        return rv

    def seq_expected_log_density(self, x: E1 | E2) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Falls back to seq_log_density(x) when no conjugate prior is set or for the taus
        parameterization.

        Args:
            x: Encoded sequences from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per sequence.

        """
        if not self.has_conj_prior or self.has_topics:
            return self.seq_log_density(x)

        x0, x1 = x
        e_log_init = self.e_log_init
        e_log_trans = self.e_log_trans

        if x1 is None:
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            num_seq = idx_mat.shape[0]
            num_states = self.n_states

            log_b = np.zeros((tot_cnt, num_states))
            for i in range(num_states):
                log_b[:, i] = self.topics[i].seq_expected_log_density(enc_data)

            rv = np.zeros(num_seq)
            for seq_i in range(num_seq):
                rows = idx_mat[seq_i, :]
                rows = rows[rows >= 0]
                if len(rows) == 0:
                    continue
                rv[seq_i] = _hmm_forward_ll(log_b[rows, :], e_log_init, e_log_trans)
        else:
            (idx, sz, enc_data), len_enc = x1
            num_states = self.n_states
            tot_cnt = len(idx)

            log_b = np.zeros((tot_cnt, num_states))
            for i in range(num_states):
                log_b[:, i] = self.topics[i].seq_expected_log_density(enc_data)

            tz = np.concatenate([[0], sz]).cumsum().astype(int)
            rv = np.zeros(len(sz))
            for seq_i in range(len(sz)):
                if sz[seq_i] == 0:
                    continue
                rv[seq_i] = _hmm_forward_ll(log_b[tz[seq_i] : tz[seq_i + 1], :], e_log_init, e_log_trans)

        if self.len_dist is not None and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def __str__(self) -> str:
        """Returns string representation of HiddenMarkovDistribution instance."""
        s1 = ",".join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        if self.taus is None:
            s4 = repr(self.taus)
        else:
            s4 = repr([list(u) for u in self.taus])
        s5 = str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.terminal_values)
        s8 = repr(self.use_numba)

        return (
            "HiddenMarkovModelDistribution([%s], %s, %s, %s, len_dist=%s, name=%s, terminal_values=%s, "
            "use_numba=%s)" % (s1, s2, s3, s4, s5, s6, s7, s8)
        )

    def compute_capabilities(self):
        from pysp.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.topics) + (() if supports(self.len_dist, Neutral) else (self.len_dist,))
        if self.has_topics or self.terminal_values is not None or self.use_numba:
            return DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")
        ready = intersect_engine_ready(children)
        return DistributionCapabilities(engine_ready=ready, kernel_status="generic_latent")

    def compute_declaration(self):
        from pysp.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        topic_children = tuple(declaration_for(topic) for topic in self.topics)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(
            child for child in topic_children + ((length,) if length is not None else ()) if child is not None
        )
        roles = tuple("state_%d_emission" % i for i, child in enumerate(topic_children) if child is not None)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="hidden_markov",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("w", constraint="simplex_vector"),
                ParameterSpec("transitions", constraint="row_simplex_matrix"),
                ParameterSpec("taus", constraint="row_simplex_matrix", differentiable=False),
            ),
            statistics=(
                StatisticSpec("num_states", kind="metadata", additive=False, scales=False),
                StatisticSpec("initial_counts"),
                StatisticSpec("state_counts"),
                StatisticSpec("transition_counts"),
                StatisticSpec("emissions", kind="tuple"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="hidden_state_sequence",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def density(self, x: list[T]) -> float:
        """Returns the density of HMM for an observed sequence x.

        See 'HiddenMarkovDistribution.log_density()' for details.

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Density of HMM for observed sequence x.

        """
        return exp(self.log_density(x))

    def _terminal_states_log_density(self, x: list[T]) -> float:
        """Forward likelihood when the sequence ends exactly at the first absorbing (terminal) state.

        ``P(x) = sum over paths z_1..z_L`` with ``z_1..z_{L-1}`` non-terminal and ``z_L`` terminal, of
        ``pi(z_1) prod A(z_t,z_{t+1}) prod b_{z_t}(x_t)``. The forward recursion only transitions *from*
        non-terminal states (entering a terminal state stops the chain) and the likelihood sums the
        final position over terminal states. ``len_dist`` is bypassed -- the length is the stopping
        time. Computed in log space (stable, no rescaling).
        """
        n = len(x)
        if n == 0:
            return -np.inf  # a terminal-states HMM always emits at least the terminal state
        k = self.n_states
        log_b = np.empty((n, k))
        for j in range(k):
            log_b[:, j] = [self.topics[j].log_density(x[t]) for t in range(n)]
        return terminal_forward_loglik(self.log_w, self.log_transitions, log_b, self._terminal_mask)

    def log_density(self, x: list[T]) -> float:
        """Returns the log-density of HMM for observed sequence x.

        Density for a sequence of length N is given by recursively evaluating the conditional density,

            p_mat(x_mat(0),x_mat(1),....,x_mat(t)) = p_mat(x_mat(t)|x_mat(0),...,x_mat(t-1)) = p_mat(x_mat(t)|Z(t))*p_mat(Z(t)|Z(t-1))*p_mat(Z(t-1)|x_mat(0),....,x_mat(t-1))

        for t = 1,2,...,N-1. p_mat(Z(0)) is given by 'w', p_mat(x_mat(t)|Z(t)) is given by emission distribution 'topics' for
        t = 0,1,...,N-1.

        The returned density is given by

            p_mat(x_mat) = p_mat(x_mat(0),x_mat(1),....,x_mat(t))*P_len(N).

        where P_len(N) is the length distribution 'len_dist', if assigned.
        Note: All calculations are done on the log scale with log-sum-exp used to prevent numerical underflow.

        If 'has_topics' is true, 'weighed_log_sum_exp' and 'log_sum' calls from pysp.utils.vector are used to handle
        the emission distributions being treated as mixture distributions with weights 'log_taus'.

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Log-density of observed HMM sequence x.

        """
        if self.terminal_states is not None:
            return self._terminal_states_log_density(x)

        if x is None or len(x) == 0:
            return self.len_dist.log_density(0)  # this will return 0.0 if NullDistribution()

        if not self.has_topics:
            log_w = self.log_w
            num_states = self.n_states
            comps = self.topics

            obs_log_likelihood = np.zeros(num_states, dtype=np.float64)
            obs_log_likelihood += log_w
            for i in range(num_states):
                obs_log_likelihood[i] += comps[i].log_density(x[0])

            if np.max(obs_log_likelihood) == -np.inf:
                return -np.inf

            max_ll = obs_log_likelihood.max()
            obs_log_likelihood -= max_ll
            np.exp(obs_log_likelihood, out=obs_log_likelihood)
            sum_ll = np.sum(obs_log_likelihood)
            retval = np.log(sum_ll) + max_ll

            for k in range(1, len(x)):
                #  p_mat(Z(t) | Z(t-1) = i) p_mat(Z(t-1) = i | x_mat(0), ..., x_mat(t-1))
                np.dot(self.transitions.T, obs_log_likelihood, out=obs_log_likelihood)
                obs_log_likelihood /= obs_log_likelihood.sum()

                # log p_mat(Z(t-1) | x_mat(0), ..., x_mat(t-1))
                np.log(obs_log_likelihood, out=obs_log_likelihood)

                # log p_mat(x_mat(t) | Z(t)=i) + log p_mat(Z(t-1)=i | x_mat(0), ..., x_mat(t-1))
                for i in range(num_states):
                    obs_log_likelihood[i] += comps[i].log_density(x[k])

                # p_mat(x_mat(t) | x_mat(0), ..., x_mat(t-1))  [prevent underflow]
                max_ll = obs_log_likelihood.max()
                if max_ll == -np.inf:
                    # x[k] is outside every state's emission support: the sequence has zero probability.
                    # Without this guard, obs_log_likelihood -= -inf would produce nan (the vectorized
                    # seq_log_density path returns -inf here, so this keeps scalar/seq consistent).
                    return -np.inf
                obs_log_likelihood -= max_ll
                np.exp(obs_log_likelihood, out=obs_log_likelihood)
                sum_ll = np.sum(obs_log_likelihood)

                # p_mat(x_mat(0), ..., x_mat(t-1), x_mat(t))
                retval += np.log(sum_ll) + max_ll

            retval += self.len_dist.log_density(len(x))

            return retval

        else:
            x_iter = iter(x)
            log_w = self.log_w
            log_taus = self.log_taus
            n_states = self.n_states
            x0 = next(x_iter)

            obs_log_density_by_topic = np.asarray([u.log_density(x0) for u in self.topics])
            log_likelihood_by_state = np.asarray(
                [log_w[i] + vec.weighted_log_sum(obs_log_density_by_topic, log_taus[i, :]) for i in range(n_states)]
            )

            for x in x_iter:
                obs_log_density_by_topic = np.asarray([u.log_density(x) for u in self.topics])
                log_likelihood_by_state = [
                    vec.weighted_log_sum(obs_log_density_by_topic, log_taus[:, i])
                    + vec.weighted_log_sum(obs_log_density_by_topic, log_taus[i, :])
                    for i in range(n_states)
                ]

            rv = vec.log_sum(log_likelihood_by_state)
            rv += self.len_dist.log_density(len(x))

            return rv

    def _terminal_states_seq_log_density(self, x: E1 | E2) -> "np.ndarray":
        """Vectorized terminal-state forward: per-sequence stopping-time likelihood from encoded emissions."""
        x0, _ = x
        (tot_cnt, _idx_bands, _has_next, len_vec, idx_mat, _idx_vec, enc_data), _, _len_enc = x0
        k = self.n_states
        log_b_all = np.empty((tot_cnt, k))
        for j in range(k):
            log_b_all[:, j] = self.topics[j].seq_log_density(enc_data)
        out = np.empty(idx_mat.shape[0], dtype=np.float64)
        for s in range(idx_mat.shape[0]):
            length = int(len_vec[s])
            if length == 0:
                out[s] = -np.inf
                continue
            log_b = log_b_all[idx_mat[s, :length], :]
            out[s] = terminal_forward_loglik(self.log_w, self.log_transitions, log_b, self._terminal_mask)
        return out

    def seq_log_density(self, x: E1 | E2) -> "np.ndarray":
        """Return vectorized log-density values for sequence-encoded observations."""
        if self.terminal_states is not None:
            return self._terminal_states_seq_log_density(x)
        x0, x1 = x
        if x1 is None:
            num_states = self.n_states
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            w = self.w
            a_mat = self.transitions

            max_len = len(idx_bands)
            num_seq = idx_mat.shape[0]

            good = idx_mat >= 0

            pr_obs = np.zeros((tot_cnt, num_states))
            ll_ret = np.zeros(num_seq)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

            with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> ll_ret sanitized below
                pr_max0 = pr_obs.max(axis=1, keepdims=True)
                pr_obs -= pr_max0
                np.exp(pr_obs, out=pr_obs)

            # Vectorized alpha pass
            band = idx_bands[0]
            alphas_prev = np.multiply(pr_obs[band[0] : band[1], :], w)
            temp = alphas_prev.sum(axis=1, keepdims=True)
            alphas_prev /= temp

            np.log(temp, out=temp)
            temp2 = pr_max0[band[0] : band[1], 0]
            ll_ret[good[:, 0]] += temp[:, 0] + temp2

            for i in range(1, max_len):
                band = idx_bands[i]
                has_next_loc = has_next[i - 1]

                alphas_next = np.dot(alphas_prev[has_next_loc, :], a_mat)
                alphas_next *= pr_obs[band[0] : band[1], :]
                pr_max = alphas_next.sum(axis=1, keepdims=True)
                alphas_next /= pr_max
                alphas_prev = alphas_next

                np.log(pr_max, out=pr_max)
                temp2 = pr_max0[band[0] : band[1], 0]
                ll_ret[good[:, i]] += pr_max[:, 0] + temp2

            # nz = len_vec != 0
            # ll_ret[nz] /= len_vec[nz]

            ll_ret[np.isnan(ll_ret)] = -np.inf

            if self.len_dist is not None:
                ll_ret += self.len_dist.seq_log_density(len_enc)

            return ll_ret

        else:
            num_states = self.n_states
            (idx, sz, enc_data), len_enc = x1

            w = self.w
            a_mat = self.transitions
            tot_cnt = len(idx)
            num_seq = len(sz)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_seq, dtype=np.float64)
            tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

            with np.errstate(invalid="ignore"):  # impossible rows have max -inf -> sanitized after the kernel
                pr_max0 = pr_obs.max(axis=1)
                pr_obs -= pr_max0[:, None]
                np.exp(pr_obs, out=pr_obs)

            alpha_buff = np.zeros((num_seq, num_states), dtype=np.float64)
            next_alpha = np.zeros((num_seq, num_states), dtype=np.float64)

            numba_seq_log_density(num_states, tz, pr_obs, w, a_mat, pr_max0, next_alpha, alpha_buff, ll_ret)
            # a sequence with an out-of-support emission yields nan from the kernel; it has zero
            # probability (the numpy path above sanitizes the same way -- keep the two paths consistent)
            ll_ret[np.isnan(ll_ret)] = -np.inf

            if self.len_dist is not None:
                ll_ret += self.len_dist.seq_log_density(len_enc)

            return ll_ret

    def backend_seq_log_density(self, x: E1 | E2, engine: Any) -> Any:
        """Engine-neutral forward scores for non-numba encoded HMM batches.

        The compiled/numba encoding remains on the legacy NumPy path.  The
        standard blocked encoding is converted through the active engine and
        composes child distribution-owned backend scores.
        """
        from pysp.stats.compute.backend import BackendScoringError, backend_seq_log_density

        if self.has_topics:
            if getattr(engine, "supports_numba", False):
                return self.seq_log_density(x)
            raise BackendScoringError("HMM backend scoring does not support taus/topic-mixture emissions.")
        if self.terminal_values is not None:
            if getattr(engine, "supports_numba", False):
                return self.seq_log_density(x)
            raise BackendScoringError("HMM backend scoring does not support terminal-value semantics.")

        x0, x1 = x
        if x1 is not None:
            if getattr(engine, "supports_numba", False):
                return self.seq_log_density(x)
            raise BackendScoringError("HMM backend scoring requires the standard non-numba encoding.")

        num_states = self.n_states
        (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
        num_seq = idx_mat.shape[0]
        if tot_cnt == 0:
            rv = engine.zeros(num_seq)
            if self.len_dist is not None and len_enc is not None:
                rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)
            return rv

        pr_obs = []
        for i in range(num_states):
            pr_obs.append(backend_seq_log_density(self.topics[i], enc_data, engine))
        pr_obs = engine.stack(pr_obs, axis=1)

        pr_max0 = engine.max(pr_obs, axis=1)
        pr_exp = engine.exp(pr_obs - pr_max0[:, None])

        ll_ret = engine.zeros(num_seq)
        w = engine.asarray(self.w)
        a_mat = engine.asarray(self.transitions)

        good0 = np.asarray(idx_mat[:, 0] >= 0, dtype=bool)
        band = idx_bands[0]
        alphas_prev = pr_exp[band[0] : band[1], :] * w
        alpha_sum = engine.sum(alphas_prev, axis=1)
        alphas_prev = alphas_prev / alpha_sum[:, None]
        if np.any(good0):
            values = engine.log(alpha_sum) + pr_max0[band[0] : band[1]]
            ll_ret = engine.index_add(ll_ret, engine.asarray(np.flatnonzero(good0)), values)

        for i in range(1, len(idx_bands)):
            band = idx_bands[i]
            has_next_loc = has_next[i - 1]
            alphas_next = engine.matmul(alphas_prev[engine.asarray(has_next_loc)], a_mat)
            alphas_next = alphas_next * pr_exp[band[0] : band[1], :]
            alpha_sum = engine.sum(alphas_next, axis=1)
            alphas_next = alphas_next / alpha_sum[:, None]
            alphas_prev = alphas_next

            good = np.asarray(idx_mat[:, i] >= 0, dtype=bool)
            if np.any(good):
                values = engine.log(alpha_sum) + pr_max0[band[0] : band[1]]
                ll_ret = engine.index_add(ll_ret, engine.asarray(np.flatnonzero(good)), values)

        ll_ret = engine.where(engine.isnan(ll_ret), engine.asarray(-np.inf), ll_ret)
        if self.len_dist is not None and len_enc is not None:
            ll_ret = ll_ret + backend_seq_log_density(self.len_dist, len_enc, engine)

        return ll_ret

    def seq_posterior(self, x: E2) -> list[np.ndarray] | None:
        """Return vectorized posterior state probabilities for encoded observations."""
        if not self.use_numba:
            return None

        x0, x1 = x

        (idx, sz, enc_data), len_enc = x1

        tot_cnt = len(idx)
        seq_cnt = len(sz)
        num_states = self.n_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        weights = np.ones(seq_cnt, dtype=np.float64)
        max_len = sz.max()
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = self.w
        tran_mat = self.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch_alphas(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)

        return [alphas[tz[i] : tz[i + 1], :] for i in range(len(tz) - 1)]

    def viterbi(self, x: list[T]) -> np.ndarray:
        """Return the most likely latent-state path for a single observation sequence."""
        nn = len(x)
        num_states = self.n_states

        v = np.zeros((nn, num_states), dtype=np.float64)
        ptr = np.zeros(nn, dtype=np.int32)
        pr_obs = np.zeros((nn, num_states), dtype=np.float64)
        enc_x = self.topics[0].dist_to_encoder().seq_encode(x)

        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

        v[0, :] += pr_obs[0, :] + self.log_w

        for t in range(1, nn):
            temp = np.zeros((num_states, num_states), dtype=np.float64)
            temp += np.reshape(v[t - 1, :], (num_states, 1))
            temp += self.log_transitions
            temp += np.reshape(pr_obs[t, :], (1, num_states))
            v[t, :] += temp.max(axis=0, keepdims=False)

        for t in range(nn - 1, -1, -1):
            ptr[t] = np.argmax(v[t, :])

        return ptr

    def latent_posterior(self, x: list[T]) -> "MarkovChainLatentPosterior":
        """Return the exact chain posterior ``q(z | x)`` over hidden states for one observation sequence.

        The returned :class:`~pysp.stats.compute.posterior.MarkovChainLatentPosterior` can
        ``.marginals()`` (forward-backward smoothing probabilities), ``.sample(rng)`` a full state path
        by FFBS, ``.mode()`` (the Viterbi path), or ``.entropy()`` (the exact chain entropy).
        """
        enc = self.topics[0].dist_to_encoder().seq_encode(list(x))
        log_b = np.empty((len(x), self.n_states))
        for k in range(self.n_states):
            log_b[:, k] = self.topics[k].seq_log_density(enc)
        return MarkovChainLatentPosterior(self.log_w, self.log_transitions, log_b)

    def posterior_predictive(self, x: list[T], seed: int | None = None) -> list[Any]:
        """Draw a new observation sequence conditioned on ``x``.

        Sample a full hidden-state path from the posterior ``q(z | x)`` by FFBS, then emit a fresh
        observation from each state's emission distribution -- "given the sequence I saw, draw a new
        sequence from the states it most likely passed through". Returns a list the length of ``x``.
        """
        rng = RandomState(seed)
        z = self.latent_posterior(x).sample(rng)
        topic_samplers = [t.sampler(seed=rng.randint(maxrandint)) for t in self.topics]
        return [topic_samplers[k].sample() for k in z]

    def seq_viterbi(self, x: E2):
        """Return Viterbi paths for sequence-encoded observation sequences."""
        x0, x1 = x
        if x1 is None:
            num_states = self.n_states
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            log_w = self.log_w
            log_a_mat = self.log_transitions

            max_len = len(idx_bands)
            num_seq = idx_mat.shape[0]

            good = idx_mat >= 0

            pr_obs = np.zeros((tot_cnt, num_states))
            v = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ptr = np.zeros(tot_cnt, dtype=np.int32)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

            # Vectorized alpha pass
            prev_band_idx = np.arange(idx_bands[0][0], idx_bands[0][1])
            v[prev_band_idx, :] += pr_obs[prev_band_idx, :] + log_w

            for i in range(1, max_len):
                nxt_band_idx = np.arange(idx_bands[i][0], idx_bands[i][1])
                has_next_loc = has_next[i - 1]

                temp = np.zeros((len(has_next_loc), num_states, num_states), dtype=np.float64)
                temp += np.reshape(v[prev_band_idx[has_next_loc], :], (-1, num_states, 1)) + log_a_mat
                temp += np.reshape(pr_obs[nxt_band_idx, :], (-1, 1, num_states))

                v[nxt_band_idx, :] += np.max(temp, axis=1)

                prev_band_idx = nxt_band_idx.copy()

            for i in range(max_len - 1, -1, -1):
                prev_band_idx = np.arange(idx_bands[i][0], idx_bands[i][1])
                ptr[prev_band_idx] += np.argmax(v[prev_band_idx, :], axis=1)

            return ptr

    def to_fisher(self, **kwargs):
        """Forward-backward Fisher view for the HMM."""
        if hasattr(self, "topics") and hasattr(self, "transitions"):
            from pysp.inference.fisher import HiddenMarkovFisherView

            return HiddenMarkovFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> "HiddenMarkovSampler":
        """Create a HiddenMarkovSampler object with seed passed.

        Note: Throws exception if 'len_dist'and 'terminal_values' are not set.

        If len_dist is set, it should be a SequenceEncodableProbabilityDistribution with data type int and support on
        non-negative integers.

        Args:
            seed (Optional[int]): Set seed for random sampling.

        Returns:
            HiddenMarkovSampler object.

        """
        if supports(self.len_dist, Neutral) and self.terminal_values is None and self.terminal_states is None:
            raise Exception(
                "HiddenMarkovSampler requires len_dist with support on non-negative integers, or terminal_"
                "values / terminal_states to be set."
            )

        return HiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HiddenMarkovEstimator":
        """Create HiddenMarkovEstimator for estimating HiddenMarkovDistribution objects from aggregated sufficient
            statistics.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics of HiddenMarkovDistribution object
                instance.

        Returns:
            HiddenMarkovEstimator object.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return HiddenMarkovEstimator(
            comp_ests,
            pseudo_count=(pseudo_count, pseudo_count),
            len_estimator=len_est,
            name=self.name,
            prior=self.get_prior(),
            terminal_states=self.terminal_states,
        )

    def dist_to_encoder(self) -> "HiddenMarkovDataEncoder":
        """Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return HiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder, use_numba=self.use_numba
        )

    def enumerator(self) -> "HiddenMarkovModelEnumerator":
        """Returns HiddenMarkovModelEnumerator iterating observation sequences in descending
        marginal probability order."""
        return HiddenMarkovModelEnumerator(self)

    _COUNT_INDEX_ITEM_CAP = 1 << 18

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """BoundedCount for the MARGINAL HMM law: a forward count DP over the trellis with an
        iterative emission-split unrank, reaching a 2**M budget structurally.

        log p(x) = logsumexp over latent paths. We count (state-path, observation) PAIRS by their
        joint cost  log w_{s0} + sum_t log trans(s_t|s_{t-1}) + sum_t log emit_{s_t}(x_t): the forward
        DP pools paths into a per-(length, end-state) count histogram, where each step convolves the
        prefix histogram with the emission's count index (choosing the emitted symbol). This is the
        HMM analogue of the Mixture bound -- a conservative UPPER bound that does NOT deduplicate an
        observation produced by multiple paths and bins by the joint (dominant-path / tropical) cost
        rather than the exact logsumexp; every unranked value still carries its exact marginal
        ``log_density``. Unranking is one iterative backward walk over t: each step does a local
        times-split (recover the emitted symbol + bucket split) and a plus-choice (recover the
        predecessor state) -- O(L), no recursion. Falls back to capped enumerate-and-bin for
        non-plain HMMs (taus / terminal_values) or emissions that cannot count structurally.
        """
        from pysp.enumeration.quantization.core import CountHistogram, CountIndex, child_count_index, leaf_count_index
        from pysp.stats.compute.pdist import EnumerationError

        if supports(self.len_dist, Neutral):
            raise EnumerationError(self, reason="no length distribution is modeled (len_dist is Null)")

        def _fallback():
            return leaf_count_index(self.enumerator(), quantizer, max_fine_bucket, max_items=self._COUNT_INDEX_ITEM_CAP)

        if getattr(self, "taus", None) is not None or getattr(self, "terminal_values", None):
            return _fallback()

        n = self.n_states
        log_w = self.log_w
        log_T = self.log_transitions  # log_T[s][s'] = log P(s'|s)

        emit: list[Any] = []
        truncated = False
        for s in range(n):
            try:
                ci, tr = child_count_index(
                    self.topics[s], "HiddenMarkovModelDistribution.topics[%d]" % s, quantizer, max_fine_bucket
                )
            except EnumerationError:
                return _fallback()
            emit.append(ci)
            truncated = truncated or tr

        lengths: list[tuple[int, float]] = []
        _LEN_CAP = 1 << 24
        for length, lp_len in child_enumerator(self.len_dist, "HiddenMarkovModelDistribution.len_dist"):
            if not isinstance(length, (int, np.integer)) or length < 0 or lp_len == -np.inf:
                continue
            if quantizer.fine_bucket(lp_len) > max_fine_bucket:
                truncated = True
                break
            lengths.append((int(length), float(lp_len)))
            if len(lengths) >= _LEN_CAP:
                truncated = True
                break
        if not lengths:
            return CountIndex(CountHistogram.empty(), lambda fb, off: (_ for _ in ()).throw(IndexError())), truncated

        max_len = max(L for L, _ in lengths)
        init_shift = [quantizer.fine_bucket(log_w[s]) if log_w[s] > -np.inf else None for s in range(n)]
        # Predecessors into next-state s': (predecessor s, log trans, fine-bucket shift).
        into: list[list[tuple[int, float, int]]] = [[] for _ in range(n)]
        for sp in range(n):
            for s in range(n):
                lt = float(log_T[s][sp])
                if lt > -np.inf:
                    into[sp].append((s, lt, quantizer.fine_bucket(lt)))

        # alpha[t][s] = count histogram of (path, obs) prefixes of length t ending in state s;
        # pooled[t][s] = the pre-emission prefix histogram (sum over predecessors), kept for unranking.
        alpha: list[dict[int, CountHistogram]] = [None, {}]
        for s in range(n):
            if init_shift[s] is None or emit[s].hist.is_empty():
                continue
            h = emit[s].hist.shift(init_shift[s]).truncate(max_fine_bucket)
            if not h.is_empty():
                alpha[1][s] = h
        pooled: list[dict[int, CountHistogram]] = [None, {}]
        for t in range(2, max_len + 1):
            prev = alpha[t - 1]
            cur: dict[int, CountHistogram] = {}
            pcur: dict[int, CountHistogram] = {}
            for sp in range(n):
                if emit[sp].hist.is_empty():
                    continue
                pool = CountHistogram.empty()
                any_pred = False
                for s, _lt, shift in into[sp]:
                    ph = prev.get(s)
                    if ph is not None and not ph.is_empty():
                        pool = pool.add(ph.shift(shift).truncate(max_fine_bucket))
                        any_pred = True
                if not any_pred or pool.is_empty():
                    continue
                ah = quantizer.convolve(pool, emit[sp].hist, max_fine_bucket=max_fine_bucket)
                if ah.is_empty():
                    continue
                pcur[sp] = pool
                cur[sp] = ah
            alpha.append(cur)
            pooled.append(pcur)
            if not cur:
                truncated = True
                break
        built = len(alpha) - 1

        total = CountHistogram.empty()
        contributing: list[tuple[int, int, float]] = []
        for L, lp_len in lengths:
            ls = quantizer.fine_bucket(lp_len)
            if L == 0:
                total = total.add(CountHistogram.delta(ls, 1))
                contributing.append((0, ls, lp_len))
                continue
            if L > built or not alpha[L]:
                truncated = True
                continue
            seqh = CountHistogram.empty()
            for s in range(n):
                h = alpha[L].get(s)
                if h is not None:
                    seqh = seqh.add(h)
            piece = seqh.shift(ls).truncate(max_fine_bucket)
            if piece.is_empty():
                continue
            total = total.add(piece)
            contributing.append((L, ls, lp_len))

        def unrank(L: int, s_end: int, b: int, o: int) -> tuple[list[Any], float]:
            seq: list[Any] = [None] * L
            lp = 0.0
            t, s = L, s_end
            while t >= 2:
                eh = emit[s].hist
                pool = pooled[t][s]
                picked = False
                for be in range(eh.base, eh.base + len(eh.data)):
                    ne = eh.count_at(be)
                    if ne == 0:
                        continue
                    bp = b - be
                    mp = pool.count_at(bp)
                    if mp == 0:
                        continue
                    block = ne * mp
                    if o < block:
                        sym, slp = emit[s].get_in_bucket(be, o // mp)
                        seq[t - 1] = sym
                        lp += slp
                        po = o % mp
                        for s_prev, lt, shift in into[s]:
                            ph = alpha[t - 1].get(s_prev)
                            if ph is None:
                                continue
                            c = ph.count_at(bp - shift)
                            if c == 0:
                                continue
                            if po < c:
                                lp += lt
                                s, b, t, o = s_prev, bp - shift, t - 1, po
                                picked = True
                                break
                            po -= c
                        if not picked:
                            raise IndexError("offset outside hmm trellis")
                        break
                    o -= block
                if not picked:
                    raise IndexError("offset outside hmm trellis")
            sym, slp = emit[s].get_in_bucket(b - init_shift[s], o)
            seq[0] = sym
            return seq, lp + slp + float(log_w[s])

        def getter(fb: int, off: int) -> tuple[Any, float]:
            o = int(off)
            for L, ls, lp_len in contributing:
                if L == 0:
                    if fb == ls:
                        if o < 1:
                            return [], lp_len
                        o -= 1
                    continue
                target = fb - ls
                cnt_L = 0
                for s in range(n):
                    h = alpha[L].get(s)
                    if h is not None:
                        cnt_L += h.count_at(target)
                if o < cnt_L:
                    for s in range(n):
                        h = alpha[L].get(s)
                        if h is None:
                            continue
                        c = h.count_at(target)
                        if o < c:
                            return unrank(L, s, target, o)
                        o -= c
                    raise IndexError("offset outside hmm fine bucket %d" % fb)
                o -= cnt_L
            raise IndexError("offset outside hmm fine bucket %d" % fb)

        return CountIndex(total, getter), truncated

    def is_canonical_copy(self, value, coarse_bin: int, quantizer) -> bool:
        """Stateless dedup: keep an observation only at its min-cost (canonical) path's bin.

        The structural index emits an observation once per state-path that can generate it; the
        canonical copy is the one at the minimal joint fine bucket. A min-plus forward pass over the
        trellis computes that minimum exactly (mirroring the count-index's fine-bucket sums), so the
        check is O(L * n_states^2) with no state. Falls back to True for non-plain HMMs.
        """
        if getattr(self, "taus", None) is not None or getattr(self, "terminal_values", None):
            return True
        if not value:
            return True  # empty observation: a single copy
        n = self.n_states
        log_w = self.log_w
        log_T = self.log_transitions
        INF = float("inf")

        def emit_fb(o, s):
            # Structural bucket (sum-of-floored sub-buckets), matching the count DP's
            # child_count_index(topics[s]); a single floor of the joint log-density would
            # mispredict and drop nested (composite/sequence) emissions.
            if self.topics[s].log_density(o) == -np.inf:
                return INF
            return self.topics[s].structural_fine_bucket(o, quantizer)

        # v[s] = minimal joint fine bucket of a length-(t+1) path-prefix ending in state s.
        v = []
        for s in range(n):
            e = emit_fb(value[0], s)
            v.append(INF if (e == INF or log_w[s] == -np.inf) else quantizer.fine_bucket(float(log_w[s])) + e)
        for t in range(1, len(value)):
            nv = [INF] * n
            for sp in range(n):
                e = emit_fb(value[t], sp)
                if e == INF:
                    continue
                best_in = INF
                for s in range(n):
                    if v[s] == INF or log_T[s][sp] == -np.inf:
                        continue
                    cand = v[s] + quantizer.fine_bucket(float(log_T[s][sp]))
                    if cand < best_in:
                        best_in = cand
                if best_in != INF:
                    nv[sp] = best_in + e
            v = nv
        min_fb = min(v)
        if min_fb == INF:
            return True
        min_fb += self.len_dist.structural_fine_bucket(len(value), quantizer)
        return coarse_bin == quantizer.coarse_bin(int(min_fb))


class _HmmPrefix:
    """A concrete observation prefix in the HMM enumeration search.

    Holds the exact log-space forward vector alpha (alpha[s] = log p(x_1..x_t, S_t = s))
    and the projection proj[s'] = logsumexp_s(alpha[s] + log_A[s, s']) used to score and
    expand all single-symbol extensions of this prefix. For the empty prefix, proj is the
    initial state log-probability vector.
    """

    __slots__ = ("t", "values", "proj")

    def __init__(self, t: int, values: tuple, proj: np.ndarray) -> None:
        self.t = t
        self.values = values
        self.proj = proj


class HiddenMarkovModelEnumerator(DistributionEnumerator):
    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        topics: Sequence[SequenceEncodableProbabilityDistribution] | None = None,
        log_w: np.ndarray | None = None,
        log_transitions: np.ndarray | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
        path_root: str | None = None,
    ) -> None:
        """Enumerates observation sequences in descending marginal probability order.

        The optional keyword arguments override the corresponding attributes of dist so HMM
        variants with the same forward semantics (e.g. SemiSupervisedHiddenMarkovModelDistribution)
        can reuse this enumerator.

        The marginal probability of an observation sequence sums over all hidden state paths
        (the forward algorithm), so enumeration is an A*-style best-first search over
        observation prefixes:

          - A shared symbol pool enumerates the deduped union of the emission supports in
            descending max-over-states emission probability.
          - Prefixes carry exact log-space forward vectors; partial nodes are scored with the
            admissible bound logsumexp_s(proj[s] + UB[s, remaining-1]) + pool_max_emission,
            where UB[s, r] = logsumexp_s'(log_A[s, s'] + max_emission[s'] + UB[s', r-1]) bounds
            any r further (transition + emission) steps out of state s. The pool-rank bound is
            also valid for all later ranks, enabling lazy sibling generation.
          - Complete sequences re-enter the heap with their exact forward log-density plus the
            length log-probability, so popped complete sequences are in true descending order.
          - Lengths are pulled lazily from the length distribution's enumerator and merged on
            a length frontier (per-length scores never exceed the length log-probability).

        Raises EnumerationError for the taus/topics parameterization (different density
        semantics), when terminal_values is set, when no length distribution is modeled, or
        when an emission distribution does not support enumeration.

        Args:
            dist (HiddenMarkovModelDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        # ``getattr`` defaults let chain-forward HMM variants (e.g. SegmentalHiddenMarkovModel) that
        # share the standard forward semantics but lack the taus/terminal_values machinery reuse this
        # enumerator without defining those attributes.
        if getattr(dist, "has_topics", False):
            raise EnumerationError(dist, reason="taus/topics parameterization is not supported")
        if getattr(dist, "terminal_values", None) is not None:
            raise EnumerationError(dist, reason="terminal_values semantics are not supported")
        len_dist = dist.len_dist if len_dist is None else len_dist
        if len_dist is None or supports(len_dist, Neutral):
            raise EnumerationError(dist, reason="no length distribution is modeled (len_dist is Null)")
        path_root = path_root if path_root is not None else type(dist).__name__

        self._topics = list(dist.topics) if topics is None else list(topics)
        self._n_states = len(self._topics)
        self._log_w = np.asarray(dist.log_w if log_w is None else log_w, dtype=np.float64)
        self._log_a = np.asarray(dist.log_transitions if log_transitions is None else log_transitions, dtype=np.float64)

        emission_streams = [
            BufferedStream(child_enumerator(topic, "%s.topics[%d]" % (path_root, s)))
            for s, topic in enumerate(self._topics)
        ]
        heads = [es.get(0) for es in emission_streams]
        self._head_max = np.asarray([h[1] if h is not None else -np.inf for h in heads], dtype=np.float64)

        topics_loc = self._topics

        def max_emission_lp(x) -> float:
            with np.errstate(divide="ignore"):
                return max(topic.log_density(x) for topic in topics_loc)

        self._pool = BufferedStream(best_first_union_max(emission_streams, [0.0] * self._n_states, max_emission_lp))
        self._emis_cache: list[np.ndarray] = []

        # UB[r][s] bounds r further (transition + emission) steps out of state s.
        self._ub: list[np.ndarray] = [np.zeros(self._n_states, dtype=np.float64)]

        len_stream = BufferedStream(child_enumerator(len_dist, "%s.len_dist" % path_root))
        self._merge = LengthFrontierMerge(len_stream, self._kbest_sequences)

    def _emissions(self, rank: int) -> np.ndarray | None:
        """Per-state emission log-densities of the pool symbol at rank; None past the pool end."""
        while len(self._emis_cache) <= rank:
            item = self._pool.get(len(self._emis_cache))
            if item is None:
                return None
            with np.errstate(divide="ignore"):
                self._emis_cache.append(
                    np.asarray([topic.log_density(item[0]) for topic in self._topics], dtype=np.float64)
                )
        return self._emis_cache[rank]

    def _ub_for(self, r: int) -> np.ndarray:
        while len(self._ub) <= r:
            prev = self._ub[-1]
            step = self._log_a + (self._head_max + prev)[None, :]
            self._ub.append(logsumexp(step, axis=1))
        return self._ub[r]

    def _kbest_sequences(self, n: int, lp_len: float):
        if n == 0:
            yield ([], lp_len)
            return
        counter = itertools.count()
        heap = []  # entries: (-score, counter, kind, payload)

        def push_candidate(parent: "_HmmPrefix", rank: int) -> None:
            if self._pool.get(rank) is None:
                return
            pool_lp = self._pool.get(rank)[1]
            remaining = n - parent.t - 1
            bound = logsumexp(parent.proj + self._ub_for(remaining)) + pool_lp + lp_len
            if bound > -np.inf:
                heapq.heappush(heap, (-bound, next(counter), "cand", (parent, rank)))

        root = _HmmPrefix(0, (), self._log_w)
        push_candidate(root, 0)

        while heap:
            neg_score, _, kind, payload = heapq.heappop(heap)
            if kind == "done":
                yield payload
                continue
            parent, rank = payload
            push_candidate(parent, rank + 1)
            x, _ = self._pool.get(rank)
            alpha = parent.proj + self._emissions(rank)
            t = parent.t + 1
            if np.max(alpha) == -np.inf:
                continue
            if t == n:
                exact = logsumexp(alpha) + lp_len
                if exact > -np.inf:
                    heapq.heappush(heap, (-exact, next(counter), "done", (list(parent.values) + [x], exact)))
            else:
                proj = logsumexp(alpha[:, None] + self._log_a, axis=0)
                child = _HmmPrefix(t, parent.values + (x,), proj)
                push_candidate(child, 0)

    def __next__(self) -> tuple[list[Any], float]:
        return next(self._merge)


class HiddenMarkovSampler(DistributionSampler):
    def __init__(self, dist: "HiddenMarkovModelDistribution", seed: int | None = None) -> None:
        """HiddenMarkovSampler object for sampling from HMM.

        If 'dist.len_dist' is set, samples HMM sequences with sequence lengths generated from 'len_dist'. If
        'dist.len_dist' is NullDistribution, 'dist.terminal_values' is must be set. Samples are generated until
        a terminal value is reached.

        Args:
            dist (HiddenMarkovModelDistribution): HiddenMarkovModelDistribution object instance to sample from.
            seed (Optional[int]): Set seed on random number generator for sampling.

        Attributes:
            num_states (int): Number of hidden states in 'dist' object.
            dist (HiddenMarkovModelDistribution): HiddenMarkovModelDistribution object instance to sample from.
            rng (RandomState): RandomState object with seed set for sampling.
            obs_samplers (List[DistributionSampler]): List of DistributionSampler objects corresponding to the emission
                distributions of 'dist'. Taken to be MixtureSampler objects if 'dist.has_topics' is True.
            len_sampler (Optional[DistributionSampler]): DistributionSampler object with data type int and support on
                non-negative integers for sampling HMM observation sequence lengths.
            terminal_set (Optional[Set[T]]): Set of values to terminate HMM sampling when calling 'sample_seq()'.
            state_sampler (MarkovChainSampler): MarkovChainSampler for sampling states of HMM.

        """
        self.num_states = dist.n_states
        self.dist = dist
        self.rng = RandomState(seed)

        if dist.has_topics:
            self.obs_samplers = [
                MixtureDistribution(dist.topics, dist.taus[i, :]).sampler(seed=self.rng.randint(0, maxrandint))
                for i in range(dist.n_states)
            ]
        else:
            self.obs_samplers = [
                dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in range(dist.n_states)
            ]

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        else:
            self.len_sampler = None

        if dist.terminal_values is None:
            self.terminal_set = None
        else:
            self.terminal_set = set(dist.terminal_values)

        self.terminal_states = dist.terminal_states  # absorbing hidden states (a set of indices) or None

        t_map = {i: {k: dist.transitions[i, k] for k in range(dist.n_states)} for i in range(dist.n_states)}
        p_map = {i: dist.w[i] for i in range(dist.n_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def _sample_emissions_batched(self, state_seqs: list[list[Any]]) -> list[list[Any]]:
        """Draw all emissions for a batch of state paths, grouped by hidden state.

        Each emission sampler is invoked once with the number of token positions assigned to its
        state and the draws are scattered back into sequence/position order. Because every emission
        sampler owns an independent RandomState and is consumed in state order, this is byte-identical
        to drawing the emissions one at a time in nested order over (sequence, position).
        """
        # Flatten (sequence_index, position) and the hidden state at each position.
        flat_states: list[int] = []
        flat_seq: list[int] = []
        flat_pos: list[int] = []
        for si, seq in enumerate(state_seqs):
            for pi, st in enumerate(seq):
                flat_states.append(st)
                flat_seq.append(si)
                flat_pos.append(pi)

        out: list[list[Any]] = [[None] * len(seq) for seq in state_seqs]
        if not flat_states:
            return out

        flat_states_arr = np.asarray(flat_states)
        for st in range(self.num_states):
            pos_mask = np.flatnonzero(flat_states_arr == st)
            count = len(pos_mask)
            if count == 0:
                continue
            drawn = self.obs_samplers[st].sample(size=count)
            for m, idx in enumerate(pos_mask):
                out[flat_seq[idx]][flat_pos[idx]] = drawn[m]
        return out

    def sample_seq(self, size: int | None = None, *, batched: bool = True) -> list[Any] | list[list[Any]]:
        """Sample iid HMM sequences.

        If size is None, 1 sample is drawn and a List[T] is returned. If size > 0, 'size' samples are drawn and a List
        of length 'size' with HMM sequences (List[T]) is returned.

        With ``batched=True`` (default) the hidden-state paths for the whole batch are drawn in a
        single vectorized pass (the MarkovChainSampler advances all chains across time at once) and
        the emissions are drawn by grouping token positions by hidden state and invoking each emission
        sampler once. Emission batching is byte-identical to the legacy nested loop (each emission
        sampler owns an independent RandomState consumed in state order), but vectorizing the state
        path changes the RNG consumption order, so the state paths (and therefore the emissions
        conditioned on them) are only statistically equivalent to ``batched=False`` -- not
        byte-identical. Set ``batched=False`` to reproduce the exact legacy output for a given seed.

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.
            batched (bool): Vectorize state-path and emission draws (default); set False for the
                legacy per-draw loop.

        Returns:
            List[T] or List[List[T]] depending on size arg.

        """
        if not batched:
            if size is None:
                n = self.len_sampler.sample()
                state_seq = self.state_sampler.sample_seq(n, batched=False)
                return [self.obs_samplers[state_seq[i]].sample() for i in range(n)]
            n = self.len_sampler.sample(size=size)
            state_seq = [self.state_sampler.sample_seq(size=nn, batched=False) for nn in n]
            return [[self.obs_samplers[j].sample() for j in nn] for nn in state_seq]

        if size is None:
            n = int(self.len_sampler.sample())
            state_seq = self.state_sampler.sample_paths([n])[0]
            return self._sample_emissions_batched([state_seq])[0]

        n = np.asarray(self.len_sampler.sample(size=size), dtype=np.int64).reshape(-1)
        state_seqs = self.state_sampler.sample_paths(n)
        return self._sample_emissions_batched(state_seqs)

    def sample_terminal(self, terminal_set: set[T]) -> list[T]:
        """Sample an HMM sequence, until a terminal value is samples from the emission distribution.

        Args:
            terminal_set (Set[T]): Set values to terminate the HMM sequence.

        Returns:
            List[T] with length determined by samples to reach the first terminating value.

        """
        z = self.state_sampler.sample_seq()
        rv = [self.obs_samplers[z].sample()]

        while rv[-1] not in terminal_set:
            z = self.state_sampler.sample_seq(v0=z)
            rv.append(self.obs_samplers[z].sample())

        return rv

    def sample_terminal_states(self, cap: int = 1_000_000) -> list[T]:
        """Sample an HMM sequence run until the hidden chain first enters an absorbing (terminal) state.

        The path ``z_1, z_2, ...`` is drawn from the chain and stops the moment ``z_L`` is terminal; the
        returned sequence emits one observation per state, so its last state is terminal and all earlier
        states are not. ``cap`` guards against a terminal state that is unreachable.
        """
        z = int(self.state_sampler.sample_seq())
        states = [z]
        while z not in self.terminal_states and len(states) < cap:
            z = int(self.state_sampler.sample_seq(v0=z))
            states.append(z)
        return [self.obs_samplers[s].sample() for s in states]

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw iid samples from HMM.

        If a 'len_sampler' is set, call 'sample_seq()' (See HiddenMarkovSampler.sample_seq() for details).
        If 'len_sampler' is the NullDistributionSampler(), 'sample_terminal()' is called. (See
        HiddenMarkovSampler.sample_terminal() for details).

        With ``batched=True`` (default) the length-distribution path uses the vectorized
        :meth:`sample_seq` (statistically equivalent, not byte-identical -- see its docstring). The
        terminal-value path is inherently sequential and always uses the legacy loop. ``batched=False``
        reproduces the exact legacy output for a given seed.

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.
            batched (bool): Vectorize state-path and emission draws on the length-distribution path
                (default); set False for the legacy per-draw loop.

        Returns:
            List[T] or List[List[T]] depending on arg size.

        """
        if self.terminal_states is not None:
            if size is None:
                return self.sample_terminal_states()
            return [self.sample_terminal_states() for _ in range(size)]

        if self.len_sampler is not None:
            return self.sample_seq(size=size, batched=batched)

        elif self.terminal_set is not None:
            if size is None:
                return self.sample_terminal(self.terminal_set)
            else:
                return [self.sample_terminal(self.terminal_set) for i in range(size)]

        else:
            raise RuntimeError("HiddenMarkovSampler requires either a length distribution or terminal value set.")


class HiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        use_numba: bool | None = False,
        keys: tuple[str | None, str | None, str | None] = (None, None, None),
        name: str | None = None,
    ) -> None:
        """HiddenMarkovAccumulator object for aggregating sufficient statistics from HMM observations.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): SequenceEncodableStatisticAccumulator
                objects for the emission distributions.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): SequenceEncodableStatisticAccumulator
                object for the length distribution.
            use_numba (bool): True if sequence encodings are for use with numba.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for initial states, transition counts,
                and emission accumulators.
            name (Optional[str]): Name for object.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): SequenceEncodableStatisticAccumulator
                objects for the emission distributions.
            num_states (int): Total number of hidden states.
            init_counts (ndarray): Track gamma_i(0), or first time point gamma for each component in Baum-Welch.
            trans_counts (ndarray): 2-d matrix tracking transition updates from Baum-Welch
                (sum_t psi_ij(t) / sum_t gamma_i(t)).
            state_counts (ndarray): Expected number of times state is observed in sequence from t=0 to t=T-2.
            len_accumulator (SequenceEncodableStatisticAccumulator): SequenceEncodableStatisticAccumulator
                object for the length distribution. Set to NullAccumulator is None is passed.
            use_numba (bool): True if sequence encodings are for use with numba.
            init_key (Optional[str]): Key for initial states.
            trans_key (Optional[str]): Key for state transitions.
            state_key (Optional[str]): Key for emission accumulators..
            name (Optional[str]): Name for object.

            _init_rng (bool): True if RandomState objects have been initialized
            _len_rng (Optional[RandomState]): RandomState for initializing length accumulator.
            _acc_rng (Optional[List[RandomState]): List of RandomState objects for initializing emission accumulators.
            _idx_rng (Optional[RandomState]): RandomState for initializing initial state draws.

        """
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = vec.zeros(self.num_states)
        self.trans_counts = vec.zeros((self.num_states, self.num_states))
        self.state_counts = vec.zeros(self.num_states)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        self.use_numba = use_numba
        self.name = name

        # Data log-likelihood accumulated as a byproduct of the E-step forward pass, only when
        # _track_ll is enabled. Used by the fused-EM fast path in optimize(reuse_estep_ll=True);
        # not part of value(). Off by default so the standard path pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: RandomState | None = None
        self._acc_rng: list[RandomState] | None = None
        self._idx_rng: RandomState | None = None

    def update(self, x: list[T], weight: float, estimate: HiddenMarkovModelDistribution) -> None:
        """Update sufficient statistics of HiddenMarkovAccumulator with one observation.

        Note: Note efficient. Should use seq_encode() for fully encoded sequence instead.

        Args:
            x (List[T]): HMM observation sequence.
            weight (float): Weight for observation.
            estimate (HiddenMarkovModelDistribution): Previous estimate of HMM.

        Returns:
            None.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Set RandomState member variables for initialize and seq_initialize consistency.

        Args:
            rng (RandomState): RandomState object used to set member RandomState objects.

        Returns:
            None.

        """
        rng_seeds = rng.randint(maxrandint, size=2 + self.num_states)
        self._idx_rng = RandomState(seed=rng_seeds[0])
        self._len_rng = RandomState(seed=rng_seeds[1])
        self._acc_rng = [RandomState(seed=rng_seeds[2 + i]) for i in range(self.num_states)]
        self._init_rng = True

    def initialize(self, x: list[T], weight: float, rng: RandomState) -> None:
        """Initialize HiddenMarkovAccumulator object with HMM sequence x.

        Args:
            x (List[T]): HMM observation sequence.
            weight (float): Weight for observation.
            rng (RandomState): Sets RandomState member values if not already set.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        n = len(x)

        self.len_accumulator.initialize(n, weight, self._len_rng)

        if n > 0:
            idx = self._idx_rng.choice(self.num_states, size=n)

            self.init_counts[idx[0]] += weight
            self.state_counts[idx[0]] += weight

            for i in range(n):
                for j in range(self.num_states):
                    w = weight if j == idx[i] else 0.0
                    self.accumulators[j].initialize(x[i], w, self._acc_rng[j])

            if n > 1:
                for i in range(1, n):
                    self.trans_counts[idx[i - 1], idx[i]] += weight
                    self.state_counts[idx[i]] += weight

    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of HiddenMarkovAccumulator.

        Note: Initialization method depends on sequence encoding for Numba or baseline numpy. Both methods do
        not call numba for initialization.

        If _init_rng is False, protected RandomState members are set from rng for the accumulators. This ensures
        initialize() method produces a consistent initialization for the same datasets.

        The input 'x' is a sequence encoded HMM sequence of iid observations produced by
        'HiddenMarkovDataEncoder.seq_encode()'. Arg x is either Tuple[None, enc] or Tuple[None, enc_numba].

        For the first case, enc is Tuple[Tuple[....], T_topic, T_len], where the first tuple is given by a Tuple of
            enc[0][0] (int): Total number of observed emissions from all HMM sequences.
            enc[0][1] (List[Tuple[int, int]]): Contains bands for t^th observation in HMM sequences stored in 'seq_x'.
            enc[0][2] (List[ndarray[int]]): List of numpy array on sequence indices that have a next observed emission.
            enc[0][3] (np.ndarray[int]): Numpy array of sequence lengths.
            enc[0][4] (np.ndarray[int]): 2-d matrix with rv[0][0] rows, and column length equal to the length of the
                largest HMM sequence. This is used to store the index of seq_x corresponding to emission x[i][t]. A -1
                is stored if the sequence length has already been met.
            enc[0][5] (ndarray): Numpy array containing lists index 'i' corresponding to x[i][t] block of 'seq_x'.
            enc[0][6] (T_topic): Sequence encoded value of 'seq_x'.
        The next two entries of the Tuple are,
            enc[1] (T_topic): Sequence encoded observation values in order. Just for seq_init consistency.
            enc[1] (Optional[T_len]): Sequence encoded value of lengths of HMM distribution. None if len_encoder is
                the NullDataEncoder.

        The first entry of enc_numba is a Tuple of length-3,
            enc_numba[0][0] (ndarray[int]): Sequence id's for observed values.
            enc_numba[0][1] (ndarray[int]): Sequence lengths for each observed HMM sequence.
            enc_numba[0][2] (T_topic): Sequence encoded observation values.
        The second entry is,
            enc_numba[1] (Optional[T_len]): Sequence encoded values of sequence lengths. None if len_encoder is
                NullDataEncoder.

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of weights for observations.
            rng (RandomState): Used to set seed on random initialization.

        Returns:
            None.

        """
        x0, x1 = x

        if x1 is None:
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), xs_enc, len_enc = x0

            if not self._init_rng:
                self._rng_initialize(rng)

            self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

            non_zero_len = len_vec != 0
            weights_nz = weights[non_zero_len]

            idx = self._idx_rng.choice(self.num_states, size=tot_cnt)

            seq_i = []
            for i in range(len(len_vec[non_zero_len])):
                seq_i.extend([i] * len_vec[non_zero_len][i])

            seq_i = np.asarray(seq_i, dtype=int)

            x_idx_i, x_group_i, x_len_i = np.unique(seq_i, return_index=True, return_counts=True)

            self.init_counts += np.bincount(idx[x_group_i], weights_nz[x_idx_i], minlength=self.num_states)
            self.state_counts += np.bincount(idx, weights_nz[seq_i], minlength=self.num_states)

            sz_next = len_vec[non_zero_len].copy() - 1
            steps = np.zeros(len(sz_next), dtype=int)
            cond = steps < sz_next

            while np.any(cond):
                prev_state = idx[x_group_i[cond] + steps[cond]]
                next_state = idx[x_group_i[cond] + steps[cond] + 1]
                temp = np.bincount(
                    prev_state * self.num_states + next_state, weights_nz[cond], minlength=self.num_states**2
                )
                self.trans_counts += np.reshape(temp, (self.num_states, self.num_states))

                steps[cond] += 1
                cond = steps < sz_next

            for j in range(self.num_states):
                w = weights[idx_vec]
                w[idx != j] = 0.0
                self.accumulators[j].seq_initialize(xs_enc, w.flatten(), self._acc_rng[j])

        else:
            (idx, sz, xs), len_enc = x1

            if not self._init_rng:
                self._rng_initialize(rng)

            self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

            tot_cnt = np.sum(sz)
            states = self._idx_rng.choice(self.num_states, size=tot_cnt)
            nz_idx, nz_idx_group, nz_idx_rep = np.unique(idx, return_index=True, return_inverse=True)
            weights_nz = weights[nz_idx]

            # Emission init parity with the numpy ("bands") branch.
            #
            # Both branches draw the SAME values from self._idx_rng (same call, same size); ``states`` here equals
            # the numpy branch's time-major draw ``idx``. They differ ONLY in how that draw is mapped onto the
            # emission observations: the numba encoding stores observations sequence-major (``xs``), whereas the
            # numpy branch builds the per-observation weight mask in TIME-MAJOR ("banded") order via ``idx_vec``
            # and applies it positionally to its sequence-major ``xs_enc`` (which is identical to ``xs``). To stay
            # bit-identical with the pinned numpy path we reconstruct that same time-major ``idx_vec`` here and
            # build the mask the same way, rather than the sequence-major ``weights[idx]`` mask used previously.
            len_vec = np.asarray(sz, dtype=int)
            non_zero_len = len_vec != 0
            nz_len_vec = len_vec[non_zero_len]
            orig_ids_nz = np.nonzero(non_zero_len)[0]
            max_len = int(nz_len_vec.max()) if nz_len_vec.size else 0
            band_seq_i = [j for t in range(max_len) for j in range(len(nz_len_vec)) if t < nz_len_vec[j]]
            idx_vec = orig_ids_nz[np.asarray(band_seq_i, dtype=int)]

            for j in range(self.num_states):
                w = weights[idx_vec]
                w[states != j] = 0.0
                self.accumulators[j].seq_initialize(xs, w.flatten(), self._acc_rng[j])

            sz_next = sz.copy()[nz_idx] - 1
            steps = np.zeros(len(sz_next), dtype=int)
            cond = steps < sz_next

            while np.any(cond):
                prev_state = states[nz_idx_group[cond] + steps[cond]]
                next_state = states[nz_idx_group[cond] + steps[cond] + 1]
                temp = np.bincount(
                    prev_state * self.num_states + next_state, weights_nz[cond], minlength=self.num_states**2
                )
                self.trans_counts += np.reshape(temp, (self.num_states, self.num_states))

                steps[cond] += 1
                cond = steps < sz_next

            self.state_counts += np.bincount(states, weights[idx], minlength=self.num_states)
            self.init_counts += np.bincount(states[nz_idx_group], weights[nz_idx], minlength=self.num_states)

    def _terminal_seq_update(self, x, weights: np.ndarray, estimate: HiddenMarkovModelDistribution) -> None:
        """Baum-Welch E-step for a terminal-state HMM (per-sequence terminal forward-backward).

        The backward pass mirrors the forward: a state ends the sequence only at the final position and
        only if it is terminal, and only non-terminal states have a future. Responsibilities (gamma) and
        transition responsibilities (xi) are accumulated into the same init/state/transition/emission
        sufficient statistics as the standard path, so the M-step is unchanged.
        """
        from pysp.stats.latent.hidden_markov import terminal_forward_backward

        x0, _ = x
        (tot_cnt, _ib, _hn, len_vec, idx_mat, _iv, enc_data), _, _le = x0
        k = self.num_states
        log_b_all = np.empty((tot_cnt, k))
        for j in range(k):
            log_b_all[:, j] = estimate.topics[j].seq_log_density(enc_data)
        log_w, log_a, term = estimate.log_w, estimate.log_transitions, estimate._terminal_mask
        weights = np.asarray(weights, dtype=np.float64)
        gamma_flat = np.zeros((tot_cnt, k))
        for s in range(idx_mat.shape[0]):
            length = int(len_vec[s])
            if length == 0:
                continue
            rows = idx_mat[s, :length]
            log_p, gamma, xi = terminal_forward_backward(log_w, log_a, log_b_all[rows, :], term)
            if gamma is None:
                continue
            ws = float(weights[s])
            self.init_counts += ws * gamma[0]
            gamma_flat[rows] += ws * gamma
            self.trans_counts += ws * xi.sum(axis=0)
        self.state_counts += gamma_flat.sum(axis=0)
        for j in range(k):
            self.accumulators[j].seq_update(enc_data, gamma_flat[:, j], estimate.topics[j])

    def seq_update(self, x, weights: np.ndarray, estimate: HiddenMarkovModelDistribution) -> None:
        """Vectorized update for HiddenMarkovAccumulator object from encoded sequence of observations.

        This is a vectorized implementation of the Baum-Welch algorithm. If use_numba, Numba functions are called
        for the alpha and beta pass. Else, a vectorized Numpy implementation of Baum-Welch is used.

        The input 'x' is a sequence encoded HMM sequence of iid observations produced by
        'HiddenMarkovDataEncoder.seq_encode()'. Arg x is either Tuple[None, enc] or Tuple[None, enc_numba].

        For the first case, enc is Tuple[Tuple[....], T_topic, T_len], where the first tuple is given by a Tuple of
            enc[0][0] (int): Total number of observed emissions from all HMM sequences.
            enc[0][1] (List[Tuple[int, int]]): Contains bands for t^th observation in HMM sequences stored in 'seq_x'.
            enc[0][2] (List[ndarray[int]]): List of numpy array on sequence indices that have a next observed emission.
            enc[0][3] (np.ndarray[int]): Numpy array of sequence lengths.
            enc[0][4] (np.ndarray[int]): 2-d matrix with rv[0][0] rows, and column length equal to the length of the
                largest HMM sequence. This is used to store the index of seq_x corresponding to emission x[i][t]. A -1
                is stored if the sequence length has already been met.
            enc[0][5] (ndarray): Numpy array containing lists index 'i' corresponding to x[i][t] block of 'seq_x'.
            enc[0][6] (T_topic): Sequence encoded value of 'seq_x'.
        The next two entries of the Tuple is,
            enc[1] (T_topic): Sequence encoded observation values in order. Just for seq_init consistency.
            enc[2] (Optional[T_len]): Sequence encoded value of lengths of HMM distribution. None if len_encoder is
                the NullDataEncoder.

        The first entry of enc_numba is a Tuple of length-3,
            enc_numba[0][0] (ndarray[int]): Sequence id's for observed values.
            enc_numba[0][1] (ndarray[int]): Sequence lengths for each observed HMM sequence.
            enc_numba[0][2] (T_topic): Sequence encoded observation values.
        The second entry is,
            enc_numba[1] (Optional[T_len]): Sequence encoded values of sequence lengths. None if len_encoder is
                NullDataEncoder.

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of weights for observation.
            estimate (HiddenMarkovModelDistribution): Previous EM estimate of HMM model.

        Returns:
            None.

        """
        if estimate.terminal_states is not None:
            self._terminal_seq_update(x, weights, estimate)
            return

        x0, x1 = x

        if x1 is None:
            num_states = self.num_states
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            w = estimate.w
            a_mat = estimate.transitions

            max_len = len(idx_bands)
            num_seq = idx_mat.shape[0]

            good = idx_mat >= 0

            pr_obs = np.zeros((tot_cnt, num_states))
            alphas = np.zeros((tot_cnt, num_states))

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

            pr_max0 = pr_obs.max(axis=1, keepdims=True)
            pr_obs -= pr_max0
            np.exp(pr_obs, out=pr_obs)

            # When the fused-EM fast path requests it, accumulate the per-sequence data
            # log-likelihood from the forward normalizers (un-floored row sums + emission max),
            # matching seq_log_density exactly. The standard path skips this entirely.
            track_ll = self._track_ll
            ll_ret = np.zeros(num_seq) if track_ll else None

            # Vectorized alpha pass
            band = idx_bands[0]
            alphas_prev = alphas[band[0] : band[1], :]
            np.multiply(pr_obs[band[0] : band[1], :], w, out=alphas_prev)
            a_sum = alphas_prev.sum(axis=1, keepdims=True)
            if track_ll:
                with np.errstate(divide="ignore"):
                    ll_ret[good[:, 0]] += np.log(a_sum[:, 0]) + pr_max0[band[0] : band[1], 0]
            a_sum[a_sum == 0] = 1.0
            alphas_prev /= a_sum

            for i in range(1, max_len):
                band = idx_bands[i]
                has_next_loc = has_next[i - 1]
                alphas_next = alphas[band[0] : band[1], :]
                np.dot(alphas_prev[has_next_loc, :], a_mat, out=alphas_next)
                alphas_next *= pr_obs[band[0] : band[1], :]
                a_sum = alphas_next.sum(axis=1, keepdims=True)
                if track_ll:
                    with np.errstate(divide="ignore"):
                        ll_ret[good[:, i]] += np.log(a_sum[:, 0]) + pr_max0[band[0] : band[1], 0]
                a_sum[a_sum == 0] = 1.0
                alphas_next /= a_sum
                alphas_prev = alphas_next

            band2 = idx_bands[-1]
            prev_beta = np.ones((band2[1] - band2[0], num_states))
            alphas[band2[0] : band2[1], :] /= alphas[band2[0] : band2[1], :].sum(axis=1, keepdims=True)

            # Vectorized beta pass
            for i in range(max_len - 2, -1, -1):
                band1 = idx_bands[i]
                band2 = idx_bands[i + 1]
                has_next_loc = has_next[i]

                next_b = pr_obs[band2[0] : band2[1], :]
                prev_a = alphas[band1[0] : band1[1], :]
                prev_a = prev_a[has_next_loc, :]

                prev_beta *= next_b

                prev_a = np.reshape(prev_a, (prev_a.shape[0], prev_a.shape[1], 1))
                next_beta2 = np.reshape(prev_beta, (prev_beta.shape[0], 1, prev_beta.shape[1]))
                xi_loc = next_beta2 * a_mat
                next_beta = xi_loc.sum(axis=2)
                next_beta_max = next_beta.max(axis=1, keepdims=True)
                next_beta_max[next_beta_max == 0] = 1.0
                next_beta /= next_beta_max

                prev_beta = np.ones((band1[1] - band1[0], num_states))
                prev_beta[has_next_loc, :] = next_beta

                xi_loc *= prev_a
                # xi_loc = np.einsum('Bi,ij,Bj->Bij', prev_a, A, next_beta)
                xi_loc_sum = xi_loc.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
                len_vec_loc = np.reshape(len_vec[good[:, i + 1]], (-1, 1, 1)) - 1
                weights_loc = np.reshape(weights[good[:, i + 1]], (-1, 1, 1))
                # xi_loc *= weights_loc/(len_vec_loc*xi_loc_sum)

                xi_loc_sum[xi_loc_sum == 0] = 1.0

                xi_loc *= weights_loc / xi_loc_sum

                temp = xi_loc.sum(axis=2)
                temp_sum = temp.sum(axis=1, keepdims=True)
                temp_sum[temp_sum == 0] = 1.0
                temp /= temp_sum

                alphas[band1[0] + has_next_loc, :] = temp

                self.trans_counts += xi_loc.sum(axis=0)

            # Aggregate sufficient statistics
            for i in range(num_states):
                # alphas[:,i] *= weights[idx_vec]/np.maximum(len_vec[idx_vec], 1.0)
                alphas[:, i] *= weights[idx_vec]
                self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            band1 = idx_bands[0]
            temp = alphas[band1[0] : band1[1], :].sum(axis=1, keepdims=True)
            temp[temp == 0] = 1.0
            alphas[band1[0] : band1[1], :] *= np.reshape(weights[good[:, 0]], (-1, 1)) / temp

            self.init_counts += alphas[band1[0] : band1[1], :].sum(axis=0)

            if track_ll:
                if estimate.len_dist is not None and len_enc is not None:
                    ll_ret = ll_ret + estimate.len_dist.seq_log_density(len_enc)
                self._seq_ll += float(np.dot(weights, ll_ret))

            if self.len_accumulator is not None:
                self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

        else:
            (idx, sz, enc_data), len_enc = x1

            tot_cnt = len(idx)
            seq_cnt = len(sz)
            num_states = estimate.n_states
            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            max_len = sz.max()
            tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

            init_pvec = estimate.w
            tran_mat = estimate.transitions

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

            pr_max = pr_obs.max(axis=1, keepdims=True)
            pr_obs -= pr_max
            np.exp(pr_obs, out=pr_obs)

            # When the fused-EM fast path requests it, compute the per-sequence data log-likelihood
            # from the already-scored emissions via the (read-only) forward kernel, reusing pr_obs so
            # no emissions are re-scored. Done before Baum-Welch (which may overwrite pr_obs).
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
                self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            if self.len_accumulator is not None:
                self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def seq_update_engine(self, x, weights, estimate: "HiddenMarkovModelDistribution", engine) -> None:
        """Engine-resident Baum-Welch E-step.

        Mirrors :meth:`seq_update` but computes the posteriors (gamma), expected transition counts
        (xi), and initial-state posteriors (pi) with :func:`hmm_engine_forward_backward`, so the
        forward-backward runs on the active engine (numpy or torch/GPU/autograd). Per-state emission
        statistics are accumulated by the existing child accumulators using the engine-computed
        gamma weights. Falls back to the host :meth:`seq_update` for the blocked (non-numba)
        encoding.
        """
        x0, x1 = x
        num_states = estimate.n_states
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_w = np.log(estimate.w)
            log_a = np.log(estimate.transitions)

        if x1 is not None:
            # numba encoding: observations are stored sequence-contiguously.
            (idx, sz, enc_data), len_enc = x1
            sz = np.asarray(sz)
            tot_cnt = int(sz.sum())
            pr_obs = np.empty((tot_cnt, num_states), dtype=np.float64)
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)
            padded, mask, offsets = hmm_pad_log_emissions(pr_obs, sz)
            scatter = None
        else:
            # blocked encoding: idx_mat[n, t] is the flat row of observation (sequence n, step t),
            # or -1 when the sequence is shorter than t. Gather it into the padded layout and keep
            # the index map for scattering gamma back to the flat emission order.
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            pr_obs = np.empty((tot_cnt, num_states), dtype=np.float64)
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)
            n_seq, tmax = idx_mat.shape
            valid = idx_mat >= 0
            padded = np.full((n_seq, tmax, num_states), -np.inf, dtype=np.float64)
            padded[valid] = pr_obs[idx_mat[valid]]
            mask = valid.astype(np.float64)
            scatter = (valid, idx_mat)

        _, gamma, xi_sum, pi = hmm_engine_forward_backward(engine, padded, log_w, log_a, mask, weights=weights_np)
        gamma = np.asarray(engine.to_numpy(gamma))
        xi_sum = np.asarray(engine.to_numpy(xi_sum))
        pi = np.asarray(engine.to_numpy(pi))

        gamma_flat = np.zeros((tot_cnt, num_states), dtype=np.float64)
        if scatter is None:
            for i in range(len(sz)):
                n = int(sz[i])
                if n > 0:
                    gamma_flat[offsets[i] : offsets[i + 1], :] = gamma[i, :n, :]
        else:
            valid, idx_mat = scatter
            gamma_flat[idx_mat[valid]] = gamma[valid]

        self.init_counts += pi.sum(axis=0)
        self.trans_counts += xi_sum
        self.state_counts += gamma_flat.sum(axis=0)
        for i in range(num_states):
            self.accumulators[i].seq_update(enc_data, gamma_flat[:, i], estimate.topics[i])
        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(len_enc, weights_np, estimate.len_dist)

    def combine(
        self, suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], T2 | None]
    ) -> "HiddenMarkovAccumulator":
        """Combine the sufficient statistics of HiddenMarkovAccumulator with suff_stat arg.

        Sufficient statistics in suff_stat are a Tuple containing:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarayy): State transition counts.
            suff_stat[4] (Sequence[T1]): Emission distribution accumulators.
            suff_stat[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Args:
            suff_stat: See above for details.

        Returns:
            HiddenMarkovAccumulator object.

        """
        num_states, init_counts, state_counts, trans_counts, acc_values, len_acc_value = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.accumulators[i].combine(acc_values[i])

        if len_acc_value is not None:
            self.len_accumulator.combine(len_acc_value)

        return self

    def value(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any], Any | None]:
        """Returns sufficient statistics of HiddenMarkovAccumulator object instance.

        Returned value rv is a Tuple containing:
            rv[0] (int): Number of hidden states.
            rv[1] (np.ndarray): Initial state counts.
            rv[2] (np.ndarray): State counts.
            rv[3] (np.ndarray): State transition counts.
            rv[4] (Sequence[T1]): Emission distribution accumulator sufficient statistics (type T1).
            rv[5] (Optional[T2]): Optional sufficient statistics of the length distribution (type T2).

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]].

        """
        len_val = self.len_accumulator.value()

        return (
            self.num_states,
            self.init_counts,
            self.state_counts,
            self.trans_counts,
            tuple([u.value() for u in self.accumulators]),
            len_val,
        )

    def from_value(
        self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], T2 | None]
    ) -> "HiddenMarkovAccumulator":
        """Set the sufficient statistics of HiddenMarkovAccumulator object instance to value x.

        Returned value x is a Tuple containing:
            x[0] (int): Number of hidden states.
            x[1] (np.ndarray): Initial state counts.
            x[2] (np.ndarray): State counts.
            x[3] (np.ndarayy): State transition counts.
            x[4] (List[T1]): Emission distribution accumulators.
            x[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Args:
            x: See above for details.

        Returns:
            HiddenMarkovAccumulator object.

        """
        num_states, init_counts, state_counts, trans_counts, accumulators, len_acc = x
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts

        for i, v in enumerate(accumulators):
            self.accumulators[i].from_value(v)

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(len_acc)

        return self

    def scale(self, c: float) -> "HiddenMarkovAccumulator":
        """Scale linear HMM sufficient statistics while preserving metadata."""
        self.init_counts *= c
        self.state_counts *= c
        self.trans_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        if self.len_accumulator is not None:
            self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge the sufficient statistics of object instance with sufficient statistics in suff_stat that have
            matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary containing sufficient statistics for corresponding keys.

        Returns:
            None.

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
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

        return None

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace the sufficient statistics of HiddenMarkovAccumulator object with matching sufficient statistics in
            arg suff_stat that have matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.accumulators = stats_dict[self.state_key]

        for u in self.accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

        return None

    def acc_to_encoder(self) -> "HiddenMarkovDataEncoder":
        """Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return HiddenMarkovDataEncoder(
            emission_encoder=emission_encoder, len_encoder=len_encoder, use_numba=self.use_numba
        )


class HiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        use_numba: bool = False,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        name: str | None = None,
    ) -> None:
        """HiddenMarkovAccumulatorFactory object for creating HiddenMarkovEstimatorAccumulator objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory object for the emission
                distributions.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution.
            use_numba (bool): Default to True.
            keys (Optional[Tuple[Optional[str],Optional[str], Optional[str]]]): Set keys for initial states, state
                transitions, and the emission distributions.
            name (Optional[str]): Name for object.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory object for the emission
                distributions.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution. Defaults
                to NullAccumulatorFactory().
            use_numba (bool): Default to True. Indicated if Numbda is to be used for 'seq_' calls.
            keys (Tuple[Optional[str],Optional[str], Optional[str]]): Set keys for initial states, state
                transitions, and the emission distributions.
            name (Optional[str]): Name for object.


        """
        self.factories = factories
        self.use_numba = use_numba
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory
        self.name = name

    def make(self) -> "HiddenMarkovAccumulator":
        """Returns a HiddenMarkovAccumulator object."""
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return HiddenMarkovAccumulator(
            [self.factories[i].make() for i in range(len(self.factories))],
            len_accumulator=len_acc,
            use_numba=self.use_numba,
            keys=self.keys,
            name=self.name,
        )


class HiddenMarkovEstimator(ParameterEstimator):
    def __init__(
        self,
        estimators: list[ParameterEstimator],
        len_estimator: ParameterEstimator | None = NullEstimator(),
        pseudo_count: tuple[float | None, float | None] | None = (None, None),
        name: str | None = None,
        keys: tuple[str | None, str | None, str | None] | None = (None, None, None),
        use_numba: bool | None = None,
        prior=None,
        steady_state_init: bool = False,
        terminal_states: set[int] | Sequence[int] | None = None,
    ) -> None:
        """HiddenMarkovEstimator object for estimating HiddenMarkovDistribution for aggregated sufficient statistics.

        Args:
            estimators (List[ParameterEstimator]): Set ParameterEstimator objects for emission distributions.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object for length distribution.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Pseudo count for initial states and
                state transitions.
            name (Optional[str]): Set name to object.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for initial states,
                transitions counts, and emission distributions.
            use_numba (Optional[bool]): If True, Numba is used for sequence encoding and vectorized functions. If
                None (default), numba is used automatically when installed (HAS_NUMBA). The numba and numpy paths
                are bit-identical for this estimator, so this only affects speed.

        Attributes:
            estimators (List[ParameterEstimator]): Set ParameterEstimator objects for emission distributions.
            len_estimator (ParameterEstimator): ParameterEstimator object for length distribution, set to NullEstimator
                if None was passed.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo count for initial states and
                state transitions. Defaults to Tuple of (None, None) if None was passed.
            name (Optional[str]): Name for object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states, transitions counts, and
                emission distributions. Defaults to Tuple of (None, None, None).
            use_numba (bool): If True, Numba is used for sequence encoding and vectorized functions.

        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.use_numba = HAS_NUMBA if use_numba is None else use_numba
        self.steady_state_init = bool(steady_state_init)
        self.terminal_states = terminal_states
        self.set_prior(prior)

    def accumulator_factory(self):
        """Returns an HiddenMarkovAccumulatorFactory object."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return HiddenMarkovAccumulatorFactory(est_factories, len_factory, self.use_numba, self.keys, self.name)

    def get_prior(self):
        """Returns the chain conjugate prior in ``(init_prior, row_priors)`` form (or None).

        Per-state emission component priors are owned by the topic estimators themselves.
        """
        if not self.has_conj_prior:
            return None
        return (self.init_prior, list(self.row_priors))

    def set_prior(self, prior) -> None:
        """Set the conjugate Dirichlet chain prior and flag whether it admits the conjugate update.

        Args:
            prior: ``(init_prior, row_priors)`` tuple or None; has_conj_prior is set when both the
                initial-state prior and all row priors are Dirichlet.

        """
        from pysp.stats.bayes.dirichlet import DirichletDistribution

        if prior is None:
            self.prior = None
            self.init_prior = None
            self.row_priors = None
            self.has_conj_prior = False
            return

        init_prior, row_priors = _unpack_hmm_chain_prior(prior)
        self.prior = prior
        self.init_prior = init_prior
        self.row_priors = row_priors
        self.has_conj_prior = isinstance(init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in row_priors
        )

    def model_log_density(self, model: "HiddenMarkovModelDistribution") -> float:
        """Log-density of the model parameters under the priors (ELBO global term).

        Sums the Dirichlet log-densities of the initial-state and transition probabilities (floored
        at a tiny constant so boundary MAP estimates score finitely) plus each topic estimator's
        model_log_density of its emission distribution. Returns the emission-only sum without a
        conjugate chain prior.

        Args:
            model (HiddenMarkovModelDistribution): Model to score.

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
            if hasattr(est, "model_log_density"):
                rv += float(est.model_log_density(topic))
        return rv

    def estimate(
        self, nobs: float | None, suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray, list[T1], T2 | None]
    ) -> "HiddenMarkovModelDistribution":
        """Estimate HiddenMarkovModel from aggregated sufficient statistics contained in arg 'suff_stat'.

        Sufficient statistics in arg 'suff_stat' are a Tuple containing:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarayy): State transition counts.
            suff_stat[4] (List[T1]): List of Sufficient statistics for the emission distribution accumulators.
                Each having type S0.
            suff_stat[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the type for the sufficient statistics of the emission accumulators. T2 is the type for the
        length accumulator.

        If pseudo_count[0] is not None, the initial counts in 'suff_stat' is re-weighted in estimation.
        If pseudo_count[1] is not None, the transition counts in 'suff_stat' are re-weighted in estimation.


        Args:
            nobs (Optional[float]): Number of observations used in estimation.
            suff_stat: See above for details.

        Returns:
            HiddenMarkovModelDistribution object.

        """
        from pysp.stats.bayes.dirichlet import DirichletDistribution

        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss)
        topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]

        if self.has_conj_prior:
            a0 = np.asarray(self.init_prior.get_parameters(), dtype=float)
            w = _hmm_map_probs(init_counts, a0)
            init_posterior = DirichletDistribution(init_counts + a0)

            transitions = np.zeros((num_states, num_states), dtype=np.float64)
            row_posteriors = []
            for i in range(num_states):
                ai = np.asarray(self.row_priors[i].get_parameters(), dtype=float)
                transitions[i, :] = _hmm_map_probs(trans_counts[i, :], ai)
                row_posteriors.append(DirichletDistribution(trans_counts[i, :] + ai))

            if self.steady_state_init:  # tie the initial distribution to the transition matrix's equilibrium
                w = stationary_distribution(transitions)

            return HiddenMarkovModelDistribution(
                topics=topics,
                w=w,
                transitions=transitions,
                taus=None,
                len_dist=len_dist,
                name=self.name,
                terminal_values=None,
                terminal_states=self.terminal_states,
                use_numba=self.use_numba,
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
            else:
                transitions = trans_counts / row_sum

        if self.steady_state_init:  # tie the initial distribution to the transition matrix's equilibrium
            w = stationary_distribution(transitions)

        return HiddenMarkovModelDistribution(
            topics=topics,
            w=w,
            transitions=transitions,
            taus=None,
            len_dist=len_dist,
            name=self.name,
            terminal_values=None,
            terminal_states=self.terminal_states,
            use_numba=self.use_numba,
        )


class HiddenMarkovDataEncoder(DataSequenceEncoder):
    def __init__(
        self,
        emission_encoder: DataSequenceEncoder,
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
        use_numba: bool = False,
    ) -> None:
        """HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations.

        Args:
            emission_encoder (DataSequenceEncoder): DataSequenceEncoder object of type T for the observed
                emission distribution values.
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder object for the length
                of sequences. Should have support of non-negative integers.
            use_numba (bool): If True, sequence encode for Numba.

        Attributes:
            emission_encoder (DataSequenceEncoder): DataSequenceEncoder object of type T for the observed
                emission distribution values.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the length of sequences.
                Should have support of non-negative integers. Set to NullDataEncoder if None.
            use_numba (bool): If True, sequence encode for Numba.

        """
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.use_numba = use_numba

    def __str__(self) -> str:
        """Returns string representation of HiddenMarkovDataEncoder object instance."""
        s = "HiddenMarkovDataEncoder(emission_encoder=" + str(self.emission_encoder) + ","
        s += "len_encoder=" + str(self.len_encoder) + ","
        s += "use_numba=" + str(self.use_numba) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        """Check if other is equivalent to HiddenMarkovDataEncoder object instance.

        Args:
            other (Object): Object to compare to HiddenMarkovDataEncoder object instance.

        Returns:
            True if other is HiddenMarkovDataEncoder with equivalent 'len_encoder' and 'use_numba', else False.

        """
        if isinstance(other, HiddenMarkovDataEncoder):
            if self.use_numba == other.use_numba:
                if self.len_encoder == other.len_encoder:
                    return True
        else:
            return False

    def _seq_encode(self, x: list[list[T]]) -> tuple[E1, None]:
        """Sequence encoding for iid HMM sequence for vectorized numpy functions that do not use numba.

        Encoding  x: List[List[T]) where x[i] the ith HMM sequence of length n_i, s.t. x[i] = [x[i][0],...,x[i][n_i]].
        Call the t^th observation in the ith HMM sequence x[i][t].

        Blocks observations of each HMM sequence into blocks of same 't' value. I.e.
            seq_x = [ x[0][0],...,x[cnt][0], x[0][1],x[1][1],...,x[cnt][1],...]
        Note: That seq_x chunks will include x[i][t] values only if the sequence x[i] is length >= t.

        The returned value rv is a Tuple[Tuple[....], T_topic, T_len], where the first tuple is given by a Tuple of
            rv[0][0] (int): Total number of observed emissions from all HMM sequences.
            rv[0][1] (List[Tuple[int, int]]): Contains bands for t^th observation in HMM sequences stored in 'seq_x'.
            rv[0][2] (List[ndarray[int]]): List of numpy array on sequence indices that have a next observed emission.
            rv[0][3] (np.ndarray[int]): Numpy array of sequence lengths.
            rv[0][4] (np.ndarray[int]): 2-d matrix with rv[0][0] rows, and column length equal to the length of the
                largest HMM sequence. This is used to store the index of seq_x corresponding to emission x[i][t]. A -1
                is stored if the sequence length has already been met.
            rv[0][5] (ndarray): Numpy array containing lists index 'i' corresponding to x[i][t] block of 'seq_x'.
            rv[0][6] (T_topic): Sequence encoded value of 'seq_x'.

        The second entry of 'rv' is given by,
            rv[1] (T_topic): Sequence encoded observation values in order. Just for seq_init consistency.
            rv[2] (Optional[T_len]): Sequence encoded value of lengths of HMM distribution. None if len_encoder is
                the NullDataEncoder.

        Args:
            x(List[List[T]]): A sequence of iid observations from an HMM distribution of type T.

        Returns:
            Tuple[rv, None].

        """
        cnt = len(x)
        len_vec = [len(u) for u in x]
        len_enc = self.len_encoder.seq_encode(len_vec)

        len_vec = np.asarray(len_vec)
        max_len = len_vec.max()
        # len_cnt = np.bincount(len_vec)

        seq_x = []
        idx_loc = 0
        idx_mat = np.zeros((cnt, max_len), dtype=int) - 1
        idx_bands = []
        has_next = []
        idx_vec = []

        for i in range(max_len):
            i0 = idx_loc
            has_next_loc = []
            for j in range(cnt):
                if i < len_vec[j]:
                    if i < (len_vec[j] - 1):
                        has_next_loc.append(idx_loc - i0)
                    idx_vec.append(j)
                    seq_x.append(x[j][i])
                    idx_mat[j, i] = idx_loc
                    idx_loc += 1

            has_next.append(np.asarray(has_next_loc))
            idx_bands.append((i0, idx_loc))

        tot_cnt = len(seq_x)
        enc_data = self.emission_encoder.seq_encode(seq_x)
        idx_vec = np.asarray(idx_vec)

        xs = []
        for xx in x:
            if len(xx) > 0:
                xs.extend(xx)
        xs_enc = self.emission_encoder.seq_encode(xs)

        rv = ((tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), xs_enc, len_enc)
        return rv, None

    def seq_encode(self, x: list[list[T]]) -> tuple[E1 | None, E2 | None]:
        """Sequence encode sequences of iid HMM observations.

        Numba sequence encoding: Return type Tuple[Tuple[np.ndarray, np.ndarray, T_topic], Optional[T_len]] where
        T_topicis the type for 'emission_encoder.seq_encode()' and T_len is the type for 'len_encoder.seq_encode()'.
        The first entry of the returned value (rv_numba) is a Tuple of length-3,

            rv_numba[0][0] (ndarray[int]): Sequence id's for observed values.
            rv_numba[0][1] (ndarray[int]): Sequence lengths for each observed HMM sequence.
            rv_numba[0][2] (T_topic): Sequence encoded observation values.
            rv_numba[1] (Optional[T_len]): Sequence encoded values of sequence lengths. None if len_encoder is
                NullDataEncoder.

        If use_numba is False, calls HiddenMarkovDataEncoder._seq_encode(x). (See '_seq_encode' for details).


        Args:
            x (List[List[T]]): A sequence of iid observations from an HMM distribution of type T.

        Returns:
            Tuple[None, rv_numba] if use_numba, else Tuple[rv, None].

        """
        if not self.use_numba:
            return self._seq_encode(x)

        sz_list = [len(xx) for xx in x]
        sz = np.asarray(sz_list, dtype=np.int32)
        idx = np.repeat(np.arange(len(x), dtype=np.int32), sz)
        xs = list(itertools.chain.from_iterable(x))

        len_enc = self.len_encoder.seq_encode(sz_list)

        xs = self.emission_encoder.seq_encode(xs)

        return None, ((idx, sz, xs), len_enc)


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount1(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Args:
        x (np.ndarray[np.float64]): Group ids of rows
        w (np.ndarray[np.float64]): N by S numpy array with rows corresponding to x
        out (np.ndarray[np.float64]): Unique values in support of x by S.

    Returns:
        Numpy 2-d array.

    """
    for i in range(len(x)):
        out[x[i], :] += w[i, :]
    return out


@numba.njit("float64[:,:](int32[:], float64[:,:], float64[:,:])", cache=True)
def vec_bincount2(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    N = len(x)
    S = number of states.
    U = unique values in x can take on.

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[:, x[j]] += w[:, j]
    return out


# ---------------------------------------------------------------------------
# Engine-routed HMM forward-backward (numpy + torch, GPU/autograd capable).
#
# A single log-space implementation expressed in ComputeEngine array ops, replacing the per-backend
# Baum-Welch kernels for the engine path. Sequences are padded to a common length with a 0/1 mask;
# the recursions freeze the carried state at padded steps so variable-length sequences need no
# per-sequence Python loop (only a loop over time steps, which torch autograd unrolls).
# ---------------------------------------------------------------------------


def hmm_pad_log_emissions(log_emit_flat, sz):
    """Pack per-sequence-contiguous (tot, S) log-emissions into padded (N, Tmax, S) + (N, Tmax) mask.

    Args:
        log_emit_flat (np.ndarray): (tot, S) log emission densities, ordered sequence-by-sequence.
        sz (np.ndarray): (N,) sequence lengths summing to tot.

    Returns:
        Tuple of (padded (N, Tmax, S) float64, mask (N, Tmax) float64, offsets (N+1,) int).
    """
    sz = np.asarray(sz, dtype=np.int64)
    n = len(sz)
    tmax = int(sz.max()) if n > 0 else 0
    num_states = log_emit_flat.shape[1]
    padded = np.full((n, tmax, num_states), -np.inf, dtype=np.float64)
    mask = np.zeros((n, tmax), dtype=np.float64)
    offsets = np.concatenate([[0], np.cumsum(sz)]).astype(np.int64)
    for i in range(n):
        s0, s1 = offsets[i], offsets[i + 1]
        if s1 > s0:
            padded[i, : s1 - s0, :] = log_emit_flat[s0:s1, :]
            mask[i, : s1 - s0] = 1.0
    return padded, mask, offsets


def hmm_engine_forward_backward(engine, log_emit, log_w, log_a, mask, weights=None):
    """Log-space forward-backward over padded sequences using ComputeEngine ops.

    Args:
        engine (ComputeEngine): Array backend (numpy or torch).
        log_emit: (N, Tmax, S) log emission densities; padded slots may be -inf (masked out).
        log_w: (S,) log initial-state probabilities.
        log_a: (S, S) log transition matrix.
        mask: (N, Tmax) 1.0 for real observations, 0.0 for padding.
        weights: Optional (N,) per-sequence weights applied to gamma/xi/pi (E-step reweighting).

    Returns:
        Tuple of:
            ll: (N,) per-sequence emission log-likelihood (add the length model separately).
            gamma: (N, Tmax, S) posterior state probabilities (weighted; padded slots zero).
            xi_sum: (S, S) expected transition counts over all sequences and steps (weighted).
            pi: (N, S) initial-state posteriors (weighted).
    """
    mask_np = np.asarray(mask)
    n, tmax = mask_np.shape[0], mask_np.shape[1]
    log_emit = engine.asarray(log_emit)
    log_w = engine.asarray(log_w)
    log_a = engine.asarray(log_a)
    m = engine.asarray(mask)
    # log_w is either a shared (S,) initial vector or a per-sequence (N, S) vector (SemiSupervised HMM).
    log_w_2d = np.asarray(log_w).ndim == 2
    num_states = int(np.asarray(log_w).shape[-1])

    # forward pass (freeze alpha at padded steps so ll reads the last valid step)
    init = log_w if log_w_2d else log_w[None, :]
    alpha = init + log_emit[:, 0, :]
    alphas = [alpha]
    for t in range(1, tmax):
        cand = engine.logsumexp(alpha[:, :, None] + log_a[None, :, :], axis=1) + log_emit[:, t, :]
        alpha = engine.where(m[:, t][:, None] > 0, cand, alpha)
        alphas.append(alpha)
    alpha_stack = engine.stack(alphas, axis=1)
    ll = engine.logsumexp(alpha, axis=1)

    # backward pass (carry beta unchanged across padded steps)
    beta = engine.asarray(np.zeros((n, num_states)))
    betas = [None] * tmax
    betas[tmax - 1] = beta
    for t in range(tmax - 2, -1, -1):
        step = log_a[None, :, :] + (log_emit[:, t + 1, :] + beta)[:, None, :]
        cand = engine.logsumexp(step, axis=2)
        beta = engine.where(m[:, t + 1][:, None] > 0, cand, beta)
        betas[t] = beta
    beta_stack = engine.stack(betas, axis=1)

    wvec = engine.asarray(np.ones(n)) if weights is None else engine.asarray(weights)

    # gamma (posterior state probabilities). Empty/all-padded sequences give -inf alpha+beta whose
    # normalization is -inf - (-inf) = NaN; zero the padded slots with a mask-select (not a
    # multiply, since NaN * 0 = NaN) so degenerate sequences contribute nothing.
    zero = engine.asarray(0.0)
    ab = alpha_stack + beta_stack
    log_gamma = ab - engine.logsumexp(ab, axis=2, keepdims=True)
    gamma = engine.exp(log_gamma) * wvec[:, None, None]
    gamma = engine.where(m[:, :, None] > 0, gamma, zero)
    pi = gamma[:, 0, :]

    # xi (expected transition counts) summed over valid transitions
    xi_sum = engine.asarray(np.zeros((num_states, num_states)))
    for t in range(tmax - 1):
        log_xi = (
            alpha_stack[:, t, :][:, :, None]
            + log_a[None, :, :]
            + (log_emit[:, t + 1, :] + beta_stack[:, t + 1, :])[:, None, :]
            - ll[:, None, None]
        )
        contrib = engine.exp(log_xi) * wvec[:, None, None]
        contrib = engine.where((m[:, t + 1] > 0)[:, None, None], contrib, zero)
        xi_sum = xi_sum + engine.sum(contrib, axis=0)

    return ll, gamma, xi_sum, pi


def _register_hmm_engine_kernel():
    """Register the engine-resident HMM kernel (idempotent; called at import)."""
    from pysp.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class HiddenMarkovModelKernel(GenericKernel):
        """HMM kernel whose E-step runs the forward-backward on the active engine.

        Scoring reuses the generic backend path. For non-numpy engines the accumulation uses
        ``HiddenMarkovAccumulator.seq_update_engine`` (the ComputeEngine forward-backward), so EM
        estimation runs on torch (GPU/autograd). The numpy engine keeps the tuned host Baum-Welch.
        """

        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("HiddenMarkovModelKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class HiddenMarkovModelKernelFactory(KernelFactory):
        """Build the engine-resident HMM kernel, falling back to the generic kernel as needed."""

        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return HiddenMarkovModelKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(HiddenMarkovModelDistribution, HiddenMarkovModelKernelFactory())


_register_hmm_engine_kernel()


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
HiddenMarkovModelAccumulator = HiddenMarkovAccumulator
HiddenMarkovModelAccumulatorFactory = HiddenMarkovAccumulatorFactory
HiddenMarkovModelDataEncoder = HiddenMarkovDataEncoder
HiddenMarkovModelEstimator = HiddenMarkovEstimator
HiddenMarkovModelSampler = HiddenMarkovSampler
