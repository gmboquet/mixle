"""First-order Markov-chain distributions over finite or structured states.

The assumed data type for the stats-space is T.

The density of Markov chain is given by for sequence of length n, x=[x[0],x[1],...,x[n-1]]

    p_mat(x) = p_mat(x[0])*p_mat(x[1]|x[0])*...*p_mat(x[n-1]|x[n-2])*P_len(n)

where p_mat(x[i+1]|x[i]) is the transition probability, p_mat(x[0]) is the init-probability, and P_len(n) is given
by the length distribution density.

Note if len(x) = 0, only log(P_len(0)) is returned.

"""

import heapq
import itertools
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState
from scipy.sparse import dok_matrix

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, LengthFrontierMerge
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
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)

T = TypeVar("T")  ### state type
T1 = TypeVar("T1")  ### Type for length distribution sufficient statsitics value.
suff_stat_type = tuple[dict[T, float], dict[T, dict[T, float]], Any | None]
enc_data_type = tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]


# --- Conjugate Dirichlet prior machinery (folded from mixle.bstats.markov_chain) ---
#
# The prior is over a FIXED ordered list of states ``states`` (length S): a Dirichlet on the
# initial-state probabilities and an independent Dirichlet on each transition row.  It is carried
# as ``prior = (states, init_prior, row_priors)`` where ``init_prior`` is a
# mixle.stats.bayes.dirichlet.DirichletDistribution and ``row_priors`` is a length-S list of the same.
# ``prior=None`` (the default) preserves the existing maximum-likelihood / pseudo-count path
# byte-identically.


def _bstats_dirichlet():
    from mixle.stats.bayes.dirichlet import DirichletDistribution

    return DirichletDistribution


def markov_chain_dirichlet_default_prior(states: Sequence[T]):
    """Returns the default ``(states, init_prior, row_priors)`` prior of unit-parameter Dirichlets.

    Args:
        states (Sequence[T]): Ordered list of the S state values the priors range over.

    Returns:
        Tuple ``(list_of_states, DirichletDistribution, list_of_DirichletDistribution)``.

    """
    dirichlet = _bstats_dirichlet()
    states = list(states)
    s = len(states)
    return (
        states,
        dirichlet(np.ones(s)),
        [dirichlet(np.ones(s)) for _ in range(s)],
    )


def _unpack_markov_chain_prior(prior):
    """Normalize the prior into ``(states, init_prior, row_priors)``.

    Accepts the ``(states, init_prior, row_priors)`` tuple form.

    """
    states, init_prior, row_priors = prior[0], prior[1], list(prior[2])
    return list(states), init_prior, row_priors


def _map_probs(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Dirichlet MAP with boundary clamp; posterior mean when degenerate.

    Mirrors mixle.bstats.markov_chain._map_probs exactly.
    """
    num = np.maximum(counts + alpha - 1.0, 0.0)
    tot = num.sum()
    if tot > 0:
        return num / tot
    cpp = counts + alpha
    return cpp / cpp.sum()


def stationary_distribution(transitions: np.ndarray) -> np.ndarray:
    """Stationary distribution ``pi`` of a row-stochastic matrix (``pi A = pi``, ``sum pi = 1``).

    Solved as the constrained least-squares system ``[(I - A^T); 1^T] pi = [0; 1]`` so it is robust for
    reducible/near-singular chains (it returns one valid stationary distribution). The result is
    clipped to non-negative and renormalized.

    Args:
        transitions (np.ndarray): a square row-stochastic transition matrix.

    Returns:
        1-d numpy array of stationary probabilities (length = number of states).
    """
    a = np.asarray(transitions, dtype=np.float64)
    n = a.shape[0]
    lhs = np.vstack([np.eye(n) - a.T, np.ones((1, n))])
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    pi = np.clip(pi, 0.0, None)
    total = pi.sum()
    return pi / total if total > 0.0 else np.full(n, 1.0 / n)


class MarkovChainDistribution(SequenceEncodableProbabilityDistribution):
    """Markov-chain distribution over finite-state sequences."""

    def __init__(
        self,
        init_prob_map: dict[T, float],
        transition_map: dict[T, dict[T, float]],
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        default_value: float = 0.0,
        name: str | None = None,
        prior=None,
    ) -> None:
        """Create a Markov-chain distribution compatible with state type ``T``.

        Args:
            init_prob_map (Dict[T, float]): Probability of each initial values of data type T.
            transition_map (Dict[T, Dict[T, float]]): Transition probability map.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length distribution for length of
                observation sequence.
            default_value (float): Default probability for value outside support.
            name (Optional[str]): Set name to MarkovChainDistribution object.
            prior: Optional ``(states, init_prior, row_priors)`` conjugate Dirichlet prior (see
                set_prior()). ``None`` (default) is a plain point model whose estimation path is
                unchanged.

        Attributes:
            init_prob_map (Dict[T, float]): Probability of each initial values of data type T.
            transition_map (Dict[T, Dict[T, float]]): Transition probability map.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length distribution for length of
                observation sequence.
            default_value (float): Default probability for value outside support.
            name (Optional[str]): Set name to MarkovChainDistribution object.
            all_vals (Set[T]): Set of all values in state-space.
            loginit_prob_map (Dict[T, float]): Dictionary mapping initial state value to log probability.
            log_transition_map (Dict[T, Dict[T, float]]): Dictionary mapping state to state transition
                log-probabilities.
            log_dv (float): Log default value.
            log_dtv (float): Log of default value scaled by number of state-values + 1.
            log1p_dv (float): Log of 1 plus default_value.
            key_map (Dict[T, int]): Maps each state-value in all_vals to integer [1, len(all_vals)+1]
            inv_key_map (List[T]): List of all state-values (keys).
            num_keys (int): Number of state-values (len(keys)).
            init_log_pvec (ndarray): Log-probabilities of each initial value. Entry 0, is  log_dv. (len == num_keys+1).
            trans_log_pvec (dok_matrix): Dictionary of keys for sparse log transition probabilities.

        """
        self.name = name
        self.init_prob_map = init_prob_map
        self.transition_map = transition_map
        self.len_dist = len_dist if len_dist is not None else NullDistribution()

        self.all_vals = (
            set(init_prob_map.keys())
            .union(set([v for u in transition_map.values() for v in u.keys()]))
            .union(transition_map.keys())
        )
        self.loginit_prob_map = {u[0]: -np.inf if u[1] == 0.0 else log(u[1]) for u in init_prob_map.items()}

        self.log_transition_map = dict(
            (key, dict((u[0], log(u[1])) for u in transition_map[key].items())) for key in transition_map.keys()
        )

        self.default_value = max(min(default_value, 1.0), 0.0)
        self.log_dv = -np.inf if default_value == 0.0 else log(self.default_value)
        self.log_dtv = -np.inf if default_value == 0.0 else (log(default_value) - np.log(len(self.all_vals) + 1))
        self.log1p_dv = log(one + self.default_value)

        num_keys = len(self.all_vals)

        keys = list(self.all_vals)
        sidx = np.argsort(keys)
        keys = [keys[i] for i in sidx]

        self.key_map = {keys[i]: i + 1 for i in range(num_keys)}
        self.inv_key_map = keys
        self.num_keys = num_keys

        self.init_log_pvec = np.zeros(num_keys + 1)
        self.trans_log_pvec = dok_matrix((num_keys + 1, num_keys + 1))

        for k1, v1 in init_prob_map.items():
            self.init_log_pvec[self.key_map.get(k1, 0.0)] = -np.inf if v1 == 0.0 else np.log(v1)

        for k1, v1 in transition_map.items():
            k1_idx = self.key_map.get(k1, 0)
            for k2, v2 in v1.items():
                self.trans_log_pvec[k1_idx, self.key_map.get(k2, 0)] = -np.inf if v2 == 0 else np.log(v2)

        self.init_log_pvec[0] = self.log_dv
        self.trans_log_pvec[:, 0] = self.log_dv
        self.trans_log_pvec[0, :] = self.log_dv - np.log(num_keys + 1)

        self.set_prior(prior)

    def get_prior(self):
        """Returns the conjugate prior in ``(states, init_prior, row_priors)`` form (or None)."""
        if not self.has_conj_prior:
            return None
        return (list(self.prior_states), self.init_prior, list(self.row_priors))

    def set_prior(self, prior) -> None:
        """Set the conjugate Dirichlet prior and precompute its digamma expectations.

        With Dirichlet ``init_prior`` and Dirichlet ``row_priors`` (each over the fixed ordered
        ``states``) this caches the digamma expectations E[ln p_k] = psi(alpha_k) - psi(sum alpha)
        used by expected_log_density and sets ``has_conj_prior`` accordingly. ``prior=None`` leaves
        the distribution a plain point model.

        Args:
            prior: ``(states, init_prior, row_priors)`` tuple or None.

        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution

        if prior is None:
            self.prior = None
            self.prior_states = None
            self.init_prior = None
            self.row_priors = None
            self.e_log_init = None
            self.e_log_trans = None
            self.has_conj_prior = False
            return

        states, init_prior, row_priors = _unpack_markov_chain_prior(prior)
        self.prior = prior
        self.prior_states = states
        self.init_prior = init_prior
        self.row_priors = row_priors

        if isinstance(init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in row_priors
        ):
            a0 = np.asarray(init_prior.get_parameters(), dtype=float)
            self.e_log_init = digamma(a0) - digamma(a0.sum())
            self.e_log_trans = np.zeros((len(states), len(states)))
            for i, row_prior in enumerate(row_priors):
                ai = np.asarray(row_prior.get_parameters(), dtype=float)
                self.e_log_trans[i, :] = digamma(ai) - digamma(ai.sum())
            self.has_conj_prior = True
        else:
            self.e_log_init = None
            self.e_log_trans = None
            self.has_conj_prior = False

    def expected_log_density(self, x: list[T]) -> float:
        """Variational E_q[log p(x)] under the Dirichlet priors over a state sequence.

        Replaces the initial/transition log-probabilities with their digamma expectations
        E[ln p_k] = psi(alpha_k) - psi(sum alpha); the length term is added as in log_density().
        Falls back to the plug-in log_density(x) when no conjugate prior is set.

        Args:
            x (List[T]): An observed Markov chain state sequence.

        Returns:
            Expected log-density of the Markov chain at x.

        """
        if not self.has_conj_prior:
            return self.log_density(x)

        rv = 0.0
        if len(x) != 0:
            idx = {s: i for i, s in enumerate(self.prior_states)}
            rv = float(self.e_log_init[idx[x[0]]])
            for i in range(1, len(x)):
                rv += float(self.e_log_trans[idx[x[i - 1]], idx[x[i]]])
        rv += self.len_dist.log_density(len(x))
        return rv

    def seq_expected_log_density(self, x: enc_data_type) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Falls back to seq_log_density(x) when no conjugate prior is set.

        Args:
            x: Encoded sequences from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per sequence.

        """
        if not self.has_conj_prior:
            return self.seq_log_density(x)

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x

        idx = {s: i for i, s in enumerate(self.prior_states)}
        loc_key_map = np.asarray([idx[u] for u in inv_key_map])

        rv = np.zeros(sz, dtype=float)
        if len(idx1) > 0:
            temp = self.e_log_trans[loc_key_map[prev_x], loc_key_map[next_x]]
            rv = np.bincount(idx1, weights=temp, minlength=sz)
        rv[idx0] += self.e_log_init[loc_key_map[init_x]]

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def compute_capabilities(self):
        """Return compute-backend metadata inherited from the optional length distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.len_dist)
        return DistributionCapabilities(
            engine_ready=child.engine_ready, kernel_status="generic", numpy_only_reason=child.numpy_only_reason
        )

    def compute_declaration(self):
        """Return the symbolic declaration for Markov-chain transition and length statistics."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = () if length is None else (length,)
        return DistributionDeclaration(
            name="markov_chain",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("init_prob_map", constraint="simplex_map"),
                ParameterSpec("transition_map", constraint="row_simplex_map"),
                ParameterSpec("default_value", constraint="unit_interval", differentiable=False),
            ),
            statistics=(
                StatisticSpec("initial_counts", kind="mapping"),
                StatisticSpec("transition_counts", kind="mapping"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="finite_state_sequence",
            children=children,
            child_roles=("length",) if length is not None else (),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self):
        """Return a constructor-style representation of the distribution."""
        s1 = repr(dict(sorted(self.init_prob_map.items(), key=lambda u: u[0])))
        temp = sorted(self.transition_map.items(), key=lambda u: u[0])
        s2 = repr(dict([(k, dict(sorted(v.items(), key=lambda u: u[0]))) for k, v in temp]))
        s3 = str(self.len_dist)
        s4 = repr(self.default_value)
        s5 = repr(self.name)

        return "MarkovChainDistribution(%s, %s, len_dist=%s, default_value=%s, name=%s)" % (s1, s2, s3, s4, s5)

    def density(self, x: list[T]) -> float:
        """Return density of MarkovChainDistribution at observed sequence x.

        Returns exponential of log_density(x). See log_density() for details.

        Args:
            x (List[T]): An observed Markov chain sequence of data type T.

        Returns:
            Density of Markov chain at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: list[T]) -> float:
        """Return log-density of MarkovChainDistribution at observed sequence x.

        Density of Markov chain is given by for sequence of length n, x=[x[0],x[1],...,x[n-1]]

            p_mat(x) = p_mat(x[0])*p_mat(x[1]|x[0])*...*p_mat(x[n-1]|x[n-2])*P_len(n)

        where p_mat(x[i+1]|x[i]) is the transition probability, p_mat(x[0]) is the init-probability, and P_len(n) is given
        by the length distribution density.

        Note if len(x) = 0, only log(P_len(0)) is returned.

        Args:
            x (List[T]): An observed Markov chain sequence of data type T.

        Returns:
            Log-density of Markov chain at x.

        """
        if len(x) == 0:
            rv = 0.0
        else:
            rv = self.loginit_prob_map.get(x[0], self.log_dv) - self.log1p_dv

            for i in range(1, len(x)):
                if x[i - 1] in self.log_transition_map:
                    rv += self.log_transition_map[x[i - 1]].get(x[i], self.log_dv) - self.log1p_dv
                else:
                    rv += self.log_dtv - self.log1p_dv

        rv += self.len_dist.log_density(len(x))

        return rv

    def seq_log_density(self, x: enc_data_type) -> np.ndarray:
        """Vectorized evaluation of log_density of Markov Chain for an encoded sequence of observations x.

        Computationally efficient implementation of log_density() for sequence encoded data x.

        The arg value x is a Tuple of length 8 with entries:

            x[0] (int): Number of total observations (number of Markov sequences).
            x[1] (ndarray[int]): Sequence index for initial state observations.
            x[2] (ndarray[int]): Sequence index for non-initial state observations in a sequence greater than len 1.
            x[3] (ndarray[int]): Numpy array of observations index in inv_key_map for initial states.
            x[4] (ndarray[int]): State-to-state index value of inv_key_map for initial state value.
            x[5] (ndarray[int]): State-to-state index value of inv_key_map for transition.
            x[6] (ndarray[T]): Maps integer index value to value in state-space (T).
            x[7] (Optional[T1]): Encoded sequence of lengths from len_encoder. None if no length distribution to be
                estimated.

        Args:
            x: See above for details.

        Returns:
            Numpy of length x[0], containing the log-density of Markov chain at each observation in x.

        """
        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x

        loc_key_map = np.asarray([self.key_map.get(u, 0) for u in inv_key_map])

        temp = self.trans_log_pvec[loc_key_map[prev_x], loc_key_map[next_x]].toarray().flatten() - self.log1p_dv
        rv = np.bincount(idx1, weights=temp, minlength=sz)
        rv[idx0] += self.init_log_pvec[loc_key_map[init_x]] - self.log1p_dv

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def backend_seq_log_density(self, x: enc_data_type, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded Markov-chain sequences."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x

        loc_key_map = engine.asarray(np.asarray([self.key_map.get(u, 0) for u in inv_key_map], dtype=np.int64))
        init_log_pvec = engine.asarray(self.init_log_pvec)
        trans_log_pvec = engine.asarray(self.trans_log_pvec.toarray())
        rv = engine.zeros(sz)

        if len(idx1) > 0:
            prev_idx = loc_key_map[engine.asarray(prev_x)]
            next_idx = loc_key_map[engine.asarray(next_x)]
            values = trans_log_pvec[prev_idx, next_idx] - self.log1p_dv
            rv = engine.index_add(rv, engine.asarray(idx1), values)

        if len(idx0) > 0:
            init_idx = loc_key_map[engine.asarray(init_x)]
            values = init_log_pvec[init_idx] - self.log1p_dv
            rv = engine.index_add(rv, engine.asarray(idx0), values)

        if len_enc is not None:
            rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)

        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["MarkovChainDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked fixed-support Markov-chain parameters."""
        from mixle.stats.compute.stacked import stacked_component_params

        labels = tuple(dists[0].inv_key_map)
        null_len_dist = supports(dists[0].len_dist, Neutral)
        if any(
            tuple(dist.inv_key_map) != labels or supports(dist.len_dist, Neutral) != null_len_dist for dist in dists
        ):
            raise ValueError("Stacked MarkovChainDistribution components require shared states and length policy.")

        length_route = None
        if not null_len_dist:
            try:
                length_route = stacked_component_params([dist.len_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "MarkovChain length child %s is not stackable: %s" % (type(dists[0].len_dist).__name__, exc)
                )

        return {
            "__pysp_component_axis__": {"init_log_p": 1, "trans_log_p": 2, "log1p_dv": 0},
            "labels": labels,
            "init_log_p": engine.asarray(np.stack([dist.init_log_pvec for dist in dists], axis=1)),
            "trans_log_p": engine.asarray(np.stack([dist.trans_log_pvec.toarray() for dist in dists], axis=2)),
            "log1p_dv": engine.asarray(np.asarray([dist.log1p_dv for dist in dists], dtype=np.float64)),
            "length_route": length_route,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: enc_data_type, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Markov-chain sequence log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        label_to_idx = {label: i + 1 for i, label in enumerate(params["labels"])}
        loc_key_map = np.asarray([label_to_idx.get(label, 0) for label in inv_key_map], dtype=np.int64)
        rv = engine.zeros((sz, int(params["num_components"])))

        if len(idx1) > 0:
            prev_idx = loc_key_map[prev_x]
            next_idx = loc_key_map[next_x]
            values = (
                params["trans_log_p"][engine.asarray(prev_idx), engine.asarray(next_idx), :]
                - params["log1p_dv"][None, :]
            )
            rv = engine.index_add(rv, engine.asarray(idx1), values)

        if len(idx0) > 0:
            init_idx = loc_key_map[init_x]
            values = params["init_log_p"][engine.asarray(init_idx), :] - params["log1p_dv"][None, :]
            rv = engine.index_add(rv, engine.asarray(idx0), values)

        if params["length_route"] is not None and len_enc is not None:
            rv = rv + stacked_component_log_density(len_enc, params["length_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: enc_data_type, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy ``(initial_counts, transition_counts, length_stat)`` statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        ww = engine.asarray(weights)
        num_components = int(params["num_components"])

        if len(idx0) > 0 and len(inv_key_map) > 0:
            init_weights = ww[engine.asarray(idx0)]
            zero_rows = init_weights * engine.asarray(0.0)
            rows = []
            init_engine = engine.asarray(init_x)
            for value_index in range(len(inv_key_map)):
                mask = init_engine == engine.asarray(value_index)
                rows.append(engine.sum(engine.where(mask[:, None], init_weights, zero_rows), axis=0))
            init_counts = np.asarray(engine.to_numpy(engine.stack(rows, axis=0)), dtype=np.float64)
        else:
            init_counts = np.zeros((len(inv_key_map), num_components), dtype=np.float64)

        observed_pairs = list(dict.fromkeys((int(prev_x[i]), int(next_x[i])) for i in range(len(prev_x))))
        if observed_pairs:
            trans_weights = ww[engine.asarray(idx1)]
            zero_rows = trans_weights * engine.asarray(0.0)
            trans_rows = []
            prev_engine = engine.asarray(prev_x)
            next_engine = engine.asarray(next_x)
            for prev_i, next_i in observed_pairs:
                mask = (prev_engine == engine.asarray(prev_i)) & (next_engine == engine.asarray(next_i))
                trans_rows.append(engine.sum(engine.where(mask[:, None], trans_weights, zero_rows), axis=0))
            trans_counts = np.asarray(engine.to_numpy(engine.stack(trans_rows, axis=0)), dtype=np.float64)
        else:
            trans_counts = np.zeros((0, num_components), dtype=np.float64)

        if params["length_route"] is None or len_enc is None:
            length_by_component = tuple(None for _ in range(num_components))
        else:
            outer_estimators = tuple(getattr(estimator, "estimators", ()))
            length_estimators = tuple(
                getattr(component_est, "len_estimator", None) for component_est in outer_estimators
            )
            length_estimator = (
                StackedEstimatorView(length_estimators) if len(length_estimators) == num_components else None
            )
            length_stats = stacked_component_sufficient_statistics(
                len_enc, ww, params["length_route"], engine, length_estimator
            )
            length_by_component = unstack_component_stats(length_stats, num_components)

        return tuple(
            (
                {
                    inv_key_map[value_index]: float(init_counts[value_index, component])
                    for value_index in range(len(inv_key_map))
                    if init_counts[value_index, component] != 0.0
                },
                _markov_chain_transition_stats(inv_key_map, observed_pairs, trans_counts, component),
                length_by_component[component],
            )
            for component in range(num_components)
        )

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for fixed-support autograd fitting."""
        init_keys = tuple(self.init_prob_map.keys())
        init_logits = tensor_param([self.init_prob_map[key] for key in init_keys], engine, torch, transform="logits")
        leaves.append(init_logits)

        trans_keys = {key: tuple(row.keys()) for key, row in self.transition_map.items()}
        trans_logits = {}
        for key, row_keys in trans_keys.items():
            logits = tensor_param(
                [self.transition_map[key][row_key] for row_key in row_keys], engine, torch, transform="logits"
            )
            trans_logits[key] = logits
            leaves.append(logits)

        len_child = None if supports(self.len_dist, Neutral) else recurse(self.len_dist, engine, torch, leaves)
        return _MarkovChainGradientFitState(self, init_keys, init_logits, trans_keys, trans_logits, len_child)

    def sampler(self, seed: int | None = None) -> "MarkovChainSampler":
        """Return a sampler for this Markov-chain distribution.

        Raises exception if length distribution (len_dist) was not specified in initialization.

        Args:
            seed (Optional[int]): Used to set the seed of random number generator for sampling.

        Returns:
            MarkovChainSampler object.

        """
        return MarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "MarkovChainEstimator":
        """Create an estimator initialized from this Markov-chain distribution.

        Args:
            pseudo_count (Optional[float]): Prior mass used to smooth initial and transition counts.

        Returns:
            MarkovChainEstimator: Estimator configured with the same length estimator and prior.

        """
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        return MarkovChainEstimator(
            pseudo_count=pseudo_count, len_estimator=len_est, name=self.name, prior=self.get_prior()
        )

    def dist_to_encoder(self) -> "MarkovChainDataEncoder":
        """Create a data encoder for Markov-chain observation sequences.

        Note: len_encoder is passed as NullDataEncoder() if len_dist is not to be estimated.

        Returns:
            MarkovChainDataEncoder: Encoder using this distribution's length encoder.

        """
        len_encoder = self.len_dist.dist_to_encoder()
        return MarkovChainDataEncoder(len_encoder=len_encoder)

    def enumerator(self) -> "MarkovChainEnumerator":
        """Returns MarkovChainEnumerator iterating state sequences in descending probability order."""
        return MarkovChainEnumerator(self)

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Structural count index: a forward DP carrying a count histogram per (length, end-state).

        log p(x) = log p_init(x0) + sum_i log p_trans(x_i|x_{i-1}) + log p(len). The forward
        recursion is lifted into the count semiring: alpha[t][s] is the histogram (over the fine
        bucket of accumulated log probability) of length-t prefixes ending in state s, with
        ``alpha[1][s] = delta(bucket(log p_init(s)))`` and
        ``alpha[t+1][s'] = sum_s alpha[t][s].shift(bucket(log p_trans(s'|s)))``. Per length L the
        sequence histogram pools the end states and shifts by the length term; the total pools
        lengths. Sequences are unranked by choosing the end state, then walking the trellis
        backward choosing predecessors by count.
        """
        from mixle.enumeration.quantization.semiring import CountSemiring
        from mixle.stats.compute.pdist import EnumerationError

        if self.default_value != 0.0:
            raise EnumerationError(self, reason="non-zero default_value gives an unbounded support")
        if supports(self.len_dist, Neutral):
            raise EnumerationError(self, reason="no length distribution is modeled (len_dist is Null)")

        sr = CountSemiring()
        # The RECURSIVE law lifted into the count semiring. alpha[t][s] is the carrier element for
        # length-t prefixes ending in state s; a transition is scale (scalar transition log-prob
        # shift) + map_values (append the symbol) + plus (pool predecessors) -- no convolution, since
        # each step adds one scalar. The carrier's reified nodes unrank *iteratively* (see _unrank),
        # so a length-L path no longer recurses O(L) deep.
        # Fixed iteration order over states; exact (default_value==0 makes the map values exact).
        init_lp = {s: float(lp) for s, lp in self.loginit_prob_map.items() if lp > -np.inf}
        state_order: list[Any] = list(init_lp.keys())
        seen = set(state_order)
        for s_prev, m in self.log_transition_map.items():
            for s_next in m:
                if s_next not in seen and m[s_next] > -np.inf:
                    seen.add(s_next)
                    state_order.append(s_next)
        # Transitions into each next-state, in predecessor state_order: (predecessor, log p_trans).
        into: dict[Any, list[tuple[Any, float]]] = {s: [] for s in state_order}
        for s_prev in state_order:
            m = self.log_transition_map.get(s_prev, {})
            for s_next in state_order:
                lp = m.get(s_next, -np.inf)
                if lp > -np.inf:
                    into[s_next].append((s_prev, float(lp)))

        truncated = False
        lengths: list[tuple[int, float]] = []
        _LEN_CAP = 1 << 24
        for length, lp_len in child_enumerator(self.len_dist, "MarkovChainDistribution.len_dist"):
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
            return sr.zero(), truncated

        max_len = max(L for L, _ in lengths)
        alpha: list[dict[Any, Any]] = [None]  # index 0 unused
        alpha.append(
            {s: sr.map_values(sr.leaf(s, init_lp[s], quantizer), lambda v: [v]) for s in state_order if s in init_lp}
        )
        for t in range(2, max_len + 1):
            prev = alpha[t - 1]
            cur: dict[Any, Any] = {}
            for s_next in state_order:
                acc = sr.zero()
                built = False
                for s_prev, lp_tr in into[s_next]:
                    ph = prev.get(s_prev)
                    if ph is None or ph.hist.is_empty():
                        continue
                    step = sr.map_values(
                        sr.scale(ph, lp_tr, quantizer, max_fine_bucket), lambda seq, s=s_next: seq + [s]
                    )
                    acc = step if not built else sr.plus(acc, step)
                    built = True
                if built and not acc.hist.is_empty():
                    cur[s_next] = acc
            alpha.append(cur)
            if not cur:
                truncated = True
                break
        built_len = len(alpha) - 1

        total = sr.zero()
        total_built = False
        for L, lp_len in lengths:
            if L == 0:
                piece = sr.map_values(sr.leaf((), lp_len, quantizer), lambda v: [])
                total = piece if not total_built else sr.plus(total, piece)
                total_built = True
                continue
            if L > built_len or not alpha[L]:
                truncated = True
                continue
            pooled = sr.zero()
            pooled_built = False
            for s in state_order:
                h = alpha[L].get(s)
                if h is not None:
                    pooled = h if not pooled_built else sr.plus(pooled, h)
                    pooled_built = True
            if not pooled_built:
                continue
            piece = sr.scale(pooled, lp_len, quantizer, max_fine_bucket)
            if piece.hist.is_empty():
                continue
            total = piece if not total_built else sr.plus(total, piece)
            total_built = True

        return total, truncated


class _MarkovChainGradientFitState:
    """Autograd state for fixed-support MarkovChainDistribution fitting."""

    def __init__(
        self,
        template: MarkovChainDistribution,
        init_keys: tuple[Any, ...],
        init_logits: Any,
        trans_keys: dict[Any, tuple[Any, ...]],
        trans_logits: dict[Any, Any],
        len_child: Any,
    ) -> None:
        self.template = template
        self.init_keys = init_keys
        self.init_logits = init_logits
        self.trans_keys = trans_keys
        self.trans_logits = trans_logits
        self.len_child = len_child

    def shadow(self, torch, shadow_child):
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow._gradient_init_keys = self.init_keys
        shadow._gradient_init_log_probs = torch.log_softmax(self.init_logits, dim=0)
        shadow._gradient_trans_keys = self.trans_keys
        shadow._gradient_trans_log_probs = {
            key: torch.log_softmax(logits, dim=0) for key, logits in self.trans_logits.items()
        }
        if self.len_child is not None:
            shadow.len_dist = shadow_child(self.len_child, torch)
        return shadow

    def score(self, enc, engine, torch, score_child):
        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = enc
        rv = engine.zeros(sz)
        inv_values = list(inv_key_map)

        if len(idx0) > 0:
            init_pos = {key: i for i, key in enumerate(self.init_keys)}
            positions = np.asarray([init_pos.get(inv_values[i], -1) for i in init_x], dtype=np.int64)
            scores = engine.zeros(len(init_x)) + self.template.log_dv - self.template.log1p_dv
            known = np.flatnonzero(positions >= 0)
            if len(known) > 0:
                log_probs = torch.log_softmax(self.init_logits, dim=0)
                scores[engine.asarray(known)] = log_probs[engine.asarray(positions[known])] - self.template.log1p_dv
            rv = engine.index_add(rv, engine.asarray(idx0), scores)

        if len(idx1) > 0:
            default_scores = []
            for prev_i in prev_x:
                prev_key = inv_values[prev_i]
                if prev_key in self.trans_logits:
                    default_scores.append(self.template.log_dv - self.template.log1p_dv)
                else:
                    default_scores.append(self.template.log_dtv - self.template.log1p_dv)
            scores = engine.asarray(np.asarray(default_scores, dtype=np.float64))
            for key, row_keys in self.trans_keys.items():
                row_positions = np.flatnonzero(np.asarray([inv_values[i] == key for i in prev_x], dtype=bool))
                if len(row_positions) == 0:
                    continue
                next_pos = {value: i for i, value in enumerate(row_keys)}
                positions = np.asarray([next_pos.get(inv_values[next_x[i]], -1) for i in row_positions], dtype=np.int64)
                known = np.flatnonzero(positions >= 0)
                if len(known) > 0:
                    target = row_positions[known]
                    log_probs = torch.log_softmax(self.trans_logits[key], dim=0)
                    scores[engine.asarray(target)] = (
                        log_probs[engine.asarray(positions[known])] - self.template.log1p_dv
                    )
            rv = engine.index_add(rv, engine.asarray(idx1), scores)

        if self.len_child is not None and len_enc is not None:
            rv = rv + score_child(self.len_child, len_enc, engine, torch)
        return rv

    def build(self, torch, build_child, detach_value):
        init_probs = torch.softmax(self.init_logits, dim=0).detach().cpu().numpy()
        init_map = {key: float(prob) for key, prob in zip(self.init_keys, init_probs)}
        trans_map = {}
        for key, row_keys in self.trans_keys.items():
            probs = torch.softmax(self.trans_logits[key], dim=0).detach().cpu().numpy()
            trans_map[key] = {value: float(prob) for value, prob in zip(row_keys, probs)}
        len_dist = (
            getattr(self.template, "len_dist", None) if self.len_child is None else build_child(self.len_child, torch)
        )
        return type(self.template)(
            init_map,
            trans_map,
            len_dist=len_dist,
            default_value=getattr(self.template, "default_value", 0.0),
            name=getattr(self.template, "name", None),
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        from mixle.stats.compute.gradient import dirichlet_alpha_tensor, markov_chain_priors, prior_family, prior_zero

        init_prior, trans_priors, len_prior = markov_chain_priors(priors, tuple(self.trans_keys.keys()))
        rv = prior_zero(torch, engine, self.init_logits)

        if prior_family(init_prior) == "dirichlet":
            alpha = dirichlet_alpha_tensor(init_prior.get("alpha"), self.init_keys, self.init_logits, engine, torch)
            rv = rv + torch.sum((alpha - 1.0) * torch.log_softmax(self.init_logits, dim=0))
        elif prior_strength != 0.0:
            alpha = 1.0 + float(prior_strength) / max(1, self.init_logits.numel())
            rv = rv + torch.sum((alpha - 1.0) * torch.log_softmax(self.init_logits, dim=0))

        for key, logits in self.trans_logits.items():
            prior = trans_priors.get(key)
            labels = self.trans_keys[key]
            if prior_family(prior) == "dirichlet":
                alpha = dirichlet_alpha_tensor(prior.get("alpha"), labels, logits, engine, torch)
                rv = rv + torch.sum((alpha - 1.0) * torch.log_softmax(logits, dim=0))
            elif prior_strength != 0.0:
                alpha = 1.0 + float(prior_strength) / max(1, logits.numel())
                rv = rv + torch.sum((alpha - 1.0) * torch.log_softmax(logits, dim=0))

        if self.len_child is not None:
            rv = rv + prior_child(self.len_child, len_prior, prior_strength, torch, engine, initial_leaves_by_id)
        return rv


def _markov_chain_transition_stats(
    inv_key_map: np.ndarray, observed_pairs: Sequence[tuple[int, int]], counts: np.ndarray, component: int
) -> dict[Any, dict[Any, float]]:
    """Return legacy nested transition-count maps for one stacked component."""
    rv: dict[Any, dict[Any, float]] = {}
    for pair_index, (prev_i, next_i) in enumerate(observed_pairs):
        prev_key = inv_key_map[prev_i]
        next_key = inv_key_map[next_i]
        if prev_key not in rv:
            rv[prev_key] = {}
        rv[prev_key][next_key] = float(counts[pair_index, component])
    return rv


class MarkovChainEnumerator(DistributionEnumerator):
    """Best-first enumerator for finite-state sequences with a modeled length law."""

    def __init__(self, dist: "MarkovChainDistribution") -> None:
        """Enumerates state sequences in descending probability order.

        Lengths come lazily from the length distribution's enumerator; within each length the
        sequences are produced by a best-first search over prefixes, scored with the admissible
        bound exact_prefix_log_prob + remaining_steps * max_transition_log_prob (each remaining
        step can contribute at most the largest single transition log-probability).

        Raises EnumerationError when default_value is non-zero (unbounded support over
        arbitrary values) or when no length distribution is modeled.

        Args:
            dist (MarkovChainDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if dist.default_value != 0.0:
            raise EnumerationError(dist, reason="non-zero default_value gives an unbounded support")
        if supports(dist.len_dist, Neutral):
            raise EnumerationError(dist, reason="no length distribution is modeled (len_dist is Null)")
        self._init = [(v, lp) for v, lp in dist.loginit_prob_map.items() if lp > -np.inf]
        self._trans = {s: [(w, lp) for w, lp in m.items() if lp > -np.inf] for s, m in dist.log_transition_map.items()}
        steps = [lp for m in self._trans.values() for _, lp in m]
        self._max_step = min(max(steps), 0.0) if steps else -np.inf
        len_stream = BufferedStream(child_enumerator(dist.len_dist, "MarkovChainDistribution.len_dist"))
        self._merge = LengthFrontierMerge(len_stream, self._kbest_paths)

    def _kbest_paths(self, n: int, lp_len: float):
        if n == 0:
            yield ([], lp_len)
            return
        counter = itertools.count()
        heap = []
        for v, lp in self._init:
            bound = lp_len + lp + (n - 1) * self._max_step
            if bound > -np.inf:
                heapq.heappush(heap, (-bound, next(counter), (v,), lp))
        while heap:
            _, _, prefix, exact = heapq.heappop(heap)
            t = len(prefix)
            if t == n:
                yield (list(prefix), exact + lp_len)
                continue
            for w, lp_step in self._trans.get(prefix[-1], ()):
                exact2 = exact + lp_step
                bound2 = lp_len + exact2 + (n - t - 1) * self._max_step
                if bound2 > -np.inf:
                    heapq.heappush(heap, (-bound2, next(counter), prefix + (w,), exact2))

    def __next__(self) -> tuple[list[Any], float]:
        return next(self._merge)


class MarkovChainSampler(DistributionSampler):
    """Sampler for Markov-chain state sequences."""

    def __init__(self, dist: "MarkovChainDistribution", seed: int | None = None) -> None:
        """Create a sampler for a Markov-chain distribution.

        Args:
            dist (MarkovChainDistribution): Distribution to sample from.
            seed (Optional[int]): Set seed of random number generator for sampling from Markov chain.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            init_prob (Tuple[List[T], List[float]): Tuple of initial state-values and probabilities.
            trans_prob (Dict[T, Tuple[List[T], List[float]]]): Dictionary mapping transition probabilities from state i
                to state j.
            len_sampler (DistributionSampler): Sample length of Markov chain sequence. Must be a DistributionSampler
                with support on non-negative integers.

        """
        self.rng = RandomState(seed)

        loc_trans = list(dist.init_prob_map.items())
        loc_probs = [v[1] for v in loc_trans]
        loc_keys = [v[0] for v in loc_trans]

        self.init_prob = (loc_keys, loc_probs)

        self.trans_prob = dict()
        for k, v in dist.transition_map.items():
            loc_trans = list(v.items())
            loc_probs = [v[1] for v in loc_trans]
            loc_keys = [v[0] for v in loc_trans]
            self.trans_prob[k] = (loc_keys, loc_probs)

        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

        # --- batched-sampling tables (built lazily) ---
        self._batch_tables = None

    def _build_batch_tables(self):
        """Precompute index-space tables for vectorized state-path sampling.

        Returns ``(states, init_idx, init_p, trans_cdf, has_row)`` where ``states`` is the ordered
        state list, ``init_idx``/``init_p`` give the initial-state categorical over those indices,
        ``trans_cdf`` is an ``(S, S)`` row-cumsum matrix (row ``i`` = transitions out of state ``i``,
        zero rows for states absent from ``trans_prob``), and ``has_row`` flags which states have an
        outgoing distribution. States absent from ``trans_prob`` are absorbing/terminal: the legacy
        loop breaks the chain there, so batched sampling leaves the remainder unfilled too.
        """
        if self._batch_tables is not None:
            return self._batch_tables

        # Ordered union of every state that can appear (initial keys + all transition keys/targets).
        states = list(self.init_prob[0])
        seen = set(states)
        for k, (keys, _probs) in self.trans_prob.items():
            if k not in seen:
                seen.add(k)
                states.append(k)
            for w in keys:
                if w not in seen:
                    seen.add(w)
                    states.append(w)
        state_to_idx = {s: i for i, s in enumerate(states)}
        n = len(states)

        init_idx = np.asarray([state_to_idx[s] for s in self.init_prob[0]], dtype=np.int64)
        init_p = np.asarray(self.init_prob[1], dtype=float)

        trans_cdf = np.zeros((n, n), dtype=float)
        has_row = np.zeros(n, dtype=bool)
        for k, (keys, probs) in self.trans_prob.items():
            row = np.zeros(n, dtype=float)
            for w, p in zip(keys, probs):
                row[state_to_idx[w]] += p
            trans_cdf[state_to_idx[k], :] = np.cumsum(row)
            has_row[state_to_idx[k]] = True

        self._batch_tables = (states, init_idx, init_p, trans_cdf, has_row)
        return self._batch_tables

    def _sample_state_paths(self, lengths: np.ndarray) -> list[list[Any]]:
        """Vectorized state-path sampling across a batch of chains.

        Loops over time (``T = max(lengths)``) drawing all live chains' next states at once via the
        transition CDF, instead of N x T scalar ``rng.choice`` calls. Chains that reach an absorbing
        state (one with no outgoing transition row) stop early, matching the legacy break.

        Note: this consumes the RNG in a different order than the per-draw legacy loop, so the output
        is statistically equivalent but NOT byte-identical to ``batched=False``.
        """
        states, init_idx, init_p, trans_cdf, has_row = self._build_batch_tables()
        lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        size = len(lengths)
        if size == 0:
            return []

        max_len = int(lengths.max()) if size else 0
        out: list[list[Any]] = [[None] * int(n) for n in lengths]
        if max_len == 0:
            return out

        init_cdf = np.cumsum(init_p)
        # cur = current integer state per chain; -1 marks a chain that has stopped (absorbed/done).
        cur = np.full(size, -1, dtype=np.int64)
        active = lengths >= 1
        if active.any():
            act_pos = np.flatnonzero(active)
            u = self.rng.random_sample(len(act_pos)) * init_cdf[-1]
            picks = init_idx[np.searchsorted(init_cdf, u, side="right")]
            cur[act_pos] = picks
            for k, pos in enumerate(act_pos):
                out[pos][0] = states[picks[k]]

        for t in range(1, max_len):
            # A chain advances at step t iff it still needs entries (lengths > t), has not stopped
            # (cur >= 0), and its current state has an outgoing row (has_row). The first two without
            # the third = sitting on an absorbing state: leave the remainder None, like the legacy break.
            needs = (lengths > t) & (cur >= 0)
            if not needs.any():
                break
            live = needs & has_row[np.where(cur >= 0, cur, 0)]
            live_pos = np.flatnonzero(live)
            if len(live_pos) == 0:
                continue
            rows = trans_cdf[cur[live_pos], :]
            u = self.rng.random_sample(len(live_pos)) * rows[:, -1]
            nxt = (rows < u[:, None]).sum(axis=1)
            cur[live_pos] = nxt
            for k, pos in enumerate(live_pos):
                out[pos][t] = states[nxt[k]]

        return out

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[Any] | list[list[Any]]:
        """Draw iid samples from Markov chain distribution.

        If size is None, sample N from len_sampler() and return a List[T] of length N, where T is the data type of
        the Markov chain. If size > 0, return a list of length size, containing List[T] data types.

        With ``batched=True`` (default) the state paths for the whole batch are drawn by looping over
        time and advancing all live chains at once through the transition matrix, instead of N x T
        scalar draws. This consumes the RNG in a different order than the legacy per-draw loop, so the
        draws are statistically equivalent but NOT byte-identical to ``batched=False``. Set
        ``batched=False`` to reproduce the exact legacy output for a given seed.

        Args:
            size (Optional[int]): Number of samples to draw. Draws 1 sample if None.
            batched (bool): Vectorize state-path draws across chains (default); set False for the
                legacy per-draw loop.

        Returns:
            List[T] or List[List[T]], depending on size arg.

        """
        if not batched:
            if size is not None:
                return [self.sample(batched=False) for i in range(size)]
            cnt = self.len_sampler.sample()
            rv = [None] * cnt
            if cnt >= 1:
                rv[0] = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
            for i in range(1, cnt):
                curr_k, curr_p = self.trans_prob[rv[i - 1]]
                rv[i] = self.rng.choice(curr_k, p=curr_p)
            return rv

        if size is None:
            cnt = int(self.len_sampler.sample())
            return self._sample_state_paths(np.asarray([cnt], dtype=np.int64))[0]

        lengths = np.asarray(self.len_sampler.sample(size=size), dtype=np.int64).reshape(-1)
        return self._sample_state_paths(lengths)

    def sample_paths(self, lengths: Sequence[int]) -> list[list[Any]]:
        """Vectorized batch of state paths, one per requested length.

        Loops over time and advances all live chains at once through the transition matrix. The RNG
        consumption order differs from per-sequence ``sample_seq`` calls, so paths are statistically
        equivalent but not byte-identical. Used by HiddenMarkovSampler for batched state-path draws.

        Args:
            lengths (Sequence[int]): Length of each chain to sample.

        Returns:
            List of state-sequences (List[T]), one per entry in ``lengths``.

        """
        return self._sample_state_paths(np.asarray(list(lengths), dtype=np.int64))

    def sample_seq(self, size: int | None = None, v0: T | None = None, *, batched: bool = False) -> T | list[T]:
        """Sample a Markov chain sequence of length 'size' conditioned on initial state 'v0'.

        If size is None, draw a sequence of length 1, returning as type T.

        If size is not None, draw a sequence of length size, returning as type List[T].

        If v0 is None, v0 is sampled from member variable 'init_prob'.

        This is the legacy per-step path (one ``rng.choice`` per transition); the ``batched`` flag is
        accepted for API symmetry but does not change behavior here. For vectorized batches of whole
        chains use :meth:`sample_paths` or ``sample(..., batched=True)``.

        Args:
            size (Optional[int]): Length of Markov chain sequence to sample.
            v0 (Optional[T]): Initial state of Markov chain sequence to sample from.
            batched (bool): Accepted for API symmetry; sample_seq is always the per-step path.

        Returns:
            T or List[T] depending on arg size.

        """
        if size is not None:
            rv = [None] * size

            prev_val = v0

            if size > 0 and prev_val is None:
                rv[0] = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
                prev_val = rv[0]

            for i in range(1, size):
                if prev_val not in self.trans_prob:
                    break

                levels, probs = self.trans_prob[prev_val]
                rv[i] = self.rng.choice(levels, p=probs)
                prev_val = rv[i]

            return rv

        else:
            prev_val = v0

            if prev_val is None:
                rv = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
            else:
                levels, probs = self.trans_prob[prev_val]
                rv = self.rng.choice(levels, p=probs)

            return rv


class MarkovChainAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for initial-state, transition, and optional length sufficient statistics."""

    def __init__(
        self,
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for Markov-chain sufficient statistics.

        Args:
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for length sufficient
                statistics.
            keys (Optional[str]): Set keys for merging sufficient statistics of MarkovChainAccumulator.

        Attributes:
            init_count_map (Dict[T, float]): Dictionary for accumulating weighted counts of initial states.
            trans_count_map (Dict[T, Dict[T, float]]): Dictionary for accumulating weighted counts of state to state
                transitions
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for length sufficient statistics.
                Set to NullAccumulator() if no length distribution is to be estimated.
            keys (Optional[str]): Keys for merging sufficient statistics of MarkovChainAccumulator.

        """
        self.init_count_map = dict()
        self.trans_count_map = dict()
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.keys = keys

    def update(self, x: list[T], weight: float, estimate: MarkovChainDistribution) -> None:
        """Update sufficient statistics of MarkovChainAccumulator with weighted observation.

        Aggregates suff stats by checking initial state of sequence, and counting all transitions. Passes length of
        sequence x to len_accumulator.

        Args:
            x (List[T]):
            weight (float): Weight for observation.
            estimate (Optional[MarkovChainDistribution]): Previous estimate for MarkovChainDistribution or None.

        Returns:
            None.

        """
        if x is not None:
            self.len_accumulator.update(len(x), weight, getattr(estimate, "len_dist", None))

        if x is not None and len(x) != 0:
            x0 = x[0]
            self.init_count_map[x0] = self.init_count_map.get(x0, zero) + weight

            for u in x[1:]:
                if x0 not in self.trans_count_map:
                    self.trans_count_map[x0] = dict()

                self.trans_count_map[x0][u] = self.trans_count_map[x0].get(u, zero) + weight
                x0 = u

    def initialize(self, x: list[T], weight: float, rng: RandomState) -> None:
        """Initialize MarkovChainAccumulator with Markov chain observation x and random number generator rng passed
            to len_accumulator.initialize().

        Args:
            x (List[T]): Single Markov chain observation.
            weight (float): Weight for observation.
            rng (RandomState): Random state passed to ``len_accumulator.initialize()``.

        Returns:
            None.

        """
        if x is not None:
            self.len_accumulator.initialize(len(x), weight, rng)

        if x is not None and len(x) != 0:
            x0 = x[0]
            self.init_count_map[x0] = self.init_count_map.get(x0, zero) + weight

            for u in x[1:]:
                if x0 not in self.trans_count_map:
                    self.trans_count_map[x0] = dict()

                self.trans_count_map[x0][u] = self.trans_count_map[x0].get(u, zero) + weight
                x0 = u

    def seq_initialize(self, x: enc_data_type, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of MarkovChainAccumulator sufficient statistics from a sequence of encoded data x.

        Note that this is the same as seq_update() for the transition and initial state updates. For len_accumulator,
        a call to seq_initialize() must be made.

        The arg value x is a Tuple of length 8 with entries:
            x[0] (int): Number of total observations (number of Markov sequences).
            x[1] (ndarray[int]): Sequence index for initial state observations.
            x[2] (ndarray[int]): Sequence index for non-initial state observations in a sequence greater than len 1.
            x[3] (ndarray[int]): Numpy array of observations index in inv_key_map for initial states.
            x[4] (ndarray[int]): State-to-state index value of inv_key_map for initial state value.
            x[5] (ndarray[int]): State-to-state index value of inv_key_map for transition.
            x[6] (ndarray[T]): Maps integer index value to value in state-space (T).
            x[7] (Optional[T1]): Encoded sequence of lengths from len_encoder. None if no length distribution to be
                estimated.

        Args:
            x: See above for details.
            weights (ndarray[float]): Weights for observations in sequence encoded x.
            rng (RandomState): Random state passed to ``len_accumulator.initialize()``.

        Returns:
            None.

        """
        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        self.len_accumulator.seq_initialize(len_enc, weights, rng)

        key_sz = len(inv_key_map)

        init_count = np.bincount(init_x, weights=weights[idx0])

        for i in range(len(init_count)):
            v = init_count[i]
            if v != 0:
                self.init_count_map[inv_key_map[i]] = self.init_count_map.get(inv_key_map[i], 0.0) + v

        # Aggregate transition weights over (prev, next) pairs in one vectorized pass,
        # then scatter only the distinct nonzero pairs into the sparse count map.
        if len(prev_x) > 0:
            flat = np.asarray(prev_x) * key_sz + np.asarray(next_x)
            trans_count = np.bincount(flat, weights=weights[idx1], minlength=key_sz * key_sz)
            nz = np.nonzero(trans_count)[0]
            for f in nz:
                k1 = inv_key_map[f // key_sz]
                k2 = inv_key_map[f % key_sz]
                v = trans_count[f]

                if k1 not in self.trans_count_map:
                    self.trans_count_map[k1] = {k2: v}
                else:
                    m = self.trans_count_map[k1]
                    m[k2] = m.get(k2, 0.0) + v

    def seq_update(self, x: enc_data_type, weights: np.ndarray, estimate: MarkovChainDistribution) -> None:
        """Vectorized update of Markov chain sufficient statistics for a sequence encoded x.

        Computationally efficient update of MarkovChainAccumulator object using vectorized numpy operations.

        Note that estimate must be passed, as the 'estimate' argument of len_accumulator.seq_update() may require
        estimate parameter to not be None.

        The arg value x is a Tuple of length 8 with entries:
            x[0] (int): Number of total observations (number of Markov sequences).
            x[1] (ndarray[int]): Sequence index for initial state observations.
            x[2] (ndarray[int]): Sequence index for non-initial state observations in a sequence greater than len 1.
            x[3] (ndarray[int]): Numpy array of observations index in inv_key_map for initial states.
            x[4] (ndarray[int]): State-to-state index value of inv_key_map for initial state value.
            x[5] (ndarray[int]): State-to-state index value of inv_key_map for transition.
            x[6] (ndarray[T]): Maps integer index value to value in state-space (T).
            x[7] (Optional[T1]): Encoded sequence of lengths from len_encoder. None if no length distribution to be
                estimated.

        Args:
            x: See above for details.
            weights (ndarray[float]): Weights for observations in sequence encoded x.
            estimate (MarkovChainDistribution): Previous estimate of MarkovChainDistribution.

        Returns:
            None.

        """
        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x

        key_sz = len(inv_key_map)

        init_count = np.bincount(init_x, weights=weights[idx0])

        for i in range(len(init_count)):
            v = init_count[i]
            if v != 0:
                self.init_count_map[inv_key_map[i]] = self.init_count_map.get(inv_key_map[i], 0.0) + v

        # Aggregate transition weights over (prev, next) pairs in one vectorized pass,
        # then scatter only the distinct nonzero pairs into the sparse count map.
        if len(prev_x) > 0:
            flat = np.asarray(prev_x) * key_sz + np.asarray(next_x)
            trans_count = np.bincount(flat, weights=weights[idx1], minlength=key_sz * key_sz)
            nz = np.nonzero(trans_count)[0]
            for f in nz:
                k1 = inv_key_map[f // key_sz]
                k2 = inv_key_map[f % key_sz]
                v = trans_count[f]

                if k1 not in self.trans_count_map:
                    self.trans_count_map[k1] = {k2: v}
                else:
                    m = self.trans_count_map[k1]
                    m[k2] = m.get(k2, 0.0) + v

        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def seq_update_engine(self, x: enc_data_type, weights: Any, estimate: MarkovChainDistribution, engine: Any) -> None:
        """Engine-resident E-step: initial-state counts are reduced on the active engine and the
        transition weights are gathered on the engine before filling the sparse count maps; the
        length accumulator is routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        key_sz = len(inv_key_map)
        w_eng = engine.asarray(weights)

        init_count = np.asarray(
            engine.to_numpy(
                engine.bincount(
                    engine.asarray(np.asarray(init_x, dtype=np.int64)),
                    weights=w_eng[np.asarray(idx0, dtype=np.int64)],
                    minlength=key_sz,
                )
            ),
            dtype=np.float64,
        )
        for i in range(len(init_count)):
            v = init_count[i]
            if v != 0:
                self.init_count_map[inv_key_map[i]] = self.init_count_map.get(inv_key_map[i], 0.0) + v

        w_trans = np.asarray(engine.to_numpy(w_eng[np.asarray(idx1, dtype=np.int64)]), dtype=np.float64)
        for i in range(len(prev_x)):
            k1 = inv_key_map[prev_x[i]]
            k2 = inv_key_map[next_x[i]]
            ww = w_trans[i]
            if k1 not in self.trans_count_map:
                self.trans_count_map[k1] = {k2: ww}
            else:
                m = self.trans_count_map[k1]
                m[k2] = m.get(k2, 0.0) + ww

        child_seq_update(
            self.len_accumulator, len_enc, w_eng, estimate.len_dist if estimate is not None else None, engine
        )

    def combine(self, suff_stat: suff_stat_type) -> "MarkovChainAccumulator":
        """Merge the sufficient statistics of arg suff_stat with MarkovChainAccumulator.

        Arg suff_stat is a Tuple of length three containing,
            suff_stat[0] (Dict[T, float]): Maps initial state values to their corresponding counts.
            suff_stat[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition counts.
            suff_stat[2] (T1): Sufficient statistic value of length accumulator. (Assumed type T1).

        Args:
            suff_stat: See above for details.

        Returns:
            MarkovChainAccumulator object.

        """
        for item in suff_stat[0].items():
            self.init_count_map[item[0]] = self.init_count_map.get(item[0], 0.0) + item[1]

        for item in suff_stat[1].items():
            if item[0] not in self.trans_count_map:
                self.trans_count_map[item[0]] = dict()

            item_map = self.trans_count_map[item[0]]
            for elem in item[1].items():
                item_map[elem[0]] = item_map.get(elem[0], 0.0) + elem[1]

        self.len_accumulator = self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> suff_stat_type:
        """Return initial-state, transition, and length sufficient statistics."""
        return self.init_count_map, self.trans_count_map, self.len_accumulator.value()

    def from_value(self, x: suff_stat_type) -> "MarkovChainAccumulator":
        """Assign MarkovChainAccumulator sufficient statistics to value of x.

        Arg x is a Tuple of length three containing,
            x[0] (Dict[T, float]): Maps initial state values to their corresponding counts.
            x[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition counts.
            x[2] (T1): Sufficient statistic value of length accumulator. (Assumed type T1).

        Args:
            x: See above for details.

        Returns:
            MarkovChainAccumulator object.

        """
        self.init_count_map = x[0]
        self.trans_count_map = x[1]
        self.len_accumulator = self.len_accumulator.from_value(x[2])

        return self

    def scale(self, c: float) -> "MarkovChainAccumulator":
        """Scale initial, transition, and length sufficient statistics by a constant."""
        for key in list(self.init_count_map.keys()):
            self.init_count_map[key] *= c
        for tmap in self.trans_count_map.values():
            for key in list(tmap.keys()):
                tmap[key] *= c
        self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, "MarkovChainAccumulator"]) -> None:
        """Aggregate the sufficient statistics of MarkovChainAccumulator with member instance key in
            stats_dict.

        Args:
            stats_dict (Dict[str, MarkovChainAccumulator]): Key of dict are the 'keys' for
                MarkovChainAccumulator that represent the same distribution.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())

            else:
                stats_dict[self.keys] = self

        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, "MarkovChainAccumulator"]) -> None:
        """Set MarkovChainAccumulator sufficient statistic member variables to the value of stats_dict with
            matching keys.

        When this accumulator's key exists in ``stats_dict``, replace its sufficient statistics with the
        statistics stored under the matching key.

        Args:
            stats_dict (Dict[str, MarkovChainAccumulator]): Maps member variable key to MarkovChainAccumulator with
                same key.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "MarkovChainDataEncoder":
        """Create a data encoder from this accumulator's length encoder.

        Note: len_encoder is passed as NullDataEncoder() if len_dist is not to be estimated.

        Returns:
            MarkovChainDataEncoder: Encoder using this accumulator's length encoder.

        """
        len_encoder = self.len_accumulator.acc_to_encoder()
        return MarkovChainDataEncoder(len_encoder=len_encoder)


class MarkovChainAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for Markov-chain sufficient-statistic accumulators."""

    def __init__(
        self, len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(), keys: str | None = None
    ) -> None:
        """Create a factory for Markov-chain accumulators.

        Args:
            len_factory (StatisticAccumulatorFactory): Factory for the Markov-chain sequence-length accumulator.
            keys (Optional[str]): Optional key for merging Markov-chain sufficient statistics.

        Attributes:
            len_factory (StatisticAccumulatorFactory): Factory for the sequence-length accumulator.
            keys (Optional[str]): Optional key for merging Markov-chain sufficient statistics.
        """
        self.len_factory = len_factory
        self.keys = keys

    def make(self) -> "MarkovChainAccumulator":
        """Return a new Markov-chain accumulator."""
        len_acc = self.len_factory.make()
        return MarkovChainAccumulator(len_accumulator=len_acc, keys=self.keys)


class MarkovChainEstimator(ParameterEstimator):
    """Estimator for finite-state Markov-chain transition maps and optional length law."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        levels: Iterable[T] | None = None,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        name: str | None = None,
        keys: str | None = None,
        prior=None,
    ) -> None:
        """Create an estimator for a Markov-chain distribution from aggregated data.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics when merged with aggregated data.
            levels (Optional[Iterable[T]]): Set of state values.
            len_estimator (Optional[ParameterEstimator]): ParameterEstimator for length of Markov sequences.
            name (Optional[str]): Set a name for instance of MarkovChainEstimator.
            keys (Optional[str]): Set keys for merging sufficient statistics of MarkovChainAccumulator objects.
            prior: Optional ``(states, init_prior, row_priors)`` conjugate Dirichlet prior. When the
                priors are all Dirichlet this enables the clamped Dirichlet MAP update (carrying the
                posterior Dirichlets forward). ``None`` (default) preserves the existing MLE /
                pseudo-count path byte-identically.

        Attributes:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics when merged with aggregated data.
            levels (Optional[Iterable[T]]): State state values previously encountered.
            len_estimator (ParameterEstimator): NullEstimator if no length distribution is to be estimated.
            name (Optional[str]): Name for instance of MarkovChainEstimator.
            keys (Optional[str]): Keys for merging sufficient statistics of MarkovChainAccumulator objects.
        """
        self.name = name
        self.pseudo_count = pseudo_count
        self.levels = levels
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.keys = keys
        self.set_prior(prior)

    def accumulator_factory(self) -> "MarkovChainAccumulatorFactory":
        """Returns MarkovChainAccumulatorFactory for creating MarkovChainAccumulator."""
        return MarkovChainAccumulatorFactory(len_factory=self.len_estimator.accumulator_factory(), keys=self.keys)

    def get_prior(self):
        """Returns the conjugate prior in ``(states, init_prior, row_priors)`` form (or None)."""
        if not self.has_conj_prior:
            return None
        return (list(self.prior_states), self.init_prior, list(self.row_priors))

    def set_prior(self, prior) -> None:
        """Set the conjugate Dirichlet prior and flag whether it admits the conjugate update.

        Args:
            prior: ``(states, init_prior, row_priors)`` tuple or None; has_conj_prior is set when
                all priors are Dirichlet.

        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution

        if prior is None:
            self.prior = None
            self.prior_states = None
            self.init_prior = None
            self.row_priors = None
            self.has_conj_prior = False
            return

        states, init_prior, row_priors = _unpack_markov_chain_prior(prior)
        self.prior = prior
        self.prior_states = states
        self.init_prior = init_prior
        self.row_priors = row_priors
        self.has_conj_prior = isinstance(init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in row_priors
        )

    def model_log_density(self, model: "MarkovChainDistribution") -> float:
        """Log-density of the model's probabilities under the Dirichlet priors.

        Sums the Dirichlet log-densities of the initial-state probabilities and each transition row
        (floored at a tiny constant so MAP estimates that sit on the simplex boundary score
        finitely). Returns 0.0 without a conjugate prior.

        Args:
            model (MarkovChainDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters.

        """
        if not self.has_conj_prior:
            return 0.0
        tiny = 1.0e-300
        states = self.prior_states
        w = np.asarray([model.init_prob_map.get(s, 0.0) for s in states], dtype=float)
        rv = float(self.init_prior.log_density(np.maximum(w, tiny)))
        for i, s in enumerate(states):
            row = model.transition_map.get(s, {})
            tvec = np.asarray([row.get(s2, 0.0) for s2 in states], dtype=float)
            rv += float(self.row_priors[i].log_density(np.maximum(tvec, tiny)))
        return rv

    def estimate(self, nobs: float | None, suff_stat: suff_stat_type) -> "MarkovChainDistribution":
        """Estimate MarkovChainDistribution from aggregated sufficient statistics from observed data.

        Arg suff_stat is a Tuple of length three containing,
            suff_stat[0] (Dict[T, float]): Maps initial state values to their aggregated counts.
            suff_stat[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition counts.
            suff_stat[2] (T1): Sufficient statistic value of length accumulator. (Assumed type T1).

        If member variable pseudo_count is set estimate1() is called to aggregated weighted sufficient statistics. Else
        estimate0() is called to obtain estimates for MarkovChainDistribution directly from arg 'suff_stat'.

        Args:
            nobs (Optional[float]): Number of observations. Passed to estimate1() or estimate2().
            suff_stat: Seed above for details.

        Returns:
            MarkovChainDistribution object.

        """
        if self.has_conj_prior:
            return self._estimate_conjugate(nobs, suff_stat)
        elif self.pseudo_count is not None:
            return self.estimate1(nobs, suff_stat)
        else:
            return self.estimate0(nobs, suff_stat)

    def _estimate_conjugate(self, nobs: float | None, suff_stat: suff_stat_type) -> "MarkovChainDistribution":
        """Clamped Dirichlet MAP estimate over the fixed prior ``states`` ordering.

        The initial-state and per-row transition probabilities are the clamped Dirichlet MAP
        (counts + alpha - 1, floored at zero and renormalized; posterior mean when degenerate) and
        the posterior Dirichlets (counts + alpha) are carried forward as the new prior. Mirrors
        mixle.bstats.markov_chain.MarkovChainEstimator.estimate exactly, mapping the stats dict-based
        sufficient statistics onto the fixed state ordering.
        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution

        init_count_map, trans_count_map, len_val = suff_stat
        states = self.prior_states
        s = len(states)
        idx = {st: i for i, st in enumerate(states)}

        init_counts = np.zeros(s, dtype=float)
        for k, v in init_count_map.items():
            if k in idx:
                init_counts[idx[k]] += v

        trans_counts = np.zeros((s, s), dtype=float)
        for k1, row in trans_count_map.items():
            if k1 not in idx:
                continue
            i = idx[k1]
            for k2, v in row.items():
                if k2 in idx:
                    trans_counts[i, idx[k2]] += v

        len_dist = self.len_estimator.estimate(nobs, len_val)

        a0 = np.asarray(self.init_prior.get_parameters(), dtype=float)
        init_probs = _map_probs(init_counts, a0)
        init_posterior = DirichletDistribution(init_counts + a0)

        trans_mat = np.zeros((s, s), dtype=float)
        row_posteriors = []
        for i in range(s):
            ai = np.asarray(self.row_priors[i].get_parameters(), dtype=float)
            trans_mat[i, :] = _map_probs(trans_counts[i, :], ai)
            row_posteriors.append(DirichletDistribution(trans_counts[i, :] + ai))

        init_prob_map = {states[i]: float(init_probs[i]) for i in range(s)}
        transition_map = {states[i]: {states[j]: float(trans_mat[i, j]) for j in range(s)} for i in range(s)}

        return MarkovChainDistribution(
            init_prob_map,
            transition_map,
            len_dist=len_dist,
            name=self.name,
            prior=(states, init_posterior, row_posteriors),
        )

    def estimate0(self, nobs: float | None, suff_stat: suff_stat_type) -> "MarkovChainDistribution":
        """Estimate MarkovChainDistribution from aggregated sufficient statistics from observed data.

        Maximum likelihood estimates for initial state probabilities, transition probabilities, and the length
        distribution are obtained directly from aggregated data in 'suff_stat'.

        Arg suff_stat is a Tuple of length three containing,
            suff_stat[0] (Dict[T, float]): Maps initial state values to their aggregated counts.
            suff_stat[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition counts.
            suff_stat[2] (T1): Sufficient statistic value of length accumulator. (Assumed type T1).

        Args:
            nobs (Optional[float]): Number of observations. Passed to estimate1() or estimate2().
            suff_stat: Seed above for details.

        Returns:
            MarkovChainDistribution object.

        """
        temp_sum = sum(suff_stat[0].values())
        init_prob_map = {k: v / temp_sum for k, v in suff_stat[0].items()}

        trans_map = dict()

        for key, tmap in suff_stat[1].items():
            temp_sum = sum(tmap.values())
            if temp_sum > 0:
                trans_map[key] = {k: v / temp_sum for k, v in tmap.items()}

        len_dist = self.len_estimator.estimate(nobs, suff_stat[2])

        return MarkovChainDistribution(init_prob_map, trans_map, len_dist=len_dist, name=self.name)

    def estimate1(self, nobs: float | None, suff_stat: suff_stat_type) -> "MarkovChainDistribution":
        """Estimate MarkovChainDistribution from aggregated sufficient statistics from observed data.

        Maximum likelihood estimates for initial state probabilities, transition probabilities, and the length
        distribution are obtained by a weighted aggregation of sufficient statistics in 'suff_stat', and member
        variables of MarkovChainEstimator object.

        Arg suff_stat is a Tuple of length three containing,
            suff_stat[0] (Dict[T, float]): Maps initial state values to their aggregated counts.
            suff_stat[1] (Dict[T, Dict[T, List[float]]]): Maps state to state transition counts.
            suff_stat[2] (T1): Sufficient statistic value of length accumulator. (Assumed type T1).

        Args:
            nobs (Optional[float]): Number of observations. Passed to estimate1() or estimate2().
            suff_stat: Seed above for details.

        Returns:
            MarkovChainDistribution object.

        """
        trans_map = dict()
        init_prob_map = dict()
        def_val = 0.0

        all_keys = set(suff_stat[0].keys())
        for u in suff_stat[1].values():
            all_keys.update(u.keys())
        if self.levels is not None:
            all_keys.update(self.levels)

        temp_sum = sum(suff_stat[0].values())
        p_cnt0 = self.pseudo_count if self.pseudo_count is not None else 0.0
        p_cnt1 = p_cnt0 / len(all_keys)

        if (temp_sum + p_cnt0) > 0:
            init_prob_map = {k: (suff_stat[0].get(k, 0.0) + p_cnt1) / (temp_sum + p_cnt0) for k in all_keys}

        a_sum = temp_sum
        for key, tmap in suff_stat[1].items():
            temp_sum = sum(tmap.values())
            a_sum += temp_sum
            if (temp_sum + p_cnt0) > 0:
                trans_map[key] = {k: (tmap.get(k, 0.0) + p_cnt1) / (temp_sum + p_cnt0) for k in all_keys}

        len_dist = self.len_estimator.estimate(nobs, suff_stat[2])

        if a_sum > 0:
            def_val = self.pseudo_count / a_sum

        return MarkovChainDistribution(
            init_prob_map, trans_map, len_dist=len_dist, default_value=def_val, name=self.name
        )


class MarkovChainDataEncoder(DataSequenceEncoder):
    """Encoder for Markov-chain state sequences and optional sequence lengths."""

    def __init__(self, len_encoder: DataSequenceEncoder = NullDataEncoder()) -> None:
        """Create an encoder for Markov-chain sequences and optional length observations.

        Args:
            len_encoder (DataSequenceEncoder): Encoder for non-negative integer sequence lengths.

        Attributes:
              len_encoder (DataSequenceEncoder): DataSequenceEncoder object that has support on non-negative integers.
                Is set to NullDataEncoder() if no length distribution is to be estimated.
        """
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "MarkovChainDataEncoder(len_encoder=" + str(self.len_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder is equivalent to this encoder.

        Note: Does not currently check for type consistency in state-values.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is MarkovChainDataEncoder with equivalent len_encoder variable.

        """
        if isinstance(other, MarkovChainDataEncoder):
            return other.len_encoder == self.len_encoder
        else:
            return False

    def seq_encode(self, x: list[list[T]]) -> enc_data_type:
        """Sequence encoding a sequence of iid Markov chain observations with data type T.

        The returned value is (rv) is a Tuple of length 8 with entries:

            rv[0] (int): Number of total observations (number of Markov sequences).
            rv[1] (ndarray[int]): Sequence index for initial state observations.
            rv[2] (ndarray[int]): Sequence index for non-initial state observations in a sequence greater than len 1.
            rv[3] (ndarray[int]): Numpy array of observations index in inv_key_map for initial states.
            rv[4] (ndarray[int]): State-to-state index value of inv_key_map for initial state value.
            rv[5] (ndarray[int]): State-to-state index value of inv_key_map for transition.
            rv[6] (ndarray[T]): Maps integer index value to value in state-space (T).
            rv[7] (Optional[T1]): Encoded sequence of lengths from len_encoder. None if no length distributon to be
                estimated.

        Args:
            x (List[List[T]]): Sequence of iid observations of Markov chain sequences.

        Returns:
            Tuple of length 8. See above for details.

        """

        init_entries = []
        pair_entries = []
        entries_idx0 = []
        entries_idx1 = []
        obs_cnt = []
        key_map = dict()

        for i in range(len(x)):
            entry = x[i]
            obs_cnt.append(len(entry))

            if len(entry) == 0:
                continue

            if entry[0] not in key_map:
                key_map[entry[0]] = len(key_map)

            prev_idx = key_map[entry[0]]
            init_entries.append(prev_idx)
            entries_idx0.append(i)

            for j in range(1, len(entry)):
                if entry[j] not in key_map:
                    key_map[entry[j]] = len(key_map)
                next_idx = key_map[entry[j]]

                pair_entries.append([prev_idx, next_idx])
                entries_idx1.append(i)
                prev_idx = next_idx

        obs_cnt = np.asarray(obs_cnt)
        init_entries = np.asarray(init_entries)
        pair_entries = np.asarray(pair_entries)
        entries_idx0 = np.asarray(entries_idx0)
        entries_idx1 = np.asarray(entries_idx1)

        inv_key_map = [None] * len(key_map)
        for k, v in key_map.items():
            inv_key_map[v] = k
        inv_key_map = np.asarray(inv_key_map)

        len_enc = self.len_encoder.seq_encode(obs_cnt)

        return (
            len(x),
            entries_idx0,
            entries_idx1,
            init_entries,
            pair_entries[:, 0],
            pair_entries[:, 1],
            inv_key_map,
            len_enc,
        )
