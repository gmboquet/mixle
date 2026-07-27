"""First-order Markov-chain distributions over finite or structured states.

The assumed data type for the stats-space is T.

The density of Markov chain is given by for sequence of length n, x=[x[0],x[1],...,x[n-1]]

    p_mat(x) = p_mat(x[0])*p_mat(x[1]|x[0])*...*p_mat(x[n-1]|x[n-2])*P_len(n)

where p_mat(x[i+1]|x[i]) is the transition probability, p_mat(x[0]) is the init-probability, and P_len(n) is given
by the length distribution density.

Note if len(x) = 0, only log(P_len(0)) is returned.

"""

import copy
import heapq
import itertools
import math
from collections.abc import Iterable, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple, TypeVar

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
    ContractError,
    DataSequenceEncoder,
    DensitySemantics,
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
enc_data_type = tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]


class NonGenerativeMarkovChainError(TypeError):
    """Raised when a fixed-length Markov factor is asked to generate whole sequences."""


class MarkovChainStatistics(NamedTuple):
    """Immutable, versioned sufficient statistics over one closed state layout."""

    schema_version: int
    states: tuple[Any, ...]
    initial_counts: tuple[float, ...]
    transition_counts: tuple[tuple[float, ...], ...]
    length_nobs: float
    length: Any | None


def _canonical_states(values: Iterable[Any], *, label: str) -> tuple[Any, ...]:
    """Return unique hashable states in a stable heterogeneous-label order."""
    try:
        values = tuple(values)
    except TypeError as exc:
        raise TypeError("%s must be an iterable of hashable states." % label) from exc
    seen = set()
    for value in values:
        try:
            duplicate = value in seen
            seen.add(value)
        except TypeError as exc:
            raise TypeError("%s contains an unhashable state %r." % (label, value)) from exc
        if duplicate:
            raise ValueError("%s contains duplicate state %r." % (label, value))
    return tuple(
        sorted(
            values,
            key=lambda value: (
                type(value).__module__,
                type(value).__qualname__,
                repr(value),
            ),
        )
    )


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a finite non-negative scalar." % label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite non-negative scalar." % label) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be a finite non-negative scalar." % label)
    return result


def _checked_length(value: Any, *, label: str) -> int:
    """Return one exact finite non-negative integer length."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise TypeError("%s must be an exact finite non-negative integer." % label)
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise TypeError("%s must be a scalar exact finite non-negative integer." % label)
    try:
        numeric = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be an exact finite non-negative integer." % label) from exc
    if not math.isfinite(numeric) or numeric < 0.0 or math.floor(numeric) != numeric:
        raise ValueError("%s must be an exact finite non-negative integer." % label)
    if numeric > np.iinfo(np.intp).max:
        raise ValueError("%s exceeds the platform index range." % label)
    return int(numeric)


def _validated_probability_map(value: Any, *, label: str) -> dict[Any, float]:
    if not isinstance(value, dict):
        raise TypeError("%s must be a dict." % label)
    result = {}
    for key, probability in value.items():
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("%s contains an unhashable state." % label) from exc
        result[key] = _finite_nonnegative(probability, label="%s[%r]" % (label, key))
    return result


def _require_simplex(probabilities: Iterable[float], *, label: str) -> None:
    total = math.fsum(float(value) for value in probabilities)
    if not math.isclose(total, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("%s must sum to 1; got %.17g." % (label, total))


def _validate_markov_prior(
    prior: Any,
    *,
    expected_states: tuple[Any, ...] | None = None,
) -> tuple[tuple[Any, ...], Any, tuple[Any, ...]]:
    """Validate one exact finite-state Dirichlet prior layout."""
    from mixle.stats.bayes.dirichlet import DirichletDistribution

    if not isinstance(prior, (tuple, list)) or len(prior) != 3:
        raise TypeError("Markov prior must be (states, init_prior, row_priors).")
    raw_states = tuple(prior[0])
    states = _canonical_states(raw_states, label="Markov prior states")
    if not states:
        raise ValueError("Markov prior states cannot be empty.")
    if raw_states != states:
        raise ValueError("Markov prior states must use the model's canonical state order.")
    if expected_states is not None and states != expected_states:
        raise ValueError("Markov prior states must exactly match the model state layout.")
    init_prior = prior[1]
    row_priors = tuple(prior[2])
    if not isinstance(init_prior, DirichletDistribution):
        raise TypeError("Markov initial prior must be a DirichletDistribution.")
    if len(row_priors) != len(states) or any(
        not isinstance(row, DirichletDistribution) for row in row_priors
    ):
        raise ValueError("Markov prior must contain one Dirichlet transition prior per state.")
    for label, distribution in (
        ("initial", init_prior),
        *((("transition[%r]" % state), row) for state, row in zip(states, row_priors)),
    ):
        alpha = np.asarray(distribution.get_parameters(), dtype=float)
        if alpha.shape != (len(states),) or np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
            raise ValueError("%s Markov prior parameters must be a positive finite state-aligned vector." % label)
    return states, init_prior, row_priors


def _validate_markov_statistics(
    value: Any,
    *,
    expected_states: tuple[Any, ...] | None = None,
    path: str,
) -> MarkovChainStatistics:
    """Validate and normalize one immutable fixed-layout statistic."""
    if not isinstance(value, MarkovChainStatistics) or value.schema_version != 1:
        raise ContractError(
            path,
            "MarkovChainStatistics schema version 1",
            type(value).__name__,
            "pass the value produced by MarkovChainAccumulator.value().",
        )
    raw_states = tuple(value.states)
    states = _canonical_states(raw_states, label="%s.states" % path)
    if not states:
        raise ValueError("%s states cannot be empty." % path)
    if raw_states != states:
        raise ContractError(
            "%s.states" % path,
            "canonical state order %r" % (states,),
            "%r" % (raw_states,),
        )
    if expected_states is not None and states != expected_states:
        raise ContractError(
            "%s.states" % path,
            "the configured state layout %r" % (expected_states,),
            "%r" % (states,),
        )
    initial = tuple(
        _finite_nonnegative(count, label="%s.initial_counts[%d]" % (path, index))
        for index, count in enumerate(value.initial_counts)
    )
    if len(initial) != len(states):
        raise ValueError("%s initial count vector must align with states." % path)
    if len(value.transition_counts) != len(states):
        raise ValueError("%s transition count matrix must have one row per state." % path)
    transitions = tuple(
        tuple(
            _finite_nonnegative(count, label="%s.transition_counts[%d][%d]" % (path, row, column))
            for column, count in enumerate(counts)
        )
        for row, counts in enumerate(value.transition_counts)
    )
    if any(len(row) != len(states) for row in transitions):
        raise ValueError("%s transition count matrix must be square and state-aligned." % path)
    return MarkovChainStatistics(
        1,
        states,
        initial,
        transitions,
        _finite_nonnegative(value.length_nobs, label="%s.length_nobs" % path),
        value.length,
    )


def _validate_effective_nobs(nobs: float | None, statistics: MarkovChainStatistics) -> None:
    if nobs is None:
        return
    checked = _finite_nonnegative(nobs, label="Markov estimate nobs")
    if not math.isclose(
        checked,
        statistics.length_nobs,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Markov estimate nobs must equal the statistic's effective row count.")


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
    states = list(_canonical_states(states, label="Markov prior states"))
    s = len(states)
    if s == 0:
        raise ValueError("Markov prior states cannot be empty.")
    return (
        states,
        dirichlet(np.ones(s)),
        [dirichlet(np.ones(s)) for _ in range(s)],
    )


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
        raw_initial = _validated_probability_map(init_prob_map, label="init_prob_map")
        if not isinstance(transition_map, dict):
            raise TypeError("transition_map must be a dict of probability-row dicts.")
        raw_transitions = {
            state: _validated_probability_map(row, label="transition_map[%r]" % (state,))
            for state, row in transition_map.items()
        }
        if isinstance(default_value, (bool, np.bool_)):
            raise TypeError("default_value must be zero for a closed finite-state Markov chain.")
        try:
            checked_default = float(default_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("default_value must be zero for a closed finite-state Markov chain.") from exc
        if not math.isfinite(checked_default) or checked_default != 0.0:
            raise ValueError(
                "default_value must be 0; unknown labels require an explicit state and probability row."
            )

        states = _canonical_states(
            set(raw_initial)
            .union(raw_transitions)
            .union(value for row in raw_transitions.values() for value in row),
            label="Markov state space",
        )
        if not states:
            raise ValueError("Markov state space cannot be empty.")
        _require_simplex(raw_initial.values(), label="init_prob_map")
        missing_rows = tuple(state for state in states if state not in raw_transitions)
        if missing_rows:
            raise ValueError("transition_map is missing rows for states %r." % (missing_rows,))

        initial = {state: raw_initial.get(state, 0.0) for state in states}
        transitions = {}
        for state in states:
            row = raw_transitions[state]
            _require_simplex(row.values(), label="transition_map[%r]" % (state,))
            transitions[state] = {next_state: row.get(next_state, 0.0) for next_state in states}

        self.init_prob_map = MappingProxyType(initial)
        self.transition_map = MappingProxyType(
            {state: MappingProxyType(row) for state, row in transitions.items()}
        )
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        if not supports(self.len_dist, Neutral):
            from mixle.stats.combinator.sequence import _validate_length_distribution

            _validate_length_distribution(self.len_dist)

        self.all_vals = frozenset(states)
        self.loginit_prob_map = MappingProxyType(
            {state: -np.inf if probability == 0.0 else float(np.log(probability)) for state, probability in initial.items()}
        )
        self.log_transition_map = MappingProxyType(
            {
                state: MappingProxyType(
                    {
                        next_state: -np.inf if probability == 0.0 else float(np.log(probability))
                        for next_state, probability in row.items()
                    }
                )
                for state, row in transitions.items()
            }
        )

        self.default_value = 0.0
        self.log_dv = -np.inf
        self.log_dtv = -np.inf
        self.log1p_dv = 0.0

        num_keys = len(states)
        keys = list(states)

        self.key_map = {keys[i]: i + 1 for i in range(num_keys)}
        self.inv_key_map = keys
        self.num_keys = num_keys

        self.init_log_pvec = np.full(num_keys + 1, -np.inf, dtype=float)
        dense_transitions = np.full((num_keys + 1, num_keys + 1), -np.inf, dtype=float)

        for k1, v1 in initial.items():
            self.init_log_pvec[self.key_map[k1]] = -np.inf if v1 == 0.0 else np.log(v1)

        for k1, v1 in transitions.items():
            k1_idx = self.key_map[k1]
            for k2, v2 in v1.items():
                dense_transitions[k1_idx, self.key_map[k2]] = -np.inf if v2 == 0 else np.log(v2)
        self.trans_log_pvec = dok_matrix(dense_transitions)

        self.set_prior(prior)

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Return the validated, canonical closed-chain tables in serializable form."""
        return {
            "init_prob_map": list(self.init_prob_map.items()),
            "transition_map": [(key, list(row.items())) for key, row in self.transition_map.items()],
            "len_dist": self.len_dist,
            "default_value": self.default_value,
            "name": self.name,
            "prior": self.get_prior(),
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Restore state through the constructor so every invariant is revalidated."""
        state = dict(state)
        init_prob_map = state["init_prob_map"]
        transition_map = state["transition_map"]
        self.__init__(
            init_prob_map if isinstance(init_prob_map, dict) else dict(init_prob_map),
            (
                transition_map
                if isinstance(transition_map, dict)
                else {key: (row if isinstance(row, dict) else dict(row)) for key, row in transition_map}
            ),
            len_dist=state.get("len_dist"),
            default_value=state.get("default_value", 0.0),
            name=state.get("name"),
            prior=state.get("prior"),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "MarkovChainDistribution":
        """Copy immutable table views by reconstructing their owned backing dictionaries."""
        payload = copy.deepcopy(
            (
                dict(self.init_prob_map),
                {state: dict(row) for state, row in self.transition_map.items()},
                self.len_dist,
                self.name,
                self.get_prior(),
            ),
            memo,
        )
        copied = type(self)(
            payload[0],
            payload[1],
            len_dist=payload[2],
            default_value=0.0,
            name=payload[3],
            prior=payload[4],
        )
        memo[id(self)] = copied
        return copied

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
        if prior is None:
            self.prior = None
            self.prior_states = None
            self.init_prior = None
            self.row_priors = None
            self.e_log_init = None
            self.e_log_trans = None
            self.has_conj_prior = False
            return

        states, init_prior, row_priors = _validate_markov_prior(
            prior,
            expected_states=tuple(self.inv_key_map),
        )
        self.prior = (states, init_prior, row_priors)
        self.prior_states = states
        self.init_prior = init_prior
        self.row_priors = row_priors

        a0 = np.asarray(init_prior.get_parameters(), dtype=float)
        self.e_log_init = digamma(a0) - digamma(a0.sum())
        self.e_log_trans = np.zeros((len(states), len(states)))
        for i, row_prior in enumerate(row_priors):
            ai = np.asarray(row_prior.get_parameters(), dtype=float)
            self.e_log_trans[i, :] = digamma(ai) - digamma(ai.sum())
        self.has_conj_prior = True

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
            if x[0] not in idx:
                return -np.inf
            rv = float(self.e_log_init[idx[x[0]]])
            for i in range(1, len(x)):
                if x[i - 1] not in idx or x[i] not in idx:
                    return -np.inf
                rv += float(self.e_log_trans[idx[x[i - 1]], idx[x[i]]])
        rv += self.len_dist.expected_log_density(len(x))
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
        loc_key_map = np.asarray([idx.get(u, -1) for u in inv_key_map], dtype=np.int64)

        rv = np.zeros(sz, dtype=float)
        if len(idx1) > 0:
            prev = loc_key_map[prev_x]
            nxt = loc_key_map[next_x]
            valid = (prev >= 0) & (nxt >= 0)
            temp = np.full(len(idx1), -np.inf, dtype=float)
            temp[valid] = self.e_log_trans[prev[valid], nxt[valid]]
            rv = np.bincount(idx1, weights=temp, minlength=sz)
        initial = loc_key_map[init_x]
        initial_scores = np.full(len(idx0), -np.inf, dtype=float)
        valid = initial >= 0
        initial_scores[valid] = self.e_log_init[initial[valid]]
        rv[idx0] += initial_scores

        if len_enc is not None:
            rv += self.len_dist.seq_expected_log_density(len_enc)

        return rv

    def compute_capabilities(self):
        """Return compute-backend metadata inherited from the optional length distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.len_dist)
        return DistributionCapabilities(
            engine_ready=child.engine_ready,
            kernel_status="numpy_only" if child.numpy_only_reason else "generic",
            numpy_only_reason=child.numpy_only_reason,
        )

    def density_semantics(self):
        """Classify a chain without a length law as a conditional fixed-length factor."""
        if supports(self.len_dist, Neutral):
            return DensitySemantics.LIKELIHOOD_FACTOR
        return self.len_dist.density_semantics()

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
        order = lambda item: (type(item[0]).__module__, type(item[0]).__qualname__, repr(item[0]))
        s1 = repr(dict(sorted(self.init_prob_map.items(), key=order)))
        temp = sorted(self.transition_map.items(), key=order)
        s2 = repr(dict((k, dict(sorted(v.items(), key=order))) for k, v in temp))
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

        # np.bincount(weights=...) silently returns an int64 (not float64) array when both `idx1` and
        # `weights` are empty -- the case for length-1 sequences (no transitions, only an initial
        # state) -- which then breaks the float `+=` below with a same-kind casting error. Mirrors the
        # `rv = np.zeros(sz, dtype=float)` + `if len(idx1) > 0` guard already used above in
        # `model_log_density` for the identical reason.
        rv = np.zeros(sz, dtype=float)
        if len(idx1) > 0:
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
    ) -> tuple[MarkovChainStatistics, ...]:
        """Return versioned per-component fixed-layout Markov statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        ww = engine.asarray(weights)
        num_components = int(params["num_components"])
        labels = tuple(params["labels"])
        label_to_position = {label: index for index, label in enumerate(labels)}

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

        length_counts = engine.sum(ww, axis=0)
        result = []
        for component in range(num_components):
            dense_initial = np.zeros(len(labels), dtype=float)
            dense_transitions = np.zeros((len(labels), len(labels)), dtype=float)
            for value_index, label in enumerate(inv_key_map):
                position = label_to_position.get(label)
                if position is not None:
                    dense_initial[position] = init_counts[value_index, component]
            for pair_index, (prev_index, next_index) in enumerate(observed_pairs):
                prev_position = label_to_position.get(inv_key_map[prev_index])
                next_position = label_to_position.get(inv_key_map[next_index])
                if prev_position is not None and next_position is not None:
                    dense_transitions[prev_position, next_position] = trans_counts[pair_index, component]
            result.append(
                MarkovChainStatistics(
                    1,
                    labels,
                    tuple(float(value) for value in dense_initial),
                    tuple(tuple(float(value) for value in row) for row in dense_transitions),
                    length_counts[component],
                    length_by_component[component],
                )
            )
        return tuple(result)

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
        if supports(self.len_dist, Neutral) or self.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR:
            raise NonGenerativeMarkovChainError(
                "MarkovChainDistribution without a proper length law is a fixed-length likelihood factor; "
                "use path_sampler(seed) with sample_seq(length) or sample_paths(lengths)."
            )
        return MarkovChainSampler(self, seed)

    def path_sampler(self, seed: int | None = None) -> "MarkovChainSampler":
        """Return a sampler for caller-supplied path lengths."""
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
            pseudo_count=pseudo_count,
            levels=self.inv_key_map,
            len_estimator=len_est,
            name=self.name,
            prior=self.get_prior(),
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
        # Parallel max-plus (tropical) scalar recursion tracking the TRUE (uncapped) worst-case
        # reachable bucket per (length, state), run in lockstep with `alpha`. Every `sr.scale` call in
        # the real recursion below truncates its OWN output to max_fine_bucket at each step, so
        # `alpha[t][s].hist.max_bucket()` can never exceed max_fine_bucket by construction -- it cannot
        # reveal whether truncation actually happened. This shadow recursion applies the SAME
        # predecessor structure with no capping at all, so its values expose exactly that.
        true_top: list[dict[Any, int] | None] = [None, {s: quantizer.fine_bucket(init_lp[s]) for s in init_lp}]
        for t in range(2, max_len + 1):
            prev = alpha[t - 1]
            prev_top = true_top[t - 1]
            cur: dict[Any, Any] = {}
            cur_top: dict[Any, int] = {}
            for s_next in state_order:
                acc = sr.zero()
                built = False
                best: int | None = None
                for s_prev, lp_tr in into[s_next]:
                    ph = prev.get(s_prev)
                    if ph is not None and not ph.hist.is_empty():
                        step = sr.map_values(
                            sr.scale(ph, lp_tr, quantizer, max_fine_bucket), lambda seq, s=s_next: seq + [s]
                        )
                        acc = step if not built else sr.plus(acc, step)
                        built = True
                    pv = prev_top.get(s_prev)
                    if pv is not None:
                        cand = pv + quantizer.fine_bucket(lp_tr)
                        if best is None or cand > best:
                            best = cand
                if built and not acc.hist.is_empty():
                    cur[s_next] = acc
                if best is not None:
                    cur_top[s_next] = best
            alpha.append(cur)
            true_top.append(cur_top)
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
            # The TRUE (uncapped) worst-case bucket among states reachable at length L, plus the
            # length term's own bucket -- the exact pre-cap top of this length's contribution,
            # independent of whatever alpha's real (self-capping) histograms can reveal on their own.
            lt = true_top[L]
            if lt and max(lt.values()) + quantizer.fine_bucket(lp_len) > max_fine_bucket:
                truncated = True
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
            len_sampler (Optional[DistributionSampler]): Length sampler, or ``None`` when the distribution
                is an unnormalized fixed-length chain factor. A length distribution is required by
                :meth:`sample`; :meth:`sample_seq` and :meth:`sample_paths` remain available with
                caller-supplied lengths.

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

        self.len_sampler = (
            None
            if supports(dist.len_dist, Neutral)
            else dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        )

        # --- batched-sampling tables (built lazily) ---
        self._batch_tables = None

    def _build_batch_tables(self):
        """Precompute index-space tables for vectorized state-path sampling.

        Returns ``(states, init_idx, init_p, trans_cdf, has_row)`` where ``states`` is the ordered
        state list, ``init_idx``/``init_p`` give the initial-state categorical over those indices,
        ``trans_cdf`` is an ``(S, S)`` row-cumsum matrix (row ``i`` = transitions out of state ``i``,
        transition CDF. Construction has already proved one complete stochastic row per state.
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
        if np.any(lengths < 0):
            raise ValueError("Markov path lengths must be non-negative.")

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
            # A chain advances at step t iff it still needs entries and has a current state.
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
        if self.len_sampler is None:
            raise RuntimeError(
                "MarkovChainSampler.sample() requires a proper length distribution; "
                "use sample_seq(length) or sample_paths(lengths) for a fixed-length chain factor."
            )

        if not isinstance(batched, (bool, np.bool_)):
            raise TypeError("batched must be bool.")
        if size is not None:
            size = _checked_length(size, label="Markov sample size")
        if not batched:
            if size is not None:
                return [self.sample(batched=False) for _ in range(size)]
            cnt = _checked_length(self.len_sampler.sample(), label="sampled Markov length")
            rv = [None] * cnt
            if cnt >= 1:
                rv[0] = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
            for i in range(1, cnt):
                curr_k, curr_p = self.trans_prob[rv[i - 1]]
                rv[i] = self.rng.choice(curr_k, p=curr_p)
            return rv

        if size is None:
            cnt = _checked_length(self.len_sampler.sample(), label="sampled Markov length")
            return self._sample_state_paths(np.asarray([cnt], dtype=np.int64))[0]

        raw_lengths = np.asarray(self.len_sampler.sample(size=size), dtype=object).reshape(-1)
        if len(raw_lengths) != size:
            raise ValueError("length sampler returned the wrong batch size.")
        lengths = np.asarray(
            [_checked_length(value, label="sampled Markov length") for value in raw_lengths],
            dtype=np.int64,
        )
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
        if isinstance(lengths, (str, bytes)) or not isinstance(lengths, (Sequence, np.ndarray)):
            raise TypeError("lengths must be a sequence of exact non-negative integers.")
        checked = np.asarray(
            [_checked_length(value, label="Markov path length") for value in lengths],
            dtype=np.int64,
        )
        return self._sample_state_paths(checked)

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
        if not isinstance(batched, (bool, np.bool_)):
            raise TypeError("batched must be bool.")
        if v0 is not None and v0 not in self.trans_prob:
            raise ValueError("v0 must be a configured Markov state.")
        if size is not None:
            size = _checked_length(size, label="Markov path length")
            rv = [None] * size

            prev_val = v0

            if size > 0:
                if prev_val is None:
                    rv[0] = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
                else:
                    rv[0] = prev_val
                prev_val = rv[0]

            for i in range(1, size):
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
        levels: Iterable[T] | None = None,
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
        self.levels = None if levels is None else _canonical_states(levels, label="Markov accumulator levels")
        if self.levels == ():
            raise ValueError("Markov accumulator levels cannot be empty.")
        self.length_nobs = 0.0

    def _check_state(self, value: Any) -> None:
        if self.levels is not None and value not in self.levels:
            raise ValueError("state %r is outside the configured Markov state layout." % (value,))

    def _states(self) -> tuple[Any, ...]:
        if self.levels is not None:
            return self.levels
        values = set(self.init_count_map)
        values.update(self.trans_count_map)
        for row in self.trans_count_map.values():
            values.update(row)
        return _canonical_states(values, label="Markov accumulated states")

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
        if isinstance(x, (str, bytes)) or not isinstance(x, Sequence):
            raise TypeError("Markov observations must be state sequences.")
        checked_weight = _finite_nonnegative(weight, label="Markov observation weight")
        self.len_accumulator.update(len(x), checked_weight, getattr(estimate, "len_dist", None))
        self.length_nobs += checked_weight

        if len(x) != 0:
            for value in x:
                self._check_state(value)
            x0 = x[0]
            self.init_count_map[x0] = self.init_count_map.get(x0, zero) + checked_weight

            for u in x[1:]:
                if x0 not in self.trans_count_map:
                    self.trans_count_map[x0] = dict()

                self.trans_count_map[x0][u] = self.trans_count_map[x0].get(u, zero) + checked_weight
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
        if isinstance(x, (str, bytes)) or not isinstance(x, Sequence):
            raise TypeError("Markov observations must be state sequences.")
        checked_weight = _finite_nonnegative(weight, label="Markov observation weight")
        self.len_accumulator.initialize(len(x), checked_weight, rng)
        self.length_nobs += checked_weight

        if len(x) != 0:
            for value in x:
                self._check_state(value)
            x0 = x[0]
            self.init_count_map[x0] = self.init_count_map.get(x0, zero) + checked_weight

            for u in x[1:]:
                if x0 not in self.trans_count_map:
                    self.trans_count_map[x0] = dict()

                self.trans_count_map[x0][u] = self.trans_count_map[x0].get(u, zero) + checked_weight
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
        weights = np.asarray(weights)
        if (
            weights.shape != (sz,)
            or not np.issubdtype(weights.dtype, np.number)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise ValueError("Markov weights must be a finite non-negative vector aligned with encoded rows.")
        weights = weights.astype(float, copy=False)
        for value in inv_key_map:
            self._check_state(value)
        self.len_accumulator.seq_initialize(len_enc, weights, rng)
        self.length_nobs += float(np.sum(weights))

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
        weights = np.asarray(weights)
        if (
            weights.shape != (sz,)
            or not np.issubdtype(weights.dtype, np.number)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise ValueError("Markov weights must be a finite non-negative vector aligned with encoded rows.")
        weights = weights.astype(float, copy=False)
        for value in inv_key_map:
            self._check_state(value)

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

        self.len_accumulator.seq_update(
            len_enc,
            weights,
            None if estimate is None else estimate.len_dist,
        )
        self.length_nobs += float(np.sum(weights))

    def seq_update_engine(self, x: enc_data_type, weights: Any, estimate: MarkovChainDistribution, engine: Any) -> None:
        """Engine-resident E-step: initial-state counts are reduced on the active engine and the
        transition weights are gathered on the engine before filling the sparse count maps; the
        length accumulator is routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        key_sz = len(inv_key_map)
        w_eng = engine.asarray(weights)
        host_weights = np.asarray(engine.to_numpy(w_eng))
        if (
            host_weights.shape != (sz,)
            or not np.issubdtype(host_weights.dtype, np.number)
            or np.any(~np.isfinite(host_weights))
            or np.any(host_weights < 0.0)
        ):
            raise ValueError("Markov weights must be a finite non-negative vector aligned with encoded rows.")
        for value in inv_key_map:
            self._check_state(value)

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
        self.length_nobs += float(np.sum(host_weights))

    def combine(self, suff_stat: MarkovChainStatistics) -> "MarkovChainAccumulator":
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
        current_states = self.levels
        if current_states is None:
            observed_states = self._states()
            current_states = observed_states or None
        checked = _validate_markov_statistics(
            suff_stat,
            expected_states=current_states,
            path="MarkovChainAccumulator.combine",
        )
        if self.levels is None:
            self.levels = checked.states
        for index, state in enumerate(checked.states):
            count = checked.initial_counts[index]
            if count:
                self.init_count_map[state] = self.init_count_map.get(state, 0.0) + count
            for next_index, next_state in enumerate(checked.states):
                count = checked.transition_counts[index][next_index]
                if count:
                    row = self.trans_count_map.setdefault(state, {})
                    row[next_state] = row.get(next_state, 0.0) + count
        self.len_accumulator.combine(checked.length)
        self.length_nobs += checked.length_nobs

        return self

    def value(self) -> MarkovChainStatistics:
        """Return initial-state, transition, and length sufficient statistics."""
        states = self._states()
        if not states:
            raise ValueError(
                "Markov statistics have no declared or observed states; configure estimator levels."
            )
        return MarkovChainStatistics(
            1,
            states,
            tuple(float(self.init_count_map.get(state, 0.0)) for state in states),
            tuple(
                tuple(float(self.trans_count_map.get(state, {}).get(next_state, 0.0)) for next_state in states)
                for state in states
            ),
            self.length_nobs,
            self.len_accumulator.value(),
        )

    def from_value(self, x: MarkovChainStatistics) -> "MarkovChainAccumulator":
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
        checked = _validate_markov_statistics(
            x,
            expected_states=self.levels,
            path="MarkovChainAccumulator.from_value",
        )
        if self.levels is None:
            self.levels = checked.states
        self.init_count_map = {
            state: checked.initial_counts[index]
            for index, state in enumerate(checked.states)
            if checked.initial_counts[index] != 0.0
        }
        self.trans_count_map = {}
        for index, state in enumerate(checked.states):
            row = {
                next_state: checked.transition_counts[index][next_index]
                for next_index, next_state in enumerate(checked.states)
                if checked.transition_counts[index][next_index] != 0.0
            }
            if row:
                self.trans_count_map[state] = row
        self.len_accumulator.from_value(checked.length)
        self.length_nobs = checked.length_nobs

        return self

    def scale(self, c: float) -> "MarkovChainAccumulator":
        """Scale initial, transition, and length sufficient statistics by a constant."""
        checked = _finite_nonnegative(c, label="Markov statistic scale")
        for key in list(self.init_count_map.keys()):
            self.init_count_map[key] *= checked
        for tmap in self.trans_count_map.values():
            for key in list(tmap.keys()):
                tmap[key] *= checked
        self.len_accumulator.scale(checked)
        self.length_nobs *= checked
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
                stats_dict[self.keys] = copy.deepcopy(self)

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
        self,
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: str | None = None,
        levels: Iterable[T] | None = None,
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
        self.levels = None if levels is None else tuple(levels)

    def make(self) -> "MarkovChainAccumulator":
        """Return a new Markov-chain accumulator."""
        len_acc = self.len_factory.make()
        return MarkovChainAccumulator(len_accumulator=len_acc, keys=self.keys, levels=self.levels)


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
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _finite_nonnegative(pseudo_count, label="Markov pseudo_count")
        )
        self.levels = None if levels is None else _canonical_states(levels, label="Markov estimator levels")
        if self.levels == ():
            raise ValueError("Markov estimator levels cannot be empty.")
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.keys = keys
        self.set_prior(prior)

    def accumulator_factory(self) -> "MarkovChainAccumulatorFactory":
        """Returns MarkovChainAccumulatorFactory for creating MarkovChainAccumulator."""
        return MarkovChainAccumulatorFactory(
            len_factory=self.len_estimator.accumulator_factory(),
            keys=self.keys,
            levels=self.levels,
        )

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
        if prior is None:
            self.prior = None
            self.prior_states = None
            self.init_prior = None
            self.row_priors = None
            self.has_conj_prior = False
            return

        states, init_prior, row_priors = _validate_markov_prior(
            prior,
            expected_states=self.levels,
        )
        if self.levels is None:
            self.levels = states
        self.prior = (states, init_prior, row_priors)
        self.prior_states = states
        self.init_prior = init_prior
        self.row_priors = row_priors
        self.has_conj_prior = True

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

    def estimate(self, nobs: float | None, suff_stat: MarkovChainStatistics) -> "MarkovChainDistribution":
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
        checked = _validate_markov_statistics(
            suff_stat,
            expected_states=self.levels,
            path="MarkovChainEstimator.estimate",
        )
        _validate_effective_nobs(nobs, checked)
        if self.has_conj_prior:
            return self._estimate_conjugate(checked)
        elif self.pseudo_count is not None:
            return self.estimate1(checked.length_nobs, checked)
        else:
            return self.estimate0(checked.length_nobs, checked)

    def _estimate_conjugate(self, suff_stat: MarkovChainStatistics) -> "MarkovChainDistribution":
        """Clamped Dirichlet MAP estimate over the fixed prior ``states`` ordering.

        The initial-state and per-row transition probabilities are the clamped Dirichlet MAP
        (counts + alpha - 1, floored at zero and renormalized; posterior mean when degenerate) and
        the posterior Dirichlets (counts + alpha) are carried forward as the new prior. Mirrors
        mixle.bstats.markov_chain.MarkovChainEstimator.estimate exactly, mapping the stats dict-based
        sufficient statistics onto the fixed state ordering.
        """
        from mixle.stats.bayes.dirichlet import DirichletDistribution

        checked = _validate_markov_statistics(
            suff_stat,
            expected_states=self.prior_states,
            path="MarkovChainEstimator._estimate_conjugate",
        )
        states = self.prior_states
        s = len(states)
        init_counts = np.asarray(checked.initial_counts, dtype=float)
        trans_counts = np.asarray(checked.transition_counts, dtype=float)

        len_dist = self.len_estimator.estimate(checked.length_nobs, checked.length)

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

    def estimate0(self, nobs: float | None, suff_stat: MarkovChainStatistics) -> "MarkovChainDistribution":
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
        checked = _validate_markov_statistics(
            suff_stat,
            expected_states=self.levels,
            path="MarkovChainEstimator.estimate0",
        )
        _validate_effective_nobs(nobs, checked)
        init_counts = np.asarray(checked.initial_counts, dtype=float)
        initial_total = float(np.sum(init_counts))
        if initial_total <= 0.0:
            raise ValueError("Markov MLE requires positive initial-state evidence or a declared prior.")
        init_prob_map = {
            state: float(init_counts[index] / initial_total)
            for index, state in enumerate(checked.states)
        }

        trans_map = {}
        for index, state in enumerate(checked.states):
            row = np.asarray(checked.transition_counts[index], dtype=float)
            total = float(np.sum(row))
            if total <= 0.0:
                raise ValueError(
                    "Markov MLE row %r requires transition evidence, pseudo-count smoothing, or a prior."
                    % (state,)
                )
            trans_map[state] = {
                next_state: float(row[next_index] / total)
                for next_index, next_state in enumerate(checked.states)
            }

        len_dist = self.len_estimator.estimate(checked.length_nobs, checked.length)

        return MarkovChainDistribution(init_prob_map, trans_map, len_dist=len_dist, name=self.name)

    def estimate1(self, nobs: float | None, suff_stat: MarkovChainStatistics) -> "MarkovChainDistribution":
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
        checked = _validate_markov_statistics(
            suff_stat,
            expected_states=self.levels,
            path="MarkovChainEstimator.estimate1",
        )
        _validate_effective_nobs(nobs, checked)
        if self.pseudo_count is None:
            return self.estimate0(checked.length_nobs, checked)
        pseudo_count = _finite_nonnegative(self.pseudo_count, label="Markov pseudo_count")
        if pseudo_count <= 0.0:
            return self.estimate0(checked.length_nobs, checked)
        per_state = pseudo_count / len(checked.states)
        init_counts = np.asarray(checked.initial_counts, dtype=float) + per_state
        init_prob_map = {
            state: float(init_counts[index] / np.sum(init_counts))
            for index, state in enumerate(checked.states)
        }
        trans_map = {}
        for index, state in enumerate(checked.states):
            row = np.asarray(checked.transition_counts[index], dtype=float) + per_state
            trans_map[state] = {
                next_state: float(row[next_index] / np.sum(row))
                for next_index, next_state in enumerate(checked.states)
            }
        len_dist = self.len_estimator.estimate(checked.length_nobs, checked.length)
        return MarkovChainDistribution(init_prob_map, trans_map, len_dist=len_dist, name=self.name)


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

        import itertools

        if isinstance(x, (str, bytes)) or not isinstance(x, (list, tuple, np.ndarray)):
            raise ContractError(
                "MarkovChainDataEncoder.seq_encode",
                "a sequence of state sequences",
                type(x).__name__,
            )
        for index, entry in enumerate(x):
            if isinstance(entry, (str, bytes)) or not isinstance(entry, (list, tuple, np.ndarray)):
                raise ContractError(
                    "MarkovChainDataEncoder.seq_encode (row %d)" % index,
                    "a state sequence",
                    type(entry).__name__,
                )

        obs_cnt = np.fromiter((len(entry) for entry in x), dtype=np.int64, count=len(x))
        n_tokens = int(obs_cnt.sum())
        flat = list(itertools.chain.from_iterable(x))
        state_types = set(map(type, flat))

        # State -> first-seen integer code. Fast path: a sortable homogeneous array lets np.unique
        # produce the codes in one vectorized pass, remapped to FIRST-SEEN order so the encoding is
        # byte-identical to the dict walk (inv_key_map order included). Any failure (mixed or
        # unorderable state types -- dicts compare where sorts cannot) falls back to the dict walk.
        codes_flat = None
        _SAFE_STATE_TYPES = (str, int, float, bool, np.str_, np.bytes_, np.integer, np.floating, np.bool_)
        if n_tokens:
            # The fast path must not let np.asarray COERCE across python types: [1, "1"] becomes two
            # equal strings under coercion while the dict walk keeps them distinct states. One cheap
            # type-homogeneity scan gates it; anything else (tuples, custom objects, mixed types)
            # takes the dict walk with the original semantics.
            if len(state_types) == 1 and issubclass(next(iter(state_types)), _SAFE_STATE_TYPES):
                try:
                    arr = np.asarray(flat)
                    if arr.dtype != object and arr.ndim == 1:
                        uniq, first_pos, inverse = np.unique(arr, return_index=True, return_inverse=True)
                        order = np.argsort(first_pos, kind="stable")
                        remap = np.empty(len(uniq), dtype=np.int64)
                        remap[order] = np.arange(len(uniq), dtype=np.int64)
                        codes_flat = remap[inverse.reshape(-1)]
                        inv_key_map = np.asarray([uniq[j] for j in order])
                except (TypeError, ValueError):  # pathological values: take the dict walk below
                    codes_flat = None
        if codes_flat is None:
            key_map: dict = {}
            codes_flat = np.empty(n_tokens, dtype=np.int64)
            pos = 0
            for value in flat:
                code = key_map.get(value)
                if code is None:
                    code = len(key_map)
                    key_map[value] = code
                codes_flat[pos] = code
                pos += 1
            inv_key_map = [None] * len(key_map)
            for k, v in key_map.items():
                inv_key_map[v] = k
            # dtype=object for heterogeneous state types: a bare asarray COERCES [1, "1"] into two
            # equal strings, silently merging states the dict walk (and every downstream key_map
            # lookup) keeps distinct -- a pre-existing hazard of the original encoder, fixed here.
            safe_scalar_type = (
                len(state_types) == 1
                and issubclass(next(iter(state_types)), _SAFE_STATE_TYPES)
            )
            if safe_scalar_type:
                inv_key_map = np.asarray(inv_key_map)
            else:
                object_values = np.empty(len(inv_key_map), dtype=object)
                object_values[:] = inv_key_map
                inv_key_map = object_values

        # Structure arrays from offsets, fully vectorized: rows of length >= 1 contribute their first
        # token to the init arrays; every within-row position >= 1 contributes a (prev, next) pair.
        starts = np.concatenate(([0], np.cumsum(obs_cnt)[:-1])) if len(x) else np.zeros(0, dtype=np.int64)
        nonempty = obs_cnt > 0
        entries_idx0 = np.nonzero(nonempty)[0]
        init_entries = codes_flat[starts[nonempty]] if n_tokens else np.zeros(0, dtype=np.int64)

        row_ids = np.repeat(np.arange(len(x), dtype=np.int64), obs_cnt)
        token_pos = np.arange(n_tokens, dtype=np.int64) - np.repeat(starts, obs_cnt)
        trans_mask = token_pos >= 1
        entries_idx1 = row_ids[trans_mask]
        next_entries = codes_flat[trans_mask] if n_tokens else np.zeros(0, dtype=np.int64)
        prev_entries = codes_flat[np.nonzero(trans_mask)[0] - 1] if n_tokens else np.zeros(0, dtype=np.int64)

        len_enc = self.len_encoder.seq_encode(obs_cnt)

        return (
            len(x),
            entries_idx0,
            entries_idx1,
            init_entries,
            prev_entries,
            next_entries,
            inv_key_map,
            len_enc,
        )

    def row_count(self, x: Any) -> int:
        """Validate encoded Markov geometry and return the outer sequence count."""
        if not isinstance(x, tuple) or len(x) != 8:
            raise ValueError("Markov encoding must be an 8-tuple.")
        size, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x
        size = _checked_length(size, label="Markov encoded row count")
        idx0 = np.asarray(idx0)
        idx1 = np.asarray(idx1)
        init_x = np.asarray(init_x)
        prev_x = np.asarray(prev_x)
        next_x = np.asarray(next_x)
        inv_key_map = np.asarray(inv_key_map)
        for label, array in (
            ("initial row indices", idx0),
            ("transition row indices", idx1),
            ("initial state codes", init_x),
            ("previous state codes", prev_x),
            ("next state codes", next_x),
        ):
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise ValueError("Markov %s must be a one-dimensional integer array." % label)
        if len(idx0) != len(init_x) or len(idx1) != len(prev_x) or len(idx1) != len(next_x):
            raise ValueError("Markov encoded state arrays are not aligned.")
        if (
            np.any(idx0 < 0)
            or np.any(idx0 >= size)
            or np.any(idx1 < 0)
            or np.any(idx1 >= size)
            or len(np.unique(idx0)) != len(idx0)
        ):
            raise ValueError("Markov encoded row indices are invalid.")
        state_count = len(inv_key_map)
        if (
            np.any(init_x < 0)
            or np.any(init_x >= state_count)
            or np.any(prev_x < 0)
            or np.any(prev_x >= state_count)
            or np.any(next_x < 0)
            or np.any(next_x >= state_count)
        ):
            raise ValueError("Markov encoded state codes are invalid.")
        initial_rows = set(idx0.tolist())
        if any(row not in initial_rows for row in idx1.tolist()):
            raise ValueError("Markov transition rows must also carry an initial state.")
        try:
            if len(set(inv_key_map.tolist())) != state_count:
                raise ValueError("Markov encoded state labels must be unique.")
        except TypeError as exc:
            raise ValueError("Markov encoded state labels must be hashable.") from exc
        if supports(self.len_encoder, Neutral):
            if (
                isinstance(len_enc, (bool, np.bool_))
                or not isinstance(len_enc, (int, np.integer))
                or int(len_enc) != size
            ):
                raise ValueError("Markov null length encoding must retain the exact row count.")
        elif self.len_encoder.row_count(len_enc) != size:
            raise ValueError("Markov length encoding does not preserve outer rows.")
        return size
