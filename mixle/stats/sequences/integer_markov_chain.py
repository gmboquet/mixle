"""Integer Markov-chain distributions with optional lagged transitions.

The data type: Sequence[int].

For a sequence ``x`` of length ``n >= lag > 0``, the log density is

    log(P(x)) = log(P_init(x[0:lag]) + sum_{j=0}^{n-lag-1} log(p_mat(x[j + lag] | x[j],...,x[j+lag-1])) +
                    log(P_len(n)),

where ``P_len`` is evaluated at the actual length and ``P_init`` generates
exactly ``lag`` bounded-integer states. Shorter sequences are outside the
declared domain.

"""

import copy
import heapq
import itertools
import math
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, LengthFrontierMerge, ProductEnumerator
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
from mixle.utils.exact import require_exact_bool

E1 = TypeVar("E1")  ## init encoding
E2 = TypeVar("E2")  ## len encoding
SS1 = TypeVar("SS1")  ## suff stat of init
SS2 = TypeVar("SS2")  ## suff-stat of length


class NonGenerativeIntegerMarkovChainError(TypeError):
    """Raised when a conditional integer-chain factor is asked to generate sequences."""


class IntegerMarkovChainStatistics(NamedTuple):
    """Immutable sufficient statistics for one fixed integer-chain support."""

    schema_version: int
    num_values: int
    lag: int
    transition_counts: tuple[tuple[tuple[int, ...], int, float], ...]
    initial_nobs: float
    initial: Any | None
    length_nobs: float
    length: Any | None


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive exact integer." % label)
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise TypeError("%s must be a positive exact integer." % label)
    try:
        numeric = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a positive exact integer." % label) from exc
    if not math.isfinite(numeric) or numeric < 1.0 or math.floor(numeric) != numeric:
        raise ValueError("%s must be a positive exact integer." % label)
    if numeric > np.iinfo(np.intp).max:
        raise ValueError("%s exceeds the platform index range." % label)
    return int(numeric)


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


def _exact_state(value: Any, *, num_values: int, label: str) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise TypeError("%s must be an exact integer state." % label)
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise TypeError("%s must be an exact integer state." % label)
    try:
        numeric = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be an exact integer state." % label) from exc
    if not math.isfinite(numeric) or math.floor(numeric) != numeric or numeric < 0.0 or numeric >= num_values:
        raise ValueError("%s must be in the declared support [0, %d)." % (label, num_values))
    return int(numeric)


def _checked_observation(
    value: Any,
    *,
    num_values: int,
    lag: int,
    label: str,
) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError("%s must be a one-dimensional sequence of integer states." % label)
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("%s must be a sequence of integer states." % label)
    states = tuple(
        _exact_state(state, num_values=num_values, label="%s[%d]" % (label, index)) for index, state in enumerate(value)
    )
    if len(states) < lag:
        raise ValueError("%s length must be at least lag=%d." % (label, lag))
    return states


def _validate_length_child(length_dist: SequenceEncodableProbabilityDistribution, *, lag: int) -> None:
    """Require an integer length law whose positive support starts at or above ``lag``."""
    from mixle.stats.combinator.sequence import _checked_length, _validate_length_distribution

    _validate_length_distribution(length_dist)
    size = length_dist.support_size()
    if size is None:
        raise TypeError(
            "IntegerMarkovChainDistribution length law must expose finite support so its minimum "
            "can be proved to be at least lag."
        )
    items = tuple(
        itertools.islice(
            child_enumerator(length_dist, "IntegerMarkovChainDistribution.len_dist"),
            size + 1,
        )
    )
    if len(items) > size:
        raise ValueError("integer Markov length enumeration must expose its exact finite support.")
    values = tuple(
        _checked_length(value, label="integer Markov length support value")
        for value, log_probability in items
        if float(log_probability) > -np.inf
    )
    if not values or len(set(values)) != len(values):
        raise ValueError("integer Markov length law must expose unique positive-mass support.")
    if any(value < lag for value in values):
        raise ValueError("integer Markov length support must contain only lengths >= lag=%d." % lag)


def _validate_initial_child(
    initial_dist: SequenceEncodableProbabilityDistribution,
    *,
    num_values: int,
    lag: int,
) -> None:
    from mixle.stats.compute.declarations import declaration_for

    declaration = declaration_for(initial_dist)
    if declaration is None or declaration.support != "sequence":
        raise TypeError("IntegerMarkovChainDistribution init_dist must declare sequence support.")
    length_dist = getattr(initial_dist, "len_dist", None)
    element_dist = getattr(initial_dist, "dist", None)
    if length_dist is None or element_dist is None:
        raise TypeError("IntegerMarkovChainDistribution init_dist must expose element and length laws.")
    from mixle.stats.combinator.sequence import _checked_length, _validate_length_distribution

    _validate_length_distribution(length_dist)
    size = length_dist.support_size()
    if size is None:
        raise TypeError("integer Markov initial length law must expose finite support.")
    items = tuple(
        itertools.islice(
            child_enumerator(length_dist, "IntegerMarkovChainDistribution.init_dist.len_dist"),
            size + 1,
        )
    )
    if len(items) > size:
        raise ValueError("integer Markov initial length enumeration exceeds its declared support.")
    lengths = tuple(
        _checked_length(value, label="integer Markov initial length support value")
        for value, log_probability in items
        if float(log_probability) > -np.inf
    )
    if set(lengths) != {lag}:
        raise ValueError("integer Markov init_dist must generate exactly lag=%d states." % lag)
    element_declaration = declaration_for(element_dist)
    lower = getattr(element_dist, "min_val", None)
    upper = getattr(element_dist, "max_val", None)
    if (
        element_declaration is None
        or element_declaration.support != "bounded_integer"
        or lower is None
        or upper is None
        or int(lower) < 0
        or int(upper) >= num_values
    ):
        raise TypeError(
            "integer Markov init_dist elements must declare bounded integer support inside [0, num_values)."
        )


def _validate_statistics(
    value: Any,
    *,
    num_values: int,
    lag: int,
    path: str,
) -> IntegerMarkovChainStatistics:
    if not isinstance(value, IntegerMarkovChainStatistics) or value.schema_version != 1:
        raise ContractError(
            path,
            "IntegerMarkovChainStatistics schema version 1",
            type(value).__name__,
            "pass the value produced by IntegerMarkovChainAccumulator.value().",
        )
    if value.num_values != num_values or value.lag != lag:
        raise ContractError(
            path,
            "statistics for num_values=%d and lag=%d" % (num_values, lag),
            "num_values=%r and lag=%r" % (value.num_values, value.lag),
        )
    transitions = []
    seen = set()
    for index, entry in enumerate(value.transition_counts):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ContractError(
                "%s.transition_counts[%d]" % (path, index),
                "(lag_prefix, next_state, count)",
                repr(entry),
            )
        prefix = _checked_observation(
            entry[0],
            num_values=num_values,
            lag=lag,
            label="%s.transition_counts[%d].prefix" % (path, index),
        )
        if len(prefix) != lag:
            raise ValueError("%s transition prefixes must have exactly lag states." % path)
        target = _exact_state(
            entry[1],
            num_values=num_values,
            label="%s.transition_counts[%d].target" % (path, index),
        )
        key = (prefix, target)
        if key in seen:
            raise ValueError("%s contains duplicate transition %r." % (path, key))
        seen.add(key)
        transitions.append((prefix, target, _finite_nonnegative(entry[2], label="%s transition count" % path)))
    canonical = tuple(sorted(transitions, key=lambda entry: (entry[0], entry[1])))
    if tuple(transitions) != canonical:
        raise ValueError("%s transition counts must use canonical lexicographic order." % path)
    initial_nobs = _finite_nonnegative(value.initial_nobs, label="%s.initial_nobs" % path)
    length_nobs = _finite_nonnegative(value.length_nobs, label="%s.length_nobs" % path)
    if not math.isclose(initial_nobs, length_nobs, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("%s initial_nobs and length_nobs must match." % path)
    return IntegerMarkovChainStatistics(
        1,
        num_values,
        lag,
        canonical,
        initial_nobs,
        value.initial,
        length_nobs,
        value.length,
    )


class IntegerMarkovChainDistribution(SequenceEncodableProbabilityDistribution):
    """Markov-chain distribution over integer-valued states."""

    def compute_capabilities(self):
        """Declare generated-compute support inherited from initial and length distributions."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(
            engine_ready=intersect_engine_ready((self.init_dist, self.len_dist)), kernel_status="generic_table"
        )

    def compute_declaration(self):
        """Return the generated-compute declaration for the integer Markov chain."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        init = None if supports(self.init_dist, Neutral) else declaration_for(self.init_dist)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(d for d in (init, length) if d is not None)
        roles = []
        if init is not None:
            roles.append("initial")
        if length is not None:
            roles.append("length")
        return DistributionDeclaration(
            name="integer_markov_chain",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("num_values", constraint="integer", differentiable=False),
                ParameterSpec("cond_dist", constraint="row_simplex_matrix"),
                ParameterSpec("lag", constraint="integer", differentiable=False),
            ),
            statistics=(
                StatisticSpec("transition_counts", kind="mapping"),
                StatisticSpec("initial", kind="child_stat"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="finite_integer_sequence",
            children=children,
            child_roles=tuple(roles),
            differentiable=False,
        )

    def __init__(
        self,
        num_values: int,
        cond_dist: list[list[float]] | np.ndarray,
        lag: int = 1,
        init_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create an integer Markov-chain distribution with finite lag.


        Args:
            num_values (int): Total number of values in support.
            cond_dist (Array-like): Should be num_vals ** lag by num_vals with transition probabilities for each
                lagged length tuple (v_0,v_1,..,v_{lag}).
            lag (int): Lag length for conditional density.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for initial states
                of Markov chain (with length lag). Should be a distribution compatible with Sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for the length of
                observations.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        Attributes:
            num_values (int): Total number of values in support.
            cond_dist (Array-like): Should be num_vals ** lag by num_vals with transition probabilities for each
                lagged length tuple (v_0,v_1,..,v_{lag}).
            lag (int): Lag length for conditional density.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for initial states
                of Markov chain (with length lag). Should be a distribution compatible with Sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for the length of
                observations.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        """
        self.num_values = _positive_integer(num_values, label="num_values")
        self.lag = _positive_integer(lag, label="lag")
        expected_shape = (self.num_values**self.lag, self.num_values)
        try:
            conditional = np.array(cond_dist, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("cond_dist must be a finite numeric transition matrix.") from exc
        if conditional.shape != expected_shape:
            raise ValueError("cond_dist must have shape %r; got %r." % (expected_shape, conditional.shape))
        if np.any(~np.isfinite(conditional)) or np.any(conditional < 0.0):
            raise ValueError("cond_dist entries must be finite and non-negative.")
        row_sums = conditional.sum(axis=1)
        if np.any(~np.isclose(row_sums, 1.0, rtol=1.0e-12, atol=1.0e-12)):
            raise ValueError("every cond_dist row must sum to 1.")
        conditional.setflags(write=False)
        self.cond_dist = conditional
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        if not isinstance(self.init_dist, SequenceEncodableProbabilityDistribution):
            raise TypeError("init_dist must be a probability distribution.")
        if not isinstance(self.len_dist, SequenceEncodableProbabilityDistribution):
            raise TypeError("len_dist must be a probability distribution.")
        if not supports(self.init_dist, Neutral):
            _validate_initial_child(
                self.init_dist,
                num_values=self.num_values,
                lag=self.lag,
            )
        if not supports(self.len_dist, Neutral):
            _validate_length_child(self.len_dist, lag=self.lag)
        self.name = name
        self.keys = keys

    def density_semantics(self):
        """Classify missing initial/length laws as conditional likelihood factors."""
        from mixle.stats.compute.pdist import join_density_semantics

        if supports(self.init_dist, Neutral) or supports(self.len_dist, Neutral):
            return DensitySemantics.LIKELIHOOD_FACTOR
        return join_density_semantics(child.density_semantics() for child in (self.init_dist, self.len_dist))

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = repr(self.num_values)
        s2 = repr(self.cond_dist.tolist())
        s3 = repr(self.lag)
        s4 = repr(self.init_dist) if self.init_dist is None else str(self.init_dist)
        s5 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.keys)

        return "IntegerMarkovChainDistribution(%s, %s, lag=%s, init_dist=%s, len_dist=%s, name=%s, keys=%s)" % (
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
        )

    def density(self, x: Sequence[int]) -> float:
        """Density of integer Markov chain evaluated at x.

        See log_density() for details.

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[int]) -> float:
        """Log-density of integer Markov chain evaluated at x.

        The initial child scores exactly the first ``lag`` states, each later
        state is scored by its lag-indexed transition row, and the length child
        scores the actual sequence length. Shorter observations are rejected.

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            Log-density evaluated at x.

        """
        states = _checked_observation(
            x,
            num_values=self.num_values,
            lag=self.lag,
            label="integer Markov observation",
        )
        rv = 0.0
        lag = self.lag

        m_shape = [self.num_values] * lag
        rv += self.init_dist.log_density(states[:lag])

        for i in range(len(states) - lag):
            idx = np.ravel_multi_index(states[i : (i + lag)], m_shape)
            rv += np.log(self.cond_dist[idx, states[i + lag]])

        rv += self.len_dist.log_density(len(states))

        return rv

    def seq_log_density(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
    ) -> np.ndarray:
        """Vectorized evaluation of log-density at every observation in encoded sequence.

        See log_density() for details on likelihood evaluation.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Actual observed sequence lengths.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x: See above for details.

        Returns:
            Log-density evaluated at each observation in encoded sequence.

        """
        self.dist_to_encoder().row_count(x)
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x

        rv = np.zeros(len(seq_len), dtype=np.float64)
        if len(seq_idx):
            left_idx = np.asarray(
                [np.ravel_multi_index(u[0], [self.num_values] * self.lag) for u in u_seq_values],
                dtype=np.intp,
            )
            right_idx = np.asarray([u[1] for u in u_seq_values], dtype=np.intp)
            with np.errstate(divide="ignore"):
                temp_prob = np.log(self.cond_dist[left_idx, right_idx])
            rv += np.bincount(
                seq_idx,
                weights=temp_prob[np.asarray(u_seq_idx, dtype=np.intp)],
                minlength=len(seq_len),
            )

        if not supports(self.init_dist, Neutral) and len(init_idx):
            rv[init_idx] += self.init_dist.seq_log_density(init_enc)

        if not supports(self.len_dist, Neutral) and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def backend_seq_log_density(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        engine: Any,
    ) -> Any:
        """Engine-neutral vectorized log-density for grouped integer Markov-chain encodings."""
        from mixle.stats.compute.backend import backend_seq_log_density

        self.dist_to_encoder().row_count(x)
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x
        rv = engine.zeros(len(seq_len))

        if len(seq_idx) > 0:
            left_idx = np.asarray(
                [np.ravel_multi_index(u[0], [self.num_values] * self.lag) for u in u_seq_values], dtype=np.int64
            )
            right_idx = np.asarray([u[1] for u in u_seq_values], dtype=np.int64)
            with np.errstate(divide="ignore"):
                transition_scores = np.log(self.cond_dist[left_idx, right_idx])
            transition_scores = transition_scores[u_seq_idx]
            rv = engine.index_add(rv, engine.asarray(seq_idx), engine.asarray(transition_scores))

        if self.init_dist is not None and init_enc is not None and len(init_idx) > 0:
            rv = engine.index_add(
                rv, engine.asarray(init_idx), backend_seq_log_density(self.init_dist, init_enc, engine)
            )

        if self.len_dist is not None and len_enc is not None:
            rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)

        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerMarkovChainDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer Markov-chain parameters for shared support/lag."""
        from mixle.stats.compute.stacked import stacked_component_params

        num_values = int(dists[0].num_values)
        lag = int(dists[0].lag)
        null_init_dist = supports(dists[0].init_dist, Neutral)
        null_len_dist = supports(dists[0].len_dist, Neutral)
        if any(
            int(dist.num_values) != num_values
            or int(dist.lag) != lag
            or supports(dist.init_dist, Neutral) != null_init_dist
            or supports(dist.len_dist, Neutral) != null_len_dist
            for dist in dists
        ):
            raise ValueError(
                "Stacked IntegerMarkovChainDistribution components require shared support, lag, and child policies."
            )

        init_route = None
        if not null_init_dist:
            try:
                init_route = stacked_component_params([dist.init_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "IntegerMarkovChain initial child %s is not stackable: %s"
                    % (type(dists[0].init_dist).__name__, exc)
                )

        length_route = None
        if not null_len_dist:
            try:
                length_route = stacked_component_params([dist.len_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "IntegerMarkovChain length child %s is not stackable: %s" % (type(dists[0].len_dist).__name__, exc)
                )

        with np.errstate(divide="ignore"):
            log_cond = np.stack([np.log(dist.cond_dist) for dist in dists], axis=2)

        return {
            "__pysp_component_axis__": {"log_cond": 2},
            "num_values": num_values,
            "lag": lag,
            "log_cond": engine.asarray(log_cond),
            "init_route": init_route,
            "length_route": length_route,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(
        cls,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        params: dict[str, Any],
        engine: Any,
    ) -> Any:
        """Return an ``(n, k)`` matrix of integer Markov-chain log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        if not isinstance(x, tuple) or len(x) != 7:
            raise ValueError("stacked integer Markov encoding must be a seven-slot tuple.")
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x
        rv = engine.zeros((len(seq_len), int(params["num_components"])))

        if len(seq_idx) > 0:
            left_idx = np.asarray(
                [np.ravel_multi_index(u[0], [params["num_values"]] * params["lag"]) for u in u_seq_values],
                dtype=np.int64,
            )
            right_idx = np.asarray([u[1] for u in u_seq_values], dtype=np.int64)
            transition_scores = params["log_cond"][engine.asarray(left_idx), engine.asarray(right_idx), :]
            transition_scores = transition_scores[engine.asarray(u_seq_idx), :]
            rv = engine.index_add(rv, engine.asarray(seq_idx), transition_scores)

        if params["init_route"] is not None and init_enc is not None and len(init_idx) > 0:
            rv = engine.index_add(
                rv, engine.asarray(init_idx), stacked_component_log_density(init_enc, params["init_route"], engine)
            )

        if params["length_route"] is not None and len_enc is not None:
            rv = rv + stacked_component_log_density(len_enc, params["length_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        weights: Any,
        params: dict[str, Any],
        engine: Any,
        estimator: Any,
    ) -> tuple[Any, ...]:
        """Return per-component versioned fixed-support sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x
        ww = engine.asarray(weights)
        num_components = int(params["num_components"])

        if len(u_seq_values) > 0:
            trans_weights = ww[engine.asarray(seq_idx)]
            zero_rows = trans_weights * engine.asarray(0.0)
            unique_idx = engine.asarray(u_seq_idx)
            rows = []
            for value_index in range(len(u_seq_values)):
                mask = unique_idx == engine.asarray(value_index)
                rows.append(engine.sum(engine.where(mask[:, None], trans_weights, zero_rows), axis=0))
            trans_counts = np.asarray(engine.to_numpy(engine.stack(rows, axis=0)), dtype=np.float64)
        else:
            trans_counts = np.zeros((0, num_components), dtype=np.float64)

        outer_estimators = tuple(getattr(estimator, "estimators", ()))

        if params["init_route"] is None or init_enc is None:
            init_by_component = tuple(None for _ in range(num_components))
        else:
            init_estimators = tuple(
                getattr(component_est, "init_estimator", None) for component_est in outer_estimators
            )
            init_estimator = StackedEstimatorView(init_estimators) if len(init_estimators) == num_components else None
            init_stats = stacked_component_sufficient_statistics(
                init_enc, ww[engine.asarray(init_idx)], params["init_route"], engine, init_estimator
            )
            init_by_component = unstack_component_stats(init_stats, num_components)

        if params["length_route"] is None or len_enc is None:
            length_by_component = tuple(None for _ in range(num_components))
        else:
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

        weights_np = np.asarray(engine.to_numpy(ww), dtype=np.float64)
        init_nobs = weights_np[np.asarray(init_idx, dtype=np.intp), :].sum(axis=0)
        length_nobs = weights_np.sum(axis=0)
        values = tuple(u_seq_values)
        return tuple(
            IntegerMarkovChainStatistics(
                1,
                int(params["num_values"]),
                int(params["lag"]),
                tuple(
                    sorted(
                        (
                            tuple(values[value_index][0]),
                            int(values[value_index][1]),
                            float(trans_counts[value_index, component]),
                        )
                        for value_index in range(len(values))
                    )
                ),
                float(init_nobs[component]),
                init_by_component[component],
                float(length_nobs[component]),
                length_by_component[component],
            )
            for component in range(num_components)
        )

    def sampler(self, seed: int | None = None) -> "IntegerMarkovChainSampler":
        """Return a sampler for this integer Markov chain."""
        if self.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR:
            raise NonGenerativeIntegerMarkovChainError(
                "IntegerMarkovChainDistribution requires proper init_dist and len_dist laws for sampling."
            )
        return IntegerMarkovChainSampler(self, seed)

    def transition_sampler(self, seed: int | None = None) -> "IntegerMarkovChainSampler":
        """Return a sampler exposing only ``sample_given`` for conditional transition use."""
        return IntegerMarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None):
        """Return an estimator initialized from this integer Markov chain."""
        init_est = self.init_dist.estimator(pseudo_count=pseudo_count)
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

        return IntegerMarkovChainEstimator(
            num_values=self.num_values,
            lag=self.lag,
            init_estimator=init_est,
            len_estimator=len_est,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerMarkovChainDataEncoder":
        """Return a data encoder for iid integer Markov-chain observations."""
        len_encoder = self.len_dist.dist_to_encoder()
        init_encoder = self.init_dist.dist_to_encoder()
        return IntegerMarkovChainDataEncoder(
            num_values=self.num_values,
            lag=self.lag,
            len_encoder=len_encoder,
            init_encoder=init_encoder,
        )

    def enumerator(self) -> "IntegerMarkovChainEnumerator":
        """Returns IntegerMarkovChainEnumerator iterating integer sequences in descending probability order."""
        return IntegerMarkovChainEnumerator(self)


class IntegerMarkovChainEnumerator(DistributionEnumerator):
    """Enumerates integer Markov-chain sequences in descending probability order."""

    def __init__(self, dist: IntegerMarkovChainDistribution) -> None:
        """Create an enumerator for integer Markov-chain sequences.

        Lengths are pulled from len_dist. For each length, a best-first search expands
        prefixes using an admissible upper bound based on the largest transition probability.

        Args:
            dist (IntegerMarkovChainDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        if dist.lag <= 0:
            raise EnumerationError(dist, reason="lag must be positive for enumeration")
        if supports(dist.init_dist, Neutral):
            raise EnumerationError(dist, reason="no initial-prefix distribution is modeled (init_dist is Null)")
        if supports(dist.len_dist, Neutral):
            raise EnumerationError(dist, reason="no length distribution is modeled (len_dist is Null)")

        with np.errstate(divide="ignore"):
            self._log_cond = np.log(np.asarray(dist.cond_dist, dtype=np.float64))

        expected_shape = (dist.num_values**dist.lag, dist.num_values)
        if self._log_cond.shape != expected_shape:
            raise EnumerationError(
                dist,
                reason="cond_dist shape must be %s for num_values=%d and lag=%d"
                % (expected_shape, dist.num_values, dist.lag),
            )

        self._choices = [(i, 0.0) for i in range(dist.num_values)]
        self._transitions: list[list[tuple[int, float]]] = []
        steps = []
        for row in self._log_cond:
            entries = [(int(i), float(lp)) for i, lp in enumerate(row) if lp > -np.inf]
            entries.sort(key=lambda u: -u[1])
            self._transitions.append(entries)
            steps.extend(lp for _, lp in entries)
        self._max_step = min(max(steps), 0.0) if steps else -np.inf
        self._shape = [dist.num_values] * dist.lag

        len_stream = BufferedStream(child_enumerator(dist.len_dist, "IntegerMarkovChainDistribution.len_dist"))
        self._merge = LengthFrontierMerge(len_stream, self._kbest_paths)

    def _init_iterator(self) -> Iterator[tuple[Any, float]]:
        if supports(self.dist.init_dist, Neutral):
            streams = [BufferedStream(iter(self._choices)) for _ in range(self.dist.lag)]
            return iter(ProductEnumerator(streams, combine=list))
        return iter(child_enumerator(self.dist.init_dist, "IntegerMarkovChainDistribution.init_dist"))

    def _valid_prefix(self, value: Any) -> tuple[int, ...]:
        try:
            prefix = _checked_observation(
                value,
                num_values=self.dist.num_values,
                lag=self.dist.lag,
                label="integer Markov initial support value",
            )
        except (TypeError, ValueError) as exc:
            raise EnumerationError(
                self.dist,
                reason="init_dist emitted an invalid integer prefix: %s" % exc,
            ) from exc
        if len(prefix) != self.dist.lag:
            raise EnumerationError(
                self.dist,
                reason="init_dist prefix length must equal lag=%d" % self.dist.lag,
            )
        return prefix

    def _bound(self, exact: float, remaining: int, lp_len: float) -> float:
        if exact == -np.inf:
            return -np.inf
        if remaining == 0:
            return exact + lp_len
        if self._max_step == -np.inf:
            return -np.inf
        return exact + remaining * self._max_step + lp_len

    def _row_index(self, prefix: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(prefix[-self.dist.lag :], self._shape))

    def _short_paths(self, n: int, lp_len: float) -> Iterator[tuple[list[int], float]]:
        raise EnumerationError(
            self.dist,
            reason="length support contains %d below lag=%d" % (n, self.dist.lag),
        )

    def _kbest_paths(self, n: int, lp_len: float) -> Iterator[tuple[list[int], float]]:
        if n == 0:
            yield ([], lp_len)
            return
        if n < self.dist.lag:
            yield from self._short_paths(n, lp_len)
            return

        counter = itertools.count()
        heap: list[tuple[float, int, tuple[int, ...], float]] = []
        init_stream = BufferedStream(self._init_iterator())
        init_rank = 0
        pending_init: tuple[tuple[int, ...], float, float] | None = None
        init_remaining = n - self.dist.lag

        def next_pending_init() -> tuple[tuple[int, ...], float, float] | None:
            nonlocal init_rank, pending_init
            while pending_init is None:
                item = init_stream.get(init_rank)
                if item is None:
                    return None
                init_rank += 1
                prefix = self._valid_prefix(item[0])
                exact = float(item[1])
                bound = self._bound(exact, init_remaining, lp_len)
                if bound == -np.inf:
                    continue
                pending_init = (prefix, exact, bound)
            return pending_init

        while True:
            frontier = next_pending_init()
            frontier_bound = -np.inf if frontier is None else frontier[2]
            if heap and -heap[0][0] >= frontier_bound:
                _, _, prefix, exact = heapq.heappop(heap)
                if len(prefix) == n:
                    yield (list(prefix), exact + lp_len)
                    continue

                row_idx = self._row_index(prefix)
                remaining = n - len(prefix) - 1
                for value, lp_step in self._transitions[row_idx]:
                    exact2 = exact + lp_step
                    bound2 = self._bound(exact2, remaining, lp_len)
                    if bound2 > -np.inf:
                        heapq.heappush(heap, (-bound2, next(counter), prefix + (value,), exact2))
            elif frontier is not None:
                prefix, exact, bound = frontier
                pending_init = None
                heapq.heappush(heap, (-bound, next(counter), prefix, exact))
            else:
                if not heap:
                    return

    def __next__(self) -> tuple[list[int], float]:
        return next(self._merge)


class IntegerMarkovChainSampler(DistributionSampler):
    """Draw integer-valued sequences from an :class:`IntegerMarkovChainDistribution`."""

    def __init__(self, dist: IntegerMarkovChainDistribution, seed: int | None) -> None:
        """Create a sampler for an integer Markov-chain distribution.

        Args:
            dist (IntegerMarkovChainDistribution): Integer Markov chain to sample from.
            seed (Optional[int]): Set the seed for random sampling.

        Attributes:
            dist (IntegerMarkovChainDistribution): Integer Markov chain to sample from.
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            trans_sampler (RandomState): Random state for sampling transitions.

        """
        rng = np.random.RandomState(seed)
        seeds = rng.randint(0, maxrandint, size=3)

        self.dist = dist
        self.rng = rng
        self.trans_sampler = np.random.RandomState(seeds[0])

        # init/len samplers are only needed for unconditional sampling; sample_given works without them
        self.init_sampler = None if supports(dist.init_dist, Neutral) else dist.init_dist.sampler(seeds[1])
        self.len_sampler = None if supports(dist.len_dist, Neutral) else dist.len_dist.sampler(seeds[2])

    def single_sample(self) -> Sequence[int]:
        """Returns a single sample from the integer Markov chain distribution."""
        if self.init_sampler is None or self.len_sampler is None:
            raise ValueError("IntegerMarkovChainSampler requires init_dist and len_dist for unconditional sampling.")
        cnt = _positive_integer(self.len_sampler.sample(), label="sampled integer Markov length")
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag

        if cnt < lag:
            raise ValueError("sampled integer Markov length must be at least lag=%d." % lag)
        rv = list(
            _checked_observation(
                self.init_sampler.sample(),
                num_values=n_val,
                lag=lag,
                label="sampled integer Markov initial prefix",
            )
        )
        if len(rv) != lag:
            raise ValueError("sampled integer Markov initial prefix must have exactly lag states.")
        for i in range(lag, cnt):
            idx = np.ravel_multi_index(rv[-lag:], m_shape)
            rv.append(int(self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :])))
        return rv

    def _sample_batched(self, size: int) -> list[Sequence[int]]:
        """Vectorized batch sample: per-chain init/length draws, then transitions across chains.

        The length and initial-state draws are taken per chain in order (byte-identical to the loop),
        then the transition step is vectorized: at each time index every live chain's lag-index is
        computed at once, the conditional rows are gathered, and all next states are drawn together.
        Because the transition draws are taken across chains rather than per chain, the transition
        portion is statistically equivalent but NOT byte-identical to ``batched=False``.
        """
        if self.init_sampler is None or self.len_sampler is None:
            raise ValueError("IntegerMarkovChainSampler requires init_dist and len_dist for unconditional sampling.")
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag

        raw_lengths = np.asarray(self.len_sampler.sample(size=size), dtype=object).reshape(-1)
        if len(raw_lengths) != size:
            raise ValueError("integer Markov length sampler returned the wrong batch size.")
        lengths = np.asarray(
            [_positive_integer(value, label="sampled integer Markov length") for value in raw_lengths],
            dtype=np.intp,
        )
        if np.any(lengths < lag):
            raise ValueError("sampled integer Markov lengths must be at least lag=%d." % lag)
        # Per-chain initial draws remain in the child's stable sampling order.
        seqs: list[list[int]] = []
        for _ in lengths:
            prefix = _checked_observation(
                self.init_sampler.sample(),
                num_values=n_val,
                lag=lag,
                label="sampled integer Markov initial prefix",
            )
            if len(prefix) != lag:
                raise ValueError("sampled integer Markov initial prefix must have exactly lag states.")
            seqs.append(list(prefix))

        max_len = int(lengths.max()) if size else 0
        for t in range(lag, max_len):
            live = np.flatnonzero((lengths >= lag) & (lengths > t))
            if len(live) == 0:
                continue
            # lag-index per live chain from its last `lag` states
            last = np.asarray([seqs[c][t - lag : t] for c in live], dtype=np.int64)
            idx = (
                np.ravel_multi_index([last[:, k] for k in range(lag)], m_shape) if lag > 0 else np.zeros(len(live), int)
            )
            rows = self.dist.cond_dist[idx, :]
            cdf = np.cumsum(rows, axis=1)
            u = self.trans_sampler.random_sample(len(live)) * cdf[:, -1]
            nxt = (cdf < u[:, None]).sum(axis=1)
            for k, c in enumerate(live):
                seqs[c].append(int(nxt[k]))
        return seqs

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[Sequence[int]] | Sequence[int]:
        """Draw iid samples from an integer Markov chain distribution.

        With ``batched=True`` (default) and ``size`` not None, the lengths and initial states are drawn
        per chain (byte-identical to the loop) and the lag-conditional transitions are vectorized across
        all live chains at each time index. The transition draws change RNG consumption order, so the
        output is statistically equivalent but NOT byte-identical to ``batched=False``. Set
        ``batched=False`` to reproduce the exact legacy per-sequence output for a given seed.

        Args:
            size (Optional[int]): If None, size is taken to be 0.
            batched (bool): Vectorize transition draws across chains (default); set False for the
                legacy per-sequence loop.

        Returns:
            Sequence[int] if size is None, else List[Sequence[int]] with length equal to size.

        """
        if size is None:
            return self.single_sample()
        if isinstance(batched, (np.bool_, bool)):
            batched = require_exact_bool(batched, "batched")
        else:
            raise TypeError("batched must be bool.")
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
            raise TypeError("size must be a non-negative integer.")
        size = int(size)
        if size < 0:
            raise ValueError("size must be a non-negative integer.")
        if not batched:
            return [self.single_sample() for _ in range(size)]
        return self._sample_batched(size)

    def sample_given(self, x: Sequence[int]) -> int:
        """Sample from the Markov chain conditioned on a given value 'x'.

        Args:
            x (Sequence[int]): Sample from Markov chain conditioned on observing 'x'.

        Returns:
            Single sample transition from integer Markov chain.

        """
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag
        states = _checked_observation(
            x,
            num_values=n_val,
            lag=lag,
            label="integer Markov conditioning path",
        )
        idx = np.ravel_multi_index(states[-lag:], m_shape)

        return self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :])


class IntegerMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate transition, initial-state, and length statistics for Markov chains."""

    def __init__(
        self,
        num_values: int,
        lag: int,
        init_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create an accumulator for integer Markov-chain sufficient statistics.

        Args:
            lag (int): The lag for the Markov chain.
            init_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the initial
                distribution.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the length
                of the observed sequences.
            keys (Optional[str]): Optional key for merging sufficient statistics with compatible accumulators.
            name (Optional[str]): Optional accumulator name.

        Attributes:
            lag (int): The lag for the Markov chain.
            trans_count_map (Dict[Tuple[Sequence[int], int], float]): Dictionary for tracking transition counts.
            init_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the initial distribution. Should
                be a sequence compatible accumulator with support on the integers. Defaults to the NullAccumulator.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the length of the observed
                sequences. Should be a sequence compatible accumulator with support on the non-negative integers.
                Defaults to the NullAccumulator.
            initial_nobs (float): Effective weight accumulated for the initial child.
            length_nobs (float): Effective weight accumulated for the length child.
            keys (Optional[str]): Optional key for merging sufficient statistics with compatible accumulators.
            name (Optional[str]): Optional accumulator name.

            _init_rng (bool): True if accumulator random states have been initialized.
            _acc_rng (Optional[RandomState]): Random state for initializing the init accumulator.
            _len_rng (Optional[RandomState]): Random state for initializing the length accumulator.

        """
        self.num_values = _positive_integer(num_values, label="integer Markov accumulator num_values")
        self.lag = _positive_integer(lag, label="integer Markov accumulator lag")
        self.trans_count_map = dict()
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.init_accumulator = init_accumulator if init_accumulator is not None else NullAccumulator()
        self.initial_nobs = 0.0
        self.length_nobs = 0.0
        self.keys = keys

        self._acc_rng = None
        self._len_rng = None
        self._init_rng = False

    def update(self, x: Sequence[int], weight: float, estimate: IntegerMarkovChainDistribution | None) -> None:
        """Update sufficient statistics with a single weighted observation.

        Args:
            x (Sequence[int]): An observation from an integer Markov chain.
            weight (float): Observation weight.
            estimate (Optional[IntegerMarkovChainDistribution]): Optional previous estimate.

        Returns:
            None.

        """
        states = _checked_observation(
            x,
            num_values=self.num_values,
            lag=self.lag,
            label="integer Markov observation",
        )
        weight = _finite_nonnegative(weight, label="integer Markov observation weight")
        self.len_accumulator.update(len(states), weight, estimate.len_dist if estimate is not None else None)
        self.length_nobs += weight
        self.init_accumulator.update(states[: self.lag], weight, estimate.init_dist if estimate is not None else None)
        self.initial_nobs += weight

        for i in range(len(states) - self.lag):
            entry = (states[i : (i + self.lag)], states[i + self.lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0.0) + weight

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize accumulator random states from ``rng``.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _acc_rng and _len_rng.

        Returns:
            None.

        """
        seeds = rng.randint(maxrandint, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._len_rng = RandomState(seed=seeds[1])
        self._init_rng = True

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics from a single observation.

        Note: Calls _rng_initialize() to ensure consistency with seq_initialize() function.

        Args:
            x (Sequence[int]): An observation from an integer Markov chain.
            weight (float): Observation weight.
            rng (RandomState): RandomState for initializing sufficient statistics.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        states = _checked_observation(
            x,
            num_values=self.num_values,
            lag=self.lag,
            label="integer Markov observation",
        )
        weight = _finite_nonnegative(weight, label="integer Markov observation weight")
        self.len_accumulator.initialize(len(states), weight, self._len_rng)
        self.length_nobs += weight
        self.init_accumulator.initialize(states[: self.lag], weight, self._acc_rng)
        self.initial_nobs += weight

        for i in range(len(states) - self.lag):
            entry = (states[i : (i + self.lag)], states[i + self.lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0.0) + weight

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        weights: np.ndarray,
        estimate: IntegerMarkovChainDistribution | None,
    ) -> None:
        """Vectorized update of sufficient statistics from an encoded sequence of observations 'x'.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Actual observed sequence lengths.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            estimate (Optional[IntegerMarkovChainDistribution]): Optional previous estimate.

        Returns:
            None.

        """
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(seq_len),):
            raise ValueError("integer Markov weights must align with encoded observations.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("integer Markov weights must be finite and non-negative.")
        init_idx = np.asarray(init_idx, dtype=np.intp)
        seq_idx = np.asarray(seq_idx, dtype=np.intp)
        seq_cnt = np.bincount(
            np.asarray(u_seq_idx, dtype=np.intp),
            weights=weights[seq_idx],
            minlength=len(u_seq_values),
        )

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_update(
            init_enc, weights[init_idx], estimate.init_dist if estimate is not None else None
        )
        self.initial_nobs += float(weights[init_idx].sum())
        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist if estimate is not None else None)
        self.length_nobs += float(weights.sum())

    def seq_update_engine(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        weights: Any,
        estimate: IntegerMarkovChainDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident E-step: per-unique-transition counts are reduced on the active engine
        before being scattered into the sparse transition dict; the init/len children are routed
        through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x

        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        if weights_np.shape != (len(seq_len),):
            raise ValueError("integer Markov weights must align with encoded observations.")
        if np.any(~np.isfinite(weights_np)) or np.any(weights_np < 0.0):
            raise ValueError("integer Markov weights must be finite and non-negative.")
        w_eng = engine.asarray(weights_np)

        seq_cnt = np.asarray(
            engine.to_numpy(
                engine.bincount(
                    engine.asarray(np.asarray(u_seq_idx, dtype=np.int64)),
                    weights=w_eng[np.asarray(seq_idx, dtype=np.int64)],
                    minlength=len(u_seq_values),
                )
            ),
            dtype=np.float64,
        )

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        init_estimate = None if estimate is None else estimate.init_dist
        len_estimate = None if estimate is None else estimate.len_dist
        child_seq_update(
            self.init_accumulator, init_enc, w_eng[np.asarray(init_idx, dtype=np.int64)], init_estimate, engine
        )
        child_seq_update(self.len_accumulator, len_enc, w_eng, len_estimate, engine)
        self.initial_nobs += float(weights_np[np.asarray(init_idx, dtype=np.intp)].sum())
        self.length_nobs += float(weights_np.sum())

    def seq_initialize(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Vectorized initialization of sufficient statistics from an encoded sequence of observations in 'x'.

        Note: Calls _rng_initialize() to ensure consistency with seq_initialize() function.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Actual observed sequence lengths.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            rng (RandomState): RandomState for initializing sufficient statistics.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(seq_len),):
            raise ValueError("integer Markov weights must align with encoded observations.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("integer Markov weights must be finite and non-negative.")
        init_idx = np.asarray(init_idx, dtype=np.intp)
        seq_idx = np.asarray(seq_idx, dtype=np.intp)
        seq_cnt = np.bincount(
            np.asarray(u_seq_idx, dtype=np.intp),
            weights=weights[seq_idx],
            minlength=len(u_seq_values),
        )

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_initialize(init_enc, weights[init_idx], self._acc_rng)
        self.initial_nobs += float(weights[init_idx].sum())
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)
        self.length_nobs += float(weights.sum())

    def combine(self, suff_stat: IntegerMarkovChainStatistics) -> "IntegerMarkovChainAccumulator":
        """Combine another versioned fixed-support statistic into this accumulator.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerMarkovChainAccumulator: This accumulator after combination.

        """
        checked = _validate_statistics(
            suff_stat,
            num_values=self.num_values,
            lag=self.lag,
            path="IntegerMarkovChainAccumulator.combine",
        )
        for prefix, target, count in checked.transition_counts:
            key = (prefix, target)
            self.trans_count_map[key] = self.trans_count_map.get(key, 0.0) + count

        if checked.initial is not None:
            self.init_accumulator = self.init_accumulator.combine(checked.initial)
        self.initial_nobs += checked.initial_nobs

        if checked.length is not None:
            self.len_accumulator = self.len_accumulator.combine(checked.length)
        self.length_nobs += checked.length_nobs

        return self

    def value(self) -> IntegerMarkovChainStatistics:
        """Return copied child statistics and immutable canonical transition counts."""
        transitions = tuple(
            (prefix, target, _finite_nonnegative(count, label="integer Markov transition count"))
            for (prefix, target), count in sorted(self.trans_count_map.items())
        )
        return IntegerMarkovChainStatistics(
            1,
            self.num_values,
            self.lag,
            transitions,
            _finite_nonnegative(self.initial_nobs, label="integer Markov initial_nobs"),
            copy.deepcopy(self.init_accumulator.value()),
            _finite_nonnegative(self.length_nobs, label="integer Markov length_nobs"),
            copy.deepcopy(self.len_accumulator.value()),
        )

    def from_value(self, x: IntegerMarkovChainStatistics) -> "IntegerMarkovChainAccumulator":
        """Restore accumulator state from a versioned fixed-support statistic.

        Args:
            x: See above for details.

        Returns:
            IntegerMarkovChainAccumulator: This accumulator after restoration.

        """
        checked = _validate_statistics(
            x,
            num_values=self.num_values,
            lag=self.lag,
            path="IntegerMarkovChainAccumulator.from_value",
        )
        self.trans_count_map = {(prefix, target): count for prefix, target, count in checked.transition_counts}
        self.initial_nobs = checked.initial_nobs
        self.length_nobs = checked.length_nobs
        if checked.initial is not None:
            self.init_accumulator = self.init_accumulator.from_value(copy.deepcopy(checked.initial))
        if checked.length is not None:
            self.len_accumulator = self.len_accumulator.from_value(copy.deepcopy(checked.length))

        return self

    def scale(self, c: float) -> "IntegerMarkovChainAccumulator":
        """Scale numeric evidence while preserving support and schema metadata."""
        c = _finite_nonnegative(c, label="integer Markov statistic scale")
        self.trans_count_map = {key: count * c for key, count in self.trans_count_map.items()}
        self.initial_nobs *= c
        self.length_nobs *= c
        self.init_accumulator.scale(c)
        self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into keyed sufficient statistics.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = copy.deepcopy(self)

        self.init_accumulator.key_merge(stats_dict)
        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics from matching keyed values.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.init_accumulator.key_replace(stats_dict)
        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerMarkovChainDataEncoder":
        """Return a data encoder built from the child accumulators."""
        len_encoder = self.len_accumulator.acc_to_encoder()
        init_encoder = self.init_accumulator.acc_to_encoder()
        return IntegerMarkovChainDataEncoder(
            num_values=self.num_values,
            lag=self.lag,
            len_encoder=len_encoder,
            init_encoder=init_encoder,
        )


class IntegerMarkovChainAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer Markov-chain accumulators with child accumulator factories."""

    def __init__(
        self,
        num_values: int,
        lag: int,
        init_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a factory for integer Markov-chain accumulators.

        Args:
            lag (int): Length of lag in Markov chain.
            init_factory (Optional[StatisticAccumulatorFactory]): Optional factory for the
                init distribution. Should be compatible with sequences of integers.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional factory for the
                length of Markov chain sequence. Should have support on non-negative integers.
            keys (Optional[str]): Optional key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.
            name (Optional[str]): Optional accumulator name.

        Attributes:
            lag (int): Length of lag in Markov chain.
            init_factory (StatisticAccumulatorFactory): Factory for the init distribution.
                Should be compatible with sequences of integers. Defaults to NullAccumulatorFactory if None.
            len_factory (StatisticAccumulatorFactory): Factory for the length of Markov
                chain sequence. Requires support on non-negative integers. Defaults to NullAccumulatorFactory if None.
            key (Optional[str]): Optional key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.
            name (Optional[str]): Optional accumulator name.

        """
        self.num_values = _positive_integer(num_values, label="integer Markov factory num_values")
        self.lag = _positive_integer(lag, label="integer Markov factory lag")
        self.init_factory = init_factory if init_factory is not None else NullAccumulatorFactory()
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys
        self.name = name

    def make(self) -> "IntegerMarkovChainAccumulator":
        """Return a new integer Markov-chain accumulator."""
        init_acc = self.init_factory.make()
        len_acc = self.len_factory.make()
        return IntegerMarkovChainAccumulator(
            self.num_values,
            self.lag,
            init_acc,
            len_acc,
            keys=self.keys,
            name=self.name,
        )


class IntegerMarkovChainEstimator(ParameterEstimator):
    """Estimate integer Markov-chain transition probabilities and child models."""

    def __init__(
        self,
        num_values: int,
        lag: int = 1,
        init_estimator: ParameterEstimator | None = NullEstimator(),
        len_estimator: ParameterEstimator | None = NullEstimator(),
        init_dist: SequenceEncodableProbabilityDistribution | None = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for integer Markov-chain distributions.

        Args:
            num_values (int): Number of values in Markov chain support.
            lag (int): Length of conditional dependence.
            init_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object compatible with
                sequences of integers.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object compatible with the
                non-negative integers.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, init_dist is fixed and not
                estimated. Must be compatible with sequences of integers.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, len_dist is fixed and not
                estimated. Must be compatible with non-negative integers.
            pseudo_count (Optional[float]): Prior mass used to smooth transition probabilities during estimation.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Optional key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        Attributes:
            num_values (int): Number of values in Markov chain support.
            lag (int): Length of conditional dependence.
            init_estimator (ParameterEstimator): Optional ParameterEstimator object compatible with
                sequences of integers. Defaults to NullEstimator.
            len_estimator (ParameterEstimator): ParameterEstimator object compatible with the non-negative integers.
                Defaults to the NullEstimator.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, init_dist is fixed and not
                estimated. Must be compatible with sequences of integers.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, len_dist is fixed and not
                estimated. Must be compatible with non-negative integers.
            pseudo_count (Optional[float]): Prior mass used to smooth transition probabilities during estimation.
            name (Optional[str]): Optional name assigned to estimated distributions.
            key (Optional[str]): Optional key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        """
        self.num_values = _positive_integer(num_values, label="integer Markov estimator num_values")
        self.lag = _positive_integer(lag, label="integer Markov estimator lag")
        self.init_estimator = init_estimator if init_estimator is not None else NullEstimator()
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.init_dist = init_dist
        self.len_dist = len_dist
        self.pseudo_count = (
            None if pseudo_count is None else _finite_nonnegative(pseudo_count, label="integer Markov pseudo_count")
        )
        if self.init_dist is not None:
            if not isinstance(self.init_dist, SequenceEncodableProbabilityDistribution):
                raise TypeError("fixed init_dist must be a probability distribution.")
            if not supports(self.init_dist, Neutral):
                _validate_initial_child(
                    self.init_dist,
                    num_values=self.num_values,
                    lag=self.lag,
                )
        if self.len_dist is not None:
            if not isinstance(self.len_dist, SequenceEncodableProbabilityDistribution):
                raise TypeError("fixed len_dist must be a probability distribution.")
            if not supports(self.len_dist, Neutral):
                _validate_length_child(self.len_dist, lag=self.lag)
        self.name = name
        self.keys = keys

    def get_prior(self):
        """Returns the implicit per-row symmetric-Dirichlet prior ``pseudo_count`` induces, or None.

        ``pseudo_count`` smoothing (``cond_mat += pseudo_count`` before row-normalizing in
        :meth:`estimate`) is the exact MAP point estimate under an independent
        ``SymmetricDirichletDistribution(pseudo_count + 1)`` prior on each row of the conditional
        matrix -- see :meth:`model_log_density`. Lets
        :func:`mixle.inference.estimation.optimize` auto-detect the ``'map'`` objective instead of
        silently tracking plain (unpenalized) MLE for a pseudo-count-regularized fit.
        """
        if self.pseudo_count is None:
            return None
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

        return SymmetricDirichletDistribution(self.pseudo_count + 1.0, dim=self.num_values)

    def model_log_density(self, model: "IntegerMarkovChainDistribution") -> float:
        """Log-density of the fitted conditional matrix under the implicit row prior (see
        :meth:`get_prior`), plus ``init_dist``/``len_dist``'s own ``model_log_density`` when the
        estimator that actually fit them (rather than a fixed, caller-supplied distribution)
        exposes one. Returns ``0.0`` (a plain MLE fit) when ``pseudo_count`` is None.
        """
        rv = 0.0
        if self.pseudo_count is not None:
            from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

            # log_density takes each row directly (cond_dist has at most num_values**lag rows -- never
            # a hot loop, so the scalar path per row needs no vectorization here).
            prior = SymmetricDirichletDistribution(self.pseudo_count + 1.0)
            rv += float(sum(prior.log_density(row) for row in model.cond_dist))
        if self.init_dist is None:  # init_dist was fit by init_estimator, not a fixed caller-supplied one
            fn = getattr(self.init_estimator, "model_log_density", None)
            if callable(fn):
                term = fn(model.init_dist)
                if term is not None:
                    rv += float(term)
        if self.len_dist is None:  # likewise for len_dist
            fn = getattr(self.len_estimator, "model_log_density", None)
            if callable(fn):
                term = fn(model.len_dist)
                if term is not None:
                    rv += float(term)
        return rv

    def accumulator_factory(self) -> "IntegerMarkovChainAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        len_factory = self.len_estimator.accumulator_factory()
        init_factory = self.init_estimator.accumulator_factory()
        return IntegerMarkovChainAccumulatorFactory(
            self.num_values,
            self.lag,
            init_factory,
            len_factory,
            keys=self.keys,
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: IntegerMarkovChainStatistics,
    ) -> "IntegerMarkovChainDistribution":
        """Estimate an integer Markov-chain distribution from versioned sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations used in aggregation of 'suff_stat'.
            suff_stat: See above for details.

        Returns:
            IntegerMarkovChainDistribution object.

        """
        checked = _validate_statistics(
            suff_stat,
            num_values=self.num_values,
            lag=self.lag,
            path="IntegerMarkovChainEstimator.estimate",
        )
        if nobs is not None and not math.isclose(
            _finite_nonnegative(nobs, label="integer Markov estimate nobs"),
            checked.length_nobs,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("integer Markov estimate nobs must match statistic length_nobs.")
        lag = self.lag

        len_dist = (
            self.len_dist
            if self.len_dist is not None
            else self.len_estimator.estimate(checked.length_nobs, checked.length)
        )
        init_dist = (
            self.init_dist
            if self.init_dist is not None
            else self.init_estimator.estimate(checked.initial_nobs, checked.initial)
        )

        num_values = self.num_values
        cond_mat = np.zeros((num_values**lag, num_values), dtype=np.float64)
        for prefix, target, count in checked.transition_counts:
            row = np.ravel_multi_index(prefix, [num_values] * lag)
            cond_mat[row, target] = count

        if self.pseudo_count is not None:
            cond_mat += self.pseudo_count

        row_sum = cond_mat.sum(axis=1, keepdims=True)
        bad_rows = row_sum.flatten() == 0.0
        if np.any(bad_rows):
            cond_mat[bad_rows, :] = 1.0
            row_sum[bad_rows] = num_values
        cond_mat /= row_sum

        return IntegerMarkovChainDistribution(
            num_values,
            cond_mat,
            init_dist=init_dist,
            lag=lag,
            len_dist=len_dist,
            name=self.name,
            keys=self.keys,
        )


class IntegerMarkovChainDataEncoder(DataSequenceEncoder):
    """Encode integer-valued sequences for vectorized Markov-chain scoring."""

    def __init__(
        self,
        num_values: int,
        lag: int,
        init_encoder: DataSequenceEncoder = NullDataEncoder(),
        len_encoder: DataSequenceEncoder = NullDataEncoder(),
    ) -> None:
        """Create an encoder for integer Markov-chain observations.

        Args:
            lag (int): Integer valued length of lag.
            init_encoder (DataSequenceEncoder): Encoder for the initial lagged value.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for the length of observed sequences.

        Attributes:
            lag (int): Integer valued length of lag.
            init_encoder (DataSequenceEncoder): Encoder for the initial lagged value. Should be a
                DataSequenceEncoder for a Sequence of distribution with support on integers.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for the length of observed sequences. Should be
                a DataSequenceEncoder with support on the integers.

        """
        self.num_values = _positive_integer(num_values, label="integer Markov encoder num_values")
        self.lag = _positive_integer(lag, label="integer Markov encoder lag")
        self.init_encoder = init_encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        rv = "IntegerMarkovChainDataEncoder(len_encoder=" + str(self.len_encoder)
        rv += ",init_encoder=" + str(self.init_encoder)
        rv += ",num_values=" + str(self.num_values) + ",lag=" + str(self.lag) + ")"
        return rv

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder is equivalent to this encoder.

        Note: Must have equivalent init_encoder and len_encoder member attributes.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is an equivalent IntegerMarkovChainDataEncoder.

        """
        if isinstance(other, IntegerMarkovChainDataEncoder):
            c0 = other.init_encoder == self.init_encoder
            c1 = other.len_encoder == self.len_encoder
            c2 = self.lag == other.lag
            c3 = self.num_values == other.num_values
            if c0 and c1 and c2 and c3:
                return True
            else:
                return False
        else:
            return False

    def seq_encode(
        self, x: list[Sequence[int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any | None, Any | None]:
        """Encode iid observations from an integer Markov chain.

        The returned tuple contains:
            seq_len (ndarray[int]): Actual observed sequence lengths.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x (List[Sequence[int]]): Sequence of iid observations from integer markov chain distribution.

        Returns:
            See above for details.


        """
        lag = self.lag

        if not isinstance(x, (list, tuple)):
            raise TypeError("integer Markov batch must be a list or tuple of observations.")
        observations = [
            _checked_observation(
                value,
                num_values=self.num_values,
                lag=self.lag,
                label="integer Markov observation[%d]" % index,
            )
            for index, value in enumerate(x)
        ]
        lens = np.asarray([len(value) for value in observations], dtype=np.int32)
        lag_cnt = len(observations)
        step_cnt = int(np.maximum(lens - lag, 0).sum())

        init_entries = np.zeros(lag_cnt, dtype=object)
        seq_entries = np.zeros(step_cnt, dtype=object)

        init_idx = []
        seq_idx = []
        seq_len = []

        i0 = 0
        i1 = 0

        for i, xx in enumerate(observations):
            seq_len.append(len(xx))

            init_idx.append(i)
            init_entries[i0] = tuple(xx[:lag])
            i0 += 1

            for j in range(len(xx) - lag):
                seq_idx.append(i)
                seq_entries[i1] = (tuple(xx[j : (j + lag)]), xx[j + lag])
                i1 += 1

        if step_cnt:
            u_seq_values, u_seq_idx = np.unique(seq_entries, return_inverse=True)
            u_seq_values = np.asarray(u_seq_values, dtype=object)
            u_seq_idx = np.asarray(u_seq_idx, dtype=np.int32)
        else:
            u_seq_values = np.empty(0, dtype=object)
            u_seq_idx = np.empty(0, dtype=np.int32)

        init_idx = np.asarray(init_idx, dtype=np.int32)
        seq_idx = np.asarray(seq_idx, dtype=np.int32)
        seq_len = np.asarray(seq_len, dtype=np.int32)

        len_enc = self.len_encoder.seq_encode(seq_len)
        init_enc = self.init_encoder.seq_encode(init_entries)

        return seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc

    def row_count(self, x: Any) -> int:
        """Return the number of rows after validating integer-chain index geometry."""
        if not isinstance(x, tuple) or len(x) != 7:
            raise ValueError("integer Markov encoded data must be a seven-slot tuple.")
        lengths, init_idx, seq_idx, unique_idx, values, _, _ = x

        def integer_vector(value: Any, label: str) -> np.ndarray:
            raw = np.asarray(value)
            if raw.ndim != 1:
                raise ValueError("%s must be one-dimensional." % label)
            try:
                numeric = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("%s must contain integers." % label) from exc
            if (
                np.any(~np.isfinite(numeric))
                or np.any(np.floor(numeric) != numeric)
                or np.any(np.abs(numeric) > np.iinfo(np.intp).max)
            ):
                raise ValueError("%s must contain exact platform-range integers." % label)
            return np.asarray(numeric, dtype=np.intp)

        lengths = integer_vector(lengths, "integer Markov lengths")
        init_idx = integer_vector(init_idx, "integer Markov initial row indices")
        seq_idx = integer_vector(seq_idx, "integer Markov transition row indices")
        unique_idx = integer_vector(unique_idx, "integer Markov unique transition indices")
        values = np.asarray(values, dtype=object)
        if values.ndim != 1:
            raise ValueError("integer Markov encoded transition values must be one-dimensional.")
        n_rows = len(lengths)
        if np.any(lengths < self.lag):
            raise ValueError("integer Markov encoded lengths must be at least lag.")
        if len(init_idx) != n_rows or not np.array_equal(init_idx, np.arange(n_rows)):
            raise ValueError("integer Markov initial row indices must cover every encoded row once.")
        expected_transitions = int(np.asarray(lengths - self.lag, dtype=np.intp).sum())
        if len(seq_idx) != expected_transitions or len(unique_idx) != expected_transitions:
            raise ValueError("integer Markov transition indices disagree with encoded lengths.")
        if len(seq_idx) and (
            np.any(seq_idx < 0)
            or np.any(seq_idx >= n_rows)
            or np.any(unique_idx < 0)
            or np.any(unique_idx >= len(values))
        ):
            raise ValueError("integer Markov transition indices are outside the encoded layout.")
        observed_per_row = np.bincount(seq_idx, minlength=n_rows)
        if not np.array_equal(observed_per_row, lengths - self.lag):
            raise ValueError("integer Markov transition row counts disagree with encoded lengths.")
        for index, value in enumerate(values):
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("integer Markov transition value %d has invalid geometry." % index)
            prefix = _checked_observation(
                value[0],
                num_values=self.num_values,
                lag=self.lag,
                label="integer Markov transition value %d prefix" % index,
            )
            if len(prefix) != self.lag:
                raise ValueError("integer Markov transition prefixes must have exactly lag states.")
            _exact_state(
                value[1],
                num_values=self.num_values,
                label="integer Markov transition value %d target" % index,
            )
        return n_rows
