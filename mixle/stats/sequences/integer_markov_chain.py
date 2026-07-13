"""Integer Markov-chain distributions with optional lagged transitions.

The data type: Sequence[int].

Consider a sequence of length n > 0 s.t. x = (x[0],x[1],...,x[n-1]). With lag > 0, we have the integer Markov chain
has a log-density given by:

    log(P(x)) = log(P_init(x[0:lag]) + sum_{j=0}^{n-1} log(p_mat(x[j + lag] | x[j], x[j+1],..,x[j+lag-1])) +
                    log(P_len(n)),

where P_len(n) is the density for the length distribution evaluated for length 'n', and P_init() is the density
for the initial distribution. If the sequence length is less than the lag, i.e. len(x) < lag, then

    log(P(x)) = log(P_len(n)).

Note: P_len() should be compatible with non-negative integers. P_init() must be compatible with sequences of ints.

"""

import heapq
import itertools
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

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

E1 = TypeVar("E1")  ## init encoding
E2 = TypeVar("E2")  ## len encoding
SS1 = TypeVar("SS1")  ## suff stat of init
SS2 = TypeVar("SS2")  ## suff-stat of length


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
        self.num_values = num_values
        self.cond_dist = np.asarray(cond_dist)
        self.lag = lag
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.name = name
        self.keys = keys

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

        Consider a sequence of length n > 0 s.t. x = (x[0],x[1],...,x[n-1]). With lag > 0, we have log-density
        given by:

            log(P(x)) = log(P_init(x[0:lag]) + sum_{j=0}^{n-1} log(p_mat(x[j + lag] | x[j], x[j+1],..,x[j+lag-1])) +
                log(P_len(n)),

        where P_len(n) is the density for the length distribution evaluated for length 'n', and P_init() is the density
        for the initial distribution. If the sequence length is less than the lag, i.e. len(x) < lag, then

            log(P(x)) = log(P_len(n)).

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            Log-density evaluated at x.

        """
        rv = 0.0
        lag = self.lag

        if len(x) >= lag:
            m_shape = [self.num_values] * lag
            rv += self.init_dist.log_density(x[:lag])

            for i in range(len(x) - lag):
                idx = np.ravel_multi_index(x[i : (i + lag)], m_shape)
                rv += np.log(self.cond_dist[idx, x[i + lag]])

        rv += self.len_dist.log_density(len(x))

        return rv

    def seq_log_density(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
    ) -> np.ndarray:
        """Vectorized evaluation of log-density at every observation in encoded sequence.

        See log_density() for details on likelihood evaluation.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
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
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x

        left_idx = [np.ravel_multi_index(u[0], [self.num_values] * self.lag) for u in u_seq_values]
        right_idx = np.asarray([u[1] for u in u_seq_values])
        temp_prob = np.log(self.cond_dist[left_idx, right_idx])
        temp_prob = temp_prob[u_seq_idx]

        rv = np.bincount(seq_idx, weights=temp_prob, minlength=len(seq_len))

        if self.init_dist is not None:
            rv[init_idx] += self.init_dist.seq_log_density(init_enc)

        if self.len_dist is not None and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def backend_seq_log_density(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        engine: Any,
    ) -> Any:
        """Engine-neutral vectorized log-density for grouped integer Markov-chain encodings."""
        from mixle.stats.compute.backend import backend_seq_log_density

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
        """Return per-component legacy ``(transition_counts, initial_stat, length_stat)`` statistics."""
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

        return tuple(
            (
                {
                    u_seq_values[value_index]: float(trans_counts[value_index, component])
                    for value_index in range(len(u_seq_values))
                },
                init_by_component[component],
                length_by_component[component],
            )
            for component in range(num_components)
        )

    def sampler(self, seed: int | None = None) -> "IntegerMarkovChainSampler":
        """Return a sampler for this integer Markov chain."""
        return IntegerMarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None):
        """Return an estimator initialized from this integer Markov chain."""
        init_est = self.init_dist.estimator()
        len_est = self.len_dist.estimator()

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
        return IntegerMarkovChainDataEncoder(lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder)

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

    def _valid_prefix(self, value: Any) -> tuple[int, ...] | None:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        if len(value) != self.dist.lag:
            return None
        try:
            prefix = tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return None
        if any(v < 0 or v >= self.dist.num_values for v in prefix):
            return None
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
        streams = [BufferedStream(iter(self._choices)) for _ in range(n)]
        return iter(ProductEnumerator(streams, combine=list, offset=lp_len))

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
                if prefix is None:
                    continue
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
        cnt = self.len_sampler.sample()
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag

        if cnt >= lag:
            rv = self.init_sampler.sample()  ## must return a list
            for i in range(lag, cnt):
                idx = np.ravel_multi_index(rv[-lag:], m_shape)
                rv.append(self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :]))
            return rv
        else:
            return []

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

        lengths = np.asarray(self.len_sampler.sample(size=size)).astype(int).reshape(-1)
        # Per-chain init draws, in order (byte-identical). Chains with length < lag yield [].
        seqs: list[list[int]] = []
        for cnt in lengths:
            if cnt >= lag:
                seqs.append(list(self.init_sampler.sample()))
            else:
                seqs.append([])

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
        if not batched:
            return [self.single_sample() for i in range(size)]
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
        idx = np.ravel_multi_index(x[-lag:], m_shape)

        return self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :])


class IntegerMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate transition, initial-state, and length statistics for Markov chains."""

    def __init__(
        self,
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
            max_value (int): Largest value encountered when accumulating sufficient statistics.
            keys (Optional[str]): Optional key for merging sufficient statistics with compatible accumulators.
            name (Optional[str]): Optional accumulator name.

            _init_rng (bool): True if accumulator random states have been initialized.
            _acc_rng (Optional[RandomState]): Random state for initializing the init accumulator.
            _len_rng (Optional[RandomState]): Random state for initializing the length accumulator.

        """
        self.lag = lag
        self.trans_count_map = dict()
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.init_accumulator = init_accumulator if init_accumulator is not None else NullAccumulator()
        self.max_value = -1
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
        lag = self.lag
        self.len_accumulator.update(
            max(len(x) - lag + 1, 0), weight, estimate.len_dist if estimate is not None else None
        )

        if len(x) >= lag:
            self.init_accumulator.update(x[:lag], weight, estimate.init_dist if estimate is not None else None)

        for i in range(len(x) - lag):
            entry = (tuple(x[i : (i + lag)]), x[i + lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight

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

        lag = self.lag

        if len(x) >= lag:
            self.len_accumulator.initialize(len(x) - lag, weight, self._len_rng)
            self.init_accumulator.initialize(x[:lag], weight, self._acc_rng)

        for i in range(len(x) - lag):
            entry = (tuple(x[i : (i + lag)]), x[i + lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        weights: np.ndarray,
        estimate: IntegerMarkovChainDistribution | None,
    ) -> None:
        """Vectorized update of sufficient statistics from an encoded sequence of observations 'x'.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
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

        seq_cnt = np.bincount(u_seq_idx, weights=weights[seq_idx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_update(
            init_enc, weights[init_idx], estimate.init_dist if estimate is not None else None
        )

        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist if estimate is not None else None)

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

    def seq_initialize(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, E1 | None, E2 | None],
        weights: np.ndarray,
        rng: RandomState,
    ) -> None:
        """Vectorized initialization of sufficient statistics from an encoded sequence of observations in 'x'.

        Note: Calls _rng_initialize() to ensure consistency with seq_initialize() function.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
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

        seq_cnt = np.bincount(u_seq_idx, weights=weights[seq_idx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_initialize(init_enc, weights[init_idx], self._acc_rng)
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

    def combine(
        self, suff_stat: tuple[dict[tuple[tuple[int, ...], int], float], SS1 | None, SS2 | None]
    ) -> "IntegerMarkovChainAccumulator":
        """Combine another sufficient-statistics tuple into this accumulator.

        Arg suff_stat is a Tuple of length 3 containing:
            suff_stat[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            suff_stat[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerMarkovChainAccumulator: This accumulator after combination.

        """
        for k, v in suff_stat[0].items():
            self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        if suff_stat[1] is not None:
            self.init_accumulator = self.init_accumulator.combine(suff_stat[1])

        if suff_stat[2] is not None:
            self.len_accumulator = self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> tuple[dict[tuple[tuple[int, ...], int], float], Any | None, Any | None]:
        """Returns sufficient statistics of integer Markov chain.

        Returned suff_stat is a Tuple of length 3 containing:
            suff_stat[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            suff_stat[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Returns:
            Tuple[Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]].

        """
        return self.trans_count_map, self.init_accumulator.value(), self.len_accumulator.value()

    def from_value(
        self, x: tuple[dict[tuple[tuple[int, ...], int], float], SS1 | None, SS2 | None]
    ) -> "IntegerMarkovChainAccumulator":
        """Restore accumulator state from aggregated sufficient statistics.

        Arg value 'x' is a Tuple of length 3 containing:
            x[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            x[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            x[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Args:
            x: See above for details.

        Returns:
            IntegerMarkovChainAccumulator: This accumulator after restoration.

        """
        self.trans_count_map = x[0]
        if x[1] is not None:
            self.init_accumulator = self.init_accumulator.from_value(x[1])

        if x[2] is not None:
            self.len_accumulator = self.len_accumulator.from_value(x[2])

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
                stats_dict[self.keys] = self

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
        return IntegerMarkovChainDataEncoder(lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder)


class IntegerMarkovChainAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer Markov-chain accumulators with child accumulator factories."""

    def __init__(
        self,
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
        self.lag = lag
        self.init_factory = init_factory if init_factory is not None else NullAccumulatorFactory()
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys
        self.name = name

    def make(self) -> "IntegerMarkovChainAccumulator":
        """Return a new integer Markov-chain accumulator."""
        init_acc = self.init_factory.make()
        len_acc = self.len_factory.make()
        return IntegerMarkovChainAccumulator(self.lag, init_acc, len_acc, keys=self.keys, name=self.name)


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
        self.num_values = num_values
        self.lag = lag
        self.init_estimator = init_estimator
        self.len_estimator = len_estimator
        self.init_dist = init_dist
        self.len_dist = len_dist
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "IntegerMarkovChainAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        len_factory = self.len_estimator.accumulator_factory()
        init_factory = self.init_estimator.accumulator_factory()
        return IntegerMarkovChainAccumulatorFactory(self.lag, init_factory, len_factory, keys=self.keys)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: tuple[dict[tuple[tuple[int, ...], int], float], SS1 | None, SS2 | None],
    ) -> "IntegerMarkovChainDistribution":
        """Estimate an integer Markov-chain distribution from aggregated sufficient statistics.

        Arg 'suff_stat' is a Tuple of length 3 containing:
            suff_stat[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            suff_stat[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Args:
            nobs (Optional[float]): Number of observations used in aggregation of 'suff_stat'.
            suff_stat: See above for details.

        Returns:
            IntegerMarkovChainDistribution object.

        """
        trans_count_map, init_ss, len_ss = suff_stat
        lag = self.lag

        len_dist = self.len_dist if self.len_dist is not None else self.len_estimator.estimate(None, len_ss)
        init_dist = self.init_dist if self.init_dist is not None else self.init_estimator.estimate(None, init_ss)

        # Honor the declared support: estimating from a data shard that happens not to contain every
        # value (common when streaming, or with rare symbols) must NOT shrink num_values, or a later shard
        # with an out-of-range value would break the index arithmetic. Fall back to the observed max only
        # if no support was declared.
        observed = (1 + max(max(max(u[0]), u[1]) for u in trans_count_map.keys())) if trans_count_map else 1
        num_values = max(self.num_values, observed) if self.num_values is not None else observed

        cond_mat = np.zeros((num_values**lag, num_values), dtype=np.float32)

        vv = list(trans_count_map.items())
        yidx = np.asarray([np.ravel_multi_index(u[0], [num_values] * lag) for u, _ in vv])
        xidx = np.asarray([u[1] for u, _ in vv])
        zidx = np.asarray([u[1] for u in vv])

        cond_mat[yidx, xidx] = zidx

        if self.pseudo_count is not None:
            cond_mat += self.pseudo_count

        row_sum = cond_mat.sum(axis=1, keepdims=True)
        bad_rows = row_sum.flatten() == 0.0
        if np.any(bad_rows):
            cond_mat[bad_rows, :] = 1.0
            row_sum[bad_rows] = num_values
        cond_mat /= row_sum

        return IntegerMarkovChainDistribution(
            num_values, cond_mat, init_dist=init_dist, lag=lag, len_dist=len_dist, name=self.name
        )


class IntegerMarkovChainDataEncoder(DataSequenceEncoder):
    """Encode integer-valued sequences for vectorized Markov-chain scoring."""

    def __init__(
        self,
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
        self.lag = lag
        self.init_encoder = init_encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        rv = "IntegerMarkovChainDataEncoder(len_encoder=" + str(self.len_encoder)
        rv += ",init_encoder=" + str(self.init_encoder) + ",lag=" + str(self.lag) + ")"
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
            if c0 and c1 and c2:
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
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
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

        cnt = len(x)
        lens = np.asarray([len(u) for u in x])
        lag_cnt = (lens >= lag).sum()
        step_cnt = np.maximum(lens - lag, 0).sum()

        init_entries = np.zeros(lag_cnt, dtype=object)
        seq_entries = np.zeros(step_cnt, dtype=object)

        init_idx = []
        seq_idx = []
        seq_len = []

        i0 = 0
        i1 = 0

        for i in range(len(x)):
            xx = x[i]
            seq_len.append(max(len(xx) - lag + 1, 0))

            if len(xx) < lag:
                continue

            init_idx.append(i)
            init_entries[i0] = tuple(xx[:lag])
            i0 += 1

            for j in range(len(xx) - lag):
                seq_idx.append(i)
                seq_entries[i1] = (tuple(xx[j : (j + lag)]), xx[j + lag])
                i1 += 1

        u_seq_values, u_seq_idx = np.unique(seq_entries, return_inverse=True)

        init_idx = np.asarray(init_idx, dtype=np.int32)
        seq_idx = np.asarray(seq_idx, dtype=np.int32)
        seq_len = np.asarray(seq_len, dtype=np.int32)

        len_enc = self.len_encoder.seq_encode(seq_len)
        init_enc = self.init_encoder.seq_encode(init_entries)

        return seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc
