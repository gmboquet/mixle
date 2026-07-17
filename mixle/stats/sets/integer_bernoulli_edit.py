"""Integer Bernoulli edit distributions over pairs of finite sets.

Data type: Tuple[Sequence[int], Sequence[int]]: An observation x = (x1, x2) is a pair of integer sets
(prev set, next set), each a subset of S = {0,1,2,...N-1}.

Assume S = {0,1,2,...N-1} is a set of integers. The Bernoulli edit set distribution considers transitions between two
random subsets. That is, let X1 and X2 be a random subsets of unique integers from S, s.t. X1 and X2 have
at most N elements.

Consider observed subsets of S x1 and x2. The density is given by

    (1) p_mat(x2 | x1) = sum_{k=0}^{N-1} p_mat(k in x2 | k in x1) + p_mat(k in x2 | k not in x1) + p_mat(k not in x2 | k in x1)
        + p_mat(k not in x2 | k not in x1).
    (2) p_mat(x1,x2) = P_init(x1)*p_mat(x2|x1).

Note: In (1) only one of the summation terms in non-zero for a given value of k. In (2), P_init() is a distribution
defining probabilities for an integer 0<=k<N being in a set (Generally a BernoulliSetDistribution is a good choice).

"""

import heapq
import itertools
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import *
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
from mixle.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
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
    child_enumerator,
)
from mixle.utils.aliasing import MISSING, coalesce_alias

T = tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]
E1 = TypeVar("E1")  ## encoded type for init
E = tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], E1 | None]
SS1 = TypeVar("SS1")  ## suff-stat of init_dist


class IntegerBernoulliEditDistribution(SequenceEncodableProbabilityDistribution):
    """Bernoulli edit set distribution: each integer independently transitions in/out between two sets."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for integer Bernoulli-edit generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    def __init__(
        self,
        log_edit_pmat: Sequence[tuple[float, float]] | np.ndarray,
        init_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an integer Bernoulli edit distribution between sets.

        Args:
            log_edit_pmat (Union[Sequence[Tuple[float, float]], np.ndarray]): num_vals by 2 (or 4) matrix of
                log-probabilities. With 2 columns, column 0 is log p(present | missing) and column 1 is
                log p(present | present); the missing-state columns are filled in by complement. With 4 columns,
                the columns are log p(missing | missing), log p(missing | present), log p(present | missing),
                log p(present | present).
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the previous set x[0].
                Should be compatible with Sequence[int] observations (e.g. IntegerBernoulliSetDistribution).
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            name (Optional[str]): Optional distribution name.
            key (Optional[str]): Key for merging sufficient statistics.
            init_dist (SequenceEncodableProbabilityDistribution): Distribution for the previous set x[0].
            num_vals (int): Number of integer values N in the set range.
            orig_log_edit_pmat (np.ndarray): The log_edit_pmat passed at construction.
            log_edit_pmat (np.ndarray): num_vals by 4 matrix of edit log-probabilities (see log_edit_pmat above).
            log_nsum (float): Sum of log p(missing | missing), the log-probability of the empty-set transition.
            log_dvec (np.ndarray): num_vals by 3 matrix of edit log-probabilities relative to
                log p(missing | missing); columns are (missing | present), (present | missing),
                (present | present).

        """
        num_vals = len(log_edit_pmat)
        self.name = name
        self.keys = keys
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.num_vals = num_vals

        pmat = np.asarray(log_edit_pmat, dtype=np.float64).copy()
        if pmat.shape[1] == 2:
            log_pmat = np.zeros((num_vals, 4), dtype=np.float64)
            log_pmat[:, 0] = np.log1p(-np.exp(pmat[:, 0]))  # p_mat(missing | missing) = 1 - p_mat(present | missing)
            log_pmat[:, 1] = np.log1p(-np.exp(pmat[:, 1]))  # p_mat(missing | present) = 1 - p_mat(present | present)
            log_pmat[:, 2] = pmat[:, 0]  # p_mat(present | missing)
            log_pmat[:, 3] = pmat[:, 1]  # p_mat(present | present)
        else:
            log_pmat = pmat

        self.orig_log_edit_pmat = pmat
        self.log_edit_pmat = log_pmat
        self.log_nsum = self.log_edit_pmat[
            np.isfinite(self.log_edit_pmat[:, 0]), 0
        ].sum()  # sum [ln p_mat(missing | missing)]
        self.log_dvec = (
            self.log_edit_pmat[:, 1:] - self.log_edit_pmat[:, 0, None]
        )  # ln p_mat (?? | ??) - ln p_mat(missing | missing)

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = repr(list(map(list, self.orig_log_edit_pmat)))
        s2 = repr(self.init_dist)
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return "IntegerBernoulliEditDistribution(%s, init_dist=%s, name=%s, keys=%s)" % (s1, s2, s3, s4)

    def density(self, x: T) -> float:
        """Density of the Bernoulli edit set distribution at observation x.

        See log_density() for details.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.

        Returns:
            Density at observation x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Log-density of the joint observation (x[0], x[1]).

        Computes log p(x[1] | x[0]) by summing per-integer edit log-probabilities for kept,
        added, and removed elements, plus log p(x[0]) under init_dist.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.

        Returns:
            Log-density at observation x.

        """
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        in10 = np.isin(xx1, xx0, invert=False)  # xx0 \cap xx1
        in01 = np.isin(xx0, xx1, invert=True)  # xx0 \cap xx1

        yy = np.ones(len(xx1), dtype=int)
        yy[in10] = 2
        rv = self.log_nsum  # ln p_mat(missing | missing) for the empty set
        rv += np.sum(self.log_dvec[xx1[in10], 2])  # ln p_mat(present | present) same stuff that was there
        rv += np.sum(self.log_dvec[xx1[~in10], 1])  # ln p_mat(present | missing) new additions
        rv += np.sum(self.log_dvec[xx0[in01], 0])  # ln p_mat(missing | present) stuff to remove
        # rv = ln p_mat(x[1] | x[0])

        # rv = ln p_mat(x[1] | x[0]) + ln(p_mat(x[0]) = ln p_mat(x[0], x[1])
        rv += self.init_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerBernoulliEditDataEncoder.seq_encode().

        Returns:
            Numpy array of log-density values, one per encoded observation.

        """
        sz, idx, xs, ys, ym, init_enc = x
        rv = np.bincount(idx, weights=self.log_dvec[xs, ys], minlength=sz)
        rv += self.log_nsum
        rv += self.init_dist.seq_log_density(init_enc)

        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded integer edit-set observations."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, xs, ys, ym, init_enc = x
        rv = engine.zeros(sz) + engine.asarray(self.log_nsum)

        if len(idx) > 0:
            contrib = engine.asarray(self.log_dvec)[engine.asarray(xs), engine.asarray(ys)]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)

        rv = rv + backend_seq_log_density(self.init_dist, init_enc, engine)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerBernoulliEditDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer edit-set parameters for shared support and init policy."""
        from mixle.stats.compute.stacked import stacked_component_params

        num_vals = int(dists[0].num_vals)
        null_init_dist = supports(dists[0].init_dist, Neutral)
        if any(int(dist.num_vals) != num_vals or supports(dist.init_dist, Neutral) != null_init_dist for dist in dists):
            raise ValueError(
                "Stacked IntegerBernoulliEditDistribution components require shared support and init policy."
            )

        init_route = None
        if not null_init_dist:
            try:
                init_route = stacked_component_params([dist.init_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "IntegerBernoulliEdit initial child %s is not stackable: %s"
                    % (type(dists[0].init_dist).__name__, exc)
                )

        return {
            "__pysp_component_axis__": {"log_dvec": 2, "log_nsum": 0},
            "num_vals": num_vals,
            "log_dvec": engine.asarray(np.stack([dist.log_dvec for dist in dists], axis=2)),
            "log_nsum": engine.asarray([dist.log_nsum for dist in dists]),
            "init_route": init_route,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of integer edit-set component log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, idx, xs, ys, ym, init_enc = x
        rv = engine.zeros((sz, int(params["num_components"]))) + params["log_nsum"][None, :]

        if len(idx) > 0:
            contrib = params["log_dvec"][engine.asarray(xs), engine.asarray(ys), :]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)

        if params["init_route"] is not None:
            rv = rv + stacked_component_log_density(init_enc, params["init_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy ``(edit_counts, total_weight, init_stat)`` statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        sz, idx, xs, ys, ym, init_enc = x
        ww = engine.asarray(weights)
        num_components = int(params["num_components"])
        num_vals = int(params["num_vals"])

        if len(idx) > 0 and num_vals > 0:
            row_weights = ww[engine.asarray(idx)]
            zero_rows = row_weights * engine.asarray(0.0)
            by_type = []
            for edit_type in range(3):
                rows = []
                for value_index in range(num_vals):
                    mask = np.bitwise_and(xs == value_index, ys == edit_type)
                    rows.append(engine.sum(engine.where(engine.asarray(mask)[:, None], row_weights, zero_rows), axis=0))
                by_type.append(engine.stack(rows, axis=1))
            edit_counts = engine.stack(by_type, axis=2)
        else:
            edit_counts = engine.zeros((num_components, num_vals, 3))

        total_count = engine.sum(ww, axis=0)
        outer_estimators = tuple(getattr(estimator, "estimators", ()))

        init_estimators = tuple(getattr(component_est, "init_est", None) for component_est in outer_estimators)

        if params["init_route"] is None or all(isinstance(init_est, NullEstimator) for init_est in init_estimators):
            init_by_component = tuple(None for _ in range(num_components))
        else:
            init_estimator = StackedEstimatorView(init_estimators) if len(init_estimators) == num_components else None
            init_stats = stacked_component_sufficient_statistics(
                init_enc, ww, params["init_route"], engine, init_estimator
            )
            init_by_component = unstack_component_stats(init_stats, num_components)

        return tuple((edit_counts[i], total_count[i], init_by_component[i]) for i in range(num_components))

    def sampler(self, seed: int | None = None) -> "IntegerBernoulliEditSampler":
        """Create a sampler for this integer Bernoulli edit distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerBernoulliEditSampler: Sampler bound to this distribution.

        """
        return IntegerBernoulliEditSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerBernoulliEditEstimator":
        """Create an IntegerBernoulliEditEstimator with matching num_vals.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics in estimation.

        Returns:
            IntegerBernoulliEditEstimator: Estimator configured with matching support size.

        """
        return IntegerBernoulliEditEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "IntegerBernoulliEditDataEncoder":
        """Return a data encoder for integer Bernoulli edit observations."""
        return IntegerBernoulliEditDataEncoder(init_encoder=self.init_dist.dist_to_encoder())

    def enumerator(self) -> "IntegerBernoulliEditEnumerator":
        """Returns IntegerBernoulliEditEnumerator iterating set-pairs in descending probability order."""
        return IntegerBernoulliEditEnumerator(self)


class IntegerBernoulliEditEnumerator(DistributionEnumerator):
    """Enumerates finite previous/next integer-set pairs in descending probability order."""

    def __init__(self, dist: IntegerBernoulliEditDistribution) -> None:
        """Create an enumerator for integer Bernoulli-edit outputs.

        Previous sets are pulled from init_dist when it is enumerable; with a Null init_dist,
        all subsets of ``{0, ..., num_vals - 1}`` are considered with log-density 0. For each
        previous set, the conditional next-set support is an independent two-choice product.

        Args:
            dist (IntegerBernoulliEditDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        self._prev_stream = BufferedStream(self._prev_iterator())
        self._next_rank = 0
        self._heap: list[tuple[float, int, int]] = []
        self._heads: dict[int, tuple[Any, float]] = {}
        self._streams: dict[int, Iterator[tuple[Any, float]]] = {}
        self._counter = itertools.count()

    def _prev_iterator(self) -> Iterator[tuple[list[int], float]]:
        if supports(self.dist.init_dist, Neutral):
            choices = [(False, 0.0), (True, 0.0)]
            streams = [BufferedStream(iter(choices)) for _ in range(self.dist.num_vals)]

            def combine(flags: tuple[bool, ...]) -> list[int]:
                return [k for k, flag in enumerate(flags) if flag]

            return iter(ProductEnumerator(streams, combine=combine))
        return iter(child_enumerator(self.dist.init_dist, "IntegerBernoulliEditDistribution.init_dist"))

    def _valid_prev(self, value: Any) -> list[int] | None:
        if not isinstance(value, (list, tuple, np.ndarray, set, frozenset)):
            return None
        try:
            vals = sorted(set(int(v) for v in value))
        except (TypeError, ValueError):
            return None
        if any(v < 0 or v >= self.dist.num_vals for v in vals):
            return None
        return vals

    def _next_stream(self, prev: list[int], lp_prev: float) -> Iterator[tuple[Any, float]]:
        prev_set: set[int] = set(prev)
        streams = []
        for k in range(self.dist.num_vals):
            if k in prev_set:
                choices = [(False, float(self.dist.log_edit_pmat[k, 1])), (True, float(self.dist.log_edit_pmat[k, 3]))]
            else:
                choices = [(False, float(self.dist.log_edit_pmat[k, 0])), (True, float(self.dist.log_edit_pmat[k, 2]))]
            choices = [(flag, lp) for flag, lp in choices if lp > -np.inf]
            choices.sort(key=lambda u: -u[1])
            streams.append(BufferedStream(iter(choices)))

        def combine(flags: tuple[bool, ...]) -> tuple[list[int], list[int]]:
            return (list(prev), [k for k, flag in enumerate(flags) if flag])

        return iter(ProductEnumerator(streams, combine=combine, offset=lp_prev))

    def _pop(self) -> tuple[Any, float]:
        _, _, sid = heapq.heappop(self._heap)
        value, lp = self._heads.pop(sid)
        try:
            nxt = next(self._streams[sid])
            self._heads[sid] = nxt
            heapq.heappush(self._heap, (-nxt[1], next(self._counter), sid))
        except StopIteration:
            del self._streams[sid]
        return (value, lp)

    def __next__(self) -> tuple[Any, float]:
        while True:
            frontier = None
            while frontier is None:
                item = self._prev_stream.get(self._next_rank)
                if item is None:
                    break
                self._next_rank += 1
                prev = self._valid_prev(item[0])
                if prev is not None:
                    frontier = (prev, float(item[1]))

            if frontier is None:
                if self._heap:
                    return self._pop()
                raise StopIteration

            if self._heap and -self._heap[0][0] >= frontier[1]:
                self._next_rank -= 1
                return self._pop()

            prev, lp_prev = frontier
            sid = self._next_rank - 1
            stream = self._next_stream(prev, lp_prev)
            try:
                head = next(stream)
            except StopIteration:
                continue
            self._streams[sid] = stream
            self._heads[sid] = head
            heapq.heappush(self._heap, (-head[1], next(self._counter), sid))


class IntegerBernoulliEditSampler(DistributionSampler):
    """Sampler for ``(previous set, next set)`` pairs from an integer Bernoulli-edit distribution."""

    def __init__(self, dist: IntegerBernoulliEditDistribution, seed: int | None = None):
        """Create a sampler for an integer Bernoulli-edit distribution.

        Args:
            dist (IntegerBernoulliEditDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (IntegerBernoulliEditDistribution): Distribution to sample from.
            init_rng (DistributionSampler): Sampler for the previous set drawn from dist.init_dist.
            next_rng (RandomState): RandomState used for sampling the next set.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.init_rng = dist.init_dist.sampler(self.rng.randint(0, maxrandint))
        self.next_rng = np.random.RandomState(self.rng.randint(0, maxrandint))

    def sample(
        self, size: int | None = None, *, batched: bool = True
    ) -> list[tuple[list[int], list[int]]] | tuple[list[int], list[int]]:
        """Draw iid (prev set, next set) observations from the distribution.

        Args:
            size (Optional[int]): Number of pairs to draw. If None, a single pair is returned.

        Returns:
            A (prev set, next set) tuple of integer lists if size is None, else a list of such tuples.

        """
        if size is None:
            temp = self.rng.rand(self.dist.num_vals)
            temp = np.log(temp)
            rv = np.zeros(self.dist.num_vals, dtype=bool)
            prev_raw = self.init_rng.sample()  # a Null init_dist yields None -> empty previous set
            prev_ob = np.asarray([] if prev_raw is None else prev_raw, dtype=int)

            rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
            rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

            return list(prev_ob), list(np.flatnonzero(rv))
        else:
            rv = []
            for i in range(size):
                rv.append(self.sample())
            return rv

    def sample_given(self, x: Sequence[Sequence[int]]) -> list[int]:
        """Draw a next set conditioned on the last set in x.

        Args:
            x (Sequence[Sequence[int]]): History of integer sets; only the last set x[-1] is conditioned on.

        Returns:
            List of integers sampled for the next set.

        """
        temp = self.rng.rand(self.dist.num_vals)
        np.log(temp, out=temp)
        rv = np.zeros(self.dist.num_vals, dtype=bool)
        prev_ob = np.asarray(x[-1], dtype=int)

        rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
        rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

        return list(np.flatnonzero(rv))


class IntegerBernoulliEditAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for removed, added, and kept counts from observed set pairs."""

    def __init__(
        self,
        num_vals: int,
        init_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for integer Bernoulli-edit sufficient statistics.

        Args:
            num_vals (int): Number of integer values N in the set range.
            init_acc (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the previous set x[0].
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            pcnt (np.ndarray): num_vals by 3 matrix of weighted counts for removed, added, and kept elements.
            key (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of integer values N in the set range.
            init_acc (SequenceEncodableStatisticAccumulator): Accumulator for the previous set x[0].
            tot_sum (float): Sum of weights for observations.

        """
        self.pcnt = np.zeros((num_vals, 3), dtype=np.float64)
        self.keys = keys
        self.num_vals = num_vals
        self.init_acc = init_acc if init_acc is not None else NullAccumulator()
        self.tot_sum = 0.0

    def update(self, x: T, weight: float, estimate: IntegerBernoulliEditDistribution | None) -> None:
        """Add weight to the removed/added/kept counts for the observed (prev set, next set) pair.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.
            weight (float): Weight for the observation.
            estimate (Optional[IntegerBernoulliEditDistribution]): Previous estimate passed to the init accumulator.

        """
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight
        self.init_acc.update(x[0], weight, estimate.init_dist if estimate is not None else None)

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with a weighted observation.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator passed to the init accumulator.

        """
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight
        self.init_acc.initialize(x[0], weight, rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: IntegerBernoulliEditDistribution | None) -> None:
        """Vectorized update of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerBernoulliEditDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (Optional[IntegerBernoulliEditDistribution]): Previous estimate passed to the init accumulator.

        """
        sz, idx, xs, ys, ym, init_enc = x

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[: len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[: len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[: len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_update(init_enc, weights, None if estimate is None else estimate.init_dist)

    def seq_update_engine(
        self, x: E, weights: Any, estimate: IntegerBernoulliEditDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of removed/added/kept edit counts (numpy or torch).

        The three weighted edit-type histograms are reduced on the active engine; the fixed-size
        count matrix is host bookkeeping. The init child is routed through the engine. Matches
        seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, idx, xs, ys, ym, init_enc = x

        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w_eng = engine.asarray(weights_np)
        xsv = np.asarray(xs)
        idxv = np.asarray(idx, dtype=np.int64)

        for col in range(3):
            sel = np.asarray(ym[col], dtype=np.int64)
            if sel.size == 0:
                continue
            col_vals = xsv[sel].astype(np.int64)
            minlen = int(col_vals.max()) + 1
            agg = np.asarray(
                engine.to_numpy(engine.bincount(engine.asarray(col_vals), weights=w_eng[idxv[sel]], minlength=minlen)),
                dtype=np.float64,
            )
            self.pcnt[: len(agg), col] += agg

        self.tot_sum += float(engine.to_numpy(engine.sum(w_eng)))

        init_estimate = None if estimate is None else estimate.init_dist
        child_seq_update(self.init_acc, init_enc, w_eng, init_estimate, engine)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerBernoulliEditDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            rng (np.random.RandomState): Random number generator passed to the init accumulator.

        """
        sz, idx, xs, ys, ym, init_enc = x

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[: len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[: len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[: len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_initialize(init_enc, weights, rng)

    def combine(self, suff_stat: tuple[np.ndarray, float, SS1 | None]) -> "IntegerBernoulliEditAccumulator":
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            This IntegerBernoulliEditAccumulator.

        """
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        self.init_acc.combine(suff_stat[2])

        return self

    def value(self) -> tuple[np.ndarray, float, Any | None]:
        """Returns the sufficient statistics: (edit counts, total weight, init suff stats)."""
        return self.pcnt, self.tot_sum, self.init_acc.value()

    def from_value(self, x: tuple[np.ndarray, float, SS1 | None]) -> "IntegerBernoulliEditAccumulator":
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            This IntegerBernoulliEditAccumulator.

        """
        self.pcnt = x[0]
        self.tot_sum = x[1]
        self.init_acc.from_value(x[2])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator's statistics into stats_dict under its key, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                temp = stats_dict[self.keys]
                stats_dict[self.keys] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.keys] = (self.pcnt, self.tot_sum)

        self.init_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the keyed statistics in stats_dict, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.keys]

        self.init_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerBernoulliEditDataEncoder":
        """Return a data encoder built from the previous-set accumulator."""
        return IntegerBernoulliEditDataEncoder(init_encoder=self.init_acc.acc_to_encoder())


class IntegerBernoulliEditAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer Bernoulli edit accumulators."""

    def __init__(
        self, num_vals: int, init_factory: StatisticAccumulatorFactory | None = None, keys: str | None = None
    ) -> None:
        """Create a factory for integer Bernoulli edit accumulators.

        Args:
            num_vals (int): Number of integer values N in the set range.
            init_factory (Optional[StatisticAccumulatorFactory]): Factory for the previous-set accumulator.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.

        Attributes:
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.
            init_factory (StatisticAccumulatorFactory): Factory for the previous-set accumulator.
            num_vals (int): Number of integer values N in the set range.

        """
        self.keys = keys
        self.init_factory = init_factory if init_factory is not None else NullAccumulatorFactory()
        self.num_vals = num_vals

    def make(self) -> "IntegerBernoulliEditAccumulator":
        """Return a new integer Bernoulli edit accumulator."""
        return IntegerBernoulliEditAccumulator(self.num_vals, init_acc=self.init_factory.make(), keys=self.keys)


class IntegerBernoulliEditEstimator(ParameterEstimator):
    """Estimate integer Bernoulli edit distributions from aggregated sufficient statistics."""

    def __init__(
        self,
        num_vals: int = MISSING,
        init_estimator: ParameterEstimator | None = NullEstimator(),
        min_prob: float = 1.0e-128,
        pseudo_count: float | None = None,
        suff_stat: np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
        num_values: int = MISSING,
    ) -> None:
        """Create an estimator for integer Bernoulli edit set distributions.

        Args:
            num_vals (int): Number of integer values N in the set range.
            init_estimator (Optional[ParameterEstimator]): Estimator for the previous set x[0].
            min_prob (float): Minimum probability for an edit transition.
            pseudo_count (Optional[float]): Prior mass used to smooth edit probabilities during estimation.
            suff_stat (Optional[np.ndarray]): num_vals by 4 matrix of edit probabilities.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.

        Attributes:
            num_vals (int): Number of integer values N in the set range.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.
            pseudo_count (Optional[float]): Prior mass used to smooth edit probabilities during estimation.
            suff_stat (Optional[np.ndarray]): num_vals by 4 matrix of edit probabilities.
            name (Optional[str]): Optional name assigned to estimated distributions.
            min_prob (float): Minimum probability for an edit transition.
            init_est (ParameterEstimator): Estimator for the previous set x[0].

        """
        self.num_vals = coalesce_alias("num_vals", num_vals, "num_values", num_values, default=MISSING)
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob
        self.init_est = init_estimator if init_estimator is not None else NullEstimator()

    def accumulator_factory(self) -> "IntegerBernoulliEditAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        return IntegerBernoulliEditAccumulatorFactory(self.num_vals, self.init_est.accumulator_factory(), self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, float, SS1 | None]
    ) -> "IntegerBernoulliEditDistribution":
        """Estimate an IntegerBernoulliEditDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Unused (kept for protocol consistency).
            suff_stat (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            IntegerBernoulliEditDistribution object.

        """
        init_dist = self.init_est.estimate(None, suff_stat[2])
        count_mat, tot_sum, _ = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            p = self.pseudo_count
            s = self.suff_stat

            s1 = count_mat[:, 0] + count_mat[:, 2]
            s0 = tot_sum - s1

            log_s1 = np.log(s1 + p * (s[:, 1] + s[:, 3]))
            log_s0 = np.log(s0 + p * (s[:, 0] + s[:, 2]))

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + p * s[:, 0]) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:, 0] + p * s[:, 1]) - log_s1
            log_pmat[:, 2] = np.log(count_mat[:, 1] + p * s[:, 2]) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:, 2] + p * s[:, 3]) - log_s1

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count

            s1 = count_mat[:, 0] + count_mat[:, 2]
            s0 = tot_sum - s1

            log_s1 = np.log(s1 + p / 2.0)
            log_s0 = np.log(s0 + p / 2.0)

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 2] = np.log(count_mat[:, 1] + (p / 4.0)) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:, 2] + (p / 4.0)) - log_s1
            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + (p / 4.0)) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:, 0] + (p / 4.0)) - log_s1

        else:
            if suff_stat[1] == 0:
                log_pmat = np.zeros((self.num_vals, 4), dtype=np.float64) + np.log(0.5)

            elif (self.min_prob is not None) and (self.min_prob > 0):
                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                nz0 = s0 != 0
                nz1 = s1 != 0

                p0 = np.ones(self.num_vals, dtype=np.float64)
                p2 = np.zeros(self.num_vals, dtype=np.float64)
                p0[nz0] = np.maximum((s0[nz0] - count_mat[nz0, 1]) / s0[nz0], self.min_prob)
                p2[nz0] = np.maximum(count_mat[nz0, 1] / s0[nz0], self.min_prob)
                z0 = p0[nz0] + p2[nz0]
                p0[nz0] /= z0
                p2[nz0] /= z0

                p1 = np.zeros(self.num_vals, dtype=np.float64)
                p3 = np.ones(self.num_vals, dtype=np.float64)
                p1[nz1] = np.maximum(count_mat[nz1, 0] / s1[nz1], self.min_prob)
                p3[nz1] = np.maximum(count_mat[nz1, 2] / s1[nz1], self.min_prob)
                z1 = p1[nz1] + p3[nz1]
                p1[nz1] /= z1
                p3[nz1] /= z1

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                with np.errstate(divide="ignore"):
                    log_pmat[:, 0] = np.log(p0)
                    log_pmat[:, 1] = np.log(p1)
                    log_pmat[:, 2] = np.log(p2)
                    log_pmat[:, 3] = np.log(p3)

            else:
                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                nz0 = s0 != 0
                nz1 = s1 != 0

                p0 = np.ones(self.num_vals, dtype=np.float64)
                p2 = np.zeros(self.num_vals, dtype=np.float64)
                p0[nz0] = (s0[nz0] - count_mat[nz0, 1]) / s0[nz0]
                p2[nz0] = count_mat[nz0, 1] / s0[nz0]

                p1 = np.zeros(self.num_vals, dtype=np.float64)
                p3 = np.ones(self.num_vals, dtype=np.float64)
                p1[nz1] = count_mat[nz1, 0] / s1[nz1]
                p3[nz1] = count_mat[nz1, 2] / s1[nz1]

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                with np.errstate(divide="ignore"):
                    log_pmat[:, 0] = np.log(p0)
                    log_pmat[:, 1] = np.log(p1)
                    log_pmat[:, 2] = np.log(p2)
                    log_pmat[:, 3] = np.log(p3)

        return IntegerBernoulliEditDistribution(log_pmat, init_dist=init_dist, name=self.name, keys=self.keys)


class IntegerBernoulliEditDataEncoder(DataSequenceEncoder):
    """Data encoder for iid ``(previous set, next set)`` observations."""

    def __init__(self, init_encoder: DataSequenceEncoder) -> None:
        """Create an encoder for ``(previous set, next set)`` observations.

        Args:
            init_encoder (DataSequenceEncoder): Encoder for the previous sets x[i][0].

        Attributes:
            init_encoder (DataSequenceEncoder): Encoder for the previous sets x[i][0].

        """
        self.init_encoder = init_encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerBernoulliEditDataEncoder(init_encoder=" + str(self.init_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent integer Bernoulli-edit encoder."""
        if isinstance(other, IntegerBernoulliEditDataEncoder):
            return other.init_encoder == self.init_encoder
        else:
            return False

    def seq_encode(
        self, x: Sequence[T]
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], Any | None]:
        """Encode a sequence of iid (prev set, next set) observations for vectorized calculations.

        Return value 'rv' is a Tuple of length 6 containing:
            rv[0] (int): Number of observed pairs.
            rv[1] (np.ndarray): Observation index for each flattened edit entry.
            rv[2] (np.ndarray): Flattened integer values of edited elements.
            rv[3] (np.ndarray): Edit type per entry: 0 (removed), 1 (added), 2 (kept).
            rv[4] (Tuple[np.ndarray, np.ndarray, np.ndarray]): Indices of entries with each edit type.
            rv[5] (Optional[Any]): Sequence encoding of the previous sets from init_encoder.

        Args:
            x (Sequence[Tuple[Sequence[int], Sequence[int]]]): Sequence of iid (prev set, next set) observations.

        Returns:
            See 'rv' above.

        """
        idx = []
        xs = []
        ys = []
        pre = []

        for i, xx in enumerate(x):
            pre.append(xx[0])

            xx0 = np.asarray(xx[0], dtype=int)
            xx1 = np.asarray(xx[1], dtype=int)

            to_add = np.isin(xx1, xx0, invert=False)
            to_rem = np.isin(xx0, xx1, invert=True)

            new_x = np.concatenate([xx0[to_rem], xx1[~to_add], xx1[to_add]])
            new_i = np.concatenate([[0] * np.sum(to_rem), [1] * np.sum(~to_add), [2] * np.sum(to_add)])

            idx.extend([i] * len(new_x))
            xs.extend(list(new_x))
            ys.extend(list(new_i))

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)
        ys = np.asarray(ys, dtype=np.int32)
        ym = (np.flatnonzero(ys == 0), np.flatnonzero(ys == 1), np.flatnonzero(ys == 2))

        init_enc = self.init_encoder.seq_encode(pre)

        return len(x), idx, xs, ys, ym, init_enc
