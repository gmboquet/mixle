"""Create, estimate, and sample from an integer step Bernoulli edit set distribution.

Defines the IntegerStepBernoulliEditDistribution, IntegerStepBernoulliEditSampler,
IntegerStepBernoulliEditAccumulatorFactory, IntegerStepBernoulliEditAccumulator,
IntegerStepBernoulliEditEstimator, and the IntegerStepBernoulliEditDataEncoder classes for use with
pysparkplug.

Data type: Tuple[Sequence[int], Sequence[int]]: An observation x = (x1, x2) is a pair of integer sets
(prev set, next set), each a subset of S = {0,1,2,...N-1}.

The density has the same form as the integer Bernoulli edit set distribution (see
pysp.stats.sets.int_edit_setdist): each integer k independently transitions in or out of the set with
probabilities p(k in x2 | k in x1), p(k in x2 | k not in x1), etc., and the previous set x1 follows an
init distribution,

    p(x1, x2) = P_init(x1) * prod_{k=0}^{N-1} p(k in/not-in x2 | k in/not-in x1).

The "step" variant differs only in estimation: after the per-element edit probabilities are computed, the
estimator fits a two-level step function to the addition probabilities p(present | missing) and the
removal probabilities p(missing | present), so that each element receives one of just two probability
levels (a high level for the top-ranked elements and a low level for the rest), chosen to maximize the
Bernoulli likelihood of the per-element estimates.

"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.combinator.null_dist import (
    NullAccumulator,
    NullAccumulatorFactory,
    NullDistribution,
    NullEstimator,
)
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.stats.sets.int_edit_setdist import IntegerBernoulliEditEnumerator
from pysp.utils.aliasing import MISSING, coalesce_alias

T = tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]
E1 = TypeVar("E1")  ## encoded type for init
E = tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], E1 | None]
SS1 = TypeVar("SS1")  ## suff-stat of init_dist


class IntegerStepBernoulliEditDistribution(SequenceEncodableProbabilityDistribution):
    """Step Bernoulli edit set distribution: each integer independently transitions in/out between two sets."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    def __init__(
        self,
        log_edit_pmat: Sequence[tuple[float, float]] | np.ndarray,
        init_dist: SequenceEncodableProbabilityDistribution | None = None,
        name: str | None = None,
    ) -> None:
        """IntegerStepBernoulliEditDistribution object defining edit probabilities between integer sets.

        Args:
            log_edit_pmat (Union[Sequence[Tuple[float, float]], np.ndarray]): num_vals by 2 (or 4) matrix of
                log-probabilities. With 2 columns, column 0 is log p(present | missing) and column 1 is
                log p(present | present); the missing-state columns are filled in by complement. With 4 columns,
                the columns are log p(missing | missing), log p(missing | present), log p(present | missing),
                log p(present | present).
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the previous set x[0].
                Should be compatible with Sequence[int] observations (e.g. IntegerBernoulliSetDistribution).
            name (Optional[str]): Set name to object instance.

        Attributes:
            name (Optional[str]): Name for object instance.
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
        """Returns string representation of IntegerStepBernoulliEditDistribution object."""
        s1 = repr(list(map(list, self.orig_log_edit_pmat)))
        s2 = repr(self.init_dist)
        s3 = repr(self.name)
        return "IntegerStepBernoulliEditDistribution(%s, init_dist=%s, name=%s)" % (s1, s2, s3)

    def density(self, x: T) -> float:
        """Density of the step Bernoulli edit set distribution at observation x.

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
                IntegerStepBernoulliEditDataEncoder.seq_encode().

        Returns:
            Numpy array of log-density values, one per encoded observation.

        """
        sz, idx, xs, ys, ym, init_enc = x
        rv = np.bincount(idx, weights=self.log_dvec[xs, ys], minlength=sz)
        rv += self.log_nsum

        rv += self.init_dist.seq_log_density(init_enc)

        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded step edit-set observations."""
        from pysp.stats.compute.backend import backend_seq_log_density

        sz, idx, xs, ys, ym, init_enc = x
        rv = engine.zeros(sz) + engine.asarray(self.log_nsum)

        if len(idx) > 0:
            contrib = engine.asarray(self.log_dvec)[engine.asarray(xs), engine.asarray(ys)]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)

        rv = rv + backend_seq_log_density(self.init_dist, init_enc, engine)
        return rv

    @classmethod
    def backend_stacked_params(
        cls, dists: Sequence["IntegerStepBernoulliEditDistribution"], engine: Any
    ) -> dict[str, Any]:
        """Return stacked step edit-set parameters for shared support and init policy."""
        from pysp.stats.compute.stacked import stacked_component_params

        num_vals = int(dists[0].num_vals)
        null_init_dist = isinstance(dists[0].init_dist, NullDistribution)
        if any(
            int(dist.num_vals) != num_vals or isinstance(dist.init_dist, NullDistribution) != null_init_dist
            for dist in dists
        ):
            raise ValueError(
                "Stacked IntegerStepBernoulliEditDistribution components require shared support and init policy."
            )

        init_route = None
        if not null_init_dist:
            try:
                init_route = stacked_component_params([dist.init_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "IntegerStepBernoulliEdit initial child %s is not stackable: %s"
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
        """Return an ``(n, k)`` matrix of step edit-set component log densities."""
        from pysp.stats.compute.stacked import stacked_component_log_density

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
        from pysp.stats.compute.stacked import (
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

    def sampler(self, seed: int | None = None) -> "IntegerStepBernoulliEditSampler":
        """Create an IntegerStepBernoulliEditSampler object from this distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerStepBernoulliEditSampler object.

        """
        return IntegerStepBernoulliEditSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerStepBernoulliEditEstimator":
        """Create an IntegerStepBernoulliEditEstimator with matching num_vals.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics in estimation.

        Returns:
            IntegerStepBernoulliEditEstimator object.

        """
        return IntegerStepBernoulliEditEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> "IntegerStepBernoulliEditDataEncoder":
        """Returns an IntegerStepBernoulliEditDataEncoder object for encoding sequences of data."""
        return IntegerStepBernoulliEditDataEncoder(init_encoder=self.init_dist.dist_to_encoder())

    def enumerator(self) -> "IntegerStepBernoulliEditEnumerator":
        """Returns IntegerStepBernoulliEditEnumerator iterating set-pairs in descending probability order."""
        return IntegerStepBernoulliEditEnumerator(self)


class IntegerStepBernoulliEditEnumerator(IntegerBernoulliEditEnumerator):
    """Enumerates finite previous/next integer-set pairs for the step edit-set distribution."""


class IntegerStepBernoulliEditSampler(DistributionSampler):
    """IntegerStepBernoulliEditSampler object for drawing (prev set, next set) pairs from an
    IntegerStepBernoulliEditDistribution instance."""

    def __init__(self, dist: IntegerStepBernoulliEditDistribution, seed: int | None = None) -> None:
        """IntegerStepBernoulliEditSampler object for sampling from an IntegerStepBernoulliEditDistribution instance.

        Args:
            dist (IntegerStepBernoulliEditDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): RandomState object with seed set if passed in args.
            dist (IntegerStepBernoulliEditDistribution): Object instance to sample from.
            init_rng (DistributionSampler): Sampler for the previous set drawn from dist.init_dist.
            next_rng (RandomState): RandomState used for sampling the next set.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.init_rng = dist.init_dist.sampler(self.rng.randint(0, maxrandint))
        self.next_rng = np.random.RandomState(self.rng.randint(0, maxrandint))

    def sample(self, size: int | None = None) -> list[tuple[list[int], list[int]]] | tuple[list[int], list[int]]:
        """Draw iid (prev set, next set) observations from the distribution.

        Args:
            size (Optional[int]): Number of pairs to draw. If None, a single pair is returned.

        Returns:
            A (prev set, next set) tuple of integer lists if size is None, else a list of such tuples.

        """
        if size is None:
            temp = np.log(self.rng.rand(self.dist.num_vals))
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

    def sample_given(self, x: Sequence[Sequence[int]]) -> Sequence[int]:
        """Draw a next set conditioned on the last set in x.

        Args:
            x (Sequence[Sequence[int]]): History of integer sets; only the last set x[-1] is conditioned on.

        Returns:
            List of integers sampled for the next set.

        """
        temp = np.log(self.rng.rand(self.dist.num_vals))
        rv = np.zeros(self.dist.num_vals, dtype=bool)
        prev_ob = np.asarray(x[-1], dtype=int)

        rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
        rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

        return list(np.flatnonzero(rv))


class IntegerStepBernoulliEditAccumulator(SequenceEncodableStatisticAccumulator):
    """IntegerStepBernoulliEditAccumulator object for accumulating removed/added/kept counts from observed
    set pairs."""

    def __init__(
        self,
        num_vals: int,
        init_acc: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: str | None = None,
    ) -> None:
        """IntegerStepBernoulliEditAccumulator object for accumulating sufficient statistics from observed data.

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
        self.key = keys
        self.num_vals = num_vals
        self.init_acc = init_acc if init_acc is not None else NullAccumulator()
        self.tot_sum = 0.0

        self._acc_rng = None
        self._init_rng = False

    def update(self, x: T, weight: float, estimate: IntegerStepBernoulliEditDistribution | None) -> None:
        """Add weight to the removed/added/kept counts for the observed (prev set, next set) pair.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.
            weight (float): Weight for the observation.
            estimate (Optional[IntegerStepBernoulliEditDistribution]): Previous estimate passed to the init
                accumulator.

        """
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight

        if self.init_acc is not None:
            if estimate is not None:
                self.init_acc.update(x[0], weight, estimate.init_dist)
            else:
                self.init_acc.update(x[0], weight, None)

    def _rng_initialize(self, rng: RandomState) -> None:
        if not self._init_rng:
            self._acc_rng = RandomState(seed=rng.randint(maxrandint))
            self._init_rng = True

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with a weighted observation.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator passed to the init accumulator.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight
        self.init_acc.initialize(x[0], weight, rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: IntegerStepBernoulliEditDistribution | None) -> None:
        """Vectorized update of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerStepBernoulliEditDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (Optional[IntegerStepBernoulliEditDistribution]): Previous estimate passed to the init
                accumulator.

        """
        sz, idx, xs, ys, ym, init_enc = x

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[: len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[: len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[: len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_update(init_enc, weights, estimate.init_dist)

    def seq_update_engine(
        self, x: E, weights: Any, estimate: IntegerStepBernoulliEditDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of removed/added/kept edit counts (numpy or torch).

        The three weighted edit-type histograms are reduced on the active engine; the fixed-size
        count matrix is host bookkeeping. The init child is routed through the engine. Matches
        seq_update.
        """
        from pysp.stats.compute.backend import child_seq_update

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

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerStepBernoulliEditDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            rng (RandomState): Random number generator passed to the init accumulator.

        """
        sz, idx, xs, ys, ym, init_enc = x

        if not self._init_rng:
            self._rng_initialize(rng)

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[: len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[: len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[: len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_initialize(init_enc, weights, rng)

    def combine(self, suff_stat: tuple[np.ndarray, float, SS1 | None]) -> "IntegerStepBernoulliEditAccumulator":
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            This IntegerStepBernoulliEditAccumulator.

        """
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        self.init_acc.combine(suff_stat[2])

        return self

    def value(self) -> tuple[np.ndarray, float, Any | None]:
        """Returns the sufficient statistics: (edit counts, total weight, init suff stats)."""
        return self.pcnt, self.tot_sum, self.init_acc.value()

    def from_value(self, x: tuple[np.ndarray, float, SS1 | None]) -> "IntegerStepBernoulliEditAccumulator":
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            This IntegerStepBernoulliEditAccumulator.

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
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

        self.init_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the keyed statistics in stats_dict, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]

        self.init_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerStepBernoulliEditDataEncoder":
        """Returns an IntegerStepBernoulliEditDataEncoder object for encoding sequences of data."""
        return IntegerStepBernoulliEditDataEncoder(init_encoder=self.init_acc.acc_to_encoder())


class IntegerStepBernoulliEditAccumulatorFactory(StatisticAccumulatorFactory):
    """IntegerStepBernoulliEditAccumulatorFactory object for creating IntegerStepBernoulliEditAccumulator objects."""

    def __init__(
        self, num_vals: int, init_factory: StatisticAccumulatorFactory | None = None, keys: str | None = None
    ) -> None:
        """IntegerStepBernoulliEditAccumulatorFactory for creating IntegerStepBernoulliEditAccumulator objects.

        Args:
            num_vals (int): Number of integer values N in the set range.
            init_factory (Optional[StatisticAccumulatorFactory]): Factory for the previous-set accumulator.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            init_factory (StatisticAccumulatorFactory): Factory for the previous-set accumulator.
            num_vals (int): Number of integer values N in the set range.

        """
        self.keys = keys
        self.init_factory = init_factory if init_factory is not None else NullAccumulatorFactory()
        self.num_vals = num_vals

    def make(self) -> "IntegerStepBernoulliEditAccumulator":
        """Returns a new IntegerStepBernoulliEditAccumulator object."""
        return IntegerStepBernoulliEditAccumulator(self.num_vals, init_acc=self.init_factory.make(), keys=self.keys)


class IntegerStepBernoulliEditEstimator(ParameterEstimator):
    """IntegerStepBernoulliEditEstimator object for estimating an IntegerStepBernoulliEditDistribution from
    aggregated sufficient statistics, with a two-level step fit to the edit probabilities."""

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
        """IntegerStepBernoulliEditEstimator object for estimating integer step Bernoulli edit set distributions.

        Args:
            num_vals (int): Number of integer values N in the set range.
            init_estimator (Optional[ParameterEstimator]): Estimator for the previous set x[0].
            min_prob (float): Minimum probability for an edit transition.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): num_vals by 4 matrix of edit probabilities.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            num_vals (int): Number of integer values N in the set range.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): num_vals by 4 matrix of edit probabilities.
            name (Optional[str]): Set name for object instance.
            min_prob (float): Minimum probability for an edit transition.
            init_est (ParameterEstimator): Estimator for the previous set x[0].

        """
        self.num_vals = coalesce_alias("num_vals", num_vals, "num_values", num_values, default=MISSING)
        if pseudo_count is not None and pseudo_count < 0.0:
            raise ValueError("IntegerStepBernoulliEditEstimator requires a non-negative pseudo_count.")
        if min_prob is not None and (min_prob < 0.0 or min_prob > 0.5):
            raise ValueError("IntegerStepBernoulliEditEstimator requires 0 <= min_prob <= 0.5.")
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob
        self.init_est = init_estimator if init_estimator is not None else NullEstimator()

    def accumulator_factory(self) -> "IntegerStepBernoulliEditAccumulatorFactory":
        """Returns an IntegerStepBernoulliEditAccumulatorFactory for creating accumulator objects."""
        init_factory = self.init_est.accumulator_factory()
        return IntegerStepBernoulliEditAccumulatorFactory(self.num_vals, init_factory, self.keys)

    def __clip_prob(self, value: float) -> float:
        if self.min_prob is None or self.min_prob <= 0.0:
            return float(value)
        return float(np.clip(value, self.min_prob, 1.0 - self.min_prob))

    def __effective_step_counts(
        self,
        count_mat: np.ndarray,
        tot_sum: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s1 = count_mat[:, 0] + count_mat[:, 2]
        s0 = tot_sum - s1

        rem_success = count_mat[:, 0].astype(np.float64).copy()
        rem_trials = s1.astype(np.float64).copy()
        add_success = count_mat[:, 1].astype(np.float64).copy()
        add_trials = s0.astype(np.float64).copy()

        if self.pseudo_count is not None and self.pseudo_count > 0.0:
            p = self.pseudo_count
            if self.suff_stat is not None:
                s = np.asarray(self.suff_stat, dtype=np.float64)
                rem_success += p * s[:, 1]
                rem_trials += p * (s[:, 1] + s[:, 3])
                add_success += p * s[:, 2]
                add_trials += p * (s[:, 0] + s[:, 2])
            else:
                rem_success += p / 4.0
                rem_trials += p / 2.0
                add_success += p / 4.0
                add_trials += p / 2.0

        return rem_success, rem_trials, add_success, add_trials

    def __get_pqk(self, successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
        """Fit a two-level (p, q) step function to per-element binomial counts.

        Sorts the elements by empirical rate, then for each split rank k assigns probability p to
        the top-k elements and q to the rest (estimated by pooled successes over pooled trials on
        each side), and keeps the split maximizing the binomial log-likelihood. Elements with no
        trials are excluded from the split search and assigned the overall pooled rate (0.5 when
        no element has trials).

        Args:
            successes (np.ndarray): Per-element success counts.
            trials (np.ndarray): Per-element trial counts.

        Returns:
            Numpy array of per-element probabilities taking at most two distinct values, clipped to
            the estimator's probability floor when one is declared.

        """
        N = len(successes)
        obs = np.flatnonzero(trials > 0)
        M = len(obs)

        if M == 0:
            return np.full(N, 0.5)

        sidx = obs[np.argsort(-(successes[obs] / trials[obs]))]
        cs = np.cumsum(successes[sidx])
        ct = np.cumsum(trials[sidx])
        tot_s = cs[-1]
        tot_t = ct[-1]

        max_ll = -np.inf
        max_params = None
        for i in range(M):
            sh, th = cs[i], ct[i]
            p = self.__clip_prob(sh / th)
            v1 = (sh * np.log(p) if sh > 0 else 0.0) + ((th - sh) * np.log1p(-p) if th > sh else 0.0)
            if i + 1 < M:
                sl, tl = tot_s - sh, tot_t - th
                q = self.__clip_prob(sl / tl)
                v2 = (sl * np.log(q) if sl > 0 else 0.0) + ((tl - sl) * np.log1p(-q) if tl > sl else 0.0)
            else:
                q = 0.0
                v2 = 0.0
            ll = v1 + v2
            if ll > max_ll:
                max_params = (p, q, i)
                max_ll = ll

        p, q, k = max_params

        arr = np.full(N, self.__clip_prob(tot_s / tot_t))
        arr[sidx[: k + 1]] = p
        arr[sidx[k + 1 :]] = q
        return arr

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, float, SS1 | None]
    ) -> "IntegerStepBernoulliEditDistribution":
        """Estimate an IntegerStepBernoulliEditDistribution from aggregated sufficient statistics.

        Per-element edit probabilities are estimated as in the non-step edit estimator, then the
        addition and removal probabilities are each replaced by a two-level step-function fit.

        Args:
            nobs (Optional[float]): Unused (kept for protocol consistency).
            suff_stat (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            IntegerStepBernoulliEditDistribution object.

        """
        init_dist = self.init_est.estimate(None, suff_stat[2])
        count_mat, tot_sum, _ = suff_stat
        rem_success, rem_trials, add_success, add_trials = self.__effective_step_counts(count_mat, tot_sum)
        arr1 = self.__get_pqk(rem_success, rem_trials)
        arr2 = self.__get_pqk(add_success, add_trials)

        log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_pmat[:, 2] = np.log(arr2)
            log_pmat[:, 0] = np.log(1 - arr2)
            log_pmat[:, 1] = np.log(arr1)
            log_pmat[:, 3] = np.log(1 - arr1)

        return IntegerStepBernoulliEditDistribution(log_pmat, init_dist=init_dist, name=self.name)


class IntegerStepBernoulliEditDataEncoder(DataSequenceEncoder):
    """IntegerStepBernoulliEditDataEncoder object for encoding sequences of iid (prev set, next set) observations."""

    def __init__(self, init_encoder: DataSequenceEncoder) -> None:
        """IntegerStepBernoulliEditDataEncoder object for encoding (prev set, next set) observations.

        Args:
            init_encoder (DataSequenceEncoder): Encoder for the previous sets x[i][0].

        Attributes:
            init_encoder (DataSequenceEncoder): Encoder for the previous sets x[i][0].

        """
        self.init_encoder = init_encoder

    def __str__(self) -> str:
        """Returns string representation of IntegerStepBernoulliEditDataEncoder object."""
        return "IntegerBernoulliEditDataEncoder(init_encoder=" + str(self.init_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Checks if other object is an equivalent IntegerStepBernoulliEditDataEncoder."""
        if isinstance(other, IntegerStepBernoulliEditDataEncoder):
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
