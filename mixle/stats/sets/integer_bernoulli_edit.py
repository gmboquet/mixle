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
from mixle.stats.combinator.null_dist import NullEstimator
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
from mixle.stats.sets.integer_bernoulli_set import (
    IntegerBernoulliSetAccumulator,
    IntegerBernoulliSetDataEncoder,
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEstimator,
    _log_complement,
    _validated_min_prob,
    _validated_num_vals,
    _validated_observation,
    _validated_sample_size,
    _validated_weight,
    _validated_weights,
)
from mixle.utils.aliasing import MISSING, coalesce_alias

T = tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]
E1 = TypeVar("E1")  ## encoded type for init
E = tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], E1 | None]
SS1 = TypeVar("SS1")  ## suff-stat of init_dist

_COUNT_ATOL = 1.0e-8


class IntegerBernoulliEditFitError(RuntimeError):
    """Raised when edit statistics do not identify a valid transition law."""


def _default_init_distribution(num_vals: int) -> IntegerBernoulliSetDistribution:
    """Return a normalized point mass on the empty previous set."""
    return IntegerBernoulliSetDistribution(np.full(num_vals, -np.inf, dtype=np.float64))


def _validated_log_kernel(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate a two/four-column edit kernel and return canonical two/four forms."""
    try:
        supplied = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Integer Bernoulli-edit kernel must be numeric") from exc
    if supplied.ndim != 2 or supplied.shape[1] not in (2, 4):
        raise ValueError("Integer Bernoulli-edit kernel must have exact shape (N, 2) or (N, 4)")
    if np.any(np.isnan(supplied)) or np.any(np.isposinf(supplied)) or np.any(supplied > 0.0):
        raise ValueError("Integer Bernoulli-edit kernel must contain finite or -inf log probabilities")
    if supplied.shape[1] == 2:
        present = supplied.copy()
    else:
        pair_norm_missing = np.logaddexp(supplied[:, 0], supplied[:, 2])
        pair_norm_present = np.logaddexp(supplied[:, 1], supplied[:, 3])
        if not np.allclose(
            pair_norm_missing,
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.allclose(
            pair_norm_present,
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Integer Bernoulli-edit transition pairs must each sum to one")
        present = supplied[:, 2:4].copy()
    full = np.empty((len(present), 4), dtype=np.float64)
    full[:, 0] = _log_complement(present[:, 0])
    full[:, 1] = _log_complement(present[:, 1])
    full[:, 2:4] = present
    return present, full


def _validated_pair(value: Any, *, num_vals: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Integer Bernoulli-edit observations must be (previous, next) pairs")
    return (
        _validated_observation(value[0], num_vals=num_vals),
        _validated_observation(value[1], num_vals=num_vals),
    )


def _validated_encoded_edits(
    value: Any,
    *,
    num_vals: int,
) -> E:
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError("Encoded integer Bernoulli edits must contain six items")
    rows = _validated_num_vals(value[0])
    row_index = np.asarray(value[1])
    labels = np.asarray(value[2])
    edit_types = np.asarray(value[3])
    if (
        row_index.ndim != 1
        or labels.ndim != 1
        or edit_types.ndim != 1
        or row_index.shape != labels.shape
        or row_index.shape != edit_types.shape
    ):
        raise ValueError("Encoded integer Bernoulli-edit arrays have invalid geometry")
    if row_index.dtype.kind not in "iu" or labels.dtype.kind not in "iu" or edit_types.dtype.kind not in "iu":
        raise TypeError("Encoded integer Bernoulli-edit arrays must be integer arrays")
    row_index = row_index.astype(np.int64, copy=False)
    labels = labels.astype(np.int64, copy=False)
    edit_types = edit_types.astype(np.int64, copy=False)
    if np.any(row_index < 0) or np.any(row_index >= rows):
        raise ValueError("Encoded integer Bernoulli-edit row indices are out of range")
    if np.any(labels < 0) or np.any(labels >= num_vals):
        raise ValueError("Encoded integer Bernoulli-edit labels are out of range")
    if np.any(edit_types < 0) or np.any(edit_types > 2):
        raise ValueError("Encoded integer Bernoulli-edit types are out of range")
    pairs = tuple(zip(row_index.tolist(), labels.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Encoded integer Bernoulli-edit rows cannot repeat a support value")
    masks = value[4]
    if not isinstance(masks, (tuple, list)) or len(masks) != 3:
        raise ValueError("Encoded integer Bernoulli-edit type masks must contain three arrays")
    checked_masks = []
    for edit_type, mask in enumerate(masks):
        mask_array = np.asarray(mask)
        if mask_array.ndim != 1 or mask_array.dtype.kind not in "iu":
            raise TypeError("Encoded integer Bernoulli-edit type masks must be integer vectors")
        mask_array = mask_array.astype(np.int64, copy=False)
        expected = np.flatnonzero(edit_types == edit_type)
        if not np.array_equal(mask_array, expected):
            raise ValueError("Encoded integer Bernoulli-edit type masks do not match edit types")
        checked_masks.append(mask_array)
    return (
        rows,
        row_index,
        labels,
        edit_types,
        tuple(checked_masks),
        value[5],
    )


def _validated_edit_statistics(
    value: Any,
    *,
    num_vals: int,
) -> tuple[np.ndarray, float, Any]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Integer Bernoulli-edit sufficient statistics must contain three items")
    try:
        counts = np.asarray(value[0], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Integer Bernoulli-edit counts must be numeric") from exc
    if counts.shape != (num_vals, 3):
        raise ValueError("Integer Bernoulli-edit counts must have exact shape (%d, 3)" % num_vals)
    total = _validated_weight(value[1], label="total weight")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("Integer Bernoulli-edit counts must be finite and non-negative")
    tolerance = _COUNT_ATOL * max(1.0, total)
    present_trials = counts[:, 0] + counts[:, 2]
    missing_trials = total - present_trials
    if np.any(present_trials > total + tolerance) or np.any(counts[:, 1] > missing_trials + tolerance):
        raise ValueError("Integer Bernoulli-edit counts do not define feasible transition partitions")
    return counts.copy(), total, value[2]


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
        init_dist: SequenceEncodableProbabilityDistribution | None = None,
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
            log_nsum (float): Sum of log p(missing | missing) over non-forced values, the log-probability
                of the empty-set transition.
            log_dvec (np.ndarray): num_vals by 3 matrix of edit log-probabilities relative to
                log p(missing | missing); columns are (missing | present), (present | missing),
                (present | present). For forced values (see below) these are the true target
                log-probabilities directly, not relative to log p(missing | missing).
            forced (np.ndarray): Boolean mask, true where p_mat(missing | missing) == 0 (log = -inf) --
                a value that can never stay missing (e.g. p_mat(present | missing) == 1.0 exactly).

        """
        present_kernel, log_pmat = _validated_log_kernel(log_edit_pmat)
        num_vals = len(present_kernel)
        self.name = name
        self.keys = keys
        self.num_vals = num_vals
        self.init_dist = _default_init_distribution(num_vals) if init_dist is None else init_dist
        if supports(self.init_dist, Neutral):
            raise ValueError(
                "IntegerBernoulliEditDistribution requires a normalized init_dist; "
                "use conditional_log_density() or sampler().sample_given() for a "
                "supplied previous set"
            )
        init_num_vals = getattr(self.init_dist, "num_vals", num_vals)
        if int(init_num_vals) != num_vals:
            raise ValueError("Integer Bernoulli-edit init_dist support size must match the transition kernel")

        self.orig_log_edit_pmat = present_kernel
        self.log_edit_pmat = log_pmat
        # log_nsum/log_dvec are a vectorized trick: start from the "every value stayed missing" baseline
        # (log_nsum) and add a per-touched-value delta (log_dvec) that cancels the baseline and lands on
        # the true target log-probability. That cancellation only works in ordinary finite arithmetic --
        # for a *forced* value (p_mat(missing | missing) == 0, log = -inf; a legal, deterministic input,
        # e.g. p_mat(present | missing) == 1.0 exactly) the baseline is -inf, so a naive delta
        # (target - (-inf)) is +inf rather than something that cancels anything. Forced values are
        # excluded from log_nsum entirely, and their log_dvec row holds the true target log-probabilities
        # directly (no baseline to cancel against). log_density/seq_log_density separately short-circuit
        # to -inf whenever an observation leaves a forced value untouched (neither added, removed, nor
        # kept), since that implies it stayed missing, which is impossible for a forced value.
        self.forced = ~np.isfinite(self.log_edit_pmat[:, 0])
        self.log_nsum = self.log_edit_pmat[~self.forced, 0].sum()  # sum [ln p_mat(missing | missing)]
        self.log_dvec = self.log_edit_pmat[:, 1:] - self.log_edit_pmat[:, 0, None]
        self.log_dvec[self.forced, :] = self.log_edit_pmat[self.forced, 1:]  # forced: true targets, not deltas
        self.orig_log_edit_pmat.setflags(write=False)
        self.log_edit_pmat.setflags(write=False)
        self.forced.setflags(write=False)
        self.log_dvec.setflags(write=False)

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
        xx0, xx1 = _validated_pair(x, num_vals=self.num_vals)
        rv = self.conditional_log_density(xx0, xx1)
        if rv == -np.inf:
            return rv
        return float(rv + self.init_dist.log_density(xx0))

    def conditional_log_density(
        self,
        previous: Sequence[int] | np.ndarray,
        next_set: Sequence[int] | np.ndarray,
    ) -> float:
        """Return ``log p(next_set | previous)`` without an initial-law factor."""
        xx0 = _validated_observation(previous, num_vals=self.num_vals)
        xx1 = _validated_observation(next_set, num_vals=self.num_vals)
        if self.forced.any():
            touched = np.zeros(self.num_vals, dtype=bool)
            touched[xx0] = True
            touched[xx1] = True
            if np.any(self.forced & ~touched):
                return -np.inf  # a forced value (missing|missing impossible) stayed missing in both sets

        in10 = np.isin(xx1, xx0, invert=False)  # xx0 \cap xx1
        in01 = np.isin(xx0, xx1, invert=True)  # xx0 \cap xx1

        rv = self.log_nsum  # ln p_mat(missing | missing) for the empty set
        rv += np.sum(self.log_dvec[xx1[in10], 2])  # ln p_mat(present | present) same stuff that was there
        rv += np.sum(self.log_dvec[xx1[~in10], 1])  # ln p_mat(present | missing) new additions
        rv += np.sum(self.log_dvec[xx0[in01], 0])  # ln p_mat(missing | present) stuff to remove
        return float(rv)

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerBernoulliEditDataEncoder.seq_encode().

        Returns:
            Numpy array of log-density values, one per encoded observation.

        """
        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=self.num_vals,
        )
        rv = np.zeros(sz, dtype=np.float64)
        if len(idx):
            rv += np.bincount(
                idx,
                weights=self.log_dvec[xs, ys],
                minlength=sz,
            )
        rv += self.log_nsum

        if self.forced.any():
            touched = np.zeros((sz, self.num_vals), dtype=bool)
            touched[idx, xs] = True
            impossible = ~touched[:, self.forced].all(axis=1)
            rv[impossible] = -np.inf  # a forced value stayed missing in both sets for these rows

        rv += self.init_dist.seq_log_density(init_enc)

        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded integer edit-set observations."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=self.num_vals,
        )
        rv = engine.zeros(sz) + engine.asarray(self.log_nsum)

        if len(idx) > 0:
            contrib = engine.asarray(np.array(self.log_dvec, copy=True))[
                engine.asarray(xs),
                engine.asarray(ys),
            ]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)
        if self.forced.any():
            touched = np.zeros((sz, self.num_vals), dtype=bool)
            touched[idx, xs] = True
            impossible = np.any(self.forced[None, :] & ~touched, axis=1)
            rv = engine.where(
                engine.asarray(impossible),
                engine.asarray(np.full(sz, -np.inf)),
                rv,
            )

        rv = rv + backend_seq_log_density(self.init_dist, init_enc, engine)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerBernoulliEditDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer edit-set parameters for shared support and init policy."""
        from mixle.stats.compute.stacked import stacked_component_params

        if not dists:
            raise ValueError("Stacked IntegerBernoulliEditDistribution parameters require at least one component.")
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
            "__pysp_component_axis__": {
                "log_dvec": 2,
                "log_nsum": 0,
                "forced": 1,
            },
            "num_vals": num_vals,
            "log_dvec": engine.asarray(np.stack([dist.log_dvec for dist in dists], axis=2)),
            "log_nsum": engine.asarray([dist.log_nsum for dist in dists]),
            "forced": engine.asarray(np.stack([dist.forced for dist in dists], axis=1)),
            "init_route": init_route,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of integer edit-set component log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=int(params["num_vals"]),
        )
        rv = engine.zeros((sz, int(params["num_components"]))) + params["log_nsum"][None, :]

        if len(idx) > 0:
            contrib = params["log_dvec"][engine.asarray(xs), engine.asarray(ys), :]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)
        forced = np.asarray(engine.to_numpy(params["forced"]), dtype=bool)
        if np.any(forced):
            touched = np.zeros((sz, int(params["num_vals"])), dtype=bool)
            touched[idx, xs] = True
            impossible = np.any(
                forced[None, :, :] & ~touched[:, :, None],
                axis=1,
            )
            rv = engine.where(
                engine.asarray(impossible),
                engine.asarray(
                    np.full(
                        (sz, int(params["num_components"])),
                        -np.inf,
                    )
                ),
                rv,
            )

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

        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=int(params["num_vals"]),
        )
        num_components = int(params["num_components"])
        num_vals = int(params["num_vals"])
        weights_np = np.asarray(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            dtype=np.float64,
        )
        expected_shape = (sz, num_components)
        if weights_np.shape != expected_shape:
            raise ValueError("Stacked integer Bernoulli-edit weights must have exact shape %r" % (expected_shape,))
        if np.any(~np.isfinite(weights_np)) or np.any(weights_np < 0.0):
            raise ValueError("Stacked integer Bernoulli-edit weights must be finite and non-negative")
        ww = engine.asarray(weights_np)

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
        return IntegerBernoulliEditEstimator(
            self.num_vals,
            init_estimator=self.init_dist.estimator(
                pseudo_count=pseudo_count,
            ),
            pseudo_count=pseudo_count,
            suff_stat=(None if pseudo_count is None else np.exp(self.log_edit_pmat)),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerBernoulliEditDataEncoder":
        """Return a data encoder for integer Bernoulli edit observations."""
        return IntegerBernoulliEditDataEncoder(
            init_encoder=self.init_dist.dist_to_encoder(),
            num_vals=self.num_vals,
        )

    def enumerator(self) -> "IntegerBernoulliEditEnumerator":
        """Returns IntegerBernoulliEditEnumerator iterating set-pairs in descending probability order."""
        return IntegerBernoulliEditEnumerator(self)


class IntegerBernoulliEditEnumerator(DistributionEnumerator):
    """Enumerates finite previous/next integer-set pairs in descending probability order."""

    def __init__(self, dist: IntegerBernoulliEditDistribution) -> None:
        """Create an enumerator for integer Bernoulli-edit outputs.

        Previous sets are pulled from the normalized ``init_dist``. For each previous set,
        the conditional next-set support is an independent two-choice product.

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
        return iter(child_enumerator(self.dist.init_dist, "IntegerBernoulliEditDistribution.init_dist"))

    def _valid_prev(self, value: Any) -> list[int] | None:
        if not isinstance(value, (list, tuple, np.ndarray, set, frozenset)):
            return None
        try:
            checked = _validated_observation(
                value,
                num_vals=self.dist.num_vals,
            )
        except (TypeError, ValueError):
            return None
        return sorted(checked.tolist())

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
            prev_raw = self.init_rng.sample()
            prev_ob = _validated_observation(
                prev_raw,
                num_vals=self.dist.num_vals,
            )

            rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
            rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

            return list(prev_ob), list(np.flatnonzero(rv))
        else:
            checked_size = _validated_sample_size(size)
            rv = []
            for _ in range(checked_size):
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
        if not x:
            raise ValueError("sample_given requires at least one previous set")
        prev_ob = _validated_observation(
            x[-1],
            num_vals=self.dist.num_vals,
        )

        rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
        rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

        return list(np.flatnonzero(rv))


class IntegerBernoulliEditAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for removed, added, and kept counts from observed set pairs."""

    def __init__(
        self,
        num_vals: int,
        init_acc: SequenceEncodableStatisticAccumulator | None = None,
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
        self.num_vals = _validated_num_vals(num_vals)
        self.pcnt = np.zeros((self.num_vals, 3), dtype=np.float64)
        self.keys = keys
        self.init_acc = IntegerBernoulliSetAccumulator(self.num_vals) if init_acc is None else init_acc
        self.tot_sum = 0.0

    def update(self, x: T, weight: float, estimate: IntegerBernoulliEditDistribution | None) -> None:
        """Add weight to the removed/added/kept counts for the observed (prev set, next set) pair.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.
            weight (float): Weight for the observation.
            estimate (Optional[IntegerBernoulliEditDistribution]): Previous estimate passed to the init accumulator.

        """
        xx0, xx1 = _validated_pair(x, num_vals=self.num_vals)
        checked_weight = _validated_weight(weight)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.init_acc.update(
            xx0,
            checked_weight,
            estimate.init_dist if estimate is not None else None,
        )
        self.pcnt[xx0[to_rem], 0] += checked_weight
        self.pcnt[xx1[~to_add], 1] += checked_weight
        self.pcnt[xx1[to_add], 2] += checked_weight

        self.tot_sum += checked_weight

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with a weighted observation.

        Args:
            x (Tuple[Sequence[int], Sequence[int]]): Observed (prev set, next set) pair of integer sets.
            weight (float): Weight for the observation.
            rng (RandomState): Random number generator passed to the init accumulator.

        """
        xx0, xx1 = _validated_pair(x, num_vals=self.num_vals)
        checked_weight = _validated_weight(weight)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.init_acc.initialize(xx0, checked_weight, rng)
        self.pcnt[xx0[to_rem], 0] += checked_weight
        self.pcnt[xx1[~to_add], 1] += checked_weight
        self.pcnt[xx1[to_add], 2] += checked_weight

        self.tot_sum += checked_weight

    def seq_update(self, x: E, weights: np.ndarray, estimate: IntegerBernoulliEditDistribution | None) -> None:
        """Vectorized update of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerBernoulliEditDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (Optional[IntegerBernoulliEditDistribution]): Previous estimate passed to the init accumulator.

        """
        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=self.num_vals,
        )
        checked_weights = _validated_weights(weights, sz)
        aggregates = [
            np.bincount(
                xs[ym[col]],
                weights=checked_weights[idx[ym[col]]],
                minlength=self.num_vals,
            )
            for col in range(3)
        ]
        self.init_acc.seq_update(
            init_enc,
            checked_weights,
            None if estimate is None else estimate.init_dist,
        )
        for col, aggregate in enumerate(aggregates):
            self.pcnt[:, col] += aggregate
        self.tot_sum += float(checked_weights.sum())

    def seq_update_engine(
        self, x: E, weights: Any, estimate: IntegerBernoulliEditDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of removed/added/kept edit counts (numpy or torch).

        The three weighted edit-type histograms are reduced on the active engine; the fixed-size
        count matrix is host bookkeeping. The init child is routed through the engine. Matches
        seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=self.num_vals,
        )

        weights_np = np.asarray(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            dtype=np.float64,
        )
        weights_np = _validated_weights(weights_np, sz)
        w_eng = engine.asarray(weights_np)
        idxv = np.asarray(idx, dtype=np.int64)

        aggregates = []
        for col in range(3):
            sel = np.asarray(ym[col], dtype=np.int64)
            if sel.size == 0:
                aggregates.append(np.zeros(self.num_vals, dtype=np.float64))
            else:
                aggregates.append(
                    np.asarray(
                        engine.to_numpy(
                            engine.bincount(
                                engine.asarray(xs[sel]),
                                weights=w_eng[idxv[sel]],
                                minlength=self.num_vals,
                            )
                        ),
                        dtype=np.float64,
                    )
                )

        init_estimate = None if estimate is None else estimate.init_dist
        child_seq_update(self.init_acc, init_enc, w_eng, init_estimate, engine)
        for col, aggregate in enumerate(aggregates):
            self.pcnt[:, col] += aggregate
        self.tot_sum += float(engine.to_numpy(engine.sum(w_eng)))

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded observations.

        Args:
            x (E): Sequence encoded (prev set, next set) observations from
                IntegerBernoulliEditDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            rng (np.random.RandomState): Random number generator passed to the init accumulator.

        """
        sz, idx, xs, ys, ym, init_enc = _validated_encoded_edits(
            x,
            num_vals=self.num_vals,
        )
        checked_weights = _validated_weights(weights, sz)
        aggregates = [
            np.bincount(
                xs[ym[col]],
                weights=checked_weights[idx[ym[col]]],
                minlength=self.num_vals,
            )
            for col in range(3)
        ]
        self.init_acc.seq_initialize(init_enc, checked_weights, rng)
        for col, aggregate in enumerate(aggregates):
            self.pcnt[:, col] += aggregate
        self.tot_sum += float(checked_weights.sum())

    def combine(self, suff_stat: tuple[np.ndarray, float, SS1 | None]) -> "IntegerBernoulliEditAccumulator":
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            This IntegerBernoulliEditAccumulator.

        """
        counts, total, init_stats = _validated_edit_statistics(
            suff_stat,
            num_vals=self.num_vals,
        )
        combined_counts, combined_total, _ = _validated_edit_statistics(
            (self.pcnt + counts, self.tot_sum + total, None),
            num_vals=self.num_vals,
        )
        self.init_acc.combine(init_stats)
        self.pcnt = combined_counts
        self.tot_sum = combined_total

        return self

    def value(self) -> tuple[np.ndarray, float, Any | None]:
        """Returns the sufficient statistics: (edit counts, total weight, init suff stats)."""
        return self.pcnt.copy(), self.tot_sum, self.init_acc.value()

    def from_value(self, x: tuple[np.ndarray, float, SS1 | None]) -> "IntegerBernoulliEditAccumulator":
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[np.ndarray, float, Optional[SS1]]): Edit counts, total weight, and init suff stats.

        Returns:
            This IntegerBernoulliEditAccumulator.

        """
        counts, total, init_stats = _validated_edit_statistics(
            x,
            num_vals=self.num_vals,
        )
        self.init_acc.from_value(init_stats)
        self.pcnt = counts
        self.tot_sum = total
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
                stats_dict[self.keys] = (self.pcnt.copy(), self.tot_sum)

        self.init_acc.key_merge(stats_dict)

    def scale(self, c: float) -> "IntegerBernoulliEditAccumulator":
        """Scale edit and initial-law sufficient statistics."""
        checked_scale = _validated_weight(c, label="scale")
        self.init_acc.scale(checked_scale)
        self.pcnt *= checked_scale
        self.tot_sum *= checked_scale
        return self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the keyed statistics in stats_dict, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.pcnt, self.tot_sum, _ = _validated_edit_statistics(
                    (*stats_dict[self.keys], None),
                    num_vals=self.num_vals,
                )

        self.init_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerBernoulliEditDataEncoder":
        """Return a data encoder built from the previous-set accumulator."""
        return IntegerBernoulliEditDataEncoder(
            init_encoder=self.init_acc.acc_to_encoder(),
            num_vals=self.num_vals,
        )


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
        self.num_vals = _validated_num_vals(num_vals)
        self.init_factory = (
            IntegerBernoulliSetEstimator(self.num_vals).accumulator_factory() if init_factory is None else init_factory
        )

    def make(self) -> "IntegerBernoulliEditAccumulator":
        """Return a new integer Bernoulli edit accumulator."""
        return IntegerBernoulliEditAccumulator(self.num_vals, init_acc=self.init_factory.make(), keys=self.keys)


class IntegerBernoulliEditEstimator(ParameterEstimator):
    """Estimate integer Bernoulli edit distributions from aggregated sufficient statistics."""

    def __init__(
        self,
        num_vals: int = MISSING,
        init_estimator: ParameterEstimator | None = None,
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
        self.num_vals = _validated_num_vals(
            coalesce_alias(
                "num_vals",
                num_vals,
                "num_values",
                num_values,
                default=MISSING,
            )
        )
        self.keys = keys
        if pseudo_count is None:
            if suff_stat is not None:
                raise ValueError("Integer Bernoulli-edit prior probabilities require a pseudo-count")
            self.pseudo_count = None
            self.suff_stat = None
        else:
            self.pseudo_count = _validated_weight(
                pseudo_count,
                label="pseudo-count",
            )
            if suff_stat is None:
                self.suff_stat = None
            else:
                try:
                    prior = np.asarray(suff_stat, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Integer Bernoulli-edit prior kernel must be numeric") from exc
                if prior.shape != (self.num_vals, 4):
                    raise ValueError(
                        "Integer Bernoulli-edit prior kernel must have exact shape (%d, 4)" % self.num_vals
                    )
                if (
                    np.any(~np.isfinite(prior))
                    or np.any(prior < 0.0)
                    or np.any(prior > 1.0)
                    or not np.allclose(
                        prior[:, 0] + prior[:, 2],
                        1.0,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.allclose(
                        prior[:, 1] + prior[:, 3],
                        1.0,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise ValueError("Integer Bernoulli-edit prior rows must define stochastic transition pairs")
                self.suff_stat = prior.copy()
        self.name = name
        self.min_prob = _validated_min_prob(min_prob)
        self.init_est = (
            IntegerBernoulliSetEstimator(
                self.num_vals,
                pseudo_count=self.pseudo_count,
            )
            if init_estimator is None
            else init_estimator
        )

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
        count_mat, tot_sum, init_stats = _validated_edit_statistics(
            suff_stat,
            num_vals=self.num_vals,
        )
        prior_weight = 0.0 if self.pseudo_count is None else self.pseudo_count
        if tot_sum == 0.0 and prior_weight == 0.0:
            raise IntegerBernoulliEditFitError(
                "Integer Bernoulli-edit fitting requires positive observation or prior weight"
            )
        init_dist = self.init_est.estimate(None, init_stats)

        present_trials = count_mat[:, 0] + count_mat[:, 2]
        missing_trials = tot_sum - present_trials
        if self.suff_stat is None:
            prior_add = np.full(self.num_vals, 0.5)
            prior_keep = np.full(self.num_vals, 0.5)
        else:
            prior_add = self.suff_stat[:, 2]
            prior_keep = self.suff_stat[:, 3]

        add_denominator = missing_trials + prior_weight
        keep_denominator = present_trials + prior_weight
        add_probability = np.zeros(self.num_vals, dtype=np.float64)
        keep_probability = np.ones(self.num_vals, dtype=np.float64)
        add_identified = add_denominator > 0.0
        keep_identified = keep_denominator > 0.0
        add_probability[add_identified] = (
            count_mat[add_identified, 1] + prior_weight * prior_add[add_identified]
        ) / add_denominator[add_identified]
        keep_probability[keep_identified] = (
            count_mat[keep_identified, 2] + prior_weight * prior_keep[keep_identified]
        ) / keep_denominator[keep_identified]
        if self.min_prob > 0.0:
            upper = 1.0 - self.min_prob
            if upper == 1.0:
                upper = float(np.nextafter(1.0, 0.0))
            add_probability = np.clip(
                add_probability,
                self.min_prob,
                upper,
            )
            keep_probability = np.clip(
                keep_probability,
                self.min_prob,
                upper,
            )
        with np.errstate(divide="ignore"):
            present_kernel = np.log(np.column_stack((add_probability, keep_probability)))
        result = IntegerBernoulliEditDistribution(
            present_kernel,
            init_dist=init_dist,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "weighted-transition-frequency",
            "regularized": prior_weight > 0.0,
            "support_size": self.num_vals,
            "repairs": (),
        }
        return result


class IntegerBernoulliEditDataEncoder(DataSequenceEncoder):
    """Data encoder for iid ``(previous set, next set)`` observations."""

    def __init__(
        self,
        init_encoder: DataSequenceEncoder | None = None,
        num_vals: int | None = None,
    ) -> None:
        """Create an encoder for ``(previous set, next set)`` observations.

        Args:
            init_encoder (DataSequenceEncoder): Encoder for the previous sets x[i][0].

        Attributes:
            init_encoder (DataSequenceEncoder): Encoder for the previous sets x[i][0].

        """
        self.num_vals = None if num_vals is None else _validated_num_vals(num_vals)
        self.init_encoder = IntegerBernoulliSetDataEncoder(self.num_vals) if init_encoder is None else init_encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerBernoulliEditDataEncoder(init_encoder=%s, num_vals=%r)" % (self.init_encoder, self.num_vals)

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent integer Bernoulli-edit encoder."""
        if isinstance(other, IntegerBernoulliEditDataEncoder):
            return other.init_encoder == self.init_encoder and other.num_vals == self.num_vals
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
        try:
            rows = tuple(x)
        except TypeError as exc:
            raise TypeError("Integer Bernoulli-edit batches must be iterable") from exc
        idx = []
        xs = []
        ys = []
        pre = []

        for i, xx in enumerate(rows):
            if not isinstance(xx, (tuple, list)) or len(xx) != 2:
                raise ValueError("Integer Bernoulli-edit observations must be (previous, next) pairs")
            xx0 = _validated_observation(xx[0], num_vals=self.num_vals)
            xx1 = _validated_observation(xx[1], num_vals=self.num_vals)
            pre.append(xx0)

            to_add = np.isin(xx1, xx0, invert=False)
            to_rem = np.isin(xx0, xx1, invert=True)

            new_x = np.concatenate([xx0[to_rem], xx1[~to_add], xx1[to_add]])
            new_i = np.concatenate([[0] * np.sum(to_rem), [1] * np.sum(~to_add), [2] * np.sum(to_add)])

            idx.extend([i] * len(new_x))
            xs.extend(list(new_x))
            ys.extend(list(new_i))

        idx = np.asarray(idx, dtype=np.int64)
        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        ym = (np.flatnonzero(ys == 0), np.flatnonzero(ys == 1), np.flatnonzero(ys == 2))

        init_enc = self.init_encoder.seq_encode(pre)

        encoded = (len(rows), idx, xs, ys, ym, init_enc)
        if self.num_vals is None:
            return encoded
        return _validated_encoded_edits(encoded, num_vals=self.num_vals)

    def row_count(self, x: E) -> int:
        """Return the validated encoded batch row count."""
        if self.num_vals is None:
            if not isinstance(x, (tuple, list)) or len(x) != 6:
                raise ValueError("Encoded integer Bernoulli edits must contain six items")
            rows = _validated_num_vals(x[0])
        else:
            rows, _, _, _, _, _ = _validated_encoded_edits(
                x,
                num_vals=self.num_vals,
            )
        child_rows = self.init_encoder.row_count(x[5])
        if child_rows != rows:
            raise ValueError("Integer Bernoulli-edit child encoding row count does not match")
        return rows
