"""Integer Bernoulli-set distributions over finite integer supports.

Let ``S = {0, 1, ..., N-1}`` be a finite integer support and let ``X`` be a random subset of ``S``.
The Bernoulli-set distribution gives each integer an independent inclusion probability ``p_k``.
The probability of an observed subset ``x`` is

    p(x) = prod_{k in x} p_k * prod_{k not in x} (1 - p_k).
"""

import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.aliasing import MISSING, coalesce_alias

_COUNT_ATOL = 1.0e-8


class IntegerBernoulliSetFitError(RuntimeError):
    """Raised when integer Bernoulli-set statistics do not identify a fit."""


def _validated_num_vals(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("num_vals must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("num_vals must be a non-negative integer") from exc
    if result < 0:
        raise ValueError("num_vals must be non-negative")
    return result


def _validated_min_prob(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("Integer Bernoulli-set min_prob must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("Integer Bernoulli-set min_prob must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0 or result > 0.5:
        raise ValueError("Integer Bernoulli-set min_prob must lie in [0, 0.5]")
    return result


def _validated_weight(value: Any, *, label: str = "weight") -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"Integer Bernoulli-set {label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Integer Bernoulli-set {label} must be a real scalar"
        ) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            f"Integer Bernoulli-set {label} must be finite and non-negative"
        )
    return result


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Integer Bernoulli-set weights must be numeric") from exc
    if result.shape != (rows,):
        raise ValueError(
            "Integer Bernoulli-set weights must have exact shape (%d,)" % rows
        )
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(
            "Integer Bernoulli-set weights must be finite and non-negative"
        )
    return result


def _validated_log_probabilities(value: Any, *, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if result.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)) or np.any(result > 0.0):
        raise ValueError(f"{label} must contain finite or -inf log probabilities")
    return result.copy()


def _log_complement(log_p: np.ndarray) -> np.ndarray:
    """Return ``log(1-exp(log_p))`` stably for validated log probabilities."""
    result = np.empty_like(log_p)
    cutoff = -np.log(2.0)
    low = log_p < cutoff
    with np.errstate(divide="ignore", invalid="ignore"):
        result[low] = np.log1p(-np.exp(log_p[low]))
        result[~low] = np.log(-np.expm1(log_p[~low]))
    return result


def _validated_observation(
    value: Any,
    *,
    num_vals: int | None,
) -> np.ndarray:
    try:
        labels = tuple(value)
    except TypeError as exc:
        raise TypeError("Integer Bernoulli-set observations must be iterable") from exc
    checked = []
    seen = set()
    for raw in labels:
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError(
                "Integer Bernoulli-set observations require exact integers"
            )
        try:
            label = operator.index(raw)
        except TypeError as exc:
            raise TypeError(
                "Integer Bernoulli-set observations require exact integers"
            ) from exc
        if label < 0 or (num_vals is not None and label >= num_vals):
            raise ValueError(
                "Integer Bernoulli-set observation is outside configured support"
            )
        if label in seen:
            raise ValueError(
                "Integer Bernoulli-set observations cannot contain duplicates"
            )
        seen.add(label)
        checked.append(label)
    return np.asarray(checked, dtype=np.int64)


def _validated_encoded_sets(
    value: Any,
    *,
    num_vals: int | None,
) -> tuple[int, np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Encoded integer Bernoulli sets must contain three items")
    rows = _validated_num_vals(value[0])
    row_index = np.asarray(value[1])
    labels = np.asarray(value[2])
    if (
        row_index.ndim != 1
        or labels.ndim != 1
        or row_index.shape != labels.shape
    ):
        raise ValueError("Encoded integer Bernoulli-set arrays have invalid geometry")
    if row_index.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
        raise TypeError("Encoded integer Bernoulli-set values must be integer arrays")
    row_index = row_index.astype(np.int64, copy=False)
    labels = labels.astype(np.int64, copy=False)
    if np.any(row_index < 0) or np.any(row_index >= rows):
        raise ValueError("Encoded integer Bernoulli-set row indices are out of range")
    if np.any(labels < 0) or (
        num_vals is not None and np.any(labels >= num_vals)
    ):
        raise ValueError("Encoded integer Bernoulli-set labels are out of range")
    pairs = tuple(zip(row_index.tolist(), labels.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError(
            "Encoded integer Bernoulli-set rows cannot contain duplicate labels"
        )
    return rows, row_index, labels


def _validated_statistics(
    value: Any,
    *,
    num_vals: int,
) -> tuple[np.ndarray, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "Integer Bernoulli-set sufficient statistics must contain two items"
        )
    try:
        counts = np.asarray(value[0], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Integer Bernoulli-set inclusion counts must be numeric"
        ) from exc
    if counts.shape != (num_vals,):
        raise ValueError(
            "Integer Bernoulli-set inclusion counts must have exact shape (%d,)"
            % num_vals
        )
    total = _validated_weight(value[1], label="total weight")
    tolerance = _COUNT_ATOL * max(1.0, total)
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError(
            "Integer Bernoulli-set inclusion counts must be finite and non-negative"
        )
    if np.any(counts > total + tolerance):
        raise ValueError(
            "Integer Bernoulli-set inclusion counts cannot exceed total weight"
        )
    return counts.copy(), total


def _validated_sample_size(value: Any) -> int:
    return _validated_num_vals(value)


class IntegerBernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """Distribution over finite sets of integer-valued Bernoulli outcomes."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the generic table-kernel capabilities for integer-set likelihoods."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Bernoulli-set statistics."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="integer_bernoulli_set",
            distribution_type=cls,
            parameters=(
                ParameterSpec("log_pvec", constraint="log_unit_interval_vector"),
                ParameterSpec("log_nvec", constraint="optional_log_unit_interval_vector", differentiable=False),
            ),
            statistics=(
                StatisticSpec("inclusion_counts"),
                StatisticSpec("total_weight"),
            ),
            support="finite_integer_set",
            differentiable=False,
        )

    def __init__(
        self,
        log_pvec: Sequence[float] | np.ndarray,
        log_nvec: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a Bernoulli set distribution on integer support ``[0, len(pvec))``.

        Args:
            log_pvec (Union[Sequence[float], np.ndarray]): Probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        Attributes:
            name (Optional[str]): Optional distribution name.
            log_pvec (np.ndarray): Probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            log_dvec (np.ndarray): Normalized probability for each integer value.
            log_nsum (float): Sum of normalized probabilities used for easily adding unobserved (missing) integer
                values in an observation.
            key (Optional[str]): Key for sharing sufficient statistics.

        """

        checked_log_p = _validated_log_probabilities(
            log_pvec,
            label="log_pvec",
        )
        num_vals = len(checked_log_p)
        derived_log_n = _log_complement(checked_log_p)
        if log_nvec is not None:
            checked_log_n = _validated_log_probabilities(
                log_nvec,
                label="log_nvec",
            )
            if checked_log_n.shape != checked_log_p.shape:
                raise ValueError("log_pvec and log_nvec must have equal lengths")
            pair_norm = np.logaddexp(checked_log_p, checked_log_n)
            if not np.allclose(pair_norm, 0.0, rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    "Each log_pvec/log_nvec pair must be complementary probabilities"
                )
        self.name = name
        self.num_vals = num_vals
        self.log_pvec = checked_log_p
        self.log_nvec = derived_log_n
        self.keys = keys
        self.log_dvec = self.log_pvec - self.log_nvec
        self.log_nsum = float(np.sum(self.log_nvec[np.isfinite(self.log_nvec)]))

        # An element with p_k = 1 is *required*: its log_dvec entry is +inf (log_nvec = -inf,
        # excluded from log_nsum). Treat it as forced membership (mirrors BernoulliSetDistribution):
        # zero contribution when present, -inf when an observation omits it -- never +inf.
        self.required = np.where(~np.isfinite(self.log_dvec) & (self.log_dvec > 0))[0]
        self.num_required = int(self.required.shape[0])
        if self.num_required:
            self.log_dvec = self.log_dvec.copy()
            self.log_dvec[self.required] = 0.0
        self.log_pvec.setflags(write=False)
        self.log_nvec.setflags(write=False)
        self.log_dvec.setflags(write=False)
        self.required.setflags(write=False)

    def __str__(self) -> str:
        s1 = repr(list(self.log_pvec))
        s2 = repr(list(self.log_nvec))
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return (
            "IntegerBernoulliSetDistribution(%s, log_nvec=%s, name=%s, keys=%s)"
            % (s1, s2, s3, s4)
        )

    def density(self, x: Sequence[int] | np.ndarray) -> float:
        """Return the probability density or mass at a single observation."""
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[int] | np.ndarray) -> float:
        """Return the log-density or log-mass at a single observation."""
        xx = _validated_observation(x, num_vals=self.num_vals)
        if self.num_required and not np.all(np.isin(self.required, xx)):
            return -np.inf
        return np.sum(self.log_dvec[xx]) + self.log_nsum

    def seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        sz, idx, xs = _validated_encoded_sets(x, num_vals=self.num_vals)
        rv = np.zeros(sz, dtype=np.float64)
        rv += np.bincount(idx, weights=self.log_dvec[xs], minlength=sz)
        rv += self.log_nsum
        if self.num_required:
            req_loc = np.isin(xs, self.required)
            req_cnt = np.bincount(idx[req_loc], minlength=sz)
            rv[req_cnt != self.num_required] = -np.inf
        return rv

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray], engine: Any) -> Any:
        """Engine-neutral log-density for encoded integer Bernoulli-set observations."""
        sz, idx, xs = _validated_encoded_sets(x, num_vals=self.num_vals)
        rv = engine.zeros(sz) + engine.asarray(self.log_nsum)
        if len(xs):
            log_dvec = engine.asarray(np.array(self.log_dvec, copy=True))
            rv = rv + engine.bincount(engine.asarray(idx), weights=log_dvec[engine.asarray(xs)], minlength=sz)
        if self.num_required:
            req_cnt = engine.zeros(sz)
            if len(xs):
                required_loc = np.isin(np.asarray(xs), self.required).astype(np.float64)
                req_cnt = engine.bincount(engine.asarray(idx), weights=engine.asarray(required_loc), minlength=sz)
            rv = engine.where(req_cnt != float(self.num_required), engine.asarray(np.full(sz, -np.inf)), rv)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerBernoulliSetDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer Bernoulli-set parameters for shared support size."""
        if not dists:
            raise ValueError(
                "Stacked IntegerBernoulliSetDistribution parameters require at least one component."
            )
        num_vals = int(dists[0].num_vals)
        if any(int(dist.num_vals) != num_vals for dist in dists):
            raise ValueError("Stacked IntegerBernoulliSetDistribution components require shared support size.")
        required = np.stack([np.isin(np.arange(num_vals), dist.required).astype(np.float64) for dist in dists], axis=1)
        return {
            "__pysp_component_axis__": {"log_dvec": 1, "log_nsum": 0, "required": 1, "num_required": 0},
            "num_vals": num_vals,
            "log_dvec": engine.asarray(np.stack([dist.log_dvec for dist in dists], axis=1)),
            "log_nsum": engine.asarray(np.asarray([dist.log_nsum for dist in dists], dtype=np.float64)),
            "required": engine.asarray(required),
            "num_required": engine.asarray(np.asarray([dist.num_required for dist in dists], dtype=np.float64)),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(
        cls, x: tuple[int, np.ndarray, np.ndarray], params: dict[str, Any], engine: Any
    ) -> Any:
        """Return an ``(n, k)`` matrix of integer Bernoulli-set log densities."""
        sz, idx, xs = _validated_encoded_sets(
            x,
            num_vals=int(params["num_vals"]),
        )
        rv = engine.zeros((sz, int(params["num_components"]))) + params["log_nsum"][None, :]
        if len(xs):
            contrib = params["log_dvec"][engine.asarray(xs), :]
            rv = engine.index_add(rv, engine.asarray(idx), contrib)
        if "num_required" in params and np.any(np.asarray(engine.to_numpy(params["num_required"])) != 0):
            req_cnt = engine.zeros((sz, int(params["num_components"])))
            if len(xs):
                req_loc = params["required"][engine.asarray(xs), :]
                req_cnt = engine.index_add(req_cnt, engine.asarray(idx), req_loc)
            rv = engine.where(req_cnt != params["num_required"][None, :], engine.asarray(-np.inf), rv)
        return rv

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[int, np.ndarray, np.ndarray], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(inclusion_counts, total_weight)`` statistics."""
        sz, idx, xs = _validated_encoded_sets(
            x,
            num_vals=int(params["num_vals"]),
        )
        weights_np = np.asarray(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            dtype=np.float64,
        )
        expected_shape = (sz, int(params["num_components"]))
        if weights_np.shape != expected_shape:
            raise ValueError(
                "Stacked integer Bernoulli-set weights must have exact shape %r"
                % (expected_shape,)
            )
        if np.any(~np.isfinite(weights_np)) or np.any(weights_np < 0.0):
            raise ValueError(
                "Stacked integer Bernoulli-set weights must be finite and non-negative"
            )
        ww = engine.asarray(weights_np)
        num_vals = int(params["num_vals"])
        if len(xs):
            row_weights = ww[engine.asarray(idx)]
            zero_rows = row_weights * engine.asarray(0.0)
            rows = []
            rel = engine.asarray(xs)
            for value_index in range(num_vals):
                mask = rel == engine.asarray(value_index)
                rows.append(engine.sum(engine.where(mask[:, None], row_weights, zero_rows), axis=0))
            pcnt = engine.stack(rows, axis=1)
        else:
            pcnt = engine.zeros((int(params["num_components"]), num_vals))
        return pcnt, engine.sum(ww, axis=0)

    def sampler(self, seed: int | None = None) -> "IntegerBernoulliSetSampler":
        """Return a sampler for drawing observations from this distribution."""
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerBernoulliSetEstimator":
        """Return an estimator for fitting this distribution from data."""
        return IntegerBernoulliSetEstimator(
            self.num_vals,
            pseudo_count=pseudo_count,
            suff_stat=(
                None if pseudo_count is None else np.exp(self.log_pvec)
            ),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return IntegerBernoulliSetDataEncoder(self.num_vals)

    def enumerator(self) -> "IntegerBernoulliSetEnumerator":
        """Returns IntegerBernoulliSetEnumerator iterating subsets in descending probability order."""
        return IntegerBernoulliSetEnumerator(self)


class IntegerBernoulliSetEnumerator(DistributionEnumerator):
    """Enumerate integer subsets in descending probability order."""

    def __init__(self, dist: IntegerBernoulliSetDistribution) -> None:
        """Enumerates subsets of {0,...,num_vals-1} in descending probability order.

        Membership is independent per integer: including k contributes log_dvec[k] to the
        log-density and excluding it contributes 0 (relative to the log_nsum offset). Each
        integer therefore yields a sorted two-choice stream, and subsets are enumerated with
        a best-first product search. Integers with p_k = 0 are exclude-only; required integers
        (p_k = 1) are include-only and contribute 0 to the log-density.

        Args:
            dist (IntegerBernoulliSetDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        required = {int(k) for k in dist.required}
        streams = []
        for k in range(dist.num_vals):
            d = dist.log_dvec[k]
            if k in required:
                choices = [(True, 0.0)]
            elif d == -np.inf:
                choices = [(False, 0.0)]
            elif d > 0.0:
                choices = [(True, float(d)), (False, 0.0)]
            else:
                choices = [(False, 0.0), (True, float(d))]
            streams.append(BufferedStream(iter(choices)))

        def combine(flags: tuple[bool, ...]) -> list[int]:
            return [k for k, f in enumerate(flags) if f]

        self._product = ProductEnumerator(streams, combine=combine, offset=float(dist.log_nsum))

    def __next__(self) -> tuple[list[int], float]:
        return next(self._product)


class IntegerBernoulliSetSampler(DistributionSampler):
    """Sample finite integer subsets by independent Bernoulli inclusion draws."""

    def __init__(self, dist: IntegerBernoulliSetDistribution, seed: int | None = None) -> None:
        """Create a sampler for an integer Bernoulli-set distribution.

        Args:
            dist (IntegerBernoulliSetDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (IntegerBernoulliSetDistribution): Distribution to sample from.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[Sequence[int]] | Sequence[int]:
        """Draw one subset or ``size`` iid subsets."""
        if size is None:
            log_u = np.log(self.rng.rand(self.dist.num_vals))
            return list(np.flatnonzero(log_u <= self.dist.log_pvec))
        else:
            checked_size = _validated_sample_size(size)
            rv = []
            for _ in range(checked_size):
                log_u = np.log(self.rng.rand(self.dist.num_vals))
                rv.append(list(np.flatnonzero(log_u <= self.dist.log_pvec)))
            return rv


class IntegerBernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate per-integer inclusion counts and total observation weight."""

    def __init__(self, num_vals: int, keys: str | None = None) -> None:
        """Create an accumulator for integer Bernoulli-set sufficient statistics.

        Args:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            pcnt (np.ndarray): Used for aggregating weighted counts of integers.
            key (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.
            tot_sum (float): Sum of weights for observations.

        """
        self.num_vals = _validated_num_vals(num_vals)
        self.pcnt = np.zeros(self.num_vals, dtype=np.float64)
        self.keys = keys
        self.tot_sum = 0.0

    def update(
        self, x: Sequence[int] | np.ndarray, weight: float, estimate: IntegerBernoulliSetDistribution | None
    ) -> None:
        """Update inclusion counts from one weighted subset."""
        xx = _validated_observation(x, num_vals=self.num_vals)
        checked_weight = _validated_weight(weight)
        self.pcnt[xx] += checked_weight
        self.tot_sum += checked_weight

    def initialize(self, x: Sequence[int] | np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize inclusion counts from one weighted subset."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[int, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: IntegerBernoulliSetDistribution | None,
    ) -> None:
        """Update inclusion counts from encoded subsets and observation weights."""
        sz, idx, xs = _validated_encoded_sets(x, num_vals=self.num_vals)
        checked_weights = _validated_weights(weights, sz)
        agg_cnt = np.bincount(
            xs,
            weights=checked_weights[idx],
            minlength=self.num_vals,
        )
        self.pcnt += agg_cnt
        self.tot_sum += float(checked_weights.sum())

    def seq_update_engine(
        self,
        x: tuple[int, np.ndarray, np.ndarray],
        weights: Any,
        estimate: IntegerBernoulliSetDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident accumulation of per-integer inclusion counts (numpy or torch).

        The weighted integer histogram is reduced on the active engine; the fixed-size count
        vector is host bookkeeping. Matches seq_update.
        """
        sz, idx, xs = _validated_encoded_sets(x, num_vals=self.num_vals)
        weights_np = np.asarray(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            dtype=np.float64,
        )
        weights_np = _validated_weights(weights_np, sz)
        w_eng = engine.asarray(weights_np)

        if xs.size > 0:
            agg_cnt = np.asarray(
                engine.to_numpy(
                    engine.bincount(
                        engine.asarray(xs),
                        weights=w_eng[np.asarray(idx, dtype=np.int64)],
                        minlength=self.num_vals,
                    )
                ),
                dtype=np.float64,
            )
            self.pcnt += agg_cnt

        self.tot_sum += float(engine.to_numpy(engine.sum(w_eng)))

    def seq_initialize(
        self, x: tuple[int, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize inclusion counts from encoded subsets."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "IntegerBernoulliSetAccumulator":
        """Merge inclusion counts and total observation weight."""
        counts, total = _validated_statistics(suff_stat, num_vals=self.num_vals)
        combined_counts, combined_total = _validated_statistics(
            (self.pcnt + counts, self.tot_sum + total),
            num_vals=self.num_vals,
        )
        self.pcnt = combined_counts
        self.tot_sum = combined_total
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return inclusion counts and total observation weight."""
        return self.pcnt.copy(), self.tot_sum

    def from_value(self, x: tuple[np.ndarray, float]) -> "IntegerBernoulliSetAccumulator":
        """Restore inclusion counts and total observation weight."""
        self.pcnt, self.tot_sum = _validated_statistics(
            x,
            num_vals=self.num_vals,
        )
        return self

    def scale(self, c: float) -> "IntegerBernoulliSetAccumulator":
        """Scale inclusion counts and total observation weight by a constant."""
        checked_scale = _validated_weight(c, label="scale")
        self.pcnt *= checked_scale
        self.tot_sum *= checked_scale
        return self

    def acc_to_encoder(self) -> "IntegerBernoulliSetDataEncoder":
        """Return the encoder compatible with Bernoulli-set sufficient statistics."""
        return IntegerBernoulliSetDataEncoder(self.num_vals)


class IntegerBernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for integer Bernoulli-set sufficient statistics."""

    def __init__(self, num_vals: int, keys: str | None = None) -> None:
        """IntegerBernoulliSetAccumulatorFactory for creating IntegerBernoulliSetAccumulator objects.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        Attributes:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        """
        self.keys = keys
        self.num_vals = _validated_num_vals(num_vals)

    def make(self) -> "IntegerBernoulliSetAccumulator":
        """Create an empty integer Bernoulli-set accumulator."""
        return IntegerBernoulliSetAccumulator(self.num_vals, keys=self.keys)


class IntegerBernoulliSetEstimator(ParameterEstimator):
    """Estimate per-integer Bernoulli inclusion probabilities from aggregate counts."""

    def __init__(
        self,
        num_vals: int = MISSING,
        min_prob: float = 1.0e-128,
        pseudo_count: float | None = None,
        suff_stat: np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
        num_values: int = MISSING,
    ) -> None:
        """Estimate integer Bernoulli set distributions from aggregated sufficient statistics.

        Args:
            num_vals (int): Number of values in integer range for the set.
            min_prob (float): Minimum probability for an integer in range of set dist.
            pseudo_count (Optional[float]): Prior mass used to smooth inclusion probabilities during estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.

        Attributes:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Key for merging sufficient statistics with compatible accumulators.
            pseudo_count (Optional[float]): Prior mass used to smooth inclusion probabilities during estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Optional name assigned to estimated distributions.
            min_prob (float): Minimum probability for an integer in range of set dist.

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
                raise ValueError(
                    "Integer Bernoulli-set prior probabilities require a pseudo-count"
                )
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
                    prior_probabilities = np.asarray(
                        suff_stat,
                        dtype=np.float64,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Integer Bernoulli-set prior probabilities must be numeric"
                    ) from exc
                if prior_probabilities.shape != (self.num_vals,):
                    raise ValueError(
                        "Integer Bernoulli-set prior probabilities must have exact "
                        "shape (%d,)" % self.num_vals
                    )
                if (
                    np.any(~np.isfinite(prior_probabilities))
                    or np.any(prior_probabilities < 0.0)
                    or np.any(prior_probabilities > 1.0)
                ):
                    raise ValueError(
                        "Integer Bernoulli-set prior probabilities must lie in [0, 1]"
                    )
                self.suff_stat = prior_probabilities.copy()
        self.name = name
        self.min_prob = _validated_min_prob(min_prob)

    def accumulator_factory(self) -> "IntegerBernoulliSetAccumulatorFactory":
        """Return a factory for integer Bernoulli-set sufficient-statistic accumulators."""
        return IntegerBernoulliSetAccumulatorFactory(self.num_vals, self.keys)

    def estimate(self, nobs: float | None, suff_stat: np.ndarray | None = None) -> "IntegerBernoulliSetDistribution":
        """Estimate an integer Bernoulli-set distribution from inclusion-count statistics."""
        counts, total = _validated_statistics(
            suff_stat,
            num_vals=self.num_vals,
        )
        prior_weight = 0.0 if self.pseudo_count is None else self.pseudo_count
        if total == 0.0 and prior_weight == 0.0:
            raise IntegerBernoulliSetFitError(
                "Integer Bernoulli-set fitting requires positive observation or prior weight"
            )
        if self.suff_stat is None:
            prior_probabilities = np.full(self.num_vals, 0.5, dtype=np.float64)
        else:
            prior_probabilities = self.suff_stat
        denominator = total + prior_weight
        probabilities = (
            counts + prior_weight * prior_probabilities
        ) / denominator
        if self.min_prob > 0.0:
            upper = 1.0 - self.min_prob
            if upper == 1.0:
                upper = float(np.nextafter(1.0, 0.0))
            probabilities = np.clip(probabilities, self.min_prob, upper)
        with np.errstate(divide="ignore"):
            log_pvec = np.log(probabilities)
        result = IntegerBernoulliSetDistribution(
            log_pvec,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "weighted-inclusion-frequency",
            "regularized": prior_weight > 0.0,
            "support_size": self.num_vals,
            "repairs": (),
        }
        return result


class IntegerBernoulliSetDataEncoder(DataSequenceEncoder):
    """Data encoder for iid integer Bernoulli-set observations."""

    def __init__(self, num_vals: int | None = None) -> None:
        self.num_vals = (
            None if num_vals is None else _validated_num_vals(num_vals)
        )

    def __str__(self) -> str:
        return "IntegerBernoulliSetDataEncoder(num_vals=%r)" % self.num_vals

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IntegerBernoulliSetDataEncoder)
            and self.num_vals == other.num_vals
        )

    def seq_encode(self, x: Sequence[Sequence[int]]) -> tuple[int, np.ndarray, np.ndarray]:
        """Encode sequences of iid observations for vectorized calculations.

        Returns 'rv':
            rv[0] (int): Total number of observations.
            rv[1] (np.ndarray): Index for flattened values of observations.
            rv[2] (np.ndarray): Flattened numpy array of integer values.

        Args:
            x (Sequence[Sequence[int]]): Sequence of integer set observations.

        Returns:
            See above for details.

        """
        try:
            rows = tuple(x)
        except TypeError as exc:
            raise TypeError(
                "Integer Bernoulli-set batches must be iterable"
            ) from exc
        row_index = []
        labels = []
        for row, observation in enumerate(rows):
            checked = _validated_observation(
                observation,
                num_vals=self.num_vals,
            )
            row_index.extend([row] * len(checked))
            labels.extend(checked.tolist())
        encoded = (
            len(rows),
            np.asarray(row_index, dtype=np.int64),
            np.asarray(labels, dtype=np.int64),
        )
        return _validated_encoded_sets(encoded, num_vals=self.num_vals)

    def row_count(self, x: tuple[int, np.ndarray, np.ndarray]) -> int:
        """Return the validated encoded batch row count."""
        rows, _, _ = _validated_encoded_sets(x, num_vals=self.num_vals)
        return rows
