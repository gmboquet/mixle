"""Integer-keyed multinomial distributions over a bounded support.

Each observation is a sequence of ``(integer_category, count)`` pairs over ``[min_val, max_val]``. Given
category probabilities ``p = (p_0, ..., p_K)`` and a trial-count distribution ``P_len(N)``, the model
scores the normalized count-vector log-mass

    log(P(x,N|p)) = log(N!) - sum_k log(x_k!) + sum_k x_k * log(p_k) + log(P_len(N))

where ``P_len(N)`` is a distribution for the number of trials. With the default neutral length factor,
the result is a normalized conditional mass for each supplied ``N`` but not a normalized joint law over
all lengths; sampling and joint enumeration therefore require a real trial-count distribution.

"""

import math
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

import mixle.utils.vector as vec
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
from mixle.stats.multivariate._multinomial_contracts import (
    canonical_integer_bag,
    exact_integer,
    finite_weight,
    log_coefficient,
    observation_weights,
    simplex,
)
from mixle.utils.aliasing import coalesce_alias

SS0 = TypeVar("SS0")
D = Sequence[tuple[int, float]]
E0 = TypeVar("E0")
E = tuple[int, np.ndarray, np.ndarray, np.ndarray, E0 | None]


def _exact_integer_array(value: Any, *, label: str, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1:
        raise ValueError("%s must be one-dimensional" % label)
    checked = [exact_integer(item, label=label, nonnegative=nonnegative) for item in raw.tolist()]
    try:
        return np.asarray(checked, dtype=np.int64)
    except OverflowError as exc:
        raise ValueError("%s values must fit signed 64-bit integers" % label) from exc


def _validate_integer_encoding(
    value: Any,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, Any | None, np.ndarray]:
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("integer multinomial encoding must be a (rows, indices, counts, values, lengths) tuple")
    rows = exact_integer(value[0], label="integer multinomial encoded row count", nonnegative=True)
    indices = _exact_integer_array(
        value[1],
        label="integer multinomial encoded row index",
        nonnegative=True,
    )
    counts = _exact_integer_array(
        value[2],
        label="integer multinomial encoded count",
        nonnegative=True,
    )
    categories = _exact_integer_array(
        value[3],
        label="integer multinomial encoded category",
    )
    if not (len(indices) == len(counts) == len(categories)):
        raise ValueError("integer multinomial encoded entry arrays must have equal length")
    if np.any(indices >= rows):
        raise ValueError("integer multinomial encoded row indices must be in [0, rows)")
    if len(indices) > 1 and np.any(indices[1:] < indices[:-1]):
        raise ValueError("integer multinomial encoded row indices must be non-decreasing")
    if len(indices) > 1:
        entries = np.column_stack((indices, categories))
        if len(np.unique(entries, axis=0)) != len(entries):
            raise ValueError("integer multinomial encodings must canonicalize duplicate row/category entries")
    totals_list = [0] * rows
    maximum = int(np.iinfo(np.int64).max)
    for row, count in zip(indices, counts):
        totals_list[int(row)] += int(count)
        if totals_list[int(row)] > maximum:
            raise ValueError("integer multinomial row totals exceed signed 64-bit range")
    totals = np.asarray(totals_list, dtype=np.int64)
    return rows, indices, counts, categories, value[4], totals


def _outside(stat_min: int, stat_counts: np.ndarray, min_val: int | None, max_val: int | None) -> bool:
    """Return whether a count statistic's support escapes a declared one.

    ``max_val`` is None when only the floor is pinned, in which case the support may grow upward.
    """
    if min_val is not None and stat_min < min_val:
        return True
    return max_val is not None and stat_min + len(stat_counts) - 1 > max_val


def _validate_integer_statistics(
    value: Any,
) -> tuple[int | None, np.ndarray | None, Any | None]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("integer multinomial sufficient statistic must be a (minimum, counts, length_stat) tuple")
    if value[1] is None:
        if value[0] is not None:
            raise ValueError("empty integer multinomial counts require minimum=None")
        return None, None, value[2]
    minimum = exact_integer(value[0], label="integer multinomial statistic minimum")
    try:
        counts = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("integer multinomial count statistic must be numeric") from exc
    if counts.ndim != 1 or counts.size == 0 or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("integer multinomial count statistic must be a nonempty finite non-negative vector")
    return minimum, counts.copy(), value[2]


class IntegerMultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Multinomial distribution over integer-keyed count maps."""

    def compute_capabilities(self):
        """Declare generated-compute support inherited from the trial-count distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.len_dist)
        return DistributionCapabilities(
            engine_ready=child.engine_ready,
            kernel_status="numpy_only" if child.numpy_only_reason else "generic_table",
            numpy_only_reason=child.numpy_only_reason,
        )

    def compute_declaration(self):
        """Return the generated-compute declaration for the integer multinomial."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = () if length is None else (length,)
        # The canonical exp-family map is the multinomial factor alone; only expose it when there is
        # no separate length (trials) distribution, so it matches seq_log_density exactly.
        exp_family = None
        if length is None:
            exp_family = ExponentialFamilySpec(
                sufficient_statistics=type(self).exp_family_sufficient_statistics,
                sufficient_statistics_from_params=type(self).exp_family_sufficient_statistics_from_params,
                natural_parameters=type(self).exp_family_natural_parameters,
                log_partition=type(self).exp_family_log_partition,
                base_measure_from_params=type(self).exp_family_base_measure_from_params,
                # T(x) is the per-category count vector and eta = log(p_vec); A = 0 and log h(x)
                # is the multinomial log coefficient on the support [min_val, min_val+K).
                # The category set depends on min_val/K so fixed_base=False; eta has -inf entries when
                # a category has p = 0, which makes the generic <eta, T> dot form NaN via 0*-inf for
                # zero-count categories, so runtime_scoring=False keeps scoring on the safe indexing
                # path while to_exponential_family still exposes the canonical map (valid where p > 0).
                fixed_base=False,
                runtime_scoring=False,
            )
        return DistributionDeclaration(
            name="integer_multinomial",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("min_val", constraint="integer", differentiable=False),
                ParameterSpec("p_vec", constraint="simplex_vector"),
            ),
            statistics=(
                StatisticSpec("min_val", kind="support_bound", additive=False, scales=False),
                StatisticSpec("count_vec", kind="count_vector"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="bounded_integer_count_vector",
            children=children,
            child_roles=("length",) if length is not None else (),
            exponential_family=exp_family,
            differentiable=False,
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return a shape-only fallback; category-aware count vectors come from ``..._from_params``."""
        return (engine.asarray(np.zeros(int(x[0]), dtype=np.float64)),)

    @staticmethod
    def exp_family_sufficient_statistics_from_params(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the per-category count vector ``T(x)`` of shape ``(sz, K)`` (counts of in-support values)."""
        sz, idx, cnt, val, _tcnt, _totals = _validate_integer_encoding(x)
        min_val = int(params["min_val"])
        k = int(np.asarray(engine.to_numpy(engine.asarray(params["p_vec"]))).reshape(-1).shape[0])
        stat = np.zeros((int(sz), k), dtype=np.float64)
        val = np.asarray(val)
        if val.shape[0] > 0:
            v = val - min_val
            keep = (v >= 0) & (v < k)
            rows = np.asarray(idx)[keep].astype(np.int64)
            np.add.at(stat, (rows, v[keep]), np.asarray(cnt, dtype=np.float64)[keep])
        return (engine.asarray(stat),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the natural parameter ``eta = log(p_vec)`` (one entry per category)."""
        return (engine.log(engine.asarray(params["p_vec"])),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the log partition ``A = 0`` (normalization is carried by ``eta = log p``)."""
        return engine.asarray(0.0)

    @staticmethod
    def exp_family_base_measure_from_params(x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return the multinomial log coefficient in support and ``-inf`` outside support."""
        sz, idx, cnt, val, _tcnt, totals = _validate_integer_encoding(x)
        min_val = int(params["min_val"])
        k = int(np.asarray(engine.to_numpy(engine.asarray(params["p_vec"]))).reshape(-1).shape[0])
        h = gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(cnt + 1.0),
            minlength=sz,
        )
        val = np.asarray(val)
        if val.shape[0] > 0:
            v = val - min_val
            out = ((v < 0) | (v >= k)) & (cnt != 0)
            if np.any(out):
                h[np.unique(np.asarray(idx)[out].astype(np.int64))] = -np.inf
        return engine.asarray(h)

    def __init__(
        self,
        min_val: int = 0,
        p_vec: list[float] = None,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        keys: str | None = None,
        prob_vec: list[float] = None,
    ) -> None:
        """Create an integer multinomial distribution.

        Args:
            min_val (int): Smallest integer category in the support.
            p_vec (Union[List[float], np.ndarray): Category probabilities. The length determines the number of
                supported integer values.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the multinomial trial
                count. ``NullDistribution`` disables the length contribution.
            name (Optional[str]): Optional distribution name used by higher-level estimators and diagnostics.
            keys (Optional[str]): Optional key used when sharing or merging sufficient statistics.

        Attributes:
            p_vec (ndarray): Probability assigned to each integer category for one trial.
            min_val (int): Smallest integer category.
            max_val (int): Largest integer category, computed as ``min_val + len(p_vec) - 1``.
            log_p_vec (ndarray): Natural logarithm of ``p_vec``.
            num_vals (int): Number of supported integer categories.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for the trial count.
            keys (Optional[str]): Key propagated to estimators for keyed statistic merging.
            name (Optional[str]): Optional distribution name.

        """
        super().__init__()
        p_vec = coalesce_alias("p_vec", p_vec, "prob_vec", prob_vec, required=False, default=None)
        if p_vec is None:
            raise ValueError("IntegerMultinomialDistribution requires a nonempty probability simplex")
        self.p_vec, input_total = simplex(
            p_vec,
            label="integer multinomial probabilities",
        )
        self.min_val = exact_integer(min_val, label="integer multinomial minimum category")
        self.max_val = self.min_val + self.p_vec.shape[0] - 1
        with np.errstate(divide="ignore"):
            self.log_p_vec = np.log(self.p_vec)
        self.log_p_vec.setflags(write=False)
        self.num_vals = self.p_vec.shape[0]
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys
        self.name = name
        self.simplex_input_total = input_total

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = repr(self.min_val)
        s2 = repr(list(self.p_vec))
        s3 = str(self.len_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return "IntegerMultinomialDistribution(%s, %s, len_dist=%s, name=%s, keys=%s)" % (
            s1,
            s2,
            s3,
            s4,
            s5,
        )

    def density(self, x: Sequence[tuple[int, float]]) -> float:
        """Evaluate the density of IntegerMultinomialDistribution at observed value x.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer category value and number of
                successes.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[tuple[int, float]]) -> float:
        """Evaluate the log-density of IntegerMultinomialDistribution at observed value x.

        Normalized conditional count-vector log-mass given by

        log(p_mat(x)) = log(n!) - sum_k log(x_k!) + sum_k x_k*log(p_k) + log(p_mat(n)),
        for x having k integer categories and n the total trial count.

        Note: x has k integer values and p_k denotes the probability of success for integer-category x_k. The
        ``len_dist`` defaults to ``NullDistribution``, whose log-density is zero for any input; this gives a
        conditional count-vector mass at each observed total, while a real length law makes it a joint mass.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer category value and number of
                successes.

        Returns:
            Log-density at x.

        """
        pairs, cc, outside = canonical_integer_bag(
            x,
            min_val=self.min_val,
            max_val=self.max_val,
        )
        if outside:
            return -np.inf
        rv = log_coefficient([count for _, count in pairs])
        for category, count in pairs:
            log_probability = self.log_p_vec[category - self.min_val]
            if not np.isfinite(log_probability):
                return -np.inf
            rv += log_probability * count
        rv += self.len_dist.log_density(cc)
        return float(rv)

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density for an encoded sequence of iid observations from integer multinomial
            distribution.

        Arg 'x' is a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[T1]): Sequence encoded number of trials for each sequence (length sz), with type T if
                length DataSequenceEncoder is not NullDataEncoder and returns type T. Else None.

        Args:
            x (See above for details): Sequence encoding of iid integer multinomial observation.

        Returns:
            Numpy array of log-density evaluated at each observation in encoding.

        """
        sz, idx, cnt, val, tcnt, totals = _validate_integer_encoding(x)
        v = val - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        nz = cnt != 0
        rv = np.zeros(len(v))
        rv.fill(-np.inf)
        # Only multiply log_p_vec by cnt where cnt != 0: a zero-count entry contributes
        # nothing even for an in-support category with log_p_vec[k] == -inf (avoids
        # (-inf) * 0 = NaN). Matches the scalar log_density path's "if cnt == 0: continue"
        # guard, including for out-of-support categories (those stay masked to -inf above
        # only when cnt != 0; zero-count rows are zeroed below regardless of support).
        um = np.bitwise_and(u, nz)
        rv[um] = self.log_p_vec[v[um]] * cnt[um]
        rv[~nz] = 0.0
        ll = np.bincount(idx, weights=rv, minlength=sz)
        ll += gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(cnt + 1.0),
            minlength=sz,
        )

        if tcnt is not None:
            ll += self.len_dist.seq_log_density(tcnt)

        return ll

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded integer count vectors."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, cnt, val, tcnt, totals = _validate_integer_encoding(x)
        ll = engine.zeros(sz)

        if len(idx) > 0:
            v = val - self.min_val
            valid = np.bitwise_and(v >= 0, v < self.num_vals)
            if self.num_vals == 0:
                contrib = engine.asarray(np.full(len(v), -np.inf))
            else:
                safe_v = np.clip(v, 0, self.num_vals - 1)
                table = engine.asarray(self.log_p_vec)
                nonzero = cnt != 0
                contrib = table[engine.asarray(safe_v)] * engine.asarray(cnt)
                contrib = engine.where(
                    engine.asarray(np.bitwise_and(valid, nonzero)),
                    contrib,
                    engine.asarray(np.full(len(v), -np.inf)),
                )
                contrib = engine.where(
                    engine.asarray(~nonzero),
                    engine.asarray(np.zeros(len(v))),
                    contrib,
                )
            ll = engine.index_add(ll, engine.asarray(idx), contrib)
        coefficient = gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(cnt + 1.0),
            minlength=sz,
        )
        ll = ll + engine.asarray(coefficient)

        if tcnt is not None:
            ll = ll + backend_seq_log_density(self.len_dist, tcnt, engine)

        return ll

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IntegerMultinomialDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer-count-vector parameters for homogeneous mixture kernels."""
        from mixle.stats.compute.stacked import stacked_component_params

        min_val = int(dists[0].min_val)
        num_vals = int(dists[0].num_vals)
        null_len_dist = supports(dists[0].len_dist, Neutral)
        if any(
            int(dist.min_val) != min_val
            or int(dist.num_vals) != num_vals
            or supports(dist.len_dist, Neutral) != null_len_dist
            for dist in dists
        ):
            raise ValueError(
                "Stacked IntegerMultinomialDistribution components require shared support and length policy."
            )

        length_route = None
        if not null_len_dist:
            try:
                length_route = stacked_component_params([dist.len_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "IntegerMultinomial length child %s is not stackable: %s" % (type(dists[0].len_dist).__name__, exc)
                )

        return {
            "__pysp_component_axis__": {"log_p": 1},
            "min_val": min_val,
            "num_vals": num_vals,
            "log_p": engine.asarray(np.stack([dist.log_p_vec for dist in dists], axis=1)),
            "length_route": length_route,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of integer-multinomial log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        sz, idx, cnt, val, tcnt, totals = _validate_integer_encoding(x)
        num_components = int(params["num_components"])
        num_vals = int(params["num_vals"])
        rv = engine.zeros((sz, num_components))

        if len(idx) > 0:
            rel = val - int(params["min_val"])
            valid = np.bitwise_and(rel >= 0, rel < num_vals)
            if num_vals == 0:
                contrib = engine.zeros((len(rel), num_components)) + engine.asarray(-np.inf)
            else:
                safe_rel = np.clip(rel, 0, num_vals - 1)
                nonzero = cnt != 0
                contrib = params["log_p"][engine.asarray(safe_rel), :] * engine.asarray(cnt)[:, None]
                contrib = engine.where(
                    engine.asarray(np.bitwise_and(valid, nonzero))[:, None],
                    contrib,
                    engine.asarray(-np.inf),
                )
                contrib = engine.where(
                    engine.asarray(~nonzero)[:, None],
                    engine.asarray(0.0),
                    contrib,
                )
            rv = engine.index_add(rv, engine.asarray(idx), contrib)
        coefficient = gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(cnt + 1.0),
            minlength=sz,
        )
        rv = rv + engine.asarray(coefficient)[:, None]

        if params["length_route"] is not None and tcnt is not None:
            rv = rv + stacked_component_log_density(tcnt, params["length_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy ``(min_val, count_vec, length_stat)`` statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        sz, idx, cnt, val, tenc, _totals = _validate_integer_encoding(x)
        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        num_vals = int(params["num_vals"])

        if len(idx) > 0 and num_vals > 0:
            rel = val - int(params["min_val"])
            valid = np.bitwise_and(rel >= 0, rel < num_vals)
            row_weights = ww[engine.asarray(idx)] * engine.asarray(cnt)[:, None]
            zero_rows = row_weights * engine.asarray(0.0)
            rows = []
            for value_index in range(num_vals):
                mask = np.bitwise_and(valid, rel == value_index)
                rows.append(engine.sum(engine.where(engine.asarray(mask)[:, None], row_weights, zero_rows), axis=0))
            count_mat = engine.stack(rows, axis=1)
        else:
            count_mat = engine.zeros((num_components, num_vals))

        if params["length_route"] is None or tenc is None:
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
                tenc, ww, params["length_route"], engine, length_estimator
            )
            length_by_component = unstack_component_stats(length_stats, num_components)

        min_val = int(params["min_val"])
        return tuple((min_val, count_mat[i], length_by_component[i]) for i in range(num_components))

    def sampler(self, seed: int | None = None) -> "IntegerMultinomialSampler":
        """Create a sampler for this integer multinomial distribution.

        Args:
            seed (Optional[int]): Set seed on random number generator used in sampling.

        Returns:
            IntegerMultinomialSampler: Sampler bound to this distribution.

        """
        if supports(self.len_dist, Neutral):
            raise ValueError(
                "IntegerMultinomialDistribution must have len_dist set to distribution with support on "
                "non-negative integers."
            )
        return IntegerMultinomialSampler(self, seed)

    def estimator(self, pseudo_count: int | None = None) -> "IntegerMultinomialEstimator":
        """Create an estimator initialized from this distribution.

        Args:
            pseudo_count (Optional[float]): Optional prior mass assigned to this distribution's current category
                probabilities during estimation.

        Returns:
            IntegerMultinomialEstimator: Estimator configured with the same support, name, and length estimator.

        """
        len_est = NullEstimator() if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)

        return IntegerMultinomialEstimator(
            min_val=self.min_val,
            max_val=self.max_val,
            len_estimator=len_est,
            pseudo_count=pseudo_count,
            suff_stat=None if pseudo_count is None else (self.min_val, self.p_vec),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        """Return a data encoder using the encoder supplied by ``len_dist``."""
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerMultinomialDataEncoder(
            len_encoder=len_encoder,
            min_val=self.min_val,
            max_val=self.max_val,
        )

    def enumerator(self, *, max_outcomes_per_length: int = 100_000) -> "IntegerMultinomialEnumerator":
        """Returns IntegerMultinomialEnumerator iterating count vectors in descending log-density order."""
        return IntegerMultinomialEnumerator(
            self,
            max_outcomes_per_length=max_outcomes_per_length,
        )


class IntegerMultinomialEnumerator(DistributionEnumerator):
    """Enumerate normalized integer count-vector masses with a modeled trial-count law."""

    def __init__(
        self,
        dist: IntegerMultinomialDistribution,
        *,
        max_outcomes_per_length: int = 100_000,
    ) -> None:
        """Create an enumerator for integer multinomial observations.

        Trial counts come from ``len_dist``. Each finite count-vector support at a length is
        materialized, scored with the exact multinomial coefficient, sorted, and merged behind the
        length probability frontier. ``max_outcomes_per_length`` bounds that materialization.
        Enumeration is refused when no trial-count law is modeled.

        Args:
            dist (IntegerMultinomialDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        self.max_outcomes_per_length = exact_integer(
            max_outcomes_per_length,
            label="integer multinomial max_outcomes_per_length",
            nonnegative=True,
        )
        if self.max_outcomes_per_length == 0:
            raise ValueError("integer multinomial max_outcomes_per_length must be positive")
        if supports(dist.len_dist, Neutral):
            raise EnumerationError(
                dist,
                reason="no trial-count distribution is modeled, so the joint support over lengths is not normalized",
            )
        len_stream = BufferedStream(child_enumerator(dist.len_dist, "IntegerMultinomialDistribution.len_dist"))
        self._merge = LengthFrontierMerge(len_stream, self._outcomes_for_length)

    def _outcomes_for_length(self, n: Any, log_length_probability: float):
        n = exact_integer(n, label="integer multinomial enumerated trial count", nonnegative=True)
        outcome_count = math.comb(n + self.dist.num_vals - 1, self.dist.num_vals - 1)
        if outcome_count > self.max_outcomes_per_length:
            raise EnumerationError(
                self.dist,
                reason="length %d has %d count vectors, exceeding max_outcomes_per_length=%d"
                % (n, outcome_count, self.max_outcomes_per_length),
            )

        def compositions(total: int, parts: int, prefix: tuple[int, ...] = ()):
            if parts == 1:
                yield prefix + (total,)
                return
            for count in range(total + 1):
                yield from compositions(total - count, parts - 1, prefix + (count,))

        outcomes = []
        for counts in compositions(n, self.dist.num_vals):
            if any(count and self.dist.p_vec[i] == 0.0 for i, count in enumerate(counts)):
                continue
            value = [(self.dist.min_val + i, count) for i, count in enumerate(counts) if count]
            score = (
                log_coefficient(counts)
                + sum(count * self.dist.log_p_vec[i] for i, count in enumerate(counts) if count)
                + float(log_length_probability)
            )
            outcomes.append((value, float(score)))
        outcomes.sort(key=lambda item: item[1], reverse=True)
        return iter(outcomes)

    def __next__(self) -> tuple[list[tuple[int, int]], float]:
        return next(self._merge)


class IntegerMultinomialSampler(DistributionSampler):
    """Draw sparse integer-category count vectors from an integer multinomial."""

    def __init__(self, dist: IntegerMultinomialDistribution, seed: int | None = None) -> None:
        """Create a sampler for an integer multinomial distribution.

        Args:
            dist (IntegerMultinomialDistribution): Distribution to sample from.
            seed (Optional[int]): Optional seed for random number generator.

        Attributes:
            dist (IntegerMultinomialDistribution): Distribution being sampled.
            rng (RandomState): Random number generator initialized from ``seed``.
            len_sampler (DistributionSampler): Sampler for the trial-count distribution.

        """
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(
        self, size: int | None = None, *, batched: bool = True
    ) -> list[tuple[int, float]] | list[list[tuple[int, float]]]:
        """Draw independent samples from an integer multinomial distribution.

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            List length size containing List[Tuple[int, float]]. If size is None, returns one sample
                List[Tuple[int, float]].

        """
        if size is None:
            cnt = self.len_sampler.sample()
            entry = self.rng.multinomial(cnt, self.dist.p_vec)
            rrv = []
            for j in np.flatnonzero(entry):
                rrv.append((j + self.dist.min_val, entry[j]))
            return rrv

        else:
            cnt = self.len_sampler.sample(size=size)
            rv = []

            for i in range(size):
                rrv = []
                entry = self.rng.multinomial(cnt[i], self.dist.p_vec)
                for j in np.flatnonzero(entry):
                    rrv.append((j + self.dist.min_val, entry[j]))
                rv.append(rrv)
            return rv


class IntegerMultinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate sparse integer-category counts and trial-count child statistics."""

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        name: str | None = None,
        keys: str | None = None,
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        fixed_support: bool | None = None,
    ) -> None:
        """Create an accumulator for integer-category count statistics.

        Args:
            min_val (Optional[int]): Smallest integer category tracked initially.
            max_val (Optional[int]): Largest integer category tracked initially.
            name (Optional[str]): Optional name carried with the accumulator.
            keys (Optional[str]): Optional key for sharing sufficient statistics with compatible accumulators.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for multinomial trial
                counts.

        Attributes:
            min_val (Optional[int]): Smallest tracked integer category.
            max_val (Optional[int]): Largest tracked integer category.
            name (Optional[str]): Optional accumulator name.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for trial counts, or
                ``NullAccumulator`` when omitted.
            count_vec (Optional[ndarray]): Weighted counts for categories from ``min_val`` through ``max_val``.
            keys (Optional[str]): Key used by ``key_merge`` and ``key_replace``.

        """
        if max_val is not None and min_val is None:
            raise ValueError("integer multinomial accumulator max_val requires min_val")
        if min_val is not None:
            min_val = exact_integer(min_val, label="integer multinomial accumulator minimum")
            if max_val is not None:
                max_val = exact_integer(max_val, label="integer multinomial accumulator maximum")
                if min_val > max_val:
                    raise ValueError("integer multinomial accumulator support minimum must not exceed maximum")
        self.min_val = min_val
        self.max_val = max_val
        # max_val is also the running maximum once observations arrive, so the configured ceiling is
        # kept separately -- otherwise a learned ceiling would start rejecting larger categories.
        self.fixed_max_val = max_val
        self.fixed_support = (min_val is not None) if fixed_support is None else bool(fixed_support)
        if self.fixed_support and min_val is None:
            raise ValueError("fixed integer multinomial support requires explicit bounds")
        self.name = name
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.count_vec = vec.zeros(max_val - min_val + 1) if min_val is not None and max_val is not None else None
        self.keys = keys

    def update(
        self, x: Sequence[tuple[int, float]], weight: float, estimate: IntegerMultinomialDistribution | None
    ) -> None:
        """Update sufficient statistics from single data observation.

        Args:
            x (Sequence[Tuple[int, float]]): Single observation of integer multinomial distribution.
            weight (float): Weight for observation.
            estimate (Optional[IntegerMultinomialDistribution]): Optional previous estimate of integer multinomial
                distribution.

        Returns:
            None.

        """
        pairs, cc, _ = canonical_integer_bag(
            x,
            min_val=self.min_val,
            max_val=self.fixed_max_val,
            reject_outside=self.fixed_support,
        )
        checked_weight = finite_weight(
            weight,
            label="integer multinomial observation weight",
        )
        if estimate is not None and not isinstance(estimate, IntegerMultinomialDistribution):
            raise TypeError("integer multinomial accumulator estimate must be an integer multinomial distribution")
        for xx, cnt in pairs:
            if self.count_vec is None:
                # A pinned floor bases the vector at min_val even when nothing that low was observed;
                # canonical_integer_bag already refused anything below it, so xx >= min_val.
                self.min_val = self.min_val if self.fixed_support else xx
                self.max_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[xx - self.min_val] += checked_weight * cnt
            elif self.max_val < xx:
                temp_vec = self.count_vec
                self.max_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[: len(temp_vec)] = temp_vec
                self.count_vec[xx - self.min_val] += checked_weight * cnt
            elif self.min_val > xx:
                temp_vec = self.count_vec
                temp_diff = self.min_val - xx
                self.min_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[temp_diff:] = temp_vec
                self.count_vec[xx - self.min_val] += checked_weight * cnt
            else:
                self.count_vec[xx - self.min_val] += checked_weight * cnt

        if estimate is None:
            self.len_accumulator.update(cc, checked_weight, None)
        else:
            self.len_accumulator.update(cc, checked_weight, estimate.len_dist)

    def initialize(self, x: Sequence[tuple[int, float]], weight: float, rng: RandomState | None) -> None:
        """Initialize IntegerMultinomialAccumulator with single observation x.

        Just calls update() method.

        Args:
            x (Sequence[Tuple[int, float]]): Single observation of integer multinomial distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Optional random state for consistency with
                SequenceEncodableStatisticAccumulator class.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_update(self, x: E, weights: np.ndarray, estimate: IntegerMultinomialDistribution | None) -> None:
        """Vectorized update of IntegerMultinomialAccumulator sufficient statistics for encoded sequence of
            independent observations x.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[E0]): Sequence encoded number of trials for each sequence (length sz), with type E0 if
                length DataSequenceEncoder is not NullDataEncoder and returns type E0.
        Args:
            x (See above): Encoded sequence of iid observations of integer multinomial distribution.
            weights (ndarray): Weights for observations in encoded sequence.
            estimate (Optional[IntegerMultinomialDistribution]): Optional previous estimate of integer multinomial
                distribution.

        Returns:
            None.

        """
        sz, idx, cnt, val, tenc, _totals = _validate_integer_encoding(x)
        checked_weights = observation_weights(
            weights,
            sz,
            label="integer multinomial observation weights",
        )
        if estimate is not None and not isinstance(estimate, IntegerMultinomialDistribution):
            raise TypeError("integer multinomial accumulator estimate must be an integer multinomial distribution")
        nonzero = cnt != 0
        active_val = val[nonzero]
        active_cnt = cnt[nonzero]
        active_idx = idx[nonzero]
        if self.fixed_support and len(active_val):
            outside = bool(np.any(active_val < self.min_val))
            if self.fixed_max_val is not None:
                outside = outside or bool(np.any(active_val > self.fixed_max_val))
            if outside:
                raise ValueError("integer multinomial category is outside the fixed accumulator support")
        if len(active_val):
            min_x = int(active_val.min())
            max_x = int(active_val.max())
            loc_cnt = np.bincount(
                active_val - min_x,
                weights=active_cnt * checked_weights[active_idx],
            )

            if self.count_vec is None:
                # A pinned floor bases the vector at min_val; the guard above already rejected
                # anything below it, so min_x >= min_val.
                self.min_val = self.min_val if self.fixed_support else min_x
                self.max_val = max_x
                self.count_vec = np.zeros(self.max_val - self.min_val + 1)

            if self.min_val > min_x or self.max_val < max_x:
                prev_min = self.min_val
                self.min_val = min(min_x, self.min_val)
                self.max_val = max(max_x, self.max_val)
                temp = self.count_vec
                prev_diff = prev_min - self.min_val
                self.count_vec = np.zeros(self.max_val - self.min_val + 1)
                self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

            min_diff = min_x - self.min_val
            self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

        if self.len_accumulator is not None:
            if estimate is None:
                self.len_accumulator.seq_update(tenc, checked_weights, None)
            else:
                self.len_accumulator.seq_update(tenc, checked_weights, estimate.len_dist)

    def seq_update_engine(
        self, x: E, weights: Any, estimate: IntegerMultinomialDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of integer-multinomial count statistics (numpy or torch).

        The weighted category histogram is reduced on the active engine; the dynamic support
        range is host bookkeeping. The length child is routed through the engine via
        child_seq_update. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, idx, cnt, val, tenc, _totals = _validate_integer_encoding(x)
        weights_np = observation_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            sz,
            label="integer multinomial observation weights",
        )
        if estimate is not None and not isinstance(estimate, IntegerMultinomialDistribution):
            raise TypeError("integer multinomial accumulator estimate must be an integer multinomial distribution")
        nonzero = cnt != 0
        active_val = val[nonzero]
        active_cnt = cnt[nonzero]
        active_idx = idx[nonzero]
        if self.fixed_support and len(active_val):
            outside = bool(np.any(active_val < self.min_val))
            if self.fixed_max_val is not None:
                outside = outside or bool(np.any(active_val > self.fixed_max_val))
            if outside:
                raise ValueError("integer multinomial category is outside the fixed accumulator support")
        if len(active_val):
            min_x = int(active_val.min())
            max_x = int(active_val.max())
            row_weights = active_cnt.astype(np.float64) * weights_np[active_idx]
            bidx = engine.asarray((active_val - min_x).astype(np.int64))
            loc_cnt = np.asarray(
                engine.to_numpy(
                    engine.bincount(
                        bidx,
                        weights=engine.asarray(row_weights),
                        minlength=max_x - min_x + 1,
                    )
                ),
                dtype=np.float64,
            )

            if self.count_vec is None:
                # A pinned floor bases the vector at min_val; the guard above already rejected
                # anything below it, so min_x >= min_val.
                self.min_val = self.min_val if self.fixed_support else min_x
                self.max_val = max_x
                self.count_vec = np.zeros(self.max_val - self.min_val + 1)

            if self.min_val > min_x or self.max_val < max_x:
                prev_min = self.min_val
                self.min_val = min(min_x, self.min_val)
                self.max_val = max(max_x, self.max_val)
                temp = self.count_vec
                prev_diff = prev_min - self.min_val
                self.count_vec = np.zeros(self.max_val - self.min_val + 1)
                self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

            min_diff = min_x - self.min_val
            self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

        if self.len_accumulator is not None:
            len_estimate = None if estimate is None else estimate.len_dist
            child_seq_update(self.len_accumulator, tenc, engine.asarray(weights_np), len_estimate, engine)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of sufficient statistics from encoded sequence of iid observations 'x'.

        This delegates to :meth:`seq_update`.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[T1]): Sequence encoded number of trials for each sequence (length sz), with type E0 if
                length DataSequenceEncoder is not NullDataEncoder and returns type E0. Else None.

        Args:
            x (See above): Encoded sequence of iid observations of integer multinomial distribution.
            weights (ndarray): Weights for observations in encoded sequence.
            rng (Optional[RandomState]): Optional random state for consistency with
                SequenceEncodableStatisticAccumulator class.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[int, np.ndarray, SS0 | None]) -> "IntegerMultinomialAccumulator":
        """Combine another integer multinomial sufficient-statistics tuple into this accumulator.

        Arg 'suff_stat' contains:
            suff_stat[0] (int): A minimum value for aggregated counts.
            suff_stat[1] (np.ndarray): Numpy array of aggregated counts.
            suff_stat[2] (Optional[SS0]): Optional sufficient statistics for the length accumulator with type SS0.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerMultinomialAccumulator object.

        """
        stat_min, stat_counts, len_stat = _validate_integer_statistics(suff_stat)
        if self.fixed_support and stat_counts is not None:
            if _outside(stat_min, stat_counts, self.min_val, self.fixed_max_val):
                raise ValueError("integer multinomial statistic lies outside fixed accumulator support")
        if self.count_vec is None and stat_counts is not None:
            self.min_val = self.min_val if self.fixed_support else stat_min
            self.max_val = stat_min + len(stat_counts) - 1
            self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
            offset = stat_min - self.min_val
            self.count_vec[offset : offset + len(stat_counts)] = stat_counts

        elif self.count_vec is not None and stat_counts is not None:
            if self.min_val == stat_min and len(self.count_vec) == len(stat_counts):
                self.count_vec += stat_counts

            else:
                min_val = min(self.min_val, stat_min)
                max_val = max(self.max_val, stat_min + len(stat_counts) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = stat_min - min_val
                i1 = (stat_min + len(stat_counts) - 1) - min_val + 1
                count_vec[i0:i1] += stat_counts

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        self.len_accumulator.combine(len_stat)

        return self

    def value(self) -> tuple[int, np.ndarray, Any | None]:
        """Return accumulated sufficient statistics.

        The returned tuple contains:
            suff_stat[0] (int): Minimum integer category represented by the count vector.
            suff_stat[1] (np.ndarray): Weighted counts for consecutive integer categories.
            suff_stat[2] (Optional[SS0]): Sufficient statistics from the length accumulator.

        Returns:
            Tuple[int, ndarray, Optional[SS0]].

        """
        return (
            self.min_val,
            None if self.count_vec is None else self.count_vec.copy(),
            self.len_accumulator.value(),
        )

    def from_value(self, x: tuple[int, np.ndarray, SS0 | None]) -> "IntegerMultinomialAccumulator":
        """Restore accumulator state from sufficient statistics.

        The input tuple contains:
            x[0] (int): Minimum integer category represented by the count vector.
            x[1] (np.ndarray): Weighted counts for consecutive integer categories.
            x[2] (Optional[SS0]): Sufficient statistics for the length accumulator.


        Args:
            x (See above for details).

        Returns:
            IntegerMultinomialAccumulator: This accumulator after restoration.

        """
        stat_min, stat_counts, len_stat = _validate_integer_statistics(x)
        if self.fixed_support and stat_counts is not None:
            if _outside(stat_min, stat_counts, self.min_val, self.fixed_max_val):
                raise ValueError("integer multinomial statistic lies outside fixed accumulator support")
        if self.fixed_support and self.fixed_max_val is not None:
            self.count_vec.fill(0.0)
            if stat_counts is not None:
                offset = stat_min - self.min_val
                self.count_vec[offset : offset + len(stat_counts)] = stat_counts
        elif self.fixed_support:
            # Floor pinned, ceiling learned: rebase the restored counts at the pinned floor.
            if stat_counts is None:
                self.max_val = None
                self.count_vec = None
            else:
                self.max_val = stat_min + len(stat_counts) - 1
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                offset = stat_min - self.min_val
                self.count_vec[offset : offset + len(stat_counts)] = stat_counts
        else:
            self.min_val = stat_min
            self.max_val = None if stat_counts is None else stat_min + len(stat_counts) - 1
            self.count_vec = None if stat_counts is None else stat_counts.copy()

        self.len_accumulator.from_value(len_stat)

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` when it has a matching key.

        Args:
            stats_dict (Dict[str, Any]): Mapping from statistic keys to accumulators.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics from ``stats_dict`` when its key is present.

        Args:
            stats_dict (Dict[str, Any]): Mapping from statistic keys to accumulators.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        """Return a data encoder using the encoder supplied by the length accumulator."""
        len_encoder = self.len_accumulator.acc_to_encoder()
        return IntegerMultinomialDataEncoder(
            len_encoder=len_encoder,
            min_val=self.min_val if self.fixed_support else None,
            max_val=self.fixed_max_val,
        )


class IntegerMultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer multinomial accumulators with optional length accumulators."""

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        name: str | None = None,
        keys: str | None = None,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        fixed_support: bool | None = None,
    ) -> None:
        """Create a factory for integer multinomial accumulators.

        Args:
            min_val (Optional[int]): Smallest integer category for new accumulators.
            max_val (Optional[int]): Largest integer category for new accumulators.
            name (Optional[str]): Optional name for new accumulators.
            keys (Optional[str]): Optional key for sharing sufficient statistics.
            len_factory (Optional[StatisticAccumulatorFactory]): Factory for the trial-count accumulator.

        Attributes:
            min_val (Optional[int]): Smallest integer category for new accumulators.
            max_val (Optional[int]): Largest integer category for new accumulators.
            name (Optional[str]): Optional name for new accumulators.
            keys (Optional[str]): Optional key for sharing sufficient statistics.
            len_factory (StatisticAccumulatorFactory): Factory for trial-count accumulators.

        """
        if max_val is not None and min_val is None:
            raise ValueError("integer multinomial accumulator factory max_val requires min_val")
        if min_val is not None:
            min_val = exact_integer(min_val, label="integer multinomial factory minimum")
            if max_val is not None:
                max_val = exact_integer(max_val, label="integer multinomial factory maximum")
                if min_val > max_val:
                    raise ValueError("integer multinomial factory minimum must not exceed maximum")
        self.min_val = min_val
        self.max_val = max_val
        self.fixed_support = (min_val is not None) if fixed_support is None else bool(fixed_support)
        self.name = name
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys

    def make(self) -> "IntegerMultinomialAccumulator":
        """Return a new integer multinomial accumulator."""
        len_acc = self.len_factory.make()
        return IntegerMultinomialAccumulator(
            min_val=self.min_val,
            max_val=self.max_val,
            name=self.name,
            keys=self.keys,
            len_accumulator=len_acc,
            fixed_support=self.fixed_support,
        )


class IntegerMultinomialEstimator(ParameterEstimator):
    """Estimate integer-category multinomial probabilities from count statistics."""

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
        name: str | None = None,
        pseudo_count: float | None = None,
        suff_stat: tuple[int, np.ndarray] | None = None,
        keys: str | None = None,
    ) -> None:
        """Estimate integer multinomial distributions from accumulated count statistics.

        Args:
            min_val (Optional[int]): Smallest integer category to include. Given alone it pins the floor and
                lets the ceiling be learned; observations below it are rejected.
            max_val (Optional[int]): Largest integer category to include. Requires ``min_val``, and together
                the two fix the support entirely.
            len_estimator (Optional[ParameterEstimator]): Estimator for the trial-count distribution.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution that fixes the trial-count distribution.
            name (Optional[str]): Optional name assigned to estimated distributions.
            pseudo_count (Optional[float]): Prior mass used to smooth category probabilities.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Prior category support and counts.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        Attributes:
            min_val (Optional[int]): Smallest integer category to include, or None when the floor is learned.
            max_val (Optional[int]): Largest integer category to include, or None when the ceiling is learned.
            len_estimator (ParameterEstimator): Estimator for trial counts, or ``NullEstimator`` when omitted.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution that fixes trial-count behavior.
            name (Optional[str]): Optional name assigned to estimated distributions.
            pseudo_count (Optional[float]): Prior mass used to smooth category probabilities.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Prior category support and counts. Ignored when both
                ``min_val`` and ``max_val`` fix the support.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        """
        # min_val alone is a pinned floor with a learned ceiling, which is what the accumulator
        # already implements: it leaves count_vec unallocated so the support grows upward, and
        # passes reject_outside=fixed_support so observations below the floor are refused. Only the
        # estimator disagreed, rejecting the combination outright -- while the line below it still
        # reads ``fixed_support = min_val is not None``, written for exactly this case. A ceiling
        # without a floor stays an error: the distribution is parameterized by (min_val, p_vec) with
        # the maximum derived, so a bare max_val has nothing to anchor to.
        if max_val is not None and min_val is None:
            raise ValueError("integer multinomial max_val requires min_val; the support is anchored at its minimum")
        if min_val is not None:
            min_val = exact_integer(min_val, label="integer multinomial estimator minimum")
            if max_val is not None:
                max_val = exact_integer(max_val, label="integer multinomial estimator maximum")
                if min_val > max_val:
                    raise ValueError("integer multinomial estimator minimum must not exceed maximum")
        if pseudo_count is not None:
            pseudo_count = finite_weight(
                pseudo_count,
                label="integer multinomial pseudo-count",
            )
        prior_min = None
        prior_prob = None
        if suff_stat is not None:
            if not isinstance(suff_stat, tuple) or len(suff_stat) != 2:
                raise ValueError("integer multinomial prior statistic must be a (minimum, probabilities) tuple")
            prior_min = exact_integer(suff_stat[0], label="integer multinomial prior minimum")
            try:
                raw_prior = np.asarray(suff_stat[1], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("integer multinomial prior probabilities must be numeric") from exc
            if raw_prior.ndim != 1 or raw_prior.size == 0 or np.any(~np.isfinite(raw_prior)) or np.any(raw_prior < 0.0):
                raise ValueError(
                    "integer multinomial prior probabilities must be a finite nonempty non-negative vector"
                )
            if raw_prior.sum() == 0.0 and pseudo_count in (None, 0.0):
                prior_prob = np.ones(len(raw_prior), dtype=np.float64) / len(raw_prior)
                prior_prob.setflags(write=False)
            else:
                prior_prob, _ = simplex(
                    raw_prior,
                    label="integer multinomial prior probabilities",
                )
            outside = min_val is not None and prior_min < min_val
            outside = outside or (max_val is not None and prior_min + len(prior_prob) - 1 > max_val)
            if outside:
                raise ValueError("integer multinomial prior support lies outside fixed support")
        self.suff_stat = None if prior_prob is None else (prior_min, prior_prob)
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.fixed_support = min_val is not None
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "IntegerMultinomialAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        if self.fixed_support:
            min_val = self.min_val
            max_val = self.max_val
        elif self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        else:
            min_val = None
            max_val = None

        len_factory = self.len_estimator.accumulator_factory()
        return IntegerMultinomialAccumulatorFactory(
            min_val=min_val,
            max_val=max_val,
            name=self.name,
            keys=self.keys,
            len_factory=len_factory,
            fixed_support=self.fixed_support,
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[int, np.ndarray, SS0 | None]
    ) -> "IntegerMultinomialDistribution":
        """Estimate a distribution from aggregated sufficient statistics.

        If ``pseudo_count`` is not set, prior sufficient statistics are ignored during estimation.

        ``suff_stat`` contains:
            suff_stat[0] (int): Minimum integer category represented by the count vector.
            suff_stat[1] (np.ndarray): Weighted counts for consecutive integer categories.
            suff_stat[2] (Optional[SS0]): Sufficient statistics for the length estimator.

        Args:
            nobs (Optional[float]): Number of observations in accumulated data.
            suff_stat: See above for details.

        Returns:
            IntegerMultinomialDistribution: Estimated distribution.

        """
        stat_min, stat_counts, len_stat = _validate_integer_statistics(suff_stat)
        len_dist = self.len_dist if self.len_dist is not None else self.len_estimator.estimate(nobs, len_stat)

        if self.fixed_support and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val
            if stat_counts is not None and (stat_min < min_val or stat_min + len(stat_counts) - 1 > max_val):
                raise ValueError("observed integer multinomial support lies outside fixed estimator support")
        else:
            supports_: list[tuple[int, int]] = []
            if stat_counts is not None:
                supports_.append((stat_min, stat_min + len(stat_counts) - 1))
            if self.suff_stat is not None:
                prior_min, prior_prob = self.suff_stat
                supports_.append((prior_min, prior_min + len(prior_prob) - 1))
            if not supports_:
                raise ValueError(
                    "cannot infer integer multinomial support from a batch containing only empty bags; "
                    "configure fixed bounds or prior probabilities"
                )
            min_val = min(bound[0] for bound in supports_)
            max_val = max(bound[1] for bound in supports_)
            if self.min_val is not None:
                # Floor pinned, ceiling learned. The accumulator already refused observations below
                # the floor, so reaching here with a lower one means a prior statistic disagrees.
                if min_val < self.min_val:
                    raise ValueError("integer multinomial prior support lies below the fixed estimator minimum")
                min_val = self.min_val

        count_vec = vec.zeros(max_val - min_val + 1)
        if stat_counts is not None:
            offset = stat_min - min_val
            count_vec[offset : offset + len(stat_counts)] += stat_counts
        if self.pseudo_count not in (None, 0.0):
            if self.suff_stat is None:
                count_vec += self.pseudo_count / len(count_vec)
            else:
                prior_min, prior_prob = self.suff_stat
                offset = prior_min - min_val
                count_vec[offset : offset + len(prior_prob)] += self.pseudo_count * prior_prob

        count_sum = float(count_vec.sum())
        if count_sum == 0.0:
            p_vec = np.ones(len(count_vec), dtype=np.float64) / len(count_vec)
        else:
            p_vec = count_vec / count_sum
        return IntegerMultinomialDistribution(
            min_val,
            p_vec,
            len_dist=len_dist,
            name=self.name,
            keys=self.keys,
        )


class IntegerMultinomialDataEncoder(DataSequenceEncoder):
    """Encode sparse integer multinomial observations for vectorized scoring."""

    def __init__(
        self,
        len_encoder: DataSequenceEncoder | None = NullDataEncoder(),
        min_val: int | None = None,
        max_val: int | None = None,
    ) -> None:
        """Create an encoder for iid integer multinomial observations.

        Args:
            len_encoder (Optional[DataSequenceEncoder]): Encoder for the trial count in each observation.

        Attributes:
            len_encoder (DataSequenceEncoder): Encoder for trial counts. Defaults to ``NullDataEncoder`` when omitted.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        if max_val is not None and min_val is None:
            raise ValueError("integer multinomial encoder max_val requires min_val")
        self.min_val = None if min_val is None else exact_integer(min_val, label="integer multinomial encoder minimum")
        self.max_val = None if max_val is None else exact_integer(max_val, label="integer multinomial encoder maximum")
        if self.min_val is not None and self.max_val is not None and self.min_val > self.max_val:
            raise ValueError("integer multinomial encoder minimum must not exceed maximum")

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerMultinomialDataEncoder(len_encoder=%s, min_val=%r, max_val=%r)" % (
            self.len_encoder,
            self.min_val,
            self.max_val,
        )

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent integer multinomial encoder.

        Note: Instance len_encoder must match as well.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is matching instance of IntegerMultinomialDataEncoder, else False.

        """
        if isinstance(other, IntegerMultinomialDataEncoder):
            return (
                self.len_encoder == other.len_encoder
                and self.min_val == other.min_val
                and self.max_val == other.max_val
            )
        else:
            return False

    def seq_encode(
        self, x: Sequence[Sequence[tuple[int, float]]]
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, Any | None]:
        """Encode a sequence of iid integer multinomial observations.

        The returned tuple contains:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[E0]): Sequence encoded number of trials for each sequence (length sz), with type E0 if
                length DataSequenceEncoder is not NullDataEncoder and returns type E0. Else None.

        Args:
            x (Sequence[Sequence[Tuple[int, float]]]): A sequence of iid integer multinomial observations in the form
                of Sequence of Tuple(s) containing integer-category and float valued number of successes.

        Returns:
            Tuple[int, ndarray[int], ndarray[float], ndarray[int], Optional[T]. See above for details.

        """
        idx = []
        cnt = []
        val = []
        tcnt = []

        for i, y in enumerate(x):
            pairs, cc, _ = canonical_integer_bag(
                y,
                min_val=self.min_val,
                max_val=self.max_val,
            )
            for category, count in pairs:
                idx.append(i)
                cnt.append(count)
                val.append(category)
            tcnt.append(cc)

        sz = len(x)
        idx = np.asarray(idx, dtype=np.int64)
        cnt = np.asarray(cnt, dtype=np.int64)
        val = np.asarray(val, dtype=np.int64)
        tcnt = np.asarray(tcnt, dtype=np.int64)

        tcnt = self.len_encoder.seq_encode(tcnt)

        return sz, idx, cnt, val, tcnt

    def row_count(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray, Any | None]) -> int:
        """Return the number of observations in a payload from :meth:`seq_encode`.

        The base implementation infers the count from the payload's shape, which is wrong for this
        encoder: ``idx``/``cnt``/``val`` are flattened over *(observation, category)* pairs, so the
        inference agreed on their common length and reported one row per pair -- 5,129 rows for
        2,000 observations. That made every ``seq_encode`` of this family fail the encoded-row
        conservation check. ``sz`` is the observation count this encoder already computes.
        """
        return int(x[0])
