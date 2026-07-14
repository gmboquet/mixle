"""Integer-keyed multinomial distributions over a bounded support.

Each observation is a sequence of ``(integer_category, count)`` pairs over ``[min_val, max_val]``. Given
category probabilities ``p = (p_0, ..., p_K)`` and a trial-count distribution ``P_len(N)``, the model
scores the unnormalized log-density

    log(P(x,N|p)) = sum_{k=0}^{K} x_k * log(p_k) + log(P_len(N))

where P_len(N) is a distribution for the number of trials in the multinomial. The multinomial coefficient
(log(N!) - sum_k log(x_k!)) is intentionally omitted, so this is a per-category scoring form rather than a
normalized probability mass over count vectors.

"""

import itertools
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

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
)
from mixle.stats.multivariate.categorical_multinomial import MultisetProductEnumerator
from mixle.utils.aliasing import coalesce_alias

SS0 = TypeVar("SS0")
D = Sequence[tuple[int, float]]
E0 = TypeVar("E0")
E = tuple[int, np.ndarray, np.ndarray, np.ndarray, E0 | None]


class IntegerMultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Multinomial distribution over integer-keyed count maps."""

    def compute_capabilities(self):
        """Declare generated-compute support inherited from the trial-count distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, capabilities_for

        child = capabilities_for(self.len_dist)
        return DistributionCapabilities(
            engine_ready=child.engine_ready, kernel_status="generic_table", numpy_only_reason=child.numpy_only_reason
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
                # T(x) is the per-category count vector and eta = log(p_vec); A = 0 and h(x) = 0 on
                # the support [min_val, min_val+K) (this density omits the multinomial coefficient).
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
        sz, idx, cnt, val, _tcnt = x
        min_val = int(params["min_val"])
        k = int(np.asarray(engine.to_numpy(engine.asarray(params["p_vec"]))).reshape(-1).shape[0])
        stat = np.zeros((int(sz), k), dtype=np.float64)
        val = np.asarray(val)
        if val.shape[0] > 0:
            v = np.rint(val - min_val).astype(np.int64)
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
        """Return ``log h = 0`` for observations whose values are all in support, ``-inf`` otherwise."""
        sz, idx, _cnt, val, _tcnt = x
        min_val = int(params["min_val"])
        k = int(np.asarray(engine.to_numpy(engine.asarray(params["p_vec"]))).reshape(-1).shape[0])
        h = np.zeros(int(sz), dtype=np.float64)
        val = np.asarray(val)
        if val.shape[0] > 0:
            v = np.rint(val - min_val).astype(np.int64)
            out = (v < 0) | (v >= k)
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
        p_vec = np.empty(0, dtype=np.float64) if p_vec is None else p_vec

        with np.errstate(divide="ignore"):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.keys = keys
            self.name = name

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        s1 = repr(self.min_val)
        s2 = repr(list(self.p_vec))
        s3 = str(self.len_dist)
        s4 = repr(self.name)
        return "IntegerMultinomialDistribution(%s, %s, len_dist=%s, name=%s)" % (s1, s2, s3, s4)

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

        Un-normalized log-density given by

        log(p_mat(x)) = sum_k x_k*log(p_k), for x having k integer categories.

        Note: x has k integer values and p_k denotes the probability of success for integer-category x_k. The
        multinomial coefficient is intentionally omitted (see the module docstring), so this is a per-category
        scoring form, not a normalized mass over count vectors.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer category value and number of
                successes.

        Returns:
            Log-density at x.

        """
        rv = 0.0
        for xx, cnt in x:
            if cnt == 0:
                # A zero-count term contributes nothing, even for an out-of-support value
                # (avoids (-inf) * 0 = NaN). Matches the seq path's base-measure masking.
                continue
            rv += (-inf if (xx < self.min_val or xx > self.max_val) else self.log_p_vec[xx - self.min_val]) * cnt
        return rv

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
        sz, idx, cnt, val, tcnt = x

        v = val - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(v))
        rv.fill(-np.inf)
        rv[u] = self.log_p_vec[v[u]]
        rv[u] *= cnt[u]
        ll = np.bincount(idx, weights=rv, minlength=sz)

        if tcnt is not None:
            ll += self.len_dist.seq_log_density(tcnt)

        return ll

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded integer count vectors."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, cnt, val, tcnt = x
        ll = engine.zeros(sz)

        if len(idx) > 0:
            v = val - self.min_val
            valid = np.bitwise_and(v >= 0, v < self.num_vals)
            if self.num_vals == 0:
                contrib = engine.asarray(np.full(len(v), -np.inf))
            else:
                safe_v = np.clip(v, 0, self.num_vals - 1)
                table = engine.asarray(self.log_p_vec)
                contrib = table[engine.asarray(safe_v)] * engine.asarray(cnt)
                contrib = engine.where(engine.asarray(valid), contrib, engine.asarray(np.full(len(v), -np.inf)))
            ll = engine.index_add(ll, engine.asarray(idx), contrib)

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

        sz, idx, cnt, val, tcnt = x
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
                contrib = params["log_p"][engine.asarray(safe_rel), :] * engine.asarray(cnt)[:, None]
                contrib = engine.where(engine.asarray(valid)[:, None], contrib, engine.asarray(-np.inf))
            rv = engine.index_add(rv, engine.asarray(idx), contrib)

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

        sz, idx, cnt, val, tenc = x
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

        if pseudo_count is None:
            return IntegerMultinomialEstimator(len_estimator=len_est, name=self.name)
        else:
            return IntegerMultinomialEstimator(
                min_val=self.min_val,
                max_val=self.max_val,
                len_estimator=len_est,
                pseudo_count=pseudo_count,
                suff_stat=(self.min_val, self.p_vec),
                name=self.name,
            )

    def dist_to_encoder(self) -> "IntegerMultinomialDataEncoder":
        """Return a data encoder using the encoder supplied by ``len_dist``."""
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)

    def enumerator(self) -> "IntegerMultinomialEnumerator":
        """Returns IntegerMultinomialEnumerator iterating count vectors in descending log-density order."""
        return IntegerMultinomialEnumerator(self)


class IntegerMultinomialEnumerator(DistributionEnumerator):
    """Enumerates integer count vectors (lists of (category, count) pairs) in descending log-density order."""

    def __init__(self, dist: IntegerMultinomialDistribution) -> None:
        """Create an enumerator for integer multinomial observations.

        IntegerMultinomialDistribution.log_density scores an observation by sum_k n_k * log(p_k)
        alone -- it includes neither the multinomial coefficient nor the trial-count (len_dist)
        contribution -- so every finite count vector over the positive-probability categories
        has positive density and the support is countably infinite. Trial counts are introduced
        lazily through a synthetic frontier: every size-n count vector scores at most
        n * log(p_max), which strictly decreases in n, so size n is instantiated only once it
        can still beat the best pending value. Within a size, count vectors are produced by a
        best-first multiset search over the probability-sorted categories. Values are emitted
        as lists of (category, count) pairs sorted by category, matching the sampler's format;
        log_prob equals log_density exactly.

        Raises EnumerationError when some category has probability one: arbitrarily large
        counts of that category then all have density one and no non-increasing complete
        ordering of the support mass exists.

        Args:
            dist (IntegerMultinomialDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        entries = [(int(dist.min_val + k), float(lp)) for k, lp in enumerate(dist.log_p_vec) if lp > -np.inf]
        if any(lp >= 0.0 for _, lp in entries):
            raise EnumerationError(
                dist,
                reason="a category has probability one, so arbitrarily large trial "
                "counts all have density one and the support mass diverges",
            )
        entries.sort(key=lambda u: -u[1])

        def combine(pairs: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
            return sorted(pairs)

        if len(entries) == 0:
            # No positive-probability category: only the empty observation has positive density.
            self._merge = iter([([], 0.0)])
        else:
            elem_buf = BufferedStream(iter(entries))
            lp_max = entries[0][1]
            len_stream = BufferedStream((n, n * lp_max) for n in itertools.count())
            self._merge = LengthFrontierMerge(
                len_stream, lambda n, lp_len: MultisetProductEnumerator(elem_buf, n, combine=combine, offset=0.0)
            )

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

    def sample(self, size: int | None = None) -> list[tuple[int, float]] | list[list[tuple[int, float]]]:
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
        self.min_val = min_val
        self.max_val = max_val
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
        cc = 0
        for xx, cnt in x:
            cc += cnt
            if self.count_vec is None:
                self.min_val = xx
                self.max_val = xx
                self.count_vec = vec.make([weight * cnt])
            elif self.max_val < xx:
                temp_vec = self.count_vec
                self.max_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[: len(temp_vec)] = temp_vec
                self.count_vec[xx - self.min_val] += weight * cnt
            elif self.min_val > xx:
                temp_vec = self.count_vec
                temp_diff = self.min_val - xx
                self.min_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[temp_diff:] = temp_vec
                self.count_vec[xx - self.min_val] += weight * cnt
            else:
                self.count_vec[xx - self.min_val] += weight * cnt

        if estimate is None:
            self.len_accumulator.update(cc, weight, None)
        else:
            self.len_accumulator.update(cc, weight, estimate.len_dist)

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
        sz, idx, cnt, val, tenc = x

        min_x = val.min()
        max_x = val.max()

        loc_cnt = np.bincount(val - min_x, weights=cnt * weights[idx])

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

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
                self.len_accumulator.seq_update(tenc, weights, None)
            else:
                self.len_accumulator.seq_update(tenc, weights, estimate.len_dist)

    def seq_update_engine(
        self, x: E, weights: Any, estimate: IntegerMultinomialDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of integer-multinomial count statistics (numpy or torch).

        The weighted category histogram is reduced on the active engine; the dynamic support
        range is host bookkeeping. The length child is routed through the engine via
        child_seq_update. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        sz, idx, cnt, val, tenc = x

        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        valv = np.asarray(val)
        min_x = int(valv.min())
        max_x = int(valv.max())

        row_weights = np.asarray(cnt, dtype=np.float64) * weights_np[np.asarray(idx)]
        bidx = engine.asarray((valv - min_x).astype(np.int64))
        loc_cnt = np.asarray(
            engine.to_numpy(engine.bincount(bidx, weights=engine.asarray(row_weights), minlength=max_x - min_x + 1)),
            dtype=np.float64,
        )

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

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
            child_seq_update(self.len_accumulator, tenc, weights, len_estimate, engine)

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
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(suff_stat[1]):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        self.len_accumulator.combine(suff_stat[2])

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
        return self.min_val, self.count_vec, self.len_accumulator.value()

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
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        self.len_accumulator.from_value(x[2])

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
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Create integer multinomial accumulators with optional length accumulators."""

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        name: str | None = None,
        keys: str | None = None,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
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
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys

    def make(self) -> "IntegerMultinomialAccumulator":
        """Return a new integer multinomial accumulator."""
        len_acc = self.len_factory.make()
        return IntegerMultinomialAccumulator(
            min_val=self.min_val, max_val=self.max_val, name=self.name, keys=self.keys, len_accumulator=len_acc
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
            min_val (Optional[int]): Smallest integer category to include when support is fixed.
            max_val (Optional[int]): Largest integer category to include when support is fixed.
            len_estimator (Optional[ParameterEstimator]): Estimator for the trial-count distribution.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution that fixes the trial-count distribution.
            name (Optional[str]): Optional name assigned to estimated distributions.
            pseudo_count (Optional[float]): Prior mass used to smooth category probabilities.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Prior category support and counts.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        Attributes:
            min_val (Optional[int]): Smallest integer category to include when support is fixed.
            max_val (Optional[int]): Largest integer category to include when support is fixed.
            len_estimator (ParameterEstimator): Estimator for trial counts, or ``NullEstimator`` when omitted.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution that fixes trial-count behavior.
            name (Optional[str]): Optional name assigned to estimated distributions.
            pseudo_count (Optional[float]): Prior mass used to smooth category probabilities.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Prior category support and counts. Ignored when both
                ``min_val`` and ``max_val`` fix the support.
            keys (Optional[str]): Optional key for sharing sufficient statistics.

        """
        self.suff_stat = suff_stat
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "IntegerMultinomialAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        len_factory = self.len_estimator.accumulator_factory()
        return IntegerMultinomialAccumulatorFactory(
            min_val=min_val, max_val=max_val, name=self.name, keys=self.keys, len_factory=len_factory
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
        len_dist = self.len_dist if self.len_dist is not None else self.len_estimator.estimate(nobs, suff_stat[2])

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            if adjusted_nobs == 0.0:
                p_vec = np.ones(len(suff_stat[1])) / float(len(suff_stat[1]))
            else:
                p_vec = (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs

            return IntegerMultinomialDistribution(
                suff_stat[0], p_vec, len_dist=len_dist, name=self.name, keys=self.keys
            )

        elif self.pseudo_count is not None and self.min_val is not None and self.max_val is not None:
            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            if adjusted_nobs == 0.0:
                p_vec = np.ones(len(count_vec)) / float(len(count_vec))
            else:
                p_vec = (count_vec + pseudo_count_per_level) / adjusted_nobs

            return IntegerMultinomialDistribution(min_val, p_vec, len_dist=len_dist, name=self.name, keys=self.keys)

        elif self.pseudo_count is not None and self.suff_stat is not None:
            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1] * self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            count_sum = count_vec.sum()
            if count_sum == 0.0:
                p_vec = np.ones(len(count_vec)) / float(len(count_vec))
            else:
                p_vec = count_vec / count_sum

            return IntegerMultinomialDistribution(min_val, p_vec, len_dist=len_dist, name=self.name, keys=self.keys)
        else:
            count_sum = suff_stat[1].sum()
            if count_sum == 0.0:
                p_vec = np.ones(len(suff_stat[1])) / float(len(suff_stat[1]))
            else:
                p_vec = suff_stat[1] / count_sum

            return IntegerMultinomialDistribution(
                suff_stat[0], p_vec, len_dist=len_dist, name=self.name, keys=self.keys
            )


class IntegerMultinomialDataEncoder(DataSequenceEncoder):
    """Encode sparse integer multinomial observations for vectorized scoring."""

    def __init__(self, len_encoder: DataSequenceEncoder | None = NullDataEncoder()) -> None:
        """Create an encoder for iid integer multinomial observations.

        Args:
            len_encoder (Optional[DataSequenceEncoder]): Encoder for the trial count in each observation.

        Attributes:
            len_encoder (DataSequenceEncoder): Encoder for trial counts. Defaults to ``NullDataEncoder`` when omitted.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerMultinomialDataEncoder(len_encoder=" + str(self.len_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an equivalent integer multinomial encoder.

        Note: Instance len_encoder must match as well.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is matching instance of IntegerMultinomialDataEncoder, else False.

        """
        if isinstance(other, IntegerMultinomialDataEncoder):
            return self.len_encoder == other.len_encoder
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
            cc = 0
            for z in y:
                idx.append(i)
                cnt.append(z[1])
                val.append(z[0])
                cc += z[1]
            tcnt.append(cc)

        sz = len(x)
        idx = np.asarray(idx, dtype=np.int32)
        cnt = np.asarray(cnt, dtype=np.float64)
        val = np.asarray(val, dtype=np.int32)
        tcnt = np.asarray(tcnt, dtype=np.int32)

        tcnt = self.len_encoder.seq_encode(tcnt)

        return sz, idx, cnt, val, tcnt
