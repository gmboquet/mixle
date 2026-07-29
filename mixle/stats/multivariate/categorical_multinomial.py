"""Categorical multinomial models over sparse value-count observations.

Let P_dist(V_k) be a distribution for a countable set of discrete observations of values V_k of type T. Denote

    p_k = P_dist(V_k),

as the probability of success for value V_k. Then sum_{k=0}^{inf} p_k = 1. Let x = (x_0, x_1,....,x_{n-1}) be a
multinomial observation for a 'n' trials, where each x_i = (V_j, n_j) for some value V_j in the observation space and
n_j is the associated number of successes for the value. (note: sum n_j = n). Then, denoting p_j = p_mat(V_j), the
normalized count-vector log mass is:

    log(p_mat(x)) = log(n!) - sum_j log(n_j!) + sum_j n_j * log(p_j) + log(P_len(n)),

where P_len(n) is a distribution for the number of trials in the multinomial having support on the non-negative
integers. When ``len_normalized=True``, the count-vector term (including its base measure) is divided by ``n`` for
nonempty bags, yielding a geometric-mean score rather than a sampling law; the length term remains unscaled.

The multinomial is assumed to have data type: Sequence[Tuple[T, float]], where T is the data type of the 'categories'.

"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream
from mixle.inference.fisher import Path
from mixle.stats.combinator.null_dist import NullAccumulator, NullAccumulatorFactory, NullDistribution, NullEstimator
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
from mixle.stats.multivariate._multinomial_contracts import (
    canonical_bag,
    exact_integer,
    finite_weight,
    log_coefficient,
    observation_weights,
)

T = TypeVar("T")  ## Generic data type for value.
T1 = TypeVar("T1")  ## encoded type for dist
T2 = TypeVar("T2")  ## encoded type for len_dist
SS1 = TypeVar("SS1")  ## suff stat type for dist
SS2 = TypeVar("SS2")  ## suff stat type for len_dist


from mixle.stats.combinator.sequence import SequenceFisherView


@dataclass(frozen=True)
class MultinomialEncodedData:
    """Validated sparse multinomial batch representation."""

    indices: np.ndarray
    inverse_totals: np.ndarray
    nonzero_totals: np.ndarray
    encoded_values: Any
    encoded_lengths: Any | None
    counts: np.ndarray
    totals: np.ndarray
    len_normalized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.len_normalized, bool):
            raise TypeError("multinomial encoded len_normalized must be a bool")
        indices = np.asarray(self.indices)
        inverse = np.asarray(self.inverse_totals, dtype=np.float64)
        nonzero = np.asarray(self.nonzero_totals)
        counts = np.asarray(self.counts)
        totals = np.asarray(self.totals)
        maximum = int(np.iinfo(np.int64).max)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("multinomial encoded indices must be a one-dimensional integer array")
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("multinomial encoded counts must be a one-dimensional integer array")
        if totals.ndim != 1 or not np.issubdtype(totals.dtype, np.integer):
            raise ValueError("multinomial encoded totals must be a one-dimensional integer array")
        rows = len(totals)
        if inverse.shape != (rows,) or np.any(~np.isfinite(inverse)) or np.any(inverse < 0.0):
            raise ValueError("multinomial encoded inverse totals must be finite with shape (rows,)")
        if nonzero.shape != (rows,) or not np.issubdtype(nonzero.dtype, np.bool_):
            raise ValueError("multinomial encoded nonzero mask must be boolean with shape (rows,)")
        if len(indices) != len(counts):
            raise ValueError("multinomial encoded entry arrays must have equal length")
        if np.any(indices < 0) or np.any(indices >= rows):
            raise ValueError("multinomial encoded indices must be in [0, rows)")
        if len(indices) > 1 and np.any(indices[1:] < indices[:-1]):
            raise ValueError("multinomial encoded indices must be non-decreasing")
        if np.any(counts <= 0) or np.any(totals < 0):
            raise ValueError("multinomial encoded counts must be positive and totals non-negative")
        if any(int(value) > maximum for value in counts) or any(int(value) > maximum for value in totals):
            raise ValueError("multinomial encoded counts and totals must fit signed 64-bit integers")
        if any(int(value) > maximum for value in indices):
            raise ValueError("multinomial encoded indices must fit signed 64-bit integers")
        computed = np.zeros(rows, dtype=object)
        for row, count in zip(indices, counts):
            computed[int(row)] += int(count)
        if any(int(computed[row]) != int(totals[row]) for row in range(rows)):
            raise ValueError("multinomial encoded totals must equal the sum of row counts")
        expected_nonzero = totals != 0
        if not np.array_equal(nonzero, expected_nonzero):
            raise ValueError("multinomial encoded nonzero mask does not match totals")
        expected_inverse = np.zeros(rows, dtype=np.float64)
        expected_inverse[expected_nonzero] = 1.0 / totals[expected_nonzero]
        if not np.array_equal(inverse, expected_inverse):
            raise ValueError("multinomial encoded inverse totals do not match totals")
        owned = (
            np.asarray(indices, dtype=np.int64).copy(),
            inverse.copy(),
            np.asarray(nonzero, dtype=bool).copy(),
            np.asarray(counts, dtype=np.int64).copy(),
            np.asarray(totals, dtype=np.int64).copy(),
        )
        for array in owned:
            array.setflags(write=False)
        object.__setattr__(self, "indices", owned[0])
        object.__setattr__(self, "inverse_totals", owned[1])
        object.__setattr__(self, "nonzero_totals", owned[2])
        object.__setattr__(self, "counts", owned[3])
        object.__setattr__(self, "totals", owned[4])

    @property
    def rows(self) -> int:
        return len(self.totals)

    def __iter__(self):
        return iter(
            (
                self.indices,
                self.inverse_totals,
                self.nonzero_totals,
                self.encoded_values,
                self.encoded_lengths,
                self.counts,
                self.totals,
            )
        )

    def __len__(self) -> int:
        return 7

    def __getitem__(self, item: int):
        return tuple(self)[item]


def _encoded(
    value: Any,
    *,
    len_normalized: bool,
    value_encoder: DataSequenceEncoder | None = None,
    length_encoder: DataSequenceEncoder | None = None,
) -> MultinomialEncodedData:
    if not isinstance(value, MultinomialEncodedData):
        raise ValueError("multinomial encoded data must be produced by MultinomialDataEncoder")
    if value.len_normalized != len_normalized:
        raise ValueError("multinomial encoded normalization semantics do not match the model")
    if value_encoder is not None:
        try:
            entries = value_encoder.row_count(value.encoded_values)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise ValueError("multinomial encoded value payload is invalid") from exc
        if entries != len(value.indices):
            raise ValueError("multinomial encoded value payload must have one row per sparse entry")
    if length_encoder is not None:
        try:
            rows = length_encoder.row_count(value.encoded_lengths)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise ValueError("multinomial encoded length payload is invalid") from exc
        if rows != value.rows:
            raise ValueError("multinomial encoded length payload must have one row per observation")
    return value


class MultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Multinomial distribution over count vectors."""

    def compute_capabilities(self):
        """Declare generated-compute support inherited from value and length children."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = (self.dist,) if supports(self.len_dist, Neutral) else (self.dist, self.len_dist)
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_table")

    def compute_declaration(self):
        """Return the generated-compute declaration for the categorical multinomial."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        value = declaration_for(self.dist)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(d for d in (value, length) if d is not None)
        roles = []
        if value is not None:
            roles.append("value")
        if length is not None:
            roles.append("length")
        return DistributionDeclaration(
            name="multinomial",
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec("values", kind="child_stat"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="count_vector",
            children=children,
            child_roles=tuple(roles),
            differentiable=False,
        )

    def to_exponential_family(self, engine: Any = None):
        """Return the multinomial exponential-family view, or ``None``.

        A multinomial over an exponential-family element is itself an exponential family with the
        shared element ``eta`` and the count-weighted sufficient statistic ``T(x) = sum_j c_j T_0(v_j)``.
        This holds only when the trial count is not separately modeled (``len_dist`` is Null) and the
        density is not length-normalized (those break the single-exp-family form); it also requires the
        value element to be an exponential family. Otherwise returns ``None``.
        """
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.exp_family import MultinomialExponentialFamilyForm, to_exponential_family

        if not supports(self.len_dist, Neutral) or self.len_normalized:
            return None
        eng = NUMPY_ENGINE if engine is None else engine
        element = to_exponential_family(self.dist, engine=eng)
        if element is None:
            return None
        return MultinomialExponentialFamilyForm(distribution=self, element=element, engine=eng)

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        len_normalized: bool = False,
        name: str | None = None,
    ) -> None:
        """Create a sparse multinomial distribution over the support of ``dist``.

        Args:
            dist: Distribution over individual trial values.
            len_dist: Optional distribution over the total number of trials.
            len_normalized: Whether to score the geometric mean per trial.
            name: Optional diagnostic name.

        Attributes:
            dist: Distribution over individual trial values.
            len_dist: Distribution over trial count.
            len_normalized: Whether log-density is normalized by total count.
            name: Optional diagnostic name.

        """
        self.dist = dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        if not isinstance(len_normalized, bool):
            raise TypeError("multinomial len_normalized must be a bool")
        self.len_normalized = len_normalized
        self.name = name

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.name)
        return "MultinomialDistribution(%s, len_dist=%s, len_normalized=%s, name=%s)" % (s1, s2, s3, s4)

    def density(self, x: Sequence[tuple[T, float]]) -> float:
        """Returns the density of multinomial evaluated at observation x.

        See log_density() for details.

        Args:
            x (Sequence[Tuple[T, float]]): Tuples of observed multinomial values and success s.t. success sum to number
                of trials.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[tuple[T, float]]) -> float:
        """Returns the log-density of multinomial evaluated at observation x.

        Let P_dist(V_k) be a distribution for a countable set of discrete observations of values V_k of type T. Denote

            p_k = P_dist(V_k),

        as the probability of success for value V_k. Then sum_{k=0}^{inf} p_k = 1. Let x = (x_0, x_1,....,x_{n-1}) be a
        multinomial observation for 'n' trials, where each x_i = (V_j, n_j) for some value V_j in the observation
        space and n_j is the associated number of success for the value. (note: sum n_j = n). Then, denoting p_j =
        p_mat(V_j), the normalized count-vector log mass is:

            log(p_mat(x)) = log(n!) - sum_j log(n_j!) + sum_j n_j * log(p_j) + log(P_len(n)),

        where P_len(n) is a distribution for the number of trials in the multinomial having support on the non-negative
        integers.

        Args:
            x (Sequence[Tuple[T, float]]): Tuples of observed multinomial values and success s.t. success sum to number
                of trials.

        Returns:
            Log-density evaluated at x.

        """
        pairs, cc = canonical_bag(x)
        rv = log_coefficient([count for _, count in pairs])
        for value, count in pairs:
            score = self.dist.log_density(value)
            if not np.isfinite(score):
                return -np.inf
            rv += score * count

        if self.len_normalized and cc > 0:
            rv /= cc

        rv += self.len_dist.log_density(cc)

        return float(rv)

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized evaluated of log-density for an encoded sequence of iid multinomial observations.

        See log_density() for details on the log-density function for MultinomialDistribution.

        Arg 'x' is a tuple of size 7 containing:
            x[0] (ndarray[int]): Observation index of sequence values.
            x[1] (ndarray[float]): Trial size for each observation.
            x[2] (ndarray[float]): Non-zero trial size indices.
            x[3] (T1): Sequence encoded flattened list of values from x.
            x[4] (Optional[T2]): Sequence encoded flatted list of trial sizes.
            x[5] (np.ndarray[float]): Flattened array of counts for values.
            x[6] (ndarray[float]): Flattened array of trial sizes.

        Args:
            x: See above for details.

        Returns:
            Numpy array of the log-density at each encoded observation of x.

        """
        encoded = _encoded(
            x,
            len_normalized=self.len_normalized,
            value_encoder=self.dist.dist_to_encoder(),
            length_encoder=self.len_dist.dist_to_encoder(),
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, totals = encoded

        ll = self.dist.seq_log_density(enc_seq)
        ll_sum = np.bincount(idx, weights=ll * enc_w, minlength=len(icnt))
        ll_sum += gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(enc_w + 1.0),
            minlength=encoded.rows,
        )

        if self.len_normalized:
            ll_sum *= icnt

        if enc_nseq is not None:
            nll = self.len_dist.seq_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum

    def backend_seq_log_density(self, x, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded count-vector observations."""
        from mixle.stats.compute.backend import backend_seq_log_density

        encoded = _encoded(
            x,
            len_normalized=self.len_normalized,
            value_encoder=self.dist.dist_to_encoder(),
            length_encoder=self.len_dist.dist_to_encoder(),
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, totals = encoded
        nseq = len(icnt)
        ll_sum = engine.zeros(nseq)

        if len(idx) > 0:
            eidx = engine.asarray(idx)
            counts = engine.asarray(enc_w)
            elem_ll = backend_seq_log_density(self.dist, enc_seq, engine) * counts
            if self.len_normalized:
                elem_ll = elem_ll * engine.asarray(icnt)[eidx]
            ll_sum = engine.index_add(ll_sum, eidx, elem_ll)
        coefficient = gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(enc_w + 1.0),
            minlength=encoded.rows,
        )
        if self.len_normalized:
            coefficient = coefficient * icnt
        ll_sum = ll_sum + engine.asarray(coefficient)

        if enc_nseq is not None:
            ll_sum = ll_sum + backend_seq_log_density(self.len_dist, enc_nseq, engine)

        return ll_sum

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[MultinomialDistribution], engine: Any) -> dict[str, Any]:
        """Return stacked child routes for homogeneous multinomial mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        len_normalized = bool(dists[0].len_normalized)
        null_len_dist = supports(dists[0].len_dist, Neutral)
        if any(
            bool(dist.len_normalized) != len_normalized or supports(dist.len_dist, Neutral) != null_len_dist
            for dist in dists
        ):
            raise ValueError("Stacked MultinomialDistribution components require matching length policy.")
        try:
            value_route = stacked_component_params([dist.dist for dist in dists], engine)
        except ValueError as exc:
            raise ValueError("Multinomial value child %s is not stackable: %s" % (type(dists[0].dist).__name__, exc))
        length_route = None
        if not null_len_dist:
            try:
                length_route = stacked_component_params([dist.len_dist for dist in dists], engine)
            except ValueError as exc:
                raise ValueError(
                    "Multinomial length child %s is not stackable: %s" % (type(dists[0].len_dist).__name__, exc)
                )
        return {
            "value_route": value_route,
            "length_route": length_route,
            "value_encoder": dists[0].dist.dist_to_encoder(),
            "length_encoder": dists[0].len_dist.dist_to_encoder(),
            "len_normalized": len_normalized,
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, ...], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of multinomial log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        encoded = _encoded(
            x,
            len_normalized=bool(params["len_normalized"]),
            value_encoder=params["value_encoder"],
            length_encoder=params["length_encoder"],
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, totals = encoded
        nseq = len(icnt)
        rv = engine.zeros((nseq, int(params["num_components"])))

        if len(idx) > 0:
            eidx = engine.asarray(idx)
            scores = stacked_component_log_density(enc_seq, params["value_route"], engine)
            scores = scores * engine.asarray(enc_w)[:, None]
            if params["len_normalized"]:
                scores = scores * engine.asarray(icnt)[eidx, None]
            rv = engine.index_add(rv, eidx, scores)
        coefficient = gammaln(totals + 1.0) - np.bincount(
            idx,
            weights=gammaln(enc_w + 1.0),
            minlength=encoded.rows,
        )
        if params["len_normalized"]:
            coefficient = coefficient * icnt
        rv = rv + engine.asarray(coefficient)[:, None]

        if params["length_route"] is not None and enc_nseq is not None:
            rv = rv + stacked_component_log_density(enc_nseq, params["length_route"], engine)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: tuple[Any, ...], weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy multinomial sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        encoded = _encoded(
            x,
            len_normalized=bool(params["len_normalized"]),
            value_encoder=params["value_encoder"],
            length_encoder=params["length_encoder"],
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, _totals = encoded
        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        outer_estimators = tuple(getattr(estimator, "estimators", ()))

        value_estimators = tuple(getattr(component_est, "estimator", None) for component_est in outer_estimators)
        value_estimator = StackedEstimatorView(value_estimators) if len(value_estimators) == num_components else None
        if len(idx) > 0:
            eidx = engine.asarray(idx)
            value_weights = ww[eidx]
            if params["len_normalized"]:
                value_weights = value_weights * engine.asarray(icnt)[eidx, None]
            value_weights = value_weights * engine.asarray(enc_w)[:, None]
        else:
            value_weights = engine.zeros((0, num_components))
        value_stats = stacked_component_sufficient_statistics(
            enc_seq, value_weights, params["value_route"], engine, value_estimator
        )
        value_by_component = unstack_component_stats(value_stats, num_components)

        if params["length_route"] is None or enc_nseq is None:
            length_by_component = tuple(None for _ in range(num_components))
        else:
            length_estimators = tuple(
                getattr(component_est, "len_estimator", None) for component_est in outer_estimators
            )
            length_estimator = (
                StackedEstimatorView(length_estimators) if len(length_estimators) == num_components else None
            )
            length_weights = ww
            length_stats = stacked_component_sufficient_statistics(
                enc_nseq, length_weights, params["length_route"], engine, length_estimator
            )
            length_by_component = unstack_component_stats(length_stats, num_components)

        return tuple((value_by_component[i], length_by_component[i]) for i in range(num_components))

    def to_fisher(self, **kwargs):
        """Structural Fisher view for the multinomial bag."""
        if hasattr(self, "dist") and hasattr(self, "len_dist"):
            return MultinomialFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> MultinomialSampler:
        """Return a sampler for iid multinomial bags.

        Args:
            seed: Optional random seed.

        Returns:
            A configured ``MultinomialSampler``.

        """
        if supports(self.len_dist, Neutral):
            raise ValueError(
                "len_dist must not be a SequenceEncodableProbabilityDistribution with support of non-negative integers."
            )
        if self.len_normalized:
            raise ValueError("len_normalized multinomial scores are geometric-mean scores, not a sampling law")
        return MultinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> MultinomialEstimator:
        """Return an estimator initialized from the value and length models.

        Args:
            pseudo_count: Optional smoothing count for child estimators.

        Returns:
            A configured ``MultinomialEstimator``.

        """
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        dist_est = self.dist.estimator(pseudo_count=pseudo_count)
        return MultinomialEstimator(dist_est, len_estimator=len_est, len_normalized=self.len_normalized, name=self.name)

    def dist_to_encoder(self) -> MultinomialDataEncoder:
        """Return an encoder for sparse multinomial observations."""
        return MultinomialDataEncoder(
            encoder=self.dist.dist_to_encoder(),
            len_encoder=self.len_dist.dist_to_encoder(),
            len_normalized=self.len_normalized,
        )

    def enumerator(self) -> MultinomialEnumerator:
        """Returns MultinomialEnumerator iterating (value, count)-pair lists in descending probability order."""
        raise EnumerationError(
            self,
            reason=(
                "exact normalized enumeration over an arbitrary category child requires a finite, "
                "deduplicated support manifest; this model does not currently expose one"
            ),
        )


class MultisetProductEnumerator:
    """Best-first enumeration of the size-n multisets drawn from a sorted (value, log_prob) stream.

    Yields (combine(pairs), log_prob) where pairs is a tuple of (value, count) entries with
    distinct values and counts summing to n, and log_prob = offset + the sum of the n chosen
    element log probs, in non-increasing order. Each multiset is represented exactly once as
    a non-decreasing tuple of ranks into the shared BufferedStream; successors increment a
    single rank while preserving sorted order (only the right-most rank of a run of equal
    ranks may move), which keeps every multiset reachable exactly once and, because the
    stream is sorted, makes successor scores monotone non-increasing.
    """

    def __init__(
        self,
        stream: BufferedStream,
        n: int,
        combine: Callable[[tuple[tuple[Any, int], ...]], Any] = list,
        offset: float = 0.0,
    ) -> None:
        """Create an enumerator for multiset product supports.

        Args:
            stream (BufferedStream): Buffered (value, log_prob) stream sorted by descending log_prob.
            n (int): Multiset size (total count); n = 0 yields the single empty multiset.
            combine (Callable): Maps the tuple of (value, count) pairs (in stream-rank order) to
                the emitted support value. Defaults to list.
            offset (float): Log-probability offset added to every emitted score.

        """
        self.stream = stream
        self.n = exact_integer(n, label="multiset enumerator size", nonnegative=True)
        self.combine = combine
        self.offset = float(offset)
        if not np.isfinite(self.offset):
            raise ValueError("multiset enumerator offset must be finite")
        self._counter = itertools.count()
        self._heap: list[tuple[float, int, tuple[int, ...]]] = []
        self._visited = set()
        self._entries: list[tuple[Any, float]] = []
        self._seen_values: set[Any] = set()
        if self.n == 0:
            # The empty multiset, carrying the offset mass alone.
            self._heap.append((-self.offset, next(self._counter), ()))
            self._visited.add(())
        elif self._entry(0) is not None:
            root = (0,) * self.n
            self._heap.append((-self._score(root), next(self._counter), root))
            self._visited.add(root)

    def __iter__(self) -> MultisetProductEnumerator:
        return self

    def _entry(self, rank: int) -> tuple[Any, float] | None:
        while len(self._entries) <= rank:
            entry = self.stream.get(len(self._entries))
            if entry is None:
                return None
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("child enumerator entries must be (value, log_probability) pairs")
            value, raw_score = entry
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError("child enumerator values must be hashable") from exc
            score = float(raw_score)
            if not np.isfinite(score) or score > 0.0:
                raise ValueError("child enumerator log probabilities must be finite and non-positive")
            if self._entries and score > self._entries[-1][1]:
                raise ValueError("child enumerator scores must be non-increasing")
            if value in self._seen_values:
                raise ValueError("child enumerator values must be unique")
            self._seen_values.add(value)
            self._entries.append((value, score))
        return self._entries[rank]

    def _score(self, idx: tuple[int, ...]) -> float:
        """Return offset plus the sum of the element log probs at ranks idx (recomputed to avoid drift)."""
        rv = self.offset
        for i in idx:
            rv += self._entry(i)[1]
        return rv

    def _pairs(self, idx: tuple[int, ...]) -> tuple[tuple[Any, int], ...]:
        """Group the non-decreasing rank tuple idx into (value, count) pairs in rank order."""
        groups: list[list[int]] = []
        for i in idx:
            if groups and groups[-1][0] == i:
                groups[-1][1] += 1
            else:
                groups.append([i, 1])
        return tuple((self._entry(i)[0], c) for i, c in groups)

    def __next__(self) -> tuple[Any, float]:
        if not self._heap:
            raise StopIteration
        _, _, idx = heapq.heappop(self._heap)
        score = self.offset if len(idx) == 0 else self._score(idx)
        value = self.combine(self._pairs(idx))
        for k in range(len(idx)):
            nxt = idx[k] + 1
            if k + 1 < len(idx) and nxt > idx[k + 1]:
                continue
            succ = idx[:k] + (nxt,) + idx[k + 1 :]
            if succ not in self._visited and self._entry(nxt) is not None:
                self._visited.add(succ)
                heapq.heappush(self._heap, (-self._score(succ), next(self._counter), succ))
        return (value, score)


class MultinomialEnumerator(DistributionEnumerator):
    """Reserved exact enumerator for categorical multinomials.

    Construction currently fails closed because arbitrary category distributions do not expose the
    finite, unique support manifest required to enumerate normalized count-vector probabilities.
    """

    def __init__(self, dist: MultinomialDistribution) -> None:
        """Refuse construction until a validated support-manifest protocol exists."""
        super().__init__(dist)
        raise EnumerationError(
            dist,
            reason=("exact normalized enumeration requires a finite, deduplicated category support manifest"),
        )

    def __next__(self) -> tuple[Any, float]:
        raise StopIteration


class MultinomialSampler(DistributionSampler):
    """Draw sparse value-count observations from a categorical multinomial."""

    def __init__(self, dist: MultinomialDistribution, seed: int | None = None) -> None:
        """Create a sampler for a categorical multinomial distribution.

        Args:
            dist (MultinomialDistribution): Distribution to sample from.
            seed (Optional[int]): Set the seed for sampling.

        Attributes:
             dist (MultinomialDistribution): Distribution to sample from.
             rng (RandomState): Random state initialized from ``seed`` when supplied.
             dist_sampler (DistributionSampler): Sampler for category values.
             len_sampler (DistributionSampler): Sampler for the number of trials.

        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(
        self, size: int | None = None, *, batched: bool = True
    ) -> Sequence[Sequence[tuple[Any, float]]] | Sequence[tuple[Any, float]]:
        """Draw samples from multinomial distribution.

        Note: If len_sampler can draw n=0, an empty list is returned for that sample.

        Args:
            size (Optional[int]): Number of iid samples to draw from multinomial.

        Returns:
            Sequence of 'size' iid observations if size is not None, else a single multinomial sample.

        """
        if size is None:
            n = exact_integer(
                self.len_sampler.sample(),
                label="sampled multinomial trial count",
                nonnegative=True,
            )
            rv = dict()
            for _ in range(n):
                v = self.dist_sampler.sample()
                if v in rv:
                    rv[v] += 1
                else:
                    rv[v] = 1
            return list(rv.items())

        else:
            sample_size = exact_integer(size, label="multinomial sample size", nonnegative=True)
            return [self.sample() for _ in range(sample_size)]


class MultinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate value and trial-count statistics for categorical multinomial data."""

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        len_normalized: bool,
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for multinomial sufficient statistics.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator for category values.
            len_normalized (bool): Take geometric mean of density.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the
                number of trials in each observation.
            keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

        Attributes:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator for category values.
            len_normalized (bool): Take geometric mean of density.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the number of trials in
                each observation, defaults to the NullAccumulator.
            keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

            _init_rng (bool): True if random states have been initialized.
            _len_rng (Optional[RandomState]): RandomState for initializing length accumulator.
            _acc_rng (Optional[RandomState]): Random states for initializing category accumulators.

        """
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.keys = keys
        if not isinstance(len_normalized, bool):
            raise TypeError("multinomial accumulator len_normalized must be a bool")
        self.len_normalized = len_normalized

        ### protected for initialization.
        self._init_rng: bool = False
        self._len_rng: RandomState | None = None
        self._acc_rng: RandomState | None = None

    def update(self, x: Sequence[tuple[T, float]], weight: float, estimate: MultinomialDistribution | None) -> None:
        """Update sufficient statistics from one weighted sparse multinomial observation.

        Args:
            x: Sparse observation as ``[(value, count), ...]``.
            weight: Observation weight.
            estimate: Optional previous multinomial estimate.

        """
        pairs, total = canonical_bag(x)
        checked_weight = finite_weight(weight, label="multinomial observation weight")
        if estimate is not None and not isinstance(estimate, MultinomialDistribution):
            raise TypeError("multinomial accumulator estimate must be a MultinomialDistribution")
        value_estimate = None if estimate is None else estimate.dist
        value_weight = checked_weight / total if self.len_normalized and total > 0 else checked_weight
        for value, count in pairs:
            self.accumulator.update(value, value_weight * count, value_estimate)
        self.len_accumulator.update(
            total,
            checked_weight,
            None if estimate is None else estimate.len_dist,
        )

    def _rng_initialize(self, rng: RandomState) -> None:
        """Set RandomState member variables for initialize and seq_initialize consistency.

        Args:
            rng: Random state used to seed child initializers.

        """
        rng_seeds = rng.randint(maxrandint, size=2)
        self._len_rng = RandomState(seed=rng_seeds[0])
        self._acc_rng = RandomState(seed=rng_seeds[1])
        self._init_rng = True

    def initialize(self, x: Sequence[tuple[T, float]], weight: float, rng: RandomState) -> None:
        """

        Args:
            x (Sequence[Tuple[T, float]]): A single observation of multinomial distribution.
            weight (float): Observation weight.
            rng (Optional[RandomState]): Random state for initialization.

        Returns:

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        pairs, total = canonical_bag(x)
        checked_weight = finite_weight(weight, label="multinomial observation weight")
        value_weight = checked_weight / total if self.len_normalized and total > 0 else checked_weight
        for value, count in pairs:
            self.accumulator.initialize(value, value_weight * count, self._acc_rng)
        self.len_accumulator.initialize(total, checked_weight, self._len_rng)

    def seq_update(self, x, weights: np.ndarray, estimate: MultinomialDistribution | None) -> None:
        """Vectorized update of encoded sequence of iid observations from multinomial distribution.

        Arg 'x' is a tuple of size 7 containing:
            x[0] (ndarray[int]): Observation index of sequence values.
            x[1] (ndarray[float]): Trial size for each observation.
            x[2] (ndarray[float]): Non-zero trial size indices.
            x[3] (T1): Sequence encoded flattened list of values from x.
            x[4] (Optional[T2]): Sequence encoded flatted list of trial sizes.
            x[5] (np.ndarray[float]): Flattened array of counts for values.
            x[6] (ndarray[float]): Flattened array of trial sizes.

        Args:
            x: See above for details.
            weights (np.ndarray): Array of observation weights.
            estimate (Optional[MultinomialDistribution]): Optional previous estimate for multinomial distribution.

        Returns:
            None.

        """
        encoded = _encoded(
            x,
            len_normalized=self.len_normalized,
            value_encoder=self.accumulator.acc_to_encoder(),
            length_encoder=self.len_accumulator.acc_to_encoder(),
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, _totals = encoded
        checked_weights = observation_weights(
            weights,
            encoded.rows,
            label="multinomial observation weights",
        )
        if estimate is not None and not isinstance(estimate, MultinomialDistribution):
            raise TypeError("multinomial accumulator estimate must be a MultinomialDistribution")
        w = checked_weights[idx] * icnt[idx] if self.len_normalized else checked_weights[idx]
        w *= enc_w

        self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)
        self.len_accumulator.seq_update(
            enc_nseq,
            checked_weights,
            estimate.len_dist if estimate is not None else None,
        )

    def seq_update_engine(self, x, weights: Any, estimate: MultinomialDistribution | None, engine: Any) -> None:
        """Engine-resident E-step: the per-value and per-length weights are formed on the active
        engine and the value/length accumulators are routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        encoded = _encoded(
            x,
            len_normalized=self.len_normalized,
            value_encoder=self.accumulator.acc_to_encoder(),
            length_encoder=self.len_accumulator.acc_to_encoder(),
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, _totals = encoded
        checked_weights = observation_weights(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            encoded.rows,
            label="multinomial observation weights",
        )
        if estimate is not None and not isinstance(estimate, MultinomialDistribution):
            raise TypeError("multinomial accumulator estimate must be a MultinomialDistribution")
        w_eng = engine.asarray(checked_weights)
        idx_a = np.asarray(idx, dtype=np.int64)
        if self.len_normalized:
            w = w_eng[idx_a] * engine.asarray(np.asarray(icnt, dtype=np.float64)[idx_a])
        else:
            w = w_eng[idx_a]
        w = w * engine.asarray(np.asarray(enc_w, dtype=np.float64))

        child_seq_update(self.accumulator, enc_seq, w, estimate.dist if estimate is not None else None, engine)
        child_seq_update(
            self.len_accumulator,
            enc_nseq,
            w_eng,
            estimate.len_dist if estimate is not None else None,
            engine,
        )

    def seq_initialize(self, x, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of of sufficient statistics for an encoded sequence of observations.

        Arg 'x' is a tuple of size 7 containing:
            x[0] (ndarray[int]): Observation index of sequence values.
            x[1] (ndarray[float]): Trial size for each observation.
            x[2] (ndarray[float]): Non-zero trial size indices.
            x[3] (T1): Sequence encoded flattened list of values from x.
            x[4] (Optional[T2]): Sequence encoded flatted list of trial sizes.
            x[5] (np.ndarray[float]): Flattened array of counts for values.
            x[6] (ndarray[float]): Flattened array of trial sizes.

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            rng (RandomState): Random state for initialization.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        encoded = _encoded(
            x,
            len_normalized=self.len_normalized,
            value_encoder=self.accumulator.acc_to_encoder(),
            length_encoder=self.len_accumulator.acc_to_encoder(),
        )
        idx, icnt, _inz, enc_seq, enc_nseq, enc_w, _totals = encoded
        checked_weights = observation_weights(
            weights,
            encoded.rows,
            label="multinomial observation weights",
        )
        w = checked_weights[idx] * icnt[idx] if self.len_normalized else checked_weights[idx]
        w = w * enc_w

        self.accumulator.seq_initialize(enc_seq, w, self._acc_rng)
        self.len_accumulator.seq_initialize(enc_nseq, checked_weights, self._len_rng)

    def combine(self, suff_stat: tuple[SS1, SS2 | None]) -> MultinomialAccumulator:
        """Merge aggregated multinomial sufficient statistics into this accumulator.

        Args:
            suff_stat: Tuple containing value-model and length-model sufficient statistics.

        Returns:
            This accumulator.

        """
        self.accumulator.combine(suff_stat[0])
        self.len_accumulator.combine(suff_stat[1])

        return self

    def value(self) -> tuple[Any, Any | None]:
        """Return sufficient statistics as ``(value_stats, length_stats)``."""
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: tuple[SS1, SS2 | None]) -> MultinomialAccumulator:
        """Replace this accumulator's sufficient statistics.

        Args:
            x: Tuple containing value-model and length-model sufficient statistics.

        Returns:
            This accumulator.

        """
        self.accumulator.from_value(x[0])
        self.len_accumulator.from_value(x[1])

        return self

    def scale(self, c: float) -> MultinomialAccumulator:
        """Scale value and length sufficient statistics through their accumulators."""
        self.accumulator.scale(c)
        self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under configured keys.

        Args:
            stats_dict: Mapping from merge keys to sufficient statistics.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

        self.accumulator.key_merge(stats_dict)
        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics from matching keys in ``stats_dict``.

        Args:
            stats_dict: Mapping from merge keys to sufficient statistics.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

        self.accumulator.key_replace(stats_dict)
        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> MultinomialDataEncoder:
        """Return an encoder compatible with sparse multinomial observations."""
        return MultinomialDataEncoder(
            encoder=self.accumulator.acc_to_encoder(),
            len_encoder=self.len_accumulator.acc_to_encoder(),
            len_normalized=self.len_normalized,
        )


class MultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Create categorical multinomial accumulators and child accumulators."""

    def __init__(
        self,
        est_factory: StatisticAccumulatorFactory,
        len_normalized: bool,
        len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
        keys: str | None = None,
    ) -> None:
        """Factory for multinomial accumulators.

        Args:
            est_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the value distribution.
            len_normalized (bool): If true, geometric mean of density is taken.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for number of trials.
            keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

        Attributes:
            est_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the value distribution.
            len_normalized (bool): If true, geometric mean of density is taken.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for number of trials.
            keys (Optional[str]): Set keys for merging sufficient statistics with objects containing matching keys.

        """
        self.est_factory = est_factory
        if not isinstance(len_normalized, bool):
            raise TypeError("multinomial accumulator factory len_normalized must be a bool")
        self.len_normalized = len_normalized
        self.len_factory = len_factory
        self.keys = keys

    def make(self) -> MultinomialAccumulator:
        """Return a fresh multinomial accumulator."""
        len_acc = self.len_factory.make()
        return MultinomialAccumulator(
            self.est_factory.make(), self.len_normalized, len_accumulator=len_acc, keys=self.keys
        )


class MultinomialEstimator(ParameterEstimator):
    """Estimate a categorical multinomial from value and length sufficient statistics."""

    def __init__(
        self,
        estimator: ParameterEstimator,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
        len_normalized: bool | None = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for sparse multinomial sufficient statistics.

        Args:
            estimator: Estimator for the value distribution.
            len_estimator: Optional estimator for the number of trials.
            len_dist: Fixed trial-count distribution, if it should not be estimated.
            len_normalized: Whether to score the geometric mean per trial.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            estimator: Estimator for the value distribution.
            len_estimator: Estimator for trial count.
            len_dist: Fixed trial-count distribution, if supplied.
            len_normalized: Whether log-density is normalized by total count.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        self.estimator = estimator
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist
        if not isinstance(len_normalized, bool):
            raise TypeError("multinomial estimator len_normalized must be a bool")
        self.len_normalized = len_normalized
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> MultinomialAccumulatorFactory:
        """Return an accumulator factory matching this estimator."""
        est_factory = self.estimator.accumulator_factory()
        len_factory = self.len_estimator.accumulator_factory()
        return MultinomialAccumulatorFactory(
            est_factory=est_factory, len_normalized=self.len_normalized, len_factory=len_factory, keys=self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[SS1, SS2 | None]) -> MultinomialDistribution:
        """Estimate a multinomial distribution from aggregated sufficient statistics.

        Args:
            nobs: Number of observations represented by ``suff_stat``.
            suff_stat: Tuple of value-model and trial-count sufficient statistics.

        Returns:
            A fitted multinomial distribution.

        """
        len_dist = self.len_estimator.estimate(nobs, suff_stat[1]) if self.len_dist is None else self.len_dist
        dist = self.estimator.estimate(nobs, suff_stat[0])
        return MultinomialDistribution(dist=dist, len_dist=len_dist, len_normalized=self.len_normalized, name=self.name)


class MultinomialDataEncoder(DataSequenceEncoder):
    """Encode sparse categorical multinomial observations for vectorized scoring."""

    def __init__(
        self,
        encoder: DataSequenceEncoder,
        len_encoder: DataSequenceEncoder,
        len_normalized: bool = False,
    ) -> None:
        """Create an encoder for sparse multinomial observations.

        ``encoder`` handles individual trial values and ``len_encoder`` handles
        total trial counts.

        Args:
            encoder: Encoder for individual trial values.
            len_encoder: Encoder for total trial count.

        Attributes:
            encoder: Encoder for individual trial values.
            len_encoder: Encoder for total trial count.

        """
        self.encoder = encoder
        self.len_encoder = len_encoder
        if not isinstance(len_normalized, bool):
            raise TypeError("multinomial encoder len_normalized must be a bool")
        self.len_normalized = len_normalized

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` uses the same length encoder.

        Args:
            other: Object to compare.

        Returns:
            ``True`` when ``other`` is a ``MultinomialDataEncoder`` with an
            equivalent length encoder.

        """
        return (
            isinstance(other, MultinomialDataEncoder)
            and other.encoder == self.encoder
            and other.len_encoder == self.len_encoder
            and other.len_normalized == self.len_normalized
        )

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "MultinomialDataEncoder(encoder=%s, len_encoder=%s, len_normalized=%r)" % (
            self.encoder,
            self.len_encoder,
            self.len_normalized,
        )

    def seq_encode(self, x: Sequence[Sequence[tuple[T, float]]]):
        """Encode iid multinomial observations for vectorized ``seq_*`` methods.

        The returned tuple contains:
            rv1 (ndarray[int]): Observation index of sequence values.
            rv2 (ndarray[float]): Trial size for each observation.
            rv3 (ndarray[float]): Non-zero trial size indices.
            rv4 (T1): Sequence encoded flattened list of values from x.
            rv5 (Optional[T2]): Sequence encoded flatted list of trial sizes.
            rv6 (np.ndarray[float]): Flattened array of counts for values.
            rv7 (ndarray[float]): Flattened array of trial sizes.

        Args:
            x (Sequence[Sequence[Tuple[T, float]]]): Sequence of iid observations of multinomial distributions.

        Returns:
            See above.

        """
        tx = []
        tidx = []
        cc = []
        ccc = []

        maximum = int(np.iinfo(np.int64).max)
        for i in range(len(x)):
            pairs, total = canonical_bag(x[i])
            if total > maximum:
                raise ValueError("multinomial row total exceeds signed 64-bit range")
            for value, count in pairs:
                tidx.append(i)
                tx.append(value)
                cc.append(count)
            ccc.append(total)

        rv1 = np.asarray(tidx, dtype=np.int64)
        rv2 = np.asarray(ccc, dtype=np.float64)
        rv3 = rv2 != 0
        rv6 = np.asarray(cc, dtype=np.int64)
        rv7 = np.asarray(ccc, dtype=np.int64)

        rv2[rv3] = 1.0 / rv2[rv3]
        # rv2[rv3] = 1.0

        rv4 = self.encoder.seq_encode(tx)

        if self.len_encoder is not None:
            rv5 = self.len_encoder.seq_encode(ccc)
        else:
            rv5 = None

        return MultinomialEncodedData(
            rv1,
            rv2,
            rv3,
            rv4,
            rv5,
            rv6,
            rv7,
            self.len_normalized,
        )

    def row_count(self, x: Any) -> int:
        """Return the number of bag observations in a validated encoding."""
        return _encoded(
            x,
            len_normalized=self.len_normalized,
            value_encoder=self.encoder,
            length_encoder=self.len_encoder,
        ).rows


# --- Fisher view(s) co-located with this family ---
class MultinomialFisherView(SequenceFisherView):
    """Fisher view for bag/count observations with a count-weighted child model.

    The model Fisher uses the canonical multinomial/count sufficient-statistic
    moments that match estimation. The multinomial coefficient is a
    base-measure term, not an accumulator statistic.
    """

    def _labels_from_children(self) -> list[Path]:
        labels = [("value",) + label for label in self.child_view.vectorizer.labels]
        if self.len_view is not None:
            labels.extend(("length",) + label for label in self.len_view.vectorizer.labels)
        return labels

    def _aggregate_weighted_flat(
        self, flat_stats: np.ndarray, idx: np.ndarray, counts: np.ndarray, totals: np.ndarray
    ) -> np.ndarray:
        out = np.zeros((len(totals), flat_stats.shape[1]), dtype=np.float64)
        if len(idx) == 0:
            return out
        weights = np.asarray(counts, dtype=np.float64)
        if self.dist.len_normalized:
            totals = np.asarray(totals, dtype=np.float64)
            inv = np.zeros_like(totals, dtype=np.float64)
            nz = totals != 0.0
            inv[nz] = 1.0 / totals[nz]
            weights = weights * inv[idx]
        np.add.at(out, idx, flat_stats * weights[:, None])
        return out

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        enc_data = _encoded(
            enc_data,
            len_normalized=self.dist.len_normalized,
            value_encoder=self.dist.dist.dist_to_encoder(),
            length_encoder=self.dist.len_dist.dist_to_encoder(),
        )
        idx, _, _, enc_seq, enc_len, counts, totals = enc_data
        idx = np.asarray(idx, dtype=np.int64)
        totals = np.asarray(totals, dtype=np.float64)
        if len(idx):
            flat = self.child_view.seq_expected_statistics(enc_seq)
            elem = self._aggregate_weighted_flat(flat, idx, np.asarray(counts, dtype=np.float64), totals)
        else:
            elem = np.zeros((len(totals), len(self.child_view.mean_statistics())), dtype=np.float64)
        blocks = [elem]
        if self.len_view is not None:
            blocks.append(self.len_view.seq_expected_statistics(enc_len))
        self._refresh_labels()
        return np.hstack(blocks) if blocks else np.zeros((len(totals), 0), dtype=np.float64)
