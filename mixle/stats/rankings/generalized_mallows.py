"""Generalized Mallows distribution over permutations under a choice of distance metric.

The Mallows family concentrates probability around a central permutation ``sigma0`` with dispersion
``theta >= 0``:

    p(sigma) = exp(-theta * d(sigma, sigma0)) / Z(theta, n),

generalizing :class:`~mixle.stats.rankings.mallows.MallowsDistribution` (Kendall-only) to any of the six
metrics in :mod:`mixle.stats.rankings._permutation_kernels`. The per-datum distance is the numba kernel;
this module supplies the metric-specific normalizer ``Z``. Three metrics have a closed-form (fast-DP)
normalizer and moment ``E_theta[d]``:

    kendall   Z = prod_{i=1}^{n-1} (1 - phi^{i+1}) / (1 - phi)          (phi = e^{-theta})
    cayley    Z = prod_{i=1}^{n-1} (1 + i phi)
    hamming   Z = sum_{m=0}^{n} C(n, m) D_m phi^m                       (D_m = subfactorial)

The other three are #P-hard and use exact algorithms within explicit resource caps::

    footrule  Z = perm(phi^{|i-j|})    exact log-domain subset permanent
    spearman  Z = perm(phi^{(i-j)^2})  exact log-domain subset permanent
    ulam      Z = sum over the exact LIS-distance histogram

Construction fails closed above a cap; a Monte-Carlo estimate is never installed
as a probability normalizer. The retained ``n_mc`` and ``seed`` arguments are
compatibility metadata only and are reported as such in computation diagnostics.

Data type: ``List[int]`` -- a full ordering, a permutation of ``0..n-1`` with ``x[r]`` the item at rank
``r`` (best first).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.rankings._contracts import (
    assignment_statistics,
    count_matrix_statistics,
    finite_nonnegative,
    log_truncated_geometric_normalizer,
    nonnegative_integer,
    permutation,
    permutation_batch,
    positive_integer,
    sample_size,
    truncated_geometric_mean,
)
from mixle.stats.rankings._contracts import (
    weights as validate_weights,
)
from mixle.stats.rankings._permutation_kernels import (
    METRICS,
    log_matrix_permanent,
    metric_id,
    seq_distance_to_center,
)
from mixle.utils.exact import require_exact_bool

_MAX_THETA = 700.0
_CLOSED_FORM = ("kendall", "cayley", "hamming")


@dataclass(frozen=True)
class GeneralizedMallowsFitDiagnostics:
    """Center-search and regularization evidence attached to a fitted model."""

    center_algorithm: str
    center_exact: bool
    centers_evaluated: int
    distance_objective: float
    regularized: bool
    pseudo_count: float

    # Every GeneralizedMallowsEstimator.estimate() fit attaches this unconditionally, so without
    # this flag to_serializable()/to_json()/model_hash() raised an unhandled SerializationError for
    # EVERY fitted instance of this family (campaign nine, D-0209) -- not registered anywhere else,
    # unlike the sibling mechanism this mirrors (ThurstoneFitDiagnostics, SpearmanRankingFitDiagnostics).
    # Unannotated on purpose: an annotated name would become a dataclass field.
    __pysp_serializable__ = True


@dataclass(frozen=True)
class GeneralizedMallowsComputationDiagnostics:
    """Exactness and algorithm provenance for scoring and sampling."""

    normalizer_algorithm: str
    normalizer_exact: bool
    sampler_algorithm: str
    sampler_exact: bool
    legacy_n_mc_ignored: int
    legacy_seed_ignored: int

    # Every GeneralizedMallowsDistribution.__init__ attaches this unconditionally (not an optional
    # constructor argument), so without this flag to_serializable()/to_json()/model_hash() raised an
    # unhandled SerializationError for EVERY instance of this family, fitted or not (campaign nine,
    # D-0209). Unannotated on purpose: an annotated name would become a dataclass field.
    __pysp_serializable__ = True


# --- metric-specific normalizer and expected distance --------------------------------------------
def _log_subfactorials(n: int) -> np.ndarray:
    """``log D_m`` for ``m = 0..n`` (rencontres / derangement counts; ``D_1 = 0`` -> ``-inf``)."""
    prev2, prev1 = 1.0, 0.0  # D_0, D_1
    logs = [0.0, -np.inf]
    for m in range(2, n + 1):
        cur = (m - 1) * (prev1 + prev2)
        logs.append(math.log(cur))
        prev2, prev1 = prev1, cur
    return np.asarray(logs[: n + 1], dtype=float)


def log_normalizer(metric: str, theta: float, n: int) -> float:
    """Return ``log Z(theta, n)`` for a closed-form metric."""
    if n <= 1:
        return 0.0
    if theta <= 0.0:
        return float(math.lgamma(n + 1))  # uniform: Z = n!
    phi = math.exp(-theta)
    if metric == "kendall":
        return float(sum(log_truncated_geometric_normalizer(theta, i) for i in range(1, n)))
    if metric == "cayley":
        return float(sum(math.log1p(i * phi) for i in range(1, n)))
    if metric == "hamming":
        m = np.arange(n + 1)
        log_binom = math.lgamma(n + 1) - np.array([math.lgamma(k + 1) + math.lgamma(n - k + 1) for k in m])
        terms = log_binom + _log_subfactorials(n) - theta * m
        return float(logsumexp(terms))
    raise ValueError(f"log_normalizer: metric {metric!r} has no closed form; use metric_log_normalizer().")


def expected_distance(metric: str, theta: float, n: int) -> float:
    """Return ``E_theta[d]`` for a closed-form metric (uniform value at ``theta = 0``)."""
    if n <= 1:
        return 0.0
    if theta <= 1e-7:  # uniform limits for the non-Kendall closed forms
        if metric == "kendall":
            return float(sum(truncated_geometric_mean(theta, i) for i in range(1, n)))
        if metric == "cayley":
            return float(sum(i / (1.0 + i) for i in range(1, n)))
        if metric == "hamming":
            return float(n - 1)  # E[fixed points] = 1 under the uniform permutation
    phi = math.exp(-theta)
    if metric == "kendall":
        return float(sum(truncated_geometric_mean(theta, i) for i in range(1, n)))
    if metric == "cayley":
        return float(sum(i * phi / (1.0 + i * phi) for i in range(1, n)))
    if metric == "hamming":
        m = np.arange(n + 1)
        log_binom = math.lgamma(n + 1) - np.array([math.lgamma(k + 1) + math.lgamma(n - k + 1) for k in m])
        terms = log_binom + _log_subfactorials(n) - theta * m
        return float(np.sum(m * np.exp(terms - logsumexp(terms))))
    raise ValueError(f"expected_distance: metric {metric!r} has no closed form.")


def solve_theta(metric: str, mean_distance: float, n: int) -> float:
    """Return the ``theta`` whose ``E_theta[d]`` matches ``mean_distance`` (monotone bisection)."""
    uniform_mean = expected_distance(metric, 0.0, n)
    if n <= 1 or mean_distance >= uniform_mean:
        return 0.0
    if mean_distance <= 0.0:
        return _MAX_THETA
    lo, hi = 0.0, 1.0
    while expected_distance(metric, hi, n) > mean_distance and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if expected_distance(metric, mid, n) > mean_distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=16)
def _exact_histogram(metric: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Full enumeration of S_n: distinct distance values and their counts (small n only)."""
    import itertools

    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    ds = seq_distance_to_center(perms, np.arange(n, dtype=np.int64), metric)
    vals, counts = np.unique(ds, return_counts=True)
    values, multiplicities = vals.astype(float), counts.astype(float)
    values.setflags(write=False)
    multiplicities.setflags(write=False)
    return values, multiplicities


def _assignment_distance_matrix(metric: str, n: int) -> np.ndarray:
    ranks = np.arange(n)
    if metric == "hamming":
        return (ranks[:, None] != ranks[None, :]).astype(np.float64)
    if metric == "footrule":
        return np.abs(ranks[:, None] - ranks[None, :]).astype(np.float64)
    if metric == "spearman":
        return ((ranks[:, None] - ranks[None, :]) ** 2).astype(np.float64)
    raise ValueError(f"{metric!r} is not an assignment-additive ranking metric.")


def _assignment_expected_distance(metric: str, theta: float, n: int) -> float:
    distance = _assignment_distance_matrix(metric, n)
    scores = -theta * distance
    log_z = log_matrix_permanent(scores)
    indices = np.arange(n)
    expected = 0.0
    for row in range(n):
        for column in range(n):
            minor = scores[np.ix_(indices[indices != row], indices[indices != column])]
            marginal = math.exp(scores[row, column] + log_matrix_permanent(minor) - log_z)
            expected += marginal * distance[row, column]
    return expected


def metric_log_normalizer(
    metric: str, theta: float, n: int, *, n_mc: int = 20000, seed: int = 0, max_exact: int = 16, max_enum: int = 9
) -> float:
    """Return an exact ``log Z`` or reject an unsupported approximation request."""
    metric_id(metric)
    theta = finite_nonnegative(theta, label="theta")
    n = positive_integer(n, label="n")
    positive_integer(n_mc, label="n_mc")
    nonnegative_integer(seed, label="seed")
    max_exact = positive_integer(max_exact, label="max_exact")
    max_enum = positive_integer(max_enum, label="max_enum")
    if metric in _CLOSED_FORM:
        return log_normalizer(metric, theta, n)
    if n <= 1:
        return 0.0
    if theta <= 0.0:
        return float(math.lgamma(n + 1))
    if metric in ("footrule", "spearman"):
        if n > max_exact:
            raise ValueError(
                f"exact {metric} normalization is capped at {max_exact} items; "
                "uncontrolled Monte Carlo normalization is not a probability contract."
            )
        return log_matrix_permanent(-theta * _assignment_distance_matrix(metric, n))
    if metric == "ulam":
        if n > max_enum:
            raise ValueError(
                f"exact Ulam normalization is capped at {max_enum} items; "
                "uncontrolled Monte Carlo normalization is not a probability contract."
            )
        vals, counts = _exact_histogram(metric, n)
        return float(logsumexp(np.log(counts) - theta * vals))
    raise ValueError(f"unsupported ranking metric {metric!r}.")


def metric_solve_theta(
    metric: str,
    mean_distance: float,
    n: int,
    *,
    n_mc: int = 20000,
    seed: int = 0,
    max_exact: int = 16,
    max_enum: int = 9,
) -> float:
    """Fit ``theta`` using exact moments, rejecting uncontrolled Monte Carlo fallbacks."""
    metric_id(metric)
    mean_distance = finite_nonnegative(mean_distance, label="mean_distance")
    n = positive_integer(n, label="n")
    positive_integer(n_mc, label="n_mc")
    nonnegative_integer(seed, label="seed")
    max_exact = positive_integer(max_exact, label="max_exact")
    max_enum = positive_integer(max_enum, label="max_enum")
    if metric in _CLOSED_FORM:
        return solve_theta(metric, mean_distance, n)
    if metric in ("footrule", "spearman"):
        if n > max_exact:
            raise ValueError(f"exact {metric} moment fitting is capped at {max_exact} items.")
        uniform_mean = _assignment_expected_distance(metric, 0.0, n)
        minimum = 0.0

        def e_dist(theta: float) -> float:
            return _assignment_expected_distance(metric, theta, n)

    elif metric == "ulam":
        if n > max_enum:
            raise ValueError(f"exact Ulam moment fitting is capped at {max_enum} items.")
        d_pop, w_pop = _exact_histogram(metric, n)
        log_w = np.log(w_pop)
        uniform_mean = float(np.sum(d_pop * np.exp(log_w - logsumexp(log_w))))
        minimum = float(d_pop.min())

        def e_dist(theta: float) -> float:
            lw = -theta * d_pop + log_w
            return float(np.sum(d_pop * np.exp(lw - logsumexp(lw))))

    else:
        raise ValueError(f"unsupported ranking metric {metric!r}.")
    if n <= 1 or mean_distance >= uniform_mean:
        return 0.0
    if mean_distance <= minimum:
        return _MAX_THETA

    lo, hi = 0.0, 1.0
    while e_dist(hi) > mean_distance and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if e_dist(mid) > mean_distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class GeneralizedMallowsDistribution(SequenceEncodableProbabilityDistribution):
    """Mallows distribution under a configurable distance ``metric`` (closed-form normalizer metrics)."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by Mallows kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Permutation distances run through dedicated numba kernels, not a compute engine.",
        )

    def __init__(
        self,
        sigma0: Sequence[int] | np.ndarray,
        theta: float = 1.0,
        metric: str = "kendall",
        name: str | None = None,
        keys: str | None = None,
        n_mc: int = 20000,
        seed: int = 0,
        max_exact: int = 16,
        max_enum: int = 9,
        fit_diagnostics: GeneralizedMallowsFitDiagnostics | None = None,
    ) -> None:
        raw_center = np.asarray(sigma0)
        if raw_center.ndim != 1 or len(raw_center) < 2:
            raise ValueError("sigma0 must be a permutation with at least two items.")
        n = len(raw_center)
        self.sigma0 = permutation(raw_center, n, label="sigma0").copy()
        self.sigma0.setflags(write=False)
        self.theta = finite_nonnegative(theta, label="theta")
        metric_id(metric)
        self.metric = metric
        self.dim = n
        self.rank0 = np.empty(n, dtype=np.int64)
        self.rank0[self.sigma0] = np.arange(n)
        self.rank0.setflags(write=False)
        self.n_mc = positive_integer(n_mc, label="n_mc")
        self.seed = nonnegative_integer(seed, label="seed")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1].")
        self.max_exact = positive_integer(max_exact, label="max_exact")
        if self.max_exact > 22:
            raise ValueError("max_exact must not exceed the exact permanent implementation limit of 22.")
        self.max_enum = positive_integer(max_enum, label="max_enum")
        self.log_z = metric_log_normalizer(
            metric, self.theta, n, n_mc=self.n_mc, seed=self.seed, max_exact=self.max_exact, max_enum=self.max_enum
        )
        normalizer_algorithm = {
            "kendall": "closed_form_repeated_insertion",
            "cayley": "closed_form_cycle_index",
            "hamming": "closed_form_rencontres",
            "footrule": "exact_log_permanent",
            "spearman": "exact_log_permanent",
            "ulam": "exact_distance_histogram",
        }[metric]
        sampler_algorithm = {
            "kendall": "exact_repeated_insertion",
            "cayley": "exact_ewens_insertion",
            "hamming": "exact_conditional_permanent",
            "footrule": "exact_conditional_permanent",
            "spearman": "exact_conditional_permanent",
            "ulam": "exact_enumerated_categorical",
        }[metric]
        self.computation_diagnostics = GeneralizedMallowsComputationDiagnostics(
            normalizer_algorithm=normalizer_algorithm,
            normalizer_exact=True,
            sampler_algorithm=sampler_algorithm,
            sampler_exact=True,
            legacy_n_mc_ignored=self.n_mc,
            legacy_seed_ignored=self.seed,
        )
        self.name = name
        self.keys = keys
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, GeneralizedMallowsFitDiagnostics):
            raise TypeError("fit_diagnostics must be a GeneralizedMallowsFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics

    def __str__(self) -> str:
        return "GeneralizedMallowsDistribution(%s, theta=%s, metric=%s, name=%s, keys=%s)" % (
            repr([int(v) for v in self.sigma0]),
            repr(self.theta),
            repr(self.metric),
            repr(self.name),
            repr(self.keys),
        )

    def distance(self, x: Sequence[int]) -> int:
        """Distance between ordering ``x`` and the central permutation under this metric.

        Raises:
            ValueError: If x is not a permutation of 0,...,n-1 (wrong length, a repeated
                element, a non-integer entry, or an element outside that range). A malformed x
                is not merely an unlikely ordering -- it isn't an ordering at all, so it is
                rejected here rather than silently scored (e.g. a repeated index would otherwise
                still produce a finite, meaningless distance, an out-of-range index can alias a
                valid rank via numpy's negative-index wraparound instead of failing loudly, and
                a fractional entry like 0.5 would otherwise be silently truncated to 0 by the
                int cast before this check ever saw it).

        """
        checked = permutation(x, self.dim, label="generalized Mallows ordering")
        return int(seq_distance_to_center(checked[None, :], self.rank0, self.metric)[0])

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of one ordering under the Mallows model."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of one ordering."""
        return -self.theta * self.distance(x) - self.log_z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings."""
        checked = permutation_batch(x, self.dim, label="generalized Mallows orderings")
        dist = seq_distance_to_center(checked, self.rank0, self.metric)
        return -self.theta * dist - self.log_z

    def sampler(self, seed: int | None = None) -> GeneralizedMallowsSampler:
        """Return a sampler for this Mallows distribution."""
        return GeneralizedMallowsSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GeneralizedMallowsEstimator:
        """Return a Mallows estimator with this metric and item count."""
        return GeneralizedMallowsEstimator(
            dim=self.dim,
            metric=self.metric,
            name=self.name,
            keys=self.keys,
            n_mc=self.n_mc,
            seed=self.seed,
            max_exact=self.max_exact,
            max_enum=self.max_enum,
            pseudo_count=pseudo_count,
            prior_center=self.sigma0,
        )

    def dist_to_encoder(self) -> GeneralizedMallowsDataEncoder:
        """Return the full-ranking encoder used by vectorized methods."""
        return GeneralizedMallowsDataEncoder(dim=self.dim)


class GeneralizedMallowsSampler(DistributionSampler):
    """Draw exact iid orderings using metric-specific finite-law samplers."""

    def __init__(self, dist: GeneralizedMallowsDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self._ulam_support: np.ndarray | None = None
        self._ulam_probabilities: np.ndarray | None = None
        self._assignment_probabilities: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def _sample_kendall(self, size: int) -> list[list[int]]:
        n, phi, theta = self.dist.dim, math.exp(-self.dist.theta), self.dist.theta
        out = []
        for _ in range(size):
            perm: list[int] = []
            for i in range(n):
                if theta <= 0.0:
                    j = self.rng.randint(0, i + 1)
                else:
                    cdf = np.cumsum(phi ** np.arange(i + 1))
                    j = int(np.searchsorted(cdf, self.rng.rand() * cdf[-1]))
                perm.insert(i - j, int(self.dist.sigma0[i]))
            out.append(perm)
        return out

    def _sample_cayley(self, size: int) -> list[list[int]]:
        output: list[list[int]] = []
        exp_negative_theta = math.exp(-self.dist.theta)
        for _ in range(size):
            relative = np.empty(self.dist.dim, dtype=np.int64)
            for index in range(self.dist.dim):
                new_cycle_probability = 1.0 / (1.0 + index * exp_negative_theta)
                if self.rng.rand() < new_cycle_probability:
                    relative[index] = index
                else:
                    predecessor = int(self.rng.randint(0, index))
                    relative[index] = relative[predecessor]
                    relative[predecessor] = index
            output.append([int(value) for value in self.dist.sigma0[relative]])
        return output

    def _sample_assignment(self, size: int) -> list[list[int]]:
        if self.dist.dim > self.dist.max_exact:
            raise ValueError(f"exact {self.dist.metric} sampling is capped at {self.dist.max_exact} items.")
        distance = _assignment_distance_matrix(self.dist.metric, self.dist.dim)
        scores = -self.dist.theta * distance[self.dist.rank0]
        output: list[list[int]] = []
        for _ in range(size):
            available = list(range(self.dist.dim))
            ordering: list[int] = []
            for rank in range(self.dist.dim):
                remaining_ranks = list(range(rank + 1, self.dist.dim))
                cache_key = (rank, tuple(available))
                probabilities = self._assignment_probabilities.get(cache_key)
                if probabilities is None:
                    log_probabilities = np.empty(len(available), dtype=np.float64)
                    for index, item in enumerate(available):
                        remaining_items = [candidate for candidate in available if candidate != item]
                        minor = scores[np.ix_(remaining_items, remaining_ranks)]
                        log_probabilities[index] = scores[item, rank] + log_matrix_permanent(minor)
                    shift = float(np.max(log_probabilities))
                    probabilities = np.exp(log_probabilities - shift)
                    probabilities /= probabilities.sum()
                    probabilities.setflags(write=False)
                    self._assignment_probabilities[cache_key] = probabilities
                selected = int(self.rng.choice(len(available), p=probabilities))
                ordering.append(available.pop(selected))
            output.append(ordering)
        return output

    def _sample_ulam(self, size: int) -> list[list[int]]:
        if self._ulam_support is None:
            import itertools

            support = np.asarray(list(itertools.permutations(range(self.dist.dim))), dtype=np.int64)
            log_probabilities = self.dist.seq_log_density(support)
            probabilities = np.exp(log_probabilities)
            probabilities /= probabilities.sum()
            self._ulam_support = support
            self._ulam_probabilities = probabilities
        indices = self.rng.choice(len(self._ulam_support), size=size, p=self._ulam_probabilities)
        return [[int(value) for value in self._ulam_support[index]] for index in indices]

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` iid orderings."""
        checked_size = sample_size(size)
        k = 1 if checked_size is None else checked_size
        if self.dist.metric == "kendall":
            draws = self._sample_kendall(k)
        elif self.dist.metric == "cayley":
            draws = self._sample_cayley(k)
        elif self.dist.metric in ("hamming", "footrule", "spearman"):
            draws = self._sample_assignment(k)
        else:
            draws = self._sample_ulam(k)
        return draws[0] if checked_size is None else draws


def _generalized_statistics(value, dim: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate consensus matrices and the exact weighted empirical rows that generated them."""
    if isinstance(value, (str, bytes)) or len(value) != 5:
        raise TypeError("generalized Mallows statistics must be a five-item tuple.")
    count, rank_count = assignment_statistics(
        (value[0], value[1]),
        dim,
        label="generalized Mallows rank statistics",
    )
    _, precede = count_matrix_statistics(
        (value[0], value[2]),
        dim,
        label="generalized Mallows precedence statistics",
        entries_per_observation=dim * (dim - 1) / 2.0,
    )
    if count == 0.0:
        if len(value[3]) != 0 or len(value[4]) != 0:
            raise ValueError("zero-weight generalized Mallows statistics require empty empirical rows.")
        return count, rank_count, precede, np.empty((0, dim), dtype=np.int64), np.empty(0)
    rows = permutation_batch(value[3], dim, label="generalized Mallows empirical rows", allow_empty=False)
    row_weights = validate_weights(value[4], len(rows))
    tolerance = 1.0e-10 * max(1.0, count)
    if not np.isclose(row_weights.sum(), count, rtol=1.0e-10, atol=tolerance):
        raise ValueError("generalized Mallows empirical weights must sum to total weight.")
    expected_rank = np.zeros((dim, dim))
    expected_precede = np.zeros((dim, dim))
    ranks = np.arange(dim)
    earlier, later = np.triu_indices(dim, 1)
    for row, weight in zip(rows, row_weights):
        np.add.at(expected_rank, (row, ranks), weight)
        np.add.at(expected_precede, (row[earlier], row[later]), weight)
    if not np.allclose(expected_rank, rank_count, rtol=1.0e-10, atol=tolerance):
        raise ValueError("generalized Mallows rank counts disagree with empirical rows.")
    if not np.allclose(expected_precede, precede, rtol=1.0e-10, atol=tolerance):
        raise ValueError("generalized Mallows precedence counts disagree with empirical rows.")
    return count, rank_count, precede, rows, row_weights


class GeneralizedMallowsAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the rank-count matrix, the precede matrix, and a bounded reservoir of orderings.

    ``rank_count[item, rank]`` and ``precede[a, b]`` give the consensus (Borda / Copeland); the reservoir
    supplies the empirical mean metric-distance to the fitted center (exact when the data fit in it).
    """

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.rank_count = np.zeros((self.dim, self.dim))
        self.precede = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.reservoir = positive_integer(reservoir, label="reservoir")
        self._res_x: list[np.ndarray] = []
        self._res_w: list[float] = []
        self.keys = keys

    def _push(self, row: np.ndarray, w: float) -> None:
        if len(self._res_x) >= self.reservoir:
            raise MemoryError(
                "generalized Mallows empirical row limit exceeded; increase reservoir rather than dropping evidence."
            )
        self._res_x.append(np.asarray(row, dtype=np.int64).copy())
        self._res_w.append(float(w))

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update rank, precedence, and reservoir statistics from one ordering."""
        checked = permutation(x, self.dim, label="generalized Mallows ordering")
        self.seq_update(checked[None, :], np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one weighted ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update rank, precedence, and reservoir statistics from encoded orderings."""
        checked = permutation_batch(x, self.dim, label="generalized Mallows orderings")
        checked_weights = validate_weights(weights, len(checked))
        if len(self._res_x) + len(checked) > self.reservoir:
            raise MemoryError(
                "generalized Mallows empirical row limit exceeded; increase reservoir rather than dropping evidence."
            )
        n = self.dim
        r_idx, rp_idx = np.triu_indices(n, 1)
        ranks = np.arange(n)
        for row, w in zip(checked, checked_weights):
            np.add.at(self.rank_count, (row, ranks), w)  # item row[r] got rank r
            np.add.at(self.precede, (row[r_idx], row[rp_idx]), w)
            self._push(row, w)
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> GeneralizedMallowsAccumulator:
        """Merge consensus statistics and reservoir samples from another accumulator."""
        count, rank_count, precede, res_x, res_w = _generalized_statistics(suff_stat, self.dim)
        if len(self._res_x) + len(res_x) > self.reservoir:
            raise MemoryError(
                "merged generalized Mallows rows exceed reservoir; increase it rather than biasing the merge."
            )
        self.count += count
        self.rank_count += rank_count
        self.precede += precede
        for row, w in zip(res_x, res_w):
            self._push(row, float(w))
        return self

    def value(self):
        """Return count, consensus matrices, and bounded reservoir contents."""
        return (
            self.count,
            self.rank_count.copy(),
            self.precede.copy(),
            [np.asarray(r).copy() for r in self._res_x],
            list(self._res_w),
        )

    def from_value(self, x) -> GeneralizedMallowsAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.rank_count, self.precede, res_x, res_w = _generalized_statistics(x, self.dim)
        if len(res_x) > self.reservoir:
            raise MemoryError("restored generalized Mallows rows exceed reservoir.")
        self._res_x = [np.asarray(r, dtype=np.int64).copy() for r in res_x]
        self._res_w = [float(w) for w in res_w]
        return self

    def acc_to_encoder(self) -> GeneralizedMallowsDataEncoder:
        """Return the ranking encoder compatible with these sufficient statistics."""
        return GeneralizedMallowsDataEncoder(dim=self.dim)


class GeneralizedMallowsAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for generalized Mallows consensus statistics."""

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.reservoir = positive_integer(reservoir, label="reservoir")
        self.keys = keys

    def make(self) -> GeneralizedMallowsAccumulator:
        """Create an empty generalized Mallows accumulator."""
        return GeneralizedMallowsAccumulator(dim=self.dim, reservoir=self.reservoir, keys=self.keys)


class GeneralizedMallowsEstimator(ParameterEstimator):
    """Estimate the central permutation (Copeland/Borda) and dispersion theta (moment match)."""

    def __init__(
        self,
        dim: int,
        metric: str = "kendall",
        theta: float | None = None,
        reservoir: int = 10000,
        name: str | None = None,
        keys: str | None = None,
        n_mc: int = 20000,
        seed: int = 0,
        max_exact: int = 16,
        max_enum: int = 9,
        pseudo_count: float | None = None,
        prior_center: Sequence[int] | np.ndarray | None = None,
        center_exact_cap: int = 7,
        allow_approximate_center: bool = False,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        metric_id(metric)
        self.metric = metric
        self.theta = None if theta is None else finite_nonnegative(theta, label="theta")
        self.reservoir = positive_integer(reservoir, label="reservoir")
        self.name = name
        self.keys = keys
        self.n_mc = positive_integer(n_mc, label="n_mc")
        self.seed = nonnegative_integer(seed, label="seed")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1].")
        self.max_exact = positive_integer(max_exact, label="max_exact")
        if self.max_exact > 22:
            raise ValueError("max_exact must not exceed 22.")
        self.max_enum = positive_integer(max_enum, label="max_enum")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        self.prior_center = (
            np.arange(self.dim, dtype=np.int64)
            if prior_center is None
            else permutation(prior_center, self.dim, label="prior_center").copy()
        )
        self.center_exact_cap = positive_integer(center_exact_cap, label="center_exact_cap", minimum=2)
        if not isinstance(allow_approximate_center, (bool, np.bool_)):
            raise TypeError("allow_approximate_center must be a Boolean.")
        self.allow_approximate_center = require_exact_bool(allow_approximate_center, "allow_approximate_center")

    def accumulator_factory(self) -> GeneralizedMallowsAccumulatorFactory:
        """Return a factory for Mallows sufficient-statistic accumulators."""
        return GeneralizedMallowsAccumulatorFactory(dim=self.dim, reservoir=self.reservoir, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> GeneralizedMallowsDistribution:
        """Estimate the central ordering and dispersion from accumulated rankings."""
        if nobs is not None:
            finite_nonnegative(nobs, label="nobs")
        count, rank_count, precede, rows, row_weights = _generalized_statistics(suff_stat, self.dim)
        res_x = [row.copy() for row in rows]
        res_w = [float(weight) for weight in row_weights]
        n = self.dim
        kw = dict(
            name=self.name,
            keys=self.keys,
            n_mc=self.n_mc,
            seed=self.seed,
            max_exact=self.max_exact,
            max_enum=self.max_enum,
        )
        pseudo_count = 0.0 if self.pseudo_count is None else self.pseudo_count
        if pseudo_count > 0.0:
            res_x = [*res_x, self.prior_center.copy()]
            res_w = [*res_w, pseudo_count]
            ranks = np.arange(n)
            rank_count = np.asarray(rank_count, dtype=float).copy()
            precede = np.asarray(precede, dtype=float).copy()
            rank_count[self.prior_center, ranks] += pseudo_count
            earlier, later = np.triu_indices(n, 1)
            np.add.at(precede, (self.prior_center[earlier], self.prior_center[later]), pseudo_count)
            count += pseudo_count
        if count <= 0.0:
            return GeneralizedMallowsDistribution(np.arange(n), 0.0, self.metric, **kw)
        x = np.asarray(res_x, dtype=np.int64)
        w = np.asarray(res_w, dtype=float)
        unique_rows, inverse = np.unique(x, axis=0, return_inverse=True)
        unique_weights = np.bincount(inverse, weights=w, minlength=len(unique_rows))
        x, w = unique_rows, unique_weights
        center_exact = n <= self.center_exact_cap
        if not center_exact and not self.allow_approximate_center:
            raise ValueError(
                f"exact generalized Mallows center search is capped at {self.center_exact_cap} items; "
                "set allow_approximate_center=True for a labeled consensus approximation."
            )
        if center_exact:
            sigma0: np.ndarray | None = None
            distance_objective = math.inf
            centers_evaluated = 0
            for candidate_tuple in permutations(range(n)):
                candidate = np.asarray(candidate_tuple, dtype=np.int64)
                candidate_rank = np.empty(n, dtype=np.int64)
                candidate_rank[candidate] = np.arange(n)
                distances = seq_distance_to_center(x, candidate_rank, self.metric)
                objective = float(np.sum(distances * w))
                centers_evaluated += 1
                if objective < distance_objective:
                    sigma0, distance_objective = candidate, objective
            if sigma0 is None:
                raise RuntimeError("generalized Mallows exact center search produced no candidates.")
            center_algorithm = "exact_enumeration"
        else:
            if self.metric == "kendall":
                scores = precede.sum(axis=1) - precede.sum(axis=0)
            else:
                mean_rank = (rank_count * np.arange(n)[None, :]).sum(axis=1) / count
                scores = -mean_rank
            sigma0 = np.argsort(-scores, kind="stable")
            approximate_rank = np.empty(n, dtype=np.int64)
            approximate_rank[sigma0] = np.arange(n)
            distance_objective = float(np.sum(seq_distance_to_center(x, approximate_rank, self.metric) * w))
            centers_evaluated = 1
            center_algorithm = "copeland_approximation" if self.metric == "kendall" else "borda_approximation"
        rank0 = np.empty(n, dtype=np.int64)
        rank0[sigma0] = np.arange(n)

        if self.theta is not None:
            theta = self.theta
        else:
            dist = seq_distance_to_center(x, rank0, self.metric)
            mean_distance = float(np.sum(dist * w) / np.sum(w))
            theta = metric_solve_theta(
                self.metric,
                mean_distance,
                n,
                n_mc=self.n_mc,
                seed=self.seed,
                max_exact=self.max_exact,
                max_enum=self.max_enum,
            )
        diagnostics = GeneralizedMallowsFitDiagnostics(
            center_algorithm=center_algorithm,
            center_exact=center_exact,
            centers_evaluated=centers_evaluated,
            distance_objective=distance_objective,
            regularized=pseudo_count > 0.0,
            pseudo_count=pseudo_count,
        )
        return GeneralizedMallowsDistribution(
            sigma0,
            theta,
            self.metric,
            fit_diagnostics=diagnostics,
            **kw,
        )


class GeneralizedMallowsDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "GeneralizedMallowsDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedMallowsDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("GeneralizedMallowsDistribution requires a non-empty sequence of orderings.")
            return permutation_batch(raw, raw.shape[1], label="generalized Mallows orderings", allow_empty=False)
        return permutation_batch(raw, self.dim, label="generalized Mallows orderings", allow_empty=False)


__all__ = [
    "GeneralizedMallowsDistribution",
    "GeneralizedMallowsSampler",
    "GeneralizedMallowsAccumulator",
    "GeneralizedMallowsAccumulatorFactory",
    "GeneralizedMallowsEstimator",
    "GeneralizedMallowsDataEncoder",
    "GeneralizedMallowsFitDiagnostics",
    "GeneralizedMallowsComputationDiagnostics",
    "METRICS",
]
