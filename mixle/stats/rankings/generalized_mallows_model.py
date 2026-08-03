"""Generalized Mallows Model (GMM): Mallows with a separate dispersion per ranking stage.

The Generalized Mallows Model (Fligner & Verducci, 1986) refines the Kendall Mallows model by giving
each stage of the ranking its own dispersion. Writing an ordering's Repeated-Insertion-Model code
``J = (J_1, ..., J_{n-1})`` relative to the central permutation ``sigma0`` (``J_i in {0..i}`` is the
back-jump of central item ``i``; ``sum_i J_i`` is the Kendall distance), the GMM makes the stages
independent truncated-geometrics:

    p(sigma) = exp(-sum_i theta_i J_i(sigma)) / Z,    Z = prod_i psi_i(theta_i),
    psi_i(theta_i) = sum_{r=0}^{i} exp(-theta_i r) = (1 - exp(-theta_i (i+1))) / (1 - exp(-theta_i)).

Everything factorizes over stages, so the normalizer, the moments ``E[J_i]``, exact RIM sampling, and
maximum likelihood are all closed form and the per-datum statistic ``J`` is a numba kernel. The model
captures rankings that are firm at the top but loose at the bottom (decreasing ``theta_i``) and vice
versa -- structure a single-dispersion Mallows cannot represent.

Data type: ``List[int]`` -- a full ordering, a permutation of ``0..n-1`` with ``x[r]`` the item at rank
``r`` (best first).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.rankings._contracts import (
    count_matrix_statistics,
    finite_nonnegative,
    log_truncated_geometric_normalizer,
    permutation,
    permutation_batch,
    positive_integer,
    sample_size,
    truncated_geometric_mean,
)
from mixle.stats.rankings._contracts import (
    weights as validate_weights,
)
from mixle.stats.rankings._permutation_kernels import seq_rim_code
from mixle.utils.exact import require_exact_bool

_MAX_THETA = 700.0


@dataclass(frozen=True)
class GeneralizedMallowsModelFitDiagnostics:
    """Center-search and regularization evidence attached to a fitted stage-wise model."""

    center_algorithm: str
    center_exact: bool
    centers_evaluated: int
    objective: float
    regularized: bool
    pseudo_count: float


def _log_psi(theta: float, m: int) -> float:
    """``log sum_{r=0}^{m} exp(-theta r)`` -- log-normalizer of a stage with values ``{0..m}``."""
    return log_truncated_geometric_normalizer(theta, m)


def _stage_mean(theta: float, m: int) -> float:
    """``E_theta[J]`` for a stage truncated-geometric on ``{0..m}``."""
    return truncated_geometric_mean(theta, m)


def _solve_stage_theta(mean_j: float, m: int) -> float:
    """``theta`` whose stage mean matches ``mean_j`` on ``{0..m}`` (monotone bisection)."""
    if m <= 0 or mean_j >= m / 2.0:
        return 0.0
    if mean_j <= 0.0:
        return _MAX_THETA
    lo, hi = 0.0, 1.0
    while _stage_mean(hi, m) > mean_j and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _stage_mean(mid, m) > mean_j:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class GeneralizedMallowsModelDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Mallows Model: a Kendall Mallows model with a per-stage dispersion vector ``theta``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy execution path used by stage-wise Mallows kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="The RIM insertion code runs through a dedicated numba kernel, not a compute engine.",
        )

    def __init__(
        self,
        sigma0: Sequence[int] | np.ndarray,
        theta: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
        fit_diagnostics: GeneralizedMallowsModelFitDiagnostics | None = None,
    ) -> None:
        raw_center = np.asarray(sigma0)
        if raw_center.ndim != 1 or len(raw_center) < 2:
            raise ValueError("sigma0 must be a permutation with at least two items.")
        n = len(raw_center)
        s0 = permutation(raw_center, n, label="sigma0").copy()
        th = np.ones(n - 1) if theta is None else np.asarray(theta, dtype=float)
        if th.shape != (n - 1,) or np.any(th < 0.0) or not np.all(np.isfinite(th)):
            raise ValueError("theta must be a length-(n-1) vector of non-negative dispersions.")
        self.sigma0 = s0
        self.sigma0.setflags(write=False)
        self.theta = th.copy()
        self.theta.setflags(write=False)
        self.dim = n
        self.log_z = float(sum(_log_psi(float(self.theta[i - 1]), i) for i in range(1, n)))
        self.name = name
        self.keys = keys
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, GeneralizedMallowsModelFitDiagnostics):
            raise TypeError("fit_diagnostics must be a GeneralizedMallowsModelFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics

    def __str__(self) -> str:
        return "GeneralizedMallowsModelDistribution(%s, theta=%s, name=%s, keys=%s)" % (
            repr([int(v) for v in self.sigma0]),
            repr([float(v) for v in self.theta]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of one ordering."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of one ordering."""
        checked = permutation(x, self.dim, label="stage-wise Mallows ordering")
        return float(self.seq_log_density(checked[None, :])[0])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings.

        Raises:
            ValueError: If any row of x is not a permutation of 0,...,n-1 (wrong length, a
                repeated element, a non-integer entry, or an element outside that range). A
                malformed row is not merely an unlikely ordering -- it isn't an ordering at all,
                so it is rejected here rather than silently scored (e.g. a repeated index would
                otherwise still produce a finite, meaningless log-density, an out-of-range or
                negative index reaches the numba RIM kernel below, which has no bounds checking
                of its own and does not raise -- it silently scores too, rather than failing
                loudly -- and a fractional entry like 0.5 would otherwise be silently truncated
                to 0 by the int cast before this check ever saw it).

        """
        checked = permutation_batch(x, self.dim, label="stage-wise Mallows orderings")
        j = seq_rim_code(checked, self.sigma0)  # (N, n-1)
        return -(j @ self.theta) - self.log_z

    def sampler(self, seed: int | None = None) -> GeneralizedMallowsModelSampler:
        """Return an exact repeated-insertion sampler for this model."""
        return GeneralizedMallowsModelSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GeneralizedMallowsModelEstimator:
        """Return a stage-wise Mallows estimator with this item count."""
        return GeneralizedMallowsModelEstimator(
            dim=self.dim,
            pseudo_count=pseudo_count,
            prior_center=self.sigma0,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> GeneralizedMallowsModelDataEncoder:
        """Return the full-ranking encoder used by vectorized methods."""
        return GeneralizedMallowsModelDataEncoder(dim=self.dim)


class GeneralizedMallowsModelSampler(DistributionSampler):
    """Exact GMM draws via the per-stage Repeated Insertion Model."""

    def __init__(self, dist: GeneralizedMallowsModelDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self) -> list[int]:
        n = self.dist.dim
        perm: list[int] = []
        for i in range(n):
            if i == 0:
                j = 0
            else:
                theta = float(self.dist.theta[i - 1])
                if theta <= 0.0:
                    j = self.rng.randint(0, i + 1)
                else:  # truncated geometric on {0..i}, P(r) ∝ exp(-theta r)
                    cdf = np.cumsum(np.exp(-theta * np.arange(i + 1)))
                    j = int(np.searchsorted(cdf, self.rng.rand() * cdf[-1]))
            perm.insert(i - j, int(self.dist.sigma0[i]))
        return perm

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` iid orderings."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(sample_size(size))]


def _stage_statistics(value, dim: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact weighted empirical support and its derived precedence matrix."""
    if isinstance(value, (str, bytes)):
        raise TypeError("stage-wise Mallows statistics must be a four-item tuple.")
    try:
        if len(value) != 4:
            raise TypeError("stage-wise Mallows statistics must be a four-item tuple.")
    except TypeError as exc:
        raise TypeError("stage-wise Mallows statistics must be a four-item tuple.") from exc
    count, precede = count_matrix_statistics(
        (value[0], value[1]),
        dim,
        label="stage-wise Mallows precedence statistics",
        entries_per_observation=dim * (dim - 1) / 2.0,
    )
    raw_rows, raw_weights = value[2], value[3]
    if count == 0.0:
        if len(raw_rows) != 0 or len(raw_weights) != 0:
            raise ValueError("zero-weight stage-wise Mallows statistics must have empty empirical support.")
        return count, precede, np.empty((0, dim), dtype=np.int64), np.empty(0, dtype=np.float64)
    rows = permutation_batch(raw_rows, dim, label="stage-wise Mallows empirical support", allow_empty=False)
    row_weights = validate_weights(raw_weights, len(rows))
    if not np.isclose(row_weights.sum(), count, rtol=1.0e-10, atol=1.0e-10 * max(1.0, count)):
        raise ValueError("stage-wise Mallows support weights must sum to total observation weight.")
    expected = np.zeros((dim, dim), dtype=np.float64)
    earlier, later = np.triu_indices(dim, 1)
    for row, weight in zip(rows, row_weights):
        np.add.at(expected, (row[earlier], row[later]), weight)
    if not np.allclose(expected, precede, rtol=1.0e-10, atol=1.0e-10 * max(1.0, count)):
        raise ValueError("stage-wise Mallows precedence counts disagree with empirical support.")
    return count, precede, rows, row_weights


class GeneralizedMallowsModelAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate precedence counts and an exact bounded weighted empirical support."""

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.precede = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.reservoir = positive_integer(reservoir, label="reservoir")
        self._empirical: dict[tuple[int, ...], float] = {}
        self.keys = keys

    def _push(self, row: np.ndarray, weight: float) -> None:
        if weight == 0.0:
            return
        key = tuple(int(value) for value in row)
        if key not in self._empirical and len(self._empirical) >= self.reservoir:
            raise MemoryError(
                "stage-wise Mallows exact empirical support exceeded the configured reservoir limit; "
                "increase reservoir rather than silently dropping evidence."
            )
        self._empirical[key] = self._empirical.get(key, 0.0) + weight

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update consensus and reservoir statistics from one weighted ordering."""
        checked = permutation(x, self.dim, label="stage-wise Mallows ordering")
        self.seq_update(checked[None, :], np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize consensus and reservoir statistics from one ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update consensus and reservoir statistics from encoded orderings."""
        checked = permutation_batch(x, self.dim, label="stage-wise Mallows orderings")
        checked_weights = validate_weights(weights, len(checked))
        prospective = set(self._empirical)
        prospective.update(
            tuple(int(value) for value in row) for row, weight in zip(checked, checked_weights) if weight > 0.0
        )
        if len(prospective) > self.reservoir:
            raise MemoryError(
                "stage-wise Mallows exact empirical support exceeded the configured reservoir limit; "
                "increase reservoir rather than silently dropping evidence."
            )
        earlier, later = np.triu_indices(self.dim, 1)
        for row, weight in zip(checked, checked_weights):
            np.add.at(self.precede, (row[earlier], row[later]), weight)
            self._push(row, float(weight))
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> GeneralizedMallowsModelAccumulator:
        """Merge observation weight, precedence counts, and reservoir samples."""
        count, precede, rows, row_weights = _stage_statistics(suff_stat, self.dim)
        prospective = set(self._empirical)
        prospective.update(tuple(int(value) for value in row) for row in rows)
        if len(prospective) > self.reservoir:
            raise MemoryError(
                "merged stage-wise Mallows support exceeds the configured reservoir limit; "
                "increase reservoir rather than making the merge order-dependent."
            )
        self.count += count
        self.precede += precede
        for row, weight in zip(rows, row_weights):
            self._push(row, float(weight))
        return self

    def value(self):
        """Return count, precedence matrix, and bounded reservoir contents."""
        keys = sorted(self._empirical)
        return (
            self.count,
            self.precede.copy(),
            [np.asarray(key, dtype=np.int64) for key in keys],
            [self._empirical[key] for key in keys],
        )

    def from_value(self, x) -> GeneralizedMallowsModelAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.precede, rows, row_weights = _stage_statistics(x, self.dim)
        if len(rows) > self.reservoir:
            raise MemoryError("restored stage-wise Mallows support exceeds the configured reservoir limit.")
        self._empirical = {tuple(int(value) for value in row): float(weight) for row, weight in zip(rows, row_weights)}
        return self

    def acc_to_encoder(self) -> GeneralizedMallowsModelDataEncoder:
        """Return the encoder compatible with these sufficient statistics."""
        return GeneralizedMallowsModelDataEncoder(dim=self.dim)


class GeneralizedMallowsModelAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for stage-wise Generalized Mallows statistics."""

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.reservoir = positive_integer(reservoir, label="reservoir")
        self.keys = keys

    def make(self) -> GeneralizedMallowsModelAccumulator:
        """Create an empty stage-wise Generalized Mallows accumulator."""
        return GeneralizedMallowsModelAccumulator(dim=self.dim, reservoir=self.reservoir, keys=self.keys)


def _fit_stage_center(
    rows: np.ndarray,
    row_weights: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit stage dispersions for one center and return them with exact negative log likelihood."""
    count = float(row_weights.sum())
    code = seq_rim_code(rows, center)
    mean_code = (code * row_weights[:, None]).sum(axis=0) / count
    theta = np.asarray([_solve_stage_theta(float(mean_code[index - 1]), index) for index in range(1, len(center))])
    log_z = float(sum(_log_psi(float(theta[index - 1]), index) for index in range(1, len(center))))
    negative_log_likelihood = float(np.sum(row_weights * (code @ theta)) + count * log_z)
    return theta, negative_log_likelihood


class GeneralizedMallowsModelEstimator(ParameterEstimator):
    """Fit a center and per-stage dispersions from exact weighted empirical support."""

    def __init__(
        self,
        dim: int,
        reservoir: int = 10000,
        pseudo_count: float | None = None,
        prior_center: Sequence[int] | np.ndarray | None = None,
        center_exact_cap: int = 7,
        allow_approximate_center: bool = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.reservoir = positive_integer(reservoir, label="reservoir")
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
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedMallowsModelAccumulatorFactory:
        """Return a factory for stage-wise Mallows sufficient-statistic accumulators."""
        return GeneralizedMallowsModelAccumulatorFactory(dim=self.dim, reservoir=self.reservoir, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> GeneralizedMallowsModelDistribution:
        """Estimate central ordering and per-stage dispersions from accumulated rankings."""
        if nobs is not None:
            finite_nonnegative(nobs, label="nobs")
        count, precede, rows, row_weights = _stage_statistics(suff_stat, self.dim)
        n = self.dim
        pseudo_count = 0.0 if self.pseudo_count is None else self.pseudo_count
        if pseudo_count > 0.0:
            rows = np.vstack((rows, self.prior_center))
            row_weights = np.concatenate((row_weights, np.asarray([pseudo_count])))
            count += pseudo_count
            earlier, later = np.triu_indices(n, 1)
            np.add.at(precede, (self.prior_center[earlier], self.prior_center[later]), pseudo_count)
        if count <= 0.0:
            return GeneralizedMallowsModelDistribution(
                np.arange(n),
                np.zeros(n - 1),
                name=self.name,
                keys=self.keys,
            )

        exact = n <= self.center_exact_cap
        if not exact and not self.allow_approximate_center:
            raise ValueError(
                f"exact stage-wise Mallows center search is capped at {self.center_exact_cap} items; "
                "set allow_approximate_center=True to request a labeled Copeland approximation."
            )
        if exact:
            best_center: np.ndarray | None = None
            best_theta: np.ndarray | None = None
            best_objective = math.inf
            centers_evaluated = 0
            for candidate_tuple in permutations(range(n)):
                candidate = np.asarray(candidate_tuple, dtype=np.int64)
                theta, objective = _fit_stage_center(rows, row_weights, candidate)
                centers_evaluated += 1
                if objective < best_objective:
                    best_center, best_theta, best_objective = candidate, theta, objective
            if best_center is None or best_theta is None:
                raise RuntimeError("stage-wise Mallows exact center search produced no candidates.")
            center, theta = best_center, best_theta
            center_algorithm = "exact_enumeration"
        else:
            scores = precede.sum(axis=1) - precede.sum(axis=0)
            center = np.argsort(-scores, kind="stable").astype(np.int64)
            theta, best_objective = _fit_stage_center(rows, row_weights, center)
            centers_evaluated = 1
            center_algorithm = "copeland_approximation"
        diagnostics = GeneralizedMallowsModelFitDiagnostics(
            center_algorithm=center_algorithm,
            center_exact=exact,
            centers_evaluated=centers_evaluated,
            objective=best_objective,
            regularized=pseudo_count > 0.0,
            pseudo_count=pseudo_count,
        )
        return GeneralizedMallowsModelDistribution(
            center,
            theta,
            name=self.name,
            keys=self.keys,
            fit_diagnostics=diagnostics,
        )


class GeneralizedMallowsModelDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "GeneralizedMallowsModelDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedMallowsModelDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("GeneralizedMallowsModelDistribution requires a non-empty sequence of orderings.")
            return permutation_batch(raw, raw.shape[1], label="stage-wise Mallows orderings", allow_empty=False)
        return permutation_batch(raw, self.dim, label="stage-wise Mallows orderings", allow_empty=False)


__all__ = [
    "GeneralizedMallowsModelDistribution",
    "GeneralizedMallowsModelSampler",
    "GeneralizedMallowsModelAccumulator",
    "GeneralizedMallowsModelAccumulatorFactory",
    "GeneralizedMallowsModelEstimator",
    "GeneralizedMallowsModelDataEncoder",
    "GeneralizedMallowsModelFitDiagnostics",
]
