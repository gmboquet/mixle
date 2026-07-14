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
from mixle.stats.rankings._permutation_kernels import seq_rim_code

_MAX_THETA = 700.0


def _log_psi(theta: float, m: int) -> float:
    """``log sum_{r=0}^{m} exp(-theta r)`` -- log-normalizer of a stage with values ``{0..m}``."""
    if m <= 0:
        return 0.0
    if theta <= 0.0:
        return math.log(m + 1)
    phi = math.exp(-theta)
    if phi <= 0.0:
        return 0.0
    return math.log1p(-(phi ** (m + 1))) - math.log1p(-phi)


def _stage_mean(theta: float, m: int) -> float:
    """``E_theta[J]`` for a stage truncated-geometric on ``{0..m}``."""
    if m <= 0:
        return 0.0
    if theta <= 1e-7:
        return m / 2.0
    phi = math.exp(-theta)
    return phi / (1.0 - phi) - (m + 1) * phi ** (m + 1) / (1.0 - phi ** (m + 1))


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
    ) -> None:
        s0 = np.asarray(sigma0, dtype=np.int64)
        n = len(s0)
        if n < 2 or not np.array_equal(np.sort(s0), np.arange(n)):
            raise ValueError("sigma0 must be a permutation of 0,...,n-1 with n >= 2.")
        th = np.ones(n - 1) if theta is None else np.asarray(theta, dtype=float)
        if th.shape != (n - 1,) or np.any(th < 0.0) or not np.all(np.isfinite(th)):
            raise ValueError("theta must be a length-(n-1) vector of non-negative dispersions.")
        self.sigma0 = s0
        self.theta = th
        self.dim = n
        self.log_z = float(sum(_log_psi(float(th[i - 1]), i) for i in range(1, n)))
        self.name = name
        self.keys = keys

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
        return float(self.seq_log_density(np.asarray(x, dtype=np.int64)[None, :])[0])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings."""
        j = seq_rim_code(x, self.sigma0)  # (N, n-1)
        return -(j @ self.theta) - self.log_z

    def sampler(self, seed: int | None = None) -> GeneralizedMallowsModelSampler:
        """Return an exact repeated-insertion sampler for this model."""
        return GeneralizedMallowsModelSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GeneralizedMallowsModelEstimator:
        """Return a stage-wise Mallows estimator with this item count."""
        return GeneralizedMallowsModelEstimator(dim=self.dim, name=self.name, keys=self.keys)

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

    def sample(self, size: int | None = None) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` iid orderings."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class GeneralizedMallowsModelAccumulator(SequenceEncodableStatisticAccumulator):
    """Precede matrix (Copeland consensus) + count + a bounded reservoir for the per-stage means."""

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = dim
        self.precede = np.zeros((dim, dim))
        self.count = 0.0
        self.reservoir = reservoir
        self._res_x: list[np.ndarray] = []
        self._res_w: list[float] = []
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update consensus and reservoir statistics from one weighted ordering."""
        self.seq_update(np.asarray([x], dtype=np.int64), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize consensus and reservoir statistics from one ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update consensus and reservoir statistics from encoded orderings."""
        n = self.dim
        r_idx, rp_idx = np.triu_indices(n, 1)
        for row, w in zip(x, weights):
            np.add.at(self.precede, (row[r_idx], row[rp_idx]), w)
            if len(self._res_x) < self.reservoir:
                self._res_x.append(np.asarray(row, dtype=np.int64))
                self._res_w.append(float(w))
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> GeneralizedMallowsModelAccumulator:
        """Merge observation weight, precedence counts, and reservoir samples."""
        count, precede, res_x, res_w = suff_stat
        self.count += count
        self.precede += precede
        for row, w in zip(res_x, res_w):
            if len(self._res_x) < self.reservoir:
                self._res_x.append(np.asarray(row, dtype=np.int64))
                self._res_w.append(float(w))
        return self

    def value(self):
        """Return count, precedence matrix, and bounded reservoir contents."""
        return self.count, self.precede, [np.asarray(r) for r in self._res_x], list(self._res_w)

    def from_value(self, x) -> GeneralizedMallowsModelAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.precede, res_x, res_w = x
        self.dim = self.precede.shape[0]
        self._res_x = [np.asarray(r, dtype=np.int64) for r in res_x]
        self._res_w = [float(w) for w in res_w]
        return self

    def acc_to_encoder(self) -> GeneralizedMallowsModelDataEncoder:
        """Return the encoder compatible with these sufficient statistics."""
        return GeneralizedMallowsModelDataEncoder(dim=self.dim)


class GeneralizedMallowsModelAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for stage-wise Generalized Mallows statistics."""

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = dim
        self.reservoir = reservoir
        self.keys = keys

    def make(self) -> GeneralizedMallowsModelAccumulator:
        """Create an empty stage-wise Generalized Mallows accumulator."""
        return GeneralizedMallowsModelAccumulator(dim=self.dim, reservoir=self.reservoir, keys=self.keys)


class GeneralizedMallowsModelEstimator(ParameterEstimator):
    """Copeland consensus for ``sigma0`` and a per-stage moment match for each ``theta_i``."""

    def __init__(self, dim: int, reservoir: int = 10000, name: str | None = None, keys: str | None = None) -> None:
        if dim is None or dim < 2:
            raise ValueError("GeneralizedMallowsModelEstimator requires dim >= 2.")
        self.dim = int(dim)
        self.reservoir = reservoir
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedMallowsModelAccumulatorFactory:
        """Return a factory for stage-wise Mallows sufficient-statistic accumulators."""
        return GeneralizedMallowsModelAccumulatorFactory(dim=self.dim, reservoir=self.reservoir, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> GeneralizedMallowsModelDistribution:
        """Estimate central ordering and per-stage dispersions from accumulated rankings."""
        count, precede, res_x, res_w = suff_stat
        n = self.dim
        if count <= 0.0:
            return GeneralizedMallowsModelDistribution(np.arange(n), name=self.name, keys=self.keys)
        scores = precede.sum(axis=1) - precede.sum(axis=0)  # Copeland
        sigma0 = np.argsort(-scores, kind="stable").astype(np.int64)
        j = seq_rim_code(np.asarray(res_x, dtype=np.int64), sigma0)  # (R, n-1)
        w = np.asarray(res_w, dtype=float)
        mean_j = (j * w[:, None]).sum(axis=0) / w.sum()
        theta = np.array([_solve_stage_theta(float(mean_j[i - 1]), i) for i in range(1, n)])
        return GeneralizedMallowsModelDistribution(sigma0, theta, name=self.name, keys=self.keys)


class GeneralizedMallowsModelDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "GeneralizedMallowsModelDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedMallowsModelDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        rv = np.asarray([list(row) for row in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("GeneralizedMallowsModelDistribution requires a non-empty sequence of orderings.")
        expected = np.arange(rv.shape[1])
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("orderings must be permutations of 0,...,n-1.")
        return rv


__all__ = [
    "GeneralizedMallowsModelDistribution",
    "GeneralizedMallowsModelSampler",
    "GeneralizedMallowsModelAccumulator",
    "GeneralizedMallowsModelAccumulatorFactory",
    "GeneralizedMallowsModelEstimator",
    "GeneralizedMallowsModelDataEncoder",
]
