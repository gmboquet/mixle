"""Bradley-Terry model for paired comparisons.

Each observation is an ordered pair ``(winner, loser)`` drawn from ``K`` items; the model gives every
item a latent worth ``w_i = exp(log_w_i)`` and sets

    P(i beats j) = w_i / (w_i + w_j) = sigmoid(log_w_i - log_w_j).

Treating the compared pair as a uniform draw over the ``C(K, 2)`` unordered pairs makes this a proper
distribution over ordered pairs:

    p(winner, loser) = (1 / C(K, 2)) * sigmoid(log_w[winner] - log_w[loser]).

Worths are identified up to a global scale, so ``log_w`` is stored centered (mean zero). Estimation is
the Zermelo / minorization-maximization fixed point (Hunter 2004), a few low-cost numba iterations over the
``K x K`` win-count matrix -- the sufficient statistic. Unlike :class:`PlackettLuceDistribution` (full
orderings) this consumes pairwise data directly.

Data type: ``Tuple[int, int]`` -- ``(winner, loser)`` item indices in ``0..K-1`` (winner != loser).
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
from mixle.utils.optional_deps import numba


@numba.njit("float64[:](float64[:, :], int64, float64)", cache=True)
def _bt_mm(wins: np.ndarray, n_iter: int, tol: float) -> np.ndarray:
    """Zermelo / MM fixed point for the Bradley-Terry worths from a win-count matrix; returns log-worths."""
    k = wins.shape[0]
    p = np.ones(k) / k
    w_tot = np.zeros(k)
    n_pair = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w_tot[i] += wins[i, j]
            n_pair[i, j] = wins[i, j] + wins[j, i]
    for _ in range(n_iter):
        newp = np.empty(k)
        for i in range(k):
            denom = 0.0
            for j in range(k):
                if j != i and n_pair[i, j] > 0.0:
                    denom += n_pair[i, j] / (p[i] + p[j])
            newp[i] = w_tot[i] / denom if denom > 0.0 else p[i]
        s = newp.sum()
        if s <= 0.0:
            break
        diff = 0.0
        for i in range(k):
            newp[i] /= s
            diff += abs(newp[i] - p[i])
        p = newp
        if diff < tol:
            break
    out = np.empty(k)
    for i in range(k):
        out[i] = math.log(p[i]) if p[i] > 0.0 else -np.inf
    return out


class BradleyTerryDistribution(SequenceEncodableProbabilityDistribution):
    """Bradley-Terry paired-comparison model with centered log-worths ``log_w``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by Bradley-Terry kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Pairwise-comparison likelihood is numpy-native; the MM fit uses a numba kernel.",
        )

    def __init__(self, log_w: Sequence[float] | np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        lw = np.asarray(log_w, dtype=float)
        if lw.ndim != 1 or lw.size < 2 or not np.all(np.isfinite(lw)):
            raise ValueError("log_w must be a finite length-K vector with K >= 2.")
        self.log_w = lw - lw.mean()  # identified up to a global scale
        self.dim = lw.size
        self.log_pairs = math.log(self.dim * (self.dim - 1) / 2.0)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "BradleyTerryDistribution(%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.log_w]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: tuple[int, int]) -> float:
        """Return the probability of one ``(winner, loser)`` comparison."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[int, int]) -> float:
        """Return the log-probability of one ``(winner, loser)`` comparison."""
        w, ell = int(x[0]), int(x[1])
        return self.log_w[w] - np.logaddexp(self.log_w[w], self.log_w[ell]) - self.log_pairs

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded pairwise comparisons."""
        w, ell = x[:, 0], x[:, 1]
        return self.log_w[w] - np.logaddexp(self.log_w[w], self.log_w[ell]) - self.log_pairs

    def sampler(self, seed: int | None = None) -> BradleyTerrySampler:
        """Return a sampler for paired-comparison outcomes."""
        return BradleyTerrySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> BradleyTerryEstimator:
        """Return an MM estimator with this distribution's item count."""
        return BradleyTerryEstimator(dim=self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> BradleyTerryDataEncoder:
        """Return the pairwise-comparison encoder used by vectorized methods."""
        return BradleyTerryDataEncoder(dim=self.dim)


class BradleyTerrySampler(DistributionSampler):
    """Draw ``(winner, loser)`` pairs: a uniform unordered pair, then a Bradley-Terry outcome."""

    def __init__(self, dist: BradleyTerryDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self) -> tuple[int, int]:
        k = self.dist.dim
        i, j = self.rng.choice(k, size=2, replace=False)
        pi = 1.0 / (1.0 + math.exp(-(self.dist.log_w[i] - self.dist.log_w[j])))
        return (int(i), int(j)) if self.rng.rand() < pi else (int(j), int(i))

    def sample(self, size: int | None = None) -> tuple[int, int] | list[tuple[int, int]]:
        """Draw one comparison outcome or ``size`` iid comparison outcomes."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class BradleyTerryAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the ``K x K`` win-count matrix: ``wins[i, j]`` = weighted count of ``i`` beating ``j``."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.wins = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: tuple[int, int], weight: float, estimate: Any) -> None:
        """Update the win-count matrix from one weighted comparison."""
        self.wins[int(x[0]), int(x[1])] += weight
        self.count += weight

    def initialize(self, x: tuple[int, int], weight: float, rng: RandomState | None) -> None:
        """Initialize win counts from one weighted comparison."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update win counts from encoded pairwise comparisons."""
        np.add.at(self.wins, (x[:, 0], x[:, 1]), weights)
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize win counts from encoded pairwise comparisons."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> BradleyTerryAccumulator:
        """Merge observation count and win-count matrix statistics."""
        self.count += suff_stat[0]
        self.wins += suff_stat[1]
        return self

    def value(self):
        """Return accumulated observation weight and win-count matrix."""
        return self.count, self.wins

    def from_value(self, x) -> BradleyTerryAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.wins = x[0], np.asarray(x[1])
        self.dim = self.wins.shape[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> BradleyTerryDataEncoder:
        """Return the encoder compatible with Bradley-Terry win-count statistics."""
        return BradleyTerryDataEncoder(dim=self.dim)


class BradleyTerryAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for Bradley-Terry win-count statistics."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> BradleyTerryAccumulator:
        """Create an empty Bradley-Terry accumulator."""
        return BradleyTerryAccumulator(dim=self.dim, keys=self.keys)


class BradleyTerryEstimator(ParameterEstimator):
    """Maximum-likelihood log-worths via the Zermelo / MM fixed point (Hunter 2004)."""

    def __init__(
        self,
        dim: int,
        pseudo_count: float | None = None,
        max_iter: int = 500,
        tol: float = 1e-10,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("BradleyTerryEstimator requires the number of items dim >= 2.")
        self.dim = int(dim)
        self.pseudo_count = pseudo_count
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BradleyTerryAccumulatorFactory:
        """Return a factory for Bradley-Terry sufficient-statistic accumulators."""
        return BradleyTerryAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> BradleyTerryDistribution:
        """Estimate centered log-worths from accumulated win counts."""
        count, wins = suff_stat
        k = self.dim
        if count <= 0.0:
            return BradleyTerryDistribution(np.zeros(k), name=self.name, keys=self.keys)
        w = np.array(wins, dtype=float)
        if self.pseudo_count:  # symmetric smoothing: each ordered pair gets a fractional prior win
            pc = float(self.pseudo_count)
            w = w + pc * (1.0 - np.eye(k))
        log_w = _bt_mm(np.ascontiguousarray(w), self.max_iter, self.tol)
        log_w[~np.isfinite(log_w)] = log_w[np.isfinite(log_w)].min() - 10.0 if np.any(np.isfinite(log_w)) else 0.0
        return BradleyTerryDistribution(log_w, name=self.name, keys=self.keys)


class BradleyTerryDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(winner, loser)`` pairs into an ``(N, 2)`` integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "BradleyTerryDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BradleyTerryDataEncoder)

    def seq_encode(self, x: Sequence[tuple[int, int]]) -> np.ndarray:
        """Validate and encode ``(winner, loser)`` pairs as an integer matrix."""
        rv = np.asarray([list(p) for p in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[1] != 2 or rv.shape[0] == 0:
            raise ValueError("BradleyTerryDistribution requires a non-empty sequence of (winner, loser) pairs.")
        if np.any(rv[:, 0] == rv[:, 1]):
            raise ValueError("a Bradley-Terry comparison must have winner != loser.")
        return rv


__all__ = [
    "BradleyTerryDistribution",
    "BradleyTerrySampler",
    "BradleyTerryAccumulator",
    "BradleyTerryAccumulatorFactory",
    "BradleyTerryEstimator",
    "BradleyTerryDataEncoder",
]
