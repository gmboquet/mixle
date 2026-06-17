"""Create, estimate, and sample from a Plackett-Luce ranking distribution.

Defines the PlackettLuceDistribution, PlackettLuceSampler, PlackettLuceAccumulatorFactory,
PlackettLuceAccumulator, PlackettLuceEstimator, and the PlackettLuceDataEncoder classes for use with
pysparkplug.

Data type: List[int] (a full ranking of K items, given as an ordering: ``x[0]`` is the index of the
top-ranked item, ``x[1]`` the second, ..., ``x[K-1]`` the last). Each datum is a permutation of
0,...,K-1.

The Plackett-Luce model assigns each item a positive worth ``w_i`` (stored as log-worths so the
representation is numerically stable and identified up to an additive constant; the constructor
normalizes ``sum_i w_i = 1``). The probability of an ordering ``x`` is the product of sequential
softmax choices

    p(x) = prod_{s=0}^{K-1} w_{x[s]} / sum_{t>=s} w_{x[t]},

so the most-preferred remaining item is drawn first, then the next, and so on. Equivalently, ranking
the items by ``log w_i + Gumbel(0, 1)`` noise yields a Plackett-Luce sample, which the sampler uses.

Maximum likelihood has no closed form; the estimator runs the Minorization-Maximization (MM) update of
Hunter (2004): ``w_i^{new} = num_i / den_i`` where ``num_i`` counts the rankings in which item ``i`` is
not last and ``den_i`` accumulates ``1 / sum_{t>=s} w_t`` over every stage ``s`` in which ``i`` is still
in contention, evaluated at the current worths. Each ``fit`` iteration performs one MM step, monotonically
increasing the likelihood.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_LOG_WORTH_FLOOR = -700.0


def _reverse_logcumsumexp(g: np.ndarray) -> np.ndarray:
    """Return ``rcl[..., s] = logsumexp(g[..., s:])`` along the last axis."""
    return np.logaddexp.accumulate(g[..., ::-1], axis=-1)[..., ::-1]


def _reverse_cumsum(g: np.ndarray) -> np.ndarray:
    """Return ``rc[..., s] = sum(g[..., s:])`` along the last axis."""
    return np.cumsum(g[..., ::-1], axis=-1)[..., ::-1]


class PlackettLuceDistribution(SequenceEncodableProbabilityDistribution):
    """Plackett-Luce distribution over orderings of K items with log-worths log_w.

    Data type: List[int] (an ordering: a permutation of 0,...,K-1, best-ranked item first).
    """

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="sequential reverse log-sum-exp over ranking stages is numpy-native.",
        )

    def __init__(
        self,
        log_w: Sequence[float] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """PlackettLuceDistribution object.

        Args:
            log_w (Union[Sequence[float], np.ndarray]): Length-K log-worths (real valued). The density is
                invariant to an additive constant; values are stored as given so the representation round
                trips exactly. The estimator emits a canonical form whose worths sum to one.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            log_w (np.ndarray): Log-worths of length K.
            dim (int): Number of items K.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        lw = np.asarray(log_w, dtype=float).copy()
        if lw.ndim != 1 or lw.size < 2 or not np.all(np.isfinite(lw)):
            raise ValueError("PlackettLuceDistribution requires a finite log-worth vector of length >= 2.")
        self.log_w = lw
        self.dim = len(lw)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation of PlackettLuceDistribution object."""
        return "PlackettLuceDistribution(%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.log_w]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of an ordering x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of an ordering x (a permutation of 0,...,K-1)."""
        idx = np.asarray(x, dtype=int)
        g = self.log_w[idx]
        rcl = _reverse_logcumsumexp(g)
        return float(np.sum(g - rcl))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for an (N, K) array of orderings."""
        g = self.log_w[x]
        rcl = _reverse_logcumsumexp(g)
        return np.sum(g - rcl, axis=1)

    def sampler(self, seed: int | None = None) -> "PlackettLuceSampler":
        """Return a sampler for drawing orderings from this distribution."""
        return PlackettLuceSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "PlackettLuceEstimator":
        """Return an MM estimator that keeps the item count fixed at this distribution's K."""
        return PlackettLuceEstimator(dim=self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "PlackettLuceDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return PlackettLuceDataEncoder(dim=self.dim)


class PlackettLuceSampler(DistributionSampler):
    """Draw iid Plackett-Luce orderings via the Gumbel-max construction."""

    def __init__(self, dist: PlackettLuceDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> list[int] | list[list[int]]:
        """Draw orderings (permutations of 0,...,K-1); a single list when size is None."""
        sz = 1 if size is None else size
        k = self.dist.dim
        gumbel = -np.log(-np.log(self.rng.rand(sz, k)))
        keys = self.dist.log_w[None, :] + gumbel
        # Higher perturbed log-worth is preferred first: sort descending.
        orderings = np.argsort(-keys, axis=1)
        rv = [[int(v) for v in row] for row in orderings]
        return rv[0] if size is None else rv


class PlackettLuceAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the Minorization-Maximization sufficient statistics for Plackett-Luce estimation.

    ``num`` (non-last appearance counts) is data-only; ``den`` accumulates the stage reciprocals
    ``1 / sum_{t>=s} w_t`` and is evaluated at the previous ``estimate`` (uniform worths when no estimate
    is supplied, which seeds the fit).
    """

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.num = np.zeros(dim)
        self.den = np.zeros(dim)
        self.count = 0.0
        self.key = keys

    def update(self, x: Sequence[int], weight: float, estimate: PlackettLuceDistribution | None) -> None:
        self.seq_update(np.asarray([x], dtype=int), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: PlackettLuceDistribution | None) -> None:
        k = self.dim
        # Numerator: every item ranked above last position is a winner at its (non-final) stage.
        nonlast = x[:, : k - 1].reshape(-1)
        np.add.at(self.num, nonlast, np.repeat(weights, k - 1))

        worths = np.ones(k) if estimate is None else np.exp(estimate.log_w)
        go = worths[x]  # (N, K) worths in ranked order
        suffix = _reverse_cumsum(go)  # suffix[n, s] = sum_{t>=s} w_{x[n,t]}
        inv_suffix = 1.0 / suffix
        prefix = np.cumsum(inv_suffix, axis=1)  # prefix[n, m] = sum_{s<=m} 1/suffix[n, s]
        # Item ranked at position t is in contention at stages 0..min(t, K-2).
        m_cols = np.minimum(np.arange(k), k - 2)
        contrib = prefix[:, m_cols] * weights[:, None]
        np.add.at(self.den, x.reshape(-1), contrib.reshape(-1))
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> "PlackettLuceAccumulator":
        count, num, den = suff_stat
        self.count += count
        self.num += num
        self.den += den
        return self

    def value(self) -> tuple[float, np.ndarray, np.ndarray]:
        return self.count, self.num, self.den

    def from_value(self, x: tuple[float, np.ndarray, np.ndarray]) -> "PlackettLuceAccumulator":
        self.count, self.num, self.den = x[0], np.asarray(x[1]), np.asarray(x[2])
        self.dim = len(self.num)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "PlackettLuceDataEncoder":
        return PlackettLuceDataEncoder(dim=self.dim)


class PlackettLuceAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PlackettLuceAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> PlackettLuceAccumulator:
        return PlackettLuceAccumulator(dim=self.dim, keys=self.keys)


class PlackettLuceEstimator(ParameterEstimator):
    """Minorization-Maximization estimator for the Plackett-Luce log-worths (item count K fixed)."""

    def __init__(
        self,
        dim: int,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("PlackettLuceEstimator requires the number of items dim >= 2.")
        self.dim = int(dim)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> PlackettLuceAccumulatorFactory:
        return PlackettLuceAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> PlackettLuceDistribution:
        count, num, den = suff_stat
        if count <= 0.0:
            return PlackettLuceDistribution(np.zeros(self.dim), name=self.name, keys=self.keys)

        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        if self.pseudo_count is not None:
            # Symmetric Dirichlet-style smoothing: nudge every worth toward uniform.
            num = num + self.pseudo_count
            den = den + self.pseudo_count * self.dim

        worths = np.where(den > 0.0, num / np.maximum(den, np.finfo(float).tiny), 0.0)
        total = float(np.sum(worths))
        if total <= 0.0 or not np.isfinite(total):
            return PlackettLuceDistribution(np.zeros(self.dim), name=self.name, keys=self.keys)

        with np.errstate(divide="ignore"):
            log_w = np.log(worths) - np.log(total)
        log_w = np.maximum(log_w, _LOG_WORTH_FLOOR)
        return PlackettLuceDistribution(log_w, name=self.name, keys=self.keys)


class PlackettLuceDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,K-1) into an (N, K) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "PlackettLuceDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlackettLuceDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        rv = np.asarray([list(row) for row in x], dtype=int)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("PlackettLuceDistribution requires a non-empty sequence of orderings.")
        k = rv.shape[1]
        expected = np.arange(k)
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("PlackettLuceDistribution orderings must be permutations of 0,...,K-1.")
        return rv
