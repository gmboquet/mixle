"""Plackett-Luce ranking distributions over full permutations.

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

import heapq
import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
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
        """Declare the NumPy execution path used by Plackett-Luce ranking kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

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
        """Create a Plackett-Luce distribution from item log-worths.

        Args:
            log_w (Union[Sequence[float], np.ndarray]): Length-K log-worths (real valued). The density is
                invariant to an additive constant; values are stored as given so the representation round
                trips exactly. The estimator emits a canonical form whose worths sum to one.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            log_w (np.ndarray): Log-worths of length K.
            dim (int): Number of items K.
            name (Optional[str]): Optional distribution name.
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
        """Return a constructor-style representation of the Plackett-Luce distribution."""
        return "PlackettLuceDistribution(%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.log_w]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of an ordering x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of an ordering ``x``.

        ``x`` may be a full ranking (a permutation of ``0,...,K-1``) or a **partial / top-m** ranking
        (an ordered list of ``m <= K`` distinct items, best first, leaving the other ``K-m`` items
        unranked). For a partial ranking the sequential-choice denominator at each stage still
        includes the unranked items:
        ``p(x) = prod_{s=0}^{m-1} w_{x[s]} / (sum_{t>=s} w_{x[t]} + sum_{u unranked} w_u)``,
        which reduces to the full-ranking density when ``m = K``.
        """
        idx = np.asarray(x, dtype=int)
        if idx.ndim != 1 or (
            idx.size and (np.any(idx < 0) or np.any(idx >= self.dim) or len(set(idx.tolist())) != idx.size)
        ):
            raise ValueError("PlackettLuceDistribution ordering must be distinct item indices in 0,...,K-1.")
        g = self.log_w[idx]
        rcl = _reverse_logcumsumexp(g)
        if 0 < idx.size < self.dim:
            mask = np.ones(self.dim, dtype=bool)
            mask[idx] = False
            rcl = np.logaddexp(rcl, float(np.logaddexp.reduce(self.log_w[mask])))
        return float(np.sum(g - rcl))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings.

        A dense ``(N, K)`` integer array (the full-ranking encoding) is scored by the vectorized
        path; a ragged sequence of variable-length orderings (the partial / top-m encoding) is scored
        per-ranking via :meth:`log_density`, which includes the unranked-item denominator term.
        """
        arr = x if isinstance(x, np.ndarray) else np.asarray(x, dtype=object)
        if arr.dtype == object or arr.ndim != 2:
            return np.array([self.log_density(row) for row in x], dtype=float)
        g = self.log_w[arr]
        rcl = _reverse_logcumsumexp(g)
        return np.sum(g - rcl, axis=1)

    def sampler(self, seed: int | None = None) -> "PlackettLuceSampler":
        """Return a sampler for drawing orderings from this distribution."""
        return PlackettLuceSampler(self, seed)

    def enumerator(self) -> "PlackettLuceEnumerator":
        """Return an exact finite enumerator over all orderings in decreasing probability order."""
        return PlackettLuceEnumerator(self)

    def estimator(self, pseudo_count: float | None = None) -> "PlackettLuceEstimator":
        """Return an MM estimator that keeps the item count fixed at this distribution's K."""
        return PlackettLuceEstimator(dim=self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "PlackettLuceDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return PlackettLuceDataEncoder(dim=self.dim)


class PlackettLuceEnumerator(DistributionEnumerator):
    """Enumerate Plackett--Luce orderings in descending probability order, lazily (A* best-first).

    A ranking's log-density is ``sum_k [theta_{x_k} - logsumexp(theta over the items not yet ranked at step k)]``.
    The state is a chosen prefix with exact partial score g; the best possible completion ranks the remaining
    items by descending worth (which minimizes the remaining logsumexp denominators), giving an exact admissible
    bound h. A* on g + h streams the rankings in exact descending order without touching the n! support.
    """

    def __init__(self, dist: PlackettLuceDistribution) -> None:
        super().__init__(dist)
        self._lw = np.asarray(dist.log_w, dtype=float)
        self._n = dist.dim
        self._counter = itertools.count()
        # heap entries: (-(g + h), tiebreak, prefix_tuple, remaining_frozenset, g)
        root_remaining = frozenset(range(self._n))
        self._heap: list = [(-self._bound(root_remaining), next(self._counter), (), root_remaining, 0.0)]

    def _bound(self, remaining: frozenset[int]) -> float:
        """Exact best-completion log-probability over ``remaining`` (rank by descending worth)."""
        if not remaining:
            return 0.0
        lw = np.sort(self._lw[list(remaining)])[::-1]  # descending worth
        # term at step k = lw[k] - logsumexp(lw[k:]); sum_k lw[k] - sum_k logsumexp(suffix_k)
        suffix = np.logaddexp.accumulate(lw[::-1])[::-1]  # logsumexp of each descending suffix
        return float(lw.sum() - suffix.sum())

    def __next__(self) -> tuple[list[int], float]:
        while self._heap:
            _, _, prefix, remaining, g = heapq.heappop(self._heap)
            if not remaining:
                return list(prefix), g  # complete ranking; g is the exact log-density
            lse_rem = float(np.logaddexp.reduce(self._lw[list(remaining)]))
            for j in remaining:
                new_g = g + (self._lw[j] - lse_rem)
                new_remaining = remaining - {j}
                f = new_g + self._bound(new_remaining)
                heapq.heappush(self._heap, (-f, next(self._counter), (*prefix, j), new_remaining, new_g))
        raise StopIteration


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
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: PlackettLuceDistribution | None) -> None:
        """Update sufficient statistics from one full ranking and its weight."""
        self.seq_update(np.asarray([x], dtype=int), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one full ranking using the uniform-worth seed."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: PlackettLuceDistribution | None) -> None:
        """Update MM sufficient statistics from encoded full rankings."""
        k = self.dim
        # Numerator: every item ranked above last position is a winner at its (non-final) stage.
        nonlast = x[:, : k - 1].reshape(-1)
        np.add.at(self.num, nonlast, np.repeat(weights, k - 1))

        worths = np.ones(k) if estimate is None else np.exp(estimate.log_w)
        go = worths[x]  # (N, K) worths in ranked order
        suffix = _reverse_cumsum(go)  # suffix[n, s] = sum_{t>=s} w_{x[n,t]}
        inv_suffix = 1.0 / np.maximum(suffix, np.finfo(np.float64).tiny)  # guard underflowed worth sums
        prefix = np.cumsum(inv_suffix, axis=1)  # prefix[n, m] = sum_{s<=m} 1/suffix[n, s]
        # Item ranked at position t is in contention at stages 0..min(t, K-2).
        m_cols = np.minimum(np.arange(k), k - 2)
        contrib = prefix[:, m_cols] * weights[:, None]
        np.add.at(self.den, x.reshape(-1), contrib.reshape(-1))
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize a batch of rankings with uniform-worth denominator statistics."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> "PlackettLuceAccumulator":
        """Merge count, numerator, and denominator arrays from another accumulator."""
        count, num, den = suff_stat
        self.count += count
        self.num += num
        self.den += den
        return self

    def value(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return the accumulated count, numerator counts, and denominator totals."""
        return self.count, self.num, self.den

    def from_value(self, x: tuple[float, np.ndarray, np.ndarray]) -> "PlackettLuceAccumulator":
        """Restore accumulator state from ``value`` output."""
        self.count, self.num, self.den = x[0], np.asarray(x[1]), np.asarray(x[2])
        self.dim = len(self.num)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from ``stats_dict`` when its key is present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "PlackettLuceDataEncoder":
        """Return the full-ranking encoder compatible with these sufficient statistics."""
        return PlackettLuceDataEncoder(dim=self.dim)


class PlackettLuceAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PlackettLuceAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> PlackettLuceAccumulator:
        """Create an empty full-ranking Plackett-Luce accumulator."""
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
        """Return a factory for MM sufficient statistics at this estimator's dimension."""
        return PlackettLuceAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> PlackettLuceDistribution:
        """Return one MM estimate from accumulated full-ranking sufficient statistics."""
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
        """Validate and encode full rankings as a dense integer matrix."""
        rv = np.asarray([list(row) for row in x], dtype=int)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("PlackettLuceDistribution requires a non-empty sequence of orderings.")
        k = rv.shape[1]
        expected = np.arange(k)
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("PlackettLuceDistribution orderings must be permutations of 0,...,K-1.")
        return rv


class PlackettLucePartialDataEncoder(DataSequenceEncoder):
    """Encode partial / top-m orderings (variable length) over a fixed item count ``dim``.

    Each datum is an ordered list of ``m <= dim`` distinct item indices (best first); the remaining
    items are left unranked. Encoded as a ragged list of 1-D int arrays.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def __str__(self) -> str:
        return "PlackettLucePartialDataEncoder(dim=%r)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlackettLucePartialDataEncoder) and other.dim == self.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Validate and encode variable-length top-m rankings as ragged arrays."""
        rows = []
        for row in x:
            r = np.asarray(list(row), dtype=int)
            if (
                r.ndim != 1
                or r.size > self.dim
                or (r.size and (r.min() < 0 or r.max() >= self.dim or len(set(r.tolist())) != r.size))
            ):
                raise ValueError("partial ordering must be distinct item indices in 0,...,dim-1.")
            rows.append(r)
        return rows


class PlackettLucePartialAccumulator(SequenceEncodableStatisticAccumulator):
    """Generalized Hunter (2004) MM statistics for partial / top-m Plackett-Luce rankings.

    For each ranking, at every non-forced stage ``s`` (at least two items still available) the chosen
    item accrues a numerator count and **all currently-available items** (the unranked tail included)
    accrue ``weight / sum_{available} w`` to the denominator, evaluated at the previous estimate's
    worths (uniform when none). On full rankings this reduces exactly to the vectorized full-ranking
    accumulator.
    """

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = int(dim)
        self.num = np.zeros(self.dim)
        self.den = np.zeros(self.dim)
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: PlackettLuceDistribution | None) -> None:
        """Update partial-ranking sufficient statistics from one observation."""
        self.seq_update([np.asarray(list(x), dtype=int)], np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one partial ranking using uniform worths."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[np.ndarray], weights: np.ndarray, estimate: PlackettLuceDistribution | None
    ) -> None:
        """Update generalized MM statistics from encoded top-m rankings."""
        k = self.dim
        worths = np.ones(k) if estimate is None else np.exp(estimate.log_w)
        for r, wgt in zip(x, np.asarray(weights, dtype=float)):
            r = np.asarray(r, dtype=int)
            avail = np.ones(k, dtype=bool)
            sum_avail = float(worths.sum())
            for s in range(r.size):
                if (k - s) >= 2:  # a genuine choice (the final forced pick of a full ranking is skipped)
                    self.den[avail] += wgt / sum_avail
                    self.num[r[s]] += wgt
                sum_avail -= worths[r[s]]
                avail[r[s]] = False
            self.count += float(wgt)

    def seq_initialize(self, x: Sequence[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize a batch of partial rankings with uniform-worth statistics."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> "PlackettLucePartialAccumulator":
        """Merge partial-ranking sufficient statistics from another accumulator."""
        count, num, den = suff_stat
        self.count += count
        self.num += num
        self.den += den
        return self

    def value(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return accumulated partial-ranking count, numerator, and denominator statistics."""
        return self.count, self.num, self.den

    def from_value(self, x: tuple[float, np.ndarray, np.ndarray]) -> "PlackettLucePartialAccumulator":
        """Restore partial-ranking accumulator state from ``value`` output."""
        self.count, self.num, self.den = x[0], np.asarray(x[1]), np.asarray(x[2])
        self.dim = len(self.num)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this partial accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from ``stats_dict`` when its key is present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "PlackettLucePartialDataEncoder":
        """Return the partial-ranking encoder compatible with this accumulator."""
        return PlackettLucePartialDataEncoder(dim=self.dim)


class PlackettLucePartialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PlackettLucePartialAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = int(dim)
        self.keys = keys

    def make(self) -> PlackettLucePartialAccumulator:
        """Create an empty partial-ranking Plackett-Luce accumulator."""
        return PlackettLucePartialAccumulator(dim=self.dim, keys=self.keys)


class PlackettLucePartialEstimator(ParameterEstimator):
    """MM estimator of Plackett-Luce log-worths from partial / top-m rankings (item count K fixed)."""

    def __init__(
        self,
        dim: int,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("PlackettLucePartialEstimator requires the number of items dim >= 2.")
        self.dim = int(dim)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> PlackettLucePartialAccumulatorFactory:
        """Return a factory for partial-ranking MM sufficient statistics."""
        return PlackettLucePartialAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> PlackettLuceDistribution:
        """Return one MM estimate from accumulated partial-ranking sufficient statistics."""
        count, num, den = suff_stat
        if count <= 0.0:
            return PlackettLuceDistribution(np.zeros(self.dim), name=self.name, keys=self.keys)
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        if self.pseudo_count is not None:
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
