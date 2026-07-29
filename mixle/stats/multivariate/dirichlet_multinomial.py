"""Dirichlet-multinomial (Polya) distribution -- an overdispersed multinomial.

The multivariate analogue of the beta-binomial: a multinomial whose category probabilities are
Dirichlet(alpha) distributed and integrated out. For a count vector ``x`` over ``K`` categories summing
to ``n``,

    P(x; alpha) = n!/prod_k x_k! * B(alpha + x) / B(alpha),    B(a) = prod_k Gamma(a_k) / Gamma(sum a),

which adds overdispersion (and category correlation) over a plain multinomial. The number of trials
``n`` is a fixed, known parameter; ``alpha`` is fit by Minka's maximum-likelihood fixed point, run from
a cumulative-count sufficient statistic so it converges inside a single ``estimate`` call.
"""

import heapq
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_DEFAULT_MAX_RECURRENCE_CELLS = 1_000_000
_DEFAULT_MAX_FRONTIER_ENTRIES = 100_000
_DEFAULT_MIN_ALPHA = 1.0e-8


class DirichletMultinomialResourceError(RuntimeError):
    """Raised before a configured enumeration or fitting resource budget is exceeded."""


@dataclass(frozen=True)
class DirichletMultinomialFitReceipt:
    """Machine-readable evidence about a concentration fixed-point fit."""

    algorithm: str
    converged: bool
    identifiable: bool
    iterations: int
    objective_history: tuple[float, ...]
    max_delta: float
    regularized_categories: tuple[int, ...]
    concentration_floor: float


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive integer" % label)
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError("%s must be a positive integer" % label) from exc
    if result <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return result


def _trial_count(value: Any, *, label: str = "n (number of trials)") -> int:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a non-negative integer" % label)
    try:
        result = index(value)
    except TypeError:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("%s must be a non-negative integer" % label) from exc
        if not np.isfinite(numeric) or numeric < 0.0 or numeric != math.floor(numeric):
            raise ValueError("%s must be a non-negative integer" % label)
        return int(numeric)
    if result < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return result


def _finite_scalar(
    value: Any,
    *,
    label: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % label)
    if positive and result <= 0.0:
        raise ValueError("%s must be positive" % label)
    if nonnegative and result < 0.0:
        raise ValueError("%s must be non-negative" % label)
    return result


def _resource_limit(value: Any, *, label: str) -> int:
    return _positive_integer(value, label=label)


def _require_recurrence_budget(dim: int, n: int, limit: int, *, operation: str, multiplier: int = 1) -> None:
    cells = multiplier * dim * max(n, 1)
    if cells > limit:
        raise DirichletMultinomialResourceError(
            "%s requires %d recurrence cells, exceeding max_recurrence_cells=%d" % (operation, cells, limit)
        )


def _count_event(value: Any, dim: int, n: int, *, label: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a numeric count vector" % label) from exc
    if raw.shape != (dim,):
        raise ValueError("%s must have exact shape (%d,)" % (label, dim))
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0) or np.any(np.floor(raw) != raw) or raw.sum() != n:
        raise ValueError("%s must contain non-negative integer counts summing to %d" % (label, n))
    return raw.astype(np.int64, copy=True)


def _count_batch(value: Any, dim: int, n: int, *, label: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a numeric count matrix" % label) from exc
    if raw.ndim != 2 or raw.shape[1] != dim:
        raise ValueError("%s must have exact shape (N, %d)" % (label, dim))
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0) or np.any(np.floor(raw) != raw) or np.any(raw.sum(axis=1) != n):
        raise ValueError("%s must contain non-negative integer count rows summing to %d" % (label, n))
    return raw.astype(np.int64, copy=True)


def _weights(value: Any, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Dirichlet-multinomial weights must be numeric") from exc
    if result.shape != (rows,):
        raise ValueError("Dirichlet-multinomial weights must have exact shape (%d,)" % rows)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("Dirichlet-multinomial weights must be finite and non-negative")
    return result


def _recurrence_statistic(
    value: Any,
    *,
    dim: int,
    n: int,
) -> tuple[np.ndarray, float, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Dirichlet-multinomial sufficient statistic must be a (recurrence, weight, n) tuple")
    logical_n = _trial_count(value[2], label="serialized Dirichlet-multinomial trial count")
    if logical_n != n:
        raise ValueError("serialized Dirichlet-multinomial trial count does not match the configured model")
    try:
        recurrence = np.asarray(value[0], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Dirichlet-multinomial recurrence statistic must be numeric") from exc
    expected_shape = (dim, max(n, 1))
    if recurrence.shape != expected_shape:
        raise ValueError("Dirichlet-multinomial recurrence statistic must have exact shape %r" % (expected_shape,))
    count = _finite_scalar(
        value[1],
        label="Dirichlet-multinomial total weight",
        nonnegative=True,
    )
    if np.any(~np.isfinite(recurrence)) or np.any(recurrence < 0.0):
        raise ValueError("Dirichlet-multinomial recurrence statistic must be finite and non-negative")
    tolerance = 1.0e-12 * max(count, 1.0)
    if n == 0:
        if np.any(recurrence != 0.0):
            raise ValueError("zero-trial Dirichlet-multinomial recurrence storage must be zero")
    else:
        if np.any(recurrence > count + tolerance):
            raise ValueError("Dirichlet-multinomial recurrence counts cannot exceed total weight")
        if np.any(np.diff(recurrence, axis=1) > tolerance):
            raise ValueError("Dirichlet-multinomial recurrence rows must be non-increasing")
    return recurrence.copy(), count, logical_n


class DirichletMultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Dirichlet-multinomial over ``K``-category count vectors summing to ``n`` (concentration ``alpha``)."""

    def __init__(
        self,
        alpha: np.ndarray,
        n: int,
        name: str | None = None,
        keys: str | None = None,
        *,
        max_recurrence_cells: int = _DEFAULT_MAX_RECURRENCE_CELLS,
        max_frontier_entries: int = _DEFAULT_MAX_FRONTIER_ENTRIES,
    ) -> None:
        try:
            a = np.asarray(alpha, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alpha must be a nonempty vector of positive finite concentrations") from exc
        if a.ndim != 1 or a.size == 0 or np.any(a <= 0.0) or not np.all(np.isfinite(a)):
            raise ValueError("alpha must be a nonempty vector of positive finite concentrations")
        self.alpha = a.copy()
        self.dim = a.shape[0]
        self.n = _trial_count(n)
        self.name = name
        self.keys = keys
        self.max_recurrence_cells = _resource_limit(
            max_recurrence_cells,
            label="max_recurrence_cells",
        )
        self.max_frontier_entries = _resource_limit(
            max_frontier_entries,
            label="max_frontier_entries",
        )
        self.alpha.setflags(write=False)
        self._sum_alpha = float(self.alpha.sum())
        self._gammaln_alpha = gammaln(self.alpha)
        self._log_const = gammaln(self.n + 1) + gammaln(self._sum_alpha) - gammaln(self.n + self._sum_alpha)

    def __str__(self) -> str:
        return (
            "DirichletMultinomialDistribution(%s, %s, name=%s, keys=%s, "
            "max_recurrence_cells=%s, max_frontier_entries=%s)"
        ) % (
            repr(self.alpha.tolist()),
            repr(self.n),
            repr(self.name),
            repr(self.keys),
            repr(self.max_recurrence_cells),
            repr(self.max_frontier_entries),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the probability mass at a single count vector ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log-mass at ``x`` (``-inf`` if any count is negative, non-integer, or the total
        is not ``n`` -- a fractional category count is not a valid Dirichlet-multinomial outcome even
        when the (real-valued) total happens to sum to ``n``)."""
        xx = np.asarray(x, dtype=np.float64)
        if (
            xx.shape != (self.dim,)
            or not np.all(np.isfinite(xx))
            or np.any(xx < 0)
            or np.any(np.floor(xx) != xx)
            or xx.sum() != self.n
        ):
            return -np.inf
        term = gammaln(xx + self.alpha) - self._gammaln_alpha - gammaln(xx + 1.0)
        return float(self._log_const + term.sum())

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-mass for a stack of count vectors, shape ``(N, K)`` (``-inf`` rows where any
        count is negative, non-integer, or the row total is not ``n``)."""
        try:
            xx = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Dirichlet-multinomial observations must be a numeric count matrix") from exc
        if xx.ndim != 2 or xx.shape[1] != self.dim:
            raise ValueError("Dirichlet-multinomial observations must have exact shape (N, %d)" % self.dim)
        bad = (
            ~np.all(np.isfinite(xx), axis=1)
            | (xx < 0).any(axis=1)
            | (np.floor(xx) != xx).any(axis=1)
            | (xx.sum(axis=1) != self.n)
        )
        rv = np.full(len(xx), -np.inf, dtype=np.float64)
        good = ~bad
        if np.any(good):
            valid = xx[good]
            term = gammaln(valid + self.alpha) - self._gammaln_alpha - gammaln(valid + 1.0)
            rv[good] = self._log_const + term.sum(axis=1)
        return rv

    def sampler(self, seed: int | None = None) -> "DirichletMultinomialSampler":
        """Return a sampler for drawing count vectors from this distribution."""
        return DirichletMultinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DirichletMultinomialEstimator":
        """Return a Minka fixed-point MLE estimator for ``alpha`` at the fixed number of trials ``n``."""
        return DirichletMultinomialEstimator(
            self.dim,
            self.n,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
            max_recurrence_cells=self.max_recurrence_cells,
            max_frontier_entries=self.max_frontier_entries,
        )

    def dist_to_encoder(self) -> "DirichletMultinomialDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return DirichletMultinomialDataEncoder(self.dim, self.n)

    def enumerator(
        self,
        *,
        max_recurrence_cells: int | None = None,
        max_frontier_entries: int | None = None,
    ) -> "DirichletMultinomialEnumerator":
        """Return an enumerator over count vectors summing to ``n`` in descending probability order."""
        return DirichletMultinomialEnumerator(
            self,
            max_recurrence_cells=self.max_recurrence_cells if max_recurrence_cells is None else max_recurrence_cells,
            max_frontier_entries=self.max_frontier_entries if max_frontier_entries is None else max_frontier_entries,
        )


class DirichletMultinomialEnumerator(DistributionEnumerator):
    """Enumerate Dirichlet-multinomial count vectors in descending probability order, lazily (A* best-first).

    The log-mass separates per category: ``log P(x) = log_const + sum_k f_k(x_k)`` with
    ``f_k(c) = gammaln(c + alpha_k) - gammaln(alpha_k) - gammaln(c + 1)``, and each ``f_k`` is monotone
    in ``c`` (increasing for ``alpha_k > 1``, decreasing for ``alpha_k < 1``), so a prefix assignment
    ``(x_1..x_j)`` with ``m`` trials left admits the admissible completion bound
    ``sum_{k>j} max(f_k(0), f_k(m))``. A* on the exact prefix score plus that bound streams count
    vectors in exact descending order without materializing the ``C(n+K-1, K-1)`` support.
    """

    def __init__(
        self,
        dist: DirichletMultinomialDistribution,
        *,
        max_recurrence_cells: int = _DEFAULT_MAX_RECURRENCE_CELLS,
        max_frontier_entries: int = _DEFAULT_MAX_FRONTIER_ENTRIES,
    ) -> None:
        super().__init__(dist)
        recurrence_limit = _resource_limit(
            max_recurrence_cells,
            label="max_recurrence_cells",
        )
        self._max_frontier_entries = _resource_limit(
            max_frontier_entries,
            label="max_frontier_entries",
        )
        _require_recurrence_budget(
            2 * dist.dim + 1,
            dist.n + 1,
            recurrence_limit,
            operation="Dirichlet-multinomial enumeration",
        )
        self._counter = itertools.count()
        counts = np.arange(dist.n + 1, dtype=np.float64)
        # f[k, c] = the per-category log-mass term of count c in category k (f[k, 0] == 0).
        self._f = (
            gammaln(counts[None, :] + dist.alpha[:, None])
            - gammaln(dist.alpha)[:, None]
            - gammaln(counts + 1.0)[None, :]
        )
        # suffix_best[j, m] = sum_{k >= j} max over c in [0, m] of f_k(c) = sum_{k >= j} max(0, f_k(m)),
        # by the per-category monotonicity -- the admissible bound on any completion of m trials.
        best = np.maximum(self._f, 0.0)
        self._suffix_best = np.zeros((dist.dim + 1, dist.n + 1))
        for j in range(dist.dim - 1, -1, -1):
            self._suffix_best[j] = self._suffix_best[j + 1] + best[j]
        # heap entries: (-(g + h), tiebreak, prefix_tuple, g); a full-length prefix carries its exact score.
        root = -(float(dist._log_const) + float(self._suffix_best[0][dist.n]))
        self._heap: list[tuple[float, int, tuple[int, ...], float]] = []
        self._push((root, next(self._counter), (), float(dist._log_const)))

    def _push(self, item: tuple[float, int, tuple[int, ...], float]) -> None:
        if len(self._heap) >= self._max_frontier_entries:
            raise DirichletMultinomialResourceError(
                "Dirichlet-multinomial enumeration exceeded max_frontier_entries=%d" % self._max_frontier_entries
            )
        heapq.heappush(self._heap, item)

    def __next__(self) -> tuple[np.ndarray, float]:
        d = self.dist
        while self._heap:
            _, _, prefix, g = heapq.heappop(self._heap)
            if len(prefix) == d.dim:
                return (np.asarray(prefix, dtype=np.int64), g)
            m = d.n - sum(prefix)
            if len(prefix) == d.dim - 1:
                # the last category is forced to the remaining total: push the completed vector exactly
                g2 = g + float(self._f[d.dim - 1, m])
                self._push((-g2, next(self._counter), prefix + (m,), g2))
                continue
            j = len(prefix)
            for c in range(m + 1):
                g2 = g + float(self._f[j, c])
                bound = g2 + float(self._suffix_best[j + 1][m - c])
                self._push((-bound, next(self._counter), prefix + (c,), g2))
        raise StopIteration


class DirichletMultinomialSampler(DistributionSampler):
    """Draw counts as ``p ~ Dirichlet(alpha)`` then ``x ~ Multinomial(n, p)``."""

    def __init__(self, dist: DirichletMultinomialDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one count vector or a stack of iid count vectors."""
        d = self.dist
        n_draws = 1 if size is None else int(size)
        p = self.rng.dirichlet(d.alpha, size=n_draws)
        out = np.array([self.rng.multinomial(d.n, pi) for pi in p])
        return out[0] if size is None else out


class DirichletMultinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate cumulative counts ``c[k, j] = sum_i w_i 1{x_ik > j}`` (the Minka digamma-recurrence stat)."""

    def __init__(
        self,
        dim: int,
        n: int,
        name: str | None = None,
        keys: str | None = None,
        *,
        max_recurrence_cells: int = _DEFAULT_MAX_RECURRENCE_CELLS,
    ) -> None:
        self.dim = _positive_integer(dim, label="Dirichlet-multinomial dimension")
        self.n = _trial_count(n)
        self.max_recurrence_cells = _resource_limit(
            max_recurrence_cells,
            label="max_recurrence_cells",
        )
        _require_recurrence_budget(
            self.dim,
            self.n,
            self.max_recurrence_cells,
            operation="Dirichlet-multinomial accumulation",
        )
        self.c = np.zeros((self.dim, max(self.n, 1)), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: DirichletMultinomialDistribution | None) -> None:
        """Accumulate Minka recurrence statistics for one count vector."""
        xx = _count_event(
            x,
            self.dim,
            self.n,
            label="Dirichlet-multinomial observation",
        )
        checked_weight = _finite_scalar(
            weight,
            label="Dirichlet-multinomial observation weight",
            nonnegative=True,
        )
        if estimate is not None and (
            not isinstance(estimate, DirichletMultinomialDistribution)
            or estimate.dim != self.dim
            or estimate.n != self.n
        ):
            raise ValueError("Dirichlet-multinomial accumulator estimate must match dimension and n")
        for k in range(self.dim):
            self.c[k, : xx[k]] += checked_weight  # j = 0 .. x_k-1
        self.count += checked_weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one count vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate Minka recurrence statistics from encoded count vectors."""
        xx = _count_batch(
            x,
            self.dim,
            self.n,
            label="Dirichlet-multinomial observations",
        )
        w = _weights(weights, len(xx))
        if estimate is not None and (
            not isinstance(estimate, DirichletMultinomialDistribution)
            or estimate.dim != self.dim
            or estimate.n != self.n
        ):
            raise ValueError("Dirichlet-multinomial accumulator estimate must match dimension and n")
        if self.n > 0:
            for k in range(self.dim):
                hist = np.bincount(xx[:, k], weights=w, minlength=self.n + 1)
                tail = np.cumsum(hist[::-1])[::-1]  # tail[v] = sum_{u>=v} hist[u]
                self.c[k, :] += tail[1 : self.n + 1]  # c[k,j] = sum_{v>j} hist[v]
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded count vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float, int]) -> "DirichletMultinomialAccumulator":
        """Merge another Dirichlet-multinomial sufficient-statistic tuple."""
        recurrence, count, _ = _recurrence_statistic(
            suff_stat,
            dim=self.dim,
            n=self.n,
        )
        self.c += recurrence
        self.count += count
        return self

    def value(self) -> tuple[np.ndarray, float, int]:
        """Return cumulative recurrence counts, total weight, and logical trial count."""
        return self.c.copy(), self.count, self.n

    def from_value(self, x: tuple[np.ndarray, float, int]) -> "DirichletMultinomialAccumulator":
        """Replace accumulator contents from recurrence statistics."""
        self.c, self.count, _ = _recurrence_statistic(
            x,
            dim=self.dim,
            n=self.n,
        )
        return self

    def acc_to_encoder(self) -> "DirichletMultinomialDataEncoder":
        """Return the encoder used by this accumulator."""
        return DirichletMultinomialDataEncoder(self.dim, self.n)


class DirichletMultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for DirichletMultinomialAccumulator."""

    def __init__(
        self,
        dim: int,
        n: int,
        name: str | None = None,
        keys: str | None = None,
        *,
        max_recurrence_cells: int = _DEFAULT_MAX_RECURRENCE_CELLS,
    ) -> None:
        self.dim = _positive_integer(dim, label="Dirichlet-multinomial dimension")
        self.n = _trial_count(n)
        self.name = name
        self.keys = keys
        self.max_recurrence_cells = _resource_limit(
            max_recurrence_cells,
            label="max_recurrence_cells",
        )
        _require_recurrence_budget(
            self.dim,
            self.n,
            self.max_recurrence_cells,
            operation="Dirichlet-multinomial accumulator factory",
        )

    def make(self) -> DirichletMultinomialAccumulator:
        """Create a fresh Dirichlet-multinomial accumulator."""
        return DirichletMultinomialAccumulator(
            self.dim,
            self.n,
            name=self.name,
            keys=self.keys,
            max_recurrence_cells=self.max_recurrence_cells,
        )


class DirichletMultinomialEstimator(ParameterEstimator):
    """Minka fixed-point maximum-likelihood estimator for the Dirichlet-multinomial concentration."""

    def __init__(
        self,
        dim: int,
        n: int,
        max_iter: int = 500,
        tol: float = 1.0e-9,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        min_alpha: float = _DEFAULT_MIN_ALPHA,
        max_recurrence_cells: int = _DEFAULT_MAX_RECURRENCE_CELLS,
        max_frontier_entries: int = _DEFAULT_MAX_FRONTIER_ENTRIES,
    ) -> None:
        if pseudo_count is not None:
            # Unlike the raw-moment method-of-moments estimators (Gumbel, Weibull, ...), this is an
            # iterative Minka fixed-point MLE over a cumulative-count recurrence statistic
            # c[k, j] = sum_i w_i 1{x_ik > j}, not a simple additive raw moment. Blending a prior
            # pseudo-sample into it correctly would require its expectation under the current
            # model -- E[c[k,j]] = pseudo_count * P(X_k > j), the tail of the coordinate-k marginal,
            # itself a Beta-Binomial(n, alpha_k, sum(alpha)-alpha_k) CDF evaluated at every
            # j = 0..n-1 for every one of the dim coordinates -- not a small, safe change to bolt
            # onto this fixed point. Refuse explicitly rather than silently ignoring pseudo_count.
            raise ValueError(
                "DirichletMultinomialEstimator does not support pseudo_count smoothing: its Minka "
                "fixed-point MLE operates on a cumulative-count recurrence statistic, not a simple "
                "additive moment, so there is no small, safe way to blend a prior pseudo-sample "
                "into it. Pass pseudo_count=None (the default)."
            )
        self.dim = _positive_integer(dim, label="Dirichlet-multinomial estimator dimension")
        self.n = _trial_count(n)
        self.max_iter = _positive_integer(max_iter, label="Dirichlet-multinomial max_iter")
        self.tol = _finite_scalar(
            tol,
            label="Dirichlet-multinomial tolerance",
            positive=True,
        )
        self.min_alpha = _finite_scalar(
            min_alpha,
            label="Dirichlet-multinomial concentration floor",
            positive=True,
        )
        self.max_recurrence_cells = _resource_limit(
            max_recurrence_cells,
            label="max_recurrence_cells",
        )
        self.max_frontier_entries = _resource_limit(
            max_frontier_entries,
            label="max_frontier_entries",
        )
        _require_recurrence_budget(
            self.dim,
            self.n,
            self.max_recurrence_cells,
            operation="Dirichlet-multinomial estimation",
        )
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> DirichletMultinomialAccumulatorFactory:
        """Return an accumulator factory for Dirichlet-multinomial statistics."""
        return DirichletMultinomialAccumulatorFactory(
            self.dim,
            self.n,
            name=self.name,
            keys=self.keys,
            max_recurrence_cells=self.max_recurrence_cells,
        )

    def _distribution(
        self,
        alpha: np.ndarray,
        receipt: DirichletMultinomialFitReceipt,
    ) -> DirichletMultinomialDistribution:
        result = DirichletMultinomialDistribution(
            alpha,
            self.n,
            name=self.name,
            keys=self.keys,
            max_recurrence_cells=self.max_recurrence_cells,
            max_frontier_entries=self.max_frontier_entries,
        )
        result.fit_receipt = receipt
        return result

    def _objective(self, alpha: np.ndarray, recurrence: np.ndarray, count: float) -> float:
        j = np.arange(self.n, dtype=np.float64)
        return float(np.sum(recurrence * np.log(alpha[:, None] + j[None, :])) - count * np.sum(np.log(alpha.sum() + j)))

    def estimate(
        self,
        nobs: float | None,
        suff_stat: tuple[np.ndarray, float, int],
    ) -> DirichletMultinomialDistribution:
        """Estimate concentration parameters by Minka's fixed-point update."""
        c, count, _ = _recurrence_statistic(
            suff_stat,
            dim=self.dim,
            n=self.n,
        )
        if count <= 0.0 or self.n == 0:
            receipt = DirichletMultinomialFitReceipt(
                algorithm="minka-fixed-point",
                converged=True,
                identifiable=False,
                iterations=0,
                objective_history=(),
                max_delta=0.0,
                regularized_categories=(),
                concentration_floor=self.min_alpha,
            )
            return self._distribution(np.ones(self.dim), receipt)
        j = np.arange(self.n, dtype=np.float64)
        alpha = np.full(self.dim, 1.0, dtype=np.float64)
        objective_history = [self._objective(alpha, c, count)]
        regularized_categories: set[int] = set()
        converged = False
        max_delta = math.inf
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            s = alpha.sum()
            # Minka: alpha_k <- alpha_k * [sum_j c[k,j]/(alpha_k+j)] / [N * sum_j 1/(s+j)]
            numer = (c / (alpha[:, None] + j[None, :])).sum(axis=1)
            denom = count * float((1.0 / (s + j)).sum())
            if not np.isfinite(denom) or denom <= 0.0 or np.any(~np.isfinite(numer)):
                raise ValueError("Dirichlet-multinomial fixed-point update produced invalid finite terms")
            raw_alpha = alpha * numer / denom
            if np.any(~np.isfinite(raw_alpha)):
                raise ValueError("Dirichlet-multinomial fixed-point concentration overflowed")
            regularized = np.flatnonzero(raw_alpha < self.min_alpha)
            regularized_categories.update(int(category) for category in regularized)
            alpha_new = np.maximum(raw_alpha, self.min_alpha)
            max_delta = float(np.max(np.abs(alpha_new - alpha)))
            objective_history.append(self._objective(alpha_new, c, count))
            alpha = alpha_new
            if max_delta < self.tol:
                converged = True
                break
        receipt = DirichletMultinomialFitReceipt(
            algorithm="minka-fixed-point",
            converged=converged,
            identifiable=True,
            iterations=iterations,
            objective_history=tuple(objective_history),
            max_delta=max_delta,
            regularized_categories=tuple(sorted(regularized_categories)),
            concentration_floor=self.min_alpha,
        )
        return self._distribution(alpha, receipt)


class DirichletMultinomialDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``K``-category count vectors as an ``(N, K)`` array."""

    def __init__(self, dim: int, n: int) -> None:
        self.dim = _positive_integer(dim, label="Dirichlet-multinomial encoder dimension")
        self.n = _trial_count(n)

    def __str__(self) -> str:
        return "DirichletMultinomialDataEncoder(dim=%d, n=%d)" % (self.dim, self.n)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirichletMultinomialDataEncoder) and other.dim == self.dim and other.n == self.n

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Validate and encode count vectors as an integral matrix."""
        return _count_batch(
            x,
            self.dim,
            self.n,
            label="Dirichlet-multinomial observations",
        )
