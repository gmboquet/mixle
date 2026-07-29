"""Shared plumbing for the non-elliptical copula cores (Clayton, Frank, Student-t).

Unlike the Gaussian copula -- whose inversion estimator has an additive sufficient statistic (the moments of
the normal scores) -- these cores fit their parameter(s) by Kendall's-tau matching or 1-D MLE, neither of
which is a running additive statistic. So their accumulator simply BUFFERS the (weighted) uniform scores and
the estimator fits from the whole buffer, the same buffer-the-rows pattern the neural leaves and
:class:`~mixle.stats.combinator.copula.CopulaDistribution` use. A copula core's ``seq_encode`` returns the raw
``u`` rows (its ``seq_log_density`` recomputes whatever transform it needs, since the parameters are not known
at encode time), so the buffered statistic is exactly the ``(u, weight)`` rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import index
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def validated_dimension(value: Any, *, minimum: int = 2, label: str = "copula dimension") -> int:
    """Return an exact integer dimension without truncating a model specification."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be an integer" % label)
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError("%s must be an integer" % label) from exc
    if result < minimum:
        raise ValueError("%s must be at least %d" % (label, minimum))
    return result


def validated_sample_size(value: Any) -> int:
    """Return an exact non-negative integer sample count."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("sample size must be a non-negative integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError("sample size must be a non-negative integer") from exc
    if result < 0:
        raise ValueError("sample size must be non-negative")
    return result


def validated_finite_scalar(value: Any, *, label: str) -> float:
    """Return one finite real scalar parameter."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % label)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % label)
    return result


def u_score_event(value: Any, dim: int, *, label: str = "copula observation") -> np.ndarray:
    """Validate one event in the exact open unit-cube support ``(0, 1)^d``."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if result.shape != (dim,):
        raise ValueError("%s must have exact shape (%d,)" % (label, dim))
    reject_out_of_unit_cube(result, label=label)
    return result


def u_score_batch(
    value: Any,
    dim: int,
    *,
    label: str = "copula observations",
    allow_empty: bool = True,
) -> np.ndarray:
    """Validate a batch in the exact open unit-cube support ``(0, 1)^d``."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if allow_empty and result.shape == (0,):
        result = np.empty((0, dim), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dim:
        raise ValueError("%s must have exact shape (N, %d)" % (label, dim))
    if not allow_empty and result.shape[0] == 0:
        raise ValueError("%s must contain at least one row" % label)
    reject_out_of_unit_cube(result, label=label)
    return result


def validated_weight(value: Any) -> float:
    """Validate one finite, non-negative observation weight."""
    result = validated_finite_scalar(value, label="copula observation weight")
    if result < 0.0:
        raise ValueError("copula observation weight must be non-negative")
    return result


def validated_weights(value: Any, rows: int) -> np.ndarray:
    """Validate weights aligned one-for-one with a copula observation batch."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("copula observation weights must be numeric") from exc
    if result.shape != (rows,):
        raise ValueError("copula observation weights must have exact shape (%d,)" % rows)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("copula observation weights must be finite and non-negative")
    return result


def validated_buffered_statistic(
    value: Any,
    dim: int,
    *,
    minimum_rows: int = 0,
    require_positive_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the serialized sufficient statistic used by buffered copula fits."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("copula sufficient statistic must be an (observations, weights) pair")
    u = u_score_batch(value[0], dim)
    w = validated_weights(value[1], len(u))
    if len(u) < minimum_rows:
        raise ValueError("copula fit requires at least %d observation rows" % minimum_rows)
    if require_positive_weight and not float(w.sum()) > 0.0:
        raise ValueError("copula fit requires positive total observation weight")
    return u, w


def reject_unsupported_pseudo_count(pseudo_count: Any, *, family: str) -> None:
    """Reject regularization that a copula estimator cannot define honestly."""
    if pseudo_count is not None:
        raise ValueError("%s pseudo-count regularization is not implemented" % family)


class UScoreEncoder(DataSequenceEncoder):
    """Encode a batch of uniform-score rows as a plain ``(n, d)`` float array (identity transform)."""

    def __init__(self, dim: int) -> None:
        self.dim = validated_dimension(dim)

    def __str__(self) -> str:
        return "UScoreEncoder(dim=%d)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UScoreEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return u_score_batch(x, self.dim).copy()


class BufferedUScoreAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer the (weighted) uniform-score rows; the copula core's estimator fits from the whole buffer."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = validated_dimension(dim)
        self.keys = keys
        self._u: list[np.ndarray] = []
        self._w: list[np.ndarray] = []

    def update(self, x: np.ndarray, weight: float, estimate: Any) -> None:
        self._u.append(u_score_event(x, self.dim).reshape(1, self.dim).copy())
        self._w.append(np.asarray([validated_weight(weight)], dtype=np.float64))

    def initialize(self, x: np.ndarray, weight: float, rng: Any) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        xb = u_score_batch(x, self.dim)
        checked_weights = validated_weights(weights, len(xb))
        if len(xb):
            self._u.append(xb.copy())
            self._w.append(checked_weights.copy())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray]) -> BufferedUScoreAccumulator:
        u, w = validated_buffered_statistic(suff_stat, self.dim)
        if len(u):
            self._u.append(u.copy())
            self._w.append(w.copy())
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray]:
        u = np.concatenate(self._u, axis=0) if self._u else np.zeros((0, self.dim))
        w = np.concatenate(self._w) if self._w else np.zeros((0,))
        return u.copy(), w.copy()

    def from_value(self, x: tuple[np.ndarray, np.ndarray]) -> BufferedUScoreAccumulator:
        u, w = validated_buffered_statistic(x, self.dim)
        self._u = [u.copy()] if len(u) else []
        self._w = [w.copy()] if len(u) else []
        return self

    def acc_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder(self.dim)


class BufferedUScoreAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = validated_dimension(dim)
        self.keys = keys

    def make(self) -> BufferedUScoreAccumulator:
        return BufferedUScoreAccumulator(self.dim, keys=self.keys)


def out_of_unit_cube(u: np.ndarray) -> np.ndarray:
    """Return a per-row mask for values outside the exact open support ``(0, 1)^d``."""
    u = np.asarray(u, dtype=np.float64)
    bad = (u <= 0.0) | (u >= 1.0) | ~np.isfinite(u)
    return np.any(bad, axis=-1)


def reject_out_of_unit_cube(u: np.ndarray, *, label: str = "copula observations") -> None:
    """Raise when an observation is not finite and strictly inside ``(0, 1)^d``."""
    u = np.asarray(u, dtype=np.float64)
    if u.size and np.any(out_of_unit_cube(u)):
        finite = u[np.isfinite(u)]
        lo = float(finite.min()) if finite.size else float("nan")
        hi = float(finite.max()) if finite.size else float("nan")
        raise ValueError("%s must be finite and lie strictly inside (0, 1); observed min=%r, max=%r" % (label, lo, hi))


def weighted_kendall_tau(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Kendall's tau between two score vectors: (concordant - discordant) / total, pair weight w_i w_j.

    The implementation is ``O(n log n)`` in time and ``O(n)`` in memory. Ties
    contribute zero to the numerator and remain in the denominator, matching
    the historical weighted tau-a definition used by these estimators.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    ww = validated_weights(w, len(aa))
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("Kendall score vectors must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)):
        raise ValueError("Kendall score vectors must be finite")
    total = 0.5 * (float(ww.sum()) ** 2 - float(np.dot(ww, ww)))
    if total <= 0.0:
        return 0.0

    order = np.lexsort((bb, aa))
    aa, bb, ww = aa[order], bb[order], ww[order]
    ranks = np.searchsorted(np.unique(bb), bb)
    tree = np.zeros(int(ranks.max(initial=-1)) + 2, dtype=np.float64)

    def prefix_sum(position: int) -> float:
        answer = 0.0
        cursor = position
        while cursor > 0:
            answer += tree[cursor]
            cursor -= cursor & -cursor
        return answer

    def add(position: int, value: float) -> None:
        cursor = position + 1
        while cursor < len(tree):
            tree[cursor] += value
            cursor += cursor & -cursor

    concordance = 0.0
    previous_weight = 0.0
    start = 0
    while start < len(aa):
        stop = start + 1
        while stop < len(aa) and aa[stop] == aa[start]:
            stop += 1
        for rank, weight in zip(ranks[start:stop], ww[start:stop]):
            below = prefix_sum(int(rank))
            through = prefix_sum(int(rank) + 1)
            above = previous_weight - through
            concordance += float(weight) * (below - above)
        for rank, weight in zip(ranks[start:stop], ww[start:stop]):
            add(int(rank), float(weight))
            previous_weight += float(weight)
        start = stop
    return concordance / total


def maximize_1d(loglik: Any, lo: float, hi: float, *, iters: int = 60) -> float:
    """Golden-section search for the argmax of a unimodal 1-D ``loglik`` on ``[lo, hi]``."""
    lo = validated_finite_scalar(lo, label="optimization lower bound")
    hi = validated_finite_scalar(hi, label="optimization upper bound")
    iters = validated_dimension(iters, minimum=1, label="optimization iteration count")
    if not lo < hi:
        raise ValueError("optimization lower bound must be less than upper bound")
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = loglik(c), loglik(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = loglik(d)
        else:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = loglik(c)
    return 0.5 * (a + b)
