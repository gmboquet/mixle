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
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class UScoreEncoder(DataSequenceEncoder):
    """Encode a batch of uniform-score rows as a plain ``(n, d)`` float array (identity transform)."""

    def __str__(self) -> str:
        return "UScoreEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UScoreEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


class BufferedUScoreAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer the (weighted) uniform-score rows; the copula core's estimator fits from the whole buffer."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys
        self._u: list[np.ndarray] = []
        self._w: list[np.ndarray] = []

    def update(self, x: np.ndarray, weight: float, estimate: Any) -> None:
        self._u.append(np.asarray(x, dtype=np.float64).reshape(1, -1))
        self._w.append(np.asarray([float(weight)], dtype=np.float64))

    def initialize(self, x: np.ndarray, weight: float, rng: Any) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        xb = np.asarray(x, dtype=np.float64)
        self._u.append(xb.reshape(len(xb), -1))
        self._w.append(np.asarray(weights, dtype=np.float64).ravel())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray]) -> BufferedUScoreAccumulator:
        u, w = suff_stat
        if len(u):
            self._u.append(np.asarray(u, dtype=np.float64).reshape(-1, self.dim))
            self._w.append(np.asarray(w, dtype=np.float64).ravel())
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray]:
        u = np.concatenate(self._u, axis=0) if self._u else np.zeros((0, self.dim))
        w = np.concatenate(self._w) if self._w else np.zeros((0,))
        return u, w

    def from_value(self, x: tuple[np.ndarray, np.ndarray]) -> BufferedUScoreAccumulator:
        u, w = x
        u = np.asarray(u, dtype=np.float64).reshape(-1, self.dim)
        self._u = [u] if len(u) else []
        self._w = [np.asarray(w, dtype=np.float64).ravel()] if len(u) else []
        return self

    def acc_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class BufferedUScoreAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> BufferedUScoreAccumulator:
        return BufferedUScoreAccumulator(self.dim, keys=self.keys)


def out_of_unit_cube(u: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    """Per-row mask: True where any coordinate of ``u`` is not in ``[0, 1]`` (beyond a small float tolerance).

    A copula's density/log-density is defined only on the unit cube ``[0, 1]^d`` (uniform marginals);
    coordinates further outside than ``tol`` are not a valid observation for ANY copula core, no matter how
    the boundary itself is handled numerically. Reduces over the last axis, so a single ``(d,)`` row gives a
    0-d (scalar) result and a batch ``(n, d)`` gives an ``(n,)`` mask -- callers use this to force out-of-range
    rows to score ``-inf`` instead of being silently ``np.clip``-ed onto a boundary corner and scored as if
    that clipped substitute were the real (impossible) observation.
    """
    u = np.asarray(u, dtype=np.float64)
    bad = (u < -tol) | (u > 1.0 + tol) | np.isnan(u)
    return np.any(bad, axis=-1)


def reject_out_of_unit_cube(u: np.ndarray, tol: float = 1.0e-9) -> None:
    """Raise ``ValueError`` if any row of ``u`` has a coordinate outside ``[0, 1]``.

    For data about to be FIT: a copula's input must already be legitimate uniform-score/PIT data, so
    (matching how :class:`~mixle.stats.univariate.continuous.beta.BetaAccumulator` and
    :class:`~mixle.stats.univariate.continuous.beta.BetaDataEncoder` treat observations outside Beta's own
    ``(0, 1)`` support) out-of-range fitting data is a hard error rather than something to silently clip onto
    the boundary and fit as though it were legitimate.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.size and np.any(out_of_unit_cube(u, tol)):
        finite = u[np.isfinite(u)]
        lo = float(finite.min()) if finite.size else float("nan")
        hi = float(finite.max()) if finite.size else float("nan")
        raise ValueError(
            "copula observations must lie in [0, 1] (uniform-score/PIT domain); got value(s) outside that "
            "range (observed min=%r, max=%r)" % (lo, hi)
        )


def weighted_kendall_tau(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Kendall's tau between two score vectors: (concordant - discordant) / total, pair weight w_i w_j.

    O(n^2) over the buffered rows -- copula cores are fit on the whole buffer, and n is a batch, not a stream.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    pair_w = w[:, None] * w[None, :]
    concordant = float(np.sum(pair_w * da * db))  # sign(da*db)=+1 concordant, -1 discordant, 0 tie
    total = float(np.sum(pair_w)) - float(np.sum(w * w))  # exclude the i==j diagonal
    return concordant / total if total > 0 else 0.0


def maximize_1d(loglik: Any, lo: float, hi: float, *, iters: int = 60) -> float:
    """Golden-section search for the argmax of a unimodal 1-D ``loglik`` on ``[lo, hi]``."""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = loglik(c), loglik(d)
    for _ in range(int(iters)):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = loglik(d)
        else:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = loglik(c)
    return 0.5 * (a + b)
