"""Defines the log-pseudo-determinant, polgamma, trigamma, and digamma inverse functions.

This module is the canonical home for mixle's hand-rolled stable-math helpers (``log_erfcx``,
``trigamma``, ``digammainv``, ``log1mexp``, ``logsubexp``, ``logsumexp``, ``softmax``,
``softmax_rows``, ``valid_integer``). Import from here rather than re-implementing privately so
the numerically-careful versions stay in one place.
"""

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

# Several scipy.special names are re-exported as part of this module's public surface
# (other modules do `from mixle.utils.special import beta, betaln, gammaln, ...`).
from scipy.special import (  # noqa: F401  -- re-exported
    beta,
    betaln,
    digamma,  # as digammaS0
    erfc,
    erfcx,
    gamma,
    gammaln,
    polygamma,
    psi,
    zeta,
)
from scipy.special import logsumexp as _scipy_logsumexp

from mixle.engines.arithmetic import *

D1 = digamma(1.0)


def log_erfcx(x: np.ndarray | float, out: np.ndarray | None = None) -> np.ndarray | float:
    """Stable natural log of the scaled complementary error function ``log(erfcx(x))``.

    ``erfcx(x) = exp(x**2) * erfc(x)`` is the scaled complementary error function, the workhorse
    of the exponentially-modified-Gaussian tail. A naive ``log(erfcx(x))`` blows up at both ends:
    for large positive ``x`` it underflows (``erfcx -> 0`` so ``log -> -inf``) and for large
    negative ``x`` it overflows (the ``exp(x**2)`` factor in ``erfcx`` -> ``inf`` so ``log -> inf``).
    Three branches keep it finite and accurate everywhere:

      * large positive ``x``: the asymptotic series
        ``erfcx(x) ~ 1/(x*sqrt(pi)) * (1 - 1/(2 x^2) + 3/(4 x^4) - ...)``, in log space;
      * large negative ``x``: ``log(erfcx(x)) = x**2 + log(erfc(x))`` with ``erfc(x) -> 2`` finite,
        so no overflow;
      * moderate ``x``: the direct ``log(erfcx(x))``.

    The branches match ``log(erfcx)`` to machine precision on their overlaps. This is what keeps
    the EMG tail from underflowing/overflowing.

    Args:
        x: Array-like or scalar argument.
        out: Optional output array (only used for the array path).

    Returns:
        ``log(erfcx(x))`` as a numpy array (array input) or float (scalar input).

    """
    # Above +hi, erfcx is small enough that the asymptotic series is accurate and avoids underflow.
    # Below -lo, erfcx's exp(x^2) factor overflows, so use x^2 + log(erfc(x)) instead.
    hi = 25.0
    lo = -25.0
    scalar_input = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    xx = np.asarray(x, dtype=np.float64)

    rv = np.empty(xx.shape, dtype=np.float64) if out is None else out

    big = xx > hi
    neg = xx < lo
    mid = ~(big | neg)

    if np.any(mid):
        with np.errstate(divide="ignore"):
            rv[mid] = np.log(erfcx(xx[mid]))

    if np.any(big):
        xb = xx[big]
        inv = 1.0 / xb
        inv2 = inv * inv  # 1/x^2 without overflowing x*x for astronomically large x
        # log of the asymptotic series 1/(x*sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6) + ...)
        series = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2 - 1.875 * inv2 * inv2 * inv2
        rv[big] = -np.log(xb) - 0.5 * math.log(math.pi) + np.log(series)

    if np.any(neg):
        xn = xx[neg]
        # erfcx(x) = exp(x^2) * erfc(x); for x << 0, erfc(x) -> 2 (finite), so this stays bounded.
        rv[neg] = xn * xn + np.log(erfc(xn))

    if scalar_input:
        return float(rv)
    return rv


def stirling2(n: int, k: int) -> int:
    """Stirling number of the second kind S(n, k).

    Counts the number of ways to partition n labeled objects into k non-empty
    unlabeled subsets. Computed with the standard recurrence
    S(n, k) = k*S(n-1, k) + S(n-1, k-1) using exact integer arithmetic.

    Args:
        n (int): Number of objects (n >= 0).
        k (int): Number of subsets (k >= 0).

    Returns:
        Integer value of S(n, k); 0 when k > n or when exactly one of n, k is 0.

    """
    if k > n or k < 0:
        return 0
    if n == 0 and k == 0:
        return 1
    if k == 0:
        return 0

    row = [1] + [0] * k
    for i in range(1, n + 1):
        upper = min(i, k)
        for j in range(upper, 0, -1):
            row[j] = j * row[j] + row[j - 1]
        row[0] = 0

    return row[k]


def logpdet(x_mat: np.ndarray) -> float:
    """Computes the log-pseudo-determinant for a symmetric dense matrix.

    Args:
        x_mat (np.ndarray): 2-d Numpy array representing a matrix.

    Returns:
        float, log-pseudo-determinant.

    """
    eigs = np.abs(np.linalg.eigvalsh(x_mat))
    eigs = eigs[eigs > np.max(eigs, initial=0.0) * max(x_mat.shape) * np.finfo(np.float64).eps]

    if len(eigs) > 0:
        return float(np.sum(np.log(eigs)))
    else:
        return -math.inf


def trigamma(y: np.ndarray | int | float | Iterable | list[float], out: np.ndarray | None = None) -> np.ndarray | float:
    """Trigamma function.

    Args:
        y (Array-like): An array-like or float/int.
        out (np.ndarray); Store output in this variable.

    Returns:
        Numpy array of trigamma function evaluated at y.

    """
    return zeta(2, y, out=out)


def digammainv(y: np.ndarray | float) -> np.ndarray | float:
    """Inverse digamma function evaluated on y.

    Args:
        y (Union[np.ndarray, float]): Numpy array of values to be evaluated or single value.

    Returns:
        Numpy array if y is numpy array else float.

    """
    if isinstance(y, np.ndarray):
        rv = np.zeros(y.shape, dtype=float)
        rv[np.isposinf(y)] = np.inf

        Q = np.isfinite(y)
        z = y[Q]
        M = z >= -2.22
        x = np.empty(z.shape, dtype=float)
        x[M] = exp(z[M]) + 0.5
        x[~M] = -1.0 / (z[~M] - D1)

        t1 = np.zeros(x.shape, dtype=float)
        t2 = np.zeros(x.shape, dtype=float)

        for i in range(5):
            digamma(x, out=t1)
            zeta(2, x, out=t2)

            t1 -= z
            t1 /= t2
            x -= t1

        rv[Q] = x
        x = rv

    else:
        x = (exp(y) + 0.5) if y >= -2.22 else (-1.0 / (y - D1))

        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)
        x -= (digamma(x) - y) / trigamma(x)

    return x


def log1mexp(x: float) -> float:
    """Return ``log(1 - exp(x))`` for ``x <= 0``, stable across the whole range.

    Uses the two-regime split (Mächler, "Accurately Computing log(1 - exp(-|a|))"):
    ``log(-expm1(x))`` when ``exp(x)`` is small and ``log1p(-exp(x))`` when it is close to 1,
    so ``1 - exp(x)`` is never formed by a catastrophically cancelling subtraction. Returns
    ``-inf`` for ``x >= 0`` (where ``1 - exp(x) <= 0``).

    Args:
        x: A non-positive log-probability ``log p`` with ``p in [0, 1]``.

    Returns:
        ``log(1 - exp(x))``; ``-inf`` when ``x >= 0``.

    """
    if x >= 0.0:
        return -math.inf
    if x > -math.log(2.0):
        return math.log(-math.expm1(x))
    return math.log1p(-math.exp(x))


def logsubexp(log_hi: float, log_lo: float) -> float:
    """Return ``log(exp(log_hi) - exp(log_lo))`` for ``log_hi >= log_lo``, computed stably.

    Evaluates ``log_hi + log1mexp(log_lo - log_hi)`` so a far-tail difference whose two operands
    are individually indistinguishable from 0 (or 1) in probability space still returns a finite
    large-negative log-mass instead of ``log(0) = -inf``. Returns ``-inf`` when the difference is
    non-positive (``log_hi <= log_lo``).

    Args:
        log_hi: Log of the larger operand.
        log_lo: Log of the smaller operand.

    Returns:
        ``log(exp(log_hi) - exp(log_lo))``; ``-inf`` if ``log_hi <= log_lo``.

    """
    if log_hi == -math.inf:
        return -math.inf
    if log_lo == -math.inf:
        return log_hi
    if log_hi <= log_lo:
        return -math.inf
    return log_hi + log1mexp(log_lo - log_hi)


def logsumexp(a: Any, axis: int | None = None) -> Any:
    """Stable ``log(sum(exp(a)))`` via the max-shift trick.

    Thin wrapper over :func:`scipy.special.logsumexp` providing mixle's canonical scalar/array
    fallback. The ``axis=None`` (full reduction) result is returned as a Python ``float`` to match
    the private re-implementations this replaces; a reduced array is returned otherwise. Empty input
    reduces to ``-inf`` and a non-finite running max propagates (``+inf`` stays ``+inf``).

    Args:
        a: Array-like of log-values.
        axis: Axis to reduce over, or ``None`` for a full reduction.

    Returns:
        ``float`` when ``axis is None``; a numpy array otherwise.

    """
    rv = _scipy_logsumexp(a, axis=axis)
    if axis is None:
        return float(rv)
    return rv


def softmax(log_scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax of ``log_scores`` along ``axis``, with an all-``-inf`` guard.

    Subtracts the per-slice maximum before exponentiating. A slice that is entirely ``-inf`` (no
    finite log-score) has no defined softmax and would otherwise yield ``nan``; it is filled with a
    uniform distribution ``1 / n`` over that axis instead.

    Args:
        log_scores: Array of log-scores.
        axis: Axis along which to normalize (default the last axis).

    Returns:
        An array the same shape as ``log_scores`` whose ``axis`` slices each sum to 1.

    """
    log_scores = np.asarray(log_scores, dtype=np.float64)
    mx = np.max(log_scores, axis=axis, keepdims=True)
    good = np.isfinite(mx)
    shifted = np.where(good, log_scores - np.where(good, mx, 0.0), -np.inf)
    e = np.exp(shifted)
    denom = e.sum(axis=axis, keepdims=True)
    rv = np.divide(e, denom, out=np.zeros_like(e), where=denom > 0.0)
    # Slices whose max was non-finite (all -inf) get a uniform distribution.
    n = log_scores.shape[axis]
    bad_slice = ~good
    if np.any(bad_slice):
        rv = np.where(np.broadcast_to(bad_slice, rv.shape), 1.0 / n, rv)
    return rv


def softmax_rows(log_scores: np.ndarray) -> np.ndarray:
    """Row-wise (``axis=1``) softmax of a ``(B, K)`` log-score matrix, with an all-``-inf`` guard.

    Convenience wrapper for :func:`softmax` with ``axis=1``: a row that is entirely ``-inf`` is
    replaced by the uniform distribution ``1 / K`` instead of yielding ``nan``.

    Args:
        log_scores: A ``(B, K)`` matrix of log-scores.

    Returns:
        A ``(B, K)`` matrix whose rows each sum to 1.

    """
    return softmax(np.asarray(log_scores, dtype=np.float64), axis=1)


def valid_integer(x: Any, *, nonneg: bool = False) -> bool:
    """Return whether ``x`` is a finite (optionally non-negative) integer value.

    Coerces ``x`` to ``float`` and checks it is finite and integer-valued (``floor(x) == x``). Any
    coercion failure returns ``False``. With ``nonneg=True`` the value must additionally be ``>= 0``.

    Args:
        x: Candidate value.
        nonneg: If ``True``, require ``x >= 0`` (e.g. a count); if ``False``, allow negatives.

    Returns:
        ``True`` if ``x`` is a valid (non-negative, if requested) integer; ``False`` otherwise.

    """
    try:
        xx = float(x)
    except Exception:
        return False
    if not (np.isfinite(xx) and math.floor(xx) == xx):
        return False
    return xx >= 0.0 if nonneg else True
