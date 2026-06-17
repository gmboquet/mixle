"""Defines the log-pseudo-determinant, polgamma, trigamma, and digamma inverse functions."""

import math
from collections.abc import Iterable

import numpy as np

# Several scipy.special names are re-exported as part of this module's public surface
# (other modules do `from pysp.utils.special import beta, betaln, gammaln, ...`).
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

from pysp.arithmetic import *

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


def polygamma_loc(n, y, out=None):
    if out is not None:
        fac2 = zeta(n + 1, y, out=out)
        fac2 *= (-1.0) ** (n + 1) * gamma(n + 1.0)
    else:
        fac2 = (-1.0) ** (n + 1) * gamma(n + 1.0) * zeta(n + 1, y)

    return fac2


def trigamma(y: np.ndarray | int | float | Iterable | list[float], out: np.ndarray | None = None) -> np.ndarray | float:
    """Trigamma function.

    Args:
        y (Array-like): An array-like or float/int.
        out (np.ndarray); Store output in this variable.

    Returns:
        Numpy array of trigamma function evaluated at y.

    """
    return zeta(2, y, out=out)


def digammainv(y: np.ndarray | float, out: np.ndarray | None = None) -> np.ndarray | float:
    """Inverse digamma function evaluated on y.

    Args:
        y (Union[np.ndarray, float]): Numpy array of values to be evaluated or single value.
        out (Optional[np.ndarray]): Deprecated. Kept for consistency with other files.

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

            # if np.any(t2 == 0) or np.any(np.isnan(t2)) or np.any(np.isinf(t2)):
            #    print('bad')
            # if np.any(np.isnan(t1)) or np.any(np.isinf(t1)):
            #    print('bad')

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
