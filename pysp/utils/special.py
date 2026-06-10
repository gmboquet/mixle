"""Defines the log-pseudo-determinant, polgamma, trigamma, and digamma inverse functions."""
from scipy.special import gammaln
from scipy.special import gammaln, betaln, beta
from scipy.special import digamma, psi  # as digammaS0
from scipy.special import zeta, gamma, polygamma
import numpy as np
import math
from pysp.arithmetic import *
from typing import Union, Optional, Any, List, Iterable

D1 = digamma(1.0)


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


def trigamma(y: Union[np.ndarray, int, float, Iterable, List[float]], out: Optional[np.ndarray] = None) \
        -> Union[np.ndarray, float]:
    """Trigamma function.

    Args:
        y (Array-like): An array-like or float/int.
        out (np.ndarray); Store output in this variable.

    Returns:
        Numpy array of trigamma function evaluated at y.

    """
    return zeta(2, y, out=out)

def digammainv(y: Union[np.ndarray, float], out: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
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
        M = (z >= -2.22)
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

        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))

    return x
