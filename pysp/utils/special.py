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


def logpdet(x_mat: np.ndarray) -> float:
    """Computes the log-pseudo-determinant for a symmetric dense matrix.

    Args:
        x_mat (np.ndarray): 2-d Numpy array representing a matrix.

    Returns:
        float, log-pseudo-determinant.

    """
    eigs = np.abs(np.linalg.eig(x_mat))
    eigs = eigs[eigs != 0]

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
        x = M * (exp(z) + 0.5) + (1.0 - M) * (-1.0 / (z - D1))

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
        m = (y >= -2.22)
        x = m * (exp(y) + 0.5) + (1.0 - m) * (-1.0 / (y - D1))

        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))

    return x
