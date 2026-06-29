"""Shared NumPy stationary-kernel primitives (RBF / Matern-3/2 / Matern-5/2).

The RBF and Matern covariance shapes were re-implemented independently in several places
(:mod:`mixle.models.sparse_gaussian_process`, :mod:`mixle.ppl.field`, :mod:`mixle.doe.calibrate`),
each with its own ``sqrt(3)``/``sqrt(5)`` constants. This module is the single source of those shapes
for the NumPy back-end. (The Torch GP in :mod:`mixle.models.gaussian_process` keeps its own autograd
kernel -- a real back-end difference, not duplication that can be shared here.)

Two layers are provided:

* shape functions that take an already-lengthscale-scaled distance and return ``amp**2 * shape`` --
  these let callers keep whatever distance/scaling arithmetic they already use (e.g. great-circle
  chord distance, or scaling the squared distance before vs after the sqrt), so existing numerical
  results are preserved bit-for-bit;
* :func:`stationary_kernel`, a self-contained ``(x1, x2, lengthscale, amplitude, name)`` helper that
  computes Euclidean pairwise distances and applies the chosen shape (the sparse-GP convention:
  scale the squared distance by ``lengthscale**2`` *before* the sqrt).

None of these add a jitter / nugget term -- that is a caller concern (the existing call sites use
different nugget values: 1e-8 on Kuu, 1e-6 on a field covariance, none on a cross-covariance), so it
stays at the call site.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SQRT3",
    "SQRT5",
    "rbf_from_scaled_sqdist",
    "matern32_from_scaled_dist",
    "matern52_from_scaled_dist",
    "exponential_from_scaled_dist",
    "stationary_kernel",
]

SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)


def rbf_from_scaled_sqdist(d2_scaled: np.ndarray, amplitude: float) -> np.ndarray:
    """RBF (squared-exponential) covariance ``amp**2 * exp(-0.5 * d2_scaled)``.

    ``d2_scaled`` is the squared distance already divided by ``lengthscale**2``.
    """
    return amplitude**2 * np.exp(-0.5 * d2_scaled)


def matern32_from_scaled_dist(r: np.ndarray, amplitude: float) -> np.ndarray:
    """Matern-3/2 covariance ``amp**2 * (1 + s) * exp(-s)`` with ``s = sqrt(3) * r``.

    ``r`` is the Euclidean distance already divided by the lengthscale.
    """
    s = SQRT3 * r
    return amplitude**2 * (1.0 + s) * np.exp(-s)


def matern52_from_scaled_dist(r: np.ndarray, amplitude: float) -> np.ndarray:
    """Matern-5/2 covariance ``amp**2 * (1 + s + s**2/3) * exp(-s)`` with ``s = sqrt(5) * r``.

    ``r`` is the Euclidean distance already divided by the lengthscale.
    """
    s = SQRT5 * r
    return amplitude**2 * (1.0 + s + s * s / 3.0) * np.exp(-s)


def exponential_from_scaled_dist(r: np.ndarray, amplitude: float) -> np.ndarray:
    """Matern-1/2 / exponential covariance ``amp**2 * exp(-r)`` (``r`` already lengthscale-scaled)."""
    return amplitude**2 * np.exp(-r)


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def stationary_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    lengthscale: float,
    amplitude: float,
    name: str,
) -> np.ndarray:
    """Pairwise stationary covariance between point sets ``x1`` and ``x2`` (Euclidean inputs).

    ``name`` is ``'rbf'``, ``'matern32'`` or ``'matern52'``. The squared distance is divided by
    ``lengthscale**2`` before the (Matern) sqrt -- the sparse-GP convention -- with the Matern sqrt
    floored at ``1e-12`` so it stays defined at zero separation. No jitter is added.
    """
    x1, x2 = _as2d(x1), _as2d(x2)
    d2 = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1) / lengthscale**2
    if name == "rbf":
        return rbf_from_scaled_sqdist(d2, amplitude)
    r = np.sqrt(np.maximum(d2, 1e-12))
    if name == "matern32":
        return matern32_from_scaled_dist(r, amplitude)
    if name == "matern52":
        return matern52_from_scaled_dist(r, amplitude)
    raise ValueError(f"unknown kernel {name!r}")
