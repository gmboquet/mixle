"""Rough paths -- the truncated path signature (iterated-integral) transform.

The *signature* of a path ``X: [0, T] -> R^d`` is the sequence of iterated integrals

    S(X) = ( 1, {int dX^i}, {int int dX^i dX^j}, ... ),

a graded tensor (level ``k`` lives in ``(R^d)^{otimes k}``) that characterizes the path up to
reparameterization and is the central object of rough-path theory and of signature features in machine
learning. For a piecewise-linear path it is computed *exactly*: each straight segment with increment
``delta`` has signature ``delta^{otimes k} / k!`` (the tensor exponential), and segments combine by Chen's
identity -- the signature of a concatenation is the truncated tensor-algebra product of the segment
signatures. The transform therefore satisfies, to machine precision, the closed form on linear paths,
Chen's multiplicativity ``S(X * Y) = S(X) (x) S(Y)``, and the factorial bound ``||S_k|| <= L^k / k!`` in
the path length ``L``.

Reference: Chen, "Integration of paths" (1958); Lyons, "Differential equations driven by rough signals",
*Rev. Mat. Iberoamericana* 14 (1998); Lyons, Caruana & Levy, *Differential Equations Driven by Rough
Paths* (2007).
"""

from math import factorial
from typing import Any

import numpy as np


def _segment_signature(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """Return the exact signature of a straight segment with increment ``delta`` (level ``k`` = ``delta^{(x)k}/k!``)."""
    levels: list[np.ndarray] = [np.array(1.0)]
    cur = np.array(1.0)
    for k in range(1, depth + 1):
        cur = np.tensordot(cur, delta, axes=0)
        levels.append(cur / factorial(k))
    return levels


def signature_tensor_product(a: list[np.ndarray], b: list[np.ndarray], depth: int) -> list[np.ndarray]:
    """Return the truncated tensor-algebra product ``(a (x) b)_n = sum_{i+j=n} a_i (x) b_j`` (Chen's product)."""
    out: list[np.ndarray] = []
    for n in range(depth + 1):
        acc: np.ndarray | None = None
        for i in range(n + 1):
            term = np.tensordot(a[i], b[n - i], axes=0)
            acc = term if acc is None else acc + term
        out.append(acc if acc is not None else np.array(0.0))
    return out


def path_signature(path: Any, depth: int) -> list[np.ndarray]:
    """Return the truncated signature of a piecewise-linear path up to level ``depth``.

    Args:
        path: array of shape ``(n_points, d)`` -- the vertices of a piecewise-linear path in ``R^d``.
        depth: truncation level ``M``; the signature is returned as a list ``[S_0, S_1, ..., S_M]`` where
            ``S_0 = 1`` (scalar) and ``S_k`` has shape ``(d,) * k``.

    Returns:
        The list of signature tensors. Computed exactly by Chen's identity over the segments.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("path must have shape (n_points, d).")
    if int(depth) < 0:
        raise ValueError("depth must be non-negative.")
    if pts.shape[0] < 2:
        return [np.array(1.0)] + [np.zeros((pts.shape[1],) * k) for k in range(1, int(depth) + 1)]
    sig = _segment_signature(pts[1] - pts[0], int(depth))
    for i in range(1, pts.shape[0] - 1):
        sig = signature_tensor_product(sig, _segment_signature(pts[i + 1] - pts[i], int(depth)), int(depth))
    return sig


def signature_norms(signature: list[np.ndarray]) -> list[float]:
    """Return the Hilbert-Schmidt norm of each signature level (level 0 omitted)."""
    return [float(np.linalg.norm(level.ravel())) for level in signature[1:]]
