"""A dependency-free UMAP core (numpy/scipy only) -- the fallback engine behind ``humap``.

``humap`` prefers umap-learn when it is installed; this module exists so a missing optional
dependency degrades to a slower-but-correct layout instead of an ImportError, and so embedding
goals (:mod:`mixle.utils.hvis.goals`) have an optimizer they can actually steer -- umap-learn's
numba SGD loop cannot take per-iteration external gradients.

It is the standard UMAP construction (McInnes, Healy & Melville 2018), deliberately minimal:

1. :func:`fuzzy_simplicial_set` -- per-row smoothed-kNN calibration (``rho`` = nearest distance,
   ``sigma`` binary-searched so the row's total membership is ``log2(k)``), then probabilistic
   t-conorm symmetrization ``W + W^T - W o W^T``.
2. :func:`fit_ab` -- least-squares fit of the low-dimensional curve ``1/(1 + a d^(2b))`` to the
   ``min_dist``/``spread`` target, exactly as umap-learn does.
3. :func:`simplicial_set_layout` -- spectral initialization (normalized-Laplacian eigenvectors,
   random fallback) and the epochs-per-sample SGD with negative sampling and per-component +-4
   move clipping, vectorized per epoch over the due edges rather than numba-jitted per edge.

The knn graph comes from the caller (``humap`` hands in MODEL distances), so nothing here embeds
raw vectors -- it lays out whatever graph the model-based affinity machinery produced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

__all__ = ["fit_ab", "fuzzy_simplicial_set", "internal_umap", "simplicial_set_layout"]

_SMOOTH_K_TOLERANCE = 1.0e-5
_MOVE_CLIP = 4.0


def fit_ab(min_dist: float, spread: float = 1.0) -> tuple[float, float]:
    """Fit ``(a, b)`` of the low-dimensional membership curve ``1/(1 + a d^(2b))`` to the target
    ``exp(-(d - min_dist)/spread)`` (1 inside ``min_dist``) -- umap-learn's own calibration."""
    from scipy.optimize import curve_fit

    grid = np.linspace(0.0, 3.0 * spread, 300)
    target = np.where(grid < min_dist, 1.0, np.exp(-(grid - min_dist) / spread))

    def curve(d, a, b):
        return 1.0 / (1.0 + a * d ** (2.0 * b))

    (a, b), _ = curve_fit(curve, grid, target, p0=(1.0, 1.0), maxfev=10000)
    return float(a), float(b)


def _smooth_knn_row(dists: np.ndarray, target: float, n_iter: int = 64) -> tuple[float, float]:
    """UMAP's per-row calibration: rho = nearest positive distance; binary-search sigma so
    ``sum_j exp(-max(0, d_j - rho)/sigma) = target``."""
    positive = dists[dists > 0.0]
    rho = float(positive.min()) if len(positive) else 0.0
    lo, hi, sigma = 0.0, np.inf, 1.0
    for _ in range(n_iter):
        val = float(np.sum(np.exp(-np.maximum(dists - rho, 0.0) / sigma)))
        if abs(val - target) < _SMOOTH_K_TOLERANCE:
            break
        if val > target:
            hi = sigma
            sigma = (lo + hi) / 2.0
        else:
            lo = sigma
            sigma = sigma * 2.0 if np.isinf(hi) else (lo + hi) / 2.0
    return rho, max(sigma, 1.0e-12)


def fuzzy_simplicial_set(knn_idx: np.ndarray, knn_dist: np.ndarray) -> scipy.sparse.coo_matrix:
    """Symmetric fuzzy graph from a kNN graph of (model) distances.

    Per-row memberships ``exp(-(d - rho)/sigma)`` with the smoothed-kNN calibration, then the
    probabilistic t-conorm ``W + W^T - W o W^T`` -- an undirected membership strength that is high
    when EITHER direction considers the edge close.
    """
    knn_idx = np.asarray(knn_idx, dtype=np.int64)
    knn_dist = np.asarray(knn_dist, dtype=np.float64)
    n, k = knn_idx.shape
    target = np.log2(max(k, 2))

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = knn_idx.ravel()
    vals = np.empty(n * k, dtype=np.float64)
    for i in range(n):
        rho, sigma = _smooth_knn_row(knn_dist[i], target)
        vals[i * k : (i + 1) * k] = np.exp(-np.maximum(knn_dist[i] - rho, 0.0) / sigma)

    w = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    w_t = w.T.tocsr()
    prod = w.multiply(w_t)
    return (w + w_t - prod).tocoo()


def _spectral_init(graph: scipy.sparse.coo_matrix, emb_dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Normalized-Laplacian eigenvector initialization, with a random fallback when the eigensolver
    does not converge (small/disconnected graphs) -- a fallback engine must never hard-fail on init."""
    n = graph.shape[0]
    try:
        from scipy.sparse.linalg import eigsh

        w = graph.tocsr()
        deg = np.asarray(w.sum(axis=1)).ravel()
        d_inv_sqrt = scipy.sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1.0e-12)))
        lap = scipy.sparse.identity(n) - d_inv_sqrt @ w @ d_inv_sqrt
        k = min(emb_dim + 1, n - 1)
        v0 = rng.uniform(-1.0, 1.0, size=n)  # ARPACK's default start vector uses the GLOBAL rng
        _, vecs = eigsh(lap, k=k, sigma=0.0, which="LM", maxiter=n * 50, v0=v0)
        y = vecs[:, 1 : emb_dim + 1]
        if y.shape[1] < emb_dim:
            raise ValueError("not enough eigenvectors")
        y = y / max(float(np.abs(y).max()), 1.0e-12) * 10.0
        return y + rng.normal(0.0, 1.0e-4, size=(n, emb_dim))
    except Exception:
        return rng.uniform(-10.0, 10.0, size=(n, emb_dim))


def simplicial_set_layout(
    graph: scipy.sparse.coo_matrix,
    emb_dim: int = 2,
    n_epochs: int | None = None,
    a: float = 1.577,
    b: float = 0.8951,
    *,
    seed: int | None = None,
    Y: np.ndarray | None = None,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    goals=None,
) -> np.ndarray:
    """UMAP's epochs-per-sample SGD on a fuzzy graph, vectorized per epoch over the due edges.

    Attractive moves apply to both endpoints, repulsive (negative-sampled) moves to the head only,
    every per-component move clipped to +-4 and the learning rate annealed linearly to zero -- the
    reference algorithm's behavior, minus numba. ``goals`` gradients are applied once per epoch at
    the same annealed rate, followed by hard-constraint projection.
    """
    from mixle.utils.hvis.goals import apply_projections, total_goal_gradient

    graph = graph.tocoo()
    n = graph.shape[0]
    keep = graph.data > graph.data.max() / max(float(n_epochs or 500), 1.0)
    heads, tails, weights = graph.row[keep], graph.col[keep], graph.data[keep]
    if n_epochs is None:
        n_epochs = 500 if n < 10000 else 200
    rng = np.random.RandomState(seed)

    if Y is None:
        y = _spectral_init(graph, emb_dim, rng)
    else:
        y = np.array(Y, dtype=np.float64)
        if y.shape != (n, emb_dim):
            raise ValueError(f"initial Y must have shape ({n}, {emb_dim}).")

    epochs_per_sample = weights.max() / weights
    next_due = epochs_per_sample.copy()

    for epoch in range(1, int(n_epochs) + 1):
        lr = learning_rate * (1.0 - epoch / float(n_epochs))
        due = next_due <= epoch
        if np.any(due):
            h, t = heads[due], tails[due]
            diff = y[h] - y[t]
            d2 = np.sum(diff * diff, axis=1)
            pos = d2 > 0.0
            coeff = np.zeros_like(d2)
            coeff[pos] = (-2.0 * a * b * d2[pos] ** (b - 1.0)) / (1.0 + a * d2[pos] ** b)
            move = np.clip(coeff[:, None] * diff, -_MOVE_CLIP, _MOVE_CLIP) * lr
            np.add.at(y, h, move)
            np.add.at(y, t, -move)

            neg = rng.randint(0, n, size=(len(h), int(negative_sample_rate)))
            diff_n = y[h][:, None, :] - y[neg]
            d2_n = np.sum(diff_n * diff_n, axis=2)
            coeff_n = (2.0 * b) / ((0.001 + d2_n) * (1.0 + a * d2_n**b))
            coeff_n[neg == h[:, None]] = 0.0  # a point is not its own negative
            move_n = np.clip(coeff_n[:, :, None] * diff_n, -_MOVE_CLIP, _MOVE_CLIP) * lr
            np.add.at(y, h, move_n.sum(axis=1))

            next_due[due] += epochs_per_sample[due]

        extra = total_goal_gradient(goals, y)
        if extra is not None:  # bounded rate semantics: applied as-is, not annealed away with the SGD rate
            y -= extra
        y = apply_projections(goals, y)

    return y


def internal_umap(
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    emb_dim: int = 2,
    *,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: int | None = None,
    seed: int | None = None,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    goals: Any = None,
) -> np.ndarray:
    """kNN graph in, embedding out: the dependency-free ``humap`` engine, end to end."""
    a, b = fit_ab(min_dist, spread)
    graph = fuzzy_simplicial_set(knn_idx, knn_dist)
    return simplicial_set_layout(
        graph,
        emb_dim=emb_dim,
        n_epochs=n_epochs,
        a=a,
        b=b,
        seed=seed,
        negative_sample_rate=negative_sample_rate,
        learning_rate=learning_rate,
        goals=goals,
    )
