"""Sparse model-distance graphs, random-projection trees, and kNN.

Builds the sparse neighbor graphs (exact blockwise top-k and the approximate
random-projection-forest variant) and the umap-style kNN arrays under the
model distance d_ij = -log s_ij. The dense affinity matrix is never
materialized.
"""

import numpy as np
import scipy.sparse

from pysp.utils.htsne.affinity import (
    _affinity_factors,
    _factor_n,
    _factor_parts,
    _factor_similarity_block,
    _factor_similarity_candidates,
    _factor_weight,
    _is_fisher_factor,
    _is_local_factor,
)


def sparse_model_distances(
    posterior_mat: np.ndarray,
    ll_mat: np.ndarray | None = None,
    k: int = 90,
    block_size: int = 1024,
    affinity: str = "bhattacharyya",
    evidence_cap: float | None = None,
) -> scipy.sparse.csr_matrix:
    """Sparse n x n matrix of model distances d_ij = -log s_ij.

    Keeps the k nearest neighbors (largest affinity) per row. Built blockwise so
    the dense n x n affinity matrix is never materialized. Distances are
    non-negative. evidence_cap as in model_log_affinity.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    k = min(k, n - 1)
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    if k <= 0:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    vals = np.empty(n * k, dtype=np.float64)

    for s0 in range(0, n, block_size):
        s1 = min(s0 + block_size, n)
        row_idx = np.arange(s0, s1, dtype=np.int64)
        log_s = np.zeros((s1 - s0, n))
        with np.errstate(divide="ignore"):
            for factor in factors:
                weight = _factor_weight(factor)
                if weight == 0.0:
                    continue
                term = np.log(np.maximum(_factor_similarity_block(factor, row_idx), 1.0e-300))
                if cap is not None:
                    np.maximum(term, -cap, out=term)
                log_s += weight * term
        log_s[np.arange(s1 - s0), np.arange(s0, s1)] = -np.inf
        s_blk = log_s

        nbr = np.argpartition(-s_blk, k - 1, axis=1)[:, :k]
        log_s = s_blk

        for bi, i in enumerate(range(s0, s1)):
            c = nbr[bi]
            d = -log_s[bi, c]
            np.maximum(d, 0.0, out=d)
            cols[i * k : (i + 1) * k] = c
            vals[i * k : (i + 1) * k] = d

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


class _RPTreeNode:
    __slots__ = ("idx", "direction", "threshold", "left", "right")

    def __init__(self, idx=None, direction=None, threshold=None, left=None, right=None):
        self.idx = idx
        self.direction = direction
        self.threshold = threshold
        self.left = left
        self.right = right


def _candidate_features(factors) -> tuple[np.ndarray, np.ndarray]:
    """Feature coordinates used only for approximate neighbor proposals."""
    row_blocks = []
    col_blocks = []
    n = _factor_n(factors[0])

    for factor in factors:
        weight = _factor_weight(factor)
        if weight == 0.0:
            continue

        if _is_local_factor(factor):
            sq = np.asarray(factor["sqrt_z"], dtype=np.float64)
            x = np.asarray(factor["x"], dtype=np.float64)
            mu = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            x = np.where(np.isfinite(x), x, mu)
            sd = np.std(x, axis=0, keepdims=True)
            x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(sd, 1.0e-8)
            rg = np.hstack((sq, 0.25 * x))
            ch = rg
        elif _is_fisher_factor(factor):
            x = np.asarray(factor["x"], dtype=np.float64)
            rg = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            ch = rg
            scale = np.sqrt(weight)
            row_blocks.append(scale * rg)
            col_blocks.append(scale * ch)
            continue
        else:
            g, h, _ = _factor_parts(factor)
            rg = np.nan_to_num(np.asarray(g, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            ch = np.nan_to_num(np.asarray(h, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        rg_norm = np.linalg.norm(rg, axis=1, keepdims=True)
        ch_norm = np.linalg.norm(ch, axis=1, keepdims=True)
        rg = rg / np.maximum(rg_norm, 1.0e-300)
        ch = ch / np.maximum(ch_norm, 1.0e-300)

        scale = np.sqrt(weight)
        row_blocks.append(scale * rg)
        col_blocks.append(scale * ch)

    if not row_blocks:
        z = np.zeros((n, 1), dtype=np.float64)
        return z, z.copy()

    return np.hstack(row_blocks), np.hstack(col_blocks)


def _build_rp_tree(
    X: np.ndarray, idx: np.ndarray, leaf_size: int, rng: np.random.RandomState, max_depth: int
) -> _RPTreeNode:
    if len(idx) <= leaf_size or max_depth <= 0:
        return _RPTreeNode(idx=idx)

    pts = X[idx]
    for _ in range(12):
        direction = rng.randn(X.shape[1])
        dn = np.linalg.norm(direction)
        if dn <= 0:
            continue
        direction /= dn
        proj = np.dot(pts, direction)
        if not np.all(np.isfinite(proj)) or proj.max() - proj.min() <= 1.0e-12:
            continue
        threshold = float(np.median(proj))
        left_mask = proj <= threshold
        if np.any(left_mask) and np.any(~left_mask):
            left = _build_rp_tree(X, idx[left_mask], leaf_size, rng, max_depth - 1)
            right = _build_rp_tree(X, idx[~left_mask], leaf_size, rng, max_depth - 1)
            return _RPTreeNode(direction=direction, threshold=threshold, left=left, right=right)

    return _RPTreeNode(idx=idx)


def _query_rp_tree(node: _RPTreeNode, x: np.ndarray) -> np.ndarray:
    while node.idx is None:
        node = node.left if float(np.dot(x, node.direction)) <= node.threshold else node.right
    return node.idx


def _augment_candidates(candidates: np.ndarray, i: int, n: int, target: int, rng: np.random.RandomState) -> np.ndarray:
    target = min(max(target, 0), n - 1)
    candidates = np.unique(candidates)
    candidates = candidates[candidates != i]

    if len(candidates) > target:
        candidates = rng.choice(candidates, size=target, replace=False)
        return np.asarray(sorted(candidates), dtype=np.int64)
    if len(candidates) >= target:
        return candidates
    if target >= n - 1:
        all_idx = np.arange(n, dtype=np.int64)
        return all_idx[all_idx != i]

    seen = set(int(j) for j in candidates)
    attempts = 0
    while len(seen) < target and attempts < 20:
        needed = target - len(seen)
        extra = rng.randint(0, n, size=max(16, needed * 3))
        for j in extra:
            jj = int(j)
            if jj != i:
                seen.add(jj)
                if len(seen) >= target:
                    break
        attempts += 1

    if len(seen) < target:
        for jj in range(n):
            if jj != i:
                seen.add(jj)
                if len(seen) >= target:
                    break

    return np.asarray(sorted(seen), dtype=np.int64)


def _candidate_log_affinity(factors, i: int, candidates: np.ndarray, cap: float | None) -> np.ndarray:
    log_s = np.zeros(len(candidates), dtype=np.float64)
    with np.errstate(divide="ignore"):
        for factor in factors:
            weight = _factor_weight(factor)
            if weight == 0.0:
                continue
            term = np.log(np.maximum(_factor_similarity_candidates(factor, i, candidates), 1.0e-300))
            if cap is not None:
                np.maximum(term, -cap, out=term)
            log_s += weight * term
    return log_s


def approx_sparse_model_distances(
    posterior_mat: np.ndarray,
    ll_mat: np.ndarray | None = None,
    k: int = 90,
    affinity: str = "bhattacharyya",
    evidence_cap: float | None = None,
    n_trees: int = 8,
    leaf_size: int | None = None,
    candidate_multiplier: int = 8,
    seed: int | None = None,
) -> scipy.sparse.csr_matrix:
    """Approximate sparse model distances without all-pairs graph construction.

    A random-projection forest proposes candidate neighbors in normalized
    model-factor coordinates. Candidate pairs are then rescored with the exact
    model affinity used by sparse_model_distances, so approximation only enters
    through candidate recall. This is local/non-distributed today, but the
    proposal/evaluation split is the intended boundary for future distributed
    graph construction.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    k = min(k, n - 1)
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    if k <= 0:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    if leaf_size is None:
        leaf_size = max(64, 2 * k)
    leaf_size = min(max(int(leaf_size), k), n)
    n_trees = max(int(n_trees), 1)
    target_candidates = min(n - 1, max(k, int(candidate_multiplier) * k, leaf_size * n_trees))

    row_feat, col_feat = _candidate_features(factors)
    rng = np.random.RandomState(seed)
    idx = np.arange(n, dtype=np.int64)
    max_depth = max(1, int(np.ceil(np.log2(max(n / float(leaf_size), 1.0)))) + 2)
    trees = [_build_rp_tree(col_feat, idx, leaf_size, rng, max_depth) for _ in range(n_trees)]

    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    vals = np.empty(n * k, dtype=np.float64)

    for i in range(n):
        cand_parts = [_query_rp_tree(tree, row_feat[i]) for tree in trees]
        candidates = np.concatenate(cand_parts) if cand_parts else np.empty(0, dtype=np.int64)
        candidates = _augment_candidates(candidates, i, n, target_candidates, rng)

        log_s = _candidate_log_affinity(factors, i, candidates, cap)
        nbr = np.argpartition(-log_s, k - 1)[:k]
        c = candidates[nbr]
        d = -log_s[nbr]
        np.maximum(d, 0.0, out=d)

        s = i * k
        cols[s : s + k] = c
        vals[s : s + k] = d

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def model_knn(
    posterior_mat: np.ndarray,
    ll_mat: np.ndarray | None = None,
    k: int = 15,
    block_size: int = 1024,
    affinity: str = "bhattacharyya",
    evidence_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """k-nearest-neighbor arrays under the model distance d_ij = -log s_ij.

    Returns (indices, distances), each n x k, sorted ascending per row with
    each point as its own first neighbor at distance 0 (the convention
    expected by umap-learn, where self counts toward n_neighbors). Built
    blockwise; the dense affinity matrix is never materialized.
    evidence_cap as in model_log_affinity.
    """
    factors = _affinity_factors(posterior_mat, ll_mat, affinity)
    n = _factor_n(factors[0])
    k = min(k, n)
    m = k - 1  # non-self neighbors
    cap = evidence_cap if (evidence_cap is not None and len(factors) > 1) else None

    knn_idx = np.empty((n, k), dtype=np.int64)
    knn_dist = np.empty((n, k), dtype=np.float64)
    knn_idx[:, 0] = np.arange(n)
    knn_dist[:, 0] = 0.0

    if m == 0:
        return knn_idx, knn_dist

    for s0 in range(0, n, block_size):
        s1 = min(s0 + block_size, n)
        row_idx = np.arange(s0, s1, dtype=np.int64)
        log_s = np.zeros((s1 - s0, n))
        with np.errstate(divide="ignore"):
            for factor in factors:
                weight = _factor_weight(factor)
                if weight == 0.0:
                    continue
                term = np.log(np.maximum(_factor_similarity_block(factor, row_idx), 1.0e-300))
                if cap is not None:
                    np.maximum(term, -cap, out=term)
                log_s += weight * term
        log_s[np.arange(s1 - s0), np.arange(s0, s1)] = -np.inf
        s_blk = log_s

        nbr = np.argpartition(-s_blk, m - 1, axis=1)[:, :m]

        for bi, i in enumerate(range(s0, s1)):
            c = nbr[bi]
            d = -log_s[bi, c]
            np.maximum(d, 0.0, out=d)
            order = np.argsort(d)
            knn_idx[i, 1:] = c[order]
            knn_dist[i, 1:] = d[order]

    return knn_idx, knn_dist
