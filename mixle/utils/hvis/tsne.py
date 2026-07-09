"""t-SNE embedding cores: exact full-matrix and sparse Barnes-Hut.

Holds the heavy-tailed student-t kernel and exact full-matrix gradient
descent (tsne_exact) as well as the sparse Barnes-Hut optimizer
(tsne_barnes_hut) with its quad/oct-tree, the numba/pure-Python repulsion
kernels, and the sparse probability-matrix construction. The numba kernel is
guarded by HAS_NUMBA exactly as before.
"""

import sys

import numpy as np
import scipy.sparse

from mixle.utils.hvis.affinity import _calibrate_row
from mixle.utils.optional_deps import HAS_NUMBA, numba


def t_kernel(tx: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heavy-tailed student-t kernel on embedding tx.

    Returns (Q, num, d2): normalized probabilities Q, the gradient weights
    num_ij = 1 / (1 + d_ij^2 / alpha), and squared distances d2. At alpha = 1
    this is the standard t-SNE kernel.
    """
    n = tx.shape[0]
    rsum = np.sum(np.square(tx), axis=1, keepdims=True)
    d2 = np.dot(-2.0 * tx, tx.T)
    d2 += rsum
    d2 += rsum.T
    np.maximum(d2, 0.0, out=d2)

    num = 1.0 / (1.0 + d2 / alpha)
    qt = num ** ((alpha + 1.0) / 2.0)
    qt[np.arange(n), np.arange(n)] = 0.0
    q = qt / qt.sum()

    return q, num, d2


def _exact_tsne_gradient(
    P: np.ndarray, Y: np.ndarray, alpha: float, min_value: float = 1.0e-128
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of KL(P || Q(Y)) for the dense heavy-tailed t-SNE kernel."""
    Q, num, _ = t_kernel(Y, alpha)
    np.maximum(Q, min_value, out=Q)

    W = (P - Q) * num
    dC = (np.sum(W, axis=1, keepdims=True) * Y - np.dot(W, Y)) * (2.0 * (alpha + 1.0) / alpha)
    return dC, Q


def update_embed(
    P: np.ndarray,
    Y: np.ndarray,
    iY: np.ndarray,
    gains: np.ndarray,
    momentum: float,
    eta: float,
    alpha: float,
    min_gain: float,
    min_value: float = 1.0e-128,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One delta-bar-delta gradient step of KL(P || Q) on the embedding Y.

    Gradient of the heavy-tailed kernel:
        dC/dy_i = (2(alpha+1)/alpha) * sum_j (p_ij - q_ij) num_ij (y_i - y_j),
    computed in matrix form (no per-row Python loop). ``center=False`` skips the mean-centering
    for gauge-fixing goals (anchors, see :mod:`mixle.utils.hvis.goals`), which centering would
    otherwise undo each step.
    """
    dC, Q = _exact_tsne_gradient(P, Y, alpha, min_value=min_value)

    inc = (dC > 0) != (iY > 0)
    gains = np.where(inc, gains + 0.2, gains * 0.8)
    np.maximum(gains, min_gain, out=gains)

    iY = momentum * iY - eta * (gains * dC)
    Y = Y + iY
    if center:
        Y -= np.mean(Y, axis=0, keepdims=True)

    return Y, iY, gains, Q


def _kl(P: np.ndarray, Q: np.ndarray) -> float:
    m = (P > 0) & (Q > 0)
    return float(np.dot(P[m], np.log(P[m]) - np.log(Q[m])))


def update_alpha(
    P: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    min_alpha: float,
    min_value: float,
    max_its: int = 30,
    step: float = 0.1,
    eps: float = 1.0e-6,
    max_alpha: float = 1.0e6,
) -> float:
    """Optimize the kernel tail parameter alpha by guarded Newton steps.

    d(-log q~_ij)/d(alpha) = 0.5 log(1 + d2/alpha) - (alpha+1) d2 / (2 alpha^2 (1 + d2/alpha)),
    and dKL/d(alpha) = sum_ij (p_ij - q_ij) d(-log q~_ij)/d(alpha) (the partition
    function term cancels because sum P = sum Q = 1). Steps are clipped to a
    +-step trust region; alpha is kept in [min_alpha, max_alpha].
    """
    Q, num, d2 = t_kernel(Y, alpha)
    np.maximum(Q, min_value, out=Q)
    kl = _kl(P, Q)

    for _ in range(max_its):
        e_a = 1.0 + d2 / alpha
        dlq = 0.5 * np.log(e_a) - ((alpha + 1.0) / (2.0 * alpha * alpha)) * (d2 / e_a)
        g = float(np.sum((P - Q) * dlq))

        if not np.isfinite(g) or g == 0.0:
            break

        prop = alpha - kl / g if g != 0 else alpha
        prop = min(max(prop, alpha * (1.0 - step)), alpha * (1.0 + step))
        new_alpha = min(max(prop, min_alpha), max_alpha)

        if new_alpha == alpha:
            break

        Q, num, d2 = t_kernel(Y, new_alpha)
        np.maximum(Q, min_value, out=Q)
        new_kl = _kl(P, Q)

        if new_kl > kl - eps:
            break
        alpha, kl = new_alpha, new_kl

    return alpha


def tsne_exact(
    P: np.ndarray,
    emb_dim: int = 2,
    alpha: float = 1.0,
    Y: np.ndarray | None = None,
    max_its: int = 1000,
    eta: float | None = None,
    momentum: float = 0.8,
    early_exaggeration: float = 12.0,
    early_its: int = 250,
    min_gain: float = 0.01,
    min_value: float = 1.0e-128,
    optimize_alpha: bool = False,
    min_alpha: float = 1.0e-6,
    max_alpha_its: int = 3,
    tol: float = 1.0e-7,
    check_every: int = 50,
    print_iter: int = 100,
    seed: int | None = None,
    out=None,
    goals=None,
) -> np.ndarray:
    """Full-matrix t-SNE on symmetrized probabilities P with convergence stopping.

    ``goals`` is an optional sequence of embedding goals (:mod:`mixle.utils.hvis.goals`): their
    gradients join the data gradient every iteration and hard constraints are re-projected after
    every step.
    """
    from mixle.utils.hvis.goals import apply_projections, goals_fix_gauge, total_goal_gradient

    center = not goals_fix_gauge(goals)
    if out is None:
        out = sys.stdout

    P = np.asarray(P, dtype=np.float64).copy()
    P /= P.sum()
    n = P.shape[0]

    if eta is None:
        eta = max(n / early_exaggeration, 50.0)

    if Y is None:
        rng = np.random.RandomState(seed)
        Y = rng.randn(n, emb_dim) * 1.0e-4
    else:
        Y = np.array(Y, dtype=np.float64)
        emb_dim = Y.shape[1]

    iY = np.zeros((n, emb_dim))
    gains = np.ones((n, emb_dim))

    P *= early_exaggeration
    np.maximum(P, min_value, out=P)

    last_kl = np.inf
    for i in range(1, max_its + 1):
        if i == early_its + 1:
            P /= early_exaggeration
            np.maximum(P, min_value, out=P)

        mom = 0.5 if i <= early_its else momentum
        Y, iY, gains, Q = update_embed(P, Y, iY, gains, mom, eta, alpha, min_gain, min_value, center=center)
        step = total_goal_gradient(goals, Y)
        if step is not None:  # goals apply as their own bounded step: rate semantics, no eta/gains coupling
            Y = Y - step
        Y = apply_projections(goals, Y, iY)

        if optimize_alpha and i > early_its:
            alpha = update_alpha(P, Y, alpha, min_alpha, min_value, max_alpha_its)

        if (i % print_iter) == 0:
            out.write("Iteration %d: alpha = %f, KL(P||Q)=%f\n" % (i, alpha, _kl(P, Q)))

        if i > early_its and (i % check_every) == 0:
            kl = _kl(P, Q)
            if last_kl - kl < tol * max(1.0, abs(last_kl)):
                break
            last_kl = kl

    return Y


def _sparse_conditional_pmat(dist_csr: scipy.sparse.csr_matrix, perplexity: float) -> scipy.sparse.csr_matrix:
    """Row-conditional probabilities on a sparse model-neighbor graph.

    dist_csr stores D_ij = -log s_ij, so calibration is performed on -D_ij,
    exactly matching the dense model-affinity path. No metric-distance
    squaring is applied.
    """
    dist_csr = dist_csr.tocsr().astype(np.float64, copy=True)
    n = dist_csr.shape[0]
    if dist_csr.shape[1] != n:
        raise ValueError("dist_csr must be square.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")

    data = np.empty_like(dist_csr.data, dtype=np.float64)
    indptr, indices = dist_csr.indptr, dist_csr.indices
    target_entropy = np.log(perplexity)

    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        neg_d = -np.asarray(dist_csr.data[start:end], dtype=np.float64)
        finite = np.isfinite(neg_d)
        if not np.all(finite):
            neg_d = neg_d.copy()
            neg_d[~finite] = -1.0e300
        data[start:end] = _calibrate_row(neg_d, target_entropy)

    return scipy.sparse.csr_matrix((data, indices.copy(), indptr.copy()), shape=dist_csr.shape)


def _csr_without_diagonal(mat: scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    coo = mat.tocoo()
    keep = coo.row != coo.col
    rv = scipy.sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape)
    rv.sum_duplicates()
    rv.eliminate_zeros()
    return rv


def _sparse_joint_pmat(dist_csr: scipy.sparse.csr_matrix, perplexity: float) -> scipy.sparse.csr_matrix:
    """Symmetrized sparse t-SNE input probabilities from model distances."""
    p_cond = _sparse_conditional_pmat(dist_csr, perplexity)
    p = _csr_without_diagonal(p_cond + p_cond.T)
    p *= 1.0 / (2.0 * p.shape[0])
    total = p.sum()
    if total > 0:
        p *= 1.0 / total
    return p


class _BHNode:
    __slots__ = ("idx", "bmin", "bmax", "center", "width", "mass", "children")

    def __init__(self, idx, bmin, bmax, center, width, children):
        self.idx = idx
        self.bmin = bmin
        self.bmax = bmax
        self.center = center
        self.width = width
        self.mass = len(idx)
        self.children = children


def _build_bh_tree(Y: np.ndarray, idx: np.ndarray, leaf_size: int) -> _BHNode:
    pts = Y[idx]
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    center = pts.mean(axis=0)
    width = float(np.max(bmax - bmin))

    if len(idx) <= leaf_size or width <= 1.0e-12:
        return _BHNode(idx, bmin, bmax, center, width, None)

    mid = 0.5 * (bmin + bmax)
    codes = np.zeros(len(idx), dtype=np.int64)
    for d in range(Y.shape[1]):
        codes |= (pts[:, d] > mid[d]).astype(np.int64) << d

    children = []
    for code in np.unique(codes):
        child_idx = idx[codes == code]
        if len(child_idx) > 0:
            children.append(_build_bh_tree(Y, child_idx, leaf_size))

    if len(children) <= 1:
        return _BHNode(idx, bmin, bmax, center, width, None)

    return _BHNode(idx, bmin, bmax, center, width, children)


def _flatten_bh_tree(root: _BHNode, emb_dim: int):
    """Flatten a Barnes-Hut tree into arrays suitable for numba traversal."""
    bmins = []
    bmaxs = []
    centers = []
    widths = []
    masses = []
    child_starts = []
    child_counts = []
    child_links = []
    leaf_starts = []
    leaf_counts = []
    leaf_links = []

    def walk(node):
        node_id = len(masses)
        bmins.append(np.asarray(node.bmin, dtype=np.float64))
        bmaxs.append(np.asarray(node.bmax, dtype=np.float64))
        centers.append(np.asarray(node.center, dtype=np.float64))
        widths.append(float(node.width))
        masses.append(int(node.mass))
        child_starts.append(0)
        child_counts.append(0)
        leaf_starts.append(0)
        leaf_counts.append(0)

        if node.children is None:
            leaf_starts[node_id] = len(leaf_links)
            leaf_idx = np.asarray(node.idx, dtype=np.int64)
            leaf_links.extend(int(i) for i in leaf_idx)
            leaf_counts[node_id] = len(leaf_idx)
        else:
            direct_children = []
            for child in node.children:
                direct_children.append(walk(child))
            child_starts[node_id] = len(child_links)
            child_links.extend(direct_children)
            child_counts[node_id] = len(direct_children)

        return node_id

    walk(root)
    empty_box = np.empty((0, emb_dim), dtype=np.float64)

    return (
        np.vstack(bmins).astype(np.float64, copy=False) if bmins else empty_box,
        np.vstack(bmaxs).astype(np.float64, copy=False) if bmaxs else empty_box,
        np.vstack(centers).astype(np.float64, copy=False) if centers else empty_box,
        np.asarray(widths, dtype=np.float64),
        np.asarray(masses, dtype=np.int64),
        np.asarray(child_starts, dtype=np.int64),
        np.asarray(child_counts, dtype=np.int64),
        np.asarray(child_links, dtype=np.int64),
        np.asarray(leaf_starts, dtype=np.int64),
        np.asarray(leaf_counts, dtype=np.int64),
        np.asarray(leaf_links, dtype=np.int64),
    )


@numba.njit(cache=True, parallel=True)
def _numba_barnes_hut_negative_forces(
    Y, bmin, bmax, center, width, mass, child_start, child_count, child_index, leaf_start, leaf_count, leaf_index, theta
):
    n, emb_dim = Y.shape
    forces = np.zeros((n, emb_dim), dtype=np.float64)
    z_terms = np.zeros(n, dtype=np.float64)
    num_nodes = width.shape[0]
    eps = 1.0e-12

    for i in numba.prange(n):
        fi = np.zeros(emb_dim, dtype=np.float64)
        zi = 0.0
        stack = np.empty(num_nodes, dtype=np.int64)
        sp = 1
        stack[0] = 0

        while sp > 0:
            sp -= 1
            node = stack[sp]
            if mass[node] == 0:
                continue

            if child_count[node] == 0:
                start = leaf_start[node]
                end = start + leaf_count[node]
                for p in range(start, end):
                    j = leaf_index[p]
                    if j == i:
                        continue
                    d2 = 0.0
                    for d in range(emb_dim):
                        diff = Y[i, d] - Y[j, d]
                        d2 += diff * diff
                    q = 1.0 / (1.0 + d2)
                    coeff = q * q
                    for d in range(emb_dim):
                        fi[d] += coeff * (Y[i, d] - Y[j, d])
                    zi += q
                continue

            d2 = 0.0
            for d in range(emb_dim):
                diff = Y[i, d] - center[node, d]
                d2 += diff * diff

            inside = True
            for d in range(emb_dim):
                if Y[i, d] < bmin[node, d] - eps or Y[i, d] > bmax[node, d] + eps:
                    inside = False
                    break

            if (not inside) and d2 > 0.0 and theta > 0.0 and width[node] / np.sqrt(d2) < theta:
                q = 1.0 / (1.0 + d2)
                coeff = mass[node] * q * q
                for d in range(emb_dim):
                    fi[d] += coeff * (Y[i, d] - center[node, d])
                zi += mass[node] * q
            else:
                start = child_start[node]
                end = start + child_count[node]
                for p in range(start, end):
                    stack[sp] = child_index[p]
                    sp += 1

        for d in range(emb_dim):
            forces[i, d] = fi[d]
        z_terms[i] = zi

    return forces, max(float(np.sum(z_terms)), 1.0e-300)


def _python_barnes_hut_negative_forces(
    Y: np.ndarray, theta: float = 0.5, leaf_size: int = 16
) -> tuple[np.ndarray, float]:
    """Pure-Python Barnes-Hut traversal used as the no-numba fallback."""
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n, emb_dim = Y.shape
    if n <= 1:
        return np.zeros_like(Y), 1.0

    theta = max(float(theta), 0.0)
    leaf_size = max(int(leaf_size), 1)
    root = _build_bh_tree(Y, np.arange(n, dtype=np.int64), leaf_size)
    forces = np.zeros_like(Y)
    z_sum = 0.0
    eps = 1.0e-12

    for i in range(n):
        yi = Y[i]
        fi = np.zeros(emb_dim, dtype=np.float64)
        zi = 0.0
        stack = [root]

        while stack:
            node = stack.pop()
            if node.mass == 0:
                continue

            if node.children is None:
                diff = yi.reshape((1, -1)) - Y[node.idx]
                d2 = np.sum(diff * diff, axis=1)
                q = 1.0 / (1.0 + d2)
                q[node.idx == i] = 0.0
                fi += np.sum((q * q)[:, None] * diff, axis=0)
                zi += float(q.sum())
                continue

            diff = yi - node.center
            d2 = float(np.dot(diff, diff))
            inside = True
            for d in range(emb_dim):
                if yi[d] < node.bmin[d] - eps or yi[d] > node.bmax[d] + eps:
                    inside = False
                    break

            if (not inside) and d2 > 0.0 and theta > 0.0 and node.width / np.sqrt(d2) < theta:
                q = 1.0 / (1.0 + d2)
                fi += node.mass * (q * q) * diff
                zi += node.mass * q
            else:
                stack.extend(node.children)

        forces[i] = fi
        z_sum += zi

    return forces, max(float(z_sum), 1.0e-300)


def _barnes_hut_negative_forces(Y: np.ndarray, theta: float = 0.5, leaf_size: int = 16) -> tuple[np.ndarray, float]:
    """Approximate t-SNE repulsive forces and normalization with Barnes-Hut.

    Returns (F, Z) where

        F_i ~= sum_j (1 + ||y_i-y_j||^2)^-2 (y_i - y_j)
        Z   ~= sum_ij (1 + ||y_i-y_j||^2)^-1.

    With theta <= 0 the traversal descends to leaves, giving the exact sums
    up to floating-point order.
    """
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n, emb_dim = Y.shape
    if n <= 1:
        return np.zeros_like(Y), 1.0

    theta = max(float(theta), 0.0)
    leaf_size = max(int(leaf_size), 1)
    if not HAS_NUMBA:
        return _python_barnes_hut_negative_forces(Y, theta=theta, leaf_size=leaf_size)

    root = _build_bh_tree(Y, np.arange(n, dtype=np.int64), leaf_size)
    return _numba_barnes_hut_negative_forces(Y, *_flatten_bh_tree(root, emb_dim), theta)


def _exact_negative_forces(Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact t-SNE repulsive forces, vectorized over all pairs."""
    rsum = np.sum(Y * Y, axis=1, keepdims=True)
    d2 = -2.0 * np.dot(Y, Y.T)
    d2 += rsum
    d2 += rsum.T
    np.maximum(d2, 0.0, out=d2)
    d2 += 1.0
    np.reciprocal(d2, out=d2)
    d2[np.arange(Y.shape[0]), np.arange(Y.shape[0])] = 0.0
    z_sum = float(d2.sum())
    d2 *= d2
    forces = np.sum(d2, axis=1, keepdims=True) * Y - np.dot(d2, Y)
    return forces, max(z_sum, 1.0e-300)


def _negative_forces(
    Y: np.ndarray, theta: float, leaf_size: int, repulsion_method: str, exact_threshold: int
) -> tuple[np.ndarray, float]:
    method = repulsion_method
    if method == "auto":
        method = "exact" if Y.shape[0] <= exact_threshold else "barnes_hut"
    if method == "exact":
        return _exact_negative_forces(Y)
    if method == "barnes_hut":
        return _barnes_hut_negative_forces(Y, theta=theta, leaf_size=leaf_size)
    raise ValueError("repulsion_method must be 'auto', 'exact', or 'barnes_hut'.")


def _dispatch_negative_forces(*args, **kwargs) -> tuple[np.ndarray, float]:
    """Call _negative_forces through the htsne package namespace.

    When this module lived as a single ``mixle.utils.hvis`` module, the
    Barnes-Hut loop read ``_negative_forces`` as a module global, so rebinding
    ``mixle.utils.hvis._negative_forces`` redirected the call. Preserving that
    hook now that the package re-exports the name: resolve the (possibly
    monkeypatched) package attribute at call time and fall back to the local
    definition.
    """
    pkg = sys.modules.get("mixle.utils.hvis")
    fn = getattr(pkg, "_negative_forces", _negative_forces) if pkg is not None else _negative_forces
    return fn(*args, **kwargs)


def _sparse_positive_forces_from_edges(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray, Y: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """Exact attractive t-SNE forces over sparse probability edges."""
    if len(data) == 0:
        return np.zeros_like(Y)

    diff = Y[rows] - Y[cols]
    d2 = np.sum(diff * diff, axis=1)
    num = 1.0 / (1.0 + d2)
    weighted = (scale * data * num)[:, None] * diff
    forces = np.zeros_like(Y)
    np.add.at(forces, rows, weighted)
    return forces


def _sparse_positive_forces_symmetric_from_edges(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray, Y: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """Exact attractive forces for upper-triangular edges from a symmetric P."""
    if len(data) == 0:
        return np.zeros_like(Y)

    diff = Y[rows] - Y[cols]
    d2 = np.sum(diff * diff, axis=1)
    num = 1.0 / (1.0 + d2)
    weighted = (scale * data * num)[:, None] * diff
    forces = np.zeros_like(Y)
    np.add.at(forces, rows, weighted)
    np.add.at(forces, cols, -weighted)
    return forces


def _sparse_positive_forces(P: scipy.sparse.csr_matrix, Y: np.ndarray) -> np.ndarray:
    """Exact attractive t-SNE forces over nonzero sparse probabilities."""
    p = P.tocoo()
    return _sparse_positive_forces_from_edges(p.row, p.col, p.data, Y)


def _sparse_tsne_kl(P: scipy.sparse.csr_matrix, Y: np.ndarray, z_sum: float | None = None) -> float:
    p = P.tocoo()
    if p.nnz == 0:
        return 0.0
    if z_sum is None:
        _, z_sum = _barnes_hut_negative_forces(Y, theta=0.0, leaf_size=1)
    return _sparse_tsne_kl_from_edges(p.row, p.col, p.data, Y, z_sum)


def _sparse_tsne_kl_from_edges(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray, Y: np.ndarray, z_sum: float
) -> float:
    if len(data) == 0:
        return 0.0
    diff = Y[rows] - Y[cols]
    d2 = np.sum(diff * diff, axis=1)
    q = np.maximum((1.0 / (1.0 + d2)) / z_sum, 1.0e-300)
    pdata = np.maximum(data, 1.0e-300)
    return float(np.dot(pdata, np.log(pdata) - np.log(q)))


def _tsne_barnes_hut_from_p(
    P: scipy.sparse.csr_matrix,
    emb_dim: int = 2,
    max_its: int = 1000,
    eta: float | None = None,
    momentum: float = 0.8,
    early_exaggeration: float = 12.0,
    early_its: int = 250,
    min_gain: float = 0.01,
    tol: float = 1.0e-7,
    check_every: int = 50,
    print_iter: int = 100,
    theta: float = 0.5,
    leaf_size: int = 16,
    repulsion_method: str = "auto",
    exact_repulsion_threshold: int = 5000,
    seed: int | None = None,
    Y: np.ndarray | None = None,
    out=None,
    goals=None,
) -> np.ndarray:
    """Barnes-Hut t-SNE on a sparse symmetric probability matrix. ``goals`` as in
    :func:`tsne_exact`: goal gradients join the data gradient each iteration, hard constraints are
    re-projected each step, and mean-centering is skipped when a goal fixes the gauge."""
    from mixle.utils.hvis.goals import apply_projections, goals_fix_gauge, total_goal_gradient

    center = not goals_fix_gauge(goals)
    if out is None:
        out = sys.stdout

    P = _csr_without_diagonal(P).astype(np.float64, copy=False)
    total = P.sum()
    if total <= 0:
        raise ValueError("P must contain positive off-diagonal probabilities.")
    P *= 1.0 / total

    n = P.shape[0]
    if eta is None:
        eta = max(n / early_exaggeration, 50.0)

    if Y is None:
        rng = np.random.RandomState(seed)
        Y = rng.randn(n, emb_dim) * 1.0e-4
    else:
        Y = np.asarray(Y, dtype=np.float64).copy()
        if Y.shape[0] != n:
            raise ValueError("initial embedding Y has the wrong number of rows.")
        emb_dim = Y.shape[1]
    if center:
        Y -= np.mean(Y, axis=0, keepdims=True)

    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    n_iter = max(int(max_its), 0)
    last_kl = np.inf
    p_edges = P.tocoo()
    p_rows, p_cols, p_data = p_edges.row, p_edges.col, p_edges.data
    p_upper = scipy.sparse.triu(P, k=1).tocoo()
    pos_rows, pos_cols, pos_data = p_upper.row, p_upper.col, p_upper.data

    for it in range(1, n_iter + 1):
        exaggeration = early_exaggeration if it <= early_its else 1.0

        pos = _sparse_positive_forces_symmetric_from_edges(pos_rows, pos_cols, pos_data, Y, exaggeration)
        neg, z_sum = _dispatch_negative_forces(
            Y,
            theta=theta,
            leaf_size=leaf_size,
            repulsion_method=repulsion_method,
            exact_threshold=exact_repulsion_threshold,
        )
        dC = 4.0 * (pos - neg / z_sum)

        inc = (dC > 0) != (iY > 0)
        gains = np.where(inc, gains + 0.2, gains * 0.8)
        np.maximum(gains, min_gain, out=gains)

        mom = 0.5 if it <= early_its else momentum
        iY = mom * iY - eta * gains * dC
        Y = Y + iY
        if center:
            Y -= np.mean(Y, axis=0, keepdims=True)
        step = total_goal_gradient(goals, Y)
        if step is not None:  # bounded rate semantics, decoupled from eta/gains (see goals module)
            Y = Y - step
        Y = apply_projections(goals, Y, iY)

        if (it % print_iter) == 0:
            _, kl_z_sum = _dispatch_negative_forces(
                Y,
                theta=theta,
                leaf_size=leaf_size,
                repulsion_method=repulsion_method,
                exact_threshold=exact_repulsion_threshold,
            )
            kl = _sparse_tsne_kl_from_edges(p_rows, p_cols, p_data, Y, kl_z_sum)
            out.write("Iteration %d: KL(P||Q)=%f\n" % (it, kl))

        if it > early_its and (it % check_every) == 0:
            _, kl_z_sum = _dispatch_negative_forces(
                Y,
                theta=theta,
                leaf_size=leaf_size,
                repulsion_method=repulsion_method,
                exact_threshold=exact_repulsion_threshold,
            )
            kl = _sparse_tsne_kl_from_edges(p_rows, p_cols, p_data, Y, kl_z_sum)
            if last_kl - kl < tol * max(1.0, abs(last_kl)):
                break
            last_kl = kl

    return Y


def _tsne_barnes_hut(
    dist_csr: scipy.sparse.csr_matrix,
    emb_dim: int,
    perplexity: float,
    max_its: int,
    eta,
    momentum: float,
    early_exaggeration: float,
    min_gain: float,
    tol: float,
    print_iter: int,
    seed: int | None,
    Y: np.ndarray | None,
    out=None,
    theta: float = 0.5,
    leaf_size: int = 16,
    repulsion_method: str = "auto",
    exact_repulsion_threshold: int = 5000,
    goals=None,
) -> np.ndarray:
    P = _sparse_joint_pmat(dist_csr, perplexity)
    return _tsne_barnes_hut_from_p(
        P,
        emb_dim=emb_dim,
        max_its=max_its,
        eta=eta,
        momentum=momentum,
        early_exaggeration=early_exaggeration,
        min_gain=min_gain,
        tol=tol,
        print_iter=print_iter,
        theta=theta,
        leaf_size=leaf_size,
        repulsion_method=repulsion_method,
        exact_repulsion_threshold=exact_repulsion_threshold,
        seed=seed,
        Y=Y,
        out=out,
        goals=goals,
    )


def tsne_barnes_hut(
    P: scipy.sparse.csr_matrix,
    emb_dim: int = 2,
    max_its: int = 1000,
    eta: float | None = None,
    momentum: float = 0.8,
    early_exaggeration: float = 12.0,
    min_gain: float = 0.01,
    tol: float = 1.0e-7,
    print_iter: int = 100,
    theta: float = 0.5,
    leaf_size: int = 16,
    repulsion_method: str = "auto",
    exact_repulsion_threshold: int = 5000,
    seed: int | None = None,
    Y: np.ndarray | None = None,
    out=None,
) -> np.ndarray:
    """Embed a precomputed sparse t-SNE probability matrix with Barnes-Hut.

    P must be a symmetric, non-negative affinity/probability matrix. It is
    normalized internally. This function is self-contained and does not call
    sklearn.
    """
    return _tsne_barnes_hut_from_p(
        P,
        emb_dim=emb_dim,
        max_its=max_its,
        eta=eta,
        momentum=momentum,
        early_exaggeration=early_exaggeration,
        min_gain=min_gain,
        tol=tol,
        print_iter=print_iter,
        theta=theta,
        leaf_size=leaf_size,
        repulsion_method=repulsion_method,
        exact_repulsion_threshold=exact_repulsion_threshold,
        seed=seed,
        Y=Y,
        out=out,
    )
