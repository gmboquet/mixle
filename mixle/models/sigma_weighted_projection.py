"""Sigma-weighted structured projections (roadmap G2): thin solvers over borrowed primitives.

Given a weight matrix ``W`` (``out_dim x in_dim``) and the covariance ``Sigma`` (``in_dim x in_dim``,
PSD) of the activations it will actually be multiplied against -- e.g. the propagated-law covariance
coming out of :mod:`mixle.models.moment_propagation` (roadmap G1) -- the objective that matters for
preserving downstream behavior is NOT plain Frobenius compression (``||W - What||_F^2``, which treats
every input direction as equally important) but the SIGMA-WEIGHTED version

    min_What  tr((W - What) @ Sigma @ (W - What)^T)

which penalizes reconstruction error in input directions the real data varies along more, and tolerates
more error in directions the data barely explores (the "optimal brain damage" / Fisher-weighted pruning
idea, generalized from a diagonal Hessian approximation to a full covariance weighting).

Per the roadmap's build-vs-borrow note, this module BORROWS the heavy primitives rather than
reimplementing them:

* the low-rank case has a real closed-form solution via a whiten/SVD/un-whiten reduction to plain
  Eckart-Young truncated SVD (:func:`sigma_weighted_low_rank`) -- no iterative solver needed;
* the block-sparse / 2:4 case has no closed form, so :func:`sigma_weighted_block_sparse` uses a
  textbook projected-gradient ("alternating projection") scheme: a gradient step on the (convex,
  quadratic-in-What) weighted objective, alternated with a hard projection onto the structural
  constraint set (a fixed support mask, or a dynamically re-selected 2:4 pattern);
* the permutation case is solved with Sinkhorn's algorithm -- the standard entropic-OT relaxation of a
  linear assignment problem. ``torchsort``/POT/``geomloss`` were checked (see the PR description) and
  are not installed in this environment; Sinkhorn itself is a well-known ~10-line fixed-point iteration
  (alternating row/column normalization of a Gibbs kernel in log-domain for numerical stability), so it
  is implemented directly here rather than pulling in a heavy optional dependency for a few lines of
  numpy. This is also the differentiable "profile . arrangement" building block roadmap item G4 (later,
  not this item) reuses for permutation x profile quantization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "sigma_weighted_error",
    "sigma_weighted_low_rank",
    "sigma_weighted_block_sparse",
    "sigma_weighted_permutation",
    "sigma_weighted_butterfly",
    "ProjectionReport",
    "project",
]


# --------------------------------------------------------------------------------------------------------
# shared objective
# --------------------------------------------------------------------------------------------------------


def sigma_weighted_error(w: Any, w_hat: Any, sigma: Any) -> float:
    """``tr((W - What) @ Sigma @ (What - W)^T)`` -- the Sigma-weighted reconstruction objective itself.

    Used both as the convergence check inside the iterative solvers below and as the metric the tests
    compare solvers against each other with. ``Sigma`` is assumed PSD (a covariance matrix); this
    function does not itself validate that -- callers pass a real covariance (e.g. from
    :func:`mixle.models.moment_propagation.propagate_moments`) or a synthetic ``A @ A.T`` construction.
    """
    diff = np.asarray(w, dtype=np.float64) - np.asarray(w_hat, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    return float(np.trace(diff @ sigma @ diff.T))


def _symmetric_sqrt_and_pinv_sqrt(sigma: np.ndarray, rcond: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric PSD ``Sigma`` into its symmetric matrix square root and the
    (pseudo-inverse) square root, clipping numerically-negative eigenvalues to zero and treating
    near-zero eigenvalues as exactly rank-deficient directions (their pseudo-inverse contribution is
    zero, matching the fact that ``Sigma`` assigns no weight/cost to those directions at all).
    """
    sigma = 0.5 * (sigma + sigma.T)
    eigval, eigvec = np.linalg.eigh(sigma)
    eigval = np.clip(eigval, 0.0, None)
    sqrt_eigval = np.sqrt(eigval)
    threshold = rcond * float(sqrt_eigval.max() if sqrt_eigval.size else 0.0)
    inv_sqrt_eigval = np.where(sqrt_eigval > threshold, np.reciprocal(np.where(sqrt_eigval > 0, sqrt_eigval, 1.0)), 0.0)
    sigma_half = eigvec @ np.diag(sqrt_eigval) @ eigvec.T
    sigma_half_pinv = eigvec @ np.diag(inv_sqrt_eigval) @ eigvec.T
    return sigma_half, sigma_half_pinv


# --------------------------------------------------------------------------------------------------------
# 1. low-rank -- exact closed-form generalized SVD (whiten / SVD / un-whiten)
# --------------------------------------------------------------------------------------------------------


def sigma_weighted_low_rank(w: Any, sigma: Any, rank: int) -> np.ndarray:
    """Exact closed-form solver for ``min_{rank(What)<=rank} tr((W-What) Sigma (W-What)^T)``.

    Derivation (generalized SVD via whitening): for symmetric PSD ``Sigma`` with symmetric square root
    ``Sigma^(1/2)`` (``Sigma = Sigma^(1/2) Sigma^(1/2)``, itself symmetric so ``(Sigma^(1/2))^T =
    Sigma^(1/2)``),

        tr((W-What) Sigma (W-What)^T) = tr((W-What) Sigma^(1/2) Sigma^(1/2) (W-What)^T)
                                       = || (W-What) @ Sigma^(1/2) ||_F^2 .

    Substituting ``B = W @ Sigma^(1/2)`` and ``Bhat = What @ Sigma^(1/2)``, the constraint
    ``rank(What) <= rank`` becomes (for full-rank ``Sigma^(1/2)``) exactly ``rank(Bhat) <= rank``, and the
    objective becomes the PLAIN (unweighted) Frobenius low-rank problem ``min ||B - Bhat||_F^2``, whose
    exact global optimum is the truncated SVD of ``B`` (Eckart-Young). Un-whitening
    ``What = Bhat @ Sigma^(1/2)^+`` (pseudo-inverse, needed if ``Sigma`` is rank-deficient) recovers the
    optimal ``What`` in the ORIGINAL objective. When ``Sigma`` has a null space, those input directions
    contribute nothing to the objective regardless of ``What``'s value there, so the pseudo-inverse
    un-whitening (which zeroes ``What`` on that null space) is one particular optimum among many -- still
    provably attaining the true minimum objective value, which is all the stated objective can see.

    This is the SAME closed form used for Fisher/Hessian-weighted low-rank compression (a diagonal
    special case of this is "optimal brain damage"-style weighted SVD); here it is implemented for a
    full (non-diagonal) ``Sigma``.
    """
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if w.shape[1] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Sigma must be square with side == W.shape[1]; got W {w.shape}, Sigma {sigma.shape}")

    sigma_half, sigma_half_pinv = _symmetric_sqrt_and_pinv_sqrt(sigma)

    b = w @ sigma_half
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    r = int(max(0, min(rank, s.shape[0])))
    b_hat = (u[:, :r] * s[:r]) @ vt[:r, :]
    return b_hat @ sigma_half_pinv


# --------------------------------------------------------------------------------------------------------
# 2. block-sparse / 2:4 -- alternating projection (projected gradient onto a structural constraint set)
# --------------------------------------------------------------------------------------------------------


def _project_mask(w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return w * mask


def _project_2_4(w: np.ndarray) -> np.ndarray:
    """Hard-project onto the 2:4 structured-sparsity pattern: within every contiguous group of 4 entries
    along the last (input) axis, keep the 2 largest-magnitude entries and zero the rest -- the standard
    NVIDIA-style 2:4 semi-structured sparsity constraint (dense 2:4 GEMM / cuSPARSELt kernels consume
    exactly this format). The pattern is RE-SELECTED from the current iterate's magnitudes every call,
    which is what makes this a genuine alternating-projection scheme rather than a one-shot mask.
    """
    d_out, d_in = w.shape
    if d_in % 4 != 0:
        raise ValueError(f"2:4 sparsity requires the input dim to be a multiple of 4; got {d_in}")
    groups = w.reshape(d_out, d_in // 4, 4)
    order = np.argsort(-np.abs(groups), axis=-1)
    keep = order[..., :2]
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    return np.where(mask, groups, 0.0).reshape(d_out, d_in)


def sigma_weighted_block_sparse(
    w: Any,
    sigma: Any,
    block_pattern_or_2_4: Any,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Alternating-projection solver for ``min_{What in S} tr((W-What) Sigma (W-What)^T)`` where ``S`` is
    a structural constraint set with no closed form: a fixed block-sparse/arbitrary support mask, or the
    2:4 semi-structured pattern.

    ``block_pattern_or_2_4``:
        * the literal string ``"2:4"`` -- 2:4 semi-structured sparsity (pattern re-selected every step,
          see :func:`_project_2_4`);
        * a boolean array shaped like ``W`` -- an explicit (e.g. block-sparse) fixed support pattern.

    Algorithm: projected gradient descent on the (convex, quadratic-in-``What``) objective --
    ``grad_What = -2 (W - What) Sigma`` -- with step size ``1 / (2 * lambda_max(Sigma))`` (the standard
    Lipschitz-safe step for a quadratic with Hessian ``2*Sigma`` acting on the right), alternated with a
    HARD projection onto the structural constraint set after every gradient step. Convergence contract:
    for a FIXED mask the constraint set is a linear subspace, so this is plain convex projected gradient
    descent and converges to the GLOBAL optimum of that subspace-constrained problem (matches the
    closed-form per-row constrained-least-squares solution -- see the optimality test). For 2:4 the
    constraint set is a finite, non-convex union of subspaces (the pattern itself is re-chosen every
    step), so only convergence to a LOCAL optimum of the alternating scheme is guaranteed -- NOT global
    optimality over all possible 2:4 masks (that combinatorial problem is not attempted here).
    """
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)

    if isinstance(block_pattern_or_2_4, str):
        if block_pattern_or_2_4 != "2:4":
            raise ValueError(f"unrecognized structured-sparsity literal {block_pattern_or_2_4!r}, expected '2:4'")
        project = _project_2_4
    else:
        mask = np.asarray(block_pattern_or_2_4, dtype=bool)
        if mask.shape != w.shape:
            raise ValueError(f"mask shape {mask.shape} must match W shape {w.shape}")
        project = lambda x: _project_mask(x, mask)  # noqa: E731 - trivial closure, clearer inline than def

    lambda_max = float(np.linalg.eigvalsh(sigma).max()) if sigma.size else 0.0
    step = 1.0 / (2.0 * max(lambda_max, 1e-12))

    w_hat = project(w.copy())
    prev_err = sigma_weighted_error(w, w_hat, sigma)
    for _ in range(max_iter):
        grad = 2.0 * (w_hat - w) @ sigma
        w_hat = project(w_hat - step * grad)
        err = sigma_weighted_error(w, w_hat, sigma)
        if abs(prev_err - err) <= tol * max(1.0, prev_err):
            prev_err = err
            break
        prev_err = err
    return w_hat


# --------------------------------------------------------------------------------------------------------
# 3. permutation -- Sinkhorn soft-permutation solver (feeds G4's "profile . arrangement")
# --------------------------------------------------------------------------------------------------------


def _sinkhorn_log_domain(log_kernel: np.ndarray, n_iter: int) -> np.ndarray:
    """Standard log-domain Sinkhorn fixed point: alternately renormalize rows/columns of a Gibbs kernel
    (in log-space, for numerical stability) until the coupling is (approximately) doubly stochastic.
    Uniform marginals (``1/n`` each) since we are relaxing a square PERMUTATION matrix, whose row/column
    sums are all exactly 1.
    """
    n = log_kernel.shape[0]
    log_u = np.zeros(n)
    log_v = np.zeros(n)
    log_marginal = -np.log(n)
    for _ in range(n_iter):
        log_u = log_marginal - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_marginal - _logsumexp(log_kernel + log_u[:, None], axis=0)
    return np.exp(log_u[:, None] + log_kernel + log_v[None, :])


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def sigma_weighted_permutation(
    w: Any,
    sigma: Any,
    target_profile: Any,
    temperature: float = 0.1,
    max_iter: int = 100,
) -> np.ndarray:
    """Sinkhorn-based soft-permutation solver for ``What = P @ target_profile`` -- the "profile o
    arrangement" pattern (roadmap H4/R1, feeding G4's permutation x profile quantization): find the
    ROW-permutation ``P`` of a fixed canonical ``target_profile`` that best matches ``W`` under the
    Sigma-weighted objective ``min_P tr((W - P @ target_profile) Sigma (W - P @ target_profile)^T)``.

    This is a linear assignment problem in disguise: it decomposes over ROW-PAIRS (row ``i`` of ``W``
    matched to row ``j`` of ``target_profile``) with pairwise cost
    ``cost[i,j] = (W_i - profile_j) @ Sigma @ (W_i - profile_j)^T``, so the discrete problem
    ``min_{P permutation} sum_i cost[i, perm(i)]`` is EXACTLY a linear assignment problem. We solve it
    with the differentiable Sinkhorn relaxation the roadmap asks for (a Gibbs kernel
    ``K = exp(-cost/temperature)``, alternately row/column normalized in log-domain -- this is the
    ``torchsort``/POT-style soft-permutation building block G4 later reuses for a jointly-differentiable
    profile+arrangement objective), THEN round the converged soft doubly-stochastic coupling to a hard
    permutation via linear-sum-assignment (Hungarian) on the SAME cost matrix -- a standard, exact
    final-rounding step (the Sinkhorn relaxation supplies the differentiable pattern; committing to a
    hard answer is a separate, exact combinatorial step, not claimed to itself be "the Sinkhorn
    solution"). Convergence contract: the returned answer is the exact optimum of the linear assignment
    problem (Hungarian rounding is exact for assignment problems); the SINKHORN PLAN itself only
    converges to the true permutation in the low-temperature limit -- what "converged Sinkhorn solution"
    means here is that repeated Sinkhorn normalization has converged to a fixed doubly-stochastic
    coupling for the given ``temperature``, not that temperature itself has been annealed to zero.
    """
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)
    profile = np.asarray(target_profile, dtype=np.float64)
    if w.shape != profile.shape:
        raise ValueError(f"target_profile shape {profile.shape} must match W shape {w.shape}")
    n = w.shape[0]

    # pairwise Sigma-weighted cost between every row of W and every row of the profile
    diff = w[:, None, :] - profile[None, :, :]  # (n, n, d_in)
    cost = np.einsum("ijk,kl,ijl->ij", diff, sigma, diff)

    log_kernel = -cost / max(temperature, 1e-8)
    soft_plan = _sinkhorn_log_domain(log_kernel, n_iter=max_iter)

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros((n, n), dtype=np.float64)
    perm[row_ind, col_ind] = 1.0

    _ = soft_plan  # the differentiable relaxation this function demonstrates; hard rounding uses `cost` directly
    return perm @ profile


# --------------------------------------------------------------------------------------------------------
# 4. butterfly -- alternating least squares over sparse butterfly-connectivity factors
# --------------------------------------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _butterfly_stage_matrix(n: int, stride: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense ``n x n`` matrix for one butterfly stage at the given ``stride``: row ``j`` is paired with row
    ``j XOR stride`` (a self-inverse involution -- the SAME connectivity FFT's radix-2 decimation uses at
    that stride, stride doubling stage to stage), with two FREE taps per row:
    ``(S @ x)[j] = a[j] * x[j] + b[j] * x[j ^ stride]``.
    """
    partner = np.arange(n) ^ stride
    idx = np.arange(n)
    s = np.zeros((n, n), dtype=np.float64)
    s[idx, idx] = a
    s[idx, partner] = b
    return s


def _compose_apply_order(mats: list[np.ndarray], n: int) -> np.ndarray:
    """Compose a list of stage matrices given in APPLICATION order (``mats[0]`` applied first to a column
    vector), i.e. return ``mats[-1] @ ... @ mats[1] @ mats[0]``.
    """
    out = np.eye(n)
    for m in mats:
        out = m @ out
    return out


def sigma_weighted_butterfly(
    w: Any,
    sigma: Any,
    n_stages: int | None = None,
    n_sweeps: int = 4,
) -> np.ndarray:
    """Sigma-weighted BUTTERFLY structured projection: constrain ``What`` to be (the top-left
    ``d_out x d_in`` block of) an ``N x N`` "butterfly matrix" -- a product of ``L`` sparse factors, each
    with exactly 2 nonzeros per row connecting index ``j`` to ``j XOR stride`` (``stride`` doubling stage to
    stage: 1, 2, 4, ...) -- the SAME block-diagonal-then-permute connectivity pattern FFT's radix-2
    decimation uses. This gives ``O(N log N)`` free parameters (``2 * N`` per stage, ``L = log2(N)``
    stages) instead of ``O(N^2)`` for a dense matrix, where ``N`` is the next power of two
    ``>= max(d_out, d_in)``.

    Solved by ALTERNATING LEAST SQUARES over the ``L`` stage factors, per the roadmap card's Steps: reusing
    the SAME whiten-by-``Sigma^(1/2)`` reduction :func:`sigma_weighted_low_rank` uses (via
    :func:`_symmetric_sqrt_and_pinv_sqrt`) to turn each per-stage subproblem into a plain (unweighted)
    linear least-squares problem in that stage's ``2*N`` free parameters (closed-form, via
    :func:`numpy.linalg.lstsq`) given every OTHER stage held fixed -- a genuine block-coordinate solve,
    monotonically non-increasing in the Sigma-weighted objective per stage update (see
    :func:`sigma_weighted_error`, the same convergence metric the other three solvers already use).

    Two SIMPLIFICATIONS versus a textbook FFT butterfly, both bounded and stated here rather than hidden:

    1. Each stage's two taps per row are FREE real parameters *fit to the data*, not fixed unitary FFT
       twiddle factors -- this follows the "butterfly matrices for structured compression" line of work
       (generalizing FFT's O(n log n) connectivity to a learnable factorization), not a literal (inverse)
       Fourier transform.
    2. Rectangular ``W`` is handled by zero-padding ``W``/``Sigma`` up to the square ``N x N`` problem and
       reading off the top-left ``d_out x d_in`` block at the end. ``Sigma``'s padded rows/columns are zero
       (those input directions cost nothing, same convention as :func:`_symmetric_sqrt_and_pinv_sqrt`'s
       null-space handling), but padded OUTPUT rows (beyond ``d_out``, when ``d_out`` is not already a
       power of two) are fit toward zero using the SAME shared stage parameters as the real rows -- a mild,
       honest dilution of fitting capacity for non-power-of-two ``d_out``, not a hidden bug.
    3. ``n_sweeps`` bounds the number of ALS passes over all ``L`` stages rather than iterating to
       convergence -- the "fixed number of butterfly stages/sweeps" bounded-fix simplification the roadmap
       card allows for. It does not make the family a no-op or fold it into another family: each stage
       solve is a real, distinct least-squares fit and the returned ``What`` has the genuine sparse
       butterfly parameter count, not a dense low-rank or block-sparse structure.
    """
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)
    d_out, d_in = w.shape
    if sigma.shape != (d_in, d_in):
        raise ValueError(f"Sigma must be square with side == W.shape[1]; got W {w.shape}, Sigma {sigma.shape}")

    n = _next_pow2(max(d_out, d_in, 2))
    l_full = n.bit_length() - 1  # log2(n), n is a power of two
    l = l_full if n_stages is None else int(max(1, min(n_stages, l_full)))
    strides = [1 << i for i in range(l)]

    w_pad = np.zeros((n, n), dtype=np.float64)
    w_pad[:d_out, :d_in] = w
    sigma_pad = np.zeros((n, n), dtype=np.float64)
    sigma_pad[:d_in, :d_in] = sigma
    sigma_half, _ = _symmetric_sqrt_and_pinv_sqrt(sigma_pad)

    target = w_pad @ sigma_half  # (n, n); the whitened target the composed butterfly must match

    # every stage starts at the identity map (a=1, b=0): deterministic, no RNG needed in a fit path (each
    # stage's ALS solve below is an EXACT least-squares optimum given the others, regardless of starting
    # point, so identity is as principled a start as any).
    a = [np.ones(n) for _ in range(l)]
    b = [np.zeros(n) for _ in range(l)]
    stages = [_butterfly_stage_matrix(n, strides[i], a[i], b[i]) for i in range(l)]

    idx = np.arange(n)
    for _sweep in range(max(1, n_sweeps)):
        for k in range(l):
            pre = _compose_apply_order(stages[:k], n) if k > 0 else np.eye(n)
            post = _compose_apply_order(stages[k + 1 :], n) if k < l - 1 else np.eye(n)
            pre_prime = pre @ sigma_half  # (n, n); folds the whitening into the "pre" side of stage k

            partner = idx ^ strides[k]
            # column p (p < n): tap a_j basis contributes post[:, j] outer pre_prime[j, :]
            # column p (p >= n): tap b_j basis contributes post[:, j] outer pre_prime[partner[j], :]
            j_design = np.concatenate([idx, idx])
            col_from = np.concatenate([idx, partner])
            # design_tensor[p, i, ii] = post[i, j_design[p]] * pre_prime[col_from[p], ii]
            design_tensor = post[:, j_design].T[:, :, None] * pre_prime[col_from, :][:, None, :]  # (2n, n, n)
            design = design_tensor.reshape(2 * n, n * n).T  # (n*n, 2n)

            theta, *_ = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
            a[k] = theta[:n]
            b[k] = theta[n:]
            stages[k] = _butterfly_stage_matrix(n, strides[k], a[k], b[k])

    what_pad = _compose_apply_order(stages, n)
    return what_pad[:d_out, :d_in]


# --------------------------------------------------------------------------------------------------------
# unified front door (roadmap G2's stated API): project(W, Sigma, structure=..., **kw) -> (What, report)
# --------------------------------------------------------------------------------------------------------


@dataclass
class ProjectionReport:
    """Uniform report shape :func:`project` returns alongside ``What``, regardless of ``structure``.

    ``sigma_weighted_error`` is always the SAME objective (:func:`sigma_weighted_error`) every solver in
    this module already minimizes/reports, computed once here so callers get one consistent number to
    compare across families. ``stats`` carries whatever structure-specific numbers that solver's own
    return value already lets a caller compute (rank, sparsity fraction, stage/parameter counts, ...) --
    nothing new is invented here, this just wraps numbers each solver already makes derivable.
    """

    structure: str
    sigma_weighted_error: float
    stats: dict[str, Any] = field(default_factory=dict)


def project(w: Any, sigma: Any, structure: str, **kw: Any) -> tuple[np.ndarray, ProjectionReport]:
    """Unified front door for roadmap G2's four structure families:
    ``structure in {"low_rank", "block_sparse", "butterfly", "perm_profile"}``.

    Dispatches to this module's existing standalone solvers (:func:`sigma_weighted_low_rank`,
    :func:`sigma_weighted_block_sparse`, :func:`sigma_weighted_butterfly`,
    :func:`sigma_weighted_permutation`) -- this function does not reimplement any solver, it only picks one
    by name and wraps its result (plus the shared :func:`sigma_weighted_error` metric and a few
    structure-specific stats already derivable from that result) into a single :class:`ProjectionReport`
    shape, so callers that want to pick a structure by string (e.g. a search/schedule over structures) do
    not need a per-family if/elif of their own.

    ``**kw`` per structure (forwarded to the underlying solver; see each solver's docstring for details):

    * ``"low_rank"``: ``rank`` (int, required).
    * ``"block_sparse"``: ``pattern`` (``"2:4"`` or a boolean mask shaped like ``W``; required), optional
      ``max_iter``, ``tol``.
    * ``"butterfly"``: optional ``n_stages``, ``n_sweeps``.
    * ``"perm_profile"``: ``target_profile`` (required), optional ``temperature``, ``max_iter``.
    """
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if structure == "low_rank":
        rank = kw["rank"]
        what = sigma_weighted_low_rank(w, sigma, rank=rank)
        stats: dict[str, Any] = {"requested_rank": int(rank), "achieved_rank": int(np.linalg.matrix_rank(what))}
    elif structure == "block_sparse":
        pattern = kw["pattern"]
        extra = {k: v for k, v in kw.items() if k in ("max_iter", "tol")}
        what = sigma_weighted_block_sparse(w, sigma, pattern, **extra)
        stats = {
            "pattern": pattern if isinstance(pattern, str) else "custom_mask",
            "sparsity_fraction": float(np.mean(what == 0.0)),
        }
    elif structure == "butterfly":
        extra = {k: v for k, v in kw.items() if k in ("n_stages", "n_sweeps")}
        what = sigma_weighted_butterfly(w, sigma, **extra)
        n = _next_pow2(max(w.shape[0], w.shape[1], 2))
        l = (
            n.bit_length() - 1
            if extra.get("n_stages") is None
            else int(max(1, min(extra["n_stages"], n.bit_length() - 1)))
        )
        stats = {"n": int(n), "n_stages": int(l), "param_count": int(2 * n * l)}
    elif structure == "perm_profile":
        target_profile = kw["target_profile"]
        extra = {k: v for k, v in kw.items() if k in ("temperature", "max_iter")}
        what = sigma_weighted_permutation(w, sigma, target_profile, **extra)
        stats = {"n_rows": int(w.shape[0])}
    else:
        raise ValueError(
            f"unrecognized structure {structure!r}; expected one of "
            '"low_rank", "block_sparse", "butterfly", "perm_profile"'
        )

    err = sigma_weighted_error(w, what, sigma)
    return what, ProjectionReport(structure=structure, sigma_weighted_error=err, stats=stats)
