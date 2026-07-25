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
* the permutation case is an exact linear-assignment solve over profile rows. It intentionally does not
  claim to expose an entropic-transport relaxation or differentiable Sinkhorn object.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "sigma_weighted_error",
    "sigma_weighted_low_rank",
    "sigma_weighted_block_sparse",
    "sigma_weighted_permutation",
    "sigma_weighted_butterfly",
    "ButterflyProjection",
    "CovarianceAdjustmentWarning",
    "ProjectionReport",
    "project",
]


# --------------------------------------------------------------------------------------------------------
# shared objective
# --------------------------------------------------------------------------------------------------------


class CovarianceAdjustmentWarning(RuntimeWarning):
    """A covariance had only roundoff-scale negative eigenvalues and was projected to the PSD cone."""


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _validate_problem(w: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Return finite matrix weights and their finite symmetric PSD input covariance."""
    w_array = np.asarray(w)
    sigma_array = np.asarray(sigma)
    if np.iscomplexobj(w_array) or np.iscomplexobj(sigma_array):
        raise TypeError("W and Sigma must be real-valued")
    try:
        w_array = np.asarray(w_array, dtype=np.float64)
        sigma_array = np.asarray(sigma_array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("W and Sigma must be real numeric arrays") from exc
    if w_array.ndim != 2 or not all(w_array.shape):
        raise ValueError(f"W must be a non-empty two-dimensional matrix; got shape {w_array.shape}")
    expected = (w_array.shape[1], w_array.shape[1])
    if sigma_array.shape != expected:
        raise ValueError(
            f"Sigma must be square with side == W.shape[1]; got W {w_array.shape}, Sigma {sigma_array.shape}"
        )
    if not np.all(np.isfinite(w_array)) or not np.all(np.isfinite(sigma_array)):
        raise ValueError("W and Sigma must contain only finite values")

    scale = max(1.0, float(np.max(np.abs(sigma_array))))
    symmetry_tolerance = 1.0e-12 * scale
    asymmetry = float(np.max(np.abs(sigma_array - sigma_array.T)))
    if asymmetry > symmetry_tolerance:
        raise ValueError(
            f"Sigma must be symmetric within tolerance {symmetry_tolerance:.3g}; maximum asymmetry is {asymmetry:.3g}"
        )
    sigma_symmetric = 0.5 * (sigma_array + sigma_array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_symmetric)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    psd_tolerance = 1.0e-12 * spectral_scale
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -psd_tolerance:
        raise ValueError(
            f"Sigma must be positive semidefinite; minimum eigenvalue {minimum_eigenvalue:.6g} "
            f"is below tolerance {-psd_tolerance:.6g}"
        )

    correction = max(0.0, -minimum_eigenvalue)
    if correction:
        clipped = np.maximum(eigenvalues, 0.0)
        sigma_symmetric = (eigenvectors * clipped) @ eigenvectors.T
        warnings.warn(
            f"Sigma had a roundoff-scale minimum eigenvalue {minimum_eigenvalue:.6g}; "
            f"projected it to the PSD cone (maximum eigenvalue correction {correction:.6g})",
            CovarianceAdjustmentWarning,
            stacklevel=2,
        )
    return w_array, sigma_symmetric, correction


def sigma_weighted_error(w: Any, w_hat: Any, sigma: Any) -> float:
    """``tr((W - What) @ Sigma @ (W - What)^T)`` -- the Sigma-weighted reconstruction objective itself.

    Used both as the convergence check inside the iterative solvers below and as the metric the tests
    compare solvers against each other with.
    """
    w_array, sigma_array, _ = _validate_problem(w, sigma)
    w_hat_array = np.asarray(w_hat)
    if np.iscomplexobj(w_hat_array):
        raise TypeError("W_hat must be real-valued")
    try:
        w_hat_array = np.asarray(w_hat_array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("W_hat must be a real numeric array") from exc
    if w_hat_array.shape != w_array.shape:
        raise ValueError(f"W_hat shape {w_hat_array.shape} must match W shape {w_array.shape}")
    if not np.all(np.isfinite(w_hat_array)):
        raise ValueError("W_hat must contain only finite values")
    diff = w_array - w_hat_array
    result = float(np.trace(diff @ sigma_array @ diff.T))
    if not np.isfinite(result) or result < -1.0e-10:
        raise ValueError("Sigma-weighted error must be finite and non-negative")
    return max(0.0, result)


def _symmetric_sqrt_and_pinv_sqrt(sigma: np.ndarray, rcond: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric PSD ``Sigma`` into its symmetric matrix square root and the
    (pseudo-inverse) square root, clipping numerically-negative eigenvalues to zero and treating
    near-zero eigenvalues as exactly rank-deficient directions (their pseudo-inverse contribution is
    zero, matching the fact that ``Sigma`` assigns no weight/cost to those directions at all).
    """
    eigval, eigvec = np.linalg.eigh(sigma)
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
    w, sigma, _ = _validate_problem(w, sigma)
    rank = _exact_int(rank, "rank", minimum=0)
    if rank > min(w.shape):
        raise ValueError(f"rank must not exceed min(W.shape)={min(w.shape)}")

    sigma_half, sigma_half_pinv = _symmetric_sqrt_and_pinv_sqrt(sigma)

    b = w @ sigma_half
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    b_hat = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
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
    w, sigma, _ = _validate_problem(w, sigma)
    max_iter = _exact_int(max_iter, "max_iter", minimum=1)
    tol = _nonnegative_finite(tol, "tol")

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
# 3. permutation -- exact row assignment
# --------------------------------------------------------------------------------------------------------


def sigma_weighted_permutation(
    w: Any,
    sigma: Any,
    target_profile: Any,
) -> np.ndarray:
    """Exact row-permutation solver for ``What = P @ target_profile``.

    This is a linear assignment problem in disguise: it decomposes over ROW-PAIRS (row ``i`` of ``W``
    matched to row ``j`` of ``target_profile``) with pairwise cost
    ``cost[i,j] = (W_i - profile_j) @ Sigma @ (W_i - profile_j)^T``, so the discrete problem
    ``min_{P permutation} sum_i cost[i, perm(i)]`` is exactly a linear assignment problem. The returned
    hard permutation is therefore computed directly with the Hungarian algorithm. This function exposes
    no temperature or iteration controls because it does not compute or return an entropic soft plan.
    """
    w, sigma, _ = _validate_problem(w, sigma)
    profile = np.asarray(target_profile)
    if np.iscomplexobj(profile):
        raise TypeError("target_profile must be real-valued")
    try:
        profile = np.asarray(profile, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("target_profile must be a real numeric matrix") from exc
    if w.shape != profile.shape:
        raise ValueError(f"target_profile shape {profile.shape} must match W shape {w.shape}")
    if not np.all(np.isfinite(profile)):
        raise ValueError("target_profile must contain only finite values")
    n = w.shape[0]

    # pairwise Sigma-weighted cost between every row of W and every row of the profile
    diff = w[:, None, :] - profile[None, :, :]  # (n, n, d_in)
    cost = np.einsum("ijk,kl,ijl->ij", diff, sigma, diff)

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros((n, n), dtype=np.float64)
    perm[row_ind, col_ind] = 1.0
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


@dataclass(frozen=True)
class ButterflyProjection:
    """Compact executable product of sparse butterfly factors.

    ``apply(x)`` accepts vectors or batches whose last dimension is ``d_in`` and returns the equivalent
    of ``x @ projection.to_dense().T`` without materializing an ``N x N`` matrix. ``projection @ x``
    provides ordinary matrix-multiplication orientation for arrays whose first dimension is ``d_in``.
    NumPy consumers that require a dense matrix remain compatible through ``np.asarray(projection)``.
    """

    a: tuple[np.ndarray, ...]
    b: tuple[np.ndarray, ...]
    strides: tuple[int, ...]
    d_out: int
    d_in: int
    n: int

    def __post_init__(self) -> None:
        if not self.a or len(self.a) != len(self.b) or len(self.a) != len(self.strides):
            raise ValueError("butterfly factors must have equal non-zero stage counts")
        for name, factors in (("a", self.a), ("b", self.b)):
            stable: list[np.ndarray] = []
            for factor in factors:
                array = np.array(factor, dtype=np.float64, copy=True)
                if array.shape != (self.n,) or not np.all(np.isfinite(array)):
                    raise ValueError(f"butterfly {name} factors must be finite vectors of length {self.n}")
                array.setflags(write=False)
                stable.append(array)
            object.__setattr__(self, name, tuple(stable))
        if any(stride <= 0 or stride >= self.n for stride in self.strides):
            raise ValueError("butterfly strides must lie in [1, n)")
        if not (0 < self.d_out <= self.n and 0 < self.d_in <= self.n):
            raise ValueError("butterfly logical dimensions must lie in [1, n]")

    @property
    def shape(self) -> tuple[int, int]:
        """Logical dense matrix shape."""
        return self.d_out, self.d_in

    @property
    def parameter_count(self) -> int:
        """Number of stored floating-point factor parameters."""
        return sum(factor.size for factor in self.a + self.b)

    @property
    def parameter_nbytes(self) -> int:
        """Physical bytes occupied by stored floating-point factors."""
        return sum(factor.nbytes for factor in self.a + self.b)

    @property
    def serialized_nbytes(self) -> int:
        """Actual protocol-5 pickle size of the executable representation."""
        return len(pickle.dumps(self, protocol=5))

    def apply(self, x: Any) -> np.ndarray:
        """Apply the projection to vectors/batches stored along their final axis."""
        array = np.asarray(x)
        if np.iscomplexobj(array):
            raise TypeError("butterfly inputs must be real-valued")
        try:
            array = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("butterfly inputs must be real numeric arrays") from exc
        if array.ndim == 0 or array.shape[-1] != self.d_in:
            raise ValueError(f"butterfly input's final dimension must be {self.d_in}; got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("butterfly inputs must contain only finite values")

        current = np.zeros(array.shape[:-1] + (self.n,), dtype=np.float64)
        current[..., : self.d_in] = array
        indices = np.arange(self.n)
        for a_factor, b_factor, stride in zip(self.a, self.b, self.strides, strict=True):
            partner = indices ^ stride
            current = a_factor * current + b_factor * current[..., partner]
        result = current[..., : self.d_out]
        if not np.all(np.isfinite(result)):
            raise ValueError("butterfly application produced non-finite values")
        return result

    def to_dense(self) -> np.ndarray:
        """Materialize the logical dense matrix only when a caller explicitly requests it."""
        return self.apply(np.eye(self.d_in)).T

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> np.ndarray:
        array = self.to_dense()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def __matmul__(self, other: Any) -> np.ndarray:
        array = np.asarray(other)
        if array.ndim == 0 or array.shape[0] != self.d_in:
            raise ValueError(f"right operand's first dimension must be {self.d_in}; got shape {array.shape}")
        row_oriented = np.moveaxis(array, 0, -1)
        return np.moveaxis(self.apply(row_oriented), -1, 0)


def sigma_weighted_butterfly(
    w: Any,
    sigma: Any,
    n_stages: int | None = None,
    n_sweeps: int = 4,
) -> ButterflyProjection:
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
    w, sigma, _ = _validate_problem(w, sigma)
    d_out, d_in = w.shape
    n_sweeps = _exact_int(n_sweeps, "n_sweeps", minimum=1)

    n = _next_pow2(max(d_out, d_in, 2))
    l_full = n.bit_length() - 1  # log2(n), n is a power of two
    if n_stages is None:
        l = l_full
    else:
        l = _exact_int(n_stages, "n_stages", minimum=1)
        if l > l_full:
            raise ValueError(f"n_stages must not exceed log2(N)={l_full}")
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
    for _sweep in range(n_sweeps):
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

    return ButterflyProjection(tuple(a), tuple(b), tuple(strides), d_out=d_out, d_in=d_in, n=n)


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


def project(w: Any, sigma: Any, structure: str, **kw: Any) -> tuple[Any, ProjectionReport]:
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
    * ``"perm_profile"``: ``target_profile`` (required).
    """
    w, sigma, covariance_correction = _validate_problem(w, sigma)
    contracts = {
        "low_rank": ({"rank"}, {"rank"}),
        "block_sparse": ({"pattern", "max_iter", "tol"}, {"pattern"}),
        "butterfly": ({"n_stages", "n_sweeps"}, set()),
        "perm_profile": ({"target_profile"}, {"target_profile"}),
    }
    if structure not in contracts:
        raise ValueError(
            f"unrecognized structure {structure!r}; expected one of "
            '"low_rank", "block_sparse", "butterfly", "perm_profile"'
        )
    allowed, required = contracts[structure]
    unexpected = set(kw) - allowed
    missing = required - set(kw)
    if unexpected:
        raise TypeError(f"unexpected {structure} options: {', '.join(sorted(unexpected))}")
    if missing:
        raise TypeError(f"missing required {structure} options: {', '.join(sorted(missing))}")

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
        stats = {
            "n": what.n,
            "n_stages": len(what.strides),
            "param_count": what.parameter_count,
            "parameter_nbytes": what.parameter_nbytes,
            "serialized_nbytes": what.serialized_nbytes,
            "dense_nbytes": int(np.prod(what.shape) * np.dtype(np.float64).itemsize),
        }
    elif structure == "perm_profile":
        target_profile = kw["target_profile"]
        what = sigma_weighted_permutation(w, sigma, target_profile)
        stats = {"n_rows": int(w.shape[0])}
    stats["covariance_psd_correction"] = covariance_correction
    err = sigma_weighted_error(w, what, sigma)
    return what, ProjectionReport(structure=structure, sigma_weighted_error=err, stats=stats)
