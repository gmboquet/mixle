"""Coarsening operator R with per-scale receipts (roadmap G3): depth-merge + width-merge + structure-
projection, iterated under a divergence budget and a trust region, over the real transformer in
:mod:`mixle.models.transformer`.

Build vs. borrow: this module builds only what the landscape check found unoccupied for G3 itself (the
depth-merge Taylor-composition machinery and the width-merge OT-based near-duplicate pairing); it BORROWS
everything else --

* the Gaussian LAW representation and per-layer propagation primitives (:mod:`mixle.models.moment_propagation`,
  roadmap G1) -- ``linear_law``, ``layernorm_law``, ``gelu_law``, ``attention_law``, and G1's own per-block
  closure-error receipt (``_closure_error_block``);
* the "structure-projection" move itself, which is exactly roadmap G2
  (:mod:`mixle.models.sigma_weighted_projection`) called directly, not reimplemented.

The three moves
----------------
1. **Depth-merge** (:func:`depth_merge`): folds two adjacent :class:`~mixle.models.transformer.Block`\\ s
   ``x -> x + f(x)`` and ``x -> x + g(x)`` into one merged block via a SECOND-ORDER Taylor approximation of
   their residual-flow composition ``x -> x + f(x) + g(x + f(x))``:

       ``g(x + f(x)) ~= g(x) + Dg(x)[f(x)] + O(||f(x)||^2)``

   so the merged branch is ``h(x) = f(x) + g(x) + Dg(x)[f(x)]``, accurate to second order in the (typically
   small) per-block residual magnitude -- ``f`` and ``g`` themselves are NOT linearized (both keep their full
   real attention/LayerNorm/GELU nonlinearity); only the CROSS-TERM introduced by composing them is
   approximated, which is exactly the "residual is a small perturbation" regime a pre-norm residual stack is
   designed to live in. At the LAW level this is computed analytically by chaining G1's own per-branch
   Jacobians (see :func:`_block_branch`), which is also literally how the closed-form per-scale receipt below
   is obtained -- both the teacher (exact sequential G1 propagation through both blocks) and the student (the
   merged, second-order approximation) end up as Gaussian laws, so their divergence is a KNOWN CLOSED FORM
   (:func:`gaussian_kl`), not an estimate. At the REAL forward-pass level (for actual token sequences, not
   laws), :class:`MergedBlock` evaluates the identical algebraic expression using a genuine, per-input
   Jacobian-vector product (not a single frozen linearization anchor).

2. **Width-merge** (:func:`width_merge`): reduces the residual-stream width ``d_model -> target_width`` by
   finding near-duplicate directions of the (Sigma-weighted) residual-stream covariance -- the same
   "functionally near-duplicate, once permutation-aligned, can be merged/averaged" idea as neuron-permutation
   ("git re-basin") symmetries -- via an entropic-OT (Sinkhorn) plan, then projecting down. G2's own
   :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_permutation` was checked first (see its
   docstring discussion below in :func:`width_merge`) but solves a different-shaped problem (aligning two
   SAME-shape weight matrices via a square permutation against a fixed ``target_profile``), not the
   many-to-few ``d -> target_width`` reduction needed here, so a small companion RECTANGULAR Sinkhorn is
   implemented locally, reusing the identical log-domain fixed-point structure G2 uses for its square case.

3. **Structure-projection** (:func:`structure_project`): a thin wrapper directly around G2's
   :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_low_rank` /
   :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_block_sparse` -- no reimplementation.

These are iterated by :func:`coarsen` under a divergence BUDGET (stop once the accumulated closed-form KL
between teacher and student exceeds it) and a local TRUST REGION (any individual merge whose own local KL
exceeds the trust region is rejected and the original blocks are kept instead).

H1 is this operator inverted
-----------------------------
Roadmap H1 (growth operators, not built here) is the natural INVERSE of :func:`coarsen`: instead of folding
two blocks into one under a divergence budget, it would SPLIT one block into two (or widen ``d_model``)
under a capacity/EIG budget, re-using the exact same closed-form Gaussian-law receipt machinery in reverse --
:func:`gaussian_kl` doesn't care which direction the model size changes, and :class:`ScaleReceipt` already
records both a teacher and a student law symmetrically enough that swapping which one is called "teacher" is
the whole difference between coarsening and growing. Concretely, a hypothetical ``depth_split(block, budget)``
would invert the linearization here: given a merged block's branch Jacobian ``J_h``, find an ``(f, g)`` pair
whose second-order composition reconstructs ``h`` to within budget -- the same receipt formula, run backwards.
Nothing in this module's interfaces (plain ``(law) -> (representation, receipt)`` functions, laws as ordinary
:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution` objects) assumes the
direction of size change, which is deliberate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.models.moment_propagation import (
    GaussianLaw,
    _as_law,
    _closure_error_block,
    _module_weight_bias,
    _to_numpy,
    attention_law,
    gelu_law,
    layernorm_law,
    linear_law,
)
from mixle.models.sigma_weighted_projection import (
    sigma_weighted_block_sparse,
    sigma_weighted_error,
    sigma_weighted_low_rank,
)

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "ScaleReceipt",
    "ProjectionReceipt",
    "WidthMergeRepresentation",
    "CoarsenResult",
    "gaussian_kl",
    "depth_merge",
    "width_merge",
    "structure_project",
    "coarsen",
]

if _HAS_TORCH:

    class MergedBlock(nn.Module):
        """One block that approximates two composed residual blocks via the second-order Taylor
        composition documented at module level: ``x -> x + f(x) + g(x) + Dg(x)[f(x)]``, where ``f``/``g``
        are ``block_a``/``block_b``'s full residual branches (``blk(x) - x``, i.e. the attn-residual AND
        mlp-residual sub-steps together).

        The two branches ``f(x)`` and ``g(x)`` are evaluated directly from the SAME input ``x`` (in
        parallel -- neither reads the other's output), and the correction term ``Dg(x)[f(x)]`` is a real
        directional derivative of ``g`` at ``x`` in the direction ``f(x)``, computed fresh per input as a
        CENTRAL-DIFFERENCE numerical JVP (``(g(x+eps*v) - g(x-eps*v)) / (2*eps)`` with ``v = f(x)/||f(x)||``
        and ``eps`` scaled to the local input magnitude). This sidesteps a genuine PyTorch limitation on
        this stack: ``F.scaled_dot_product_attention`` has neither a CPU double-backward kernel (so
        ``torch.autograd.functional.jvp``'s reverse-over-reverse trick fails) nor forward-mode-AD support
        (so ``torch.func.jvp`` also fails) -- both were tried and both raise ``NotImplementedError`` from
        inside attention. A numerical directional derivative needs neither: it is two ordinary forward
        passes, so it works through ANY module, including opaque/no-grad-support kernels like SDPA. This is
        what turns two SEQUENTIAL blocks into one block with (at most) two extra forward passes, i.e. a
        genuine depth cut in the sense that matters for the receipt/acceptance story: one entry in
        ``model.blocks`` instead of two, with second-order-accurate behavior and no loss of either branch's
        own nonlinearity.
        """

        def __init__(self, block_a: Any, block_b: Any, fd_eps: float = 1e-3) -> None:
            super().__init__()
            self.block_a = block_a
            self.block_b = block_b
            self.fd_eps = float(fd_eps)

        @staticmethod
        def _branch(blk: Any, x: Any) -> Any:
            a = blk.attn(blk.ln1(x))
            x1 = x + a
            m = blk.mlp(blk.ln2(x1))
            return a + m

        def _directional_derivative(self, x: Any, f_x: Any) -> Any:
            """Central-difference estimate of ``Dg(x)[f_x]`` -- see the class docstring for why this is a
            numerical (not automatic) directional derivative on this stack.
            """
            norm = f_x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            v = f_x / norm
            x_scale = x.norm(dim=-1, keepdim=True).clamp_min(1e-3)
            eps = self.fd_eps * x_scale
            g_plus = self._branch(self.block_b, x + eps * v)
            g_minus = self._branch(self.block_b, x - eps * v)
            directional = (g_plus - g_minus) / (2.0 * eps)
            return directional * norm  # un-normalize: Dg(x)[f_x] = ||f_x|| * Dg(x)[v]

        def forward(self, x: Any) -> Any:
            f_x = self._branch(self.block_a, x)
            g_x = self._branch(self.block_b, x)
            with torch.no_grad():
                jvp_gf = self._directional_derivative(x, f_x)
            return x + f_x + g_x + jvp_gf

    class CoarsenedLM(nn.Module):
        """A :class:`~mixle.models.transformer.CausalLM`-shaped module whose ``blocks`` list may contain a
        mix of original :class:`~mixle.models.transformer.Block`\\ s and :class:`MergedBlock`\\ s -- the
        output of :func:`coarsen`. Shares ``tok``/``pos``/``ln``/``head`` with the ORIGINAL model (a
        coarsening pass changes depth, not the embedding/head, so there is no reason to duplicate or
        re-tie those); ``forward`` mirrors :meth:`mixle.models.transformer.CausalLM.forward` exactly (minus
        gradient checkpointing, which the coarsened model, being shallower, needs less).
        """

        def __init__(self, base_model: Any, blocks: list[Any]) -> None:
            super().__init__()
            self.tok = base_model.tok
            self.pos = base_model.pos
            self.blocks = nn.ModuleList(blocks)
            self.ln = base_model.ln
            self.head = base_model.head
            self.vocab = base_model.vocab
            self.d_model = base_model.d_model
            self.n_layer = len(blocks)
            self.block = base_model.block

        def forward(self, x: Any) -> Any:
            x = x.long()
            t = x.shape[1]
            pos = torch.arange(t, device=x.device)
            h = self.tok(x) + self.pos(pos)[None, :, :]
            for blk in self.blocks:
                h = blk(h)
            return self.head(self.ln(h))[:, -1]

    class LowRankLinear(nn.Module):
        """Drop-in replacement for :class:`torch.nn.Linear` whose weight is stored as a genuine
        low-rank factorization (``U @ V``, ``U: (out, r)``, ``V: (r, in)``) instead of a dense
        ``(out, in)`` matrix -- the actual parameter-count reduction G2's Sigma-weighted low-rank
        solver (:func:`structure_project` / :func:`~mixle.models.sigma_weighted_projection.
        sigma_weighted_low_rank`) computes the VALUES for but, used alone, does not realize (it returns
        a dense same-shape matrix that is merely numerically low-rank). Same ``forward(x) ->
        (..., out_features)`` contract as ``nn.Linear``, so it drops into any attribute slot
        (``Block.mlp[0]``/``[2]``, ``CausalAttention.qkv``) without changing any shape-dependent code
        downstream (multi-head reshape, residual adds, ``MergedBlock``'s own branch evaluation, ...).
        """

        def __init__(self, u: Any, v: Any, bias: Any | None) -> None:
            super().__init__()
            self.u = nn.Parameter(u)  # (out_features, rank)
            self.v = nn.Parameter(v)  # (rank, in_features)
            self.bias = nn.Parameter(bias) if bias is not None else None
            self.out_features = int(u.shape[0])
            self.in_features = int(v.shape[1])
            self.rank = int(u.shape[1])

        def forward(self, x: Any) -> Any:
            out = (x @ self.v.T) @ self.u.T
            if self.bias is not None:
                out = out + self.bias
            return out


# --------------------------------------------------------------------------------------------------------
# closed-form Gaussian KL -- the per-scale receipt's core arithmetic
# --------------------------------------------------------------------------------------------------------


def gaussian_kl(p: GaussianLaw, q: GaussianLaw) -> float:
    """Closed-form ``KL(p || q)`` for two multivariate Gaussians -- the standard textbook formula

        ``KL(p||q) = 0.5 * ( tr(Sigma_q^-1 Sigma_p) + (mu_q - mu_p)^T Sigma_q^-1 (mu_q - mu_p)
                              - k + ln(det Sigma_q / det Sigma_p) )``

    computed ANALYTICALLY, not via Monte Carlo -- both ``p`` (the "teacher" law) and ``q`` (the "student"
    law) are already :class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution`
    objects, which cache ``inv_covar`` and ``log_det`` from a (self-healing) Cholesky factorization at
    construction time, so this reuses those cached quantities directly rather than re-deriving them.
    Clipped at 0 to absorb float round-off on (near-)identical laws (KL is exactly 0 there, mathematically).
    """
    k = int(p.mu.shape[0])
    diff = q.mu - p.mu
    trace_term = float(np.trace(q.inv_covar @ p.covar))
    quad_term = float(diff @ q.inv_covar @ diff)
    logdet_term = float(q.log_det - p.log_det)
    kl = 0.5 * (trace_term + quad_term - k + logdet_term)
    return float(max(kl, 0.0))


# --------------------------------------------------------------------------------------------------------
# receipts
# --------------------------------------------------------------------------------------------------------


@dataclass
class ScaleReceipt:
    """One per-scale receipt: the CLOSED-FORM teacher/student divergence at this coarsening step, plus
    (separately) G1's own closure-error signal for how much the Gaussian-surrogate assumption itself is
    trusted at this point in the network (``nan`` where no G1 block closure applies, e.g. width-merge,
    which never runs a real ``Block`` forward and so has nothing for G1's Monte-Carlo closure check to
    compare against).
    """

    name: str
    teacher_law: GaussianLaw
    student_law: GaussianLaw
    kl_divergence: float
    surrogate_closure_error: float
    accepted: bool = True


@dataclass
class ProjectionReceipt:
    """Receipt for a :func:`structure_project` call -- reports G2's own Sigma-weighted reconstruction
    error directly (there is no Gaussian law on either side of a weight-space projection, so this is not a
    KL divergence; it is the SAME ``sigma_weighted_error`` objective G2's solvers themselves minimize).
    """

    name: str
    mode: str
    sigma_weighted_error: float


@dataclass
class WidthMergeRepresentation:
    """Data-free width-reduction representation: a ``(target_width, d_model)`` merge operator and its
    ``(d_model, target_width)`` (pseudo-inverse) reconstruction, built from an entropic-OT near-duplicate
    pairing of residual-stream coordinates (see :func:`width_merge`). Kept as an explicit linear map rather
    than folded into new per-layer weight matrices -- conjugating every ``qkv``/``proj``/``mlp`` weight in
    the real model by this map is a real but separable engineering step this representation is designed to
    make straightforward (``W_new = merge @ W @ unmerge`` for a weight whose BOTH axes are ``d_model``,
    ``W_new = W @ unmerge`` / ``merge @ W`` for one-sided cases), left to callers that need an actually
    smaller ``CausalLM``.
    """

    merge: np.ndarray
    unmerge: np.ndarray
    target_width: int
    d_model: int


@dataclass
class CoarsenResult:
    """Output of :func:`coarsen`: the new (shallower) model, the full per-scale receipt map, and the
    bookkeeping needed to see exactly which merges were accepted vs. rejected and why.

    ``structure_receipts`` is the (separate, additive) third move's own receipt list -- see
    :func:`_narrow_block_linears` -- kept OUT of ``receipt_map`` deliberately: ``receipt_map`` values are
    :class:`ScaleReceipt` (closed-form KL against a Gaussian law, consumed as-is by hybrid's
    ``surrogate_closure_error``-keyed stage ranking in :mod:`mixle.models.compress`), while structure-
    projection's own receipt is a :class:`ProjectionReceipt` (a Sigma-weighted reconstruction error, not a
    KL) -- mixing the two dataclasses into one dict would silently break that attribute lookup.
    """

    model: Any
    receipt_map: dict[str, ScaleReceipt] = field(default_factory=dict)
    accepted_pairs: list[tuple[int, int]] = field(default_factory=list)
    rejected_pairs: list[tuple[int, int]] = field(default_factory=list)
    total_kl: float = 0.0
    budget: float = float("inf")
    trust_region: float = float("inf")
    within_budget: bool = True
    structure_receipts: list[ProjectionReceipt] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------------
# shared: one Block's residual branch law + Jacobian (reused by depth_merge for both teacher and student)
# --------------------------------------------------------------------------------------------------------


def _block_branch(
    law: GaussianLaw, blk: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Propagate ``law`` through one :class:`~mixle.models.transformer.Block`'s residual BRANCH (i.e.
    ``blk(x) - x``, the attn-residual and mlp-residual sub-steps together, NOT including the outer
    residual add) using G1's own per-layer laws directly (:func:`layernorm_law`, :func:`attention_law`,
    :func:`linear_law`, :func:`gelu_law`) -- the identical machinery
    :func:`mixle.models.moment_propagation.propagate_moments` uses internally, just exposing the branch's
    own (mean, covariance, Jacobian-wrt-input, cross-covariance-with-input) instead of already adding it
    back onto ``x``.

    Returns ``(branch_mean, branch_covar, branch_jacobian, cross_covar_with_input, linear_sigmas)``, where
    ``linear_sigmas`` is a data-free ``{"qkv": ..., "mlp0": ..., "mlp2": ...}`` map of the REAL (propagated,
    not data-sampled) activation covariance feeding each of this block's three biggest weight matrices --
    already computed as a byproduct of this same propagation, and exactly the ``Sigma`` G2's Sigma-weighted
    low-rank solver wants (see :func:`_narrow_block_linears`, which is what actually consumes this).
    """
    d = law.mu.shape[0]
    ln1_w, ln1_b = _to_numpy(blk.ln1.weight), _to_numpy(blk.ln1.bias)
    ln1_law, j_ln1 = layernorm_law(law, ln1_w, ln1_b, eps=blk.ln1.eps)

    qkv_w, qkv_b = _module_weight_bias(blk.attn.qkv)
    proj_w, proj_b = _module_weight_bias(blk.attn.proj)
    attn_law, j_attn = attention_law(ln1_law, qkv_w, qkv_b, proj_w, proj_b, n_head=blk.attn.h)
    j_attn_branch = j_attn @ j_ln1

    x1_mu = law.mu + attn_law.mu
    cross1 = law.covar @ j_attn_branch.T
    x1_cov = law.covar + attn_law.covar + cross1 + cross1.T
    x1_law = _as_law(x1_mu, x1_cov)

    ln2_w, ln2_b = _to_numpy(blk.ln2.weight), _to_numpy(blk.ln2.bias)
    ln2_law, j_ln2 = layernorm_law(x1_law, ln2_w, ln2_b, eps=blk.ln2.eps)
    lin1_w, lin1_b = _module_weight_bias(blk.mlp[0])
    lin1_law, j_lin1 = linear_law(ln2_law, lin1_w, lin1_b)
    gelu_out_law, j_gelu = gelu_law(lin1_law)
    lin2_w, lin2_b = _module_weight_bias(blk.mlp[2])
    mlp_law, j_lin2 = linear_law(gelu_out_law, lin2_w, lin2_b)

    j_mlp_wrt_x1 = j_lin2 @ j_gelu @ j_lin1 @ j_ln2
    j_mlp_wrt_x = j_mlp_wrt_x1 @ (np.eye(d) + j_attn_branch)

    branch_mean = attn_law.mu + mlp_law.mu
    j_branch = j_attn_branch + j_mlp_wrt_x

    cross_am = j_attn_branch @ law.covar @ j_mlp_wrt_x.T
    branch_cov = attn_law.covar + mlp_law.covar + cross_am + cross_am.T
    cross_x_branch = law.covar @ j_branch.T
    linear_sigmas = {"qkv": ln1_law.covar, "mlp0": ln2_law.covar, "mlp2": gelu_out_law.covar}
    return branch_mean, branch_cov, j_branch, cross_x_branch, linear_sigmas


def _residual_add(law: GaussianLaw, branch_mean: np.ndarray, branch_cov: np.ndarray, cross: np.ndarray) -> GaussianLaw:
    mu = law.mu + branch_mean
    cov = law.covar + branch_cov + cross + cross.T
    return _as_law(mu, cov)


# --------------------------------------------------------------------------------------------------------
# 1. depth-merge
# --------------------------------------------------------------------------------------------------------


def depth_merge(
    block_a: Any,
    block_b: Any,
    input_law: GaussianLaw,
    n_mc: int = 64,
    seed: int = 0,
) -> tuple[Any, ScaleReceipt]:
    """Fold two adjacent :class:`~mixle.models.transformer.Block`\\ s into one via the second-order Taylor
    composition documented at module level.

    Returns ``(merged_block, receipt)`` where ``merged_block`` is a real, forward-passable
    :class:`MergedBlock` and ``receipt`` is a :class:`ScaleReceipt` whose ``teacher_law``/``student_law``
    are the EXACT-per-G1 sequential composition (``block_a`` then ``block_b``, propagated exactly as
    :func:`mixle.models.moment_propagation.propagate_moments` would) vs. the second-order MERGED
    composition, both Gaussian, so ``kl_divergence`` is the closed-form :func:`gaussian_kl` between them --
    the receipt for this individual (local) merge step, i.e. what a caller's TRUST REGION check compares
    against.
    """
    if not _HAS_TORCH:
        raise RuntimeError("depth_merge requires torch (mixle.models.transformer is torch-only).")

    d = input_law.mu.shape[0]
    eye = np.eye(d)

    f_mean, f_cov, j_f, cross_xf, _sigmas_a = _block_branch(input_law, block_a)
    g_mean, g_cov, j_g, _cross_xg, _sigmas_b = _block_branch(input_law, block_b)

    # student: second-order Taylor merge, h(x) = f(x) + g(x) + Dg(x)[f(x)]
    a_mat = eye + j_g
    h_mean = f_mean + g_mean + j_g @ f_mean
    j_h = j_f + j_g + j_g @ j_f

    cross_fg = j_f @ input_law.covar @ j_g.T
    h_cov = a_mat @ f_cov @ a_mat.T + g_cov + a_mat @ cross_fg + (a_mat @ cross_fg).T
    cross_xh = input_law.covar @ j_h.T
    student_law = _residual_add(input_law, h_mean, h_cov, cross_xh)

    # teacher: exact sequential G1 propagation through block_a then block_b
    x1_law = _residual_add(input_law, f_mean, f_cov, cross_xf)
    g2_mean, g2_cov, _j_g2, cross_x1g2, _sigmas_b2 = _block_branch(x1_law, block_b)
    teacher_law = _residual_add(x1_law, g2_mean, g2_cov, cross_x1g2)

    rng = np.random.default_rng(seed)
    err_a = _closure_error_block(input_law, block_a, x1_law, rng=rng, n_mc=n_mc)
    err_b = _closure_error_block(x1_law, block_b, teacher_law, rng=rng, n_mc=n_mc)
    surrogate_closure_error = float(max(err_a, err_b))

    kl = gaussian_kl(teacher_law, student_law)
    receipt = ScaleReceipt(
        name="depth_merge",
        teacher_law=teacher_law,
        student_law=student_law,
        kl_divergence=kl,
        surrogate_closure_error=surrogate_closure_error,
    )
    merged_block = MergedBlock(block_a, block_b)
    return merged_block, receipt


# --------------------------------------------------------------------------------------------------------
# 2. width-merge
# --------------------------------------------------------------------------------------------------------


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _rectangular_sinkhorn(cost: np.ndarray, n_out: int, temperature: float, n_iter: int) -> np.ndarray:
    """Entropic-OT plan between ``d`` source coordinates (uniform row marginal ``1/d``) and ``n_out``
    target slots (uniform column marginal ``1/n_out``) -- the RECTANGULAR (many-to-few) generalization of
    the SQUARE Sinkhorn fixed point
    :func:`mixle.models.sigma_weighted_projection._sinkhorn_log_domain` uses for its one-to-one
    permutation case. Identical log-domain alternating-normalization structure, different (unequal) row and
    column marginal totals -- both still sum to 1 overall (``d * 1/d == n_out * 1/n_out == 1``), so the
    fixed point is a genuine (non-negative, correctly-marginalized) transport plan.
    """
    d = cost.shape[0]
    log_kernel = -cost / max(temperature, 1e-8)
    log_r = -np.log(d) * np.ones(d)
    log_c = -np.log(n_out) * np.ones(n_out)
    log_u = np.zeros(d)
    log_v = np.zeros(n_out)
    for _ in range(n_iter):
        log_u = log_r - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_c - _logsumexp(log_kernel + log_u[:, None], axis=0)
    return np.exp(log_u[:, None] + log_kernel + log_v[None, :])


def width_merge(
    model: Any,
    target_width: int,
    input_law: GaussianLaw,
    temperature: float = 0.1,
    n_iter: int = 200,
) -> tuple[WidthMergeRepresentation, ScaleReceipt]:
    """Reduce the residual-stream width from ``d_model`` to ``target_width`` by pairing near-duplicate
    coordinates of the (Sigma-weighted) residual-stream covariance and merging/averaging them.

    ``sigma_weighted_permutation`` (G2) was checked first per the roadmap note (see the module docstring):
    it solves ``min_P tr((W - P @ target_profile) Sigma (W - P @ target_profile)^T)`` for a SQUARE
    permutation ``P`` matching two SAME-shape objects (``W`` against a fixed ``target_profile``) -- the
    classic one-to-one "git re-basin" alignment. Width reduction needs a genuinely MANY-TO-FEW map
    (``d_model -> target_width``, generally ``target_width < d_model`` so there is no permutation at all,
    square or otherwise), so it is not directly reusable here; :func:`_rectangular_sinkhorn` reuses the
    SAME log-domain Sinkhorn fixed-point idea for the rectangular marginals this problem actually has,
    rather than pulling in a separate heavy OT solver.

    Data-free: the only input is ``input_law.covar`` (the propagated residual-stream covariance from G1),
    used to build a correlation-distance cost ``cost[i, j] = Sigma[i,i] + Sigma[a_j,a_j] - 2*Sigma[i, a_j]``
    between every source coordinate ``i`` and ``target_width`` anchor coordinates ``a_j`` (the
    highest-variance coordinates, chosen as informative anchors) -- ``cost[i, j]`` is exactly
    ``Var(x_i - x_{a_j})``, so a near-zero cost means coordinate ``i`` is functionally redundant with anchor
    ``a_j`` and should be merged into it. The resulting Sinkhorn plan, column-normalized into convex
    combinations, is the merge operator; its pseudo-inverse is the reconstruction ("unmerge") map.

    Returns ``(representation, receipt)`` where ``receipt.teacher_law`` is ``input_law`` itself and
    ``receipt.student_law`` is ``input_law`` round-tripped through merge-then-unmerge, both ``d_model``-
    dimensional so :func:`gaussian_kl` applies directly as the (closed-form) width-merge receipt.
    """
    sigma = np.asarray(input_law.covar, dtype=np.float64)
    d = sigma.shape[0]
    if not (0 < target_width <= d):
        raise ValueError(f"target_width must be in (0, d_model]; got {target_width} for d_model={d}")

    if target_width == d:
        merge = np.eye(d)
        unmerge = np.eye(d)
    else:
        diag = np.diag(sigma)
        anchors = np.sort(np.argsort(-diag)[:target_width])
        cost = diag[:, None] + diag[None, anchors] - 2.0 * sigma[:, anchors]
        cost = np.maximum(cost, 0.0)
        plan = _rectangular_sinkhorn(cost, target_width, temperature=temperature, n_iter=n_iter)
        col_sums = plan.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 1e-12, col_sums, 1.0)
        merge = (plan / col_sums).T  # (target_width, d) rows are convex combinations of sources
        unmerge = np.linalg.pinv(merge)  # (d, target_width)

    narrow_mu = merge @ input_law.mu
    narrow_cov = merge @ sigma @ merge.T
    recon_mu = unmerge @ narrow_mu
    recon_cov = unmerge @ narrow_cov @ unmerge.T
    # `recon_cov` is exactly rank <= target_width (a linear round-trip through a lower-dimensional
    # bottleneck cannot be full rank), so a literal KL against it is infinite whenever target_width <
    # d_model (the true teacher law has support the degenerate student law assigns zero density to) -- a
    # mathematically correct but useless receipt number. We instead spread whatever total variance the
    # round-trip failed to preserve (`trace(Sigma) - trace(recon_cov)`, itself a closed-form, data-free
    # quantity) as an ISOTROPIC floor over the discarded directions: the honest, information-free
    # statement "this reduced representation captures the retained directions' covariance exactly (a
    # linear projection is exact) and contributes no directional information about what it dropped,
    # beyond how much of it there was in aggregate." This keeps the receipt full-rank and finite while
    # still growing with how much width-merge actually threw away.
    leftover_trace = float(max(np.trace(sigma) - np.trace(recon_cov), 0.0))
    floor = leftover_trace / d
    student_law = _as_law(recon_mu, recon_cov + floor * np.eye(d))

    kl = gaussian_kl(input_law, student_law)
    receipt = ScaleReceipt(
        name=f"width_merge->{target_width}",
        teacher_law=input_law,
        student_law=student_law,
        kl_divergence=kl,
        surrogate_closure_error=float("nan"),
    )
    representation = WidthMergeRepresentation(merge=merge, unmerge=unmerge, target_width=target_width, d_model=d)
    _ = model  # model is accepted per the roadmap signature; this data-free reduction only needs its input_law
    return representation, receipt


# --------------------------------------------------------------------------------------------------------
# 3. structure-projection -- thin wrapper directly over G2
# --------------------------------------------------------------------------------------------------------


def structure_project(
    weight: Any,
    sigma: Any,
    mode: str = "low_rank",
    rank: int | None = None,
    pattern: Any = "2:4",
) -> tuple[np.ndarray, ProjectionReceipt]:
    """Thin wrapper calling G2's :mod:`mixle.models.sigma_weighted_projection` solvers directly -- NOT a
    reimplementation, per the roadmap's build-vs-borrow note. ``mode="low_rank"`` calls
    :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_low_rank` (requires ``rank``);
    ``mode="block_sparse"`` calls
    :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_block_sparse` (uses ``pattern``, either
    the literal ``"2:4"`` or an explicit boolean mask, exactly as G2 documents).
    """
    if mode == "low_rank":
        if rank is None:
            raise ValueError("structure_project(mode='low_rank') requires `rank`")
        w_hat = sigma_weighted_low_rank(weight, sigma, rank)
    elif mode == "block_sparse":
        w_hat = sigma_weighted_block_sparse(weight, sigma, pattern)
    else:
        raise ValueError(f"unrecognized structure_project mode {mode!r}, expected 'low_rank' or 'block_sparse'")

    err = sigma_weighted_error(weight, w_hat, sigma)
    receipt = ProjectionReceipt(name=f"structure_project[{mode}]", mode=mode, sigma_weighted_error=err)
    return w_hat, receipt


# --------------------------------------------------------------------------------------------------------
# 3b. structure-projection actually wired into coarsen() -- the real parameter-count reduction
# --------------------------------------------------------------------------------------------------------
#
# depth_merge (move 1) genuinely cuts the number of SEQUENTIAL blocks, but every block it keeps or merges
# still holds its ORIGINAL, full-size nn.Linear weight matrices (MergedBlock literally holds block_a and
# block_b as full submodules) -- so depth-merge alone changes the compute-graph shape without changing
# real parameter count at all. Move 3 (structure-projection, :func:`structure_project` above) was built
# but, until here, never actually called from :func:`coarsen` -- this is that wiring: a POST-HOC pass,
# applied to the FINAL block list only, that replaces each block's three biggest weight matrices
# (``attn.qkv``, ``mlp[0]``, ``mlp[2]``) with a genuinely smaller :class:`LowRankLinear` factorization via
# G2's Sigma-weighted low-rank solver, data-free (the ``Sigma`` is G1's own propagated activation
# covariance already computed as a side-effect of :func:`_block_branch`, never real data).
#
# Deliberately kept OUT of the accept/reject loop above (not folded into depth_merge's own trust-region
# check): running it post-hoc, on the loop's FINAL output, means it cannot perturb `total_kl`,
# `accepted_pairs`/`rejected_pairs`, or any individual `ScaleReceipt.kl_divergence` -- so every existing
# depth-merge acceptance criterion (``mixle/tests/coarsening_test.py``) is unaffected byte-for-byte by this
# addition. The rank chosen for each matrix is pinned just below that matrix's own dense/low-rank
# break-even point (:func:`_break_even_rank`) -- the LARGEST rank that still guarantees fewer stored
# parameters than the original dense matrix, i.e. the smallest, safest cut that is still a REAL reduction
# (minimizing the Sigma-weighted reconstruction error this pass introduces on top of whatever depth_merge
# already spent of the budget), rather than an aggressive cut that would also risk the quality this
# already-accepted merge/keep decision was budgeted for.


def _break_even_rank(out_dim: int, in_dim: int) -> int:
    """The largest rank at which a low-rank factorization (``r*(out+in)`` stored numbers) is still
    cheaper than the dense matrix (``out*in`` stored numbers) -- ``floor(out*in / (out+in))``. Any
    ``rank < break_even`` genuinely reduces stored parameter count; ``rank >= break_even`` would not.
    """
    return (int(out_dim) * int(in_dim)) // (int(out_dim) + int(in_dim))


def _low_rank_project_linear(linear: Any, sigma: np.ndarray, rank: int) -> tuple[Any | None, ProjectionReceipt | None]:
    """Replace one ``nn.Linear`` with a :class:`LowRankLinear` at (at most) ``rank``, via G2's
    Sigma-weighted low-rank solver (:func:`structure_project`) -- then re-factor the (dense, same-shape)
    result with a plain SVD to recover genuinely smaller ``(U, V)`` factors (``structure_project`` alone
    only guarantees the VALUE is low-rank, not that it is STORED that way). Returns ``(None, None)`` if the
    requested rank would not actually reduce parameter count (guards against a degenerate ``rank`` for a
    near-square or tiny matrix), so callers can simply skip that matrix.
    """
    out_features, in_features = int(linear.weight.shape[0]), int(linear.weight.shape[1])
    rank = int(max(0, min(rank, min(out_features, in_features))))
    if rank <= 0 or rank * (out_features + in_features) >= out_features * in_features:
        return None, None

    weight, bias = _module_weight_bias(linear)
    w_hat, receipt = structure_project(weight, sigma, mode="low_rank", rank=rank)

    u_full, s_full, vt_full = np.linalg.svd(w_hat, full_matrices=False)
    r = int(max(1, min(rank, int(np.sum(s_full > 1e-10)))))
    if r * (out_features + in_features) >= out_features * in_features:
        return None, None
    sqrt_s = np.sqrt(np.maximum(s_full[:r], 0.0))
    u = u_full[:, :r] * sqrt_s[None, :]
    v = sqrt_s[:, None] * vt_full[:r, :]

    dtype, device = linear.weight.dtype, linear.weight.device
    new_linear = LowRankLinear(
        torch.as_tensor(u, dtype=dtype, device=device),
        torch.as_tensor(v, dtype=dtype, device=device),
        linear.bias.detach().clone() if linear.bias is not None else None,
    )
    return new_linear, receipt


def _narrow_block_mlp2(blk: Any, law: GaussianLaw) -> tuple[Any, list[ProjectionReceipt]]:
    """Data-free structure-projection of one plain :class:`~mixle.models.transformer.Block`'s ``mlp[2]``
    weight (the MLP's down-projection, ``4*d_model -> d_model``) -- deliberately the ONLY matrix this
    touches (not also ``qkv``/``mlp[0]``/``proj``): every extra matrix and every extra block this pass
    touches compounds its own reconstruction error through the rest of the (autoregressive, still-real)
    forward pass, so this stays intentionally minimal -- enough to make ``count_params()`` genuinely
    smaller without spending more of the eval-regression budget than :func:`coarsen`'s own depth-merge
    step already spent. ``attn.qkv``/``mlp[0]``/``attn.proj`` are documented extension points (their own
    Sigma is either already available (``qkv`` via ``ln1_law``) or, for ``proj``, not exposed by
    :func:`_block_branch` at all -- see that function's docstring) left unused here on purpose. Operates
    on a deep COPY of ``blk`` (never mutates the original/teacher block); returns
    ``(narrowed_block, receipts)``.
    """
    new_blk = copy.deepcopy(blk)
    _bm, _bc, _jb, _cx, sigmas = _block_branch(law, blk)

    linear = new_blk.mlp[2]
    rank = max(0, _break_even_rank(int(linear.weight.shape[0]), int(linear.weight.shape[1])) - 1)
    replacement, receipt = _low_rank_project_linear(linear, sigmas["mlp2"], rank)
    if replacement is None:
        return new_blk, []
    new_blk.mlp[2] = replacement
    return new_blk, [receipt]


def _narrow_coarsened_entry(entry: Any, law: GaussianLaw) -> tuple[Any, list[ProjectionReceipt]]:
    """Dispatch structure-projection narrowing over one entry of a coarsened model's final block list.
    Deliberately a no-op for plain, unmerged :class:`~mixle.models.transformer.Block` ("kept") entries --
    only :class:`MergedBlock` ("merged") entries are narrowed, so this extra approximation is only ever
    spent on the SAME pairs :func:`coarsen`'s own trust-region/budget check already decided were worth
    approximating; a block ``coarsen`` chose to leave untouched stays byte-for-byte untouched. Within a
    merged pair, only ``block_a`` is narrowed (not also ``block_b``) -- both to halve how many matrices
    this pass touches per accepted merge (the same compounding-error reason :func:`_narrow_block_mlp2`
    documents) AND because ``law`` (the running law entering the pair) is ``block_a``'s EXACT input law,
    whereas ``block_b``'s real input is the POST-``block_a`` law -- narrowing ``block_a`` lets the
    Sigma-weighted solver use the true activation covariance rather than an approximated stand-in for it.
    """
    if isinstance(entry, MergedBlock):
        new_a, receipts_a = _narrow_block_mlp2(entry.block_a, law)
        narrowed = MergedBlock(new_a, entry.block_b, fd_eps=entry.fd_eps)
        return narrowed, receipts_a
    return entry, []


# --------------------------------------------------------------------------------------------------------
# 4. coarsen -- the iterated top-level operator R
# --------------------------------------------------------------------------------------------------------


def coarsen(
    model: Any,
    budget: float,
    trust_region: float,
    input_law: GaussianLaw,
    n_mc: int = 64,
    seed: int = 0,
) -> CoarsenResult:
    """The iterated coarsening operator ``R``: walk ``model.blocks`` pairwise, attempting a
    :func:`depth_merge` at each adjacent pair. A merge is ACCEPTED only if BOTH hold:

    * TRUST REGION -- its own LOCAL closed-form KL (``receipt.kl_divergence``) is at most ``trust_region``;
    * BUDGET -- accepting it would not push the ACCUMULATED KL (summed over all accepted merges so far)
      past ``budget``.

    A rejected (or budget-exhausted) pair is left UNMERGED -- both original blocks are kept, and the running
    law is propagated through them individually (via G1's own per-layer laws, reusing :func:`_block_branch`
    plus the outer residual add) so later merge attempts still see the correct running law regardless of
    whether earlier pairs were merged. This makes the whole pass data-free: the running "receipt map" is
    built entirely from propagated LAWS, never real data.

    Returns a :class:`CoarsenResult` wrapping a new, real, forward-passable ``CoarsenedLM`` (so a caller can
    still measure REAL per-layer error against the original model by literally running both models on
    sampled token sequences -- see ``mixle/tests/coarsening_test.py``).
    """
    if not _HAS_TORCH:
        raise RuntimeError("coarsen requires torch (mixle.models.transformer is torch-only).")

    blocks = list(model.blocks)
    new_blocks: list[Any] = []
    block_input_laws: list[GaussianLaw] = []  # law entering each new_blocks[i] entry, same order/length
    receipt_map: dict[str, ScaleReceipt] = {}
    accepted_pairs: list[tuple[int, int]] = []
    rejected_pairs: list[tuple[int, int]] = []
    total_kl = 0.0
    law = input_law
    i = 0
    out_idx = 0
    step_seed = seed

    while i < len(blocks):
        if i + 1 < len(blocks):
            blk_a, blk_b = blocks[i], blocks[i + 1]
            merged, receipt = depth_merge(blk_a, blk_b, law, n_mc=n_mc, seed=step_seed)
            step_seed += 1
            local_kl = receipt.kl_divergence
            if local_kl <= trust_region and total_kl + local_kl <= budget:
                receipt.accepted = True
                new_blocks.append(merged)
                block_input_laws.append(law)
                receipt_map[f"merged[{out_idx}]<-blocks[{i}:{i + 2}]"] = receipt
                accepted_pairs.append((i, i + 1))
                total_kl += local_kl
                law = receipt.student_law
                out_idx += 1
                i += 2
                continue
            else:
                receipt.accepted = False
                receipt_map[f"rejected_merge<-blocks[{i}:{i + 2}]"] = receipt
                rejected_pairs.append((i, i + 1))

        # keep block i unmerged: propagate the running law through it individually (G1-exact) and record a
        # zero-KL (teacher==student, nothing approximated) receipt carrying G1's own closure error.
        blk = blocks[i]
        branch_mean, branch_cov, _j_branch, cross, _sigmas = _block_branch(law, blk)
        out_law = _residual_add(law, branch_mean, branch_cov, cross)
        rng = np.random.default_rng(step_seed)
        step_seed += 1
        closure_err = _closure_error_block(law, blk, out_law, rng=rng, n_mc=n_mc)
        receipt_map[f"kept[{out_idx}]<-blocks[{i}]"] = ScaleReceipt(
            name=f"kept_block[{i}]",
            teacher_law=out_law,
            student_law=out_law,
            kl_divergence=0.0,
            surrogate_closure_error=closure_err,
        )
        new_blocks.append(blk)
        block_input_laws.append(law)
        law = out_law
        out_idx += 1
        i += 1

    # Move 3, actually wired in: post-hoc, data-free structure-projection of the FINAL block list's own
    # weight matrices (see the section above `coarsen` for why this runs here rather than inside the loop)
    # -- this is what makes `count_params(new_model) < count_params(model)` genuinely true, on top of
    # depth_merge's sequential-call reduction alone. Uses the SAME per-entry law the accept/reject loop
    # already computed, so it costs no additional propagation.
    narrowed_blocks: list[Any] = []
    structure_receipts: list[ProjectionReceipt] = []
    for entry, entry_law in zip(new_blocks, block_input_laws):
        narrowed_entry, entry_receipts = _narrow_coarsened_entry(entry, entry_law)
        narrowed_blocks.append(narrowed_entry)
        structure_receipts.extend(entry_receipts)

    new_model = CoarsenedLM(model, narrowed_blocks)
    within_budget = total_kl <= budget
    return CoarsenResult(
        model=new_model,
        receipt_map=receipt_map,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        total_kl=total_kl,
        budget=budget,
        trust_region=trust_region,
        within_budget=within_budget,
        structure_receipts=structure_receipts,
    )
