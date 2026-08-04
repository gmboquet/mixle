"""Coarsening operator R with per-scale receipts (roadmap G3): depth grouping plus structure
projection under a divergence budget and a trust region, over the real transformer in
:mod:`mixle.models.transformer`.

Build vs. borrow: this module builds only what the landscape check found unoccupied for G3 itself (the
depth-merge Taylor-composition machinery and an experimental width-representation helper); it BORROWS
everything else --

* the Gaussian LAW representation and per-layer propagation primitives (:mod:`mixle.models.moment_propagation`,
  roadmap G1) -- ``linear_law``, ``layernorm_law``, ``gelu_law``, ``attention_law``, and G1's own per-block
  closure-error receipt (``_closure_error_block``);
* the "structure-projection" move itself, which is exactly roadmap G2
  (:mod:`mixle.models.sigma_weighted_projection`) called directly, not reimplemented.

The implemented model moves and representation helper
-----------------------------------------------------
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

2. **Experimental width representation**
   (:func:`experimental_width_merge_representation`): constructs a residual-stream map
   ``d_model -> target_width`` by
   finding near-duplicate directions of the (Sigma-weighted) residual-stream covariance -- the same
   "functionally near-duplicate, once permutation-aligned, can be merged/averaged" idea as neuron-permutation
   ("git re-basin") symmetries -- via an entropic-OT (Sinkhorn) plan, then projecting down. G2's own
   :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_permutation` was checked first (see its
   docstring discussion below) but solves a different-shaped problem (aligning two
   SAME-shape weight matrices via a square permutation against a fixed ``target_profile``), not the
   many-to-few ``d -> target_width`` reduction needed here, so a small companion RECTANGULAR Sinkhorn is
   implemented locally, reusing the identical log-domain fixed-point structure G2 uses for its square case.

   This helper does not rewire a model and is not invoked by :func:`coarsen`.

3. **Structure-projection** (:func:`structure_project`): a thin wrapper directly around G2's
   :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_low_rank` /
   :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_block_sparse` -- no reimplementation.

Depth grouping and structure projection are applied by :func:`coarsen` under a divergence BUDGET and a
local TRUST REGION. The width representation is deliberately separate until normalization, attention,
embedding, and output-head rewiring can produce a genuine narrower executable model.

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
import hashlib
import warnings
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
    "MergedBlock",
    "CoarsenedLM",
    "LowRankLinear",
    "ScaleReceipt",
    "ProjectionReceipt",
    "WidthMergeRepresentation",
    "CoarseningMetrics",
    "CoarsenResult",
    "gaussian_kl",
    "depth_merge",
    "experimental_width_merge_representation",
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
        parallel -- neither reads the other's output), and the correction term ``Dg(x)[f(x)]`` is a
        directional derivative of the full tensor-valued branch ``g`` at ``x`` in the full tensor
        direction ``f(x)``. It is computed fresh per input with one scalar central-difference step:
        ``(g(x + eps*f(x)) - g(x - eps*f(x))) / (2*eps)``. This sidesteps a genuine PyTorch limitation on
        this stack: ``F.scaled_dot_product_attention`` has neither a CPU double-backward kernel (so
        ``torch.autograd.functional.jvp``'s reverse-over-reverse trick fails) nor forward-mode-AD support
        (so ``torch.func.jvp`` also fails) -- both were tried and both raise ``NotImplementedError`` from
        inside attention. A numerical directional derivative needs neither: it is two ordinary forward
        passes, so it works through ANY module, including opaque/no-grad-support kernels like SDPA. This
        groups two sequential blocks into one logical manifest entry, but it is not itself a compute or
        parameter reduction: the merged forward performs four block-branch evaluations and retains both
        component blocks. :class:`CoarseningMetrics` makes those costs explicit. The finite-difference
        evaluations remain attached to autograd: training
        differentiates the exact function used by the forward pass, including both component blocks and
        the correction term.
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
            if not np.isfinite(self.fd_eps) or self.fd_eps <= 0.0:
                raise ValueError("fd_eps must be a positive finite scalar")
            # A directional derivative uses one scalar step for the full input
            # direction. Per-token normalization/steps would perturb a
            # different direction and are invalid for attention, where every
            # output token can depend on every input token.
            rms = x.detach().double().square().mean().sqrt().clamp_min(1e-3)
            eps = torch.as_tensor(self.fd_eps, dtype=x.dtype, device=x.device) * rms.to(dtype=x.dtype)
            g_plus = self._branch(self.block_b, x + eps * f_x)
            g_minus = self._branch(self.block_b, x - eps * f_x)
            directional = (g_plus - g_minus) / (2.0 * eps)
            return directional

        def forward(self, x: Any) -> Any:
            f_x = self._branch(self.block_a, x)
            g_x = self._branch(self.block_b, x)
            jvp_gf = self._directional_derivative(x, f_x)
            return x + f_x + g_x + jvp_gf

    class CoarsenedLM(nn.Module):
        """An independently owned :class:`~mixle.models.transformer.CausalLM`-compatible module.

        ``blocks`` may contain ordinary transformer blocks or :class:`MergedBlock` entries. The
        embedding, output head, buffers, and supplied blocks are deep-copied as one ownership boundary,
        preserving weight tying inside the copy without aliasing the source model. The forward contract
        intentionally matches ``CausalLM``: global ``position_ids``, all-position logits, output scaling,
        and scalar or per-entry gradient-checkpoint controls are supported.
        """

        def __init__(self, base_model: Any, blocks: list[Any]) -> None:
            super().__init__()
            owned = copy.deepcopy(base_model)
            self.tok = owned.tok
            self.pos = owned.pos
            self.blocks = nn.ModuleList(copy.deepcopy(blocks))
            self.ln = owned.ln
            self.head = owned.head
            self.vocab = int(base_model.vocab)
            self.d_model = int(base_model.d_model)
            self.n_layer = len(blocks)
            self.n_head = int(base_model.n_head)
            self.block = int(base_model.block)
            self.gradient_checkpointing = copy.deepcopy(getattr(base_model, "gradient_checkpointing", False))
            if hasattr(owned, "mup_output_multiplier"):
                self.register_buffer(
                    "mup_output_multiplier",
                    owned.mup_output_multiplier.detach().clone(),
                    persistent=True,
                )

        def _checkpoint_block(self, index: int) -> bool:
            policy = self.gradient_checkpointing
            if isinstance(policy, bool):
                return policy
            return policy[index]

        def _validate_checkpoint_policy(self) -> None:
            policy = self.gradient_checkpointing
            if isinstance(policy, bool):
                return
            if not isinstance(policy, (list, tuple)):
                raise ValueError("gradient_checkpointing must be a bool or a per-block list/tuple of bools")
            if len(policy) != len(self.blocks):
                raise ValueError(f"gradient_checkpointing has {len(policy)} entries; expected {len(self.blocks)}")
            if any(not isinstance(value, bool) for value in policy):
                raise ValueError("every per-block gradient_checkpointing entry must be a bool")

        def _validated_ids(
            self,
            value: Any,
            *,
            name: str,
            expected_shapes: tuple[tuple[int, ...], ...],
            upper_bound: int,
        ) -> Any:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if tuple(value.shape) not in expected_shapes:
                expected = " or ".join(str(shape) for shape in expected_shapes)
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
            if value.device != self.tok.weight.device:
                raise ValueError(f"{name} is on {value.device}; model token embeddings are on {self.tok.weight.device}")
            if value.dtype == torch.bool or (not value.is_floating_point() and not value.dtype.is_signed):
                raise ValueError(f"{name} must use a signed integer or floating-point dtype")
            if value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} must contain only finite values")
                if not bool((value == torch.round(value)).all()):
                    raise ValueError(f"{name} must contain integer-valued entries")
            ids = value.long()
            if not bool(((ids >= 0) & (ids < upper_bound)).all()):
                raise ValueError(f"{name} entries must lie in [0, {upper_bound})")
            return ids

        def forward(
            self,
            x: Any,
            *,
            position_ids: Any = None,
            return_all_logits: bool = False,
        ) -> Any:
            if not isinstance(x, torch.Tensor):
                raise TypeError("x must be a torch tensor")
            if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("x must have non-empty shape (batch, sequence)")
            batch, sequence = x.shape
            if sequence > self.block:
                raise ValueError(f"x sequence length {sequence} exceeds configured block size {self.block}")
            if not isinstance(return_all_logits, bool):
                raise ValueError("return_all_logits must be a boolean")
            self._validate_checkpoint_policy()
            x = self._validated_ids(
                x,
                name="x",
                expected_shapes=((batch, sequence),),
                upper_bound=self.vocab,
            )
            if position_ids is None:
                position_ids = torch.arange(sequence, device=x.device)
            position_ids = self._validated_ids(
                position_ids,
                name="position_ids",
                expected_shapes=((sequence,), (batch, sequence)),
                upper_bound=self.block,
            )
            if position_ids.ndim == 1:
                position_embeddings = self.pos(position_ids)[None, :, :]
            else:
                position_embeddings = self.pos(position_ids)
            hidden = self.tok(x) + position_embeddings
            for index, block in enumerate(self.blocks):
                if self._checkpoint_block(index) and self.training and torch.is_grad_enabled():
                    hidden = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden,
                        use_reentrant=False,
                    )
                else:
                    hidden = block(hidden)
            logits = self.head(self.ln(hidden))
            logits = logits * getattr(self, "mup_output_multiplier", 1.0)
            return logits if return_all_logits else logits[:, -1]

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

        @property
        def weight(self) -> Any:
            """Materialized weight view for linear-module protocol consumers.

            The factors remain the only registered parameters. Moment
            propagation and subsequent structure edits can nevertheless treat
            this module as a linear map without assuming dense storage.
            """
            return self.u @ self.v

else:
    # These four are nn.Module subclasses, so they cannot be defined without torch -- but they are in
    # __all__ and mixle.models.__init__ imports them by name at module scope. Leaving them undefined
    # made `import mixle.models` fail on a torch-less install with "cannot import name 'CoarsenedLM'
    # ... Did you mean: 'CoarsenResult'?", which reads like a typo in mixle rather than a missing
    # optional dependency, and took down every torch-free surface in the package with it. Bind
    # placeholders that import cleanly and name the real problem the moment anyone touches them.
    def _requires_torch(name: str) -> type:
        class _MissingTorch:
            _mixle_requires = "torch"

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise ImportError(
                    f"mixle.models.coarsening.{name} requires torch, which is not installed. "
                    f"Install it with `pip install mixle[torch]`."
                )

        _MissingTorch.__name__ = name
        _MissingTorch.__qualname__ = name
        return _MissingTorch

    MergedBlock = _requires_torch("MergedBlock")
    CoarsenedLM = _requires_torch("CoarsenedLM")
    LowRankLinear = _requires_torch("LowRankLinear")


# --------------------------------------------------------------------------------------------------------
# closed-form Gaussian KL -- the per-scale receipt's core arithmetic
# --------------------------------------------------------------------------------------------------------


def gaussian_kl(p: GaussianLaw, q: GaussianLaw) -> float:
    """Closed-form ``KL(p || q)`` for two multivariate Gaussians -- the standard textbook formula::

        KL(p||q) = 0.5 * ( tr(Sigma_q^-1 Sigma_p) + (mu_q - mu_p)^T Sigma_q^-1 (mu_q - mu_p)
                              - k + ln(det Sigma_q / det Sigma_p) )

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
    divergence_basis: str = "full_rank_gaussian"
    regularized_kl_divergence: float | None = None
    assumptions: tuple[str, ...] = ()
    sensitivity: dict[str, float] = field(default_factory=dict)
    scope: str = "local_candidate"
    artifact_digest: str | None = None


@dataclass
class ProjectionReceipt:
    """Receipt for a :func:`structure_project` call -- reports G2's own Sigma-weighted reconstruction
    error directly (there is no Gaussian law on either side of a weight-space projection, so this is not a
    KL divergence; it is the SAME ``sigma_weighted_error`` objective G2's solvers themselves minimize).
    """

    name: str
    mode: str
    sigma_weighted_error: float
    accepted: bool = True
    final_kl_divergence: float | None = None
    artifact_digest: str | None = None


@dataclass
class WidthMergeRepresentation:
    """Data-free width-reduction representation: a ``(target_width, d_model)`` merge operator and its
    ``(d_model, target_width)`` (pseudo-inverse) reconstruction, built from an entropic-OT near-duplicate
    pairing of residual-stream coordinates (see
    :func:`experimental_width_merge_representation`). Kept as an explicit linear map rather
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


@dataclass(frozen=True)
class CoarseningMetrics:
    """Structural measurements for the source and returned executable models.

    ``estimated_block_evaluations`` counts ordinary block-branch evaluations
    in one forward: a plain block costs one and the current numerical Taylor
    merge costs four. FLOPs and latency are deliberately not fabricated
    without shapes or calibration inputs.
    """

    source_parameter_count: int
    result_parameter_count: int
    source_parameter_bytes: int
    result_parameter_bytes: int
    source_manifest_entries: int
    result_manifest_entries: int
    source_leaf_blocks: int
    result_leaf_blocks: int
    source_estimated_block_evaluations: int
    result_estimated_block_evaluations: int
    latency_seconds: float | None = None
    latency_basis: str = "unmeasured: no calibration input was supplied"


@dataclass
class CoarsenResult:
    """Output of :func:`coarsen`: the new (shallower) model, the full per-scale receipt map, and the
    bookkeeping needed to see exactly which merges were accepted vs. rejected and why.

    ``structure_receipts`` is the third move's own receipt list -- see
    :func:`_narrow_block_mlp2` -- kept OUT of ``receipt_map`` deliberately: ``receipt_map`` values are
    :class:`ScaleReceipt` (closed-form KL against a Gaussian law, consumed as-is by hybrid's
    ``surrogate_closure_error``-keyed stage ranking in :mod:`mixle.models.compress`), while structure-
    projection's own receipt is a :class:`ProjectionReceipt` (a Sigma-weighted reconstruction error, not a
    KL) -- mixing the two dataclasses into one dict would silently break that attribute lookup. The
    ``final_artifact`` scale receipt is authoritative for the returned model: its end-to-end law,
    ``total_kl``, and digest are computed after every accepted structure projection.
    """

    model: Any
    receipt_map: dict[str, ScaleReceipt] = field(default_factory=dict)
    accepted_pairs: list[tuple[int, int]] = field(default_factory=list)
    rejected_pairs: list[tuple[int, int]] = field(default_factory=list)
    depth_only_kl: float = 0.0
    total_kl: float = 0.0
    budget: float = float("inf")
    trust_region: float = float("inf")
    within_budget: bool = True
    structure_receipts: list[ProjectionReceipt] = field(default_factory=list)
    artifact_digest: str | None = None
    metrics: CoarseningMetrics | None = None


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


def experimental_width_merge_representation(
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

    This is explicitly a representation helper, not a model transformation:
    it does not rewire transformer weights, normalization, attention heads, or
    output heads and therefore does not return or claim a smaller executable
    model.

    ``receipt.kl_divergence`` reports the true support result. It is infinite
    for a genuine rank reduction because the full-rank teacher assigns mass
    outside the rank-deficient round-trip support. For callers that explicitly
    assume isotropic downstream noise, ``regularized_kl_divergence`` and the
    receipt's sensitivity values report that separate surrogate.
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
    leftover_trace = float(max(np.trace(sigma) - np.trace(recon_cov), 0.0))
    floor = leftover_trace / d
    sensitivity: dict[str, float] = {}
    assumptions: tuple[str, ...] = ()
    regularized_kl: float | None = None
    if target_width < d:
        # A rank-deficient Gaussian has no density with respect to the
        # full-dimensional teacher measure, hence KL(teacher || round-trip) is
        # infinite. Keep a regularized law only so the optional, explicitly
        # assumed downstream-noise surrogate remains inspectable.
        effective_floor = max(floor, np.finfo(np.float64).eps * max(float(np.trace(sigma)) / d, 1.0))
        student_law = _as_law(recon_mu, recon_cov + effective_floor * np.eye(d))
        regularized_kl = gaussian_kl(input_law, student_law)
        for multiplier in (0.5, 1.0, 2.0):
            assumed_law = _as_law(
                recon_mu,
                recon_cov + multiplier * effective_floor * np.eye(d),
            )
            sensitivity[f"isotropic_noise_x{multiplier:g}"] = gaussian_kl(input_law, assumed_law)
        kl = float("inf")
        basis = "singular_support"
        assumptions = (
            "regularized_kl_divergence assumes isotropic downstream noise equal to discarded trace per dimension",
        )
    else:
        student_law = input_law
        kl = 0.0
        basis = "full_rank_gaussian"

    receipt = ScaleReceipt(
        name=f"width_merge->{target_width}",
        teacher_law=input_law,
        student_law=student_law,
        kl_divergence=kl,
        surrogate_closure_error=float("nan"),
        divergence_basis=basis,
        regularized_kl_divergence=regularized_kl,
        assumptions=assumptions,
        sensitivity=sensitivity,
    )
    representation = WidthMergeRepresentation(merge=merge, unmerge=unmerge, target_width=target_width, d_model=d)
    return representation, receipt


def width_merge(
    model: Any,
    target_width: int,
    input_law: GaussianLaw,
    temperature: float = 0.1,
    n_iter: int = 200,
) -> tuple[WidthMergeRepresentation, ScaleReceipt]:
    """Compatibility wrapper for the experimental representation helper.

    ``model`` is intentionally unused: this function does not transform a
    model. New code should call
    :func:`experimental_width_merge_representation`, whose name makes that
    boundary explicit.
    """
    warnings.warn(
        "width_merge() returns only an experimental representation and does not "
        "rewire or shrink `model`; use experimental_width_merge_representation()",
        FutureWarning,
        stacklevel=2,
    )
    _ = model
    return experimental_width_merge_representation(
        target_width=target_width,
        input_law=input_law,
        temperature=temperature,
        n_iter=n_iter,
    )


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
# Depth grouping retains both full component blocks and performs more block-branch evaluations, so it is
# not described as parameter, FLOP, or latency compression. Structure projection supplies the actual
# parameter reduction by replacing a selected dense matrix with :class:`LowRankLinear`. Every proposed
# projection is propagated through the complete candidate and accepted only when the final executable
# model remains inside both the aggregate budget and trust region. The rank chosen for each matrix is
# pinned just below that matrix's own dense/low-rank
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


def _flatten_blocks(entries: list[Any]) -> list[Any]:
    """Return the ordered executable leaf blocks from a transformed manifest.

    A second coarsening pass must operate on the model it was given rather than
    assuming every manifest entry has the concrete ``Block`` layout. Flattening
    a prior ``MergedBlock`` preserves its execution order and exposes ordinary
    block-protocol leaves, including leaves whose linears have low-rank storage.
    """
    flattened: list[Any] = []
    for entry in entries:
        if isinstance(entry, MergedBlock):
            flattened.extend(_flatten_blocks([entry.block_a, entry.block_b]))
        else:
            flattened.append(entry)
    return flattened


def _propagate_entry_law(law: GaussianLaw, entry: Any) -> GaussianLaw:
    """Propagate the law through the executable function of one manifest entry."""
    if isinstance(entry, MergedBlock):
        d = law.mu.shape[0]
        eye = np.eye(d)
        f_mean, f_cov, j_f, _cross_xf, _sigmas_a = _block_branch(law, entry.block_a)
        g_mean, g_cov, j_g, _cross_xg, _sigmas_b = _block_branch(law, entry.block_b)
        a_mat = eye + j_g
        branch_mean = f_mean + g_mean + j_g @ f_mean
        branch_jacobian = j_f + j_g + j_g @ j_f
        cross_fg = j_f @ law.covar @ j_g.T
        branch_cov = a_mat @ f_cov @ a_mat.T + g_cov + a_mat @ cross_fg + (a_mat @ cross_fg).T
        cross = law.covar @ branch_jacobian.T
        return _residual_add(law, branch_mean, branch_cov, cross)
    branch_mean, branch_cov, _jacobian, cross, _sigmas = _block_branch(law, entry)
    return _residual_add(law, branch_mean, branch_cov, cross)


def _propagate_sequence_law(input_law: GaussianLaw, entries: list[Any]) -> GaussianLaw:
    law = input_law
    for entry in entries:
        law = _propagate_entry_law(law, entry)
    return law


def _entry_input_laws(input_law: GaussianLaw, entries: list[Any]) -> list[GaussianLaw]:
    laws: list[GaussianLaw] = []
    law = input_law
    for entry in entries:
        laws.append(law)
        law = _propagate_entry_law(law, entry)
    return laws


def _artifact_digest(model: Any) -> str:
    """Content digest of the returned executable state and manifest types."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    for entry in model.blocks:
        digest.update(type(entry).__module__.encode("utf-8"))
        digest.update(type(entry).__qualname__.encode("utf-8"))
    return digest.hexdigest()


def _parameter_measurements(model: Any) -> tuple[int, int]:
    parameters = list(model.parameters())
    return (
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    )


def _estimated_block_evaluations(entries: list[Any]) -> int:
    total = 0
    for entry in entries:
        if isinstance(entry, MergedBlock):
            # f(x), g(x), g(x + eps*f(x)), g(x - eps*f(x))
            total += 4
        else:
            total += 1
    return total


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
    for value, name in ((budget, "budget"), (trust_region, "trust_region")):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError(f"{name} must be a non-negative real scalar")
        if np.isnan(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a non-negative real scalar")
    budget = float(budget)
    trust_region = float(trust_region)

    blocks = _flatten_blocks(list(model.blocks))
    new_blocks: list[Any] = []
    block_input_laws: list[GaussianLaw] = []  # law entering each new_blocks[i] entry, same order/length
    receipt_map: dict[str, ScaleReceipt] = {}
    accepted_pairs: list[tuple[int, int]] = []
    # Each accepted merge, in acceptance order: (original left-block index, merged block, receipt
    # key). Kept so the end-to-end check below can back off to a PREFIX of the accepted merges
    # without recomputing any of them -- the first k merges of the pass are identical whether or not
    # later merges happen, because dropping a merge only changes the law downstream of it.
    accepted_merges: list[tuple[int, Any, str]] = []
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
                accepted_merges.append((i, merged, f"merged[{out_idx}]<-blocks[{i}:{i + 2}]"))
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

    # The local merge sum above is useful for choosing candidates, but it is
    # not an end-to-end bound. Measure the complete executable candidate against the complete
    # leaf-model law, and when its composed error violates the caller's budget, BACK OFF to the
    # largest prefix of the accepted merges that fits -- not to the identity.
    #
    # The rollback here used to be all-or-nothing, and that made the result non-monotone in
    # ``budget``: a larger budget admits a locally-greedier merge set, the larger set's composed
    # error can breach the ceiling the smaller set stayed under, and the caller then got back an
    # UNCOMPRESSED model at budget 0.3 where budget 0.1 had compressed -- observed on the checkpoint
    # family ladder, where the identical three rung ratios were dealt to different rungs on
    # different platforms purely by which side of this cliff each landed on. Asking for a looser
    # budget must never yield a strictly worse artifact than a tighter one when a feasible subset of
    # the work exists.
    #
    # Prefixes are the right backoff set: the first k accepted merges are byte-identical to the full
    # pass (a dropped merge only changes laws DOWNSTREAM of itself), so each candidate needs no new
    # Monte Carlo work -- one law propagation and one KL per step, largest prefix first. The empty
    # prefix is the original block list, whose end-to-end divergence is zero by construction, so the
    # loop always terminates in a feasible state and the old behaviour survives as its worst case.
    teacher_output_law = _propagate_sequence_law(input_law, blocks)
    depth_output_law = _propagate_sequence_law(input_law, new_blocks)
    depth_final_kl = gaussian_kl(teacher_output_law, depth_output_law)
    if depth_final_kl > budget:
        for keep in range(len(accepted_merges) - 1, -1, -1):
            kept = accepted_merges[:keep]
            merge_at = {index: merged for index, merged, _key in kept}
            prefix_blocks: list[Any] = []
            i = 0
            while i < len(blocks):
                if i in merge_at:
                    prefix_blocks.append(merge_at[i])
                    i += 2
                else:
                    prefix_blocks.append(blocks[i])
                    i += 1
            prefix_output_law = _propagate_sequence_law(input_law, prefix_blocks)
            prefix_kl = gaussian_kl(teacher_output_law, prefix_output_law) if keep else 0.0
            if prefix_kl <= budget:
                dropped = accepted_merges[keep:]
                for index, _merged, key in dropped:
                    receipt_map[key].accepted = False
                    rejected_pairs.append((index, index + 1))
                accepted_pairs = [(index, index + 1) for index, _merged, _key in kept]
                accepted_merges = kept
                new_blocks = prefix_blocks
                block_input_laws = _entry_input_laws(input_law, new_blocks)
                depth_output_law = prefix_output_law if keep else teacher_output_law
                depth_final_kl = prefix_kl
                break
    depth_only_kl = depth_final_kl

    # Budget every structure projection against the final executable model.
    # A failed proposal is rolled back individually, while later proposals may
    # still be considered against the last accepted artifact.
    final_blocks = list(new_blocks)
    structure_receipts: list[ProjectionReceipt] = []
    final_kl = depth_final_kl
    for index in range(len(final_blocks)):
        entry_laws = _entry_input_laws(input_law, final_blocks)
        entry = final_blocks[index]
        entry_law = entry_laws[index]
        narrowed_entry, entry_receipts = _narrow_coarsened_entry(entry, entry_law)
        if not entry_receipts:
            continue
        candidate_blocks = list(final_blocks)
        candidate_blocks[index] = narrowed_entry
        candidate_output_law = _propagate_sequence_law(input_law, candidate_blocks)
        candidate_kl = gaussian_kl(teacher_output_law, candidate_output_law)
        accepted = candidate_kl <= budget and abs(candidate_kl - final_kl) <= trust_region
        for receipt in entry_receipts:
            receipt.accepted = accepted
            receipt.final_kl_divergence = candidate_kl
        structure_receipts.extend(entry_receipts)
        if accepted:
            final_blocks = candidate_blocks
            depth_output_law = candidate_output_law
            final_kl = candidate_kl

    new_model = CoarsenedLM(model, final_blocks)
    artifact_digest = _artifact_digest(new_model)
    for receipt in receipt_map.values():
        receipt.artifact_digest = artifact_digest
    for receipt in structure_receipts:
        receipt.artifact_digest = artifact_digest
    receipt_map["final_artifact"] = ScaleReceipt(
        name="final_artifact",
        teacher_law=teacher_output_law,
        student_law=depth_output_law,
        kl_divergence=final_kl,
        surrogate_closure_error=float("nan"),
        accepted=final_kl <= budget,
        scope="final_artifact",
        artifact_digest=artifact_digest,
    )

    source_parameters, source_bytes = _parameter_measurements(model)
    result_parameters, result_bytes = _parameter_measurements(new_model)
    result_leaves = _flatten_blocks(list(new_model.blocks))
    metrics = CoarseningMetrics(
        source_parameter_count=source_parameters,
        result_parameter_count=result_parameters,
        source_parameter_bytes=source_bytes,
        result_parameter_bytes=result_bytes,
        source_manifest_entries=len(model.blocks),
        result_manifest_entries=len(new_model.blocks),
        source_leaf_blocks=len(blocks),
        result_leaf_blocks=len(result_leaves),
        source_estimated_block_evaluations=_estimated_block_evaluations(list(model.blocks)),
        result_estimated_block_evaluations=_estimated_block_evaluations(list(new_model.blocks)),
    )
    total_kl = final_kl
    within_budget = final_kl <= budget
    return CoarsenResult(
        model=new_model,
        receipt_map=receipt_map,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        depth_only_kl=depth_only_kl,
        total_kl=total_kl,
        budget=budget,
        trust_region=trust_region,
        within_budget=within_budget,
        structure_receipts=structure_receipts,
        artifact_digest=artifact_digest,
        metrics=metrics,
    )
