"""Growth operators (roadmap H1): net2net widening + progressive depth stacking -- G3's
:mod:`mixle.models.coarsening` run backwards, over the real transformer in
:mod:`mixle.models.transformer`.

Where G3's :func:`~mixle.models.coarsening.coarsen` FOLDS capacity down (depth-merge, width-merge,
structure-projection) under a divergence budget, H1 SPLITS capacity up, function-preservingly, so that a
rung-(k) checkpoint can be grown into a rung-(k+1) initialization and continued-trained rather than
retrained from scratch. Concretely, the two moves here are genuine inverses of G3's shrink operators:

1. **Net2Net widening** (:func:`net2net_widen`, :func:`widen_block` -- Chen, Goodfellow & Shlens, "Net2Net:
   Accelerating Learning via Knowledge Transfer", 2016): the real duplication rule. Given two consecutive
   ``nn.Linear`` layers ``L_in: R^d -> R^h`` and ``L_out: R^h -> R^o`` composed as ``L_out(act(L_in(x)))``,
   widen the hidden dimension ``h -> h'`` by choosing a mapping ``g: {0,...,h'-1} -> {0,...,h-1}`` that is
   the identity on the first ``h`` indices and, for each new index, COPIES a (systematically or randomly
   chosen) existing column ``j``. The new ``L_in`` copies row ``j`` of the old ``L_in`` (and its bias) into
   the new row -- the widened hidden unit computes literally the same pre-activation as its source, so any
   pointwise activation between the two layers is also identical on the new unit. The new ``L_out`` copies
   column ``j`` of the old ``L_out`` into the new column, and then DIVIDES every column of the new
   ``L_out`` by its post-widening replication count (``1`` for never-duplicated original columns, ``k+1``
   for a column duplicated ``k`` extra times) -- this is the step that keeps the SUM ``L_out(h)`` invariant
   despite now having ``k+1`` identical copies of that unit's activation feeding into it. This is the
   textbook Net2WiderNet rule, not an approximation: the pre-widening and post-widening compositions compute
   the exact same function on any input, up to floating-point round-off (verified below by an actual
   forward-pass receipt, per this track's stated invariant).

2. **Progressive depth stacking** (:func:`insert_block`): the direct inverse of G3's ``depth_merge`` --
   instead of folding two adjacent blocks ``x -> x + f(x)``, ``x -> x + g(x)`` into one merged block via a
   second-order Taylor approximation of ``f`` and ``g``'s composition, SPLIT capacity by inserting a brand
   new block at ``position`` whose residual branches are zero-initialized. A :class:`~mixle.models.transformer.Block`
   has TWO separate residual adds (``x = x + attn(ln1(x))``; ``x = x + mlp(ln2(x))``), so BOTH final
   linears that get summed back onto the residual stream -- attention's ``proj`` and the MLP's second
   Linear -- have their weight and bias set to exactly zero, so the new block computes ``x -> x + 0 + 0 =
   x``, an exact identity, immediately after insertion. The residual connection is what makes this trivial:
   unlike ``depth_merge``'s Taylor approximation (needed because composing two already-nonlinear branches
   has no closed form), inserting an literal no-op block needs no
   approximation at all -- the new block's output is bit-for-bit identical to its input by construction,
   so the WHOLE model's output is unchanged by insertion, exactly (not to second order).

**Function-preservation receipt.** Per the track's stated invariant ("all function-preserving edits carry
an output-parity receipt at the moment of the edit"), every growth move here is checked with a REAL
forward-pass comparison (:func:`verify_output_parity`) of the model before vs. after growth on the same
input batch -- not a closed-form law-level argument the way G3's KL receipts are. Net2Net widening and
zero-init depth stacking are ALGEBRAICALLY exact (not second-order truncations like ``depth_merge``), so
the measured max-abs/max-rel differences here are expected to sit at plain float round-off, not at some
larger truncation-controlled scale.

Symmetric naming with G3, deliberately: :func:`net2net_widen`/:func:`widen_block` undo ``width_merge``,
:func:`insert_block` undoes ``depth_merge``, and :func:`verify_output_parity` plays the same role here that
:func:`~mixle.models.coarsening.gaussian_kl` plays there -- a real, verifiable, reported number, not a
hand-wave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "ParityReceipt",
    "GrowthReceipt",
    "verify_output_parity",
    "net2net_widen",
    "widen_block",
    "insert_block",
]


@dataclass
class ParityReceipt:
    """A real, measured before/after forward-pass comparison -- the function-preservation receipt this
    track requires "at the moment of the edit". ``max_abs_diff``/``max_rel_diff`` are computed over every
    output element on the SAME input batch; ``within_tolerance`` is ``max_abs_diff <= tolerance``.
    """

    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    within_tolerance: bool
    batch_shape: tuple[int, ...]


@dataclass
class GrowthReceipt:
    """Receipt for one growth operation: its own name, the width/depth change it made, and the
    :class:`ParityReceipt` from a real forward-pass comparison -- filled in by the caller (``widen_block``,
    ``insert_block``) once the grown module/model exists, via :func:`verify_output_parity`.
    """

    name: str
    parity: ParityReceipt | None = None


def verify_output_parity(
    model_before: Any,
    model_after: Any,
    batch: Any,
    tolerance: float = 1e-5,
) -> ParityReceipt:
    """The actual before/after forward-pass comparison used by every growth operator below: run
    ``model_before`` and ``model_after`` on the SAME ``batch`` (in ``eval()`` mode, no-grad, so dropout/BN
    -- not present in :class:`~mixle.models.transformer.Block` today, but this keeps the receipt honest if
    that ever changes -- doesn't inject spurious stochastic differences) and report the real measured
    max absolute and max relative difference between their outputs.

    This is the module's one hard requirement per the track's stated invariant: "every growth operation
    must be verified via a real forward-pass comparison... showing the output is unchanged to a stated
    bitwise/numerical tolerance." Both `net2net_widen` (an exact algebraic identity) and `insert_block` (an
    exact zero-residual identity) are expected to pass at a tight tolerance close to float precision.
    """
    if not _HAS_TORCH:
        raise RuntimeError("verify_output_parity requires torch.")

    was_training_before = model_before.training
    was_training_after = model_after.training
    model_before.eval()
    model_after.eval()
    try:
        with torch.no_grad():
            out_before = model_before(batch)
            out_after = model_after(batch)
    finally:
        model_before.train(was_training_before)
        model_after.train(was_training_after)

    diff = (out_after - out_before).abs()
    max_abs_diff = float(diff.max().item())
    denom = out_before.abs().clamp_min(1e-8)
    max_rel_diff = float((diff / denom).max().item())

    return ParityReceipt(
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        tolerance=float(tolerance),
        within_tolerance=max_abs_diff <= tolerance,
        batch_shape=tuple(batch.shape),
    )


# --------------------------------------------------------------------------------------------------------
# 1. net2net widening -- the real Net2WiderNet duplication rule
# --------------------------------------------------------------------------------------------------------


def _duplication_mapping(old_width: int, new_width: int, rng: np.random.Generator, systematic: bool) -> np.ndarray:
    """Build the index mapping ``g: {0,...,new_width-1} -> {0,...,old_width-1}`` that is the identity on
    the first ``old_width`` indices and, for each of the ``new_width - old_width`` new indices, duplicates
    an existing source index. ``systematic=True`` cycles through source indices ``0, 1, 2, ...`` in order
    (deterministic, reproducible even with a bad seed); ``systematic=False`` draws source indices uniformly
    at random via ``rng`` -- Net2Net's own paper notes either choice preserves the function exactly, since
    what matters for exactness is only the DIVISION step below, not which columns get duplicated.
    """
    if new_width < old_width:
        raise ValueError(f"net2net widening requires new_width >= old_width; got {new_width} < {old_width}")
    n_new = new_width - old_width
    if n_new == 0:
        return np.arange(old_width)
    if systematic:
        extra = np.arange(n_new) % old_width
    else:
        extra = rng.integers(0, old_width, size=n_new)
    return np.concatenate([np.arange(old_width), extra])


def net2net_widen(
    linear_in: Any,
    linear_out: Any,
    new_width: int,
    seed: int = 0,
    systematic: bool = True,
) -> tuple[Any, Any, GrowthReceipt]:
    """Widen the hidden dimension between two consecutive ``nn.Linear`` layers
    ``linear_in: R^d -> R^h``, ``linear_out: R^h -> R^o`` (composed as
    ``linear_out(act(linear_in(x)))`` for any pointwise activation ``act``, or no activation at all) to
    ``new_width = h'`` via the real Net2Net duplication rule (see module docstring): a mapping ``g`` picks,
    for each new hidden unit, an existing unit to copy; ``linear_in``'s new row is copied verbatim from its
    source row (so the new unit's pre-activation, and hence any pointwise post-activation, is IDENTICAL to
    its source); ``linear_out``'s new column is copied from its source column and then every column
    (including the original, now-duplicated ones) is divided by its total replication count, so the SUM
    ``linear_out(h)`` is invariant.

    Returns ``(new_linear_in, new_linear_out, receipt)`` -- ``receipt.parity`` is left ``None`` here (no
    model to run a forward pass on at this granularity); callers needing the parity number should use
    :func:`widen_block` or :func:`verify_output_parity` directly on an assembled module.
    """
    if not _HAS_TORCH:
        raise RuntimeError("net2net_widen requires torch.")

    old_width = linear_in.out_features
    if linear_out.in_features != old_width:
        raise ValueError(
            f"linear_in.out_features ({old_width}) must equal linear_out.in_features ({linear_out.in_features})"
        )
    if new_width < old_width:
        raise ValueError(f"net2net_widen only grows width; got new_width={new_width} < old_width={old_width}")

    rng = np.random.default_rng(seed)
    mapping = _duplication_mapping(old_width, new_width, rng, systematic=systematic)
    # replication_count[j] = how many new hidden units (including the original itself) trace back to
    # source unit j -- this is exactly the divisor net2net's outgoing-weight rule requires.
    replication_count = np.bincount(mapping, minlength=old_width).astype(np.float64)

    with torch.no_grad():
        device = linear_in.weight.device
        dtype = linear_in.weight.dtype
        map_idx = torch.as_tensor(mapping, device=device, dtype=torch.long)

        d_in = linear_in.in_features
        new_linear_in = nn.Linear(d_in, new_width, bias=linear_in.bias is not None)
        new_linear_in.weight.copy_(linear_in.weight[map_idx, :])
        if linear_in.bias is not None:
            new_linear_in.bias.copy_(linear_in.bias[map_idx])
        new_linear_in.to(device=device, dtype=dtype)

        d_out = linear_out.out_features
        divisor = torch.as_tensor(replication_count[mapping], device=device, dtype=dtype)
        new_linear_out = nn.Linear(new_width, d_out, bias=linear_out.bias is not None)
        new_col = linear_out.weight[:, map_idx] / divisor[None, :]
        new_linear_out.weight.copy_(new_col)
        if linear_out.bias is not None:
            new_linear_out.bias.copy_(linear_out.bias)  # bias is unaffected: it's added once, not summed over h
        new_linear_out.to(device=device, dtype=dtype)

    receipt = GrowthReceipt(name=f"net2net_widen[{old_width}->{new_width}]")
    return new_linear_in, new_linear_out, receipt


# --------------------------------------------------------------------------------------------------------
# 2. widen_block -- net2net widening applied coherently across one whole transformer Block
# --------------------------------------------------------------------------------------------------------


def _residual_widen_mapping(old_d: int, n_head: int, r: int) -> np.ndarray:
    """Build the ``d_model``-widening index map used by :func:`widen_block`: a UNIFORM duplication (every
    source coordinate duplicated exactly ``r`` times -- required, see :func:`widen_block`'s docstring, for
    exact LayerNorm-statistic preservation) that additionally respects head boundaries (each head's own
    ``head_dim`` slice widens using the identical local pattern, so a new head-dim coordinate never traces
    back to a *different* head's source coordinate -- required for the attention correction below to be
    well-defined per head).
    """
    head_dim_old = old_d // n_head
    local_map = np.tile(np.arange(head_dim_old), r)  # length head_dim_old * r, each source repeated r times
    return np.concatenate([local_map + h * head_dim_old for h in range(n_head)])


def _widen_attention(attn: Any, new_d: int, mapping: np.ndarray, r: int) -> Any:
    """Widen a :class:`~mixle.models.transformer.CausalAttention` from ``old_d`` (implicit in ``attn``) to
    ``new_d`` using the shared, uniform, head-respecting ``mapping`` (see :func:`_residual_widen_mapping`)
    and replication factor ``r``, EXACTLY -- not merely duplicated, because both ``qkv`` and ``proj`` are
    layers whose input is the (now-duplicated) residual stream, and duplicated inputs get SUMMED by any
    matmul, so every layer reading a widened axis needs its incoming weights divided by ``r`` to keep that
    sum unchanged (the same correction :func:`net2net_widen` applies to its ``linear_out``, here needed on
    BOTH sides because ``d_model`` -- unlike net2net's classical hidden layer -- is read by many
    consecutive layers, not produced once and consumed once).

    A second, attention-specific correction is needed for exactness: duplicating ``Q`` and ``K``'s
    per-head-dim coordinates ``r``-fold (even after the ``/r`` correction above) inflates the raw
    ``Q . K`` dot product by a factor of ``r`` (summing ``r`` identical terms per original coordinate
    instead of one), and ``scaled_dot_product_attention``'s own ``1 / sqrt(head_dim)`` scaling changes too
    (``head_dim`` itself grew by ``r``) -- so the attention LOGITS would come out scaled by ``sqrt(r)``
    relative to the original (a softmax "temperature" change, not merely a constant offset it would be
    invariant to). Scaling only ``Q`` (not ``K``, not ``V``) by an EXTRA ``1 / sqrt(r)`` cancels this
    exactly: raw ``Q' . K' = r * (Q . K)`` becomes ``sqrt(r) * (Q . K)`` after the extra ``Q``-only scale,
    and dividing by ``sqrt(head_dim_new) = sqrt(r) * sqrt(head_dim_old)`` brings the logit back to EXACTLY
    the original ``(Q . K) / sqrt(head_dim_old)`` -- so softmax attention weights, and hence the whole
    attention output, are bit-identical (up to float round-off) to the pre-widening computation. ``K`` and
    ``V`` need no such correction: with unchanged softmax weights, ``V``'s plain duplication alone makes
    the weighted-sum output duplicate-consistent (``O'_i = O_{mapping[i]}``), matching every other widened
    residual-stream surface in the block.
    """
    from mixle.models.transformer import CausalAttention

    old_d = mapping.shape[0] // r
    with torch.no_grad():
        device = attn.qkv.weight.device
        dtype = attn.qkv.weight.dtype
        map_idx = torch.as_tensor(mapping, device=device, dtype=torch.long)
        sqrt_r = float(np.sqrt(r))

        new_attn = CausalAttention(new_d, attn.h)

        old_qkv_w = attn.qkv.weight.view(3, old_d, old_d)  # (qkv, out=d, in=d)
        old_qkv_b = attn.qkv.bias.view(3, old_d)
        # both axes read/duplicate the residual-stream mapping; divide by r for the widened INPUT axis sum.
        gathered_w = old_qkv_w[:, map_idx, :][:, :, map_idx] / r  # (3, new_d, new_d)
        gathered_b = old_qkv_b[:, map_idx]  # (3, new_d)
        new_qkv_w = gathered_w.clone()
        new_qkv_b = gathered_b.clone()
        new_qkv_w[0] = gathered_w[0] / sqrt_r  # Q third: extra attention-logit-scale correction
        new_qkv_b[0] = gathered_b[0] / sqrt_r
        new_attn.qkv.weight.copy_(new_qkv_w.reshape(3 * new_d, new_d))
        new_attn.qkv.bias.copy_(new_qkv_b.reshape(3 * new_d))

        # proj: input axis is the (duplicate-consistent) concatenated attention output -> divide by r;
        # output axis lands back on the residual stream -> plain row duplication (no division).
        new_proj_w = attn.proj.weight[:, map_idx][map_idx, :] / r
        new_attn.proj.weight.copy_(new_proj_w)
        new_attn.proj.bias.copy_(attn.proj.bias[map_idx])

        new_attn.to(device=device, dtype=dtype)
    return new_attn


def widen_block(block: Any, new_d_model: int, seed: int = 0, systematic: bool = True) -> tuple[Any, GrowthReceipt]:
    """Widen an entire transformer :class:`~mixle.models.transformer.Block`'s ``d_model`` from its current
    width ``old_d`` to ``new_d_model``, applying net2net-style widening COHERENTLY across every
    ``d_model``-shaped surface (both LayerNorms, attention ``qkv``/``proj``, and the MLP's two Linears) so
    the block's overall function is unchanged, not just one isolated Linear pair.

    ``new_d_model`` must be an exact integer multiple of ``old_d`` (``r = new_d_model // old_d``), and every
    source coordinate is duplicated EXACTLY ``r`` times (see :func:`_residual_widen_mapping`). This
    uniform-ratio restriction is what makes ``LayerNorm`` exact under duplication: ``LayerNorm``'s
    mean/variance are population statistics over the FULL width, and duplicating a value ``v`` a
    NON-uniform number of times changes the weighted mean/variance relative to the original -- with a
    UNIFORM ``r``-fold duplication, every source value contributes the same relative weight it always did
    (``r`` copies out of ``r * old_d`` total, same as ``1`` copy out of ``old_d``), so the widened mean and
    (population) variance are algebraically identical to the pre-widening ones, and ``ln1``/``ln2``'s
    elementwise ``weight``/``bias`` need only a plain coordinate-duplicate (no division) to match.

    Every OTHER layer that reads the widened residual stream (``qkv``, ``proj``, the MLP's first Linear)
    divides its incoming weights by ``r`` to correct for the fact that a plain matmul SUMS over its
    (now ``r``-times-duplicated) input axis -- exactly analogous to net2net's own outgoing-weight
    correction, just needed on the input side here because ``d_model`` is read repeatedly through the
    block rather than produced once. Attention gets one further correction (see :func:`_widen_attention`)
    for the ``Q . K`` dot-product scale. The MLP's OWN internal hidden width (``4 * d_model``) widens by
    ordinary (non-uniform, arbitrary) net2net duplication, exactly as in :func:`net2net_widen`, since it is
    not part of the residual stream and has no LayerNorm-style population-statistic constraint.

    Returns ``(new_block, receipt)`` where ``receipt.parity`` is filled in by an actual forward-pass
    comparison (:func:`verify_output_parity`) of ``block`` vs. ``new_block`` on the SAME ``old_d``-wide
    random batch (both blocks accept the same input width at the moment of growth).
    """
    if not _HAS_TORCH:
        raise RuntimeError("widen_block requires torch.")

    from mixle.models.transformer import Block

    old_d = block.ln1.weight.shape[0]
    if new_d_model < old_d:
        raise ValueError(f"widen_block only grows width; got new_d_model={new_d_model} < old d_model={old_d}")
    n_head = block.attn.h
    if new_d_model % n_head != 0:
        raise ValueError(f"new_d_model={new_d_model} must be divisible by n_head={n_head}")
    if new_d_model % old_d != 0:
        raise ValueError(
            f"widen_block requires new_d_model to be an exact multiple of old d_model={old_d} "
            f"(uniform duplication ratio, needed for exact LayerNorm/attention preservation); "
            f"got new_d_model={new_d_model}"
        )
    r = new_d_model // old_d

    rng = np.random.default_rng(seed)
    mapping = _residual_widen_mapping(old_d, n_head, r)

    new_block = Block(new_d_model, n_head)

    with torch.no_grad():
        device = block.ln1.weight.device
        dtype = block.ln1.weight.dtype
        map_idx = torch.as_tensor(mapping, device=device, dtype=torch.long)

        # LayerNorms: plain coordinate-duplicate weight/bias -- exact given the uniform-r duplication (see
        # docstring above for why this is algebraically exact, not an approximation).
        new_block.ln1.weight.copy_(block.ln1.weight[map_idx])
        new_block.ln1.bias.copy_(block.ln1.bias[map_idx])
        new_block.ln1.eps = block.ln1.eps
        new_block.ln2.weight.copy_(block.ln2.weight[map_idx])
        new_block.ln2.bias.copy_(block.ln2.bias[map_idx])
        new_block.ln2.eps = block.ln2.eps

        # Attention: qkv/proj widened together, sharing this block's mapping and ratio.
        new_block.attn = _widen_attention(block.attn, new_d_model, mapping, r)

        # MLP: first Linear's input axis reads the widened residual stream (/r correction); its OWN hidden
        # output axis (old_hidden -> new_hidden) widens by ordinary net2net duplication (arbitrary mapping,
        # no residual-uniformity constraint). Second Linear's input axis undoes that hidden duplication
        # (net2net's classical divide-by-replication-count rule); its output axis lands back on the
        # (widened) residual stream, so it plain-duplicates rows via `mapping` (no division).
        old_hidden = block.mlp[0].out_features
        new_hidden = 4 * new_d_model
        hidden_mapping = _duplication_mapping(old_hidden, new_hidden, rng, systematic=systematic)
        hidden_map_idx = torch.as_tensor(hidden_mapping, device=device, dtype=torch.long)
        hidden_replication = np.bincount(hidden_mapping, minlength=old_hidden).astype(np.float64)
        hidden_divisor = torch.as_tensor(hidden_replication[hidden_mapping], device=device, dtype=dtype)

        lin1 = block.mlp[0]
        new_lin1 = nn.Linear(new_d_model, new_hidden)
        new_lin1.weight.copy_(lin1.weight[hidden_map_idx, :][:, map_idx] / r)
        new_lin1.bias.copy_(lin1.bias[hidden_map_idx])

        lin2 = block.mlp[2]
        new_lin2 = nn.Linear(new_hidden, new_d_model)
        new_col = lin2.weight[:, hidden_map_idx] / hidden_divisor[None, :]
        new_lin2.weight.copy_(new_col[map_idx, :])
        new_lin2.bias.copy_(lin2.bias[map_idx])

        new_block.mlp[0] = new_lin1.to(device=device, dtype=dtype)
        new_block.mlp[2] = new_lin2.to(device=device, dtype=dtype)

        new_block.to(device=device, dtype=dtype)

    receipt = GrowthReceipt(name=f"widen_block[{old_d}->{new_d_model}]")
    batch = torch.randn(2, 5, old_d)
    receipt.parity = _verify_widen_parity(block, new_block, mapping, batch, tolerance=1e-4)
    return new_block, receipt


def _verify_widen_parity(
    block: Any, new_block: Any, mapping: np.ndarray, batch: Any, tolerance: float
) -> ParityReceipt:
    """:func:`verify_output_parity` for :func:`widen_block` specifically: ``block`` and ``new_block`` take
    DIFFERENT-width inputs (``old_d`` vs. ``new_d_model``), so a plain same-batch comparison does not apply
    directly. Instead: run ``block`` on ``batch`` (``old_d``-wide) to get ``y_old``; separately widen
    ``batch`` itself via the SAME duplication ``mapping`` (``x_new[..., k] = x_old[..., mapping[k]]`` -- a
    duplicate-consistent input, exactly the kind ``widen_block``'s construction assumes it will receive)
    and run ``new_block`` on that to get ``y_new``. Net2net widening's guarantee is that ``y_new`` is
    itself duplicate-consistent with ``y_old`` via the SAME mapping (``y_new[..., k] == y_old[...,
    mapping[k]]``), so the real forward-pass comparison is between ``y_new`` and ``y_old[..., mapping]``
    (both ``new_d_model``-wide) -- a genuine per-element check across every widened output coordinate, not
    just the pre-existing ones.
    """
    map_idx = torch.as_tensor(mapping, device=batch.device, dtype=torch.long)
    was_training_before = block.training
    was_training_after = new_block.training
    block.eval()
    new_block.eval()
    try:
        with torch.no_grad():
            y_old = block(batch)
            y_new = new_block(batch[..., map_idx])
    finally:
        block.train(was_training_before)
        new_block.train(was_training_after)

    expected = y_old[..., map_idx]
    diff = (y_new - expected).abs()
    max_abs_diff = float(diff.max().item())
    denom = expected.abs().clamp_min(1e-8)
    max_rel_diff = float((diff / denom).max().item())
    return ParityReceipt(
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        tolerance=float(tolerance),
        within_tolerance=max_abs_diff <= tolerance,
        batch_shape=tuple(batch.shape),
    )


# --------------------------------------------------------------------------------------------------------
# 3. insert_block -- progressive depth stacking, the direct inverse of G3's depth_merge
# --------------------------------------------------------------------------------------------------------


def insert_block(model: Any, position: int, seed: int = 0) -> tuple[Any, GrowthReceipt]:
    """Progressive depth stacking: insert a brand-new, near-identity
    :class:`~mixle.models.transformer.Block` into ``model.blocks`` at index ``position`` (``0 <= position
    <= len(model.blocks)``), function-preservingly, by ZERO-INITIALIZING BOTH of the new block's residual
    branches -- attention's ``proj`` (the last thing summed onto the residual stream in
    ``x + self.attn(self.ln1(x))``) AND the MLP's second Linear (the last thing summed on in
    ``x + self.mlp(self.ln2(x))``), weight and bias both -- so the new block computes
    ``x -> (x + 0) + 0 = x``, an EXACT identity (not a second-order approximation the way G3's
    ``depth_merge`` needs, since there is no composition of two nonlinear branches to linearize here -- a
    literal no-op block needs no approximation at all). Zeroing only one of the two branches is NOT
    sufficient: a :class:`~mixle.models.transformer.Block` has two separate residual adds, and either one
    left non-zero would perturb ``x`` before the block's output is reached.

    This is the direct inverse move of ``depth_merge``: where that operator folds two adjacent blocks into
    one via a Taylor-approximated composition, this operator splits capacity by inserting one new block
    that starts as a no-op and is left for the optimizer to grow into useful capacity during continued
    training -- exactly the roadmap's "rung-(k) checkpoints initialize rung-(k+1) function-preservingly"
    role.

    Returns ``(new_model, receipt)`` -- a fresh :class:`~mixle.models.transformer.CausalLM`-shaped module
    (built by shallow-copying the input model's embeddings/head and inserting the new block into a fresh
    ``blocks`` list, mirroring :class:`~mixle.models.coarsening.CoarsenedLM`'s own approach on the shrink
    side) whose ``forward`` is identical to ``model``'s at the moment of insertion, verified by an actual
    forward pass on real token ids via :func:`verify_output_parity`.
    """
    if not _HAS_TORCH:
        raise RuntimeError("insert_block requires torch.")

    from mixle.models.transformer import Block

    n_blocks = len(model.blocks)
    if not (0 <= position <= n_blocks):
        raise ValueError(f"position must be in [0, {n_blocks}]; got {position}")

    d_model = model.d_model
    n_head = model.blocks[0].attn.h if n_blocks > 0 else model.n_head

    new_block = Block(d_model, n_head)
    with torch.no_grad():
        # A Block has TWO separate residual adds -- x = x + attn(ln1(x)); x = x + mlp(ln2(x)) -- so BOTH
        # branches' final linear (the thing actually summed back onto x) must be zeroed for the whole
        # block to be an exact identity; zeroing only one branch still leaves the other perturbing x.
        new_block.attn.proj.weight.zero_()
        new_block.attn.proj.bias.zero_()
        new_block.mlp[2].weight.zero_()
        new_block.mlp[2].bias.zero_()
    new_block.to(device=model.tok.weight.device, dtype=model.tok.weight.dtype)

    blocks = list(model.blocks)
    blocks.insert(position, new_block)

    new_model = _StackedLM(model, blocks)

    receipt = GrowthReceipt(name=f"insert_block[pos={position}]")
    rng = np.random.default_rng(seed)
    block_size = model.block
    vocab = model.vocab
    ids = rng.integers(0, vocab, size=(2, min(5, block_size)))
    batch = torch.as_tensor(ids, device=model.tok.weight.device, dtype=torch.float32)
    receipt.parity = verify_output_parity(model, new_model, batch, tolerance=1e-5)
    return new_model, receipt


if _HAS_TORCH:

    class _StackedLM(nn.Module):
        """A :class:`~mixle.models.transformer.CausalLM`-shaped module produced by :func:`insert_block`:
        shares ``tok``/``pos``/``ln``/``head`` with the ORIGINAL model (depth stacking changes depth, not
        the embedding/head), with ``blocks`` replaced by the caller-supplied list that includes the new
        near-identity block. Mirrors :class:`~mixle.models.coarsening.CoarsenedLM`'s structure on the grow
        side, and :meth:`forward` matches
        :meth:`mixle.models.transformer.CausalLM.forward` exactly (minus gradient checkpointing).
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
            self.n_head = base_model.n_head
            self.block = base_model.block

        def forward(self, x: Any) -> Any:
            x = x.long()
            t = x.shape[1]
            pos = torch.arange(t, device=x.device)
            h = self.tok(x) + self.pos(pos)[None, :, :]
            for blk in self.blocks:
                h = blk(h)
            return self.head(self.ln(h))[:, -1]
