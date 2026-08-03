"""H3: structure-edit schedule during training -- the neural half of ConditionalJIT (roadmap H).

D5 (:mod:`mixle.inference.conditional_jit_controller`) built a generic learned ``ActionType``
registry and explicitly left ``STRUCTURE_EDIT`` as a documented EXTENSION POINT, not implemented
(see its module docstring and :data:`~mixle.inference.conditional_jit_controller.ACTION_TYPE_REGISTRY`).
This module is that wiring: a real action space of architecture edits --

* **grow** (H1, :mod:`mixle.experimental.growth_operators`) -- ``insert_block`` (depth); block-only
  ``widen_block`` is deliberately unavailable here until a verified whole-model width edit exists;
* **prune / depth-merge** (G3, :mod:`mixle.models.coarsening`) -- ``depth_merge``;
* **rank change** (G2, :mod:`mixle.models.sigma_weighted_projection`) -- ``sigma_weighted_low_rank``;
* **2:4 sparsity** (I4 -- no standalone I4 PR had landed when this module was built; the underlying
  2:4 projection primitive already exists as part of G2's own module,
  :func:`~mixle.models.sigma_weighted_projection.sigma_weighted_block_sparse` with
  ``pattern="2:4"``, so a snapshot (non-ramped) 2:4 projection IS wired here -- see
  :data:`STRUCTURE_EDIT_REGISTRY`'s note on ``"sparsity_2_4"`` for exactly what is and is not
  covered);
* **MoE expert add/merge** (H2 -- not landed when this module was built) -- SCAFFOLDED ONLY: the
  action-type name is registered and raises a clear ``NotImplementedError`` from
  :func:`apply_structure_edit`, per the roadmap item's explicit "optional, document don't block"
  instruction.

under one uniform interface (:func:`apply_structure_edit`), gated by an F4-style training-health
check plus a real function-preservation/output-parity check (:func:`should_apply_edit`), driven by
a :class:`StructureEditController` that extends D5's ``LearnedController``/``ActionType`` machinery
with a REAL ``STRUCTURE_EDIT`` arm (reusing D5's own :mod:`mixle.task.bandit` wiring pattern, per
that module's "reusable brain" note), and exercised end-to-end by
:func:`train_with_adaptive_structure` -- a real training loop that starts small and grows/edits
structure as training proceeds, per the round's controller decision.

Note on ``mixle.inference.conditional_jit_controller.ACTION_TYPE_REGISTRY``: that dict's own
``STRUCTURE_EDIT`` entry is left untouched here (D5's own test pins its "EXTENSION POINT" text) --
this module's :data:`STRUCTURE_EDIT_REGISTRY` is a SEPARATE, more detailed registry of the actual
edit-type strings :func:`apply_structure_edit` accepts (``"grow_insert"``, ``"grow_widen_block"``,
``"prune_depth_merge"``, ``"rank_reduce"``, ``"sparsity_2_4"``, ``"moe_expert_add"``), not a
replacement for D5's coarser action-type-level registry.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.experimental.growth_operators import (
    ParityReceipt,
    insert_block,
    verify_output_parity,
)
from mixle.inference.conditional_jit_controller import (
    ActionType,
    ControllerAction,
    LearnedController,
)
from mixle.models.coarsening import ProjectionReceipt, depth_merge
from mixle.models.moment_propagation import GaussianLaw
from mixle.models.sigma_weighted_projection import (
    sigma_weighted_block_sparse,
    sigma_weighted_error,
    sigma_weighted_low_rank,
)
from mixle.task.bandit import UCB1
from mixle.utils.parallel.training_health import TrainingHealthMonitor, flop_config_from_causal_lm

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

if _HAS_TORCH:
    from mixle.models.coarsening import CoarsenedLM

__all__ = [
    "STRUCTURE_EDIT_REGISTRY",
    "AdaptiveTrainingResult",
    "StructureEditController",
    "StructureEditReceipt",
    "StructureEditState",
    "apply_structure_edit",
    "health_report_from_monitor",
    "should_apply_edit",
    "train_with_adaptive_structure",
]


STRUCTURE_EDIT_REGISTRY: dict[str, str] = {
    "grow_insert": (
        "IMPLEMENTED. Wraps H1's mixle.experimental.growth_operators.insert_block: inserts a "
        "zero-init (exact-identity) Block into model.blocks -- depth growth, whole-model, "
        "tok/pos/head-safe."
    ),
    "grow_widen_block": (
        "UNAVAILABLE, fail-closed. H1's growth_operators.widen_block widens only one internal Block; "
        "it cannot be substituted for a whole language model. This scheduler requires every accepted "
        "edit to return a complete model, so width growth remains unavailable until embeddings, every "
        "block, normalization, tied output head, metadata, and optimizer-state migration are widened "
        "under one verified whole-model operation."
    ),
    "prune_depth_merge": (
        "IMPLEMENTED. Wraps G3's mixle.models.coarsening.depth_merge: folds two adjacent Blocks "
        "into one MergedBlock via the second-order Taylor composition, assembled into a full "
        "CoarsenedLM. NOT exact (second-order approximation, unlike the grow ops) -- expect a "
        "real, non-zero output-parity diff, which is exactly what the gate in should_apply_edit "
        "is for."
    ),
    "rank_reduce": (
        "IMPLEMENTED. Wraps G2's mixle.models.sigma_weighted_projection.sigma_weighted_low_rank: "
        "replaces one nn.Linear's weight with its Sigma-weighted rank-r projection on a deep-copied "
        "model. An approximation (rank < full rank changes the function); the real forward-pass "
        "parity receipt is what a caller's gate should judge it by."
    ),
    "sparsity_2_4": (
        "IMPLEMENTED as a snapshot (one-shot) projection, not a ramp. No standalone I4 PR had "
        "landed when this module was built, but the underlying 2:4 projection primitive already "
        "exists as part of G2's own module (sigma_weighted_block_sparse(pattern='2:4')), so this "
        "wraps that directly. A genuine I4 'ramp' (mask sparsity fraction increased gradually over "
        "several rounds) is NOT implemented here -- this action type applies the full 2:4 pattern "
        "in one edit; a future I4 item extending this to a gradual ramp would call this repeatedly "
        "with an intermediate mask."
    ),
    "moe_expert_add": (
        "SCAFFOLD ONLY, not implemented. H2 (MoE expert add/merge/upcycling) had not landed when "
        "this module was built. apply_structure_edit(..., 'moe_expert_add', ...) raises "
        "NotImplementedError with this message. Wiring it, once H2 lands, means calling H2's "
        "expert-add/merge primitive here under the same apply_structure_edit(model, edit_type, "
        "params) -> (new_model, receipt) contract every other edit type already follows."
    ),
}


# --------------------------------------------------------------------------------------------------------
# 1. apply_structure_edit -- the uniform interface over H1/G3/G2/(I4) real ops
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureEditReceipt:
    """One structure edit's receipt: which edit, the real forward-pass :class:`ParityReceipt` used by
    the function-preservation gate (see :func:`should_apply_edit`), and the edit-specific ``detail``
    object (a :class:`~mixle.experimental.growth_operators.GrowthReceipt`,
    :class:`~mixle.models.coarsening.ScaleReceipt`, or
    :class:`~mixle.models.sigma_weighted_projection.ProjectionReceipt`-shaped record, whichever the
    underlying H1/G3/G2 op returns) for anyone wanting the edit's own native receipt too.
    """

    edit_type: str
    parity: ParityReceipt | None
    detail: Any = None
    candidate_kind: str = "whole_model"


def _random_batch(model: Any, n: int = 4, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, model.vocab, size=(n, model.block))
    return torch.as_tensor(ids, device=model.tok.weight.device, dtype=torch.long)


def _require_whole_model_candidate(model_before: Any, candidate: Any) -> None:
    """Reject edit results that cannot safely replace the complete training model."""
    if not isinstance(model_before, torch.nn.Module) or not isinstance(candidate, torch.nn.Module):
        raise TypeError("structure edits must replace a torch module with a complete torch module.")
    required = ("blocks", "vocab", "block", "n_layer", "parameters")
    if any(not hasattr(candidate, name) for name in required):
        raise TypeError("structure-edit candidate is not a complete model.")
    if any(candidate is block for block in model_before.blocks):
        raise TypeError("structure-edit candidate is an internal block, not a complete model.")
    if not isinstance(candidate.blocks, torch.nn.ModuleList) or candidate.n_layer != len(candidate.blocks):
        raise TypeError("structure-edit candidate has inconsistent model/block metadata.")
    before_parameter = next(model_before.parameters(), None)
    candidate_parameter = next(candidate.parameters(), None)
    if before_parameter is None or candidate_parameter is None:
        raise TypeError("structure-edit models must contain trainable parameters.")
    if candidate_parameter.device != before_parameter.device:
        raise ValueError("structure-edit candidate must remain on the source model's device.")


def _finalize_candidate(
    model: Any,
    candidate: Any,
    edit_type: str,
    detail: Any,
    params: dict[str, Any],
    *,
    seed: int,
) -> tuple[Any, StructureEditReceipt]:
    _require_whole_model_candidate(model, candidate)
    batch = params.get("parity_batch")
    if batch is None:
        batch = _random_batch(model, seed=seed)
    tolerance = params.get("tolerance", 1e-5)
    parity = verify_output_parity(model, candidate, batch, tolerance=tolerance)
    if hasattr(detail, "parity"):
        detail.parity = parity
    return candidate, StructureEditReceipt(
        edit_type=edit_type,
        parity=parity,
        detail=detail,
        candidate_kind="whole_model",
    )


def apply_structure_edit(
    model: Any, edit_type: str, params: dict[str, Any] | None = None
) -> tuple[Any, StructureEditReceipt]:
    """Apply one structure edit to ``model`` and return ``(new_model, receipt)`` -- the uniform
    interface every ``STRUCTURE_EDIT`` action funnels through, wrapping H1/G3/G2's real ops (see
    :data:`STRUCTURE_EDIT_REGISTRY` for exactly what each ``edit_type`` does and does not cover).

    Every implemented edit type returns a complete, forward-passable model and a real forward-pass
    :class:`ParityReceipt` (computed via
    :func:`~mixle.experimental.growth_operators.verify_output_parity` on the SAME random batch, or
    ``params["parity_batch"]`` if supplied) -- the function-preservation half of
    :func:`should_apply_edit`'s gate. Block-only width growth is rejected rather than returned as if it
    were a replacement model.
    """
    if not _HAS_TORCH:
        raise RuntimeError("apply_structure_edit requires torch.")
    params = dict(params or {})

    if edit_type == "grow_insert":
        position = int(params.get("position", len(model.blocks)))
        seed = int(params.get("seed", 0))
        new_model, growth_receipt = insert_block(model, position=position, seed=seed)
        return _finalize_candidate(model, new_model, edit_type, growth_receipt, params, seed=seed)

    if edit_type == "grow_widen_block":
        raise NotImplementedError(STRUCTURE_EDIT_REGISTRY["grow_widen_block"])

    if edit_type == "prune_depth_merge":
        position = int(params.get("position", 0))
        input_law: GaussianLaw = params["input_law"]
        n_mc = int(params.get("n_mc", 64))
        seed = int(params.get("seed", 0))
        blocks = list(model.blocks)
        if position + 1 >= len(blocks):
            raise ValueError(
                f"prune_depth_merge needs an adjacent pair; position={position} is out of range for "
                f"{len(blocks)} blocks."
            )
        merged, scale_receipt = depth_merge(blocks[position], blocks[position + 1], input_law, n_mc=n_mc, seed=seed)
        new_blocks = blocks[:position] + [merged] + blocks[position + 2 :]
        new_model = CoarsenedLM(model, new_blocks)
        return _finalize_candidate(model, new_model, edit_type, scale_receipt, params, seed=seed)

    if edit_type == "rank_reduce":
        select_linear: Callable[[Any], Any] = params["select_linear"]
        sigma = params["sigma"]
        rank = int(params["rank"])
        seed = int(params.get("seed", 0))
        new_model = copy.deepcopy(model)
        linear = select_linear(new_model)
        w = linear.weight.detach().cpu().numpy().astype(np.float64)
        w_hat = sigma_weighted_low_rank(w, sigma, rank)
        err = sigma_weighted_error(w, w_hat, sigma)
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(w_hat, dtype=linear.weight.dtype, device=linear.weight.device))
        detail = ProjectionReceipt(name=f"rank_reduce[rank={rank}]", mode="low_rank", sigma_weighted_error=err)
        return _finalize_candidate(model, new_model, edit_type, detail, params, seed=seed)

    if edit_type == "sparsity_2_4":
        select_linear = params["select_linear"]
        sigma = params["sigma"]
        seed = int(params.get("seed", 0))
        new_model = copy.deepcopy(model)
        linear = select_linear(new_model)
        w = linear.weight.detach().cpu().numpy().astype(np.float64)
        w_hat = sigma_weighted_block_sparse(w, sigma, params.get("pattern", "2:4"))
        err = sigma_weighted_error(w, w_hat, sigma)
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(w_hat, dtype=linear.weight.dtype, device=linear.weight.device))
        detail = ProjectionReceipt(name="sparsity_2_4", mode="block_sparse", sigma_weighted_error=err)
        return _finalize_candidate(model, new_model, edit_type, detail, params, seed=seed)

    if edit_type == "moe_expert_add":
        raise NotImplementedError(STRUCTURE_EDIT_REGISTRY["moe_expert_add"])

    raise ValueError(f"unrecognized edit_type {edit_type!r}; expected one of {sorted(STRUCTURE_EDIT_REGISTRY)}")


# --------------------------------------------------------------------------------------------------------
# 2. gating -- F4 training-health + function-preservation, both required
# --------------------------------------------------------------------------------------------------------


def health_report_from_monitor(monitor: TrainingHealthMonitor, lookback: int = 5) -> dict[str, Any]:
    """Build the ``health_report`` :func:`should_apply_edit` expects from a real F4
    :class:`~mixle.utils.parallel.training_health.TrainingHealthMonitor`: healthy iff no anomaly was
    raised in the last ``lookback`` observed steps (an anomaly from steps ago should not permanently
    block future edits; a RECENT one -- loss spiking, NaN/Inf grads, a restart discontinuity -- should).
    """
    if not monitor.records:
        return {"healthy": True, "recent_anomalies": []}
    last_step = monitor.records[-1].step
    recent = [a.kind for a in monitor.anomalies if a.step > last_step - lookback]
    return {"healthy": len(recent) == 0, "recent_anomalies": recent}


def should_apply_edit(health_report: dict[str, Any], parity_check: ParityReceipt | None) -> bool:
    """The H3 gate: commit to a structure edit only if BOTH hold --

    (a) ``health_report`` (see :func:`health_report_from_monitor`) reports no recent F4 anomaly
        (don't structurally edit a model mid-anomaly: a loss spike, NaN/Inf grad, or restart
        discontinuity means the current state is not trustworthy to branch a structural decision
        from);
    (b) ``parity_check`` (a real :class:`ParityReceipt` from :func:`apply_structure_edit`, per H1/D6's
        established output-divergence pattern) reports the edit is within its stated tolerance.

    Otherwise the edit is skipped for this round -- the caller keeps training the UNedited model and
    may try again (a different edit, or the same one) at a later round.
    """
    healthy = bool(health_report.get("healthy", True))
    parity_ok = parity_check is not None and bool(parity_check.within_tolerance)
    return healthy and parity_ok


# --------------------------------------------------------------------------------------------------------
# 3. StructureEditController -- wires a REAL STRUCTURE_EDIT arm into D5's action space
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureEditState:
    """One round's controller-visible state for the structure-edit decision: the running loss EMA and
    its recent slope (the plateau signal), current depth, and whether F4 currently reports healthy --
    small and specific to "should I consider editing structure right now", mirroring D5's own
    ``ControllerState`` role but for this different decision.
    """

    round_index: int
    loss_ema: float
    loss_slope: float
    n_layer: int
    healthy: bool


_DEFAULT_EDIT_MOVES: tuple[dict[str, Any], ...] = (
    {"edit_type": "none"},
    {"edit_type": "grow_insert"},
)


class StructureEditController(LearnedController[StructureEditState, ControllerAction]):
    """The real ``ActionType.STRUCTURE_EDIT`` arm D5 left as an extension point (see this module's
    docstring): an online bandit -- reusing :mod:`mixle.task.bandit` exactly as D5's own
    :class:`~mixle.inference.conditional_jit_controller.BanditController` does, per that module's
    "reusable brain" note -- over a small discrete set of edit "moves" (default: ``{no_edit,
    grow_insert}``; any :func:`apply_structure_edit`-shaped ``{"edit_type": ..., **params}`` dict may
    be added). ``select_action`` returns a ``ControllerAction`` tagged
    ``ActionType.STRUCTURE_EDIT`` whose ``payload`` carries the chosen move
    (``budget_fraction`` is unused by this action type, set to ``1.0`` for interface symmetry with D5's
    other actions).

    At capacity (``state.n_layer >= max_layer``, when ``max_layer`` is set) only ``"none"`` is legal,
    so growth arms are skipped without consulting/perturbing the bandit -- a forced move never counts
    as an exploration pull.
    """

    def __init__(
        self,
        *,
        edit_moves: tuple[dict[str, Any], ...] = _DEFAULT_EDIT_MOVES,
        max_layer: int | None = None,
        ucb_c: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.edit_moves = tuple(edit_moves)
        if len(self.edit_moves) < 2:
            raise ValueError("StructureEditController needs at least two distinct edit_moves.")
        self.max_layer = max_layer
        self.bandit = UCB1(len(self.edit_moves), c=ucb_c, seed=seed)

    def select_action(self, state: StructureEditState) -> ControllerAction:
        if self.max_layer is not None and state.n_layer >= self.max_layer:
            return ControllerAction(
                action_type=ActionType.STRUCTURE_EDIT,
                budget_fraction=1.0,
                payload={"edit_type": "none", "arm": None},
            )
        arm = self.bandit.select()
        move = dict(self.edit_moves[arm])
        move["arm"] = arm
        return ControllerAction(action_type=ActionType.STRUCTURE_EDIT, budget_fraction=1.0, payload=move)

    def update(
        self, state: StructureEditState, action: ControllerAction, realized_gain: float, realized_cost: float
    ) -> None:
        arm = action.payload.get("arm")
        if arm is None:  # a forced "none" at capacity was never a real bandit pull
            return
        reward = float(realized_gain) / max(float(realized_cost), 1.0e-12)
        self.bandit.update(int(arm), reward)


# --------------------------------------------------------------------------------------------------------
# 4. train_with_adaptive_structure -- the actual adaptive training loop
# --------------------------------------------------------------------------------------------------------


@dataclass
class AdaptiveTrainingResult:
    """Output of :func:`train_with_adaptive_structure`: the final (possibly grown/edited) model, the
    REAL measured total compute (sum of F4's own ``theoretical_flops_per_iter`` over every step, at
    whatever the model's shape was AT that step -- so growth rounds correctly cost more from the
    round they take effect, not before), and the edit/health bookkeeping.
    """

    model: Any
    total_flops: float
    steps: int
    final_loss: float
    reached_target: bool
    edits_applied: list[tuple[int, str]] = field(default_factory=list)
    edits_rejected: list[tuple[int, str, str]] = field(default_factory=list)
    optimizer_state_transfers: list[tuple[int, int]] = field(default_factory=list)
    controller_outcomes: list[tuple[int, str, float, float]] = field(default_factory=list)
    health_report: dict[str, Any] = field(default_factory=dict)


def _copy_optimizer_value(value: Any, parameter: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone().to(device=parameter.device)
    return copy.deepcopy(value)


def _rebuild_optimizer_preserving_state(
    optimizer: Any,
    model_before: Any,
    model_after: Any,
) -> tuple[Any, int]:
    """Rebuild the scheduler-owned optimizer and retain state only for unchanged parameters."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("adaptive structure training currently owns and migrates a torch.optim.AdamW optimizer.")
    accepted_options = inspect.signature(torch.optim.AdamW).parameters
    optimizer_options = {
        key: value for key, value in optimizer.defaults.items() if key != "params" and key in accepted_options
    }
    new_optimizer = torch.optim.AdamW(model_after.parameters(), **optimizer_options)
    old_by_name = dict(model_before.named_parameters())
    transferred = 0
    for name, parameter in model_after.named_parameters():
        old_parameter = old_by_name.get(name)
        same_object = parameter in optimizer.state
        same_value = (
            old_parameter is not None
            and old_parameter.shape == parameter.shape
            and torch.equal(old_parameter.detach(), parameter.detach())
        )
        source = parameter if same_object else old_parameter if same_value else None
        if source is None or source not in optimizer.state or source.shape != parameter.shape:
            continue
        new_optimizer.state[parameter] = {
            key: _copy_optimizer_value(value, parameter) for key, value in optimizer.state[source].items()
        }
        transferred += 1
    return new_optimizer, transferred


def train_with_adaptive_structure(
    initial_model: Any,
    make_batch: Callable[[int, np.random.Generator], tuple[Any, Any]],
    target_loss: float,
    *,
    max_steps: int = 2000,
    max_layer: int = 3,
    batch_size: int = 64,
    lr: float = 5e-3,
    min_steps_before_edit: int = 80,
    plateau_window: int = 40,
    plateau_eps: float = 0.01,
    parity_tolerance: float = 1e-4,
    health_lookback: int = 5,
    seed: int = 0,
    controller: StructureEditController | None = None,
) -> AdaptiveTrainingResult:
    """Train ``initial_model`` (expected small) toward ``target_loss``, letting a
    :class:`StructureEditController` decide when/how to grow structure as training proceeds.

    Each step: one AdamW step on a batch from ``make_batch(batch_size, rng)`` (real cross-entropy
    loss, real backward pass), fed into a real F4
    :class:`~mixle.utils.parallel.training_health.TrainingHealthMonitor` (loss, grad-norm). A loss-EMA
    PLATEAU DETECTOR (no improvement over the last ``plateau_window``
    steps, past ``min_steps_before_edit`` steps since the last edit, and below ``max_layer``) is what
    decides WHEN to even consider a structure edit -- the controller is consulted only at plateau
    moments, mirroring how a real scheduler would not burn an edit decision every single step. When
    consulted, the controller's chosen move is applied via :func:`apply_structure_edit` and gated by
    :func:`should_apply_edit` (a real F4 health check plus the edit's own real output-parity receipt)
    before being committed -- a rejected edit is simply skipped, the run keeps training the unedited
    model, and the plateau window resets so a fresh signal is required before trying again.
    An accepted edit retains AdamW state for parameters proven unchanged. Its controller reward is
    delayed until a subsequent training interval and uses measured EMA improvement divided by only
    that interval's FLOPs; acceptance itself is never treated as a fabricated unit gain.

    Stops as soon as the loss EMA drops below ``target_loss`` (after a short warmup), or at
    ``max_steps``. Returns an :class:`AdaptiveTrainingResult` with the REAL measured total compute.
    """
    if not _HAS_TORCH:
        raise RuntimeError("train_with_adaptive_structure requires torch.")

    model = initial_model
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 100)
    monitor = TrainingHealthMonitor()
    controller = controller or StructureEditController(max_layer=max_layer, seed=seed)

    total_flops = 0.0
    ema: float | None = None
    ema_hist: list[float] = []
    steps_since_edit = 0
    edits_applied: list[tuple[int, str]] = []
    edits_rejected: list[tuple[int, str, str]] = []
    optimizer_state_transfers: list[tuple[int, int]] = []
    controller_outcomes: list[tuple[int, str, float, float]] = []
    pending_evaluation: tuple[StructureEditState, ControllerAction, str, float, float] | None = None
    last_decision_flops = 0.0
    step = 0
    reached_target = False

    def settle_pending_evaluation(current_ema: float, current_flops: float) -> None:
        nonlocal pending_evaluation, last_decision_flops
        if pending_evaluation is None:
            return
        prior_state, prior_action, prior_edit_type, baseline_ema, start_flops = pending_evaluation
        incremental_flops = current_flops - start_flops
        if incremental_flops <= 0.0:
            return
        measured_gain = max(baseline_ema - current_ema, 0.0)
        controller.update(
            prior_state,
            prior_action,
            realized_gain=measured_gain,
            realized_cost=incremental_flops,
        )
        controller_outcomes.append((prior_state.round_index, prior_edit_type, measured_gain, incremental_flops))
        last_decision_flops = current_flops
        pending_evaluation = None

    for step in range(max_steps):
        x, y = make_batch(batch_size, rng)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e9)
        opt.step()

        cfg = flop_config_from_causal_lm(model, model.block)
        total_flops += cfg.flops_per_iter(batch_size)

        loss_value = float(loss.item())
        ema = loss_value if ema is None else 0.9 * ema + 0.1 * loss_value
        ema_hist.append(ema)
        monitor.observe_step(step, loss_value, grad_norm=float(grad_norm.item()))
        steps_since_edit += 1

        if ema < target_loss and step > 30:
            settle_pending_evaluation(ema, total_flops)
            reached_target = True
            break

        evaluation_ready = (
            steps_since_edit > min_steps_before_edit
            and len(ema_hist) > plateau_window
            and (ema_hist[-1] - ema_hist[-plateau_window]) > -plateau_eps
        )
        if evaluation_ready:
            settle_pending_evaluation(ema, total_flops)
        plateaued = model.n_layer < max_layer and evaluation_ready
        if plateaued:
            state = StructureEditState(
                round_index=step,
                loss_ema=ema,
                loss_slope=ema_hist[-1] - ema_hist[-plateau_window],
                n_layer=model.n_layer,
                healthy=health_report_from_monitor(monitor, lookback=health_lookback)["healthy"],
            )
            action = controller.select_action(state)
            edit_type = action.payload.get("edit_type", "none")
            if edit_type == "none":
                # a real bandit pull, not a no-op skip: feed back a reward so this arm's pull count
                # advances and UCB1 moves on to explore the next arm next time, rather than getting
                # stuck re-selecting an unplayed "none" forever (see UCB1.select's unplayed-first rule).
                incremental_flops = total_flops - last_decision_flops
                controller.update(state, action, realized_gain=0.0, realized_cost=incremental_flops)
                controller_outcomes.append((step, edit_type, 0.0, incremental_flops))
                last_decision_flops = total_flops
            else:
                edit_params = {key: value for key, value in action.payload.items() if key not in {"arm", "edit_type"}}
                edit_params.setdefault("position", 0)
                edit_params.setdefault("seed", step)
                edit_params.setdefault("tolerance", parity_tolerance)
                edit_params.setdefault("parity_batch", x.detach())
                candidate_model, receipt = apply_structure_edit(
                    model,
                    edit_type,
                    edit_params,
                )
                health = health_report_from_monitor(monitor, lookback=health_lookback)
                if should_apply_edit(health, receipt.parity):
                    if receipt.candidate_kind != "whole_model":
                        raise TypeError("accepted structure edits must identify a complete replacement model.")
                    _require_whole_model_candidate(model, candidate_model)
                    previous_model = model
                    opt, transferred = _rebuild_optimizer_preserving_state(opt, previous_model, candidate_model)
                    model = candidate_model
                    edits_applied.append((step, edit_type))
                    optimizer_state_transfers.append((step, transferred))
                    pending_evaluation = (state, action, edit_type, float(ema), total_flops)
                    steps_since_edit = 0
                    ema_hist = []
                else:
                    reason = "unhealthy" if not health["healthy"] else "parity_out_of_tolerance"
                    edits_rejected.append((step, edit_type, reason))
                    incremental_flops = total_flops - last_decision_flops
                    controller.update(state, action, realized_gain=0.0, realized_cost=incremental_flops)
                    controller_outcomes.append((step, edit_type, 0.0, incremental_flops))
                    last_decision_flops = total_flops
                    steps_since_edit = 0  # require a fresh plateau signal before trying again

    if ema is not None:
        settle_pending_evaluation(ema, total_flops)

    return AdaptiveTrainingResult(
        model=model,
        total_flops=total_flops,
        steps=step + 1,
        final_loss=float(ema) if ema is not None else float("nan"),
        reached_target=reached_target,
        edits_applied=edits_applied,
        edits_rejected=edits_rejected,
        optimizer_state_transfers=optimizer_state_transfers,
        controller_outcomes=controller_outcomes,
        health_report=monitor.report(),
    )
