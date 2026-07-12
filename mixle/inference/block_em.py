"""Block-coordinate-ascent EM scheduler -- greedy gain-per-cost block selection (workstream D3).

The estimator tree is an update IR. After an initial full sweep, this scheduler ranks mixture
components by the complete-data ``Q`` improvement measured the last time that component was
updated, divided by a structural parameter-count cost. It updates the highest-value blocks within
a per-round budget and bounds starvation with a forced periodic update.

Correctness backbone (unchanged from the rest of the D-track): this module is a SCHEDULING
optimization only. Any interleaving of a partial E-step (fresh log-density for the active
blocks, cached log-density for the inactive ones) and a per-block conditional M-step (only the
active blocks are re-estimated; every other block's model object is carried forward byte-for-
byte unchanged) is a generalized-EM coordinate update. A nonnegative complete-data Q gain
certifies that the observed objective cannot decrease. Periodic and final exact observed-
likelihood audits verify that certificate; a candidate without a finite certificate falls back
to an exact observed-objective gate.

The scheduler reuses :class:`~mixle.inference.freeze_rollup.FreezeRollupCache` only for blocks
left inactive in the current round. It does not permanently freeze a component from a heuristic
residual; ``max_skip_rounds`` guarantees every nonzero-weight block is revisited. Callers wanting
the explicit near-zero-weight permanent-freeze policy can use ``run_em_freeze_rollup`` directly.

The measured gain is tied to the real E-step responsibilities: for active component ``k`` it is
``sum_i gamma_ik [log p_new(x_i,k) - log p_old(x_i,k)]``, including the weight-coordinate term.
It is therefore a stale scheduling estimate, not a convergence proof. Correctness comes from the
Q certificate and observed-objective audits; ranking only changes work order.

Degeneration to vanilla full-tree EM: when every eligible block's gain-per-cost score is the
same (within ``tie_tol``) -- either because the blocks truly are indistinguishable, or because
the caller passes ``full_tree_every_round=True`` as an explicit escape hatch -- the scheduler has
no real choice to make, so it updates every eligible block, exactly like a plain full-tree EM
round would. See ``mixle.tests.block_em_test`` for a literal test of this property.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.conditional_jit_controller import (
    BanditController,
    ControllerAction,
    ControllerState,
    DesignModelController,
    LearnedController,
)
from mixle.inference.freeze_rollup import (
    DensityMatrixProfile,
    FreezeRollupCache,
    _combine,
    _component_log_density_matrix_profiled,
    _log_density_from_matrix,
    _m_step,
    _resolve_payload,
    _updated_component_log_density_columns_profiled,
    _updated_component_log_density_matrix_profiled,
)
from mixle.inference.transaction import MutableStateSnapshot, has_mutable_state
from mixle.stats.latent.mixture import MixtureDistribution, MixtureEstimator, _component_enc

_DEFAULT_ACCEPT_TOLERANCE = 1.0e-9
_DEFAULT_BUDGET_FRACTION = 0.5
_DEFAULT_TIE_TOL = 1.0e-9
_DEFAULT_BOUNDARY_WEIGHT_STEP = 1.0 - float(np.sqrt(np.finfo(np.float64).eps))
_KNOWN_POLICIES = ("greedy", "learned_bandit", "learned_design_model")


@dataclass(frozen=True)
class BlockEMAssumptionReceipt:
    """Observed checks for assumptions connecting selected work to saved work."""

    declared_budget_fraction: float
    eligible_cost: float
    selected_cost: float
    forced_cost: float
    expected_density_evaluations: int
    observed_density_evaluations: int
    expected_score_reuses: int
    observed_score_reuses: int
    schedule_reused: bool = False
    full_refresh: bool = False
    cost_basis: str = "structural_parameter_count"
    expected_responsibility_columns: int = 0
    observed_responsibility_columns: int = 0
    incremental_normalizer: bool = False
    normalizer_audit_max_abs_error: float = 0.0
    normalizer_logspace_rows: int = 0
    failures: tuple[str, ...] = ()

    @property
    def budget_utilization(self) -> float:
        budget = self.declared_budget_fraction * self.eligible_cost
        return self.selected_cost / budget if budget > 0.0 else 0.0

    @property
    def assumptions_hold(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class BlockEMTimingReceipt:
    """Wall-time decomposition for one scheduler round."""

    scheduling_seconds: float
    estep_density_seconds: float
    responsibility_seconds: float
    snapshot_seconds: float
    mstep_seconds: float
    candidate_density_seconds: float
    validation_seconds: float
    accounting_seconds: float
    total_seconds: float
    component_density_seconds: tuple[tuple[str, int, float], ...] = ()

    @property
    def density_seconds(self) -> float:
        return self.estep_density_seconds + self.candidate_density_seconds

    @property
    def orchestration_seconds(self) -> float:
        modeled = self.density_seconds + self.responsibility_seconds + self.mstep_seconds
        return max(0.0, self.total_seconds - modeled)


@dataclass
class BlockEMStats:
    """One round's accounting for the block-EM scheduler -- the acceptance-criteria receipt.

    ``n_log_density_evals`` records removed model work, while ``wall_time_seconds`` records actual
    elapsed time. ``objective`` is either the exact observed-data likelihood or its certified EM
    lower bound, as disclosed by ``objective_exact``; ``measured_q_gain`` is the complete-data
    gain used for the next scheduling decision.

    ``acceptance_basis == "vanilla_escape"`` marks a round where repeated rejections triggered one
    unconditional full-tree EM step (see ``escape_after_rejections``). ``stop_reason`` is set on
    the FINAL entry when the run terminated early; ``"rejection_livelock"`` means the scheduler
    kept producing rejected proposals even after a vanilla escape had been spent.
    """

    round_index: int
    n_components: int
    n_active: int
    n_frozen: int
    n_zero_weight: int
    n_dormant: int
    n_scheduled_inactive: int
    n_log_density_evals: int
    n_responsibility_columns: int
    objective: float
    accepted: bool = True
    acceptance_basis: str = "observed"
    objective_exact: bool = True
    degenerate_round: bool = False
    measured_q_gain: float = 0.0
    certified_q_gain: float = 0.0
    wall_time_seconds: float = 0.0
    final_audit_seconds: float = 0.0
    assumptions: BlockEMAssumptionReceipt | None = None
    timing: BlockEMTimingReceipt | None = None
    stop_reason: str | None = None

    @property
    def active_fraction(self) -> float:
        """Fraction of components genuinely (re-)evaluated this round, out of all components."""
        return self.n_active / self.n_components if self.n_components else 1.0


def _block_scores(
    model: MixtureDistribution,
    eligible: list[int],
    last_q_gain: dict[int, float],
    measured_cost: dict[int, float] | None = None,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float], str]:
    """Return last measured data-Q gain per cost unit for eligible blocks.

    Cost is the MEASURED per-component density seconds from earlier rounds' timing receipts when
    every eligible block has one (``cost_model="measured"``); otherwise the structural parameter
    count proxy. Mixed units are never compared: measured pricing engages all-or-nothing.
    """
    gain_per_cost: dict[int, float] = {}
    cost: dict[int, float] = {}
    residual: dict[int, float] = {}
    q_gain: dict[int, float] = {}
    use_measured = measured_cost is not None and all(measured_cost.get(idx, 0.0) > 0.0 for idx in eligible)
    basis = "measured_seconds" if use_measured and eligible else "structural_parameter_count"
    for idx in eligible:
        gain = max(float(last_q_gain.get(idx, 0.0)), 0.0)
        if use_measured:
            block_cost = float(measured_cost[idx])
        else:
            block_cost = float(max(_parameter_count(model.components[idx]), 1))
        cost[idx] = block_cost
        gain_per_cost[idx] = gain / block_cost
        residual[idx] = gain
        q_gain[idx] = gain
    return gain_per_cost, cost, residual, q_gain, basis


def _parameter_count(root: Any) -> int:
    """Count numeric parameters recursively without sampling or scoring a component."""
    seen: set[int] = set()

    def count(value: Any) -> int:
        if value is None or isinstance(value, (str, bytes, bytearray, bool)):
            return 0
        if isinstance(value, (int, float, np.integer, np.floating)):
            return 1
        ident = id(value)
        if ident in seen:
            return 0
        seen.add(ident)
        if isinstance(value, np.ndarray):
            return int(value.size) if np.issubdtype(value.dtype, np.number) else 0
        parameters = getattr(value, "parameters", None)
        if callable(parameters):
            return sum(int(p.numel()) for p in parameters())
        if isinstance(value, dict):
            return sum(count(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(child) for child in value)
        if hasattr(value, "__dict__"):
            return sum(count(child) for name, child in vars(value).items() if not name.startswith("_"))
        return 0

    return max(count(root), 1)


def _active_responsibilities(
    ll_mat: np.ndarray,
    log_w: np.ndarray,
    log_density: np.ndarray,
    active: tuple[int, ...],
) -> np.ndarray:
    """Compute exact posterior columns only for active mixture components."""

    if ll_mat.ndim != 2 or ll_mat.shape[0] != len(log_density):
        raise ValueError("component scores and row normalizers do not align.")
    if not active:
        return np.empty((ll_mat.shape[0], 0), dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        responsibilities = np.exp(ll_mat[:, active] + log_w[list(active)] - log_density[:, None])
    if not np.isfinite(responsibilities).all():
        raise FloatingPointError("sparse responsibilities require finite, possible observation rows.")
    return responsibilities


def _pseudo_weight_counts(estimator: MixtureEstimator) -> np.ndarray:
    """Return additive pseudo-counts matching ``_mixture_weights`` semantics."""

    if estimator.pseudo_count is None:
        return np.zeros(estimator.num_components, dtype=np.float64)
    if estimator.suff_stat is None:
        return np.full(
            estimator.num_components,
            float(estimator.pseudo_count) / estimator.num_components,
            dtype=np.float64,
        )
    prior = np.asarray(estimator.suff_stat, dtype=np.float64)
    if prior.shape != (estimator.num_components,):
        raise ValueError("mixture weight sufficient statistic has the wrong shape.")
    return prior * float(estimator.pseudo_count)


def _constrained_block_weights(
    estimator: MixtureEstimator,
    model: MixtureDistribution,
    active: tuple[int, ...],
    active_counts: np.ndarray,
    nobs: int,
    boundary_step: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Maximize mixture-weight Q over active weights plus inactive total mass.

    Individual inactive-weight ratios are held fixed. Their common scale is one
    additional coordinate, so the current weights are feasible and this remains
    a genuine conditional maximization. Only active counts and
    ``nobs - sum(active_counts)`` are needed.
    """

    if estimator.has_conj_prior and estimator.fixed_weights is None:
        raise NotImplementedError("sparse block weights do not support conjugate weight priors.")
    if estimator.w_min > 0.0 and estimator.fixed_weights is None:
        raise NotImplementedError("sparse block weights do not support a positive weight floor.")
    if len(active) != len(active_counts):
        raise ValueError("active component indices and responsibility counts do not align.")
    if not 0.0 < boundary_step <= 1.0:
        raise ValueError("boundary_step must be in (0, 1].")
    if estimator.fixed_weights is not None:
        return np.asarray(estimator.fixed_weights, dtype=np.float64).copy(), 1.0

    active_array = np.asarray(active, dtype=int)
    inactive_array = np.asarray([idx for idx in range(model.num_components) if idx not in active], dtype=int)
    pseudo = _pseudo_weight_counts(estimator)
    active_augmented = np.asarray(active_counts, dtype=np.float64) + pseudo[active_array]
    inactive_count = max(0.0, float(nobs) - float(np.sum(active_counts)))
    inactive_augmented = inactive_count + float(np.sum(pseudo[inactive_array]))
    total = float(np.sum(active_augmented)) + inactive_augmented
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("sparse block weight update has no finite mass.")

    weights = np.zeros(model.num_components, dtype=np.float64)
    weights[active_array] = active_augmented / total
    if inactive_array.size == 0:
        return weights, 0.0
    old_inactive_mass = float(np.sum(model.w[inactive_array]))
    new_inactive_mass = inactive_augmented / total
    if old_inactive_mass <= 0.0:
        if new_inactive_mass > 1.0e-15:
            raise FloatingPointError("zero-weight inactive aggregate acquired unexplained responsibility mass.")
        return weights, 0.0
    inactive_scale = new_inactive_mass / old_inactive_mass
    weights[inactive_array] = model.w[inactive_array] * inactive_scale
    creates_boundary = np.any((model.w > 0.0) & (weights == 0.0))
    if creates_boundary and boundary_step < 1.0:
        weights = (1.0 - boundary_step) * model.w + boundary_step * weights
        inactive_scale = (1.0 - boundary_step) + boundary_step * inactive_scale
    return weights, inactive_scale


def _sparse_block_m_step(
    enc_data: Any,
    estimator: MixtureEstimator,
    model: MixtureDistribution,
    active: tuple[int, ...],
    responsibilities: np.ndarray,
    boundary_weight_step: float,
) -> tuple[MixtureDistribution, np.ndarray, float]:
    """Re-estimate active leaves and the active-plus-inactive-mass weight block."""

    counts = responsibilities.sum(axis=0)
    components = list(model.components)
    for position, idx in enumerate(active):
        if model.zw[idx] or counts[position] <= 0.0:
            continue
        enc_i = _component_enc(enc_data, idx)
        accumulator = estimator.estimators[idx].accumulator_factory().make()
        accumulator.seq_update(enc_i, responsibilities[:, position], model.components[idx])
        components[idx] = estimator.estimators[idx].estimate(float(counts[position]), accumulator.value())
    weights, inactive_scale = _constrained_block_weights(
        estimator,
        model,
        active,
        counts,
        responsibilities.shape[0],
        boundary_step=boundary_weight_step,
    )
    return MixtureDistribution(components, weights, name=estimator.name), counts, inactive_scale


def _incremental_candidate_log_density(
    current_log_density: np.ndarray,
    old_active_responsibilities: np.ndarray,
    old_inactive_scores: np.ndarray,
    old_inactive_log_w: np.ndarray,
    new_active_scores: np.ndarray,
    new_active_log_w: np.ndarray,
    inactive_scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Replace active weighted terms in an exact row normalizer in ``O(N * active)``."""

    old_inactive_mass = np.maximum(0.0, 1.0 - old_active_responsibilities.sum(axis=1))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        new_active_mass = np.exp(new_active_scores + new_active_log_w - current_log_density[:, None]).sum(axis=1)
    relative_mass = inactive_scale * old_inactive_mass + new_active_mass
    impossible = ~np.isfinite(relative_mass) | (relative_mass <= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate = current_log_density + np.log(relative_mass)
    cancellation_rows = old_inactive_mass <= np.sqrt(np.finfo(np.float64).eps)
    if np.any(cancellation_rows):
        inactive_values = (
            old_inactive_scores[cancellation_rows] + old_inactive_log_w
            if old_inactive_scores.shape[1]
            else np.empty((int(np.count_nonzero(cancellation_rows)), 0), dtype=np.float64)
        )
        if inactive_values.shape[1]:
            old_inactive_log = np.logaddexp.reduce(inactive_values, axis=1)
            new_inactive_log = (
                old_inactive_log + np.log(inactive_scale)
                if inactive_scale > 0.0
                else np.full_like(old_inactive_log, -np.inf)
            )
        else:
            new_inactive_log = np.full(int(np.count_nonzero(cancellation_rows)), -np.inf)
        new_active_log = np.logaddexp.reduce(
            new_active_scores[cancellation_rows] + new_active_log_w,
            axis=1,
        )
        candidate[cancellation_rows] = np.logaddexp(new_inactive_log, new_active_log)
        impossible[cancellation_rows] = ~np.isfinite(candidate[cancellation_rows])
    if np.any(impossible):
        candidate = candidate.copy()
        candidate[impossible] = 0.0
    return candidate, impossible, int(np.count_nonzero(cancellation_rows))


def _select_active(
    eligible: list[int],
    scores: dict[int, float],
    cost: dict[int, float],
    *,
    budget_fraction: float,
    full_tree_every_round: bool,
    tie_tol: float,
    forced: set[int] | None = None,
) -> tuple[set[int], bool]:
    """Return ``(active_indices, degenerate)`` -- the block-selection decision for this round.

    ``degenerate`` is True whenever the scheduler had no real choice to make (see module
    docstring): ``full_tree_every_round`` was requested, there is nothing eligible to schedule,
    or every eligible score is indistinguishable within ``tie_tol``. In every such case the whole
    eligible set is selected -- the literal "reduces to vanilla full-tree EM" behavior.
    """
    if not eligible:
        return set(), True
    if full_tree_every_round:
        return set(eligible), True

    values = [scores[idx] for idx in eligible]
    spread = max(values) - min(values)
    if spread < tie_tol:
        return set(eligible), True

    forced = set() if forced is None else set(forced) & set(eligible)
    ranked_forced = sorted(forced, key=lambda i: (-scores[i], i))
    ranked_optional = sorted((idx for idx in eligible if idx not in forced), key=lambda i: (-scores[i], i))
    total_cost = sum(cost[idx] for idx in eligible)
    budget = max(budget_fraction, 0.0) * total_cost

    # Starvation prevention is part of the same budget decision. The old path
    # selected a budgeted set and then unioned overdue blocks on top, silently
    # turning a 50% budget into roughly 66% work on the reference fixture.
    active = set(ranked_forced)
    spent = sum(cost[idx] for idx in ranked_forced)
    for idx in ranked_optional:
        if active and spent >= budget:
            break
        if active and spent + cost[idx] > budget:
            continue
        active.add(idx)
        spent += cost[idx]
    return active, False


def is_block_em_eligible(model: Any, estimator: Any) -> bool:
    """Whether ``model``/``estimator`` are a plain ``MixtureDistribution``/``MixtureEstimator`` pair
    the D3 scheduler can drive. Lives here (not in estimation.py) so the high-level estimation driver
    never imports a concrete distribution type -- see compute_metadata_test.py's layering check."""
    return isinstance(model, MixtureDistribution) and isinstance(estimator, MixtureEstimator)


def run_block_em(
    enc_data: Any,
    estimator: MixtureEstimator,
    initial_model: MixtureDistribution,
    *,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    cache: FreezeRollupCache | None = None,
    budget_fraction: float = _DEFAULT_BUDGET_FRACTION,
    full_tree_every_round: bool = False,
    tie_tol: float = _DEFAULT_TIE_TOL,
    accept_tolerance: float = _DEFAULT_ACCEPT_TOLERANCE,
    q_gain_tol: float = 1.0e-6,  # keep in sync w/ FreezeRollupCache defaults
    weight_tol: float = 1.0e-4,
    weight_delta_tol: float = 1.0e-8,
    freeze_patience: int = 3,
    stall_patience: int = 5,
    escape_after_rejections: int = 3,
    max_skip_rounds: int = 2,
    boundary_weight_step: float = _DEFAULT_BOUNDARY_WEIGHT_STEP,
    full_refresh_interval: int | None = 10,
    schedule_wave_rounds: int = 1,
    objective_audit_interval: int | None | str = 10,
    compute_dtype: Any = None,
    cost_model: str = "structural",
    policy: str | LearnedController = "greedy",
    controller: LearnedController | None = None,
) -> tuple[MixtureDistribution, list[BlockEMStats]]:
    """Run block-coordinate-ascent EM over a :class:`MixtureDistribution` (workstream D3, D5).

    The first round updates every component. Later rounds rank blocks by the observed Q gain from
    their last update per structural-cost unit, select the highest-value blocks within
    ``budget_fraction``, reuse cached component scores for untouched blocks, and transactionally
    gate the resulting partial M-step on observed likelihood.

    ``schedule="auto"`` at the :func:`mixle.inference.estimation.optimize` layer dispatches here;
    ``budget_fraction=1.0`` or ``full_tree_every_round=True`` both degenerate this to (numerically
    indistinguishable from) vanilla full-tree EM -- see ``mixle.tests.block_em_test`` for a test
    of that literal property.

    ``delta`` convergence is gated by ``stall_patience`` consecutive small-gain rounds rather
    than a single one: a partial (budget-throttled) round can legitimately show a tiny total-objective
    gain even while a still-improving block simply wasn't scheduled that round (its turn is
    coming), so a single-round ``delta`` check -- fine for vanilla EM, where every round touches
    every block -- would risk declaring convergence early here. Requiring the plateau to persist
    for ``stall_patience`` rounds is the same style of robustness D2's own ``freeze_patience``
    already uses for its (structurally identical) "has this genuinely stopped moving" question.

    ``escape_after_rejections`` bounds a REJECTION livelock. A rejected round restores the exact
    prior state (score caches included), so the next round regenerates a near-identical proposal:
    when every proposal the scheduler can produce regresses observed likelihood (seen in practice
    with stochastic GradLeaf M-steps once the closed-form blocks reach their conditional fixed
    point), the loop would otherwise pay full proposal cost forever without moving. After that many
    CONSECUTIVE rejected rounds, one exact vanilla full-tree EM step is taken UNCONDITIONALLY
    (``acceptance_basis="vanilla_escape"``) -- monotone for closed-form leaves by the EM theorem,
    and for stochastic leaves it is precisely the take-the-step-and-let-the-next-E-step-resort
    semantics vanilla ``optimize`` uses. If a second full rejection streak accumulates without any
    normally-accepted round after the escape, no operator available makes progress from this point:
    the run stops early and the final receipt carries ``stop_reason="rejection_livelock"``.

    ``max_skip_rounds`` bounds starvation: a purely greedy top-score-wins ranking can, on a real
    fixture, rank the same eligible block last round after round (its own gain-per-cost score
    stays a little below its rivals') and never actually get a turn -- which is still valid
    coordinate ascent (the gated objective still only goes up), but converges to a WORSE fixed point than vanilla
    EM would reach in the same number of rounds, since a legitimately-still-moving block is
    parked indefinitely instead of merely delayed. Any eligible block skipped
    ``max_skip_rounds`` rounds in a row is therefore force-included in ``active`` regardless of
    its score (an aging boost) -- guaranteeing every eligible block gets a turn at least once
    every ``max_skip_rounds + 1`` rounds, which is what makes "same target objective, fewer evals" (as
    opposed to just "monotone but permanently stuck") an honest comparison against vanilla EM.

    ``objective_audit_interval`` controls exact observed-likelihood audits of Q-certified rounds;
    ``None`` disables periodic audits, but the final model is always audited. A non-finite or
    negative Q certificate always triggers an exact fallback gate. ``schedule_wave_rounds`` can
    amortize one selection over several rounds, but defaults to one because repeated coordinates
    can delay complementary updates; starvation immediately invalidates a pending wave.

    ``boundary_weight_step`` relaxes only a sparse weight update that would turn a previously
    positive component into an exact zero. The relaxed point lies on the segment from the current
    feasible weights to the conditional maximizer, so concavity preserves Q ascent while avoiding
    an irreversible floating-point simplex boundary. Its default leaves a machine-precision-derived
    interior margin. Components with exactly zero responsibility retain their existing parameters;
    positive components below ``weight_tol`` remain dormant between full refreshes rather than
    consuming ordinary block budget.

    ``full_refresh_interval`` schedules a globally coupled EM update periodically. This bounds
    stale block interactions and gives tiny-but-positive components a dense responsibility update;
    ``None`` disables refreshes. Refresh rounds are disclosed separately from score degeneracy and
    reset any pending schedule wave.

    ``policy`` (workstream D5, :mod:`mixle.inference.conditional_jit_controller`) selects WHO picks
    the per-round ``budget_fraction`` that feeds the exact same :func:`_select_active` ranking
    above: ``"greedy"`` (the default) uses the fixed ``budget_fraction`` argument every round,
    unchanged D3 behavior. ``"learned_bandit"``/``"learned_design_model"`` instead ask a
    :class:`~mixle.inference.conditional_jit_controller.LearnedController` for this round's budget
    (constructing a default :class:`~mixle.inference.conditional_jit_controller.BanditController`/
    :class:`~mixle.inference.conditional_jit_controller.DesignModelController` if ``controller`` is
    not supplied) and feed it back the round's REALIZED gain (this round's observed-objective improvement,
    ``round_value - current_value``, i.e. exactly what this round's active blocks achieved) and
    realized model-work cost (``evals_e + evals_c``) after every round -- so the controller learns
    from the same accept/reject-gated,
    provably-monotone rounds D3 already runs, never from an invented parallel objective. Passing an
    already-constructed :class:`~mixle.inference.conditional_jit_controller.LearnedController`
    directly as ``policy`` is also accepted (equivalent to passing it as ``controller`` with
    ``policy="learned_bandit"``/``"learned_design_model"``) and is how a caller warm-starts a
    controller across several calls/fits -- the SAME controller object keeps learning, since it is
    only ever mutated in place, never copied.

    Returns ``(final_model, history)`` where ``history[i]`` is round ``i``'s
    :class:`BlockEMStats`.

    ``q_gain_tol``, ``weight_delta_tol``, and ``freeze_patience`` remain accepted for API
    compatibility with earlier scheduler releases, but permanent freeze detection is no longer
    part of this path. ``weight_tol`` controls temporary dormancy only. Use
    :func:`mixle.inference.freeze_rollup.run_em_freeze_rollup` when those controls are desired.
    """
    if not isinstance(initial_model, MixtureDistribution):
        raise TypeError("run_block_em requires a MixtureDistribution model.")
    if (
        objective_audit_interval is not None
        and not isinstance(objective_audit_interval, str)
        and objective_audit_interval < 1
    ):
        raise ValueError("objective_audit_interval must be positive, None, or 'auto'.")
    if isinstance(objective_audit_interval, str) and objective_audit_interval != "auto":
        raise ValueError("objective_audit_interval accepts an int, None, or 'auto'; got %r." % objective_audit_interval)
    if cost_model not in ("structural", "measured"):
        raise ValueError("cost_model must be 'structural' or 'measured'; got %r." % (cost_model,))
    if schedule_wave_rounds < 1:
        raise ValueError("schedule_wave_rounds must be positive.")
    if not 0.0 < boundary_weight_step <= 1.0:
        raise ValueError("boundary_weight_step must be in (0, 1].")
    if full_refresh_interval is not None and full_refresh_interval < 1:
        raise ValueError("full_refresh_interval must be positive or None.")
    if escape_after_rejections < 1:
        raise ValueError("escape_after_rejections must be a positive integer.")
    if isinstance(policy, LearnedController):
        controller = policy
        policy = "learned"
    elif policy == "learned_bandit":
        controller = controller if controller is not None else BanditController()
        policy = "learned"
    elif policy == "learned_design_model":
        controller = controller if controller is not None else DesignModelController()
        policy = "learned"
    elif policy == "greedy":
        controller = None
    else:
        raise ValueError(
            "policy must be one of %r, or a LearnedController instance; got %r." % (_KNOWN_POLICIES, policy)
        )
    cache = (
        FreezeRollupCache(
            q_gain_tol=q_gain_tol,
            weight_tol=weight_tol,
            weight_delta_tol=weight_delta_tol,
            freeze_patience=freeze_patience,
        )
        if cache is None
        else cache
    )
    enc_payload = _resolve_payload(enc_data)

    model = initial_model
    history: list[BlockEMStats] = []
    old_value: float | None = None
    last_q_gain: dict[int, float] = {}
    skip_streak: dict[int, int] = {}
    stall_streak = 0
    rejected_streak = 0
    escapes_since_accept = 0
    current_ll_mat: np.ndarray | None = None
    current_log_density: np.ndarray | None = None
    current_impossible: np.ndarray | None = None
    mutable_state = has_mutable_state(model, estimator)
    if objective_audit_interval == "auto":
        # Certificate-aware audit cadence: a tree of exact/GEM updates carries the monotone
        # certificate, so its Q-certified rounds need only sparse exact audits; stochastic
        # (mutable) updates get dense ones. Mirrors the convergence-certificate tiers.
        objective_audit_interval = 5 if mutable_state else 20
    measured_cost: dict[int, float] = {}
    wave_active: set[int] | None = None
    wave_eligible: set[int] | None = None
    wave_degenerate = False
    wave_remaining = 0

    for round_index in range(max(1, int(max_its))):
        round_started = time.perf_counter()
        escape_round = False
        if rejected_streak >= escape_after_rejections:
            if escapes_since_accept:
                # A vanilla escape was already spent since the last normally-accepted round and the
                # scheduler is STILL only producing rejected proposals: no operator available makes
                # progress from here. Stop instead of paying full proposal cost forever.
                if history:
                    history[-1].stop_reason = "rejection_livelock"
                break
            escape_round = True
        # Permanent freezing is intentionally not used here. The block scheduler already bounds
        # temporary inactivity with max_skip_rounds; permanently freezing from a heuristic residual
        # would change the reachable fixed point. The separate freeze_rollup API remains available
        # for callers that explicitly opt into its near-zero-weight policy.
        frozen_idx: set[int] = set()
        positive = [idx for idx in range(model.num_components) if idx not in frozen_idx and not model.zw[idx]]
        dormant = (
            set()
            if full_tree_every_round or budget_fraction >= 1.0
            else {idx for idx in positive if model.w[idx] < weight_tol}
        )
        eligible = [idx for idx in positive if idx not in dormant]
        scores, cost, residual, q_gain, cost_basis = _block_scores(
            model, positive, last_q_gain, measured_cost if cost_model == "measured" else None
        )

        controller_state: ControllerState | None = None
        controller_action: ControllerAction | None = None
        round_budget_fraction = budget_fraction
        if controller is not None:
            controller_state = ControllerState.from_scores(round_index, eligible, scores, cost, residual, q_gain)
            controller_action = controller.select_action(controller_state)
            # A learned policy augments the declared greedy safety baseline; it cannot silently
            # starve more work than the caller's budget_fraction permits. This keeps cold-start
            # exploration bounded while still allowing a controller to spend more when useful.
            round_budget_fraction = max(float(budget_fraction), controller_action.budget_fraction)

        starved = {idx for idx in eligible if skip_streak.get(idx, 0) >= max(0, int(max_skip_rounds))}
        full_refresh = round_index > 0 and (
            escape_round
            or not eligible
            or (full_refresh_interval is not None and round_index % full_refresh_interval == 0)
        )
        schedule_reused = (
            round_index > 0
            and not full_refresh
            and controller is None
            and wave_remaining > 0
            and wave_active is not None
            and wave_eligible == set(eligible)
            and starved <= wave_active
        )
        if round_index == 0:
            # Bootstrap every block's observed Q-gain from one real full sweep. This replaces the
            # former self-sampled entropy proxy and gives the next round a data-linked ranking.
            active, degenerate = set(positive), True
        elif full_refresh:
            score_values = [scores[idx] for idx in eligible]
            naturally_degenerate = (
                full_tree_every_round or not score_values or max(score_values) - min(score_values) < tie_tol
            )
            active, degenerate = set(positive), naturally_degenerate
        elif schedule_reused:
            active = set(wave_active)
            degenerate = wave_degenerate
            wave_remaining -= 1
        else:
            active, degenerate = _select_active(
                eligible,
                scores,
                cost,
                budget_fraction=round_budget_fraction,
                full_tree_every_round=full_tree_every_round,
                tie_tol=tie_tol,
                forced=starved,
            )
        if round_index == 0 or full_refresh:
            # The bootstrap sweep exists to measure every block; it is not a scheduling
            # decision and must not turn the next round into another full-tree sweep.
            wave_active = None
            wave_eligible = None
            wave_remaining = 0
        elif not schedule_reused:
            wave_active = set(active)
            wave_eligible = set(eligible)
            wave_degenerate = degenerate
            wave_remaining = schedule_wave_rounds - 1
        for idx in eligible:
            skip_streak[idx] = 0 if idx in active else skip_streak.get(idx, 0) + 1
        for idx in list(skip_streak):
            if idx not in eligible:
                del skip_streak[idx]
        scheduled_inactive = frozen_idx | (set(positive) - active)
        scheduling_seconds = time.perf_counter() - round_started

        reused_current_matrix = current_ll_mat is not None
        if current_ll_mat is None:
            ll_mat, evals_e, estep_profile = _component_log_density_matrix_profiled(
                model, enc_payload, cache, scheduled_inactive, compute_dtype=compute_dtype
            )
        else:
            ll_mat = current_ll_mat
            evals_e = 0
            estep_profile = DensityMatrixProfile(0, 0, model.num_components, 0, 0.0, 0.0, 0.0, ())
        expected_estep_evals = 0 if reused_current_matrix else sum(not value for value in model.zw)
        sparse_round = (
            current_log_density is not None
            and current_impossible is not None
            and not np.any(current_impossible)
            and 0 < len(active) < len(positive)
            and not (estimator.has_conj_prior and estimator.fixed_weights is None)
            and not (estimator.w_min > 0.0 and estimator.fixed_weights is None)
        )
        active_indices = tuple(sorted(active))
        responsibility_started = time.perf_counter()
        if sparse_round:
            log_density = current_log_density
            gamma_active = _active_responsibilities(ll_mat, model.log_w, log_density, active_indices)
            gamma = None
            responsibility_columns = len(active_indices)
        else:
            log_density, gamma = _combine(ll_mat, model.log_w)
            gamma_active = gamma[:, active_indices]
            responsibility_columns = model.num_components
        current_value = float(np.sum(log_density))
        responsibility_seconds = time.perf_counter() - responsibility_started
        exact_delta = None if old_value is None else current_value - old_value

        snapshot_started = time.perf_counter()
        transaction = MutableStateSnapshot.capture(model, estimator) if mutable_state else None
        snapshot_seconds = time.perf_counter() - snapshot_started
        mstep_started = time.perf_counter()
        if sparse_round:
            candidate, active_counts, inactive_scale = _sparse_block_m_step(
                enc_payload,
                estimator,
                model,
                active_indices,
                gamma_active,
                boundary_weight_step,
            )
            counts = None
        else:
            if gamma is None:  # pragma: no cover - established by the branch above
                raise RuntimeError("dense block M-step requires dense responsibilities.")
            candidate = _m_step(enc_payload, estimator, model, gamma, scheduled_inactive)
            counts = gamma.sum(axis=0)
            active_counts = counts[list(active_indices)]
            inactive_scale = 1.0
        mstep_seconds = time.perf_counter() - mstep_started
        if sparse_round:
            candidate_indices, candidate_columns, evals_c, candidate_profile = (
                _updated_component_log_density_columns_profiled(
                    candidate,
                    enc_payload,
                    active,
                    ll_mat.shape[0],
                    compute_dtype=compute_dtype,
                )
            )
            ll_mat_c = None
        else:
            ll_mat_c, evals_c, candidate_profile = _updated_component_log_density_matrix_profiled(
                candidate, enc_payload, active, ll_mat, compute_dtype=compute_dtype
            )
            candidate_indices = active_indices
            candidate_columns = ll_mat_c[:, candidate_indices]
        accounting_started = time.perf_counter()
        measured_q_gain: dict[int, float] = {}
        emission_q_gain = 0.0
        active_weight_q_gain = 0.0
        for position, idx in enumerate(candidate_indices):
            component_gain = float(np.dot(gamma_active[:, position], candidate_columns[:, position] - ll_mat[:, idx]))
            emission_q_gain += component_gain
            count = float(active_counts[position])
            if count <= 0.0 or not np.isfinite(candidate.log_w[idx]) or not np.isfinite(model.log_w[idx]):
                weight_gain = 0.0
            else:
                weight_gain = float(count * (candidate.log_w[idx] - model.log_w[idx]))
            active_weight_q_gain += weight_gain
            measured_q_gain[idx] = component_gain + weight_gain

        if sparse_round:
            inactive_count = max(0.0, float(ll_mat.shape[0]) - float(np.sum(active_counts)))
            if inactive_count <= 0.0:
                inactive_weight_q_gain = 0.0
            elif inactive_scale > 0.0:
                inactive_weight_q_gain = inactive_count * np.log(inactive_scale)
            else:
                inactive_weight_q_gain = -np.inf
            weight_q_gain = active_weight_q_gain + inactive_weight_q_gain
        else:
            weight_q_gain = 0.0
            if counts is None:  # pragma: no cover - established by the dense branch
                raise RuntimeError("dense Q certificate requires dense counts.")
            for idx in range(model.num_components):
                if counts[idx] <= 0.0:
                    continue
                if not np.isfinite(candidate.log_w[idx]) or not np.isfinite(model.log_w[idx]):
                    weight_q_gain = -np.inf
                    break
                weight_q_gain += float(counts[idx] * (candidate.log_w[idx] - model.log_w[idx]))
        certified_q_gain = emission_q_gain + weight_q_gain
        q_accepted = np.isfinite(certified_q_gain) and certified_q_gain + accept_tolerance >= 0.0
        periodic_audit = objective_audit_interval is not None and ((round_index + 1) % objective_audit_interval == 0)
        pre_validation_accounting_seconds = time.perf_counter() - accounting_started
        validation_started = time.perf_counter()
        normalizer_audit_failed = False
        normalizer_audit_error = 0.0
        normalizer_logspace_rows = 0
        if sparse_round:
            inactive_indices = tuple(idx for idx in range(model.num_components) if idx not in active)
            candidate_log_density, candidate_impossible, normalizer_logspace_rows = _incremental_candidate_log_density(
                log_density,
                gamma_active,
                ll_mat[:, inactive_indices],
                model.log_w[list(inactive_indices)],
                candidate_columns,
                candidate.log_w[list(candidate_indices)],
                inactive_scale,
            )
            if periodic_audit:
                audit_matrix = ll_mat.copy()
                audit_matrix[:, candidate_indices] = candidate_columns
                audited_log_density = _log_density_from_matrix(audit_matrix, candidate.log_w)
                scale = max(1.0, float(np.max(np.abs(audited_log_density))))
                audit_span = objective_audit_interval if objective_audit_interval is not None else 1
                audit_tolerance = (
                    64.0 * np.finfo(np.float64).eps * scale * max(1, audit_span) * max(1, len(candidate_indices))
                )
                normalizer_audit_error = float(np.max(np.abs(candidate_log_density - audited_log_density)))
                normalizer_audit_failed = normalizer_audit_error > audit_tolerance
                candidate_log_density = audited_log_density
                candidate_impossible = np.all(
                    np.isneginf(audit_matrix + candidate.log_w),
                    axis=1,
                )
        else:
            if ll_mat_c is None:  # pragma: no cover - established by the dense branch
                raise RuntimeError("dense candidate validation requires a component-score matrix.")
            candidate_log_density = _log_density_from_matrix(ll_mat_c, candidate.log_w)
            candidate_impossible = np.all(np.isneginf(ll_mat_c + candidate.log_w), axis=1)
        candidate_value = float(np.sum(candidate_log_density))
        validation_seconds = time.perf_counter() - validation_started
        post_validation_accounting_started = time.perf_counter()
        audit_failed = q_accepted and candidate_value + accept_tolerance < current_value

        observed_accepted = np.isfinite(candidate_value) and candidate_value + accept_tolerance >= current_value
        # An escape round takes the exact vanilla full-tree step UNCONDITIONALLY (monotone for
        # closed-form leaves; deliberate take-the-step semantics for stochastic ones) -- gating it
        # would just re-reject the proposal the livelock keeps regenerating.
        accepted = (
            (escape_round and np.isfinite(candidate_value))
            or (q_accepted and not audit_failed)
            or (not q_accepted and observed_accepted)
        )
        if accepted:
            model = candidate
            if sparse_round:
                ll_mat[:, candidate_indices] = candidate_columns
                current_ll_mat = ll_mat
            else:
                current_ll_mat = ll_mat_c
            current_log_density = candidate_log_density
            current_impossible = candidate_impossible
            round_value = candidate_value
            objective_exact = True
            if escape_round:
                acceptance_basis = "vanilla_escape"
            elif not q_accepted:
                acceptance_basis = "observed_fallback"
            elif periodic_audit:
                acceptance_basis = "q_certified_audited"
            elif sparse_round:
                acceptance_basis = "q_certified_incremental"
            else:
                acceptance_basis = "q_certified"
            for idx, gain in measured_q_gain.items():
                last_q_gain[idx] = max(float(gain), 0.0)
            rejected_streak = 0
            escapes_since_accept = 1 if escape_round else 0
        else:
            if transaction is not None:
                transaction.restore()
            round_value = current_value
            objective_exact = True
            acceptance_basis = "rejected"
            for idx in active:
                last_q_gain[idx] = 0.0
            rejected_streak += 1

        if controller is not None and controller_state is not None and controller_action is not None:
            realized_gain = max(0.0, round_value - current_value)
            realized_cost = float(evals_e + evals_c)
            controller.update(controller_state, controller_action, realized_gain, realized_cost)

        selected_cost = float(sum(cost[idx] for idx in active))
        eligible_cost = float(sum(cost[idx] for idx in eligible))
        forced_cost = float(sum(cost[idx] for idx in starved))
        expected_evals = expected_estep_evals + sum(not candidate.zw[idx] for idx in active)
        expected_reuses = (model.num_components if reused_current_matrix else 0) + model.num_components - len(active)
        observed_evals = evals_e + evals_c
        observed_reuses = (
            estep_profile.cache_hits
            + estep_profile.reused_columns
            + candidate_profile.cache_hits
            + candidate_profile.reused_columns
        )
        failures = []
        if observed_evals != expected_evals:
            failures.append("density_evaluation_count_mismatch")
        if observed_reuses != expected_reuses:
            failures.append("score_reuse_count_mismatch")
        expected_responsibility_columns = len(active_indices) if sparse_round else model.num_components
        if responsibility_columns != expected_responsibility_columns:
            failures.append("responsibility_column_count_mismatch")
        if audit_failed:
            failures.append("q_certificate_audit_failed")
        if normalizer_audit_failed:
            failures.append("incremental_normalizer_audit_failed")
        budget_cost = round_budget_fraction * eligible_cost
        if (
            not degenerate
            and not full_refresh
            and selected_cost > budget_cost + 1.0e-12
            and forced_cost <= budget_cost + 1.0e-12
        ):
            failures.append("scheduler_budget_exceeded_without_forced_work")
        for profile in (estep_profile, candidate_profile):
            for cost_idx, seconds in profile.component_evaluation_seconds:
                if seconds > 0.0:
                    measured_cost[cost_idx] = float(seconds)
        assumptions = BlockEMAssumptionReceipt(
            declared_budget_fraction=round_budget_fraction,
            cost_basis=cost_basis,
            eligible_cost=eligible_cost,
            selected_cost=selected_cost,
            forced_cost=forced_cost,
            expected_density_evaluations=expected_evals,
            observed_density_evaluations=observed_evals,
            expected_score_reuses=expected_reuses,
            observed_score_reuses=observed_reuses,
            schedule_reused=schedule_reused,
            full_refresh=full_refresh,
            expected_responsibility_columns=expected_responsibility_columns,
            observed_responsibility_columns=responsibility_columns,
            incremental_normalizer=sparse_round,
            normalizer_audit_max_abs_error=normalizer_audit_error,
            normalizer_logspace_rows=normalizer_logspace_rows,
            failures=tuple(failures),
        )
        accounting_seconds = pre_validation_accounting_seconds + (
            time.perf_counter() - post_validation_accounting_started
        )
        wall_time_seconds = float(time.perf_counter() - round_started)
        timing = BlockEMTimingReceipt(
            scheduling_seconds=scheduling_seconds,
            estep_density_seconds=estep_profile.elapsed_seconds,
            responsibility_seconds=responsibility_seconds,
            snapshot_seconds=snapshot_seconds,
            mstep_seconds=mstep_seconds,
            candidate_density_seconds=candidate_profile.elapsed_seconds,
            validation_seconds=validation_seconds,
            accounting_seconds=accounting_seconds,
            total_seconds=wall_time_seconds,
            component_density_seconds=tuple(
                ("estep", idx, seconds) for idx, seconds in estep_profile.component_evaluation_seconds
            )
            + tuple(("candidate", idx, seconds) for idx, seconds in candidate_profile.component_evaluation_seconds),
        )
        n_zero = int(np.count_nonzero(model.zw)) if not accepted else int(np.count_nonzero(candidate.zw))
        history.append(
            BlockEMStats(
                round_index=round_index,
                n_components=model.num_components,
                n_active=len(active),
                n_frozen=len(frozen_idx),
                n_zero_weight=n_zero,
                n_dormant=len(dormant),
                n_scheduled_inactive=len(set(positive) - active),
                n_log_density_evals=evals_e + evals_c,
                n_responsibility_columns=responsibility_columns,
                objective=round_value,
                accepted=accepted,
                acceptance_basis=acceptance_basis,
                objective_exact=objective_exact,
                degenerate_round=degenerate,
                measured_q_gain=float(sum(measured_q_gain.values())) if accepted else 0.0,
                certified_q_gain=float(certified_q_gain) if np.isfinite(certified_q_gain) else float("-inf"),
                wall_time_seconds=wall_time_seconds,
                assumptions=assumptions,
                timing=timing,
            )
        )

        if delta is not None and exact_delta is not None and 0.0 <= exact_delta < delta:
            stall_streak += 1
            if stall_streak >= max(1, int(stall_patience)):
                break
        else:
            stall_streak = 0
        old_value = current_value

    if history and current_ll_mat is not None:
        audit_started = time.perf_counter()
        final_value = float(np.sum(_log_density_from_matrix(current_ll_mat, model.log_w)))
        audit_seconds = time.perf_counter() - audit_started
        if final_value + accept_tolerance < history[-1].objective:
            raise RuntimeError("final observed objective violated the certified EM lower bound.")
        history[-1].objective = final_value
        history[-1].objective_exact = True
        history[-1].final_audit_seconds = audit_seconds

    return model, history
