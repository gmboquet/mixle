"""Block-coordinate-ascent EM scheduler -- greedy gain-per-cost block selection (workstream D3).

Frame (see the ConditionalJIT track, D1-D6): **the estimator tree is an IR**. D1
(:mod:`mixle.inference.node_report`) instruments every node with a per-round residual/Q-gain
report, an ``update_kind`` classification, and an E/M cost proxy. D2
(:mod:`mixle.inference.freeze_rollup`) spent that report on one particular schedule: freeze a
subtree once it looks converged and skip it forever after. D3 generalizes the scheduling
question one level up: on EVERY round, rank the blocks (mixture components) that are NOT
already D2-frozen by D1's gain-per-cost (``q_gain / (e_step_cost + m_step_cost)``) and update
only the highest-value ones within a per-round cost budget, leaving the rest untouched for that
round -- a genuine block-coordinate-ascent / ECM schedule, not just "freeze forever".

Correctness backbone (unchanged from the rest of the D-track): this module is a SCHEDULING
optimization only. Any interleaving of a partial E-step (fresh log-density for the active
blocks, cached log-density for the inactive ones) and a per-block conditional M-step (only the
active blocks are re-estimated; every other block's model object is carried forward byte-for-
byte unchanged) is still coordinate ascent on the SAME Neal-Hinton free energy F vanilla EM
climbs -- so an accept/reject gate on the round's candidate objective (identical in spirit to
:class:`mixle.inference.em.MonotonicEM` and D2's own ``run_em_freeze_rollup``) is what turns
"should be monotone" into "IS monotone, mechanically, every round" here too.

Composition with D2: a component D2's :class:`~mixle.inference.freeze_rollup.FreezeRollupCache`
already reports frozen (:func:`mixle.inference.freeze_rollup.detect_frozen`) is, from this
scheduler's point of view, exactly a "zero e-step cost, zero m-step cost" block -- it is excluded
from the gain-per-cost ranking entirely (nothing to rank: it costs nothing and is never
scheduled to move) and is served for free from the SAME cache this module reuses for its own
this-round-only inactive blocks. A block the scheduler chooses not to update THIS round is, from
:class:`FreezeRollupCache`'s point of view, indistinguishable from a D2-frozen block for exactly
one round: its parameter signature has not moved, so the cached per-datum log-density is still a
byte-identical cache hit -- no separate caching mechanism is needed for D3, only a wider set of
"don't touch this round" indices fed into the same ``frozen=`` parameter D2 already threads
through :func:`~mixle.inference.freeze_rollup._component_log_density_matrix` and
:func:`~mixle.inference.freeze_rollup._m_step`.

Gain estimate, documented per the D1 module's own note that later track items may re-estimate
a cheap proxy fresh every round rather than reuse a stale report: this module calls
:func:`mixle.inference.node_report.node_report` on every eligible component EVERY round (a small,
fixed-size Monte-Carlo self-residual -- ``_DEFAULT_MC_SAMPLES`` samples, not a real-data pass),
always with the SAME seed across components (unlike :func:`mixle.inference.node_report.
flat_report_table`, which offsets the seed per row for a deduplicated tree walk) so that
structurally-identical components draw directly-comparable residuals -- this is what makes the
"no useful discrimination to make" acceptance criterion (identical components => identical
scores => no real ranking) hold deterministically rather than by RNG luck.

Degeneration to vanilla full-tree EM: when every eligible block's gain-per-cost score is the
same (within ``tie_tol``) -- either because the blocks truly are indistinguishable, or because
the caller passes ``full_tree_every_round=True`` as an explicit escape hatch -- the scheduler has
no real choice to make, so it updates every eligible block, exactly like a plain full-tree EM
round would. See ``mixle.tests.block_em_test`` for a literal test of this property.
"""

from __future__ import annotations

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
    FreezeRollupCache,
    _combine,
    _component_log_density_matrix,
    _m_step,
    _resolve_payload,
    detect_frozen,
)
from mixle.inference.node_report import node_report
from mixle.stats.latent.mixture import MixtureDistribution, MixtureEstimator

_DEFAULT_ACCEPT_TOLERANCE = 1.0e-9
_DEFAULT_BUDGET_FRACTION = 0.5
_DEFAULT_TIE_TOL = 1.0e-9
_SCORE_SEED = 0  # fixed, shared across every component -- see module docstring on the tie test.
_KNOWN_POLICIES = ("greedy", "learned_bandit", "learned_design_model")


@dataclass
class BlockEMStats:
    """One round's accounting for the block-EM scheduler -- the acceptance-criteria receipt.

    Mirrors :class:`mixle.inference.freeze_rollup.FreezeRollupStats` (same
    ``n_log_density_evals`` wall-clock proxy and real Neal-Hinton ``objective``), plus the extra
    fields specific to gain-per-cost SCHEDULING within a round rather than permanent freezing.
    """

    round_index: int
    n_components: int
    n_active: int
    n_frozen: int
    n_zero_weight: int
    n_scheduled_inactive: int
    n_log_density_evals: int
    objective: float
    accepted: bool = True
    degenerate_round: bool = False

    @property
    def active_fraction(self) -> float:
        """Fraction of components genuinely (re-)evaluated this round, out of all components."""
        return self.n_active / self.n_components if self.n_components else 1.0


def _block_scores(
    model: MixtureDistribution,
    eligible: list[int],
    prev_residual: dict[int, float],
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """Return ``(gain_per_cost, cost, residual, q_gain)`` for each ``eligible`` component index.

    ``gain`` is the D1 Q-gain (residual improvement since the last round this component was
    scored) when available, else the raw residual itself (a component never scored before has an
    unknown Q-gain but a high residual is still a legitimate "there is a lot of room to improve
    here" proxy for round 1). ``cost`` is D1's E-step-cost + M-step-cost proxy. Every component
    is scored with the SAME fixed seed (see module docstring) so structurally-identical
    components are directly comparable -- required for the degenerate-to-vanilla-EM test to be
    deterministic rather than RNG-dependent. ``residual``/``q_gain`` (the raw, un-normalized D1
    fields) are returned alongside the derived ``gain_per_cost``/``cost`` purely so a D5 learned
    policy can build a :class:`~mixle.inference.conditional_jit_controller.ControllerState`
    without a second pass over the tree.
    """
    gain_per_cost: dict[int, float] = {}
    cost: dict[int, float] = {}
    residual: dict[int, float] = {}
    q_gain: dict[int, float] = {}
    for idx in eligible:
        report = node_report(
            model.components[idx],
            field_path=str(idx),
            seed=_SCORE_SEED,
            nobs=1.0,
            prev_residual=prev_residual.get(idx),
        )
        prev_residual[idx] = report.residual
        gain = report.q_gain if report.q_gain is not None else report.residual
        if not np.isfinite(gain):
            gain = 0.0
        block_cost = max(report.e_step_cost + report.m_step_cost, 1.0e-12)
        cost[idx] = block_cost
        gain_per_cost[idx] = gain / block_cost
        residual[idx] = report.residual if np.isfinite(report.residual) else 0.0
        q_gain[idx] = gain
    return gain_per_cost, cost, residual, q_gain


def _select_active(
    eligible: list[int],
    scores: dict[int, float],
    cost: dict[int, float],
    *,
    budget_fraction: float,
    full_tree_every_round: bool,
    tie_tol: float,
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

    ranked = sorted(eligible, key=lambda i: -scores[i])
    total_cost = sum(cost[idx] for idx in eligible)
    budget = max(budget_fraction, 0.0) * total_cost

    active: set[int] = set()
    spent = 0.0
    for idx in ranked:
        if active and spent >= budget:
            break
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
    max_skip_rounds: int = 2,
    policy: str | LearnedController = "greedy",
    controller: LearnedController | None = None,
) -> tuple[MixtureDistribution, list[BlockEMStats]]:
    """Run block-coordinate-ascent EM over a :class:`MixtureDistribution` (workstream D3, D5).

    Each round: rank every component D2 does not already report frozen by D1 gain-per-cost,
    select the highest-value ones within ``budget_fraction`` of the round's total eligible cost
    (the rest are left untouched for this round), do a fresh partial E-step (cached log-density
    reused for every untouched/frozen component -- same mechanism D2's
    :class:`~mixle.inference.freeze_rollup.FreezeRollupCache` already provides), a per-block
    conditional M-step over only the active components, and an accept/reject gate on the round's
    real objective -- exactly D2's own monotone-F machinery, reused rather than reimplemented.

    ``schedule="auto"`` at the :func:`mixle.inference.estimation.optimize` layer dispatches here;
    ``budget_fraction=1.0`` or ``full_tree_every_round=True`` both degenerate this to (numerically
    indistinguishable from) vanilla full-tree EM -- see ``mixle.tests.block_em_test`` for a test
    of that literal property.

    ``delta`` convergence is gated by ``stall_patience`` consecutive small-gain rounds rather
    than a single one: a partial (budget-throttled) round can legitimately show a tiny total-F
    gain even while a still-improving block simply wasn't scheduled that round (its turn is
    coming), so a single-round ``delta`` check -- fine for vanilla EM, where every round touches
    every block -- would risk declaring convergence early here. Requiring the plateau to persist
    for ``stall_patience`` rounds is the same style of robustness D2's own ``freeze_patience``
    already uses for its (structurally identical) "has this genuinely stopped moving" question.

    ``max_skip_rounds`` bounds STARVATION: a purely greedy top-score-wins ranking can, on a real
    fixture, rank the same eligible block last round after round (its own gain-per-cost score
    stays a little below its rivals') and never actually get a turn -- which is still valid
    coordinate ascent (F still only goes up), but converges to a WORSE fixed point than vanilla
    EM would reach in the same number of rounds, since a legitimately-still-moving block is
    parked indefinitely instead of merely delayed. Any eligible block skipped
    ``max_skip_rounds`` rounds in a row is therefore force-included in ``active`` regardless of
    its score (an aging boost) -- guaranteeing every eligible block gets a turn at least once
    every ``max_skip_rounds + 1`` rounds, which is what makes "same target F, fewer evals" (as
    opposed to just "monotone but permanently stuck") an honest comparison against vanilla EM.

    ``policy`` (workstream D5, :mod:`mixle.inference.conditional_jit_controller`) selects WHO picks
    the per-round ``budget_fraction`` that feeds the exact same :func:`_select_active` ranking
    above: ``"greedy"`` (the default) uses the fixed ``budget_fraction`` argument every round,
    unchanged D3 behavior. ``"learned_bandit"``/``"learned_design_model"`` instead ask a
    :class:`~mixle.inference.conditional_jit_controller.LearnedController` for this round's budget
    (constructing a default :class:`~mixle.inference.conditional_jit_controller.BanditController`/
    :class:`~mixle.inference.conditional_jit_controller.DesignModelController` if ``controller`` is
    not supplied) and feed it back the round's REALIZED gain (this round's own F improvement,
    ``round_value - current_value``, i.e. exactly what this round's active blocks achieved) and
    REALIZED cost (``evals_e + evals_c``, the same wall-clock proxy :class:`BlockEMStats` already
    reports) after every round -- so the controller learns from the SAME accept/reject-gated,
    provably-monotone rounds D3 already runs, never from an invented parallel objective. Passing an
    already-constructed :class:`~mixle.inference.conditional_jit_controller.LearnedController`
    directly as ``policy`` is also accepted (equivalent to passing it as ``controller`` with
    ``policy="learned_bandit"``/``"learned_design_model"``) and is how a caller warm-starts a
    controller across several calls/fits -- the SAME controller object keeps learning, since it is
    only ever mutated in place, never copied.

    Returns ``(final_model, history)`` where ``history[i]`` is round ``i``'s
    :class:`BlockEMStats`.
    """
    if not isinstance(initial_model, MixtureDistribution):
        raise TypeError("run_block_em requires a MixtureDistribution model.")
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
    prev_residual: dict[int, float] = {}
    skip_streak: dict[int, int] = {}
    stall_streak = 0

    for round_index in range(max(1, int(max_its))):
        frozen_idx = detect_frozen(cache, model)
        eligible = [idx for idx in range(model.num_components) if idx not in frozen_idx and not model.zw[idx]]
        scores, cost, residual, q_gain = _block_scores(model, eligible, prev_residual)

        controller_state: ControllerState | None = None
        controller_action: ControllerAction | None = None
        round_budget_fraction = budget_fraction
        if controller is not None:
            controller_state = ControllerState.from_scores(round_index, eligible, scores, cost, residual, q_gain)
            controller_action = controller.select_action(controller_state)
            round_budget_fraction = controller_action.budget_fraction

        active, degenerate = _select_active(
            eligible,
            scores,
            cost,
            budget_fraction=round_budget_fraction,
            full_tree_every_round=full_tree_every_round,
            tie_tol=tie_tol,
        )
        starved = {idx for idx in eligible if skip_streak.get(idx, 0) >= max(0, int(max_skip_rounds))}
        if starved - active:
            active = active | starved
        for idx in eligible:
            skip_streak[idx] = 0 if idx in active else skip_streak.get(idx, 0) + 1
        for idx in list(skip_streak):
            if idx not in eligible:
                del skip_streak[idx]
        scheduled_inactive = frozen_idx | (set(eligible) - active)

        ll_mat, evals_e = _component_log_density_matrix(model, enc_payload, cache, scheduled_inactive)
        log_density, gamma = _combine(ll_mat, model.log_w)
        current_value = float(np.sum(log_density))
        if old_value is None:
            old_value = current_value

        candidate = _m_step(enc_payload, estimator, model, gamma, scheduled_inactive)
        candidate_frozen = detect_frozen(cache, candidate)
        candidate_inactive = candidate_frozen | (set(eligible) - active)
        ll_mat_c, evals_c = _component_log_density_matrix(candidate, enc_payload, cache, candidate_inactive)
        candidate_log_density, _ = _combine(ll_mat_c, candidate.log_w)
        candidate_value = float(np.sum(candidate_log_density))

        accepted = np.isfinite(candidate_value) and candidate_value + accept_tolerance >= current_value
        if accepted:
            model = candidate
            round_value = candidate_value
        else:
            round_value = current_value

        if controller is not None and controller_state is not None and controller_action is not None:
            realized_gain = max(0.0, round_value - current_value)
            realized_cost = float(evals_e + evals_c)
            controller.update(controller_state, controller_action, realized_gain, realized_cost)

        n_zero = int(np.count_nonzero(model.zw)) if not accepted else int(np.count_nonzero(candidate.zw))
        history.append(
            BlockEMStats(
                round_index=round_index,
                n_components=model.num_components,
                n_active=len(active),
                n_frozen=len(frozen_idx),
                n_zero_weight=n_zero,
                n_scheduled_inactive=len(set(eligible) - active),
                n_log_density_evals=evals_e + evals_c,
                objective=round_value,
                accepted=accepted,
                degenerate_round=degenerate,
            )
        )

        if delta is not None and 0.0 <= round_value - old_value < delta:
            stall_streak += 1
            if stall_streak >= max(1, int(stall_patience)):
                break
        else:
            stall_streak = 0
        old_value = round_value

    return model, history
