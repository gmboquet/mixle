"""D5: the LEARNED controller over the D1-D4/D6 estimator-tree IR (workstream ConditionalJIT track).

The estimator tree is an update IR. :mod:`mixle.inference.block_em` exposes per-round observed
complete-data Q gains, structural costs, and realized objective gains. This module lets a policy
choose the scheduler's budget from those receipts, either online from realized gain/cost or from
logged history.

Correctness backbone (unchanged from the rest of the D-track): a learned scheduling POLICY can
only ever change WHICH blocks get how much budget THIS round -- never what a block's update
computes, never the accept/reject observed-objective gate D2/D3 already enforce. The scheduler
directly evaluates observed-data log likelihood after each partial update, so that remains the audit receipt;
a bad learned decision costs speed (a wasted round, a slow warmup), never correctness. This module
therefore never runs its own EM and never touches ``accepted``/``objective`` bookkeeping -- it only
chooses the ``budget_fraction`` knob :func:`mixle.inference.block_em.run_block_em`'s existing
greedy ranking (:func:`mixle.inference.block_em._select_active`) already accepts, so "learned" and
"greedy" are two policies over the exact same scheduling loop, not two different loops.

Two learning modes:

* **Online bandits** (:class:`BanditController`) -- reuses :mod:`mixle.task.bandit` (this
  codebase's existing multi-armed-bandit module, built for exactly this "pick an
  arm/observe-a-reward, no offline data needed" loop) rather than reimplementing UCB1/Thompson
  sampling. The action space is a small discretized set of ``budget_fraction`` levels (an "arm");
  the reward is the round's realized observed-objective gain per evaluated component. Learns round-by-round DURING a fit, and
  carries over (the same policy object, still adapting) across multiple fits if the caller reuses
  it -- see the offline-vs-warm-start framing in ``mixle.tests.conditional_jit_controller_test``.
* **Offline DesignModel** (:class:`DesignModelController`) -- reuses
  :class:`mixle.task.edge.DesignModel` (this codebase's existing design-space surrogate: a GP
  fitted on logged ``(point, quality, fingerprint)`` rows, warm-startable across different but
  related tasks via its fingerprint machinery -- see :mod:`mixle.task.edge`'s own docstring)
  rather than reinventing an offline contextual bandit. ``budget_fraction`` is treated as the
  (continuous, 1-D) design point; the current round's aggregated scheduling features
  (:class:`ControllerState`) are the fingerprint DesignModel already conditions proposals on, so a
  DesignModel trained on logged rounds from OTHER fit problems can propose a good budget for a
  brand-new, held-out problem with ZERO online exploration.

Action-type registry, not a closed enum: :class:`ActionType` lists every action kind the D-track
roadmap names for D5 (``BLOCK_SELECTION``, ``BUDGET_ALLOCATION``, plus the future
``STRUCTURE_EDIT`` -- H3's structure-edit schedule / G2's projections / the evolve-ops population
search -- and ``BACKEND_CHOICE`` -- D6's compile-economics ``RespecializationDecision``). Only
``BLOCK_SELECTION`` and ``BUDGET_ALLOCATION`` are REAL, implemented action types today (and in
this module they are the SAME knob: choosing a budget_fraction is exactly what determines which
blocks D3's existing ranking selects). ``STRUCTURE_EDIT``/``BACKEND_CHOICE`` are documented
extension points (see :data:`ACTION_TYPE_REGISTRY`), not wired to real H3/G2/D6 machinery here --
that wiring is future work, explicitly out of scope for D5 per the roadmap item.

Reusable "brain" note (the roadmap's own phrase: D5 "shares its brain with F5 and the I1/J1 method
pickers"): :class:`LearnedController` is deliberately generic over its own state/action/reward
types (``Generic[StateT, ActionT]``, a plain ``select_action(state) -> action`` /
``update(state, action, realized_gain, realized_cost) -> None`` surface with no block-EM-specific
assumptions baked into the base class). A future F5 (backend/method-picker for some other stage)
or I1/J1 (other "which method" pickers named in the roadmap, not built here) could instantiate
:class:`BanditController` or :class:`DesignModelController` directly against THEIR OWN state/action
types by subclassing :class:`LearnedController` the same way this module's two concrete
controllers do -- reusing the bandit/DesignModel wiring pattern without needing block-EM's
``ControllerState``/``ControllerAction`` dataclasses at all. Nothing here builds F5/I1/J1
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

import numpy as np

from mixle.task.bandit import UCB1, ThompsonGaussian
from mixle.task.edge import DesignModel

__all__ = [
    "ACTION_TYPE_REGISTRY",
    "ActionType",
    "BanditController",
    "ControllerAction",
    "ControllerState",
    "DesignModelController",
    "LearnedController",
    "STATE_FEATURE_DIM",
]

# The fixed-length feature vector every ControllerState carries (see ControllerState.as_vector):
# (n_eligible, mean_residual, mean_q_gain, mean_cost, score_spread).
STATE_FEATURE_DIM = 5

_DEFAULT_BUDGET_LEVELS = (0.15, 0.3, 0.5, 0.7, 0.85, 1.0)


class ActionType(StrEnum):
    """The D5 action-type registry (see module docstring): a registry, not a closed enum -- future
    track items are expected to add new members and new :class:`LearnedController` subclasses that
    consume them, without needing to touch this module's two REAL, implemented action types."""

    BLOCK_SELECTION = "block_selection"
    BUDGET_ALLOCATION = "budget_allocation"
    STRUCTURE_EDIT = "structure_edit"
    BACKEND_CHOICE = "backend_choice"


ACTION_TYPE_REGISTRY: dict[ActionType, str] = {
    ActionType.BLOCK_SELECTION: (
        "IMPLEMENTED here. Realized as a derived consequence of BUDGET_ALLOCATION: the controller "
        "picks a budget_fraction and mixle.inference.block_em._select_active's existing "
        "gain-per-cost ranking turns that budget into an actual set of active block indices -- so "
        "block selection and budget allocation are one learned knob, not two."
    ),
    ActionType.BUDGET_ALLOCATION: (
        "IMPLEMENTED here. ControllerAction.budget_fraction, chosen per round by BanditController "
        "(a discretized arm) or DesignModelController (a continuous DesignModel proposal)."
    ),
    ActionType.STRUCTURE_EDIT: (
        "EXTENSION POINT, not implemented here. A future action would carry a payload describing "
        "an evolve-op / learn_structure call / G2 projection (H3's structure-edit schedule, not "
        "yet built). Wire it by adding a new ControllerAction.payload schema and a new "
        "LearnedController subclass (or a new arm/action space on an existing one) that reads "
        "H3-shaped state and emits this action type -- see the module docstring's 'reusable brain' "
        "note."
    ),
    ActionType.BACKEND_CHOICE: (
        "EXTENSION POINT, not implemented here. A future action would carry a payload shaped like "
        "D6's RespecializationDecision (origin/backend-respecialization: action / estimated_cost / "
        "estimated_benefit / net_benefit fields, already documented there as 'the compile "
        "economics exposed to D5'). Wiring it means feeding those fields into ControllerState and "
        "emitting a BACKEND_CHOICE ControllerAction that mixle.inference.backend_respecialization "
        "would apply -- not done here since D6 is a sibling branch, not an ancestor, of this one."
    ),
}


@dataclass(frozen=True)
class ControllerState:
    """One round's controller-visible state aggregated over the eligible blocks.

    The observed Q-gain and structural-cost receipts are aggregated to a fixed-size vector (one
    row per ROUND, not per node) because the eligible block set can change size round to round
    (freezing, zero-weight collapse) while both the bandit arm space and DesignModel's fingerprint
    need a fixed dimension.
    """

    round_index: int
    n_eligible: int
    mean_residual: float
    mean_q_gain: float
    mean_cost: float
    score_spread: float

    def as_vector(self) -> tuple[float, float, float, float, float]:
        """The fixed-length (:data:`STATE_FEATURE_DIM`) feature vector, e.g. for a DesignModel
        fingerprint or any future contextual (LinUCB-style) bandit."""
        return (
            float(self.n_eligible),
            float(self.mean_residual),
            float(self.mean_q_gain),
            float(self.mean_cost),
            float(self.score_spread),
        )

    @classmethod
    def from_scores(
        cls,
        round_index: int,
        eligible: list[int],
        gain_per_cost: dict[int, float],
        cost: dict[int, float],
        residual: dict[int, float],
        q_gain: dict[int, float],
    ) -> ControllerState:
        """Build a :class:`ControllerState` from the same per-block dicts D3's greedy scheduler
        already computes each round (:func:`mixle.inference.block_em._block_scores`) -- no
        separate feature-extraction pass over the tree is needed."""
        if not eligible:
            return cls(
                round_index=round_index,
                n_eligible=0,
                mean_residual=0.0,
                mean_q_gain=0.0,
                mean_cost=0.0,
                score_spread=0.0,
            )
        residuals = [residual.get(i, 0.0) for i in eligible]
        gains = [q_gain.get(i, 0.0) for i in eligible]
        costs = [cost.get(i, 0.0) for i in eligible]
        scores = [gain_per_cost.get(i, 0.0) for i in eligible]
        return cls(
            round_index=round_index,
            n_eligible=len(eligible),
            mean_residual=float(np.nanmean(residuals)) if residuals else 0.0,
            mean_q_gain=float(np.nanmean(gains)) if gains else 0.0,
            mean_cost=float(np.mean(costs)) if costs else 0.0,
            score_spread=float(max(scores) - min(scores)) if scores else 0.0,
        )


@dataclass(frozen=True)
class ControllerAction:
    """One controller decision for the current round.

    ``budget_fraction`` is the real, implemented knob (see :data:`ACTION_TYPE_REGISTRY`);
    ``payload`` is reserved for future ``STRUCTURE_EDIT``/``BACKEND_CHOICE`` action data and is
    unused by both controllers in this module.
    """

    action_type: ActionType
    budget_fraction: float
    payload: dict = field(default_factory=dict)


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class LearnedController(Generic[StateT, ActionT]):
    """Generic state -> action learned scheduling policy (D5).

    A drop-in alternative to a hand-written greedy heuristic: anywhere a scheduler currently reads
    some per-round state and applies a FIXED rule to pick an action, a ``LearnedController`` can
    read the same state and apply a TRAINED rule instead, updating from the REALIZED outcome after
    the fact. See the module docstring's "reusable brain" note -- this base class carries no
    block-EM-specific assumptions, so a future F5/I1/J1 method-picker item could subclass it
    directly for its own ``StateT``/``ActionT``.
    """

    def select_action(self, state: StateT) -> ActionT:
        """Return this round's action given ``state``."""
        raise NotImplementedError

    def update(self, state: StateT, action: ActionT, realized_gain: float, realized_cost: float) -> None:
        """Feed back the REALIZED gain/cost of ``action`` taken in ``state`` -- the online-learning
        signal every concrete controller trains from."""
        raise NotImplementedError


class BanditController(LearnedController[ControllerState, ControllerAction]):
    """Online contextual-free multi-armed bandit over discretized ``budget_fraction`` levels.

    Reuses :mod:`mixle.task.bandit` (:class:`~mixle.task.bandit.UCB1` by default, or
    :class:`~mixle.task.bandit.ThompsonGaussian`) rather than reimplementing a bandit algorithm.
    Needs NO logged/offline data: it can start learning cold, round 1 of the very first fit it
    sees, from nothing but realized gain/cost -- the "online bandits ... no offline training data
    needed" mode from the roadmap item. The state (:class:`ControllerState`) is accepted by
    :meth:`select_action`/:meth:`update` for interface symmetry with :class:`DesignModelController`
    and any future contextual variant, but neither UCB1 nor ThompsonGaussian actually conditions on
    it (both are the classic context-FREE multi-armed setting) -- a genuinely contextual variant
    (e.g. LinUCB keyed on :meth:`ControllerState.as_vector`) is a natural extension of this class
    that this module does not need to build to satisfy "online bandits ... given the state is a
    real feature vector" (the DesignModel mode already covers the contextual case, see its
    docstring).
    """

    def __init__(
        self,
        *,
        budget_levels: tuple[float, ...] = _DEFAULT_BUDGET_LEVELS,
        algorithm: str = "ucb1",
        ucb_c: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.budget_levels = tuple(sorted(set(float(b) for b in budget_levels)))
        if len(self.budget_levels) < 2:
            raise ValueError("BanditController needs at least two distinct budget_levels.")
        self.algorithm = algorithm
        if algorithm == "ucb1":
            self.bandit = UCB1(len(self.budget_levels), c=ucb_c, seed=seed)
        elif algorithm == "thompson":
            self.bandit = ThompsonGaussian(len(self.budget_levels), seed=seed)
        else:
            raise ValueError("algorithm must be 'ucb1' or 'thompson', got %r." % (algorithm,))

    def select_action(self, state: ControllerState) -> ControllerAction:
        arm = self.bandit.select()
        return ControllerAction(
            action_type=ActionType.BUDGET_ALLOCATION,
            budget_fraction=self.budget_levels[arm],
            payload={"arm": arm},
        )

    def update(
        self, state: ControllerState, action: ControllerAction, realized_gain: float, realized_cost: float
    ) -> None:
        arm = action.payload.get("arm")
        if arm is None:
            arm = self.budget_levels.index(action.budget_fraction)
        reward = float(realized_gain) / max(float(realized_cost), 1.0e-12)
        self.bandit.update(int(arm), reward)


class DesignModelController(LearnedController[ControllerState, ControllerAction]):
    """Offline controller wrapping :class:`mixle.task.edge.DesignModel`.

    Fit on logged ``(state, action, gain, cost)`` tuples collected across MANY prior fits (pass a
    pre-populated ``design=`` to warm-start from history, or let one accumulate rows via repeated
    :meth:`update` calls across several fits before relying on :meth:`select_action`). Treats
    ``budget_fraction`` as a 1-D continuous design point and :meth:`ControllerState.as_vector` as
    the DesignModel fingerprint -- DesignModel's own cross-task warm-start machinery (see
    :mod:`mixle.task.edge`) is exactly "condition the proposal on which task this is", which is
    what makes this mode work on a NEW, held-out fit problem's state vector with no online
    exploration of that new problem at all, given a DesignModel already trained on OTHER problems.

    Before at least two logged rows exist, :meth:`select_action` cannot fit a GP and falls back to
    ``default_budget`` (an honest cold-start, not a fabricated proposal).
    """

    def __init__(
        self,
        *,
        bounds: tuple[float, float] = (0.05, 1.0),
        default_budget: float = 0.5,
        design: DesignModel | None = None,
        seed: int | None = None,
    ) -> None:
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.default_budget = float(default_budget)
        self.design = (
            design
            if design is not None
            else DesignModel(signature="d5-budget-controller", n_constraints=0, n_fingerprint=STATE_FEATURE_DIM)
        )
        self.seed = seed

    def select_action(self, state: ControllerState) -> ControllerAction:
        fingerprint = state.as_vector()
        if len(self.design) < 2:
            return ControllerAction(action_type=ActionType.BUDGET_ALLOCATION, budget_fraction=self.default_budget)
        point = self.design.propose([self.bounds], seed=self.seed, fingerprint=fingerprint)
        budget = float(np.clip(point[0], self.bounds[0], self.bounds[1]))
        return ControllerAction(action_type=ActionType.BUDGET_ALLOCATION, budget_fraction=budget)

    def update(
        self, state: ControllerState, action: ControllerAction, realized_gain: float, realized_cost: float
    ) -> None:
        reward = float(realized_gain) / max(float(realized_cost), 1.0e-12)
        self.design.add([action.budget_fraction], reward, [], fingerprint=list(state.as_vector()))
