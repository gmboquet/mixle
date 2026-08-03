"""A sequential-exploration world with synthetic ground truth.

This module provides a compact, dependency-free exploration environment inside
core Mixle: hidden true targets, typed actions with costs,
``step(action) -> observation`` revealing noisy evidence, budget-exhaustion
episode end, and ``score()`` equal to targets correctly identified. It is
seeded, deterministic, and small enough for fast task-policy tests.

    world = ExplorationWorld(n_cells=30, n_targets=4, budget=40, seed=0)
    obs = world.step({"type": "survey", "cell": 3})   # low-cost: sharpens that cell's prospectivity read
    obs = world.step({"type": "drill", "cell": 3})     # costly: reveals ground truth, scores if correct
    world.score()                                       # targets correctly identified so far
    world.done                                          # budget exhausted

Two baseline policies are included (random, greedy-by-prospectivity) as the sanity check that the
world has learnable signal at all -- a policy that reads the evidence should beat one that doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

SURVEY_COST = 1
DRILL_COST = 5


@dataclass
class ExplorationWorld:
    """One episode over a synthetic mineral-style exploration world: ``n_cells`` candidate sites,
    ``n_targets`` of them hidden true targets, each cell's TRUE target status correlated with a
    latent "geology" feature that a survey partially reveals as a noisy prospectivity reading."""

    n_cells: int
    n_targets: int
    budget: int
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (("n_cells", self.n_cells), ("n_targets", self.n_targets), ("budget", self.budget)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if self.n_targets < 0 or self.n_targets > self.n_cells:
            raise ValueError("n_targets must be between zero and n_cells")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        self.n_cells = int(self.n_cells)
        self.n_targets = int(self.n_targets)
        self.budget = int(self.budget)
        self.seed = int(self.seed)
        rng = np.random.RandomState(self.seed)
        self._geology = rng.normal(size=self.n_cells)
        target_idx = rng.choice(self.n_cells, size=self.n_targets, replace=False)
        self._is_target = np.zeros(self.n_cells, dtype=bool)
        self._is_target[target_idx] = True
        # a target's geology reads systematically higher, but with enough noise that a raw look is
        # only weak evidence -- surveying a cell narrows that noise, giving a real reason to explore.
        self._geology = self._geology + self._is_target * 2.0
        self._survey_noise = np.full(self.n_cells, 1.5)  # per-cell prospectivity read noise, shrinks on survey
        self._drilled = np.zeros(self.n_cells, dtype=bool)
        self._rng = rng
        self.remaining_budget = self.budget
        self.done = self.remaining_budget < min(SURVEY_COST, DRILL_COST)
        self.history: list[dict[str, Any]] = []
        self._correct_drills = 0

    def prospectivity(self, cell: int) -> float:
        """The world's own current noisy read of ``cell`` -- what a policy actually gets to see."""
        return float(self._geology[cell] + self._rng.normal(scale=self._survey_noise[cell]))

    @staticmethod
    def action_cost(action: dict[str, Any]) -> int:
        """Return an action's exact cost without changing world state."""
        if not isinstance(action, dict):
            raise ValueError("actions must be dictionaries")
        kind = action.get("type")
        if kind == "survey":
            return SURVEY_COST
        if kind == "drill":
            return DRILL_COST
        raise ValueError(f"unknown action type {kind!r}")

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply one typed action (plain dict, so a fitted plan model can score/sample over the same
        action vocabulary): ``{"type": "survey", "cell": i}`` or ``{"type": "drill", "cell": i}``.
        Returns a plain-dict observation. Raises nothing on an over-budget action -- it is simply
        refused (recorded, zero effect) once ``done``, so a policy that keeps acting past budget
        exhaustion degrades gracefully rather than crashing."""
        if not isinstance(action, dict):
            return {"type": None, "cell": None, "accepted": False, "reason": "invalid_action"}
        kind = action.get("type")
        raw_cell = action.get("cell")
        # MXR-080-1895: this was ``int(raw_cell)``, which is a coercion, not a check. ``cell="3"`` and
        # ``cell=3.7`` were both ACCEPTED -- the second silently truncating to a different cell than the
        # caller named, and both spending real budget on an action the world never validated. Exactness
        # matches the constructor's own rule for every other integer field. ``Integral`` still admits
        # numpy integers, which is what a learned policy's ``argmax`` produces. Refusal stays graceful
        # (a recorded, zero-effect observation) rather than raising: that is this method's declared
        # contract for a bad action, and a policy that emits one should degrade, not crash.
        if isinstance(raw_cell, bool) or not isinstance(raw_cell, Integral):
            cell = -1
        else:
            cell = int(raw_cell)
        if self.done or not (0 <= cell < self.n_cells):
            obs = {"type": kind, "cell": cell, "accepted": False, "reason": "done_or_invalid_cell"}
            self.history.append(obs)
            return obs

        try:
            cost = self.action_cost(action)
        except ValueError:
            obs = {"type": kind, "cell": cell, "accepted": False, "reason": "unknown_action_or_over_budget"}
            self.history.append(obs)
            self.done = self.remaining_budget < min(SURVEY_COST, DRILL_COST)
            return obs
        if cost > self.remaining_budget:
            obs = {"type": kind, "cell": cell, "accepted": False, "reason": "unknown_action_or_over_budget"}
            self.history.append(obs)
            self.done = self.remaining_budget < min(SURVEY_COST, DRILL_COST)
            return obs

        self.remaining_budget -= cost
        if kind == "survey":
            self._survey_noise[cell] = max(0.2, self._survey_noise[cell] * 0.4)
            obs = {"type": "survey", "cell": cell, "accepted": True, "prospectivity": self.prospectivity(cell)}
        else:
            is_target = bool(self._is_target[cell])
            if is_target and not self._drilled[cell]:
                self._correct_drills += 1
            self._drilled[cell] = True
            obs = {"type": "drill", "cell": cell, "accepted": True, "is_target": is_target}

        self.done = self.remaining_budget < min(SURVEY_COST, DRILL_COST)
        self.history.append(obs)
        return obs

    def score(self) -> int:
        """Targets correctly identified so far: distinct true-target cells actually drilled."""
        return self._correct_drills

    def action_menu(self) -> list[dict[str, Any]]:
        """Every currently affordable action (undrilled cells only for drills)."""
        if self.done:
            return []
        surveys = (
            [{"type": "survey", "cell": c} for c in range(self.n_cells)] if self.remaining_budget >= SURVEY_COST else []
        )
        drills = (
            [{"type": "drill", "cell": c} for c in range(self.n_cells) if not self._drilled[c]]
            if self.remaining_budget >= DRILL_COST
            else []
        )
        return surveys + drills


@dataclass
class EpisodeResult:
    """Score, action count, and trace captured from one exploration episode."""

    score: int
    n_actions: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    n_attempts: int = 0
    stop_reason: str = "unknown"


def run_episode(
    policy,
    *,
    n_cells: int,
    n_targets: int,
    budget: int,
    seed: int,
    max_steps: int = 10_000,
) -> EpisodeResult:
    """Drive ``policy(world) -> action`` (a plain dict, or ``None`` to end early) until the world's
    budget is exhausted, the policy stops itself, an action is refused, or ``max_steps`` is
    reached. The finite cap protects callers from non-progressing custom policies."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    world = ExplorationWorld(n_cells=n_cells, n_targets=n_targets, budget=budget, seed=seed)
    n_actions = 0
    n_attempts = 0
    stop_reason = "environment_done" if world.done else "running"
    while not world.done and n_attempts < max_steps:
        action = policy(world)
        if action is None:
            stop_reason = "policy_stopped"
            break
        obs = world.step(action)
        n_attempts += 1
        if not obs.get("accepted", False):
            stop_reason = "action_refused"
            break
        n_actions += 1
    else:
        stop_reason = "environment_done" if world.done else "step_limit"
    return EpisodeResult(
        score=world.score(),
        n_actions=n_actions,
        trace=world.history,
        n_attempts=n_attempts,
        stop_reason=stop_reason,
    )


def random_policy(world: ExplorationWorld) -> dict[str, Any] | None:
    """Choose a random currently valid action from the world's action menu."""
    menu = world.action_menu()
    if not menu:
        return None
    idx = world._rng.randint(0, len(menu))
    return menu[idx]


_SURVEYED_ENOUGH_THRESHOLD = 0.65  # comfortably above one survey's exact result (1.5 * 0.4 == 0.6000...1
# in float64) so the boundary check never flips on rounding -- and comfortably below the post-survey
# result (0.24), so it never falsely treats an unsurveyed cell as surveyed.


def greedy_prospectivity_policy(world: ExplorationWorld) -> dict[str, Any] | None:
    """Survey every undrilled cell once (low-cost information), then drill highest-read-prospectivity
    cells first -- a fixed heuristic baseline for learned or diagnosis-directed policies."""
    undrilled = [c for c in range(world.n_cells) if not world._drilled[c]]
    if not undrilled:
        return None
    surveyed_enough = all(world._survey_noise[c] <= _SURVEYED_ENOUGH_THRESHOLD for c in undrilled)
    if not surveyed_enough:
        target = next(c for c in undrilled if world._survey_noise[c] > _SURVEYED_ENOUGH_THRESHOLD)
        return {"type": "survey", "cell": target}
    best = max(undrilled, key=world.prospectivity)
    return {"type": "drill", "cell": best}


__all__ = [
    "DRILL_COST",
    "SURVEY_COST",
    "EpisodeResult",
    "ExplorationWorld",
    "greedy_prospectivity_policy",
    "random_policy",
    "run_episode",
]
