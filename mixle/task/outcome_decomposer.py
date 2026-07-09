"""Outcome-trained decomposer for exploration plans.

Candidate plans are proposed by sampling a fitted :class:`~mixle.task.plan_model.PlanModel`, executing
them in the :mod:`~mixle.task.explore_world`
    world, keep verifiably successful traces (score above a quantile of that round's own scores), refit
    the plan model on successes, iterate a few rounds. Training signal is only world score -- verifiable
by construction, never a proxy or a teacher's opinion.

    decomposer = train_outcome_decomposer(seed_worlds=40, n_cells=20, n_targets=3, budget=30)
    decomposer.plan_model.sample(rng)          # a plan shaped by what actually worked, not just imitation
    evaluate_decomposer(decomposer, ...)       # mean score on held-out seeds

For a useful deployment, compare the outcome-refit model with both the imitation-only model (round 0,
before any outcome refitting) and the greedy heuristic on held-out world seeds at matched budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mixle.task.explore_world import ExplorationWorld, greedy_prospectivity_policy, run_episode
from mixle.task.plan_model import PlanModel, fit_plan_model
from mixle.task.traces import AgentTrace


def _as_traces(type_sequences: list[list[str]]) -> list[AgentTrace]:
    return [AgentTrace(request="", plan=[{"tool": t} for t in seq]) for seq in type_sequences]


def imitation_traces(policy, *, n_worlds: int, n_cells: int, n_targets: int, budget: int, seed_offset: int = 0):
    """Run ``policy`` over ``n_worlds`` seeded episodes and return each episode's ACCEPTED action-type
    sequence used to fit the round-0 imitation model."""
    out = []
    for i in range(n_worlds):
        result = run_episode(policy, n_cells=n_cells, n_targets=n_targets, budget=budget, seed=seed_offset + i)
        out.append([step["type"] for step in result.trace if step.get("accepted")])
    return out


def execute_plan(plan_types: list[str], *, n_cells: int, n_targets: int, budget: int, seed: int) -> int:
    """Execute a plan (a sequence of action types, e.g. ``["survey", "survey", "drill", ...]``) in a
    fresh seeded world: at each step, "survey" targets the undrilled cell with the noisiest current
    read (most to gain), "drill" targets the undrilled cell with the highest current prospectivity
    read -- the plan model decides the order and mix of action types; this fixed rule decides which cell,
    the same division of labor the plan/tool-name abstraction uses everywhere else in this plan.
    Returns the world's final score."""
    world = ExplorationWorld(n_cells=n_cells, n_targets=n_targets, budget=budget, seed=seed)
    for kind in plan_types:
        if world.done:
            break
        undrilled = [c for c in range(world.n_cells) if not world._drilled[c]]
        if not undrilled:
            break
        if kind == "survey":
            cell = max(undrilled, key=lambda c: world._survey_noise[c])
        elif kind == "drill":
            cell = max(undrilled, key=world.prospectivity)
        else:
            continue
        world.step({"type": kind, "cell": cell})
    return world.score()


@dataclass
class RoundStats:
    """Candidate-generation statistics for one outcome-decomposition round."""

    round: int
    mean_score: float
    n_candidates: int
    n_kept: int


@dataclass
class OutcomeTrainedDecomposer:
    """Outcome-trained plan model, baseline imitation model, and per-round statistics."""

    plan_model: PlanModel
    imitation_model: PlanModel  # round-0, kept for the acceptance comparison
    rounds: list[RoundStats] = field(default_factory=list)


def train_outcome_decomposer(
    *,
    seed_worlds: int,
    n_cells: int,
    n_targets: int,
    budget: int,
    k_candidates: int = 30,
    success_quantile: float = 0.6,
    rounds: int = 3,
    seed: int = 0,
) -> OutcomeTrainedDecomposer:
    """Train a plan model by sampling, executing, keeping high-outcome plans, and refitting."""
    rng = np.random.RandomState(seed)
    imitation = imitation_traces(
        greedy_prospectivity_policy, n_worlds=seed_worlds, n_cells=n_cells, n_targets=n_targets, budget=budget
    )
    imitation_model = fit_plan_model(_as_traces(imitation))
    model = imitation_model
    history = []
    for r in range(rounds):
        candidates = [model.sample(rng) for _ in range(k_candidates)]
        scores = [
            execute_plan(c, n_cells=n_cells, n_targets=n_targets, budget=budget, seed=int(rng.randint(0, 2**31 - 1)))
            for c in candidates
        ]
        threshold = float(np.quantile(scores, success_quantile)) if scores else 0.0
        kept = [c for c, s in zip(candidates, scores) if s >= threshold and s > 0 and c]
        history.append(
            RoundStats(round=r, mean_score=float(np.mean(scores)), n_candidates=len(candidates), n_kept=len(kept))
        )
        if kept:
            model = fit_plan_model(_as_traces(kept))
    return OutcomeTrainedDecomposer(plan_model=model, imitation_model=imitation_model, rounds=history)


def evaluate_plan_model(
    model: PlanModel, *, seeds, n_cells: int, n_targets: int, budget: int, rng_seed: int = 0
) -> float:
    """Mean world score of ``model``'s sampled plan, executed once per held-out seed."""
    rng = np.random.RandomState(rng_seed)
    scores = []
    for s in seeds:
        plan = model.sample(rng)
        scores.append(execute_plan(plan, n_cells=n_cells, n_targets=n_targets, budget=budget, seed=s))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_greedy_heuristic(*, seeds, n_cells: int, n_targets: int, budget: int) -> float:
    """Return the mean score of the built-in greedy policy across held-out seeds."""
    scores = [
        run_episode(greedy_prospectivity_policy, n_cells=n_cells, n_targets=n_targets, budget=budget, seed=s).score
        for s in seeds
    ]
    return float(np.mean(scores)) if scores else 0.0


__all__ = [
    "OutcomeTrainedDecomposer",
    "RoundStats",
    "evaluate_greedy_heuristic",
    "evaluate_plan_model",
    "execute_plan",
    "imitation_traces",
    "train_outcome_decomposer",
]
