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
from numbers import Integral

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
    """Candidate-generation and independent promotion audit for one training round."""

    round: int
    mean_score: float
    n_candidates: int
    n_kept: int
    candidates: tuple[tuple[str, ...], ...]
    candidate_scores: tuple[float, ...]
    kept_indices: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    threshold: float
    audit_seeds: tuple[int, ...]
    audit_plan_seed: int
    incumbent_audit_score: float
    proposed_audit_score: float
    promoted: bool
    proposal_error: str | None


@dataclass
class OutcomeTrainedDecomposer:
    """Outcome-trained plan model, baseline imitation model, and per-round statistics."""

    plan_model: PlanModel
    imitation_model: PlanModel  # round-0, kept for the acceptance comparison
    rounds: list[RoundStats] = field(default_factory=list)
    imitation_seeds: tuple[int, ...] = ()


def _draw_unique_seeds(rng: np.random.RandomState, count: int, used: set[int]) -> tuple[int, ...]:
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = int(rng.randint(0, 2**31 - 1))
        if candidate not in used:
            used.add(candidate)
            seeds.append(candidate)
    return tuple(seeds)


def _score_plan_panel(
    plan: list[str],
    *,
    seeds: tuple[int, ...],
    n_cells: int,
    n_targets: int,
    budget: int,
) -> float:
    return float(
        np.mean(
            [
                execute_plan(plan, n_cells=n_cells, n_targets=n_targets, budget=budget, seed=world_seed)
                for world_seed in seeds
            ]
        )
    )


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
    selection_worlds: int = 5,
    audit_worlds: int = 10,
) -> OutcomeTrainedDecomposer:
    """Train by selection on common worlds and promotion on an independent audit panel."""
    for name, value in (
        ("seed_worlds", seed_worlds),
        ("k_candidates", k_candidates),
        ("rounds", rounds),
        ("selection_worlds", selection_worlds),
        ("audit_worlds", audit_worlds),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    try:
        success_quantile = float(success_quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("success_quantile must be finite and in [0, 1]") from exc
    if not np.isfinite(success_quantile) or not 0 <= success_quantile <= 1:
        raise ValueError("success_quantile must be finite and in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if seed_worlds >= 2**31 - 1:
        raise ValueError("seed_worlds is too large for the available independent seed range")
    # Validate the shared world configuration before any training work or partial receipts exist.
    ExplorationWorld(n_cells=n_cells, n_targets=n_targets, budget=budget, seed=0)

    rng = np.random.RandomState(seed)
    used_world_seeds: set[int] = set()
    imitation_seed_base = int(rng.randint(0, 2**31 - 1 - seed_worlds))
    imitation_seeds = tuple(imitation_seed_base + index for index in range(seed_worlds))
    used_world_seeds.update(imitation_seeds)
    imitation = imitation_traces(
        greedy_prospectivity_policy,
        n_worlds=seed_worlds,
        n_cells=n_cells,
        n_targets=n_targets,
        budget=budget,
        seed_offset=imitation_seed_base,
    )
    imitation_model = fit_plan_model(_as_traces(imitation))
    model = imitation_model
    history = []
    for r in range(rounds):
        candidates = [model.sample(rng) for _ in range(k_candidates)]
        selection_seeds = _draw_unique_seeds(rng, int(selection_worlds), used_world_seeds)
        scores = [
            _score_plan_panel(
                candidate,
                seeds=selection_seeds,
                n_cells=n_cells,
                n_targets=n_targets,
                budget=budget,
            )
            for candidate in candidates
        ]
        threshold = float(np.quantile(scores, success_quantile))
        kept_indices = tuple(
            index
            for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
            if score >= threshold and score > 0 and candidate
        )
        kept = [candidates[index] for index in kept_indices]
        audit_seeds = _draw_unique_seeds(rng, int(audit_worlds), used_world_seeds)
        audit_plan_seed = int(rng.randint(0, 2**31 - 1))
        incumbent_audit_score = evaluate_plan_model(
            model,
            seeds=audit_seeds,
            n_cells=n_cells,
            n_targets=n_targets,
            budget=budget,
            rng_seed=audit_plan_seed,
        )
        proposal_error = None
        proposed_model = model
        if len(kept) >= 2:
            try:
                proposed_model = fit_plan_model(_as_traces(kept))
            except ValueError as exc:
                proposal_error = str(exc)
        proposed_audit_score = evaluate_plan_model(
            proposed_model,
            seeds=audit_seeds,
            n_cells=n_cells,
            n_targets=n_targets,
            budget=budget,
            rng_seed=audit_plan_seed,
        )
        promoted = (
            len(kept) >= 2
            and proposal_error is None
            and proposed_audit_score > incumbent_audit_score
        )
        history.append(
            RoundStats(
                round=r,
                mean_score=float(np.mean(scores)),
                n_candidates=len(candidates),
                n_kept=len(kept),
                candidates=tuple(tuple(candidate) for candidate in candidates),
                candidate_scores=tuple(float(score) for score in scores),
                kept_indices=kept_indices,
                selection_seeds=selection_seeds,
                threshold=threshold,
                audit_seeds=audit_seeds,
                audit_plan_seed=audit_plan_seed,
                incumbent_audit_score=incumbent_audit_score,
                proposed_audit_score=proposed_audit_score,
                promoted=promoted,
                proposal_error=proposal_error,
            )
        )
        if promoted:
            model = proposed_model
    return OutcomeTrainedDecomposer(
        plan_model=model,
        imitation_model=imitation_model,
        rounds=history,
        imitation_seeds=imitation_seeds,
    )


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
