"""Outcome-trained plan refinement beyond imitation.

Imitating harvested/teacher decompositions (:func:`~mixle.task.sft_plan.sft_planner`) can only reproduce
known workflows. This expert-iteration step samples candidate plans from the
current planner (:func:`~mixle.task.sft_plan.sample_plans`), verifies each with
an executable checker, and retrains the plan-writing LM on verified-successful
candidates::

    planner = sft_planner(teacher, requests, tools)              # imitation baseline
    planner, report = outcome_refine_planner(planner, tasks, verify_fn)
    report.solve_rate_before, report.solve_rate_after            # measured, not assumed

``verify_fn(task, plan) -> bool`` must be an executable or ground-truth check,
such as a :class:`~mixle.doe.oracle.VerifiableOracle` for the
plan-decomposition domain.

This module implements one propose-verify-retrain round on a synthetic
tool-world. The full expert-iteration outer loop, DPO preference learning over
plan pairs, experiment-design-as-planning, and orchestrator runtime are separate
surfaces.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from mixle.task.sft_plan import _PROMPT_SEP, GenerativePlanner, _serialize_plan, sample_plans


@dataclass
class RefinementReport:
    """Measured account of one outcome-refinement round."""

    tasks: int
    verified_gain_pairs: int  # how many new verified-successful plans entered the training signal
    solve_rate_before: float
    solve_rate_after: float
    candidate_solve_rate: float
    accepted: bool
    discovery_tasks: int
    evaluation_tasks: int


def _solved(planner: GenerativePlanner, task: str, verify_fn: Callable[[str, list[dict]], bool]) -> bool:
    plan = planner.try_plan(task)
    return plan is not None and verify_fn(task, plan)


def outcome_refine_planner(
    planner: GenerativePlanner,
    tasks: Sequence[str],
    verify_fn: Callable[[str, list[dict]], bool],
    *,
    k: int = 5,
    temperature: float = 0.8,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 0,
    discovery_frac: float = 0.5,
) -> tuple[GenerativePlanner, RefinementReport]:
    """Run one propose-verify-retrain round and return the planner plus report.

    For each task: sample ``k`` candidate plans (:func:`~mixle.task.sft_plan.sample_plans`), keep the
    ones ``verify_fn`` accepts, and for tasks with at least one
    verified success -- add the highest-scoring verified candidate as a new supervised-fine-tuning pair.
    Fine-tunes the LM on every such pair in one ``fit_pairs`` call. ``solve_rate_before``/``_after`` are
    measured on the same held-out ``tasks`` via the planner's own single-shot ``try_plan`` (matched
    budget), before and after the retrain -- not an aggregate over
    the k samples used to harvest the training signal.
    """
    if not callable(verify_fn):
        raise TypeError("verify_fn must be callable")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(temperature) or temperature <= 0.0 or not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("temperature and lr must be finite and positive")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(discovery_frac) or not 0.0 < discovery_frac < 1.0:
        raise ValueError("discovery_frac must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    tasks = [str(task) for task in tasks]
    if len(tasks) < 2 or len(set(tasks)) != len(tasks):
        raise ValueError("at least two unique tasks are required for independent discovery and evaluation")
    order = np.random.RandomState(seed).permutation(len(tasks))
    n_discovery = min(len(tasks) - 1, max(1, int(round(len(tasks) * discovery_frac))))
    discovery = [tasks[i] for i in order[:n_discovery]]
    evaluation = [tasks[i] for i in order[n_discovery:]]
    solved_before = sum(1 for task in evaluation if _solved(planner, task, verify_fn))
    candidate = copy.deepcopy(planner)

    new_pairs: list[tuple[list[int], list[int]]] = []
    for i, task in enumerate(discovery):
        samples = sample_plans(planner, task, n=k, temperature=temperature, seed=seed + i)
        verified = [plan for plan, _score in samples if plan is not None and verify_fn(task, plan)]
        if not verified:
            continue
        best_plan = verified[0]  # sample_plans returns highest-score-first; keep the first verified plan
        prompt = candidate.codec.encode(str(task) + _PROMPT_SEP)
        completion = candidate.codec.encode(_serialize_plan(best_plan))
        new_pairs.append((prompt, completion))

    if new_pairs:
        candidate.lm.fit_pairs(new_pairs, epochs=epochs, lr=lr, seed=seed)

    solved_after = sum(1 for task in evaluation if _solved(candidate, task, verify_fn))
    before_rate = solved_before / len(evaluation)
    candidate_rate = solved_after / len(evaluation)
    accepted = bool(new_pairs and candidate_rate >= before_rate)
    active = candidate if accepted else planner
    report = RefinementReport(
        tasks=len(tasks),
        verified_gain_pairs=len(new_pairs),
        solve_rate_before=before_rate,
        solve_rate_after=candidate_rate if accepted else before_rate,
        candidate_solve_rate=candidate_rate,
        accepted=accepted,
        discovery_tasks=len(discovery),
        evaluation_tasks=len(evaluation),
    )
    return active, report


__all__ = ["RefinementReport", "outcome_refine_planner"]
