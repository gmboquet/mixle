"""Search a student recipe with ``mixle.doe`` -- find a small model that matches the teacher for the least compute.

Distillation has knobs (feature width, hidden size, epochs, learning rate) that trade fidelity against training
cost. Rather than grid-search them, :func:`tune_recipe` runs GP Bayesian optimization (``mixle.doe.minimize``)
over the recipe space, distilling and scoring a handful of candidates and homing in on the best. The objective
is held-out **agreement** with the teacher, optionally minus a compute penalty (``cost_weight``) so the search
prefers the *cheapest* recipe that still matches -- the "minimize train time" pillar made concrete. Returns the
re-distilled winner as a callable :class:`~mixle.task.model.TaskModel` plus the full search history.

The recipe space is a few interpretable axes with sensible defaults; override ``space`` to widen or pin them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.distill import agreement, distill
from mixle.task.model import TaskModel


@dataclass
class RecipeSpace:
    """The tunable axes of a distillation recipe and how a unit-cube point decodes into concrete knobs."""

    dim_choices: Sequence[int] = (128, 256, 512, 1024)
    hidden_range: tuple[int, int] = (16, 128)
    epochs_range: tuple[int, int] = (50, 400)
    log10_lr_range: tuple[float, float] = (-3.0, -1.0)
    n: int = 4  # n-gram order is fixed by default (cheap to pin; widen via a custom space if needed)

    def dims(self) -> int:
        return 4

    def decode(self, point: np.ndarray) -> dict[str, Any]:
        p = np.clip(np.asarray(point, dtype=np.float64), 0.0, 1.0)
        dim = int(self.dim_choices[min(len(self.dim_choices) - 1, int(p[0] * len(self.dim_choices)))])
        hidden = int(round(self.hidden_range[0] + p[1] * (self.hidden_range[1] - self.hidden_range[0])))
        epochs = int(round(self.epochs_range[0] + p[2] * (self.epochs_range[1] - self.epochs_range[0])))
        lr = float(10.0 ** (self.log10_lr_range[0] + p[3] * (self.log10_lr_range[1] - self.log10_lr_range[0])))
        return {"n": self.n, "dim": dim, "hidden": [hidden], "epochs": epochs, "lr": lr}

    def cost(self, recipe: dict[str, Any]) -> float:
        """Relative training cost of a recipe in [0, 1] (params x steps, normalized by the space's max)."""
        hi = self.dim_choices[-1] * self.hidden_range[1] * self.epochs_range[1]
        c = recipe["dim"] * recipe["hidden"][0] * recipe["epochs"]
        return float(c) / float(hi)

    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.dims()


@dataclass
class TuneResult:
    """The outcome of a recipe search: the winning model, its recipe and scores, and the full BO history."""

    model: TaskModel
    recipe: dict[str, Any]
    agreement: float
    score: float
    cost: float
    history: Any = field(default=None)


def tune_recipe(
    teacher: Callable[..., Any],
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    space: RecipeSpace | None = None,
    n_init: int = 4,
    n_iter: int = 8,
    cost_weight: float = 0.0,
    seed: int = 0,
    task: str = "",
) -> TuneResult:
    """Bayesian-optimize the distillation recipe; return the best re-distilled :class:`TaskModel`.

    Maximizes held-out ``agreement(student, teacher, val_texts)`` minus ``cost_weight * relative_train_cost``.
    Set ``cost_weight > 0`` to prefer the cheapest recipe that still matches the teacher. ``teacher`` is called
    once per candidate on ``val_texts`` (cached across the search) and once per candidate on ``train_texts``.
    """
    from mixle.doe import minimize

    space = space or RecipeSpace()
    train_texts = [str(t) for t in train_texts]
    val_texts = [str(t) for t in val_texts]
    val_truth = _teacher_labels(teacher, val_texts)

    trials: list[dict[str, Any]] = []

    def objective(point: np.ndarray) -> float:
        recipe = space.decode(point)
        student = distill(teacher, train_texts, labels=labels, seed=seed, task=task, **recipe)
        agree = agreement(student, val_truth, val_texts)
        cost = space.cost(recipe)
        score = agree - cost_weight * cost
        trials.append({"recipe": recipe, "agreement": agree, "cost": cost, "score": score, "model": student})
        return score

    result = minimize(objective, space.bounds(), n_init=n_init, n_iter=n_iter, seed=seed, maximize=True)
    best = max(trials, key=lambda t: t["score"])
    return TuneResult(
        model=best["model"],
        recipe=best["recipe"],
        agreement=best["agreement"],
        score=best["score"],
        cost=best["cost"],
        history=result,
    )


def _teacher_labels(teacher: Callable[..., Any], texts: list[str]) -> list[Any]:
    out = teacher(texts)
    if isinstance(out, (list, tuple)) and len(out) == len(texts):
        return list(out)
    return [teacher(t) for t in texts]
