"""Search a student recipe with ``mixle.doe`` -- find a compact model that matches the teacher for the least compute.

Distillation has knobs (feature width, hidden size, epochs, learning rate) that trade fidelity against training
cost. Rather than grid-search them, :func:`tune_recipe` runs GP Bayesian optimization (``mixle.doe.minimize``)
over the recipe space, distilling and scoring a handful of candidates and homing in on the best. The objective
is held-out **agreement** with the teacher, optionally minus a compute penalty (``cost_weight``) so the search
prefers the lowest-cost recipe that still matches. Returns the
re-distilled winner as a callable :class:`~mixle.task.model.TaskModel` plus the full search history.

The recipe space is a few interpretable axes with sensible defaults; override ``space`` to widen or pin them.

``tune_recipe_for_routing`` is the routing-ready sibling: it runs the same search, then calibrates the winning
recipe into a :class:`~mixle.task.calibrate.CalibratedTaskModel` on data the search never touched -- so a task
gets an automatically right-sized model (search picks the complexity) that is *also* immediately ``decide()``-able
for :class:`~mixle.task.cascade.Cascade` / :class:`~mixle.task.router.Router`, with no separate calibration step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.distill import _fit_density_gate, _split_for_calibration, agreement, distill
from mixle.task.model import TaskModel


@dataclass
class RecipeSpace:
    """The tunable axes of a distillation recipe and how a unit-cube point decodes into concrete knobs."""

    dim_choices: Sequence[int] = (128, 256, 512, 1024)
    hidden_range: tuple[int, int] = (16, 128)
    epochs_range: tuple[int, int] = (50, 400)
    log10_lr_range: tuple[float, float] = (-3.0, -1.0)
    n: int = 4  # n-gram order is fixed by default; widen via a custom space if needed

    def __post_init__(self) -> None:
        choices = tuple(self.dim_choices)
        if (
            not choices
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in choices)
            or tuple(sorted(set(int(value) for value in choices))) != tuple(int(value) for value in choices)
        ):
            raise ValueError("dim_choices must be unique positive integers in increasing order")
        for name, bounds in (
            ("hidden_range", self.hidden_range),
            ("epochs_range", self.epochs_range),
        ):
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must be an increasing pair of positive integers")
        if (
            not isinstance(self.log10_lr_range, tuple)
            or len(self.log10_lr_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
                for value in self.log10_lr_range
            )
            or self.log10_lr_range[0] > self.log10_lr_range[1]
        ):
            raise ValueError("log10_lr_range must be an increasing pair of finite numbers")
        if isinstance(self.n, bool) or not isinstance(self.n, Integral) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        self.dim_choices = choices
        self.hidden_range = (int(self.hidden_range[0]), int(self.hidden_range[1]))
        self.epochs_range = (int(self.epochs_range[0]), int(self.epochs_range[1]))
        self.log10_lr_range = (
            float(self.log10_lr_range[0]),
            float(self.log10_lr_range[1]),
        )
        self.n = int(self.n)

    def dims(self) -> int:
        """Return the normalized recipe-search dimensionality."""
        return 4

    def decode(self, point: np.ndarray) -> dict[str, Any]:
        """Decode a normalized design point into a distillation recipe."""
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (self.dims(),) or not np.all(np.isfinite(p)):
            raise ValueError(f"recipe point must have shape ({self.dims()},) with finite values")
        p = np.clip(p, 0.0, 1.0)
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
        """Return normalized DOE bounds for recipe search."""
        return [(0.0, 1.0)] * self.dims()


@dataclass
class TuneResult:
    """The outcome of a recipe search: the winning model, its recipe and scores, and the full BO history."""

    model: TaskModel
    recipe: dict[str, Any]
    agreement: float
    score: float
    cost: float
    selection_agreement: float
    selection_score: float
    test_size: int
    history: Any = field(default=None)


def _validated_count(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _validated_cost_weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("cost_weight must be a finite non-negative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("cost_weight must be a finite non-negative number")
    return result


def _split_search_test(
    texts: list[str],
    *,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if (
        isinstance(test_frac, bool)
        or not isinstance(test_frac, Real)
        or not np.isfinite(test_frac)
        or not 0.0 < float(test_frac) < 1.0
    ):
        raise ValueError("test_frac must be a finite fraction strictly between 0 and 1")
    if len(texts) < 4:
        raise ValueError("val_texts needs at least four rows for disjoint search and test panels")
    n_test = max(2, int(round(len(texts) * float(test_frac))))
    if n_test >= len(texts):
        raise ValueError("test_frac leaves no recipe-search rows")
    order = np.random.RandomState(seed).permutation(len(texts))
    test_indices = order[:n_test]
    search_indices = order[n_test:]
    return (
        [texts[int(index)] for index in search_indices],
        [texts[int(index)] for index in test_indices],
    )


def tune_recipe(
    teacher: Callable[..., Any],
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    *,
    test_texts: Sequence[str] | None = None,
    test_frac: float = 0.25,
    labels: Sequence[str] | None = None,
    space: RecipeSpace | None = None,
    n_init: int = 4,
    n_iter: int = 8,
    cost_weight: float = 0.0,
    seed: int = 0,
    task: str = "",
) -> TuneResult:
    """Bayesian-optimize a recipe and independently evaluate the selected artifact.

    Candidate selection maximizes agreement on the search panel minus
    ``cost_weight * relative_train_cost``. ``agreement``/``score`` in the
    returned result are measured once on an untouched test panel;
    ``selection_agreement``/``selection_score`` retain the adaptive search
    objective. If ``test_texts`` is omitted, ``val_texts`` is deterministically
    split into disjoint search and test panels.
    """
    from mixle.doe import minimize

    if not callable(teacher):
        raise TypeError("teacher must be callable")
    n_init = _validated_count(n_init, "n_init", minimum=1)
    n_iter = _validated_count(n_iter, "n_iter", minimum=0)
    cost_weight = _validated_cost_weight(cost_weight)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    seed = int(seed)
    if space is None:
        space = RecipeSpace()
    if not isinstance(space, RecipeSpace):
        raise TypeError("space must be a RecipeSpace")
    if isinstance(train_texts, (str, bytes)) or isinstance(val_texts, (str, bytes)):
        raise TypeError("train_texts and val_texts must be sequences of strings")
    train_texts = list(train_texts)
    val_texts = list(val_texts)
    if not train_texts or any(not isinstance(text, str) or not text for text in train_texts):
        raise ValueError("train_texts must contain non-empty strings")
    if any(not isinstance(text, str) or not text for text in val_texts):
        raise ValueError("val_texts must contain non-empty strings")
    if test_texts is None:
        search_texts, independent_texts = _split_search_test(
            val_texts,
            test_frac=test_frac,
            seed=seed,
        )
    else:
        if isinstance(test_texts, (str, bytes)):
            raise TypeError("test_texts must be a sequence of strings")
        search_texts = val_texts
        independent_texts = list(test_texts)
        if not search_texts:
            raise ValueError("val_texts must be non-empty")
        if not independent_texts or any(not isinstance(text, str) or not text for text in independent_texts):
            raise ValueError("test_texts must contain non-empty strings")
    search_truth = _teacher_labels(teacher, search_texts)
    independent_truth = _teacher_labels(teacher, independent_texts)

    trials: list[dict[str, Any]] = []

    def objective(point: np.ndarray) -> float:
        recipe = space.decode(point)
        student = distill(teacher, train_texts, labels=labels, seed=seed, task=task, **recipe)
        agree = agreement(student, search_truth, search_texts)
        cost = space.cost(recipe)
        score = agree - cost_weight * cost
        if (
            not np.isfinite(agree)
            or not 0.0 <= agree <= 1.0
            or not np.isfinite(cost)
            or cost < 0.0
            or not np.isfinite(score)
        ):
            raise ValueError("recipe trial produced invalid agreement, cost, or score")
        trials.append({"recipe": recipe, "agreement": agree, "cost": cost, "score": score, "model": student})
        return score

    result = minimize(objective, space.bounds(), n_init=n_init, n_iter=n_iter, seed=seed, maximize=True)
    if not trials:
        raise RuntimeError("recipe search completed without evaluating any trials")
    best = max(trials, key=lambda t: t["score"])
    test_agreement = agreement(best["model"], independent_truth, independent_texts)
    if not np.isfinite(test_agreement) or not 0.0 <= test_agreement <= 1.0:
        raise ValueError("selected recipe produced invalid independent agreement")
    test_score = test_agreement - cost_weight * best["cost"]
    return TuneResult(
        model=best["model"],
        recipe=best["recipe"],
        agreement=test_agreement,
        score=test_score,
        cost=best["cost"],
        selection_agreement=best["agreement"],
        selection_score=best["score"],
        test_size=len(independent_texts),
        history=result,
    )


def _teacher_labels(teacher: Callable[..., Any], texts: list[str]) -> list[Any]:
    out = teacher(texts)
    if isinstance(out, (str, bytes)) or not isinstance(out, Sequence):
        raise ValueError("teacher must return one batched label sequence")
    labels = list(out)
    if len(labels) != len(texts):
        raise ValueError("teacher must return exactly one label per input")
    return labels


def _teacher_from_cache(known: dict[str, Any], teacher: Callable[..., Any]) -> Callable[[list[str]], list[Any]]:
    """A teacher wrapper answering from ``known`` (text -> label) first, so previously-labeled text is never
    re-queried -- the teacher is assumed a deterministic function of the text, same as everywhere else in
    distillation, so caching by text content changes no result, only how many real teacher calls it costs."""

    def wrapped(texts: list[str]) -> list[Any]:
        misses = [t for t in texts if t not in known]
        if misses:
            known.update(zip(misses, _teacher_labels(teacher, misses)))
        return [known[t] for t in texts]

    return wrapped


@dataclass
class CalibratedTuneResult:
    """The outcome of a routing-ready recipe search: the calibrated winner, its recipe and scores, and history."""

    model: CalibratedTaskModel
    recipe: dict[str, Any]
    agreement: float
    score: float
    cost: float
    selection_agreement: float
    selection_score: float
    test_size: int
    history: Any = field(default=None)


def tune_recipe_for_routing(
    teacher: Callable[..., Any],
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    space: RecipeSpace | None = None,
    n_init: int = 4,
    n_iter: int = 8,
    cost_weight: float = 0.0,
    calibration_frac: float = 0.3,
    alpha: float = 0.1,
    seed: int = 0,
    task: str = "",
    density_gate: bool = False,
    density_gate_alpha: float = 0.05,
) -> CalibratedTuneResult:
    """Optimize a distillation recipe and calibrate the winning model for routing.

    The search holds back a ``calibration_frac`` slice of ``val_texts`` before
    evaluating candidate recipes. That slice does not score candidates or
    influence the search; it is used afterward to calibrate the winning model
    into a :class:`~mixle.task.calibrate.CalibratedTaskModel`. The result is a
    task-specific recipe whose complexity and epoch budget were selected from
    data and whose model can be passed directly to a
    :class:`~mixle.task.cascade.Cascade` or :class:`~mixle.task.router.Router`.

    Teacher calls are shared through one cache. ``train_texts`` are queried
    once for the whole search rather than once per trial, and validation inputs
    that appear in both calibration and search slices are not queried twice.
    Every distinct input is priced once, no matter how many candidate recipes
    the search evaluates.

    ``density_gate=True`` wires the same OOD escalation as :func:`~mixle.task.distill.distill_for_routing`: a
    gate fit on ``train_texts``, its floor calibrated on the disjoint ``cal_texts`` slice.
    """
    val_texts = [str(t) for t in val_texts]
    val_truth = _teacher_labels(teacher, val_texts)
    search_texts, _search_labels, cal_texts, cal_labels = _split_for_calibration(
        val_texts, val_truth, calibration_frac, seed
    )
    # Shared across every trial: train_texts is identical candidate to candidate, so this cache avoids
    # tune_recipe's normal per-trial re-query as well as validation/calibration overlap.
    cached_teacher = _teacher_from_cache(dict(zip(val_texts, val_truth)), teacher)
    result = tune_recipe(
        cached_teacher,
        train_texts,
        search_texts,
        labels=labels,
        space=space,
        n_init=n_init,
        n_iter=n_iter,
        cost_weight=cost_weight,
        seed=seed,
        task=task,
    )
    gate = (
        _fit_density_gate(result.model, train_texts, cal_texts, alpha=density_gate_alpha, seed=seed)
        if density_gate
        else None
    )
    calibrated = CalibratedTaskModel(result.model, alpha=alpha, density_gate=gate).calibrate(cal_texts, cal_labels)
    return CalibratedTuneResult(
        model=calibrated,
        recipe=result.recipe,
        agreement=result.agreement,
        score=result.score,
        cost=result.cost,
        selection_agreement=result.selection_agreement,
        selection_score=result.selection_score,
        test_size=result.test_size,
        history=result.history,
    )
