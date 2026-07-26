"""Active labeling -- spend the teacher's expensive labels only where they buy the most, not at random.

Labeling is the dominant cost of building a task model: every label is a frontier call or a human minute. Random
labeling wastes most of them on examples the student already gets right. This is experimental design applied to
the labeling decision (the discrete-pool analogue of ``mixle.doe`` active learning): label a small seed, fit a
student, then repeatedly query the teacher *only* for the pool examples the student is most unsure about (and,
optionally, most novel), refit, and continue until the budget runs out. The same student quality is reached for
far fewer labels -- direct money saved.

Acquisitions score the student's own predictions (uncertainty as a ranking, which needs no calibrated
probability) and can blend in the generative density (:class:`mixle.task.density.DensityGate`) for diversity:

  * ``margin``   -- smallest gap between the top two class scores (the classic, robust default);
  * ``entropy``  -- highest predictive entropy;
  * ``least_confidence`` -- lowest top-class score;
  * ``random``   -- the baseline this is meant to beat.

``active_distill`` returns the student plus a per-round log of labels-spent vs. agreement, so the labeling
efficiency is auditable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.distill import agreement, distill_from_labels
from mixle.task.model import TaskModel


def _validated_probability_matrix(values: Any, n_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n_rows or matrix.shape[1] == 0:
        raise ValueError(f"predictions must have shape ({n_rows}, n_classes) with n_classes > 0")
    if (
        np.any(~np.isfinite(matrix))
        or np.any((matrix < 0.0) | (matrix > 1.0))
        or not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9)
    ):
        raise ValueError("predictions must contain finite row-stochastic probabilities in [0, 1]")
    return matrix


def _entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def _margin(p: np.ndarray) -> np.ndarray:
    s = np.sort(p, axis=1)
    return 1.0 - (s[:, -1] - s[:, -2]) if p.shape[1] >= 2 else 1.0 - s[:, -1]


def _least_confidence(p: np.ndarray) -> np.ndarray:
    return 1.0 - p.max(axis=1)


_ACQ = {"margin": _margin, "entropy": _entropy, "least_confidence": _least_confidence}


def acquisition_scores(student: TaskModel, texts: Sequence[str], method: str = "margin") -> np.ndarray:
    """Informativeness of each unlabeled text under the student (higher = more worth labeling)."""
    if method == "random":
        return np.zeros(len(texts))
    if method not in _ACQ:
        raise ValueError(f"unknown acquisition {method!r}; expected one of {sorted(_ACQ) + ['random']}")
    prob = student.adapter.proba_batch(student.model, list(texts))
    return _ACQ[method](_validated_probability_matrix(prob, len(texts)))


@dataclass
class ActiveResult:
    """The actively-distilled student plus an audit trail of labels spent vs. quality reached each round."""

    model: TaskModel
    labels_used: int  # every teacher label purchased, including validation
    training_labels_used: int = 0
    validation_labels_used: int = 0
    teacher_queries: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    labeled_texts: list[str] = field(default_factory=list)
    labeled_labels: list[Any] = field(default_factory=list)


def active_distill(
    teacher: Callable[..., Any],
    pool: Sequence[str],
    *,
    budget: int,
    seed_size: int = 20,
    rounds: int = 5,
    acquisition: str = "margin",
    labels: Sequence[str] | None = None,
    recipe: dict[str, Any] | None = None,
    val_texts: Sequence[str] | None = None,
    seed: int = 0,
) -> ActiveResult:
    """Distill from ``pool`` under a labeling ``budget``, querying the teacher only for the most informative items.

    Labels a ``seed_size`` random seed, then over ``rounds`` adds the top-scoring unlabeled examples (by
    ``acquisition``) until ``budget`` labels are spent, refitting the student each round. If ``val_texts`` is
    given, the teacher labels it once and each round's agreement on it is logged.
    """
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget <= 0:
        raise ValueError("budget must be an exact positive integer")
    if isinstance(seed_size, (bool, np.bool_)) or not isinstance(seed_size, (int, np.integer)) or seed_size <= 0:
        raise ValueError("seed_size must be an exact positive integer")
    if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, (int, np.integer)) or rounds <= 0:
        raise ValueError("rounds must be an exact positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an exact integer")
    rng = np.random.RandomState(int(seed))
    pool = [str(t) for t in pool]
    if not pool:
        raise ValueError("pool must be non-empty")
    recipe = dict(recipe or {})
    label_space = [str(label) for label in labels] if labels is not None else None
    if label_space is not None and (not label_space or len(set(label_space)) != len(label_space)):
        raise ValueError("labels must be a non-empty sequence of unique values")

    teach = _batched_teacher(teacher)
    validation_texts = [str(text) for text in val_texts] if val_texts is not None else []
    if len(validation_texts) >= budget:
        raise ValueError("budget must exceed the number of validation labels so at least one training label remains")
    val_truth = teach(validation_texts) if validation_texts else None
    validation_labels_used = len(validation_texts)
    training_budget = int(budget) - validation_labels_used

    remaining = list(range(len(pool)))
    rng.shuffle(remaining)
    take = min(int(seed_size), training_budget, len(remaining))
    chosen = remaining[:take]
    remaining = remaining[take:]

    labeled_texts = [pool[i] for i in chosen]
    labeled_labels = list(teach(labeled_texts))
    if label_space is None:
        label_space = sorted({str(y) for y in labeled_labels})
    else:
        unknown = sorted({str(value) for value in labeled_labels} - set(label_space))
        if unknown:
            raise ValueError(f"teacher returned labels outside the declared label space: {unknown!r}")

    history: list[dict[str, Any]] = []
    student = _fit(labeled_texts, labeled_labels, label_space, recipe, seed)
    _log_round(
        history,
        student,
        labeled_texts,
        validation_texts,
        val_truth,
        acquisition,
        validation_labels_used,
    )

    per_round = max(1, (training_budget - take) // int(rounds))
    while len(labeled_labels) < training_budget and remaining:
        k = min(per_round, training_budget - len(labeled_labels), len(remaining))
        cand_texts = [pool[i] for i in remaining]
        if acquisition == "random":
            pick_local = list(range(k))
        else:
            scores = acquisition_scores(student, cand_texts, acquisition)
            pick_local = list(np.argsort(scores)[::-1][:k])
        picked = [remaining[j] for j in pick_local]
        remaining = [i for j, i in enumerate(remaining) if j not in set(pick_local)]

        new_texts = [pool[i] for i in picked]
        labeled_texts += new_texts
        new_labels = list(teach(new_texts))
        labeled_labels += new_labels
        observed_space = sorted({str(value) for value in labeled_labels})
        if labels is None:
            label_space = observed_space
        else:
            unknown = sorted(set(observed_space) - set(label_space))
            if unknown:
                raise ValueError(f"teacher returned labels outside the declared label space: {unknown!r}")
        student = _fit(labeled_texts, labeled_labels, label_space, recipe, seed)
        _log_round(
            history,
            student,
            labeled_texts,
            validation_texts,
            val_truth,
            acquisition,
            validation_labels_used,
        )

    teacher_queries = validation_labels_used + len(labeled_labels)
    return ActiveResult(
        model=student,
        labels_used=teacher_queries,
        training_labels_used=len(labeled_labels),
        validation_labels_used=validation_labels_used,
        teacher_queries=teacher_queries,
        history=history,
        labeled_texts=labeled_texts,
        labeled_labels=labeled_labels,
    )


def _fit(texts, labels_list, label_space, recipe, seed):
    return distill_from_labels(texts, labels_list, labels=label_space, seed=seed, **recipe)


def _log_round(history, student, labeled_texts, val_texts, val_truth, acquisition, validation_labels_used):
    row = {
        "labels_used": validation_labels_used + len(labeled_texts),
        "training_labels_used": len(labeled_texts),
        "validation_labels_used": validation_labels_used,
        "acquisition": acquisition,
    }
    if val_texts:
        row["val_agreement"] = agreement(student, val_truth, list(val_texts))
    history.append(row)


def _batched_teacher(teacher: Callable[..., Any]) -> Callable[[list[str]], list[Any]]:
    def batched(texts: list[str]) -> list[Any]:
        if not texts:
            return []
        out = teacher(texts)
        if not isinstance(out, (list, tuple)) or len(out) != len(texts):
            raise ValueError("teacher must return exactly one label per batched input")
        return list(out)

    return batched
