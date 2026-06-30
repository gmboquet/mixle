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
    return _ACQ[method](prob)


@dataclass
class ActiveResult:
    """The actively-distilled student plus an audit trail of labels spent vs. quality reached each round."""

    model: TaskModel
    labels_used: int
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
    rng = np.random.RandomState(seed)
    pool = [str(t) for t in pool]
    recipe = dict(recipe or {})
    label_space = list(labels) if labels is not None else None

    teach = _batched_teacher(teacher)
    val_truth = teach(list(val_texts)) if val_texts is not None else None

    remaining = list(range(len(pool)))
    rng.shuffle(remaining)
    take = min(seed_size, budget, len(remaining))
    chosen = remaining[:take]
    remaining = remaining[take:]

    labeled_texts = [pool[i] for i in chosen]
    labeled_labels = list(teach(labeled_texts))
    if label_space is None:  # lock the label set from the seed so refits keep a stable head
        label_space = sorted({str(y) for y in labeled_labels})

    history: list[dict[str, Any]] = []
    student = _fit(labeled_texts, labeled_labels, label_space, recipe, seed)
    _log_round(history, student, labeled_texts, val_texts, val_truth, acquisition)

    per_round = max(1, (budget - take) // max(1, rounds))
    while len(labeled_labels) < budget and remaining:
        k = min(per_round, budget - len(labeled_labels), len(remaining))
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
        labeled_labels += list(teach(new_texts))
        student = _fit(labeled_texts, labeled_labels, label_space, recipe, seed)
        _log_round(history, student, labeled_texts, val_texts, val_truth, acquisition)

    return ActiveResult(
        model=student,
        labels_used=len(labeled_labels),
        history=history,
        labeled_texts=labeled_texts,
        labeled_labels=labeled_labels,
    )


def _fit(texts, labels_list, label_space, recipe, seed):
    return distill_from_labels(texts, labels_list, labels=label_space, seed=seed, **recipe)


def _log_round(history, student, labeled_texts, val_texts, val_truth, acquisition):
    row = {"labels_used": len(labeled_texts), "acquisition": acquisition}
    if val_texts is not None:
        row["val_agreement"] = agreement(student, val_truth, list(val_texts))
    history.append(row)


def _batched_teacher(teacher: Callable[..., Any]) -> Callable[[list[str]], list[Any]]:
    def batched(texts: list[str]) -> list[Any]:
        if not texts:
            return []
        out = teacher(texts)
        if isinstance(out, (list, tuple)) and len(out) == len(texts):
            return list(out)
        return [teacher(t) for t in texts]

    return batched
