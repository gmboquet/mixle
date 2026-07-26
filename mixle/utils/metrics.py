"""Classification metrics for Mixle model evaluation.

The module provides likelihood-based classification summaries, ROC/AUC helpers,
search-depth ranking metrics, and paired operating-point utilities used by
examples and validation notebooks.
"""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution

T = TypeVar("T")


def classify(data: Sequence[T], model: SequenceEncodableProbabilityDistribution, labels: list[T] | None = None):
    """Classification of sequence of iid observation from model predictions. Labels may be provided.

    Returns
    Args:
        data (Sequence[T]): Sequence of iid observations for classification.
        model (SequenceEncodableProbabilityDistribution): Distribution for classification.
        labels (Optional[List[T]]): List of labels for the data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    """
    if isinstance(data, (str, bytes)):
        raise TypeError("classification data must be a sequence of (label, observation) rows")
    rows = list(data)
    if not rows:
        raise ValueError("classify requires at least one observation")
    if any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in rows):
        raise ValueError("classification rows must contain exactly (label, observation)")
    cnt = len(rows)
    data_labels = [row[0] for row in rows]

    encoder = model.dist_to_encoder()

    if labels is None:
        try:
            labels = list(dict.fromkeys(data_labels))
        except TypeError as exc:
            raise ValueError("classification labels must be hashable") from exc
    else:
        if isinstance(labels, (str, bytes)):
            raise TypeError("labels must be a sequence of unique class values")
        labels = list(labels)
    if not labels:
        raise ValueError("labels must contain at least one class")
    try:
        if len(set(labels)) != len(labels):
            raise ValueError("labels must contain unique class values")
        label_index = {label: index for index, label in enumerate(labels)}
        missing = [label for label in dict.fromkeys(data_labels) if label not in label_index]
    except TypeError as exc:
        raise ValueError("classification labels must be hashable") from exc
    if missing:
        raise ValueError(f"observed labels are outside the declared support: {missing!r}")

    class_scores = np.empty((cnt, len(labels)), dtype=float)
    for label, column in label_index.items():
        loc_data = [(label, row[1]) for row in rows]
        scores = np.asarray(model.seq_log_density(encoder.seq_encode(loc_data)), dtype=float)
        if scores.shape != (cnt,):
            raise ValueError(f"model must return exactly one score per row; got {scores.shape} for {cnt} rows")
        if not np.all(np.isfinite(scores)):
            raise ValueError("classification log scores must be finite")
        class_scores[:, column] = scores

    max_scores = class_scores.max(axis=1, keepdims=True)
    probabilities = np.exp(class_scores - max_scores)
    normalizers = probabilities.sum(axis=1, keepdims=True)
    if (
        probabilities.shape != (cnt, len(labels))
        or not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(normalizers))
        or np.any(normalizers <= 0.0)
    ):
        raise ValueError("classification scores could not be normalized into probabilities")
    probabilities /= normalizers

    true_labels = np.asarray([label_index[label] for label in data_labels], dtype=int)
    class_prob = probabilities[np.arange(cnt), true_labels]
    class_diff = probabilities - class_prob[:, None]
    class_rank = (class_diff >= 0).sum(axis=1) - 1
    data_labels = np.asarray(data_labels)
    class_probabilities = {label: probabilities[:, column].copy() for label, column in label_index.items()}

    return class_rank, class_prob, data_labels, class_probabilities


def roc_curve(pos_x: list[float] | np.ndarray, neg_x: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create ROC curve.

    Args:
        pos_x (Union[List[float], np.ndarray]): Probs for positive classifications.
        neg_x (Union[List[float], np.ndarray]): Probs for negative classifications.

    Returns:
        Tuple of true-positive rate and false-positive rate arrays.

    """
    pos_x = np.asarray(pos_x, dtype=np.float64)
    neg_x = np.asarray(neg_x, dtype=np.float64)
    if pos_x.ndim != 1 or neg_x.ndim != 1:
        raise ValueError("positive and negative scores must be one-dimensional")
    if pos_x.size == 0 or neg_x.size == 0:
        raise ValueError("roc_curve requires at least one positive and one negative score")
    if not np.all(np.isfinite(pos_x)) or not np.all(np.isfinite(neg_x)):
        raise ValueError("ROC scores must be finite")

    scores = np.concatenate((pos_x, neg_x))
    is_positive = np.concatenate((np.ones(pos_x.size, dtype=bool), np.zeros(neg_x.size, dtype=bool)))
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    is_positive = is_positive[order]
    # A threshold includes every row sharing its score at once. Ordering one
    # class ahead of the other inside a tie invents discrimination.
    block_starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    positives = np.add.reduceat(is_positive.astype(int), block_starts)
    block_sizes = np.diff(np.r_[block_starts, scores.size])
    negatives = block_sizes - positives
    pd = np.r_[0.0, np.cumsum(positives) / pos_x.size]
    fa = np.r_[0.0, np.cumsum(negatives) / neg_x.size]
    return pd, fa


def auc(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Trapezoidal area under a curve.

    Args:
        x: X-axis coordinates, such as false-positive rates.
        y: Y-axis coordinates, such as true-positive rates.

    Returns:
        Non-negative trapezoidal area after sorting by x.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.size < 2:
        raise ValueError("auc requires at least two curve points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("curve coordinates must be finite")
    order = np.argsort(x, kind="stable")
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y[order], x[order]))


def roc_auc(pos_x: list[float] | np.ndarray, neg_x: list[float] | np.ndarray) -> float:
    """Area under the ROC curve for positive and negative scores."""
    pd, fa = roc_curve(pos_x, neg_x)
    return auc(fa, pd)


def roc_percentiles(
    pos_x: list[float] | np.ndarray,
    neg_x: list[float] | np.ndarray,
    perc_points: list[float] | np.ndarray,
) -> np.ndarray:
    """Return false-alarm/probability-detection pairs at requested detection percentiles."""

    perc_points = np.asarray(perc_points, dtype=float)
    if (
        perc_points.ndim != 1
        or not np.all(np.isfinite(perc_points))
        or np.any((perc_points < 0.0) | (perc_points > 1.0))
    ):
        raise ValueError("perc_points must be a finite one-dimensional vector in [0, 1]")
    pd, fa = roc_curve(pos_x, neg_x)
    rv = []

    for i in range(len(perc_points)):
        points = pd <= perc_points[i]

        if np.sum(points) == 0:
            continue

        y = np.max(pd[points])
        x = np.max(fa[pd == y])
        rv.append([x, y])

    return np.asarray(rv, dtype=float).reshape(-1, 2)


def ranking_depth(x, k=None, comp_func=lambda a, b: a == b):
    """Return the first rank depth at which each target appears in ranked candidate lists."""

    if k is not None:
        retval = np.zeros((len(x), k))
        retval.fill(np.nan)
    else:
        retval = []

    idx = 0
    for entry in x:
        scores = np.asarray([u[1] for u in entry[1]])
        matches = np.asarray([comp_func(entry[0], u[0]) for u in entry[1]])

        sidx = np.argsort(-scores)

        matches = matches[sidx]
        scores = scores[sidx]

        ranks = np.arange(len(sidx))[matches]

        if k is not None:
            sz = min(k, len(ranks))
            retval[idx, :sz] = ranks[:sz]
        else:
            retval.append(ranks)

        idx += 1

    return retval
