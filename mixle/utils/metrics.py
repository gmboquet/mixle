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
    cnt = len(data)
    data_labels = [u[0] for u in data]

    encoder = model.dist_to_encoder()

    if labels is None:
        labels = sorted(set(data_labels))

    class_ll = np.zeros((len(data), len(labels)))
    u_labels, true_labels = np.unique(data_labels, return_inverse=True)

    other_labs = sorted(set(labels).difference(list(u_labels)))
    u_label_map = dict(zip(list(u_labels) + other_labs, range(len(u_labels) + len(other_labs))))

    for label in labels:
        idx = u_label_map[label]
        loc_data = [(label, u[1]) for u in data]
        class_ll[:, idx] = model.seq_log_density(encoder.seq_encode(loc_data))

    max_ll = class_ll.max(axis=1, keepdims=True)
    class_ll -= max_ll
    np.exp(class_ll, out=class_ll)
    class_ll /= class_ll.sum(axis=1, keepdims=True)

    class_prob = class_ll[np.arange(cnt), true_labels]
    class_diff = class_ll - class_prob[:, None]
    class_rank = (class_diff >= 0).sum(axis=1) - 1
    data_labels = np.asarray(data_labels)
    class_ll = {label: class_ll[:, u_label_map[label]] for label in labels}

    return class_rank, class_prob, data_labels, class_ll


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
    if len(pos_x) == 0 or len(neg_x) == 0:
        raise ValueError("roc_curve requires at least one positive and one negative score")

    res = np.zeros((len(pos_x) + len(neg_x), 2))
    res[: len(pos_x), 0] = pos_x
    res[: len(pos_x), 1] = 1
    res[len(pos_x) :, 0] = neg_x
    res[len(pos_x) :, 1] = 0

    sidx = np.argsort(-res[:, 0])
    res = res[sidx, :]

    pd = np.cumsum(res[:, 1])
    fa = np.cumsum(1 - res[:, 1])

    pd /= float(len(pos_x))
    fa /= float(len(neg_x))

    return np.concatenate(([0.0], pd)), np.concatenate(([0.0], fa))


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
    order = np.argsort(x)
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

    pd, fa = roc_curve(pos_x, neg_x)
    rv = []

    for i in range(len(perc_points)):
        points = pd <= perc_points[i]

        if np.sum(points) == 0:
            continue

        y = np.max(pd[points])
        x = np.max(fa[pd == y])
        rv.append([x, y])

    return np.asarray(rv)


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
