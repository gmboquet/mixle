"""Task-sufficient projection ``pi_T`` for receiver-specific beliefs.

A :class:`~mixle.reason.modality.ModalityView` can carry a full structured
belief, while a receiver for task ``T`` may need only the smallest projection
that preserves the distinctions relevant to that task. This module builds that
operator on existing closed-form projection tools from
:mod:`mixle.inference.project`: components that ``task`` cannot distinguish
are moment-matched into one Gaussian, while components the task can
distinguish are kept separate.

This is task-specific projection rather than generic compression. A projection
built for one task should be validated before being reused for another.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.project import collapse_mixture
from mixle.stats.latent.mixture import MixtureDistribution


@dataclass
class TaskReadout:
    """Task readout used to decide which mixture components can be merged.

    ``label(mean)`` maps a component mean to a discrete readout value. Components
    sharing a readout are indistinguishable for this task and may be merged;
    components with different readouts remain separate.
    """

    name: str
    label: Callable[[np.ndarray], Hashable]


def task_sufficient_projection(mixture: Any, task: TaskReadout) -> MixtureDistribution:
    """``pi_T(mixture)``: collapse ``mixture``'s components into groups sharing ``task.label``.

    Components are grouped by ``task.label(component_mean)``. Groups with more
    than one component are moment-matched by
    :func:`~mixle.inference.project.collapse_mixture`; singleton groups pass
    through unchanged. The result never has more components than the input.
    """
    w = np.asarray(mixture.w, dtype=float)
    means = _component_means(mixture)

    groups: dict[Hashable, list[int]] = {}
    for k in range(len(w)):
        groups.setdefault(task.label(means[k]), []).append(k)

    merged_dists = [_merge_group(mixture, idx) for idx in groups.values()]
    merged_w = np.asarray([float(w[idx].sum()) for idx in groups.values()])
    return MixtureDistribution(merged_dists, merged_w / merged_w.sum())


def read_out(mixture: Any, task: TaskReadout, x: Any) -> Hashable:
    """Return the task label of the component most responsible for ``x``.

    The same readout applies to a full or projected belief, so a projection can
    be evaluated by the task labels it preserves.
    """
    w = np.asarray(mixture.w, dtype=float)
    means = _component_means(mixture)
    log_post = np.array(
        [np.log(max(w[k], 1e-300)) + float(mixture.components[k].log_density(x)) for k in range(len(w))]
    )
    return task.label(means[int(np.argmax(log_post))])


def _component_means(mixture: Any) -> np.ndarray:
    if hasattr(mixture, "mu") and hasattr(mixture, "sig2"):  # GaussianMixtureDistribution: (K, d) directly
        return np.asarray(mixture.mu, dtype=float)
    means = []
    for c in mixture.components:
        if hasattr(c, "covar"):  # MultivariateGaussianDistribution(mu, covar)
            means.append(np.asarray(c.mu, dtype=float).ravel())
        elif hasattr(c, "sigma2"):  # univariate GaussianDistribution(mu, sigma2)
            means.append(np.array([float(c.mu)]))
        else:
            raise ValueError(
                f"component {type(c).__name__} is not Gaussian; task_sufficient_projection needs a "
                "Gaussian mixture belief (see mixle.inference.project for the same restriction)."
            )
    return np.asarray(means)


def _merge_group(mixture: Any, idx: list[int]) -> Any:
    if len(idx) == 1:
        return mixture.components[idx[0]]
    sub_w = np.asarray(mixture.w, dtype=float)[idx]
    sub = MixtureDistribution([mixture.components[i] for i in idx], sub_w / sub_w.sum())
    return collapse_mixture(sub)
