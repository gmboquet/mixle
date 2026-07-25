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
    projection_error: Callable[[tuple[Any, ...], np.ndarray, Any], float] | None = None
    max_projection_error: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TaskReadout.name must be a non-empty string.")
        if not callable(self.label):
            raise TypeError("TaskReadout.label must be callable.")
        if self.projection_error is not None and not callable(self.projection_error):
            raise TypeError("TaskReadout.projection_error must be callable when provided.")
        self.max_projection_error = float(self.max_projection_error)
        if not np.isfinite(self.max_projection_error) or self.max_projection_error < 0.0:
            raise ValueError("max_projection_error must be finite and non-negative.")


def task_sufficient_projection(mixture: Any, task: TaskReadout) -> MixtureDistribution:
    """``pi_T(mixture)``: collapse ``mixture``'s components into groups sharing ``task.label``.

    Components are grouped by ``task.label(component_mean)``. Groups with more
    than one component are moment-matched by
    :func:`~mixle.inference.project.collapse_mixture`; singleton groups pass
    through unchanged. The result never has more components than the input.
    """
    w = _validated_weights(mixture)
    means = _component_means(mixture)

    groups: dict[Hashable, list[int]] = {}
    for k in range(len(w)):
        groups.setdefault(task.label(means[k]), []).append(k)

    merged_dists = [_merge_group(mixture, idx, task) for idx in groups.values()]
    merged_w = np.asarray([float(w[idx].sum()) for idx in groups.values()])
    return MixtureDistribution(merged_dists, merged_w / merged_w.sum())


def read_out(mixture: Any, task: TaskReadout, x: Any) -> Hashable:
    """Return the task label of the component most responsible for ``x``.

    The same readout applies to a full or projected belief, so a projection can
    be evaluated by the task labels it preserves.
    """
    w = _validated_weights(mixture)
    means = _component_means(mixture)
    log_post = np.array(
        [np.log(max(w[k], 1e-300)) + float(mixture.components[k].log_density(x)) for k in range(len(w))]
    )
    if np.isnan(log_post).any() or np.isposinf(log_post).any():
        raise ValueError("component scoring returned an invalid posterior responsibility.")
    if not np.isfinite(log_post).any():
        raise ValueError("observation is impossible under every mixture component; task readout is undefined.")
    return task.label(means[int(np.argmax(log_post))])


def _validated_weights(mixture: Any) -> np.ndarray:
    components = tuple(getattr(mixture, "components", ()))
    w = np.asarray(getattr(mixture, "w", ()), dtype=float)
    if not components or w.shape != (len(components),):
        raise ValueError("mixture must contain one weight per non-empty component.")
    if not np.isfinite(w).all() or np.any(w < 0.0) or not np.isclose(float(w.sum()), 1.0, atol=1e-10):
        raise ValueError("mixture weights must be finite, non-negative, and normalized.")
    return w


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


def _merge_group(mixture: Any, idx: list[int], task: TaskReadout) -> Any:
    if len(idx) == 1:
        return mixture.components[idx[0]]
    if task.projection_error is None:
        raise ValueError(
            f"task {task.name!r} groups distinct components but supplies no projection_error verifier; "
            "equal labels at component means do not prove task sufficiency."
        )
    sub_w = np.asarray(mixture.w, dtype=float)[idx]
    sub = MixtureDistribution([mixture.components[i] for i in idx], sub_w / sub_w.sum())
    merged = collapse_mixture(sub)
    error = float(task.projection_error(tuple(sub.components), np.asarray(sub.w, dtype=float), merged))
    if not np.isfinite(error) or error < 0.0:
        raise ValueError("projection_error must return a finite non-negative bound.")
    if error > task.max_projection_error:
        raise ValueError(
            f"task projection error {error} exceeds the declared bound {task.max_projection_error} "
            f"for task {task.name!r}."
        )
    return merged
