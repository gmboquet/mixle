"""Task-sufficient projection ``pi_T`` (workstream F4): compress a mixture belief for one receiver's
task, keeping only the distinctions that task's readout can tell apart.

Cross-context in this plan is a *task-sufficient projection*, not a compressed blob: a
:class:`~mixle.reason.modality.ModalityView` carries a full structured belief (workstream F1); a
receiver with task ``T`` should get the smallest view of that belief that still answers ``T`` --
different receivers, different projections of the *same* belief. This module builds that operator on
mixle's existing closed-form projection tools (:mod:`mixle.inference.project`: exact Gaussian mixture
collapse, Runnalls reduction) rather than a new compression scheme: components of the belief that
``task`` cannot distinguish are exactly moment-matched into one Gaussian (nothing task-relevant is
lost, because the task never looked at what distinguished them); components ``task`` *can* distinguish
are kept apart.

This is deliberately not generic compression -- a projection built for a different task is not expected
to serve this one well (see ``task_projection_test.py``'s mismatched-projection control), and that gap is
the falsifiable claim this operator has to earn.
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
    """A receiver's task ``T``, reduced to the one thing a projection needs: which components of a
    belief it can tell apart. ``label(mean)`` maps a mixture component's mean to a discrete readout
    value (e.g. a predicted class); components sharing a readout are indistinguishable *for this task*
    and can be losslessly (for ``T``) merged, components with different readouts must stay separate.
    """

    name: str
    label: Callable[[np.ndarray], Hashable]


def task_sufficient_projection(mixture: Any, task: TaskReadout) -> MixtureDistribution:
    """``pi_T(mixture)``: collapse ``mixture``'s components into groups sharing ``task.label``.

    Groups components by ``task.label(component_mean)``; within a group (size > 1), the components are
    exactly moment-matched (:func:`~mixle.inference.project.collapse_mixture`, closed form -- no
    samples) onto one Gaussian. Groups of size 1 pass through unchanged. Returns a
    :class:`~mixle.stats.latent.mixture.MixtureDistribution` with at most as many components as
    ``mixture`` had distinct task labels -- far smaller when many components share a label, and never
    larger than the input.
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
    """The receiver's decision for observation ``x``: ``task.label`` of whichever component of
    ``mixture`` has the highest posterior responsibility for ``x``.

    This is how a receiver consumes a belief -- projected or full -- that it did not itself choose the
    component grouping of: it reads off the task's own label from the winning component's mean, so a
    projection built for a *different* task is scored fairly (and, when it discarded information ``T``
    needed, honestly worse) rather than being handed an answer key.
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
