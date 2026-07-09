"""Embedding goals: declarative objectives layered onto an HViS embedding.

The data term (t-SNE row-KL / UMAP cross-entropy) says "preserve the model's neighborhood
structure"; a goal says what else the layout is FOR. Goals steer the same optimization -- they are
not a post-hoc warp of a finished layout, which would break exactly the neighborhood structure the
embedding exists to show.

**Rate semantics (the stability contract).** A goal's ``gradient(Y)`` returns a bounded
per-iteration DISPLACEMENT, applied as ``Y -= gradient`` once per optimizer iteration -- decoupled
from the data optimizer's learning rate and adaptive gains on purpose. t-SNE's data gradient lives
on probability scale and is driven with ``eta ~ n/12`` and delta-bar-delta gains; a raw quadratic
penalty routed through that machinery is amplified by orders of magnitude and diverges. Under rate
semantics every goal's weight is a fraction-per-step in ``(0, 1]``, contraction-stable by
construction regardless of the optimizer it rides along with.

Three goals, and two headline features are special cases:

* :class:`Anchor` -- ANCHORING: pin chosen points to given coordinates. Hard (``weight=None``) is
  an exact projection every step; soft (``weight`` in ``(0, 1]``) closes that fraction of the
  remaining gap per step -- unconditionally stable exponential relaxation. Anchors fix the
  translational gauge, so the optimizers skip their usual mean-centering (they would fight).
* :class:`LabelCohesion` -- PARTIAL LABELING: labels for any subset of points (``None`` =
  unlabeled). Labeled points move toward their label centroid at ``weight`` per step; an optional
  ``margin`` hinge pushes centroid pairs apart. Unlabeled points are shaped only by the data term
  -- semi-supervision, not relabeling.
* :class:`AxisAlign` -- a layout GOAL: a per-point scalar (time, depth, severity, ...) should run
  along a chosen embedding axis, ascending the Pearson correlation along its scale-normalized
  direction (translation/scale invariant, so it orders the axis without dictating layout scale).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["Anchor", "AxisAlign", "LabelCohesion", "apply_projections", "goals_fix_gauge", "total_goal_gradient"]


class Anchor:
    """Pin ``indices`` to ``coordinates``: hard when ``weight`` is None (exact projection each
    step), soft for ``weight`` in ``(0, 1]`` (close that fraction of the remaining gap per step)."""

    fixes_gauge = True

    def __init__(self, indices: Sequence[int], coordinates: Any, weight: float | None = None) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.coordinates = np.atleast_2d(np.asarray(coordinates, dtype=np.float64))
        if self.coordinates.shape[0] != len(self.indices):
            raise ValueError(f"coordinates must have one row per index ({len(self.indices)}).")
        if weight is not None and not 0.0 < weight <= 1.0:
            raise ValueError("soft anchor weight is a per-step fraction in (0, 1] (None = hard pin).")
        self.weight = None if weight is None else float(weight)

    @property
    def hard(self) -> bool:
        return self.weight is None

    def gradient(self, y: np.ndarray) -> np.ndarray:
        return np.zeros_like(y)  # both anchor kinds act in project(): relaxation is already a step

    def project(self, y: np.ndarray) -> np.ndarray:
        if self.hard:
            y[self.indices] = self.coordinates
        else:
            y[self.indices] += self.weight * (self.coordinates - y[self.indices])
        return y


class LabelCohesion:
    """Partial labels shape the layout: each labeled point moves toward its label's centroid at
    ``weight`` (a per-step fraction in ``(0, 1]``); with ``margin``, centroid pairs closer than
    ``margin`` are pushed apart (every member displaced alike, which moves the centroid by exactly
    the intended amount). ``labels`` has one entry per point; ``None`` marks a point unlabeled."""

    fixes_gauge = False

    def __init__(self, labels: Sequence[Any], weight: float = 0.1, margin: float | None = None) -> None:
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight is a per-step fraction in (0, 1].")
        self.labels = list(labels)
        self.weight = float(weight)
        self.margin = None if margin is None else float(margin)
        self._groups: dict[Any, np.ndarray] = {}
        for label in sorted({lab for lab in self.labels if lab is not None}, key=str):
            self._groups[label] = np.asarray([i for i, lab in enumerate(self.labels) if lab == label], dtype=np.int64)
        if not self._groups:
            raise ValueError("LabelCohesion needs at least one labeled point.")

    def gradient(self, y: np.ndarray) -> np.ndarray:
        if len(self.labels) != y.shape[0]:
            raise ValueError(f"labels cover {len(self.labels)} points but the embedding has {y.shape[0]}.")
        grad = np.zeros_like(y)
        centroids = {label: y[idx].mean(axis=0) for label, idx in self._groups.items()}
        for label, idx in self._groups.items():
            grad[idx] += self.weight * (y[idx] - centroids[label])  # Y -= grad: move toward the centroid
        if self.margin is not None and len(self._groups) > 1:
            labels = list(self._groups)
            n_pairs = len(labels) * (len(labels) - 1) // 2
            for a_pos, a in enumerate(labels):
                for b in labels[a_pos + 1 :]:
                    diff = centroids[a] - centroids[b]
                    dist = float(np.linalg.norm(diff))
                    gap = self.margin - dist
                    if gap <= 0:
                        continue
                    push = (self.weight * gap / n_pairs) * (diff / max(dist, 1.0e-12))
                    grad[self._groups[a]] -= push  # Y -= grad: a-members move +push (apart), b-members -push
                    grad[self._groups[b]] += push
        return grad


class AxisAlign:
    """A per-point scalar should run along embedding axis ``axis``: each step ascends the Pearson
    correlation ``r(y[:, axis], values)`` along its scale-normalized direction (bounded norm <= 2,
    so ``weight`` -- recommended at most ~1 -- is a stable per-step rate in embedding units). Pass
    ``-values`` to reverse direction."""

    fixes_gauge = False

    def __init__(self, values: Sequence[float], axis: int = 0, weight: float = 0.5) -> None:
        if weight <= 0.0:
            raise ValueError("weight must be positive.")
        self.values = np.asarray(values, dtype=np.float64)
        self.axis = int(axis)
        self.weight = float(weight)
        centered = self.values - self.values.mean()
        norm = float(np.linalg.norm(centered))
        if norm <= 0:
            raise ValueError("AxisAlign values must not be constant.")
        self._v_unit = centered / norm

    def gradient(self, y: np.ndarray) -> np.ndarray:
        if len(self.values) != y.shape[0]:
            raise ValueError(f"values cover {len(self.values)} points but the embedding has {y.shape[0]}.")
        grad = np.zeros_like(y)
        u = y[:, self.axis]
        u_centered = u - u.mean()
        u_norm = float(np.linalg.norm(u_centered))
        if u_norm <= 1.0e-12:  # degenerate axis (e.g. the first init steps): no direction to prefer yet
            return grad
        u_unit = u_centered / u_norm
        r = float(u_unit @ self._v_unit)
        direction = self._v_unit - r * u_unit  # the ascent direction on r, normalized to scale-free units
        grad[:, self.axis] = -self.weight * direction  # Y -= grad ascends the correlation
        return grad


def total_goal_gradient(goals: Sequence[Any] | None, y: np.ndarray) -> np.ndarray | None:
    """Summed per-step goal displacement at ``y`` (applied as ``Y -= result``), or None when there
    are no goals so the optimizers skip the add entirely."""
    if not goals:
        return None
    total = np.zeros_like(y)
    for goal in goals:
        total += goal.gradient(y)
    return total


def apply_projections(goals: Sequence[Any] | None, y: np.ndarray, velocity: np.ndarray | None = None) -> np.ndarray:
    """Apply anchor projections/relaxations after a step; zero the velocity of hard-pinned rows so
    momentum cannot accumulate against the pin."""
    if not goals:
        return y
    for goal in goals:
        project = getattr(goal, "project", None)
        if project is None:
            continue
        y = project(y)
        if velocity is not None and getattr(goal, "hard", False):
            velocity[goal.indices] = 0.0
    return y


def goals_fix_gauge(goals: Sequence[Any] | None) -> bool:
    """True when any goal pins absolute coordinates -- the optimizers must then skip mean-centering
    (centering re-translates the cloud every step, which would fight the pins)."""
    return bool(goals) and any(getattr(goal, "fixes_gauge", False) for goal in goals)
