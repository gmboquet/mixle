"""Twisted composition for mixture components with a shared base density.

Mixture components can share one base density modulo a declared group action,
instead of each independently learning a density from scratch on its own slice
of data. Concretely, :class:`CyclicGroup`
acts on a periodic coordinate by rotation of its ``(cos, sin)`` embedding (an exact, Jacobian-1 change of
variables, so a fitted embedding density scores identically whichever group element aligned a point into it);
:func:`fit_twisted_mixture` pools every group's data into ONE shared base density after undoing each group's
twist, so the shared density is fit on the union -- effectively ``|groups|`` times the data for the same
parameter count as fitting one group alone.

Use this as an experimental modeling option. If the shared-base model does not
beat independently fit per-group models at matched per-component capacity on
held-out per-group log likelihood, keep the independent baseline as the default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CyclicGroup:
    """Z_``order`` acting on a periodic real-valued coordinate of period ``period`` by rotation.

    Each group element ``k`` in ``{0, ..., order - 1}`` is realized concretely as a rotation of the
    coordinate's ``(cos, sin)`` embedding by angle ``2*pi*k/order`` -- an orthogonal (norm- and
    Jacobian-preserving) transform, so composing group elements is exactly addition mod ``order``
    (:meth:`compose`), and a density fit on the embedding is unaffected by which element aligned a point into
    it (the twist is undone before scoring, not baked into the density).
    """

    order: int
    period: float = 1.0

    def embed(self, x: Sequence[float]) -> np.ndarray:
        """The periodic coordinate's ``(cos, sin)`` embedding, shape ``(..., 2)``."""
        theta = 2.0 * np.pi * np.asarray(x, dtype=np.float64) / self.period
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    def _rotation(self, k: int) -> np.ndarray:
        angle = 2.0 * np.pi * (k % self.order) / self.order
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def act(self, embedded: np.ndarray, k: int) -> np.ndarray:
        """Rotate an ``(..., 2)`` embedding by group element ``k`` (the forward twist)."""
        return np.asarray(embedded, dtype=np.float64) @ self._rotation(k).T

    def inverse_act(self, embedded: np.ndarray, k: int) -> np.ndarray:
        """Undo group element ``k``'s twist -- ``act(inverse_act(v, k), k) == v``."""
        return self.act(embedded, -k)

    def compose(self, k1: int, k2: int) -> int:
        """The group element equivalent to applying ``k1`` then ``k2`` -- addition mod ``order``."""
        return (k1 + k2) % self.order


@dataclass
class TwistedMixtureResult:
    """A single shared base density plus the group whose elements twist it into each group's local factor."""

    base_density: Any
    group: CyclicGroup

    def log_density(self, x: Sequence[float], k: int) -> np.ndarray:
        """``log p(x | group=k)``: undo ``k``'s twist, then score under the shared base density."""
        aligned = self.group.inverse_act(self.group.embed(x), k)
        enc = self.base_density.dist_to_encoder().seq_encode([row for row in aligned])
        return np.asarray(self.base_density.seq_log_density(enc), dtype=np.float64)


def _fit_density(rows: list[np.ndarray], *, n_components: int, seed: int, max_its: int) -> Any:
    import mixle.stats as st
    from mixle.inference import optimize

    est = st.MixtureEstimator([st.DiagonalGaussianEstimator(dim=2)] * n_components)
    return optimize(rows, est, max_its=max_its, rng=np.random.RandomState(seed), out=None)


def fit_twisted_mixture(
    group: CyclicGroup,
    data_by_group: dict[int, Sequence[float]],
    *,
    n_components: int = 2,
    seed: int = 0,
    max_its: int = 50,
) -> TwistedMixtureResult:
    """Fit ONE base density on every group's data pooled together after undoing each group's twist.

    ``data_by_group`` maps a group element ``k`` to that group's (small) sample of the periodic coordinate.
    Every sample is embedded and rotated back by its own group's ``inverse_act`` before pooling -- so the
    fitted ``n_components``-component density sees ``sum(len(v) for v in data_by_group.values())`` points,
    not just one group's slice, for the same parameter count as :func:`fit_independent_mixtures` spends on a
    SINGLE group.
    """
    rows: list[np.ndarray] = []
    for k, xs in data_by_group.items():
        aligned = group.inverse_act(group.embed(xs), k)
        rows.extend(list(aligned))
    return TwistedMixtureResult(_fit_density(rows, n_components=n_components, seed=seed, max_its=max_its), group)


def fit_independent_mixtures(
    group: CyclicGroup,
    data_by_group: dict[int, Sequence[float]],
    *,
    n_components: int = 2,
    seed: int = 0,
    max_its: int = 50,
) -> dict[int, Any]:
    """The untwisted baseline: one independently-fit ``n_components``-component density per group.

    Same per-group parameter count as the shared base density in :func:`fit_twisted_mixture`, but ``|groups|``
    times the total parameters overall, and each fit sees only its own group's (small) sample -- the
    comparison :func:`fit_twisted_mixture` is measured against.
    """
    out: dict[int, Any] = {}
    for k, xs in data_by_group.items():
        rows = list(group.embed(xs))
        out[k] = _fit_density(rows, n_components=n_components, seed=seed + int(k), max_its=max_its)
    return out


def independent_log_density(models: dict[int, Any], group: CyclicGroup, x: Sequence[float], k: int) -> np.ndarray:
    """Score ``x`` under group ``k``'s independently-fit density (the baseline sibling of
    :meth:`TwistedMixtureResult.log_density`)."""
    enc = models[k].dist_to_encoder().seq_encode([row for row in group.embed(x)])
    return np.asarray(models[k].seq_log_density(enc), dtype=np.float64)
