"""Edge-preserving and discrete-composition priors for latent fields.

The Gaussian-Markov / GP field prior is smooth -- it blurs sharp material boundaries and cannot express a
field that takes a few discrete values (a composition of distinct materials). These priors fix that, and
they plug into the field surface as *data-less proxies*: a prior is a proxy whose log-likelihood is the
negative penalty, so ``joint([Gaussian(...), TotalVariation(over=field, shape=...)])`` can include it directly.

- :func:`TotalVariation` -- a smoothed total-variation penalty on the field's gradient, which preserves
  sharp edges where the smooth prior would round them (the standard regularizer for piecewise-constant
  images / sharp inclusions).
- :func:`Potts` -- a multi-well penalty pulling each node toward one of a few given levels, encoding a
  discrete material composition (a continuous relaxation of the Potts model).

Both are most useful with ``how='map'`` (the edge-preserving / discrete reconstruction is the point; the
posterior is genuinely non-Gaussian, so Laplace/Gauss-Newton only approximate it around the mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from mixle.ppl._grid import _grid_faces
from mixle.ppl.field import Proxy


class _PenaltyProxy(Proxy):
    """A data-less proxy whose log-likelihood is the negative of a field penalty (a prior term)."""

    def __init__(self, penalty, prefix):
        self._penalty = penalty
        self.prefix = prefix

    def loglik(self, field_t, params, torch):
        return -self._penalty(field_t, torch)


def _positive_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


@dataclass(frozen=True)
class _TotalVariationPenalty:
    face_a: tuple[int, ...]
    face_b: tuple[int, ...]
    n_nodes: int
    weight: float
    eps: float

    def __call__(self, field_t, torch):
        values = field_t.reshape(-1)
        if values.numel() != self.n_nodes:
            raise ValueError(f"field contains {values.numel()} values; grid geometry requires {self.n_nodes}")
        device = values.device
        a = torch.as_tensor(self.face_a, dtype=torch.long, device=device)
        b = torch.as_tensor(self.face_b, dtype=torch.long, device=device)
        difference = values[a] - values[b]
        return self.weight * torch.sum(torch.sqrt(difference * difference + self.eps * self.eps))


@dataclass(frozen=True)
class _PottsPenalty:
    levels: tuple[float, ...]
    weight: float

    def __call__(self, field_t, torch):
        well = torch.ones_like(field_t)
        for level in self.levels:
            well = well * (field_t - level) ** 2
        return self.weight * torch.sum(well)


def _field_of(over):
    return over.field if hasattr(over, "field") else over


def TotalVariation(over, shape, *, weight: float = 1.0, eps: float = 1e-3) -> tuple:
    """A smoothed total-variation prior on the field over a structured ``shape`` grid: ``weight * sum over
    neighbour pairs sqrt((f_a - f_b)^2 + eps^2)``. Edge-preserving (it does not penalize a jump as harshly
    as the squared GMRF prior). Returns the ``(field, proxy)`` pair for :func:`joint`."""
    field = _field_of(over)
    g = _grid_faces(shape, 1.0)
    penalty = _TotalVariationPenalty(
        face_a=tuple(int(index) for index in g["face_a"]),
        face_b=tuple(int(index) for index in g["face_b"]),
        n_nodes=int(g["n"]),
        weight=_positive_finite(weight, "weight"),
        eps=_positive_finite(eps, "eps"),
    )
    return field, _PenaltyProxy(penalty, "tv")


def Potts(over, levels, *, weight: float = 1.0) -> tuple:
    """A discrete-composition prior: ``weight * sum_i prod_k (f_i - level_k)^2`` -- a multi-well potential
    whose minima are the given ``levels``, pulling the field toward a few discrete material values (a
    smooth relaxation of the Potts model). Combine with :func:`TotalVariation` for piecewise-constant
    regions. Returns the ``(field, proxy)`` pair for :func:`joint`."""
    field = _field_of(over)
    try:
        level_array = np.asarray(levels, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise TypeError("levels must be a finite one-dimensional numeric collection") from error
    if level_array.size < 2:
        raise ValueError("levels must contain at least two distinct finite values")
    if not np.all(np.isfinite(level_array)):
        raise ValueError("levels must contain only finite values")
    lv = tuple(float(value) for value in np.unique(level_array))
    if len(lv) < 2:
        raise ValueError("levels must contain at least two distinct finite values")
    penalty = _PottsPenalty(levels=lv, weight=_positive_finite(weight, "weight"))
    return field, _PenaltyProxy(penalty, "potts")
