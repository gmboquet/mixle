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


def _field_of(over):
    return over.field if hasattr(over, "field") else over


def TotalVariation(over, shape, *, weight: float = 1.0, eps: float = 1e-3) -> tuple:
    """A smoothed total-variation prior on the field over a structured ``shape`` grid: ``weight * sum over
    neighbour pairs sqrt((f_a - f_b)^2 + eps^2)``. Edge-preserving (it does not penalize a jump as harshly
    as the squared GMRF prior). Returns the ``(field, proxy)`` pair for :func:`joint`."""
    field = _field_of(over)
    g = _grid_faces(shape, 1.0)
    fa, fb = g["face_a"], g["face_b"]

    def penalty(field_t, torch):
        a = torch.as_tensor(fa, dtype=torch.long)
        b = torch.as_tensor(fb, dtype=torch.long)
        d = field_t[a] - field_t[b]
        return float(weight) * torch.sum(torch.sqrt(d * d + eps * eps))

    return field, _PenaltyProxy(penalty, "tv")


def Potts(over, levels, *, weight: float = 1.0) -> tuple:
    """A discrete-composition prior: ``weight * sum_i prod_k (f_i - level_k)^2`` -- a multi-well potential
    whose minima are the given ``levels``, pulling the field toward a few discrete material values (a
    smooth relaxation of the Potts model). Combine with :func:`TotalVariation` for piecewise-constant
    regions. Returns the ``(field, proxy)`` pair for :func:`joint`."""
    field = _field_of(over)
    lv = [float(v) for v in np.asarray(levels, dtype=float).ravel()]

    def penalty(field_t, torch):
        well = torch.ones_like(field_t)
        for level in lv:
            well = well * (field_t - level) ** 2
        return float(weight) * torch.sum(well)

    return field, _PenaltyProxy(penalty, "potts")
