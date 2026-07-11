"""P11 (experimental) -- certified model properties by abstract interpretation over the tree.

G1 propagates *distributions* through the model tree; the verification sibling propagates *sets*
(here, intervals). Closed-form leaves have exact range arithmetic, and the combinators are few and
typed, so a property of a composed model can be turned from a *measured* receipt into a *proven*
one: a certified bound holds for EVERY point in an input box, not just the sampled ones.

This module certifies two properties for Gaussian leaves and their mixtures:

* :func:`certified_density_bounds` -- sound lower/upper bounds on the density over an input box
  (the mixture bound is the weighted interval sum of its components);
* :func:`certify_density_monotonic` -- whether the density is provably monotone (one sign of the
  derivative) over the box; a mixture is certified monotone only when all components agree.

Soundness (the certified interval really contains every value) and tightness (it is not absurdly
loose) are checked against dense grid evaluation in the test, per the card's validation plan.

Exploratory ``mixle.experimental`` code (P11 card).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _gauss(x: np.ndarray, mu: float, s2: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2.0 * s2)) / np.sqrt(2.0 * np.pi * s2)


def _components(model: Any) -> tuple[list, np.ndarray]:
    if hasattr(model, "components"):
        w = np.asarray(getattr(model, "w", getattr(model, "weights", None)), dtype=float)
        return list(model.components), w / w.sum()
    return [model], np.array([1.0])  # a bare leaf is a 1-component mixture


def _gaussian_density_range(mu: float, s2: float, lo: float, hi: float) -> tuple[float, float]:
    """Exact [min, max] of a Gaussian density over the box [lo, hi]."""
    x_max = min(max(mu, lo), hi)  # closest point to the mode
    x_min = lo if abs(lo - mu) >= abs(hi - mu) else hi  # farthest point from the mode
    return float(_gauss(np.array([x_min]), mu, s2)[0]), float(_gauss(np.array([x_max]), mu, s2)[0])


def certified_density_bounds(model: Any, lo: float, hi: float) -> tuple[float, float]:
    """Sound [lower, upper] bound on ``model``'s density over ``[lo, hi]`` (contains every value)."""
    comps, w = _components(model)
    dmin = dmax = 0.0
    for wk, c in zip(w, comps):
        lo_k, hi_k = _gaussian_density_range(float(c.mu), float(c.sigma2), lo, hi)
        dmin += wk * lo_k
        dmax += wk * hi_k
    return float(dmin), float(dmax)


def certify_density_monotonic(model: Any, lo: float, hi: float) -> str:
    """Certify the density is monotone over the box: 'increasing', 'decreasing', or 'not certified'.

    A Gaussian density rises toward its mode and falls after it, so it is monotone on any box that
    does not straddle the mode. A mixture is certified monotone only when every component is
    monotone in the same direction (a sound, not complete, rule).
    """
    comps, _ = _components(model)
    directions = set()
    for c in comps:
        mu = float(c.mu)
        if hi <= mu:
            directions.add("increasing")
        elif lo >= mu:
            directions.add("decreasing")
        else:
            return "not certified"  # this component straddles its mode
    if directions == {"increasing"}:
        return "increasing"
    if directions == {"decreasing"}:
        return "decreasing"
    return "not certified"  # components disagree


def grid_density_range(model: Any, lo: float, hi: float, n: int = 2001) -> tuple[float, float]:
    """Empirical [min, max] of the density on a dense grid (the validation reference)."""
    xs = np.linspace(lo, hi, n)
    comps, w = _components(model)
    dens = sum(wk * _gauss(xs, float(c.mu), float(c.sigma2)) for wk, c in zip(w, comps))
    return float(np.min(dens)), float(np.max(dens))


def looseness(model: Any, lo: float, hi: float, *, n: int = 2001) -> float:
    """Ratio of the certified interval width to the true (grid) width -- 1.0 is exactly tight."""
    clo, chi = certified_density_bounds(model, lo, hi)
    glo, ghi = grid_density_range(model, lo, hi, n)
    true_w = ghi - glo
    return (chi - clo) / true_w if true_w > 1e-12 else 1.0
