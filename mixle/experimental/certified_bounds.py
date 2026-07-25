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

from dataclasses import dataclass
from typing import Any

import numpy as np


def _gauss(x: np.ndarray, mu: float, s2: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2.0 * s2)) / np.sqrt(2.0 * np.pi * s2)


@dataclass(frozen=True)
class DensityBoundReceipt:
    """Certified interval together with the assumptions used by the proof."""

    lower: float
    upper: float
    interval: tuple[float, float]
    component_means: tuple[float, ...]
    component_variances: tuple[float, ...]
    normalized_weights: tuple[float, ...]
    rule: str = "gaussian-weighted-interval-sum/v1"

    def __iter__(self):
        """Preserve historical ``lower, upper = certified_density_bounds(...)`` unpacking."""
        return iter((self.lower, self.upper))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        return (self.lower, self.upper)[index]


def _validated_interval(lo: Any, hi: Any) -> tuple[float, float]:
    if (
        isinstance(lo, (bool, np.bool_))
        or isinstance(hi, (bool, np.bool_))
        or not np.isscalar(lo)
        or not np.isscalar(hi)
    ):
        raise ValueError("lo and hi must be finite scalar bounds.")
    lo_value, hi_value = float(lo), float(hi)
    if not np.isfinite(lo_value) or not np.isfinite(hi_value) or lo_value > hi_value:
        raise ValueError("lo and hi must be finite and satisfy lo <= hi.")
    return lo_value, hi_value


def _validated_gaussian(component: Any) -> tuple[float, float]:
    if not hasattr(component, "mu") or not hasattr(component, "sigma2"):
        raise TypeError("certification supports only Gaussian leaves with mu and sigma2.")
    try:
        mu, sigma2 = float(component.mu), float(component.sigma2)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gaussian mean and variance must be finite scalars.") from exc
    if not np.isfinite(mu) or not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("Gaussian mean must be finite and variance must be finite and positive.")
    return mu, sigma2


def _components(model: Any) -> tuple[list[Any], np.ndarray]:
    if hasattr(model, "components"):
        components = list(model.components)
        if not components:
            raise ValueError("mixture must contain at least one component.")
        raw_weights = getattr(model, "w", getattr(model, "weights", None))
        if raw_weights is None:
            raise ValueError("mixture must expose one weight per component.")
        try:
            weights = np.asarray(raw_weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("mixture weights must be finite non-negative scalars.") from exc
        if (
            weights.shape != (len(components),)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isfinite(weights.sum())
            or weights.sum() <= 0.0
        ):
            raise ValueError("mixture weights must match components and be finite, non-negative, and non-zero.")
        for component in components:
            _validated_gaussian(component)
        return components, weights / weights.sum()
    _validated_gaussian(model)
    return [model], np.array([1.0])  # a bare leaf is a 1-component mixture


def _gaussian_density_range(mu: float, s2: float, lo: float, hi: float) -> tuple[float, float]:
    """Exact [min, max] of a Gaussian density over the box [lo, hi]."""
    x_max = min(max(mu, lo), hi)  # closest point to the mode
    x_min = lo if abs(lo - mu) >= abs(hi - mu) else hi  # farthest point from the mode
    return float(_gauss(np.array([x_min]), mu, s2)[0]), float(_gauss(np.array([x_max]), mu, s2)[0])


def certified_density_bounds(model: Any, lo: float, hi: float) -> DensityBoundReceipt:
    """Certify a density interval and record every validated proof assumption."""
    lo, hi = _validated_interval(lo, hi)
    comps, w = _components(model)
    dmin = dmax = 0.0
    means = []
    variances = []
    for wk, c in zip(w, comps):
        mu, sigma2 = _validated_gaussian(c)
        means.append(mu)
        variances.append(sigma2)
        lo_k, hi_k = _gaussian_density_range(mu, sigma2, lo, hi)
        dmin += wk * lo_k
        dmax += wk * hi_k
    return DensityBoundReceipt(
        lower=float(dmin),
        upper=float(dmax),
        interval=(lo, hi),
        component_means=tuple(means),
        component_variances=tuple(variances),
        normalized_weights=tuple(float(weight) for weight in w),
    )


def certify_density_monotonic(model: Any, lo: float, hi: float) -> str:
    """Certify the density is monotone over the box: 'increasing', 'decreasing', or 'not certified'.

    A Gaussian density rises toward its mode and falls after it, so it is monotone on any box that
    does not straddle the mode. A mixture is certified monotone only when every component is
    monotone in the same direction (a sound, not complete, rule).
    """
    lo, hi = _validated_interval(lo, hi)
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
    lo, hi = _validated_interval(lo, hi)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or int(n) < 2:
        raise ValueError("n must be an integer of at least 2.")
    xs = np.linspace(lo, hi, n)
    comps, w = _components(model)
    dens = sum(wk * _gauss(xs, float(c.mu), float(c.sigma2)) for wk, c in zip(w, comps))
    return float(np.min(dens)), float(np.max(dens))


def looseness(model: Any, lo: float, hi: float, *, n: int = 2001) -> float:
    """Ratio of the certified interval width to the true (grid) width -- 1.0 is exactly tight."""
    receipt = certified_density_bounds(model, lo, hi)
    glo, ghi = grid_density_range(model, lo, hi, n)
    true_w = ghi - glo
    return (receipt.upper - receipt.lower) / true_w if true_w > 1e-12 else 1.0
