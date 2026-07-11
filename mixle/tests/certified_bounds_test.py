"""P11 (experimental) -- certified model properties by interval propagation.

The proof obligation is *soundness*: a certified bound must contain every value the model takes on
the box, and a certified monotonicity direction must actually hold -- verified here against dense
grid evaluation. The bounds must also be usefully tight (the card's kill criterion: not >100x the
empirical range).
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.certified_bounds import (
    certified_density_bounds,
    certify_density_monotonic,
    grid_density_range,
    looseness,
)
from mixle.stats import GaussianDistribution, MixtureDistribution

G = GaussianDistribution(1.0, 2.0)
MIX = MixtureDistribution([GaussianDistribution(-2, 1.0), GaussianDistribution(3, 0.5)], [0.4, 0.6])

BOXES = [(G, 2.0, 4.0), (G, -1.0, 3.0), (G, -5.0, -2.0), (MIX, -4.0, 5.0), (MIX, 0.0, 2.0)]


@pytest.mark.parametrize("model, lo, hi", BOXES)
def test_bounds_are_sound(model, lo, hi) -> None:
    clo, chi = certified_density_bounds(model, lo, hi)
    glo, ghi = grid_density_range(model, lo, hi)
    assert clo <= glo + 1e-9, f"certified lower {clo} exceeds true min {glo}"
    assert chi >= ghi - 1e-9, f"certified upper {chi} below true max {ghi}"


@pytest.mark.parametrize("model, lo, hi", BOXES)
def test_bounds_are_tight_enough(model, lo, hi) -> None:
    assert looseness(model, lo, hi) < 100.0, "certified bound is >100x looser than empirical (kill criterion)"


def test_gaussian_bounds_are_exactly_tight() -> None:
    assert np.isclose(looseness(G, 2.0, 4.0), 1.0, atol=1e-6)


def test_monotonicity_certified_correctly() -> None:
    assert certify_density_monotonic(G, 2.0, 4.0) == "decreasing"  # right of mode 1.0
    assert certify_density_monotonic(G, -2.0, 0.0) == "increasing"  # left of mode
    assert certify_density_monotonic(G, 0.0, 3.0) == "not certified"  # straddles mode
    assert certify_density_monotonic(MIX, 4.0, 6.0) == "decreasing"  # right of both modes


def test_certified_monotonicity_is_sound() -> None:
    """When a direction is certified, the density really is monotone on the grid (the proof holds)."""
    for model, lo, hi in [(G, 2.0, 4.0), (G, -2.0, 0.0), (MIX, 4.0, 6.0), (MIX, -6.0, -3.0)]:
        verdict = certify_density_monotonic(model, lo, hi)
        if verdict == "not certified":
            continue
        xs = np.linspace(lo, hi, 4001)
        comps, w = (
            (model.components, np.asarray(model.w)) if hasattr(model, "components") else ([model], np.array([1.0]))
        )
        dens = sum(
            wk * np.exp(-((xs - c.mu) ** 2) / (2 * c.sigma2)) / np.sqrt(2 * np.pi * c.sigma2) for wk, c in zip(w, comps)
        )
        diffs = np.diff(dens)
        if verdict == "increasing":
            assert np.all(diffs >= -1e-9), "certified increasing but density decreases somewhere"
        else:
            assert np.all(diffs <= 1e-9), "certified decreasing but density increases somewhere"


def test_mixture_disagreement_is_not_certified() -> None:
    # A box that is left of one mode and right of the other -> components disagree -> not certified.
    assert certify_density_monotonic(MIX, -1.0, 1.0) == "not certified"


def test_determinism() -> None:
    assert certified_density_bounds(MIX, -4.0, 5.0) == certified_density_bounds(MIX, -4.0, 5.0)
