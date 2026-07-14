"""N4: biodiversity impact & habitat connectivity (graph resistance on a habitat-cost raster)."""

from __future__ import annotations

import numpy as np

from mixle.analysis.biodiversity import (
    fragmentation_impact,
    habitat_connectivity,
    least_cost_corridor,
    resistance_raster,
)

_ROWS, _COLS = 5, 9


class _FakeHabitat:
    """Minimal duck-typed stand-in exposing only ``.mean`` (mirrors N6's ``habitat.mean``-only usage)."""

    def __init__(self, mean: np.ndarray) -> None:
        self.mean = mean


def _flat(row: int, col: int, cols: int = _COLS) -> int:
    return row * cols + col


def _two_patch_grid(rows: int = _ROWS, cols: int = _COLS) -> np.ndarray:
    """A suitability grid: two full-column patches (left/right) joined by a single-row corridor.

    Everything else is low-suitability "matrix"; row ``2`` (the middle row), across the interior
    columns, is the only high-suitability route connecting the two patch columns.
    """
    suitability = np.full((rows, cols), 1e-6)
    suitability[:, 0] = 1.0
    suitability[:, cols - 1] = 1.0
    suitability[2, 1 : cols - 1] = 1.0
    return suitability


def test_resistance_raster_inverts_suitability_with_a_floor():
    habitat = _FakeHabitat(np.array([1.0, 0.1, 0.0]))
    resistance = resistance_raster(habitat, floor=1e-3)
    assert resistance.shape == (3,)
    assert np.isclose(resistance[0], 1.0)
    assert np.isclose(resistance[1], 10.0)
    assert np.isclose(resistance[2], 1.0 / 1e-3)  # floor keeps this finite rather than inf


def test_least_cost_corridor_follows_the_high_suitability_row():
    resistance = resistance_raster(_FakeHabitat(_two_patch_grid()), floor=1e-3)
    cost, path = least_cost_corridor(resistance, _flat(2, 0), _flat(2, _COLS - 1))
    assert path[0] == _flat(2, 0)
    assert path[-1] == _flat(2, _COLS - 1)
    assert np.isclose(cost, float(_COLS - 1))  # COLS-1 unit-resistance edges straight along row 2


def test_least_cost_corridor_returns_inf_when_no_path_exists():
    resistance = np.array([[1.0, np.inf], [np.inf, 1.0]])
    cost, path = least_cost_corridor(resistance, 0, 3)
    assert cost == float("inf")
    assert path == []


def test_fragmentation_impact_on_corridor_footprint_severs_connectivity_off_corridor_does_not():
    suitability = _two_patch_grid()
    resistance = resistance_raster(_FakeHabitat(suitability), floor=1e-3)

    sources = [_flat(r, 0) for r in range(_ROWS)]
    sinks = [_flat(r, _COLS - 1) for r in range(_ROWS)]

    on_corridor_mask = np.zeros((_ROWS, _COLS), dtype=bool)
    on_corridor_mask[2, 3:6] = True  # 3 cells, dead center of the one and only corridor

    off_corridor_mask = np.zeros((_ROWS, _COLS), dtype=bool)
    off_corridor_mask[0, 3:6] = True  # equal area (3 cells), off the corridor row entirely

    on_corridor = fragmentation_impact(resistance, on_corridor_mask, sources, sinks)
    off_corridor = fragmentation_impact(resistance, off_corridor_mask, sources, sinks)

    # baseline numbers must agree regardless of which footprint is being scored against it
    assert np.isclose(on_corridor["corridor_resistance_baseline"], off_corridor["corridor_resistance_baseline"])
    assert np.isclose(on_corridor["connectivity_baseline"], off_corridor["connectivity_baseline"])

    # severing the only corridor strictly raises movement cost and strictly lowers connectivity
    assert on_corridor["corridor_resistance_mined"] > on_corridor["corridor_resistance_baseline"]
    assert on_corridor["connectivity_mined"] < on_corridor["connectivity_baseline"]

    # an equal-area footprint that misses the corridor leaves both metrics within tolerance
    assert np.isclose(
        off_corridor["corridor_resistance_mined"], off_corridor["corridor_resistance_baseline"], rtol=1e-9
    )
    assert np.isclose(off_corridor["connectivity_mined"], off_corridor["connectivity_baseline"], rtol=0.05)

    assert set(on_corridor) == {
        "corridor_resistance_baseline",
        "corridor_resistance_mined",
        "connectivity_baseline",
        "connectivity_mined",
        "delta",
        "mincut_edges",
    }
    assert on_corridor["delta"] > 0.0
    assert len(on_corridor["mincut_edges"]) >= 1


def test_habitat_connectivity_drops_to_near_zero_when_corridor_is_severed():
    suitability = _two_patch_grid()
    resistance = resistance_raster(_FakeHabitat(suitability), floor=1e-3)
    sources = [_flat(r, 0) for r in range(_ROWS)]
    sinks = [_flat(r, _COLS - 1) for r in range(_ROWS)]

    baseline = habitat_connectivity(resistance, sources, sinks)
    mined = resistance.copy()
    mined[2, 3:6] = np.inf
    severed = habitat_connectivity(mined, sources, sinks)

    assert baseline > 0.0
    assert severed < baseline
    assert severed < 0.1 * baseline  # only the low-conductance background "matrix" routes remain
