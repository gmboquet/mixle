"""N4: biodiversity impact & habitat connectivity (graph resistance on a habitat-cost raster)."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.biodiversity import (
    _conductance_edges,
    _graph_laplacian,
    _reachable,
    effective_conductance,
    fragmentation_impact,
    habitat_connectivity,
    least_cost_corridor,
    max_flow_connectivity,
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


# --------------------------------------------------------------------------------------------------
# MXR-080-0071: `habitat_connectivity` claimed a circuit-theory effective-conductance metric but was
# actually maximum flow -- a different quantity (series conductances add reciprocally; max flow only sees
# the tightest cut). `effective_conductance` is the corrected Dirichlet/Laplacian boundary-value solve;
# `habitat_connectivity` is now a plain alias for it. `max_flow_connectivity` is the old (still useful,
# now distinctly-named) max-flow metric, kept available but never conflated with the circuit-theory one.
# --------------------------------------------------------------------------------------------------
def test_effective_conductance_unit_square_opposite_corners_matches_circuit_theory_not_max_flow():
    # The audit's own concrete repro: a unit-conductance square (2x2 grid, every cell resistance 1)
    # between opposite corners. Two parallel 2-edge (series-resistance-2) paths -> textbook effective
    # resistance 1 -> effective conductance 1. A max-flow solver instead sums the two paths' raw
    # capacities (1 + 1 = 2), ignoring that they are two resistors in series, not two direct arcs.
    resistance = np.ones((2, 2))  # nodes: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1); opposite corners = 0, 3

    assert effective_conductance(resistance, sources=[0], sinks=[3]) == pytest.approx(1.0)
    assert habitat_connectivity(resistance, sources=[0], sinks=[3]) == pytest.approx(1.0)
    # the max-flow-based metric is kept, under its own name, and is legitimately still 2 for THIS
    # (bottleneck-only) definition -- never the same quantity as effective_conductance.
    assert max_flow_connectivity(resistance, sources=[0], sinks=[3]) == pytest.approx(2.0)


def test_effective_conductance_matches_hand_computed_series_and_parallel_topologies():
    # Straight 1x4 line, HETEROGENEOUS resistances: a simple path graph has no branching, so effective
    # resistance is exactly the sum of edge costs (mean of adjacent cell resistances) -- no series/
    # parallel ambiguity possible.
    line = np.array([1.0, 2.0, 3.0, 4.0])
    edge_costs = [0.5 * (1.0 + 2.0), 0.5 * (2.0 + 3.0), 0.5 * (3.0 + 4.0)]
    assert effective_conductance(line, sources=[0], sinks=[3]) == pytest.approx(1.0 / sum(edge_costs))

    # Two unit resistors purely in series (1x3 line, unit resistance): R = 1 + 1 = 2 -> G = 0.5.
    series = np.ones(3)
    assert effective_conductance(series, sources=[0], sinks=[2]) == pytest.approx(0.5)

    # Two unit resistors purely in parallel: source = the shared middle cell, sinks = both end cells
    # shorted together -- two direct conductance-1 edges from the source terminal to the sink terminal,
    # which add: G = 1 + 1 = 2.
    assert effective_conductance(series, sources=[1], sinks=[0, 2]) == pytest.approx(2.0)

    # 2x2 unit-resistance square, ADJACENT corners: a direct edge (conductance 1) in parallel with the
    # other 3-edge path around the square (series resistance 3 -> conductance 1/3). Parallel
    # conductances add: G = 1 + 1/3 = 4/3.
    square = np.ones((2, 2))
    assert effective_conductance(square, sources=[0], sinks=[1]) == pytest.approx(4.0 / 3.0)


def test_effective_conductance_disconnected_sources_and_sinks_is_exactly_zero():
    # Two 1x2 dumbbells separated by an impassable (+inf resistance) cell: no path from source to sink
    # exists at all. The linear solve must still be well-defined (each side settles to its own
    # terminal's potential everywhere reachable, so no gradient / no current anywhere) and return 0.0
    # exactly, not raise or return something spurious.
    resistance = np.array([1.0, 1.0, np.inf, 1.0, 1.0])
    assert effective_conductance(resistance, sources=[0], sinks=[4]) == 0.0


def test_effective_conductance_ignores_an_isolated_island_elsewhere_in_the_raster():
    # A cell (or group of cells) with no path to EITHER terminal must not perturb the answer or make the
    # reduced linear system singular. Cells 0,1 are connected to each other (source/sink); cell 2 is
    # impassable; cell 3 is a free node with no path to anything.
    resistance = np.array([1.0, 1.0, np.inf, 1.0])
    assert effective_conductance(resistance, sources=[0], sinks=[1]) == pytest.approx(1.0)


def test_effective_conductance_current_conservation_on_the_two_patch_grid():
    # Internal-consistency check independent of any hand-picked closed form: total current leaving the
    # source terminal must equal total current entering the sink terminal (Kirchhoff's current law), and
    # every free (non-terminal) node's net current must be exactly zero (what the linear solve enforces).
    suitability = _two_patch_grid()
    resistance = resistance_raster(_FakeHabitat(suitability), floor=1e-3)
    sources = [_flat(r, 0) for r in range(_ROWS)]
    sinks = [_flat(r, _COLS - 1) for r in range(_ROWS)]

    n, edges = _conductance_edges(resistance)
    boundary = set(sources) | set(sinks)
    reached = _reachable(n, edges, boundary)
    free = [i for i in range(n) if i not in boundary and i in reached]
    laplacian = _graph_laplacian(n, edges)
    v = np.zeros(n)
    v[sources] = 1.0
    v[sinks] = 0.0
    if free:
        free_idx = np.array(free)
        boundary_idx = np.array(sorted(boundary))
        l_ff = laplacian[np.ix_(free_idx, free_idx)]
        l_fd = laplacian[np.ix_(free_idx, boundary_idx)]
        v[free_idx] = np.linalg.solve(l_ff, -(l_fd @ v[boundary_idx]))

    current_out_source = float(np.sum(laplacian[sources, :] @ v))
    current_into_sink = float(-np.sum(laplacian[sinks, :] @ v))
    assert current_out_source == pytest.approx(current_into_sink, rel=1e-9)

    free_residual = laplacian[free, :] @ v if free else np.array([])
    if free_residual.size:
        assert np.max(np.abs(free_residual)) < 1e-8

    assert effective_conductance(resistance, sources, sinks) == pytest.approx(current_out_source, rel=1e-9)


# --------------------------------------------------------------------------------------------------
# MXR-080-0072: the super-source/-sink arc capacity used to be a hard-coded `1e6`, which silently capped
# reported connectivity for any network whose real conductances exceed it (e.g. two cells of resistance
# `1e-9`, conductance `1e9`). The capacity is now derived dynamically from the network itself.
# --------------------------------------------------------------------------------------------------
def test_connectivity_not_silently_capped_at_the_old_1e6_constant_for_high_conductance_cells():
    # Two cells of resistance 1e-9 -> conductance 1e9, far above the old fixed 1e6 cap.
    resistance = np.array([1e-9, 1e-9])
    conductance = effective_conductance(resistance, sources=[0], sinks=[1])
    assert conductance > 1_000_000.0
    assert conductance == pytest.approx(1e9, rel=1e-6)

    # the max-flow-based metric shares the same (now-fixed) terminal-capacity derivation.
    max_flow_value = max_flow_connectivity(resistance, sources=[0], sinks=[1])
    assert max_flow_value > 1_000_000.0
    assert max_flow_value == pytest.approx(1e9, rel=1e-6)


# --------------------------------------------------------------------------------------------------
# Input validation (MXR-080-0072): resistance, floor, raster dimensions, terminal indices, and
# source/sink disjointness must all be validated rather than silently mishandled.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [float("nan"), 0.0, -1.0, -np.inf])
def test_effective_conductance_rejects_invalid_resistance_entries(bad_value):
    resistance = np.array([1.0, 1.0, bad_value, 1.0])
    with pytest.raises(ValueError):
        effective_conductance(resistance, sources=[0], sinks=[3])


def test_effective_conductance_accepts_plus_inf_resistance_as_the_impassable_sentinel():
    # +inf is the documented "impassable cell" convention (see least_cost_corridor/fragmentation_impact)
    # and must NOT be rejected -- only NaN/-inf/zero/negative are invalid.
    resistance = np.array([1.0, 1.0, np.inf, 1.0])
    assert effective_conductance(resistance, sources=[0], sinks=[1]) == pytest.approx(1.0)


def test_effective_conductance_rejects_empty_raster():
    with pytest.raises(ValueError):
        effective_conductance(np.array([]), sources=[], sinks=[])


@pytest.mark.parametrize(
    "sources,sinks",
    [
        ([], [1]),  # empty sources
        ([0], []),  # empty sinks
        ([5], [1]),  # source out of bounds (raster has 4 cells: 0..3)
        ([0], [5]),  # sink out of bounds
        ([0, 1], [1, 2]),  # overlapping node (1) on both sides
        ([0], [0]),  # identical single node on both sides
    ],
)
def test_effective_conductance_rejects_invalid_terminals(sources, sinks):
    resistance = np.ones(4)
    with pytest.raises(ValueError):
        effective_conductance(resistance, sources=sources, sinks=sinks)


def test_effective_conductance_dedupes_repeated_terminal_indices_without_error():
    # a node repeated within the SAME side is harmless (the same physical cell) and must not raise, nor
    # double-count that node's current.
    resistance = np.ones((2, 2))
    assert effective_conductance(resistance, sources=[0, 0], sinks=[3]) == pytest.approx(1.0)
