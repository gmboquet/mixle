"""Biodiversity impact, habitat connectivity, and reclamation offsets (workstream N; N4 + N6).

Two related pieces share this module because they share the same underlying object -- an N1
:class:`~mixle.analysis.sdm.HabitatModel`'s fitted suitability field:

* **N4 -- connectivity** (:func:`resistance_raster`, :func:`least_cost_corridor`,
  :func:`habitat_connectivity`, :func:`fragmentation_impact`): graph resistance on a habitat-cost raster.
  Suitability is inverted into a per-cell movement *cost*; :class:`mixle.relations.ShortestPath` gives the
  cheapest corridor between two patches and :func:`mixle.relations.max_flow` over the equivalent
  conductance graph gives a circuit-theory "effective current" connectivity between patch sets.
  :func:`fragmentation_impact` scores a candidate mine footprint (H3) by how much it raises corridor
  resistance / drops connectivity relative to the unmined baseline -- the population-viability proxy N6
  and J's objective consume.
* **N6 -- reclamation ecology & biodiversity offsets** (:func:`habitat_offset_liability`,
  :func:`no_net_loss_constraint`): prices the habitat impact of a mine footprint as a liability the same
  shape as J6's other priced terms (reclamation/remediation, health, carbon) and emits the companion
  no-net-loss hard constraint, so biodiversity offsets trade off against grade/cost/carbon inside ONE
  risk-adjusted objective instead of being a separate side calculation. ``habitat_offset_liability``/
  ``no_net_loss_constraint`` work off the same "lost habitat-hectare-equivalents" quantity: the fitted
  suitability field (``HabitatModel.mean``, i.e. ``lambda_c``) times per-cell area, summed over whatever
  footprint of cells a candidate mine plan disturbs.

Every function here reads only duck-typed attributes off ``habitat`` (``.mean``, optionally
``.cell_area``) or takes plain arrays, so anything satisfying the IC-1 ``Posterior`` surface over a
suitability field -- in particular N1's ``HabitatModel`` -- works; the ``HabitatModel`` type hint is a
forward reference (evaluated only under ``TYPE_CHECKING``), so this module has no hard runtime dependency
on ``mixle.analysis.sdm``. The N4 connectivity functions likewise never edit ``mixle.relations`` symbols,
only call the frozen ``ShortestPath``/``max_flow``/``min_cut`` surface -- imported lazily inside each
function (rather than at module scope) because ``mixle.relations`` transitively imports ``mixle.stats``,
and this module is loaded from ``mixle.analysis.__init__`` while ``mixle.stats``/``mixle.inference``/
``mixle.analysis``/``mixle.reason`` are, in some import orders, still themselves mid-initialization
(a pre-existing, order-sensitive circular-import chain across those four packages); deferring the import
to call time avoids perturbing that chain's timing instead of trying to fix it here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mixle.analysis.sdm import HabitatModel

__all__ = [
    "resistance_raster",
    "least_cost_corridor",
    "habitat_connectivity",
    "fragmentation_impact",
    "habitat_offset_liability",
    "no_net_loss_constraint",
]

# A numerically-safe stand-in for "infinite" super-source/-sink attachment capacity in the connectivity
# max-flow graph: far above any real cell-to-cell conductance (bounded by ``1 / floor`` on the resistance
# side) so the super-arcs never themselves bottleneck the flow, without risking inf-arithmetic (inf - inf)
# in `mixle.relations.max_flow`'s residual-graph bookkeeping.
_BIG_CAPACITY = 1.0e6


def _rook_offsets(ndim: int) -> list[tuple[int, ...]]:
    """Unit steps along each axis (+/-1): axis-aligned ("rook"; 4-connected in 2-D) grid neighbours."""
    offsets = []
    for axis in range(ndim):
        for step in (-1, 1):
            offset = [0] * ndim
            offset[axis] = step
            offsets.append(tuple(offset))
    return offsets


def _neighbor_flat_indices(flat_idx: int, shape: tuple[int, ...], offsets: list[tuple[int, ...]]) -> list[int]:
    """In-bounds rook-neighbour flat (row-major/``np.ravel``) indices of ``flat_idx`` on a ``shape`` grid."""
    coord = np.unravel_index(int(flat_idx), shape)
    out = []
    for offset in offsets:
        nb = tuple(c + o for c, o in zip(coord, offset, strict=True))
        if all(0 <= x < s for x, s in zip(nb, shape, strict=True)):
            out.append(int(np.ravel_multi_index(nb, shape)))
    return out


def _edge_cost(flat_resistance: np.ndarray, i: int, j: int) -> float:
    """Movement cost of the edge between adjacent cells ``i``/``j``: the mean of their per-cell resistance."""
    return float(0.5 * (flat_resistance[i] + flat_resistance[j]))


def resistance_raster(habitat: HabitatModel, *, floor: float = 1e-3) -> np.ndarray:
    """Per-cell movement cost from a fitted suitability field: ``cost_c = 1 / max(suitability_c, floor)``.

    Reads only ``habitat.mean`` (N1's fitted intensity field ``lambda_c``); ``floor`` keeps near-zero
    suitability cells at a large-but-finite cost (near-impermeable) rather than blowing up to ``inf``, so
    the resulting raster is always safe to feed straight into :func:`least_cost_corridor` /
    :func:`habitat_connectivity` without any further cleanup.
    """
    suitability = np.asarray(habitat.mean, dtype=np.float64)
    return 1.0 / np.maximum(suitability, float(floor))


def least_cost_corridor(resistance: np.ndarray, patch_a: int, patch_b: int) -> tuple[float, list[int]]:
    """Cheapest movement path between two cells of a resistance raster (the least-cost corridor).

    ``resistance`` is an ``n``-dimensional cost raster (e.g. from :func:`resistance_raster`); ``patch_a``/
    ``patch_b`` are flat (row-major/``np.ravel``) cell indices into it. Delegates to
    :class:`mixle.relations.ShortestPath` over the rook-adjacency grid graph with each edge weighted by
    :func:`_edge_cost` (mean resistance of its two endpoint cells); a cell resistance of ``inf`` (a mined
    footprint, :func:`fragmentation_impact`) makes every edge touching it non-finite and so unusable,
    effectively removing that cell from the graph. Lower resistance = better connected.

    ``ShortestPath``'s underlying engine (``mixle.relations.best_first_paths``) is a pure best-first search
    with no closed/visited set -- intentional there, since it is also used to enumerate *k* best (not just
    the single best) paths, where revisiting a state via a costlier path is legitimate output. On a cyclic
    grid graph asked for only its single best path (``k=1``), that same lack of a closed set is
    combinatorially explosive (every back-and-forth detour is a distinct, never-pruned path). Since
    ``successors(state)`` is called exactly once per pop and pops come out in non-decreasing cost order,
    the first call for a given state is provably its cheapest arrival (Dijkstra's invariant, valid here
    because every edge cost is non-negative) -- so a small closed set local to this call, never touching
    ``relations.py``, turns the search back into ordinary Dijkstra: each state's successors are computed
    once and ``[]`` (already settled) on every later call.

    Returns:
        ``(corridor_resistance, path)``: the total path resistance and the visited flat cell indices from
        ``patch_a`` to ``patch_b`` inclusive; ``(inf, [])`` if no finite-cost path exists.
    """
    from mixle.relations import ShortestPath

    r = np.asarray(resistance, dtype=np.float64)
    shape = r.shape
    flat = r.reshape(-1)
    offsets = _rook_offsets(r.ndim)
    settled: set[int] = set()

    def successors(node: int) -> list[tuple[int, float]]:
        if node in settled:  # already expanded via an earlier (cheaper-or-equal) pop; nothing new to offer
            return []
        settled.add(node)
        out = []
        for nb in _neighbor_flat_indices(node, shape, offsets):
            cost = _edge_cost(flat, node, nb)
            if np.isfinite(cost):
                out.append((nb, cost))
        return out

    target = int(patch_b)
    relation = ShortestPath(int(patch_a), successors, is_goal=lambda c: c == target, sense="min")
    solution = relation.solve()
    if solution is None:
        return float("inf"), []
    return float(solution.objective), list(solution.value)


def _conductance_network(
    resistance: np.ndarray, sources: Sequence[int], sinks: Sequence[int]
) -> tuple[np.ndarray, int, int]:
    """Build the ``(n + 2, n + 2)`` super-source/-sink conductance capacity matrix for max-flow/min-cut.

    Node ``i < n`` is grid cell ``i`` (row-major flat index); node ``n`` is a super-source wired to every
    cell in ``sources`` and node ``n + 1`` a super-sink wired from every cell in ``sinks``, each at
    :data:`_BIG_CAPACITY`. Rook-adjacent cells get a *symmetric* capacity (current can flow either way)
    equal to the conductance ``1 / edge_cost`` of the movement cost between them (the circuit-theory
    analogue of :func:`least_cost_corridor`'s edge cost); a non-finite edge cost (a mined-out cell) yields
    zero conductance, i.e. no arc.
    """
    r = np.asarray(resistance, dtype=np.float64)
    shape = r.shape
    n = r.size
    flat = r.reshape(-1)
    offsets = _rook_offsets(r.ndim)
    cap = np.zeros((n + 2, n + 2), dtype=np.float64)
    source_node, sink_node = n, n + 1
    for node in range(n):
        for nb in _neighbor_flat_indices(node, shape, offsets):
            if nb <= node:
                continue  # each undirected pair visited once, from its lower-indexed endpoint
            cost = _edge_cost(flat, node, nb)
            conductance = 1.0 / cost if np.isfinite(cost) and cost > 0.0 else 0.0
            cap[node, nb] = conductance
            cap[nb, node] = conductance
    for s in sources:
        cap[source_node, int(s)] = _BIG_CAPACITY
    for t in sinks:
        cap[int(t), sink_node] = _BIG_CAPACITY
    return cap, source_node, sink_node


def habitat_connectivity(resistance: np.ndarray, sources: Sequence[int], sinks: Sequence[int]) -> float:
    """Effective connectivity ("current") between ``sources`` and ``sinks`` over a resistance raster.

    Builds a conductance graph (arc capacity ``1 / edge_cost`` between rook-adjacent cells) with a
    super-source over ``sources`` and a super-sink over ``sinks``, then returns
    :func:`mixle.relations.max_flow`'s value between them -- the maximum current the habitat network can
    carry from the source patch(es) to the sink patch(es), the standard circuit-theory proxy for landscape
    connectivity (McRae et al. 2008). Higher = better connected.
    """
    from mixle.relations import max_flow

    cap, source_node, sink_node = _conductance_network(resistance, sources, sinks)
    value, _flow = max_flow(cap, source_node, sink_node)
    return float(value)


def fragmentation_impact(
    resistance: np.ndarray,
    footprint_mask: np.ndarray,
    sources: Sequence[int],
    sinks: Sequence[int],
) -> dict:
    """Habitat-connectivity impact of a mine footprint: baseline vs. mined-out corridor/connectivity.

    Sets every ``footprint_mask`` cell's resistance to ``inf`` (impassable / zero conductance -- the H3
    mine plan) and recomputes both :func:`least_cost_corridor` (between the first ``sources``/``sinks``
    cell, taken as the representative patch-to-patch corridor endpoints -- a judgment call: the public API
    only takes cell *sets* here, not a single designated pair) and :func:`habitat_connectivity` (over the
    full ``sources``/``sinks`` sets) before and after. This is the population-viability proxy N6/J feed
    into the biodiversity-offset objective: a footprint that severs the only corridor raises
    ``corridor_resistance`` and drops ``connectivity`` sharply; one that misses every real corridor leaves
    both essentially unchanged.

    Returns:
        A dict with ``corridor_resistance_baseline``/``corridor_resistance_mined``,
        ``connectivity_baseline``/``connectivity_mined``, ``delta`` (``connectivity_baseline -
        connectivity_mined``, i.e. connectivity *lost*), and ``mincut_edges`` -- the *baseline* network's
        minimum-cut arcs (:func:`mixle.relations.min_cut`), i.e. the single weakest link a footprint would
        need to sever to maximally damage connectivity.
    """
    from mixle.relations import min_cut

    resistance = np.asarray(resistance, dtype=np.float64)
    mask = np.asarray(footprint_mask, dtype=bool)
    if mask.shape != resistance.shape:
        raise ValueError(f"footprint_mask shape {mask.shape} does not match resistance shape {resistance.shape}")
    mined = resistance.copy()
    mined[mask] = np.inf

    sources = list(sources)
    sinks = list(sinks)
    patch_a, patch_b = int(sources[0]), int(sinks[0])

    corridor_baseline, _ = least_cost_corridor(resistance, patch_a, patch_b)
    corridor_mined, _ = least_cost_corridor(mined, patch_a, patch_b)

    connectivity_baseline = habitat_connectivity(resistance, sources, sinks)
    connectivity_mined = habitat_connectivity(mined, sources, sinks)

    cap, source_node, sink_node = _conductance_network(resistance, sources, sinks)
    n = resistance.size
    _cut_value, _side, cut_edges = min_cut(cap, source_node, sink_node)
    mincut_edges = [(u, v) for u, v in cut_edges if u < n and v < n]

    return {
        "corridor_resistance_baseline": corridor_baseline,
        "corridor_resistance_mined": corridor_mined,
        "connectivity_baseline": connectivity_baseline,
        "connectivity_mined": connectivity_mined,
        "delta": connectivity_baseline - connectivity_mined,
        "mincut_edges": mincut_edges,
    }


def _lost_equivalents(plan_footprint: Any, habitat: HabitatModel) -> tuple[np.ndarray, float]:
    """Per-cell and total "lost habitat-hectare-equivalents" over the footprint.

    ``per_cell_c = footprint_c * suitability_c * area_c``; the total is its sum. ``area`` falls back to
    all-ones (unit cells) when ``habitat`` carries no ``cell_area`` attribute.
    """
    footprint = np.asarray(plan_footprint, dtype=bool)
    suitability = np.asarray(habitat.mean, dtype=np.float64)
    if footprint.shape != suitability.shape:
        raise ValueError(
            f"plan_footprint shape {footprint.shape} does not match habitat.mean shape {suitability.shape}"
        )
    area = np.asarray(getattr(habitat, "cell_area", np.ones_like(suitability)), dtype=np.float64)
    if area.shape != suitability.shape:
        raise ValueError(f"habitat.cell_area shape {area.shape} does not match habitat.mean shape {suitability.shape}")
    per_cell = footprint.astype(np.float64) * suitability * area
    return per_cell, float(per_cell.sum())


def habitat_offset_liability(
    plan_footprint: np.ndarray,
    habitat: HabitatModel,
    *,
    offset_ratio: float,
    unit_offset_cost: float,
) -> float:
    """Priced biodiversity-offset liability of disturbing ``plan_footprint`` (a J6 priced-objective term).

    ``lost_equivalents = sum_{c in footprint} suitability_c * area_c`` (suitability = N1's fitted
    ``HabitatModel.mean``); the liability is ``offset_ratio * lost_equivalents * unit_offset_cost`` -- an
    additive dollar term the same shape J6's ``priced_liabilities`` already sums for carbon/health/
    remediation (workstream-J.md J6). ``offset_ratio=0`` or ``unit_offset_cost=0`` reduces this to zero,
    i.e. no biodiversity-offset requirement.

    Because ``lost_equivalents`` is linear in the boolean footprint, the *per-cell rate*
    ``offset_ratio * unit_offset_cost * suitability_c * area_c`` is itself a valid per-block deduction a
    MILP-based optimizer (H4/J6's ``risk_adjusted_plan``) can net directly out of expected per-block
    profit -- this function is the scalar evaluator for a given (candidate or solved) footprint.
    """
    _, lost = _lost_equivalents(plan_footprint, habitat)
    return float(offset_ratio) * lost * float(unit_offset_cost)


def no_net_loss_constraint(
    plan_footprint: np.ndarray,
    habitat: HabitatModel,
    *,
    offset_ratio: float,
) -> dict:
    """Hard no-net-loss constraint payload: created/purchased offsets >= ``offset_ratio * lost_equivalents``.

    Returns a dict carrying both the raw quantities (``lost_equivalents``, ``per_cell_lost_equivalents``,
    ``required_offset``) and a solver-agnostic linear-constraint row in this repo's standard ``coeffs @ x
    <= bound`` convention (``mixle.relations``/``mixle.stochastic_opt``'s ``a_ub`` rows), expressed over a
    single ``offsets_created`` decision variable: ``coeffs=[-1.0]``, ``bound=-required_offset`` encodes
    ``-offsets_created <= -required_offset``, i.e. ``offsets_created >= required_offset``. Placing that row
    (and the ``offsets_created`` column it references) into the wider extraction/offset-purchase decision
    space is H4/J6's job -- this module never edits their MILP variable indexing, only hands them the row.
    """
    per_cell, lost = _lost_equivalents(plan_footprint, habitat)
    required = float(offset_ratio) * lost
    return {
        "lost_equivalents": lost,
        "per_cell_lost_equivalents": per_cell,
        "required_offset": required,
        "variable": "offsets_created",
        "coeffs": np.array([-1.0]),
        "bound": -required,
        "sense": ">=",
    }
