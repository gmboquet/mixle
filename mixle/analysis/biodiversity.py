"""Biodiversity impact, habitat connectivity, and reclamation offsets (workstream N; N4 + N6).

Two related pieces share this module because they share the same underlying object -- an N1
:class:`~mixle.analysis.sdm.HabitatModel`'s fitted suitability field:

* **N4 -- connectivity** (:func:`resistance_raster`, :func:`least_cost_corridor`,
  :func:`effective_conductance` / :func:`habitat_connectivity`, :func:`max_flow_connectivity`,
  :func:`fragmentation_impact`): graph resistance on a habitat-cost raster. Suitability is inverted into a
  per-cell movement *cost*; :class:`mixle.relations.ShortestPath` gives the cheapest corridor between two
  patches, and the equivalent conductance graph feeds two genuinely different connectivity metrics:
  :func:`effective_conductance` (aliased as :func:`habitat_connectivity`) solves the Dirichlet/Laplacian
  boundary-value problem that circuit theory actually means by "effective conductance" / "effective
  current" (McRae et al. 2008's Circuitscape method -- fix the source patch cells at potential 1 and the
  sink patch cells at potential 0, solve for the harmonic potential everywhere else, and read off the
  current the network carries at that unit potential difference), while :func:`max_flow_connectivity` is
  the separate, bottleneck-only maximum-flow quantity over the same conductance graph. An earlier revision
  used max-flow *as* the circuit-theory metric (MXR-080-0071): that is wrong, because series conductances
  add reciprocally while max-flow only sees the tightest cut -- the two agree only on a single-edge
  network. Use :func:`habitat_connectivity` for the population-viability proxy; :func:`max_flow_connectivity`
  is kept only because bottleneck throughput is occasionally independently useful on its own terms, never
  as a stand-in for the other. :func:`fragmentation_impact` scores a candidate development footprint (a
  mine footprint, in this module's H3 worked instantiation) by how much it raises corridor resistance /
  drops :func:`habitat_connectivity` relative to the undisturbed baseline -- the population-viability
  proxy N6 and J's objective consume. Nothing about the connectivity graph is mining-specific: the same
  resistance-raster/least-cost-corridor/effective-conductance construction is standard landscape-ecology
  practice for any development footprint (a road, a subdivision, a mine) cutting through habitat.
* **N6 -- reclamation ecology & biodiversity offsets** (:func:`habitat_offset_liability`,
  :func:`no_net_loss_constraint`): prices the habitat impact of a development footprint as a liability
  the same shape as J6's other priced terms (reclamation/remediation, health, carbon) and emits the
  companion no-net-loss hard constraint, so biodiversity offsets trade off against grade/cost/carbon
  inside ONE risk-adjusted objective instead of being a separate side calculation.
  ``habitat_offset_liability``/``no_net_loss_constraint`` work off the same "lost habitat-hectare-
  equivalents" quantity: the fitted suitability field (``HabitatModel.mean``, i.e. ``lambda_c``) times
  per-cell area, summed over whatever footprint of cells a candidate disturbance plan (a mine plan, in
  this module's worked instantiation) disturbs. Suitability, area, and every economic rate/cost feeding
  these two functions are validated finite and sign-correct before they reach a solver or accounting
  payload (MXR-080-0073) -- see :func:`_lost_equivalents` and the ``_require_*`` helpers below.

Every function here reads only duck-typed attributes off ``habitat`` (``.mean``, optionally
``.cell_area``) or takes plain arrays, so anything satisfying the IC-1 ``Posterior`` surface over a
suitability field -- in particular N1's ``HabitatModel`` -- works; the ``HabitatModel`` type hint is a
forward reference (evaluated only under ``TYPE_CHECKING``), so this module has no hard runtime dependency
on ``mixle.analysis.sdm``. :func:`effective_conductance` / :func:`habitat_connectivity` use SciPy's sparse
linear solver; :func:`max_flow_connectivity` and :func:`fragmentation_impact` use this module's sparse
float-capacity push-relabel implementation. Only :func:`least_cost_corridor` calls the frozen
``ShortestPath`` surface, imported lazily because ``mixle.relations`` transitively imports ``mixle.stats``
and this module can be loaded while the statistics/analysis import graph is still initializing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from mixle.analysis.sdm import HabitatModel

__all__ = [
    "resistance_raster",
    "least_cost_corridor",
    "effective_conductance",
    "habitat_connectivity",
    "max_flow_connectivity",
    "fragmentation_impact",
    "habitat_offset_liability",
    "no_net_loss_constraint",
]


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
    return float(0.5 * flat_resistance[i] + 0.5 * flat_resistance[j])


# ---------------------------------------------------------------------------
# Input validation (MXR-080-0072 / MXR-080-0073). Every public entry point below validates its numeric
# inputs' finiteness and physical sign before it reaches a linear solve or an accounting payload, rather
# than silently clamping (the old `resistance_raster`) or propagating NaN/negative values into a
# supposedly hard constraint.
# ---------------------------------------------------------------------------
def _require_finite(value: np.ndarray | float, name: str) -> np.ndarray:
    """Raise unless every entry of ``value`` is finite (no NaN/+-inf)."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return arr


def _require_finite_nonnegative(value: np.ndarray | float, name: str) -> np.ndarray:
    """Raise unless every entry of ``value`` is finite and ``>= 0``."""
    arr = _require_finite(value, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return arr


def _require_finite_positive(value: np.ndarray | float, name: str) -> np.ndarray:
    """Raise unless every entry of ``value`` is finite and strictly ``> 0``."""
    arr = _require_finite(value, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive, got {value!r}")
    return arr


def _require_valid_resistance(resistance: np.ndarray) -> np.ndarray:
    """Raise unless every cell of a resistance raster is finite-and-positive, or the ``+inf`` sentinel.

    ``+inf`` is accepted deliberately: it is this module's standing convention for an impassable/removed
    cell (see :func:`least_cost_corridor`, :func:`fragmentation_impact`), and it correctly yields zero
    conductance on every edge touching it. NaN, ``-inf``, zero, and negative values are always invalid --
    each would make the implied conductance ``1 / resistance`` undefined, sign-flipped, or infinite.
    """
    r = np.asarray(resistance, dtype=np.float64)
    if r.size == 0:
        raise ValueError("resistance raster must not be empty (every dimension must have size >= 1)")
    if np.any(np.isnan(r)):
        raise ValueError("resistance must not contain NaN")
    if np.any(r == -np.inf):
        raise ValueError("resistance must not contain -inf")
    if np.any(r <= 0.0):
        raise ValueError(
            "resistance must be strictly positive; use +inf to mark an impassable/removed cell, not 0 or "
            "a negative value"
        )
    return r


def _require_valid_terminals(sources: Sequence[int], sinks: Sequence[int], n: int) -> tuple[list[int], list[int]]:
    """Raise unless ``sources``/``sinks`` are non-empty, in-bounds, and disjoint node-index sets.

    Returns the de-duplicated, sorted node-index lists (a node repeated within one side is harmless --
    it is the same physical cell -- but must not appear on both sides).
    """

    def exact(values: Sequence[int], label: str) -> list[int]:
        indices: set[int] = set()
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{label} entries must be exact integer node indices, got {value!r}")
            indices.add(int(value))
        return sorted(indices)

    src = exact(sources, "sources")
    snk = exact(sinks, "sinks")
    if not src:
        raise ValueError("sources must not be empty")
    if not snk:
        raise ValueError("sinks must not be empty")
    for label, idxs in (("sources", src), ("sinks", snk)):
        for i in idxs:
            if not (0 <= i < n):
                raise ValueError(f"{label} index {i} out of bounds for a raster with {n} cell(s)")
    overlap = set(src) & set(snk)
    if overlap:
        raise ValueError(f"sources and sinks must be disjoint; overlapping node(s): {sorted(overlap)}")
    return src, snk


def _require_binary_mask(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return an owned Boolean mask, accepting only bool or exact integer 0/1 entries."""
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} shape {raw.shape} does not match required shape {shape}")
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=True)
    if raw.dtype.kind not in "iu" or np.any((raw != 0) & (raw != 1)):
        raise TypeError(f"{name} must contain only Boolean or exact integer 0/1 entries")
    return raw.astype(bool)


def resistance_raster(habitat: HabitatModel, *, floor: float = 1e-3) -> np.ndarray:
    """Per-cell movement cost from a fitted suitability field: ``cost_c = 1 / max(suitability_c, floor)``.

    Reads only ``habitat.mean`` (N1's fitted intensity field ``lambda_c``); ``floor`` keeps near-zero
    suitability cells at a large-but-finite cost (near-impermeable) rather than blowing up to ``inf``, so
    the resulting raster is always safe to feed straight into :func:`least_cost_corridor` /
    :func:`habitat_connectivity` without any further cleanup.

    Raises:
        ValueError: if ``floor`` is not finite and strictly positive, or if ``habitat.mean`` contains a
            NaN/Inf/negative entry. An earlier revision silently clamped negative suitability into the
            same near-impermeable cost as zero suitability (MXR-080-0073), hiding invalid upstream model
            output; a negative suitability now always raises instead of being clamped. Zero suitability
            remains a legitimate input (clamped to ``floor``, as documented above).
    """
    floor = float(_require_finite_positive(floor, "floor"))
    suitability = _require_finite_nonnegative(np.asarray(habitat.mean, dtype=np.float64), "habitat.mean")
    return 1.0 / np.maximum(suitability, floor)


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

    r = _require_valid_resistance(resistance)
    src, snk = _require_valid_terminals([patch_a], [patch_b], r.size)
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

    target = snk[0]
    relation = ShortestPath(src[0], successors, is_goal=lambda c: c == target, sense="min")
    solution = relation.solve()
    if solution is None:
        return float("inf"), []
    return float(solution.objective), list(solution.value)


def _conductance_edges(resistance: np.ndarray) -> tuple[int, list[tuple[int, int, float]]]:
    """Rook-adjacency conductance edges of a resistance raster: ``(n_cells, [(i, j, conductance), ...])``.

    Each undirected pair is emitted once (from its lower-indexed endpoint) with conductance ``1 /
    edge_cost`` (:func:`_edge_cost`); a non-finite or non-positive edge cost -- either endpoint at the
    ``+inf`` impassable-cell sentinel -- yields conductance ``0`` and is omitted entirely (no arc).
    """
    r = np.asarray(resistance, dtype=np.float64)
    shape = r.shape
    n = r.size
    flat = r.reshape(-1)
    offsets = _rook_offsets(r.ndim)
    edges: list[tuple[int, int, float]] = []
    for node in range(n):
        for nb in _neighbor_flat_indices(node, shape, offsets):
            if nb <= node:
                continue  # each undirected pair visited once, from its lower-indexed endpoint
            cost = _edge_cost(flat, node, nb)
            if np.isfinite(cost) and cost > 0.0:
                if cost < 1.0 / np.finfo(np.float64).max:
                    raise ValueError(f"resistance between cells {node} and {nb} implies unrepresentable conductance")
                conductance = 1.0 / cost
                if not np.isfinite(conductance):
                    raise ValueError(f"resistance between cells {node} and {nb} implies unrepresentable conductance")
                edges.append((node, nb, conductance))
    return n, edges


def _graph_laplacian(n: int, edges: list[tuple[int, int, float]]) -> sparse.csr_matrix:
    """Sparse weighted graph Laplacian with memory proportional to nodes plus edges."""
    diagonal = np.zeros(n, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for i, j, c in edges:
        diagonal[i] += c
        diagonal[j] += c
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-c, -c))
    rows.extend(range(n))
    cols.extend(range(n))
    values.extend(diagonal.tolist())
    return sparse.coo_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()


def _reachable(n: int, edges: list[tuple[int, int, float]], start: set[int]) -> set[int]:
    """Node indices reachable from ``start`` via positive-conductance edges (breadth-first search)."""
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i, j, _c in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    seen = set(start)
    queue = deque(start)
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def effective_conductance(resistance: np.ndarray, sources: Sequence[int], sinks: Sequence[int]) -> float:
    """Circuit-theory effective conductance between ``sources`` and ``sinks`` over a resistance raster.

    Solves the standard Dirichlet/Laplacian boundary-value problem for landscape connectivity (McRae et
    al. 2008; the "Circuitscape" formulation): every ``sources`` cell is held at potential ``1`` and every
    ``sinks`` cell at potential ``0`` -- equivalent to shorting each terminal's cells together with an
    ideal conductor, with no arbitrary coupling constant needed (unlike a max-flow super-source/-sink).
    The remaining free cells' potentials solve the reduced linear system ``L_free @ v_free = -L_free,fixed
    @ v_fixed`` for the graph Laplacian ``L`` (edge weight = conductance = ``1 / edge_cost``,
    :func:`_edge_cost`); the total current flowing out of the source terminal at this unit potential
    difference *is* the effective conductance (``I = G @ V``, ``V = 1``), read off directly from the
    Laplacian: current out of node ``i`` is ``(L @ v)[i]``, which is exactly ``0`` at every free node (that
    is what the linear solve enforces) and the true injected/withdrawn current at every fixed node.

    This is a genuinely different quantity from maximum flow (:func:`max_flow_connectivity`): conductances
    in series add reciprocally (two unit conductances in series give an effective conductance of ``0.5``,
    not ``1``), so a max-flow solver -- which only sees bottleneck capacity and ignores series resistance
    entirely -- systematically overstates connectivity whenever current has to cross more than one edge to
    reach the sink. On a unit-conductance square between opposite corners this function returns the
    textbook value ``1``; a max-flow solver over the same graph returns ``2`` (the sum of the two parallel
    corner-to-corner paths' raw capacities, MXR-080-0071).

    Cells with no path at all to either terminal (an isolated island cut off by ``+inf``-resistance
    neighbours on every side) carry no current and are excluded from the linear solve rather than left to
    make it singular; a raster whose ``sources`` have no path to its ``sinks`` at all correctly yields
    ``0.0`` (the linear system still solves -- each side simply settles to its own terminal's potential
    everywhere reachable, so no potential gradient, hence no current, exists anywhere).

    Raises:
        ValueError: on a NaN/negative/zero resistance entry (``+inf``, the impassable-cell sentinel, is
            fine), an empty raster, an out-of-bounds source/sink index, or overlapping source/sink sets.
    """
    r = _require_valid_resistance(resistance)
    n = r.size
    src, snk = _require_valid_terminals(sources, sinks, n)

    _, edges = _conductance_edges(r)
    if not edges:
        return 0.0
    scale = max(c for _i, _j, c in edges)
    scaled_edges = [(i, j, c / scale) for i, j, c in edges]
    if any(c == 0.0 or not np.isfinite(c) for _i, _j, c in scaled_edges):
        raise ValueError("conductance dynamic range cannot be represented after stable rescaling")
    boundary = set(src) | set(snk)
    reached = _reachable(n, scaled_edges, boundary)
    free = [i for i in range(n) if i not in boundary and i in reached]

    laplacian = _graph_laplacian(n, scaled_edges)
    v = np.zeros(n, dtype=np.float64)
    v[src] = 1.0
    v[snk] = 0.0

    if free:
        free_idx = np.array(free)
        boundary_idx = np.array(sorted(boundary))
        L_ff = laplacian[free_idx][:, free_idx]
        L_fd = laplacian[free_idx][:, boundary_idx]
        rhs = -(L_fd @ v[boundary_idx])
        v[free_idx] = spsolve(L_ff, rhs)

    current_scaled = float(np.sum(laplacian[src, :] @ v))
    if not np.isfinite(current_scaled) or current_scaled < -1e-12:
        raise ValueError("effective conductance is not a finite non-negative value")
    result = max(0.0, current_scaled) * scale
    if not np.isfinite(result):
        raise ValueError("effective conductance exceeds the representable finite range")
    return result


def habitat_connectivity(resistance: np.ndarray, sources: Sequence[int], sinks: Sequence[int]) -> float:
    """Domain-friendly alias for :func:`effective_conductance`, the circuit-theory connectivity metric.

    Kept under this name because it is the pre-existing public entry point N4/N6/J callers and tests
    already use; see :func:`effective_conductance` for the full Dirichlet/Laplacian derivation and its
    McRae et al. (2008) citation. Prior to MXR-080-0071 this function actually computed maximum flow
    (now :func:`max_flow_connectivity`) despite the circuit-theory claim in its docstring -- a different,
    systematically-inflated quantity for any network with more than one edge between source and sink. It
    now computes the real thing.
    """
    return effective_conductance(resistance, sources, sinks)


@dataclass(slots=True)
class _FlowArc:
    """Mutable residual-network arc."""

    target: int
    reverse: int
    capacity: float


def _sparse_max_flow(
    resistance: np.ndarray, sources: Sequence[int], sinks: Sequence[int]
) -> tuple[float, list[tuple[int, int]]]:
    """Float-capacity FIFO push-relabel flow using O(nodes + edges) storage."""
    r = _require_valid_resistance(resistance)
    n = r.size
    src, snk = _require_valid_terminals(sources, sinks, n)
    _, edges = _conductance_edges(r)
    if not edges:
        return 0.0, []
    scale = max(c for _i, _j, c in edges)
    scaled_edges = [(i, j, c / scale) for i, j, c in edges]
    if any(c == 0.0 or not np.isfinite(c) for _i, _j, c in scaled_edges):
        raise ValueError("conductance dynamic range cannot be represented after stable rescaling")
    source_node, sink_node = n, n + 1
    node_count = n + 2
    graph: list[list[_FlowArc]] = [[] for _ in range(node_count)]
    original: list[tuple[int, int]] = []

    def add_arc(i: int, j: int, capacity: float, *, internal: bool = False) -> None:
        forward = _FlowArc(j, len(graph[j]), capacity)
        reverse = _FlowArc(i, len(graph[i]), 0.0)
        graph[i].append(forward)
        graph[j].append(reverse)
        if internal:
            original.append((i, j))

    for i, j, c in scaled_edges:
        add_arc(i, j, c, internal=True)
        add_arc(j, i, c, internal=True)
    terminal_capacity = sum(c for _i, _j, c in scaled_edges) + 1.0
    for s in src:
        add_arc(source_node, s, terminal_capacity)
    for t in snk:
        add_arc(t, sink_node, terminal_capacity)

    tolerance = 0.0
    height = np.zeros(node_count, dtype=np.int64)
    excess = np.zeros(node_count, dtype=np.float64)
    current = np.zeros(node_count, dtype=np.int64)
    queued = np.zeros(node_count, dtype=bool)
    active: deque[int] = deque()
    height[source_node] = node_count

    def enqueue(node: int) -> None:
        if node not in (source_node, sink_node) and excess[node] > tolerance and not queued[node]:
            queued[node] = True
            active.append(node)

    for arc in graph[source_node]:
        amount = arc.capacity
        if amount <= tolerance:
            continue
        arc.capacity = 0.0
        graph[arc.target][arc.reverse].capacity += amount
        excess[source_node] -= amount
        excess[arc.target] += amount
        enqueue(arc.target)

    while active:
        node = active.popleft()
        queued[node] = False
        while excess[node] > tolerance:
            arc_index = int(current[node])
            if arc_index >= len(graph[node]):
                residual_heights = [int(height[arc.target]) for arc in graph[node] if arc.capacity > tolerance]
                if not residual_heights:
                    break
                height[node] = min(residual_heights) + 1
                current[node] = 0
                continue
            arc = graph[node][arc_index]
            if arc.capacity > tolerance and height[node] == height[arc.target] + 1:
                amount = min(excess[node], arc.capacity)
                arc.capacity = max(0.0, arc.capacity - amount)
                graph[arc.target][arc.reverse].capacity += amount
                excess[node] -= amount
                excess[arc.target] += amount
                enqueue(arc.target)
            else:
                current[node] += 1
        enqueue(node)

    reached = {source_node}
    queue = deque([source_node])
    while queue:
        u = queue.popleft()
        for arc in graph[u]:
            if arc.capacity > tolerance and arc.target not in reached:
                reached.add(arc.target)
                queue.append(arc.target)
    cut = [(i, j) for i, j in original if i in reached and j not in reached]
    result = float(excess[sink_node]) * scale
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("maximum-flow connectivity is not a finite non-negative value")
    return result, cut


def max_flow_connectivity(resistance: np.ndarray, sources: Sequence[int], sinks: Sequence[int]) -> float:
    """Maximum-flow connectivity between ``sources`` and ``sinks`` over a resistance raster's conductance graph.

    Builds a conductance graph (arc capacity ``1 / edge_cost`` between rook-adjacent cells) with a
    super-source over ``sources`` and a super-sink over ``sinks``, then returns
    :func:`mixle.relations.max_flow`'s value between them -- the maximum flow (bottleneck-limited
    throughput) the habitat network can carry from the source patch(es) to the sink patch(es).

    This is NOT the circuit-theory effective-conductance quantity (:func:`effective_conductance` /
    :func:`habitat_connectivity`): max flow only sees the tightest cut and entirely ignores series
    resistance, so it systematically overstates connectivity whenever current must cross more than one
    edge to reach the sink (MXR-080-0071). This function exists as a distinctly-named, independently
    useful bottleneck-throughput metric in its own right -- never use it as a stand-in for the other.
    """
    value, _cut = _sparse_max_flow(resistance, sources, sinks)
    return value


def fragmentation_impact(
    resistance: np.ndarray,
    footprint_mask: np.ndarray,
    sources: Sequence[int],
    sinks: Sequence[int],
) -> dict:
    """Habitat-connectivity impact of a development footprint: baseline vs. disturbed corridor/connectivity.

    Sets every ``footprint_mask`` cell's resistance to ``inf`` (impassable / zero conductance -- an H3
    mine plan, in this module's worked instantiation) and recomputes both :func:`least_cost_corridor`
    (between the first ``sources``/``sinks``
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
    resistance = _require_valid_resistance(resistance)
    src, snk = _require_valid_terminals(sources, sinks, resistance.size)
    mask = _require_binary_mask(footprint_mask, resistance.shape, "footprint_mask")
    mined = resistance.copy()
    mined[mask] = np.inf

    patch_a, patch_b = src[0], snk[0]

    corridor_baseline, _ = least_cost_corridor(resistance, patch_a, patch_b)
    corridor_mined, _ = least_cost_corridor(mined, patch_a, patch_b)

    connectivity_baseline = habitat_connectivity(resistance, src, snk)
    connectivity_mined = habitat_connectivity(mined, src, snk)
    _cut_value, mincut_edges = _sparse_max_flow(resistance, src, snk)

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

    Raises:
        ValueError: if ``habitat.mean`` or ``habitat.cell_area`` contains a NaN/Inf/negative entry, or if
            ``plan_footprint``/``habitat.cell_area`` do not match ``habitat.mean``'s shape. A NaN or
            negative habitat field used to propagate silently into the no-net-loss accounting below
            (MXR-080-0073); it is now rejected here, at the one place both public functions read it from.
            Also if the per-cell product or its sum is not representable in float64: checking only the
            INPUTS for finiteness left ``suitability * area`` (and the reduction over it) free to overflow
            to ``inf`` from entirely finite inputs near the float64 ceiling, which then published an
            infinite offset requirement and an infinite priced liability from functions whose whole
            contract is that they are validated finite accounting quantities (MXR-080-1572).
    """
    suitability = _require_finite_nonnegative(np.asarray(habitat.mean, dtype=np.float64), "habitat.mean")
    footprint = _require_binary_mask(plan_footprint, suitability.shape, "plan_footprint")
    area = _require_finite_positive(
        np.asarray(getattr(habitat, "cell_area", np.ones_like(suitability)), dtype=np.float64), "habitat.cell_area"
    )
    if area.shape != suitability.shape:
        raise ValueError(f"habitat.cell_area shape {area.shape} does not match habitat.mean shape {suitability.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        per_cell = footprint.astype(np.float64) * suitability * area
        _require_finite(per_cell, "lost habitat-hectare-equivalents (suitability * cell_area over the footprint)")
        total = float(per_cell.sum())
    _require_finite(total, "total lost habitat-hectare-equivalents")
    return per_cell, total


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
    i.e. no biodiversity-offset requirement -- both remain valid, meaningful inputs and are not rejected.

    Because ``lost_equivalents`` is linear in the boolean footprint, the *per-cell rate*
    ``offset_ratio * unit_offset_cost * suitability_c * area_c`` is itself a valid per-block deduction a
    MILP-based optimizer (H4/J6's ``risk_adjusted_plan``) can net directly out of expected per-block
    profit -- this function is the scalar evaluator for a given (candidate or solved) footprint.

    Raises:
        ValueError: if ``offset_ratio``/``unit_offset_cost`` is negative or non-finite, or if
            ``_lost_equivalents`` rejects ``plan_footprint``/``habitat``. A negative ratio or cost used to
            silently turn habitat damage into a "profit" (MXR-080-0073); both are now rejected. Also if
            the priced product itself is not representable in float64 -- finite inputs near the float64
            ceiling used to publish an infinite liability (MXR-080-1572).
    """
    offset_ratio = float(_require_finite_nonnegative(offset_ratio, "offset_ratio"))
    unit_offset_cost = float(_require_finite_nonnegative(unit_offset_cost, "unit_offset_cost"))
    _, lost = _lost_equivalents(plan_footprint, habitat)
    with np.errstate(over="ignore", invalid="ignore"):
        liability = offset_ratio * lost * unit_offset_cost
    return float(_require_finite(liability, "habitat offset liability (offset_ratio * lost * unit_offset_cost)"))


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

    Because that row is emitted ALREADY normalized into the ``<=`` convention, its ``"sense"`` label is
    ``"<="`` -- the label describes the row that is actually here, not the inequality the row expresses
    about ``offsets_created``. It used to read ``">="`` (MXR-080-1570), which made the payload internally
    contradictory: :func:`mixle.analysis.objective.hard_constraints` and
    :func:`mixle.stochastic_opt.risk_adjusted_plan` both trust the label and negate a ``">="`` row into
    ``<=`` form, so a second negation turned the no-net-loss FLOOR into the CEILING
    ``offsets_created <= required_offset`` -- permitting zero offsets and forbidding any offset purchase
    above the bare minimum, i.e. exactly the opposite of a no-net-loss guarantee.

    Raises:
        ValueError: if ``offset_ratio`` is negative or non-finite, or if ``_lost_equivalents`` rejects
            ``plan_footprint``/``habitat`` (NaN/negative suitability or area). A NaN or negative
            ``lost_equivalents`` used to be able to propagate into ``required_offset``, silently weakening
            or inverting a constraint that is supposed to be a hard floor (MXR-080-0073); both inputs are
            now validated before this payload is built. Also if ``required_offset`` itself is not
            representable in float64 -- finite inputs near the float64 ceiling used to publish an infinite
            bound, which is not a hard constraint but an unsatisfiable row (MXR-080-1572).
    """
    offset_ratio = float(_require_finite_nonnegative(offset_ratio, "offset_ratio"))
    per_cell, lost = _lost_equivalents(plan_footprint, habitat)
    with np.errstate(over="ignore", invalid="ignore"):
        required = offset_ratio * lost
    # MXR-080-1572: an infinite bound is not a hard constraint, it is an unsatisfiable row.
    required = float(_require_finite(required, "required_offset (offset_ratio * lost_equivalents)"))
    return {
        "lost_equivalents": lost,
        "per_cell_lost_equivalents": per_cell,
        "required_offset": required,
        "variable": "offsets_created",
        # already-normalized <= row (see docstring): labelling it ">=" made every downstream assembler
        # negate it a second time and invert the constraint (MXR-080-1570).
        "coeffs": np.array([-1.0]),
        "bound": -required,
        "sense": "<=",
    }
