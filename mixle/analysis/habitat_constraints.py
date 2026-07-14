"""N2 -- critical-habitat & listed-species constraints into production (workstream N; IC-9, IC-12, IC-1).

Turns N1's fitted habitat-suitability field (:class:`~mixle.analysis.sdm.HabitatModel`, IC-12/IC-1) and
a set of typed, citation-backed listed-species records (``mixle_knowledge.contracts.ListedSpecies`` --
this module never imports that package, it only duck-types the two attributes it needs, so core mixle
carries no hard dependency on the paired ``mixle-knowledge`` contract) into the one thing a mine-planning
network optimizer needs to respect a no-mine constraint: a boolean exclusion mask over blocks, folded
into an IC-9-shaped network payload. This mirrors G9's ``mixle_pde/reclamation.py:apply_env_constraints``
almost exactly -- same "excluded blocks become forbidden nodes / zero-capacity arcs" payload shape -- but
for critical-habitat/listed-species law rather than seepage/subsidence risk.

Per the work-plan non-goal, this module never imports or calls the network-flow solver itself (no
dependency on ``mixle.relations``, and no edit to it); it only produces the payload H1's
``min_cost_flow``/``network_design`` (or H4's stochastic optimizer) reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- type-checking only, no runtime dependency on either package
    from mixle_knowledge.contracts import ListedSpecies

    from mixle.analysis.sdm import HabitatModel

__all__ = ["critical_habitat_exclusion", "apply_habitat_constraints"]


def _dilate_conservatively(mask: np.ndarray, buffer_cells: int) -> np.ndarray:
    """Grow a boolean block mask outward by ``buffer_cells`` adjacent cells on either side.

    Blocks are treated as a flat, ordered sequence (the same abstract cell-index convention N1's
    ``HabitatModel``/``fit_sdm`` use -- resolving a real 2-D/3-D adjacency from a ``crs``-referenced grid
    is covariate/CRS ingest, out of scope here). Each round of dilation only ever turns a cell on, never
    off, so the result is always a superset of ``mask`` -- the buffer is a conservative expansion, not a
    smoothing.
    """
    if buffer_cells <= 0:
        return mask
    grown = mask.copy()
    for _ in range(int(buffer_cells)):
        shifted_left = np.zeros_like(grown)
        shifted_left[1:] = grown[:-1]
        shifted_right = np.zeros_like(grown)
        shifted_right[:-1] = grown[1:]
        grown = grown | shifted_left | shifted_right
    return grown


def critical_habitat_exclusion(
    habitat: HabitatModel,
    listed: Sequence[ListedSpecies],
    *,
    suitability_cut: float,
    buffer_cells: int = 0,
) -> np.ndarray:
    """Boolean no-mine mask over blocks (work-plan algorithm steps 1-2).

    For every ``listed`` record whose ``critical_habitat`` flag is set, the fitted suitability field's
    own ``habitat.critical_habitat_mask(suitability_cut)`` (IC-12) contributes its ``lambda_c >=
    suitability_cut`` cells to the exclusion; a record with ``critical_habitat=False`` (a species that is
    tracked but has no critical-habitat designation) contributes nothing. If no ``listed`` record
    qualifies, nothing is excluded. The per-species contribution is unioned (``OR``) across every
    qualifying species -- with a single shared ``habitat`` field this collapses to one mask, but the
    union step is kept explicit so a future multi-field caller (one ``HabitatModel`` per species) composes
    the same way. When the fit itself is prior-dominated (``habitat``'s own honesty flag: not enough
    presence data has yet informed the field), there isn't enough evidence to clear *any* block, so
    the mask is excluded conservatively in full (every block is treated as potential critical habitat)
    rather than optimistically reporting whatever the under-determined field happens to say.

    ``buffer_cells`` conservatively dilates the resulting mask outward (:func:`_dilate_conservatively`)
    to approximate a regulatory buffer around a designation boundary.

    A real ``CriticalHabitatDesignation`` polygon (a regulator's own mapped boundary, independent of the
    fitted suitability field) is not rasterized here: that requires resolving its ``SpatialBounds`` (real
    ``crs`` coordinates) onto this function's abstract block-index grid, which is exactly the
    covariate/CRS-ingest machinery N1 scoped out (B-series, not yet landed) -- see the PR notes for this
    documented gap. Once a grid-registration utility lands, folding a designation in is an additional
    ``OR`` into ``mask`` before the buffer dilation, using the same conservative-inclusion rule.

    Args:
        habitat: N1's fitted habitat-suitability field (IC-12 ``HabitatModel``, satisfies IC-1
            ``Posterior``).
        listed: ``ListedSpecies`` records (``mixle_knowledge.contracts``); only ``critical_habitat``
            and, when informativeness must be checked, the model's own honesty flag are consulted here
            -- every other field (citation, jurisdiction, listing status) is provenance the caller and
            downstream audit trail carry, not logic this function branches on.
        suitability_cut: the fitted-intensity threshold passed to ``critical_habitat_mask``.
        buffer_cells: conservative dilation radius, in blocks, applied after the union.

    Returns:
        A ``(K,)`` boolean array, ``True`` where the block is excluded (no-mine).
    """
    num_cells = int(np.asarray(habitat.mean).shape[0])
    qualifies = [species for species in listed if bool(getattr(species, "critical_habitat", False))]
    if not qualifies:
        return np.zeros(num_cells, dtype=bool)

    rng = np.random.default_rng(0)
    honesty = habitat.derived_quantity(lambda draws: draws, 2, rng)
    if bool(getattr(honesty, "prior_dominated", False)):
        return np.ones(num_cells, dtype=bool)

    mask = np.zeros(num_cells, dtype=bool)
    base_mask = np.asarray(habitat.critical_habitat_mask(suitability_cut), dtype=bool)
    for _species in qualifies:
        mask = mask | base_mask

    return _dilate_conservatively(mask, buffer_cells)


def apply_habitat_constraints(network: dict[str, Any], exclusion_mask: np.ndarray) -> dict[str, Any]:
    """Fold a critical-habitat exclusion mask into an IC-9-shaped network payload (work-plan algorithm
    step 3); H1/H4 read this, this module never calls the solver.

    ``network`` is a plain mapping over the reference block network, in exactly
    :func:`mixle.relations.min_cost_flow`'s frozen ``(cap, cost, supply)`` shape: ``"cap"``/``"cost"``
    are ``(n, n)`` arc matrices, ``"supply"`` is the optional length-``n`` node supply vector, and
    ``"block_nodes"`` (optional, defaults to ``arange(len(exclusion_mask))``) maps each block index to
    its node index in that network -- the same convention G9's ``apply_env_constraints`` uses.

    Excluded blocks become forbidden nodes: every arc touching one has its capacity zeroed (no flow can
    originate from, or land back on, a no-mine block), and, if that node carried supply, the supply is
    zeroed too (there is nothing left to extract there). Any ``nodes``/``arcs``/``fixed_costs``/
    ``demands`` the caller supplied (the shape :func:`mixle.relations.network_design` itself takes) are
    passed through unchanged alongside ``forbidden_nodes``, so a fixed-charge caller can exclude the same
    nodes from its own arc set.

    Args:
        network: the reference network payload (``cap``/``cost``/``supply``/``block_nodes``, plus any
            ``network_design``-shaped pass-through fields).
        exclusion_mask: the ``(K,)`` boolean no-mine mask from :func:`critical_habitat_exclusion`.

    Returns:
        A new dict: ``cap``/``cost`` (habitat-adjusted), ``forbidden_nodes`` (sorted node-id list), and
        ``supply`` when the caller provided one, plus any pass-through fields.
    """
    exclusion_mask = np.asarray(exclusion_mask, dtype=bool)

    cap = np.array(network["cap"], dtype=float, copy=True)
    cost = np.array(network["cost"], dtype=float, copy=True)
    if cap.shape != cost.shape or cap.ndim != 2 or cap.shape[0] != cap.shape[1]:
        raise ValueError("network['cap'] and network['cost'] must be equal-shape square (n, n) arrays.")

    block_nodes = np.asarray(network.get("block_nodes", np.arange(exclusion_mask.shape[0])), dtype=int)
    if block_nodes.shape[0] != exclusion_mask.shape[0]:
        raise ValueError("network['block_nodes'] must have one entry per block.")

    forbidden_nodes = sorted(int(node) for node, excluded in zip(block_nodes, exclusion_mask) if excluded)
    for node in forbidden_nodes:
        cap[node, :] = 0.0
        cap[:, node] = 0.0

    result: dict[str, Any] = {"cap": cap, "cost": cost, "forbidden_nodes": forbidden_nodes}
    if "supply" in network:
        supply = np.array(network["supply"], dtype=float, copy=True)
        if forbidden_nodes:
            supply[forbidden_nodes] = 0.0
        result["supply"] = supply
    for passthrough in ("nodes", "arcs", "fixed_costs", "demands"):
        if passthrough in network:
            result[passthrough] = network[passthrough]

    return result
