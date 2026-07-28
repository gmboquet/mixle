"""N2 -- critical-habitat & listed-species constraints into a network optimizer (workstream N; IC-9,
IC-12, IC-1).

A general regulatory-exclusion constraint: turns N1's fitted habitat-suitability field
(:class:`~mixle.analysis.sdm.HabitatModel`, IC-12/IC-1) and a set of typed, citation-backed
listed-species records (``mixle_knowledge.contracts.ListedSpecies`` -- this module never imports that
package, it only duck-types the two attributes it needs, so core mixle carries no hard dependency on
the paired ``mixle-knowledge`` contract) into a boolean exclusion mask over network nodes, folded into
an IC-9-shaped network payload -- the one thing ANY siting/routing network optimizer needs to respect
a hard no-build constraint over critical habitat or listed-species range, not just a mine-planning
network optimizer (this module's worked instantiation, hence "block" throughout for "node"). This
mirrors G9's ``mixle_pde/reclamation.py:apply_env_constraints`` almost exactly -- same "excluded
blocks become forbidden nodes / zero-capacity arcs" payload shape -- but for critical-habitat/listed-
species law rather than seepage/subsidence risk.

Per the work-plan non-goal, this module never imports or calls the network-flow solver itself (no
dependency on ``mixle.relations``, and no edit to it); it only produces the payload H1's
``min_cost_flow``/``network_design`` (or H4's stochastic optimizer) reads.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- type-checking only, no runtime dependency on either package
    from mixle_knowledge.contracts import ListedSpecies

    from mixle.analysis.sdm import HabitatModel

__all__ = ["critical_habitat_exclusion", "apply_habitat_constraints"]


def _nonnegative_integer(value: Any, *, name: str) -> int:
    """Validate a topology radius/count without Boolean or fractional coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean nonnegative integer.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-Boolean nonnegative integer.") from exc
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative, got {integer}.")
    return int(integer)


def _exact_flag(obj: Any, name: str, *, context: str) -> bool:
    """Read a Boolean policy/honesty flag off ``obj``, requiring an actual Boolean.

    A missing attribute is a legitimate absent flag and reads as ``False``; a *present* one must be a
    real Boolean. ``bool("false")`` is ``True``, so a designation or honesty flag carried through
    serialized text could invert the statutory exclusion it names -- making a species qualify as
    critical habitat, or marking a habitat field prior-dominated, on the strength of the string
    (MXR-080-1588). These flags decide a legal exclusion, so a non-Boolean is a caller error.
    """
    value = getattr(obj, name, None)
    if value is None:
        return False
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"critical_habitat_exclusion: {context} {name} must be an actual Boolean, got {value!r}")
    return bool(value)


def _finite_scalar(value: Any, *, name: str) -> float:
    """Validate one finite non-Boolean numeric scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean finite scalar.")
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite scalar.")
    scalar = float(arr)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return scalar


def _boolean_or_binary_mask(raw_mask: Any, *, name: str, expected_length: int | None = None) -> np.ndarray:
    """Return a one-dimensional Boolean mask, accepting only Boolean or exact numeric 0/1 values."""
    arr = np.asarray(raw_mask)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional Boolean/binary mask, got shape {arr.shape}.")
    if expected_length is not None and arr.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},), got {arr.shape}.")
    if arr.dtype == np.bool_:
        return np.array(arr, dtype=bool, copy=True)
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain only Boolean or numeric binary values.")
    if not np.all(np.isfinite(arr)) or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} must contain only finite binary values 0 or 1.")
    return np.array(arr, dtype=bool, copy=True)


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
    for _ in range(buffer_cells):
        shifted_left = np.zeros_like(grown)
        shifted_left[1:] = grown[:-1]
        shifted_right = np.zeros_like(grown)
        shifted_right[:-1] = grown[1:]
        grown = grown | shifted_left | shifted_right
    return grown


def _species_models(habitat: Any) -> dict[str, Any]:
    """Normalize ``habitat`` into a ``{species_id: HabitatModel}`` mapping (MXR-080-1591).

    Accepts either a single :class:`~mixle.analysis.sdm.HabitatModel` -- keyed by its own
    ``species_id`` -- or an explicit species-to-model mapping. A model with no ``species_id`` is
    rejected: without an identity there is nothing to match against, which is exactly how a field
    fitted for species A came to impose species B's legal exclusion.
    """
    models = habitat if isinstance(habitat, dict) else {None: habitat}
    resolved: dict[str, Any] = {}
    for key, model in models.items():
        identity = getattr(model, "species_id", None)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(
                "critical_habitat_exclusion: every HabitatModel must carry a non-empty species_id so "
                "the field can be matched to the listed species whose exclusion it imposes; got "
                f"species_id={identity!r}."
            )
        if key is not None and key != identity:
            raise ValueError(
                f"critical_habitat_exclusion: habitat mapping key {key!r} does not match the model's own "
                f"species_id {identity!r}."
            )
        if identity in resolved:
            raise ValueError(f"critical_habitat_exclusion: two habitat models supplied for species {identity!r}.")
        resolved[identity] = model
    return resolved


def critical_habitat_exclusion(
    habitat: HabitatModel | dict[str, HabitatModel],
    listed: Sequence[ListedSpecies],
    *,
    suitability_cut: float,
    buffer_cells: int = 0,
) -> np.ndarray:
    """Boolean no-mine mask over blocks (work-plan algorithm steps 1-2).

    For every ``listed`` record whose ``critical_habitat`` flag is set, THAT SPECIES' OWN fitted
    suitability field contributes its ``lambda_c >= suitability_cut`` cells (via IC-12's
    ``critical_habitat_mask``) to the exclusion; a record with ``critical_habitat=False`` (a species
    that is tracked but has no critical-habitat designation) contributes nothing. If no ``listed``
    record qualifies, nothing is excluded. The per-species contributions are unioned (``OR``).

    ``habitat`` is either a single :class:`~mixle.analysis.sdm.HabitatModel` or an explicit
    ``{species_id: HabitatModel}`` mapping, and every qualifying listed species must have a model whose
    ``species_id`` matches it exactly (MXR-080-1591). Previously any qualifying record applied whatever
    single field happened to be passed in, and no comparison was made at all: a model fitted for species
    A could impose species B's legal exclusion, and several listed species all collapsed onto species
    A's same field. A qualifying species with no matching model raises rather than borrowing another
    species' range.

    When a matched fit is prior-dominated (``habitat``'s own honesty flag: not enough presence data has
    yet informed the field), there isn't enough evidence to clear *any* block, so the mask is excluded
    conservatively in full (every block is treated as potential critical habitat) rather than
    optimistically reporting whatever the under-determined field happens to say.

    ``buffer_cells`` conservatively dilates the resulting mask outward (:func:`_dilate_conservatively`)
    to approximate a regulatory buffer around a designation boundary.

    A real ``CriticalHabitatDesignation`` polygon (a regulator's own mapped boundary, independent of the
    fitted suitability field) is not rasterized here: that requires resolving its ``SpatialBounds`` (real
    ``crs`` coordinates) onto this function's abstract block-index grid, which is exactly the
    covariate/CRS-ingest machinery N1 scoped out (B-series, not yet landed) -- see the PR notes for this
    documented gap. Once a grid-registration utility lands, folding a designation in is an additional
    ``OR`` into ``mask`` before the buffer dilation, using the same conservative-inclusion rule.

    Args:
        habitat: N1's fitted habitat-suitability field(s) (IC-12 ``HabitatModel``, satisfies IC-1
            ``Posterior``) -- one model, or a ``{species_id: HabitatModel}`` mapping.
        listed: ``ListedSpecies`` records (``mixle_knowledge.contracts``); ``critical_habitat``,
            ``species_id`` (to match the record to its own fitted field) and, when informativeness must
            be checked, the matched model's own honesty flag are consulted here -- every other field
            (citation, jurisdiction, listing status) is provenance the caller and downstream audit trail
            carry, not logic this function branches on.
        suitability_cut: the fitted-intensity threshold passed to ``critical_habitat_mask``.
        buffer_cells: conservative dilation radius, in blocks, applied after the union.

    Returns:
        A ``(K,)`` boolean array, ``True`` where the block is excluded (no-mine).
    """
    suitability_cut = _finite_scalar(suitability_cut, name="suitability_cut")
    if suitability_cut < 0.0:
        raise ValueError(f"suitability_cut must be nonnegative, got {suitability_cut}.")
    buffer_cells = _nonnegative_integer(buffer_cells, name="buffer_cells")
    models = _species_models(habitat)

    num_cells: int | None = None
    field_means: dict[str, np.ndarray] = {}
    for identity, model in models.items():
        habitat_mean = np.asarray(model.mean)
        if habitat_mean.ndim != 1 or habitat_mean.size == 0:
            raise ValueError(
                f"habitat.mean must be a non-empty one-dimensional cell field, got shape {habitat_mean.shape}."
            )
        if not np.all(np.isfinite(habitat_mean)):
            raise ValueError("habitat.mean must be finite.")
        if num_cells is None:
            num_cells = habitat_mean.shape[0]
        elif habitat_mean.shape[0] != num_cells:
            raise ValueError(
                "critical_habitat_exclusion: every habitat model must share one block grid; species "
                f"{identity!r} has {habitat_mean.shape[0]} cells, expected {num_cells}."
            )
        field_means[identity] = habitat_mean
    assert num_cells is not None  # _species_models always yields at least one entry

    qualifies = [species for species in listed if _exact_flag(species, "critical_habitat", context="listed species")]
    if not qualifies:
        return np.zeros(num_cells, dtype=bool)

    mask = np.zeros(num_cells, dtype=bool)
    rng = np.random.default_rng(0)
    for species in qualifies:
        identity = getattr(species, "species_id", None)
        # Exact match only (MXR-080-1591): borrowing another species' fitted range to impose this
        # species' statutory exclusion is not a conservative fallback, it is the wrong exclusion.
        if not isinstance(identity, str) or identity not in models:
            raise ValueError(
                f"critical_habitat_exclusion: no habitat model supplied for listed species {identity!r} "
                f"(models available for {sorted(models)}). A field fitted for one species cannot stand in "
                "for another species' critical-habitat designation."
            )
        model = models[identity]
        honesty = model.derived_quantity(lambda draws: draws, 2, rng)
        if _exact_flag(honesty, "prior_dominated", context=f"habitat model for {identity!r}"):
            # Not enough evidence to clear any block for this species; that alone excludes everything.
            return np.ones(num_cells, dtype=bool)
        raw_base_mask = np.asarray(model.critical_habitat_mask(suitability_cut))
        if raw_base_mask.dtype != np.bool_:
            raise ValueError("habitat.critical_habitat_mask must return a Boolean array.")
        mask = mask | _boolean_or_binary_mask(
            raw_base_mask,
            name="habitat.critical_habitat_mask",
            expected_length=num_cells,
        )

    return _dilate_conservatively(mask, buffer_cells)


def _validate_block_nodes(raw_block_nodes: Any, *, n_blocks: int, n_nodes: int) -> np.ndarray:
    """Validate ``network['block_nodes']`` as exact, in-range, duplicate-free node ids (MXR-080-0092).

    ``block_nodes[i]`` is the network-node index block ``i`` maps onto -- a *positional* array, not a
    set. A bare ``dtype=int`` cast used to run here before any validation: a fractional id (e.g. ``2.7``)
    silently truncated to the wrong node, a negative id (e.g. ``-1``) silently exercised NumPy's
    from-the-end indexing instead of raising, and an out-of-range id only failed once
    :func:`apply_habitat_constraints`'s mutation loop reached it -- after every earlier id in the same
    call had already been applied to the local ``cap`` copy.

    Every id must be an exact integer (an exact-integer-valued float such as ``2.0`` is accepted, a
    fractional one such as ``2.7`` is not -- matching :func:`mixle.analysis.coverage._rarefaction_sizes`'s
    convention for a caller-supplied id/size array) and lie in ``[0, n_nodes)``, the network's own node
    range. Two blocks aliasing the same network node can only arise from a caller mistake (e.g. a
    mis-sized ``block_nodes`` array), so duplicates are rejected rather than silently double-processing
    one node.
    """
    arr = np.asarray(raw_block_nodes)
    if arr.ndim != 1 or arr.shape[0] != n_blocks:
        raise ValueError(
            f"network['block_nodes'] must be a 1-D array with one entry per block ({n_blocks}), got "
            f"shape {arr.shape!r}."
        )
    if arr.dtype == np.bool_ or (
        arr.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in arr.tolist())
    ):
        raise ValueError("network['block_nodes'] must not contain Boolean node identities.")
    if np.issubdtype(arr.dtype, np.integer):
        ids = arr.astype(np.int64)
    elif np.issubdtype(arr.dtype, np.floating):
        farr = arr.astype(np.float64)
        if not np.all(np.isfinite(farr)):
            raise ValueError("network['block_nodes'] must be finite (no NaN or Inf).")
        if not np.array_equal(farr, np.trunc(farr)):
            raise ValueError("network['block_nodes'] must be exact integers (fractional ids are not supported).")
        ids = farr.astype(np.int64)
    else:
        raise ValueError("network['block_nodes'] must contain numeric exact integer node ids.")

    if ids.size and (ids.min() < 0 or ids.max() >= n_nodes):
        raise ValueError(
            f"network['block_nodes'] must be within [0, {n_nodes}) (the network's node count), got {ids.tolist()!r}."
        )

    seen: set[int] = set()
    duplicates: set[int] = set()
    for node_id in ids.tolist():
        if node_id in seen:
            duplicates.add(node_id)
        seen.add(node_id)
    if duplicates:
        raise ValueError(
            f"network['block_nodes'] must not contain duplicate node ids, got {ids.tolist()!r} "
            f"(duplicated: {sorted(duplicates)})"
        )

    return ids


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

    Every input is validated *before* any exclusion is applied, so a rejected call never leaves a
    partially-mutated result (MXR-080-0092): ``cap``/``cost`` must be equal-shape, square, and finite, and
    ``cap`` must additionally be nonnegative (a negative arc capacity has no meaning); ``supply``, when
    supplied, must be a finite length-``n`` vector; and ``block_nodes`` must be exact, unique node ids in
    ``[0, n)`` (see :func:`_validate_block_nodes` for the fractional/negative/out-of-range failure modes
    this replaces).

    Args:
        network: the reference network payload (``cap``/``cost``/``supply``/``block_nodes``, plus any
            ``network_design``-shaped pass-through fields).
        exclusion_mask: the ``(K,)`` boolean no-mine mask from :func:`critical_habitat_exclusion`.

    Returns:
        A new dict: ``cap``/``cost`` (habitat-adjusted), ``forbidden_nodes`` (sorted node-id list), and
        ``supply`` when the caller provided one, plus any pass-through fields.

    Raises:
        ValueError: if ``cap``/``cost`` are not equal-shape finite square arrays, if ``cap`` contains a
            negative entry, if ``supply`` is not a finite length-``n`` vector, or if ``block_nodes`` is not
            a length-matched array of exact, unique, in-range node ids.
    """
    exclusion_mask = _boolean_or_binary_mask(exclusion_mask, name="exclusion_mask")

    cap = np.array(network["cap"], dtype=float, copy=True)
    cost = np.array(network["cost"], dtype=float, copy=True)
    if cap.shape != cost.shape or cap.ndim != 2 or cap.shape[0] != cap.shape[1]:
        raise ValueError("network['cap'] and network['cost'] must be equal-shape square (n, n) arrays.")
    if not np.all(np.isfinite(cap)):
        raise ValueError("network['cap'] must be finite (no NaN or Inf).")
    if np.any(cap < 0.0):
        raise ValueError("network['cap'] must be nonnegative (a negative arc capacity is not meaningful).")
    if not np.all(np.isfinite(cost)):
        raise ValueError("network['cost'] must be finite (no NaN or Inf).")
    n_nodes = cap.shape[0]

    block_nodes = _validate_block_nodes(
        network.get("block_nodes", np.arange(exclusion_mask.shape[0])),
        n_blocks=exclusion_mask.shape[0],
        n_nodes=n_nodes,
    )

    supply: np.ndarray | None = None
    if "supply" in network:
        supply = np.array(network["supply"], dtype=float, copy=True)
        if supply.shape != (n_nodes,):
            raise ValueError(
                f"network['supply'] must be a length-{n_nodes} vector matching the network's node count, "
                f"got shape {supply.shape!r}."
            )
        if not np.all(np.isfinite(supply)):
            raise ValueError("network['supply'] must be finite (no NaN or Inf).")

    # Every check above must pass before any exclusion is applied below -- a rejected call must not leave
    # `cap`/`supply` partially mutated (MXR-080-0092).
    forbidden_nodes = sorted(int(node) for node, excluded in zip(block_nodes, exclusion_mask) if excluded)
    for node in forbidden_nodes:
        cap[node, :] = 0.0
        cap[:, node] = 0.0

    result: dict[str, Any] = {"cap": cap, "cost": cost, "forbidden_nodes": forbidden_nodes}
    if supply is not None:
        if forbidden_nodes:
            supply[forbidden_nodes] = 0.0
        result["supply"] = supply
    for passthrough in ("nodes", "arcs", "fixed_costs", "demands"):
        if passthrough in network:
            result[passthrough] = network[passthrough]

    return result
