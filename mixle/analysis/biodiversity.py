"""Reclamation ecology & biodiversity offsets (workstream N, N6).

Prices the habitat impact of a mine footprint as a liability the same shape as J6's other priced terms
(reclamation/remediation, health, carbon) and emits the companion no-net-loss hard constraint, so
biodiversity offsets trade off against grade/cost/carbon inside ONE risk-adjusted objective instead of
being a separate side calculation.

``habitat_offset_liability``/``no_net_loss_constraint`` both work off the same "lost habitat-hectare-
equivalents" quantity: the fitted suitability field (an N1 :class:`~mixle.analysis.sdm.HabitatModel`'s
``mean``, i.e. ``lambda_c``) times per-cell area, summed over whatever footprint of cells a candidate mine
plan disturbs. Multiplying by ``offset_ratio`` (how many equivalent hectares must be created/purchased per
hectare lost) and ``unit_offset_cost`` (dollars per offset hectare-equivalent) turns that into a priced
dollar liability; requiring the created/purchased offset to meet-or-exceed ``offset_ratio * lost`` is the
companion hard (no-net-loss) constraint.

Only ``habitat.mean`` (and, if present, ``habitat.cell_area``) is read, so any object satisfying the IC-1
``Posterior`` surface over a suitability field -- in particular N1's ``HabitatModel`` -- works here; the
type hint is a forward reference (evaluated only under ``TYPE_CHECKING``) so this module has no hard
runtime dependency on ``mixle.analysis.sdm``.

This module is the seed of N4's ``analysis/biodiversity.py`` (connectivity metrics: ``resistance_raster``,
``least_cost_corridor``, ``habitat_connectivity``, ``fragmentation_impact`` -- workstream-N.md N4, not
implemented here; see the module Notes in the N6 PR body). N6 only appends the two offset/liability
functions below, per its own Non-goals ("no connectivity metric (N4)").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mixle.analysis.sdm import HabitatModel

__all__ = ["habitat_offset_liability", "no_net_loss_constraint"]


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
