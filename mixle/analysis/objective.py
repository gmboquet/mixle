"""J6 -- priced-liability and hard-constraint assembly for a MILP-based selection objective: wiring,
not new physics (work-plan Sec.7-J, the grand synthesis).

A general pattern for any per-item binary-selection MILP (:func:`mixle.stochastic_opt.
two_stage_stochastic_plan` is this module's worked instantiation, turning grade uncertainty -- IC-1
`Posterior.samples` -- into a CVaR-penalized block-extraction decision): several independent
"pricing" tasks each want to add one more externality term or hard constraint to the same objective
without knowing about each other -- J2's `monte_carlo_npv` (revenue/cost DCF), a G9-style no-mine/
buffer zone, a K6-style public-health exposure cost, an L6-style carbon/emissions price, an N6-style
biodiversity offset (`mixle.analysis.biodiversity.habitat_offset_liability`/`no_net_loss_constraint`,
already landed and explicitly written against this module's shape). None of those tasks are this
module's dependency -- this is the reverse: J6 is the pluggable *framework* they register priced
terms/constraints INTO, so the dependency edge runs G9/K6/L6/N6 -> J6, never J6 -> them
(workstream-J.md J6 header note). That means this module only fixes the *shape* a priced term or a
hard constraint must have to plug in; it derives none of the underlying cost models itself
(Non-goals).

Two pieces:

  * :func:`priced_liabilities` -- nets a plan's per-block environmental/health/carbon externalities
    (remediation, health, carbon) into one additive per-block dollar array (``"total"``) plus the raw
    per-term breakdown, in exactly the shape :func:`mixle.stochastic_opt.risk_adjusted_plan` consumes via
    its ``liabilities`` parameter (``liabilities["total"]``, subtracted from per-block revenue before the
    MILP solves) -- and the same shape any other priced term (N6's ``habitat_offset_liability``, or a
    future workstream's) adds itself into, without ever importing this module.
  * :func:`hard_constraints` -- assembles the ``constraints`` dict `risk_adjusted_plan` consumes: an
    optional hard ``no_mine_mask`` (G9 no-mine/buffer polygons -- blocks forced to ``x_b = 0``) and an
    optional list of linear activity ``caps`` (K6 exposure budgets, L6 water budgets -- ``coeffs @ x <=
    bound`` rows over the block decision vector), normalizing any ``">="``-sense cap into the solver's
    standard ``<=`` convention.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["hard_constraints", "priced_liabilities"]


def _plan_get(plan: Any, key: str) -> Any:
    """Read ``key`` off ``plan``, whether a mapping (dict) or an attribute-bearing object -- the same
    mapping-or-object convention :mod:`mixle.analysis.valuation`'s ``schedule``/``plan`` arguments use."""
    if isinstance(plan, dict):
        return plan[key]
    return getattr(plan, key)


# ---------------------------------------------------------------------------
# Input validation (MXR-080-0106). Every economic quantity assembled into the objective is validated for
# finiteness and economic sign before it can be netted into a liability or a hard constraint, rather than
# silently converted (``np.asarray``/``float``) and trusted: a negative "cost" would net out of revenue as
# a profit increase instead of a liability, a NaN would propagate unguarded into the MILP objective, and
# ``np.asarray(..., dtype=bool)`` on non-boolean data coerces via nonzero-ness -- ``bool(float("nan"))``
# is ``True`` -- turning malformed mask data into a hard exclusion the caller never intended.
# ---------------------------------------------------------------------------
def _require_finite(value: np.ndarray | float, name: str) -> np.ndarray:
    """Raise unless every entry of ``value`` is finite (no NaN/+-inf)."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return arr


def _require_finite_nonnegative(value: np.ndarray | float, name: str) -> np.ndarray:
    """Raise unless every entry of ``value`` is finite and ``>= 0`` -- the sign convention every priced
    liability (a cost or a price) in this module must hold, since a negative "cost" nets out of revenue
    as a profit increase rather than a liability."""
    arr = _require_finite(value, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return arr


def _require_bool_mask(value: Any, name: str) -> np.ndarray:
    """Raise unless ``value`` is a genuinely boolean array (dtype ``bool``), rather than data that is
    merely "truthy" under an implicit ``dtype=bool`` cast -- which maps any nonzero value, including
    ``NaN`` (``bool(float("nan"))`` is ``True``: NaN is only falsy under ``==`` comparisons, never under
    general truthiness), silently to a hard ``True`` exclusion."""
    arr = np.asarray(value)
    if arr.dtype != np.bool_:
        raise ValueError(f"{name} must be an actual boolean array (dtype bool), got dtype {arr.dtype} ({value!r})")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D boolean array, got shape {arr.shape}")
    return arr


def _require_finite_1d(value: Any, name: str) -> np.ndarray:
    """Raise unless ``value`` converts to a finite, genuinely one-dimensional float array -- rejects
    NaN/+-inf entries plus higher-dimensional or ragged input."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return arr


def priced_liabilities(
    plan: Any,
    *,
    carbon_price: float,
    health_cost: Callable[[np.ndarray], np.ndarray],
    remediation_cost: Callable[[np.ndarray], np.ndarray],
) -> dict:
    """Net a plan's per-block environmental/health/carbon liabilities into one additive dollar array.

    ``plan`` is a mapping or attribute-bearing object exposing three per-block, length-``n_blocks``
    array-likes:

    - ``grade``: the block grade (or other geology/complexity proxy) fed to ``remediation_cost`` -- a
      G9-style environmental remediation-cost model keyed on what is actually mined.
    - ``exposure``: the block's public-health exposure proxy fed to ``health_cost`` -- a K6-style
      health-cost model.
    - ``emissions``: the block's carbon-equivalent emissions, priced directly at ``carbon_price``
      (an L6-style market/regulatory carbon price is a multiplier, not a fitted cost model, so no
      callable is needed for it).

    Returns a dict with per-block arrays ``"remediation"``, ``"health"``, ``"carbon"``, their elementwise
    sum ``"total"`` (the single per-block liability :func:`mixle.stochastic_opt.risk_adjusted_plan`
    nets out of revenue), and the scalar roll-ups ``"remediation_total"``/``"health_total"``/
    ``"carbon_total"``/``"grand_total"`` for reporting.

    Raises:
        ValueError: if ``carbon_price``, ``plan``'s ``emissions``, or either cost callable's return value
            is non-finite or negative -- a negative liability would net out of revenue as a profit
            increase rather than a cost, and a NaN would propagate unguarded into the objective -- or if
            any of ``plan``'s ``grade``/``exposure``/``emissions`` or a cost callable's return value has
            the wrong shape.
    """
    carbon_price = float(carbon_price)
    _require_finite_nonnegative(carbon_price, "priced_liabilities: carbon_price")

    grade = np.asarray(_plan_get(plan, "grade"), dtype=np.float64)
    exposure = np.asarray(_plan_get(plan, "exposure"), dtype=np.float64)
    emissions = np.asarray(_plan_get(plan, "emissions"), dtype=np.float64)

    n_blocks = grade.shape[0]
    if exposure.shape != (n_blocks,) or emissions.shape != (n_blocks,):
        raise ValueError(
            "priced_liabilities: plan's grade/exposure/emissions must all be length-n_blocks 1-D arrays; "
            f"got grade {grade.shape}, exposure {exposure.shape}, emissions {emissions.shape}"
        )
    _require_finite_nonnegative(emissions, "priced_liabilities: plan's emissions")

    remediation = np.asarray(remediation_cost(grade), dtype=np.float64)
    health = np.asarray(health_cost(exposure), dtype=np.float64)
    carbon = carbon_price * emissions

    for name, arr in (("remediation_cost(grade)", remediation), ("health_cost(exposure)", health)):
        if arr.shape != (n_blocks,):
            raise ValueError(f"priced_liabilities: {name} must return a length-n_blocks array; got {arr.shape}")

    remediation = _require_finite_nonnegative(remediation, "priced_liabilities: remediation_cost(grade)")
    health = _require_finite_nonnegative(health, "priced_liabilities: health_cost(exposure)")

    total = remediation + health + carbon
    return {
        "remediation": remediation,
        "health": health,
        "carbon": carbon,
        "total": total,
        "remediation_total": float(remediation.sum()),
        "health_total": float(health.sum()),
        "carbon_total": float(carbon.sum()),
        "grand_total": float(total.sum()),
    }


def hard_constraints(*, no_mine_mask: Any | None = None, caps: list[dict] | None = None) -> dict:
    """Assemble the ``constraints`` dict :func:`mixle.stochastic_opt.risk_adjusted_plan` consumes.

    - ``no_mine_mask``: an optional boolean array-like, length ``n_blocks``. ``True`` entries are a
      G9-style no-mine/buffer-zone polygon's enclosed blocks -- hard-fixed to ``x_b = 0`` (stronger than
      an inequality row: those blocks can never be selected, regardless of the rest of the objective).
    - ``caps``: an optional list of linear activity caps -- a K6 exposure budget, an L6 water budget, or
      any other ``coeffs @ x <sense> bound`` row over the block decision vector. Each entry is a dict
      with ``"coeffs"`` (length-``n_blocks`` array-like) and ``"bound"`` (float), plus an optional
      ``"sense"`` (``"<="`` by default; ``">="`` is negated into the solver's standard ``<=`` form so
      ``risk_adjusted_plan`` never has to special-case it). This is the same ``coeffs``/``bound``/
      ``sense`` field naming :func:`mixle.analysis.biodiversity.no_net_loss_constraint` already uses for
      its own linear-constraint payload.

    Returns a dict with whichever of ``"no_mine_mask"``/``"caps"`` were supplied (both are omitted, i.e.
    an empty dict, when neither argument is given -- meaning no hard constraint at all).

    Raises:
        ValueError: if ``no_mine_mask`` is not a genuine 1-D boolean array (data that is merely truthy
            under an implicit bool cast, e.g. a float array or a NaN entry, is rejected rather than
            silently coerced), if any cap's ``"coeffs"`` is not a finite 1-D array, if any cap's
            ``"bound"`` is not finite, or if a cap's ``"sense"`` is not ``"<="``/``">="``.
    """
    constraints: dict[str, Any] = {}
    if no_mine_mask is not None:
        constraints["no_mine_mask"] = _require_bool_mask(no_mine_mask, "hard_constraints: no_mine_mask")
    if caps:
        normalized: list[dict] = []
        for cap in caps:
            coeffs = _require_finite_1d(cap["coeffs"], "hard_constraints: cap['coeffs']")
            bound = float(cap["bound"])
            _require_finite(bound, "hard_constraints: cap['bound']")
            sense = cap.get("sense", "<=")
            if sense == ">=":
                coeffs = -coeffs
                bound = -bound
            elif sense != "<=":
                raise ValueError(f"hard_constraints: cap 'sense' must be '<=' or '>=', got {sense!r}")
            normalized.append({"coeffs": coeffs, "bound": bound})
        constraints["caps"] = normalized
    return constraints
