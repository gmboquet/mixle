"""J6 -- the risk-adjusted economic objective: wiring, not new physics (work-plan Sec.7-J, the grand
synthesis).

H4's `mixle.stochastic_opt.two_stage_stochastic_plan` already turns grade uncertainty (IC-1
`Posterior.samples`) into a CVaR-penalized block-extraction decision. J's other tasks each price one more
externality against the same MILP: J2's `monte_carlo_npv` (revenue/cost DCF), a G9-style no-mine/buffer
zone, a K6-style public-health exposure cost, an L6-style carbon/emissions price, an N6-style biodiversity
offset (`mixle.analysis.biodiversity.habitat_offset_liability`/`no_net_loss_constraint`, already landed and
explicitly written against this module's shape). None of those tasks are this module's dependency --
this is the reverse: J6 is the pluggable *framework* they register priced terms/constraints INTO, so the
dependency edge runs G9/K6/L6/N6 -> J6, never J6 -> them (workstream-J.md J6 header note). That means this
module only fixes the *shape* a priced term or a hard constraint must have to plug in; it derives none of
the underlying cost models itself (Non-goals).

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
    """
    grade = np.asarray(_plan_get(plan, "grade"), dtype=np.float64)
    exposure = np.asarray(_plan_get(plan, "exposure"), dtype=np.float64)
    emissions = np.asarray(_plan_get(plan, "emissions"), dtype=np.float64)

    n_blocks = grade.shape[0]
    if exposure.shape != (n_blocks,) or emissions.shape != (n_blocks,):
        raise ValueError(
            "priced_liabilities: plan's grade/exposure/emissions must all be length-n_blocks 1-D arrays; "
            f"got grade {grade.shape}, exposure {exposure.shape}, emissions {emissions.shape}"
        )

    remediation = np.asarray(remediation_cost(grade), dtype=np.float64)
    health = np.asarray(health_cost(exposure), dtype=np.float64)
    carbon = float(carbon_price) * emissions

    for name, arr in (("remediation_cost(grade)", remediation), ("health_cost(exposure)", health)):
        if arr.shape != (n_blocks,):
            raise ValueError(f"priced_liabilities: {name} must return a length-n_blocks array; got {arr.shape}")

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
    """
    constraints: dict[str, Any] = {}
    if no_mine_mask is not None:
        constraints["no_mine_mask"] = np.asarray(no_mine_mask, dtype=bool)
    if caps:
        normalized: list[dict] = []
        for cap in caps:
            coeffs = np.asarray(cap["coeffs"], dtype=np.float64)
            bound = float(cap["bound"])
            sense = cap.get("sense", "<=")
            if sense == ">=":
                coeffs = -coeffs
                bound = -bound
            elif sense != "<=":
                raise ValueError(f"hard_constraints: cap 'sense' must be '<=' or '>=', got {sense!r}")
            normalized.append({"coeffs": coeffs, "bound": bound})
        constraints["caps"] = normalized
    return constraints
