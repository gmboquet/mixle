"""Mine economics: parametric cost curves and capex/opex roll-up (work-plan Sec.7-J, J4).

J's objective function needs a `$/t` cost that is a function of *where* the ore is (depth), *what* it
is (grade), and *how fast* it is mined (throughput) before J2 can turn price paths + posterior grade
draws into an NPV distribution, and before H's block-level optimizers (`mixle.stochastic_opt`,
`mixle.relations`) have a `block_cost` to subtract from revenue:

  * :func:`cost_curve` -- parametric mining + processing cost in `$/t`, monotone increasing in haul/
    pumping depth, complexity-adjusted by grade, and shaped like the classic economies-of-scale curve
    in throughput: cheapest at the plant's design capacity, more expensive both under- and
    over-utilized.
  * :func:`capex_opex` -- rolls a period-by-period mine plan (tonnage, depth, grade, throughput, plus
    any lumpy capital spend) up into total capital and total operating cost, via :func:`cost_curve`.

This module is created here (J4, Wave 1) and extended by J2 (Wave 2) with `monte_carlo_npv` /
`NPVDistribution` once J1's price paths exist -- see that task's docstring for the DCF assembly. Both
:func:`cost_curve`'s output (a plain `$/t` array, one entry per block/period) and :func:`capex_opex`'s
totals are ordinary ``numpy``/``float`` values with no dependency on J1/J2/H1, so they slot directly
into `mixle.stochastic_opt.two_stage_stochastic_plan`'s `block_cost` argument today (H1's
`mixle.relations.min_cost_flow` had not landed on `release/0.8.0` as of this PR -- see that function's
own docstring for the same repo-boundary note).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["capex_opex", "cost_curve"]

# Default parameters, used for any key the caller's `params` dict omits. Chosen to be dimensionally
# sane toy defaults ($/t and $/t-per-metre in the low single digits), not a claim about any real mine.
_DEFAULTS: dict[str, float] = {
    "base_cost": 0.0,  # $/t floor: cost at zero depth, reference grade, design-capacity throughput
    "haul_cost_per_m": 0.0,  # $/t per metre of depth: haulage + dewatering/pumping, linear in depth
    "grade_complexity_coef": 0.0,  # $/t, scales the 1/grade metallurgical-complexity penalty
    "throughput_scale_coef": 0.0,  # $/t, scales the (Q/Q* - 1)^2 economies-of-scale penalty
    "design_capacity": 1.0,  # Q*: throughput at which the economies-of-scale term is zero
    "capex_fixed": 0.0,  # $, one-off development/construction capital independent of tonnage
    "capex_per_tonne": 0.0,  # $/t, sustaining capital that scales with total tonnage mined
}


def _param(params: dict, key: str) -> float:
    return float(params[key]) if key in params else _DEFAULTS[key]


def cost_curve(depth: Any, grade: Any, throughput: Any, *, params: dict) -> np.ndarray:
    """Parametric mining + processing cost in `$/t`, as a function of depth, grade, and throughput.

    ``depth``, ``grade``, and ``throughput`` are broadcastable array-likes (one entry per block or per
    scheduling period; scalars broadcast against the others). ``params`` recognizes (all optional,
    defaulting to zero/no-effect):

    - ``base_cost``: `$/t` floor cost.
    - ``haul_cost_per_m``: `$/t` per metre of depth -- haulage and pumping/dewatering cost, modeled as
      linear in depth, so the curve is strictly increasing in ``depth`` whenever this is positive.
    - ``grade_complexity_coef``: `$/t` scale of a ``1 / grade`` metallurgical-complexity penalty --
      lower-grade ore needs proportionally more material handled and processed per unit of recovered
      metal, so this term falls as grade rises.
    - ``throughput_scale_coef`` / ``design_capacity``: the plant has one throughput, ``design_capacity``
      (``Q*``), at which fixed costs are spread most efficiently; cost rises quadratically away from
      it in *either* direction -- ``throughput_scale_coef * ((Q - Q*) / Q*) ** 2`` -- capturing both
      under-utilized fixed-cost drag below ``Q*`` and overtime/expediting/accelerated-wear cost above
      it (the classic "decreasing then rising past design capacity" U-shaped average-cost curve).

    Returns the elementwise `$/t` cost, broadcast to the common shape of the three inputs.
    """
    d = np.asarray(depth, dtype=np.float64)
    g = np.asarray(grade, dtype=np.float64)
    q = np.asarray(throughput, dtype=np.float64)

    if np.any(g <= 0.0):
        raise ValueError("cost_curve: grade must be strictly positive (used as a 1/grade complexity term)")
    q_star = _param(params, "design_capacity")
    if q_star <= 0.0:
        raise ValueError("cost_curve: params['design_capacity'] must be strictly positive")
    if np.any(q <= 0.0):
        raise ValueError("cost_curve: throughput must be strictly positive")

    base = _param(params, "base_cost")
    haul = _param(params, "haul_cost_per_m") * d
    complexity = _param(params, "grade_complexity_coef") / g
    scale = _param(params, "throughput_scale_coef") * ((q - q_star) / q_star) ** 2

    return base + haul + complexity + scale


def _plan_get(plan: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off ``plan``, whether it is a mapping (dict) or an attribute-bearing object."""
    if isinstance(plan, dict):
        return plan.get(key, default)
    return getattr(plan, key, default)


def capex_opex(plan: Any, *, params: dict) -> tuple[float, float]:
    """Roll a mine plan's tonnage/depth/grade/throughput profile up into (total capex, total opex).

    ``plan`` is a mapping or attribute-bearing object exposing, per scheduling period:

    - ``tonnage``: array-like, tonnes mined/processed each period (required).
    - ``depth``, ``grade``, ``throughput``: array-likes (or scalars, broadcast against ``tonnage``)
      fed to :func:`cost_curve` to get each period's `$/t`.
    - ``capex_schedule`` (optional): array-like of lumpy capital spend per period (e.g. pre-strip,
      plant construction, fleet purchases); summed into total capex on top of the params below.

    ``params`` is passed through to :func:`cost_curve` for the opex side, plus two capex-only keys:
    ``capex_fixed`` (one-off, tonnage-independent capital) and ``capex_per_tonne`` (sustaining capital
    that scales with total tonnage mined over the plan).

    Total opex is ``sum(tonnage * cost_curve(depth, grade, throughput, params=params))``; total capex is
    ``capex_fixed + capex_per_tonne * sum(tonnage) + sum(capex_schedule)``. Returns ``(capex, opex)``,
    both plain floats -- the totals :func:`monte_carlo_npv` (J2) discounts into a DCF, and the same
    `$/t` curve this function calls is what feeds `block_cost` for H's optimizers.
    """
    tonnage = np.asarray(_plan_get(plan, "tonnage"), dtype=np.float64)
    depth = _plan_get(plan, "depth")
    grade = _plan_get(plan, "grade")
    throughput = _plan_get(plan, "throughput")

    per_period_cost = cost_curve(depth, grade, throughput, params=params)
    opex_total = float(np.sum(tonnage * per_period_cost))

    total_tonnage = float(np.sum(tonnage))
    capex_total = _param(params, "capex_fixed") + _param(params, "capex_per_tonne") * total_tonnage
    capex_schedule = _plan_get(plan, "capex_schedule", None)
    if capex_schedule is not None:
        capex_total += float(np.sum(np.asarray(capex_schedule, dtype=np.float64)))

    return capex_total, opex_total
