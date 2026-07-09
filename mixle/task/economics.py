"""Cost comparisons for local distillation, cascades, and teacher-only serving.

Every routing choice has a cost model. This module combines a conformal
escalation rate (:meth:`mixle.task.calibrate.CalibratedTaskModel.escalation_rate`,
the empirical ``p_escalate``) with unit costs:

  * **frontier-only** -- pay ``c_frontier`` for every request, forever.
  * **local-only** -- distill once (``n_label`` teacher calls + training), then pay ``c_local`` per request.
  * **cascade** -- run the low-cost local model first, escalate only the ambiguous fraction: per request
    ``c_local + p_escalate * c_frontier``, with the singletons covered at ``1 - alpha``.

:func:`break_even_volume` is the request count at which a distilled route
recovers its one-time setup cost. :func:`recommend_route` picks the lowest-cost
route at a given volume, optionally constrained by a maximum tolerated
escalation rate, and reports the savings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostModel:
    """Unit costs in any consistent currency."""

    c_frontier: float  # cost of one request served by the expensive teacher/frontier model
    c_local: float = 0.0  # cost of one request served by the local distilled model
    c_label: float = 0.0  # cost of one teacher label during distillation
    train_cost: float = 0.0  # one-time cost to train/tune the student (compute)

    def setup_cost(self, n_label: int) -> float:
        """Return the one-time label and training cost for a local model."""
        return n_label * self.c_label + self.train_cost


def cascade_cost_per_request(cost: CostModel, p_escalate: float) -> float:
    """Expected per-request cost of the cascade: always run local, escalate the ``p_escalate`` fraction."""
    return cost.c_local + float(p_escalate) * cost.c_frontier


def break_even_volume(cost: CostModel, n_label: int, *, p_escalate: float = 0.0) -> float:
    """Requests after which a distilled route undercuts frontier-only (``inf`` if it never does).

    Setup is amortized against the per-request saving ``c_frontier - per_request(route)``. With ``p_escalate=0``
    this is the local-only break-even; pass the model's escalation rate for the cascade break-even.
    """
    per_req = cascade_cost_per_request(cost, p_escalate)
    saving = cost.c_frontier - per_req
    if saving <= 0:
        return float("inf")
    return cost.setup_cost(n_label) / saving


@dataclass(frozen=True)
class RoutePlan:
    """Costed route comparison for a fixed request volume."""

    route: str  # "frontier_only" | "local_only" | "cascade"
    volume: int
    per_request: float
    total: float
    savings_vs_frontier: float
    p_escalate: float
    break_even: float
    options: dict[str, float]  # route -> total cost at this volume (incl. setup for distilled routes)


def recommend_route(
    cost: CostModel,
    *,
    volume: int,
    n_label: int,
    p_escalate: float,
    max_escalation: float | None = None,
) -> RoutePlan:
    """Pick the lowest-cost route over ``volume`` requests.

    ``local_only`` is offered only when the caller explicitly disallows
    escalation by setting ``max_escalation == 0``. Otherwise the cascade route
    keeps local answers for calibrated inputs and escalates the remaining
    traffic to the teacher.
    """
    frontier_total = volume * cost.c_frontier
    cascade_total = cost.setup_cost(n_label) + volume * cascade_cost_per_request(cost, p_escalate)
    local_total = cost.setup_cost(n_label) + volume * cost.c_local

    options: dict[str, float] = {"frontier_only": frontier_total, "cascade": cascade_total}
    if max_escalation == 0:
        options["local_only"] = local_total
    if max_escalation is not None and p_escalate > max_escalation:
        options.pop("cascade", None)  # cascade escalates too rarely-or-often for the caller's bar

    route = min(options, key=lambda r: options[r])
    total = options[route]
    per_req = total / volume if volume else float("inf")
    return RoutePlan(
        route=route,
        volume=volume,
        per_request=per_req,
        total=total,
        savings_vs_frontier=frontier_total - total,
        p_escalate=float(p_escalate),
        break_even=break_even_volume(cost, n_label, p_escalate=p_escalate),
        options=options,
    )


def select_alpha_for_cost(
    model: Any,
    cal_texts: Sequence[Any],
    cal_labels: Sequence[Any],
    probe_texts: Sequence[Any],
    cost: CostModel,
    *,
    volume: int,
    n_label: int,
    alphas: Sequence[float] = (0.01, 0.05, 0.1, 0.15, 0.2, 0.3),
) -> tuple[float, RoutePlan, dict[float, RoutePlan]]:
    """Select ``alpha`` from a :class:`CostModel` target.

    The sweep connects :func:`recommend_route` to the calibration step so threshold selection reflects
    both model behavior and the caller's cost assumptions.

    ``model`` is anything with the
    :class:`~mixle.task.calibrate.CalibratedTaskModel` shape: a mutable
    ``alpha`` attribute, ``calibrate(texts, labels)``, and
    ``escalation_rate(texts)``. For each candidate in ``alphas``, this
    recalibrates ``model`` and measures its realized escalation rate on
    ``probe_texts`` (a held-out slice disjoint from ``cal_texts``), then scores
    that escalation rate with :func:`recommend_route` over ``volume`` requests.
    The winner is the alpha whose recommended route is lowest-cost overall;
    ``model`` is left calibrated at that winning alpha. Returns
    ``(best_alpha, best_plan, plan_by_alpha)`` so the full sweep remains
    auditable.
    """
    plans: dict[float, RoutePlan] = {}
    for a in alphas:
        model.alpha = float(a)
        model.calibrate(cal_texts, cal_labels)
        p_escalate = model.escalation_rate(probe_texts)
        plans[a] = recommend_route(cost, volume=volume, n_label=n_label, p_escalate=p_escalate)

    best_alpha = min(plans, key=lambda a: plans[a].total)
    model.alpha = float(best_alpha)
    model.calibrate(cal_texts, cal_labels)
    return best_alpha, plans[best_alpha], plans
