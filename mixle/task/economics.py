"""The cost arithmetic that decides whether to spend the GPU: distill-and-serve-local vs. cascade vs. frontier.

GPU time is not free, so every routing choice is an economic one. This module turns the conformal escalation
rate (:meth:`mixle.task.calibrate.CalibratedTaskModel.escalation_rate`, the empirical ``p_escalate``) and a few
unit costs into dollars:

  * **frontier-only** -- pay ``c_frontier`` for every request, forever.
  * **local-only** -- distill once (``n_label`` teacher calls + training), then pay ``c_local`` per request.
  * **cascade** -- run the cheap local model first, escalate only the ambiguous fraction: per request
    ``c_local + p_escalate * c_frontier``, with the singletons covered at ``1 - alpha``.

:func:`break_even_volume` is the request count at which a distilled route pays back its one-time setup;
:func:`recommend_route` picks the cheapest route at a given volume (optionally constrained to a maximum tolerated
escalation rate) and reports the savings. This is the "is it worth it?" question answered with a number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Unit costs (any consistent currency). ``c_local`` is the amortized local per-request cost (~0 on CPU)."""

    c_frontier: float  # cost of one request served by the expensive teacher/frontier model
    c_local: float = 0.0  # cost of one request served by the local distilled model
    c_label: float = 0.0  # cost of one teacher label during distillation
    train_cost: float = 0.0  # one-time cost to train/tune the student (compute)

    def setup_cost(self, n_label: int) -> float:
        """One-time cost to stand up a local model: label ``n_label`` examples + train."""
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
    """The costed comparison of routes at a given volume, with the recommended choice and its savings."""

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
    """Pick the cheapest route over ``volume`` requests; honor an optional cap on tolerated escalation.

    ``local_only`` is treated as a cascade with no escalation *only when* its quality is acceptable to the
    caller -- here it is offered whenever ``max_escalation`` is not exceeded by ``p_escalate`` (the cascade
    already answers the easy cases locally and escalates the rest, so it dominates pure local-only on accuracy;
    local-only is kept as the floor cost when escalation is disallowed entirely, ``max_escalation == 0``).
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
