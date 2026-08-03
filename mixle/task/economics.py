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

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class CostModel:
    """Unit costs in any consistent currency."""

    c_frontier: float  # cost of one request served by the expensive teacher/frontier model
    c_local: float = 0.0  # cost of one request served by the local distilled model
    c_label: float = 0.0  # cost of one teacher label during distillation
    train_cost: float = 0.0  # one-time cost to train/tune the student (compute)

    def __post_init__(self) -> None:
        for name in ("c_frontier", "c_local", "c_label", "train_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, float(value))

    def setup_cost(self, n_label: int) -> float:
        """Return the one-time label and training cost for a local model."""
        _nonnegative_int(n_label, "n_label")
        return n_label * self.c_label + self.train_cost


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be an exact non-negative integer")
    return int(value)


def _probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def cascade_cost_per_request(cost: CostModel, p_escalate: float) -> float:
    """Expected per-request cost of the cascade: always run local, escalate the ``p_escalate`` fraction."""
    if not isinstance(cost, CostModel):
        raise TypeError("cost must be a CostModel")
    return cost.c_local + _probability(p_escalate, "p_escalate") * cost.c_frontier


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
    local_only_certified: bool = False,
) -> RoutePlan:
    """Pick the lowest-cost route over ``volume`` requests.

    ``local_only`` is never inferred from an escalation preference: forcing
    ambiguous requests through a local model changes serving semantics. It is
    offered only when ``local_only_certified=True`` and the measured
    escalation probability is exactly zero.
    """
    if not isinstance(cost, CostModel):
        raise TypeError("cost must be a CostModel")
    volume = _nonnegative_int(volume, "volume")
    n_label = _nonnegative_int(n_label, "n_label")
    p_escalate = _probability(p_escalate, "p_escalate")
    if max_escalation is not None:
        max_escalation = _probability(max_escalation, "max_escalation")
    if not isinstance(local_only_certified, bool):
        raise ValueError("local_only_certified must be boolean")
    if local_only_certified and p_escalate != 0.0:
        raise ValueError("local_only_certified requires p_escalate == 0")
    frontier_total = volume * cost.c_frontier
    cascade_total = cost.setup_cost(n_label) + volume * cascade_cost_per_request(cost, p_escalate)
    local_total = cost.setup_cost(n_label) + volume * cost.c_local

    options: dict[str, float] = {"frontier_only": frontier_total, "cascade": cascade_total}
    if local_only_certified:
        options["local_only"] = local_total
        options.pop("cascade")
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


def _attribute_snapshot(model: Any) -> Callable[[], None]:
    """Return a callable that puts ``model``'s instance attributes back the way they are right now.

    MXR-080-1895, the rollback half of :func:`select_alpha_for_cost`'s transactional sweep. The
    calibration state this function disturbs is rebound *attributes*:
    :class:`~mixle.task.calibrate.CalibratedTaskModel` keeps its threshold in ``qhat`` and its target
    in ``alpha``, and ``calibrate()`` rebinds ``qhat`` and nothing else. Snapshotting ``__dict__`` and
    restoring it therefore returns that model to its exact prior state, including dropping any
    attribute the sweep added.

    Two stated limits. It is a SHALLOW snapshot: an attribute that is mutated in place (a list
    appended to, an array written through, a nested model's weights updated by a fit) is not undone,
    because the object it points at is the same object. And a model defined with ``__slots__`` has no
    ``__dict__``; for those only ``alpha`` is restored, which is the one attribute the duck-typed
    contract in :func:`select_alpha_for_cost` actually names. Neither case can be handled generically
    without knowing what a caller's ``calibrate()`` touches, and guessing would be worse than saying
    so: the alternative that was in place -- leaving the failed state entirely alone -- is strictly
    worse than both.
    """
    state = getattr(model, "__dict__", None)
    if isinstance(state, dict):
        snapshot = dict(state)

        def restore() -> None:
            state.clear()
            state.update(snapshot)

        return restore

    alpha = getattr(model, "alpha", None)

    def restore_alpha() -> None:
        if alpha is not None:
            model.alpha = alpha

    return restore_alpha


def select_alpha_for_cost(
    model: Any,
    cal_texts: Sequence[Any],
    cal_labels: Sequence[Any],
    probe_texts: Sequence[Any],
    cost: CostModel,
    *,
    certification_texts: Sequence[Any],
    certification_labels: Sequence[Any],
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
    recalibrates ``model`` on policy-selection calibration data and measures
    its realized escalation rate on ``probe_texts``, then scores
    that escalation rate with :func:`recommend_route` over ``volume`` requests.
    The winner is the alpha whose recommended route is lowest-cost overall;
    only after selection is complete is ``model`` calibrated at that alpha on
    the independent ``certification_texts``/``certification_labels``. Thus the
    coverage-bearing threshold never participates in policy selection. Returns
    ``(best_alpha, best_plan, plan_by_alpha)`` so the full sweep remains
    auditable; the plans are policy-selection cost estimates, while the model's
    final threshold is the independent certification result.

    The in-place recalibration is all-or-nothing (MXR-080-1895): if any step raises, ``model``'s
    attributes are restored to what they were on entry before the exception propagates, so a failed
    sweep can never leave a serving model carrying a threshold fitted on policy-selection rows. See
    :func:`_attribute_snapshot` for what "restored" does and does not cover.
    """
    if not isinstance(cost, CostModel):
        raise TypeError("cost must be a CostModel")
    for attribute in ("calibrate", "escalation_rate"):
        if not callable(getattr(model, attribute, None)):
            raise TypeError(f"model must expose callable {attribute}()")
    cal_texts = list(cal_texts)
    cal_labels = list(cal_labels)
    probe_texts = list(probe_texts)
    certification_texts = list(certification_texts)
    certification_labels = list(certification_labels)
    if not cal_texts or len(cal_texts) != len(cal_labels):
        raise ValueError("policy calibration texts and labels must be non-empty and aligned")
    if not probe_texts:
        raise ValueError("probe_texts must be non-empty")
    if not certification_texts or len(certification_texts) != len(certification_labels):
        raise ValueError("certification texts and labels must be non-empty and aligned")

    def row_keys(rows: Sequence[Any]) -> set[str]:
        return {json.dumps(row, sort_keys=True, default=repr) for row in rows}

    selection_keys = row_keys(cal_texts)
    probe_keys = row_keys(probe_texts)
    certification_keys = row_keys(certification_texts)
    if selection_keys & probe_keys or selection_keys & certification_keys or probe_keys & certification_keys:
        raise ValueError("policy calibration, probe, and certification rows must be disjoint")

    alpha_values = list(alphas)
    if not alpha_values or len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must be a non-empty sequence of unique values")
    alpha_values = [_probability(alpha, "alpha") for alpha in alpha_values]
    if any(alpha in (0.0, 1.0) for alpha in alpha_values):
        raise ValueError("alpha candidates must be strictly between 0 and 1")
    # MXR-080-1895: this sweep recalibrates `model` in place once per candidate, which is its declared
    # contract -- but only when it RUNS TO COMPLETION. A failure partway (an unavailable scoring
    # backend, a bad calibration row) used to abandon the model wherever the sweep had got to: alpha
    # set to a candidate that was never selected, and a threshold fitted on the POLICY-SELECTION rows
    # this function's own docstring promises never bear coverage. A caller that caught the error and
    # kept serving was serving an uncertified threshold, with nothing in the object saying so.
    #
    # So the sweep is transactional: either it finishes and the model carries the certification-set
    # threshold, or it raises and the model is left exactly as it was handed over.
    restore = _attribute_snapshot(model)
    try:
        plans: dict[float, RoutePlan] = {}
        for a in alpha_values:
            model.alpha = a
            model.calibrate(cal_texts, cal_labels)
            p_escalate = model.escalation_rate(probe_texts)
            plans[a] = recommend_route(cost, volume=volume, n_label=n_label, p_escalate=p_escalate)

        best_alpha = min(plans, key=lambda a: plans[a].total)
        model.alpha = best_alpha
        model.calibrate(certification_texts, certification_labels)
    except BaseException:
        restore()
        raise
    return best_alpha, plans[best_alpha], plans
