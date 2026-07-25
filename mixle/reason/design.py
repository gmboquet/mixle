"""Budgeted selection over cross-modal evidence whose outcomes are already observed.

:meth:`~mixle.reason.store.CrossModalStore.next_evidence` and this module score the realized evidence produced
from stored payloads. That is useful for cost-aware selection from an existing evidence corpus, but it is not
pre-acquisition expected information gain: hidden future outcomes require a declared predictive observation
model. :func:`select_evidence_batch` therefore requires an explicit ``evidence_is_observed=True`` assertion and
fails closed otherwise.

:func:`select_evidence_batch` performs the selection; the returned :class:`AcquisitionPlan` carries the chosen
``(index, fidelity, gain, cost)`` trail, the total nats gained, and the assimilated belief.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.reason.store import CrossModalStore, _apply, _query_entropy


@dataclass
class AcquisitionPlan:
    """A budgeted evidence-acquisition plan: the chosen items, the nats they bought, and the final belief."""

    items: list[tuple[int, str, float, float]] = field(default_factory=list)  # (index, fidelity, gain_nats, cost)
    total_cost: float = 0.0
    total_gain: float = 0.0  # prior query entropy - final query entropy (nats)
    belief: Any = None

    @property
    def indices(self) -> list[int]:
        """Return selected evidence item indices."""
        return [i for i, _f, _g, _c in self.items]


def select_evidence_batch(
    store: CrossModalStore,
    belief: Any,
    *,
    budget: float,
    query: Any = None,
    fine_cost: float = 1.0,
    coarse_cost: float = 0.2,
    fidelities: Sequence[str] = ("coarse", "fine"),
    candidates: Sequence[int] | None = None,
    max_items: int | None = None,
    min_gain: float = 1e-9,
    evidence_is_observed: bool = False,
) -> AcquisitionPlan:
    """Select already-observed evidence by realized gain under a total ``budget``.

    This function evaluates the actual evidence built from each payload, so it is not a
    pre-acquisition expected-information-gain planner. Callers must explicitly set
    ``evidence_is_observed=True`` to confirm the candidate payloads are already observed and selection
    does not leak hidden outcomes. A future acquisition requires a predictive observation model.
    """
    if evidence_is_observed is not True:
        raise ValueError(
            "select_evidence_batch evaluates realized payload evidence and is only valid with "
            "evidence_is_observed=True; hidden-outcome acquisition requires a predictive observation model."
        )
    budget = float(budget)
    fine_cost = float(fine_cost)
    coarse_cost = float(coarse_cost)
    min_gain = float(min_gain)
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError("budget must be finite and non-negative.")
    if not np.isfinite(fine_cost) or fine_cost <= 0.0 or not np.isfinite(coarse_cost) or coarse_cost <= 0.0:
        raise ValueError("fine_cost and coarse_cost must be finite and positive.")
    if not np.isfinite(min_gain) or min_gain < 0.0:
        raise ValueError("min_gain must be finite and non-negative.")
    if max_items is not None:
        if (
            isinstance(max_items, (bool, np.bool_))
            or not isinstance(max_items, (int, np.integer))
            or max_items < 0
        ):
            raise ValueError("max_items must be an exact non-negative integer or None.")
        max_items = int(max_items)
    cost_of = {"coarse": float(coarse_cost), "fine": float(fine_cost)}
    build_of: dict[str, Callable[[Any], Any]] = {"coarse": store.coarse, "fine": store.fine}
    fids = list(fidelities)
    if not fids or len(set(fids)) != len(fids) or any(fidelity not in build_of for fidelity in fids):
        raise ValueError("fidelities must be a non-empty unique sequence containing only 'coarse' and 'fine'.")

    pool = list(range(len(store.payloads))) if candidates is None else list(candidates)
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= len(store.payloads)
        for index in pool
    ):
        raise ValueError("candidate indices must be exact integers within the store.")
    pool = [int(index) for index in pool]
    if len(set(pool)) != len(pool):
        raise ValueError("candidate indices must be unique.")
    prior_entropy = _query_entropy(belief, query)
    if not np.isfinite(prior_entropy):
        raise ValueError("query entropy must be finite.")
    plan = AcquisitionPlan(belief=belief)
    remaining = set(pool)

    while remaining and (max_items is None or len(plan.items) < max_items):
        before = _query_entropy(plan.belief, query)
        if not np.isfinite(before):
            raise ValueError("query entropy must remain finite during selection.")
        best = None  # (gain_per_cost, idx, fidelity, evidence, gain, cost)
        for idx in remaining:
            payload = store.payloads[idx]
            for fidelity in fids:
                cost = cost_of[fidelity]
                if plan.total_cost + cost > budget:
                    continue
                ev = build_of[fidelity](payload)
                gain = before - _query_entropy(_apply(plan.belief, ev), query)
                if not np.isfinite(gain):
                    raise ValueError("candidate evidence produced a non-finite information gain.")
                if gain <= min_gain:
                    continue
                gpc = gain / max(cost, 1e-12)
                if best is None or gpc > best[0]:
                    best = (gpc, idx, fidelity, ev, gain, cost)
        if best is None:
            break
        _, idx, fidelity, ev, gain, cost = best
        plan.belief = _apply(plan.belief, ev)
        plan.total_cost += cost
        plan.items.append((idx, fidelity, float(gain), cost))
        remaining.discard(idx)

    plan.total_gain = float(prior_entropy - _query_entropy(plan.belief, query))
    if not np.isfinite(plan.total_gain):
        raise ValueError("selected evidence produced a non-finite total information gain.")
    return plan
