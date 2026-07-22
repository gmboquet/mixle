"""Precedence-constrained selection and scheduling: maximum-weight closure and time-phased capacity scheduling.

Two general combinatorial-optimization problems recur whenever items each carry a value (positive or
negative) and a *precedence* relation: item ``b`` cannot be selected/scheduled before its predecessor
items. Two questions sit on top of that model:

* **Maximum-weight closure** (:func:`maximum_weight_closure`) -- ignoring time, which subset of items
  maximizes total value subject to precedence? Picard's construction reduces this to a single
  min-cut: connect a source to every positive-value item with capacity equal to its value, connect
  every negative-value item to a sink with capacity equal to its magnitude, and add an
  effectively-infinite-capacity arc ``b -> pred`` for every precedence pair -- an infinite arc can
  never be cut, so the source side of the min-cut can never contain an item without also containing
  its predecessors, which is exactly the closure property. Solved with :func:`mixle.relations.max_flow`
  / :func:`mixle.relations.min_cut` (Edmonds-Karp is a valid, if not asymptotically optimal, pseudoflow
  substitute for this fixed-size combinatorial problem).
* **Time-phased scheduling** (:func:`schedule_activities`) -- given a per-period capacity and a
  discount rate, *when* (which period, if ever) should each item be scheduled to maximize net present
  value, honoring precedence at every period boundary and never exceeding capacity? This is a
  mixed-integer program (binary ``x[b, t]`` = item ``b`` scheduled in period ``t``) solved by
  :func:`mixle.relations.branch_and_bound_milp`.

Neither construction is domain-specific: any "which interdependent items are worth taking, and when"
problem reduces the same way -- project selection under budget/dependency constraints, task
scheduling with prerequisites, VLSI partitioning. The canonical worked instantiation of the first is
open-pit mine planning's "ultimate pit limit" (Lerchs-Grossmann): each block's economic value (ore net
of processing cost, or negative for waste net of removal cost) plus the precedence a stable slope
imposes (a block cannot be extracted before the material a stable slope requires be removed above it)
is exactly the maximum-weight closure problem; the second's canonical instantiation is that same
block model's time-phased extraction schedule under mill/mining capacity. Both are used that way
elsewhere in this codebase, but the functions here take plain items, values, and precedence pairs --
nothing about a block, an ore, or a mine.

    >>> import numpy as np
    >>> value = np.array([-1.0, 10.0])       # item 0 is a cost, item 1 depends on it
    >>> precedence = [(1, 0)]                # item 1 requires item 0 first
    >>> maximum_weight_closure(value, precedence)
    array([ True,  True])
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.relations import branch_and_bound_milp, min_cut

__all__ = ["maximum_weight_closure", "schedule_activities"]


def maximum_weight_closure(value: Any, precedence: Sequence[tuple[int, int]]) -> np.ndarray:
    """The maximum-weight closure of a precedence DAG (Picard's max-flow reduction).

    ``value`` is a length-``n`` array of per-item net value (positive or negative). ``precedence``
    lists ``(b, pred)`` pairs meaning item ``b`` requires item ``pred`` be selected first. Returns a
    length-``n`` boolean mask, ``True`` for every item in the value-maximizing, precedence-closed
    selection: a super-source connects to every positive-value item with capacity equal to its value,
    every negative-value item connects to a super-sink with capacity equal to its magnitude, and each
    precedence pair becomes an effectively-infinite-capacity arc ``b -> pred`` -- such an arc is never
    cut, so the min-cut's source side can never hold ``b`` without also holding ``pred``, which is
    exactly the closure property. The source side of the minimum cut (:func:`mixle.relations.min_cut`)
    is the optimal closure. "Effectively infinite" is a finite big-M (the total absolute value plus
    one) rather than a literal ``np.inf``: :func:`mixle.relations.max_flow` reconstructs flow as
    ``cap - residual``, and an infinite ``cap`` on an arc that ever carries flow leaves ``residual``
    also infinite, so that subtraction is ``inf - inf = nan`` -- which then makes
    :func:`mixle.relations.min_cut`'s reachability BFS silently drop the arc (``nan > tol`` is false),
    breaking the closure guarantee. A finite big-M that no optimal cut could ever prefer to sever
    avoids the NaN while keeping the same combinatorial meaning.

    Open-pit mine planning calls this the ultimate pit limit: each block's value is ore net of
    processing cost (positive) or waste net of removal cost (negative), and ``precedence`` lists the
    blocks a stable slope forces to be removed before each block -- but the reduction above has no
    block, ore, or slope in it.
    """
    value = np.asarray(value, dtype=np.float64)
    n = value.size
    source, sink = n, n + 1
    big_m = float(np.abs(value).sum()) + 1.0  # no min-cut ever prefers severing a precedence arc
    cap = np.zeros((n + 2, n + 2), dtype=np.float64)
    for b in range(n):
        if value[b] > 0.0:
            cap[source, b] = value[b]
        elif value[b] < 0.0:
            cap[b, sink] = -value[b]
    for b, pred in precedence:
        if not (0 <= b < n and 0 <= pred < n):
            raise ValueError(f"precedence pair {(b, pred)} references an item outside 0..{n - 1}")
        cap[b, pred] = big_m
    _, source_side, _ = min_cut(cap, source, sink)
    mask = np.zeros(n, dtype=bool)
    for node in source_side:
        if node < n:
            mask[node] = True
    return mask


def _cumulative_precedence_rows(
    n: int, n_periods: int, precedence: Sequence[tuple[int, int]], idx: Any
) -> tuple[list[np.ndarray], list[float]]:
    """Rows encoding, for every precedence pair and every period ``t``: cumulative-scheduled(b) by
    ``t`` ``<=`` cumulative-scheduled(pred) by ``t`` -- ``pred`` can never trail ``b``."""
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for b, pred in precedence:
        if not (0 <= b < n and 0 <= pred < n):
            raise ValueError(f"precedence pair {(b, pred)} references an item outside 0..{n - 1}")
        for t in range(n_periods):
            row = np.zeros(n * n_periods)
            for tau in range(t + 1):
                row[idx(b, tau)] += 1.0
                row[idx(pred, tau)] -= 1.0
            rows.append(row)
            rhs.append(0.0)
    return rows, rhs


def _solve_schedule_window(
    value: np.ndarray,
    precedence: Sequence[tuple[int, int]],
    capacity: np.ndarray,
    n_periods: int,
    discount: float,
    period_offset: int,
) -> tuple[float, np.ndarray]:
    """Exact MILP solve of the schedule over one contiguous window of periods.

    ``value``/``precedence`` are already restricted to the candidate items for this window (local
    indices); ``period_offset`` is the *global* period the window's period-0 corresponds to, used
    only to discount the objective consistently across windows.
    """
    n = value.size

    def idx(b: int, t: int) -> int:
        return b * n_periods + t

    n_vars = n * n_periods
    disc = 1.0 / (1.0 + discount) ** (period_offset + np.arange(n_periods))
    c = np.zeros(n_vars)
    for b in range(n):
        for t in range(n_periods):
            c[idx(b, t)] = value[b] * disc[t]

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for b in range(n):  # each item scheduled at most once across this window
        row = np.zeros(n_vars)
        for t in range(n_periods):
            row[idx(b, t)] = 1.0
        rows.append(row)
        rhs.append(1.0)
    for t in range(n_periods):  # per-period capacity
        row = np.zeros(n_vars)
        for b in range(n):
            row[idx(b, t)] = 1.0
        rows.append(row)
        rhs.append(float(capacity[t]))
    prec_rows, prec_rhs = _cumulative_precedence_rows(n, n_periods, precedence, idx)
    rows.extend(prec_rows)
    rhs.extend(prec_rhs)

    a_ub = np.array(rows) if rows else np.zeros((0, n_vars))
    b_ub = np.array(rhs) if rhs else np.zeros(0)
    bounds = [(0.0, 1.0)] * n_vars
    result = branch_and_bound_milp(c, a_ub, b_ub, integer=list(range(n_vars)), bounds=bounds, sense="max")
    if result is None:
        # scheduling nothing always satisfies every constraint, so this should be unreachable.
        raise ValueError("schedule_activities: MILP came back infeasible on a window that admits the empty schedule")
    npv, x_flat = result
    x = x_flat.reshape(n, n_periods)
    period = np.full(n, -1, dtype=np.int64)
    for b in range(n):
        scheduled_at = np.where(x[b] > 0.5)[0]
        if scheduled_at.size:
            period[b] = period_offset + int(scheduled_at[0])
    return float(npv), period


# Above this many (item x period) binary variables, solve a rolling horizon instead of one MILP:
# branch-and-bound over the full horizon is exact but its worst case is exponential in n_vars, so
# large instances are chunked into windows that are each solved exactly and then committed.
_DIRECT_MILP_LIMIT = 400


def schedule_activities(
    value: Any,
    precedence: Sequence[tuple[int, int]],
    capacity: Any,
    n_periods: int,
    *,
    discount: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Time-phased schedule maximizing discounted value under precedence and per-period capacity.

    ``value`` is a length-``n`` array of per-item net value; ``precedence`` lists ``(b, pred)`` pairs
    as in :func:`maximum_weight_closure`; ``capacity`` is a length-``n_periods`` array bounding how
    many items may be scheduled in each period (one item = one unit of capacity: callers wanting
    heterogeneous per-item resource use can pre-scale ``capacity`` or split a heavy item into unit
    sub-items). Binary ``x[b, t]`` = item ``b`` scheduled in period ``t``; precedence is enforced at
    every period boundary (cumulative scheduling of ``pred`` can never trail ``b``'s), each item is
    scheduled at most once (never, if it is never worth it), and the objective discounts each
    period's value by ``(1 + discount) ** t``. Solved exactly by
    :func:`mixle.relations.branch_and_bound_milp`; instances large enough to make the full horizon's
    MILP impractical are solved as a rolling horizon -- windows of periods solved exactly in
    sequence, each committing its items before the next window is built.

    Returns ``(npv, period)`` where ``period[b]`` is the 0-indexed period item ``b`` is scheduled in,
    or ``-1`` if it is never scheduled.

    Open-pit mine planning uses this for time-phased block extraction scheduling under mill/mining
    capacity, but nothing here is block- or mine-specific: it is a general precedence-constrained,
    capacity-limited scheduling MILP.
    """
    value = np.asarray(value, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    n = value.size
    if capacity.size != n_periods:
        raise ValueError(f"capacity must have length n_periods={n_periods}, got {capacity.size}")

    if n * n_periods <= _DIRECT_MILP_LIMIT:
        return _solve_schedule_window(value, precedence, capacity, n_periods, discount, period_offset=0)

    # Rolling horizon: repeatedly solve the exact MILP over every still-unscheduled item for a small
    # window of upcoming periods, honoring precedence against items already committed in earlier
    # windows, then fix that window's decisions and move on.
    window = max(1, _DIRECT_MILP_LIMIT // max(n, 1))
    pred_of: dict[int, list[int]] = defaultdict(list)
    for b, pred in precedence:
        pred_of[b].append(pred)
    period = np.full(n, -1, dtype=np.int64)
    scheduled = np.zeros(n, dtype=bool)
    npv_total = 0.0
    t = 0
    while t < n_periods:
        w = min(window, n_periods - t)
        candidates = [b for b in range(n) if not scheduled[b]]
        if not candidates:
            break
        local_of = {b: i for i, b in enumerate(candidates)}
        local_precedence = [
            (local_of[b], local_of[pred])
            for b in candidates
            for pred in pred_of.get(b, [])
            if not scheduled[pred] and pred in local_of
        ]
        npv_w, period_w = _solve_schedule_window(
            value[candidates], local_precedence, capacity[t : t + w], w, discount, period_offset=t
        )
        for i, b in enumerate(candidates):
            if period_w[i] != -1:
                scheduled[b] = True
                period[b] = period_w[i]
        npv_total += npv_w
        t += w
    return float(npv_total), period
