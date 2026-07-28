"""Precedence-constrained selection and scheduling: maximum-weight closure and time-phased capacity scheduling.

Two general combinatorial-optimization problems recur whenever items each carry a value (positive or
negative) and a *precedence* relation: item ``b`` may not be taken without its predecessor items --
in the scheduling problem, ``b`` may not be scheduled in an earlier period than any of them. (That
is "never later than", not "strictly earlier": a predecessor and its dependent may share a period.
:func:`schedule_activities` spells out what that does and does not guarantee.) Two questions sit on
top of that model:

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
    if not np.isfinite(value).all():
        raise ValueError("maximum_weight_closure: every item value must be finite")
    n = value.size
    source, sink = n, n + 1
    # The big-M has to stay finite (see above) AND stay representable. Summing float64 magnitudes
    # overflows for large-but-perfectly-finite values -- with -1e308 and 1.5e308 the sum was inf,
    # max_flow's cap - residual reconstruction went to NaN, min_cut's reachability BFS silently
    # dropped the precedence arc (NaN > tol is false), and the returned mask selected a dependent
    # without its predecessor: closure violated on finite input. A closure is invariant under a
    # positive rescaling of every value, and halving is exact in binary floating point, so scale the
    # whole instance down by whatever power of two makes the reduction representable.
    peak = float(np.max(np.abs(value), initial=0.0))
    if peak > np.finfo(np.float64).max / max(n, 1):  # sum of magnitudes is at most n * peak
        value = np.ldexp(value, -((n - 1).bit_length() + 1))
    big_m = float(np.abs(value).sum()) + 1.0  # no min-cut ever prefers severing a precedence arc
    cap = np.zeros((n + 2, n + 2), dtype=np.float64)
    for b in range(n):
        if value[b] > 0.0:
            cap[source, b] = value[b]
        elif value[b] < 0.0:
            cap[b, sink] = -value[b]
    edges = []
    for b, pred in precedence:
        if not (0 <= b < n and 0 <= pred < n):
            raise ValueError(f"precedence pair {(b, pred)} references an item outside 0..{n - 1}")
        cap[b, pred] = big_m
        edges.append((b, pred))
    _, source_side, _ = min_cut(cap, source, sink)
    mask = np.zeros(n, dtype=bool)
    for node in source_side:
        if node < n:
            mask[node] = True
    # Post-verify the property this whole reduction exists to guarantee. A cut that leaves a
    # dependent without its predecessor means the min-cut did not see the big-M arc it was given,
    # and returning that mask as "the maximum-weight closure" would be a false certificate.
    violated = [(b, pred) for b, pred in edges if mask[b] and not mask[pred]]
    if violated:
        raise ValueError(f"maximum_weight_closure: min-cut returned a non-closed selection at edges {violated}")
    return mask


def _cumulative_precedence_rows(
    n: int, n_periods: int, precedence: Sequence[tuple[int, int]], idx: Any
) -> tuple[list[np.ndarray], list[float]]:
    """Rows encoding, for every precedence pair and every period ``t``: cumulative-scheduled(b) by
    ``t`` ``<=`` cumulative-scheduled(pred) by ``t`` -- ``pred`` can never trail ``b``.

    Non-strict at each boundary by design: the two may share a period. See
    :func:`schedule_activities` for why that is the intended reading and what to do when it is not.
    """
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


# Window size for the explicitly-approximate mode="rolling" horizon, in (item x period) binaries:
# each window is a MILP solved exactly and then committed. This is a performance knob for a mode the
# caller has to ask for by name -- it must never decide, on its own, which problem gets solved.
_DIRECT_MILP_LIMIT = 400


def schedule_activities(
    value: Any,
    precedence: Sequence[tuple[int, int]],
    capacity: Any,
    n_periods: int,
    *,
    discount: float = 0.0,
    mode: str = "exact",
) -> tuple[float, np.ndarray]:
    """Time-phased schedule maximizing discounted value under precedence and per-period capacity.

    ``value`` is a length-``n`` array of finite per-item net value; ``precedence`` lists ``(b, pred)``
    pairs as in :func:`maximum_weight_closure`; ``capacity`` is a length-``n_periods`` array bounding
    how many items may be scheduled in each period (one item = one unit of capacity: callers wanting
    heterogeneous per-item resource use can pre-scale ``capacity`` or split a heavy item into unit
    sub-items). Binary ``x[b, t]`` = item ``b`` scheduled in period ``t``; precedence is enforced at
    every period boundary, each item is scheduled at most once (never, if it is never worth it), and
    the objective discounts each period's value by ``(1 + discount) ** t``. Solved exactly by
    :func:`mixle.relations.branch_and_bound_milp`.

    **Precedence here means "never later than", not "strictly earlier".** The constraint is that
    ``pred``'s cumulative scheduling can never trail ``b``'s *at any period boundary*, so a
    predecessor and its dependent may legitimately share a period -- with one period, capacity two,
    values ``[1, 10]`` and ``precedence=[(1, 0)]`` both items are scheduled at period 0. That is
    same-boundary closure, the right model when a period is an accounting bucket (a year's mining is
    not internally ordered) and the wrong one when a period is a strict hand-off. For strict
    "finishes before the next one starts", give each item its own period's worth of capacity, or
    model the delay explicitly rather than reading a stronger guarantee into this one. For the same
    reason a cyclic precedence relation is *satisfiable* rather than rejected: a two-cycle forces its
    items into the same period, which is what mutual "never later than" constraints mean. The
    ordering that a strict reading would call a contradiction only exists under strict semantics.

    ``capacity`` entries are finite and non-negative; since items consume one unit each, a fractional
    capacity floors (``0.5`` schedules nothing in that period). ``discount`` must be finite and
    greater than ``-1``: at ``-1`` the discount factor divides by zero, and an infinite one silently
    zeroes all future value rather than discounting it.

    ``mode`` selects the semantics, and only the caller may change them:

    * ``"exact"`` (default) always solves the declared maximum-NPV MILP over the full horizon.
    * ``"rolling"`` is an explicitly approximate myopic rolling horizon -- windows of periods solved
      exactly in sequence, each committing its items before the next window is built. It carries **no
      optimality bound**: because a window cannot see reward that only arrives after it, the
      heuristic can refuse an entire profitable dependency chain. On a 50-item, 10-period instance
      with unit capacity and a 10-item chain (nine predecessors worth -1, terminal item worth 100) it
      returned no schedule and NPV 0, where the full MILP takes the chain in periods 0-9 for NPV 91.

    Above ``_DIRECT_MILP_LIMIT`` item-period binaries this used to switch to ``"rolling"`` on its own,
    so an ordinary optimum-shaped result silently answered a different problem than the one declared.
    Large instances now stay exact unless the caller opts into the approximation.

    Returns ``(npv, period)`` where ``period[b]`` is the 0-indexed period item ``b`` is scheduled in,
    or ``-1`` if it is never scheduled.

    Open-pit mine planning uses this for time-phased block extraction scheduling under mill/mining
    capacity, but nothing here is block- or mine-specific: it is a general precedence-constrained,
    capacity-limited scheduling MILP.
    """
    if mode not in ("exact", "rolling"):
        raise ValueError(f"mode must be 'exact' or 'rolling', got {mode!r}")
    # The calendar, the resource model and the discount rate define the problem; none of them was
    # checked, so out-of-domain values became either a wrong answer or an error blaming the solver.
    # A capacity of -1 made every schedule infeasible and reported that the always-feasible empty
    # schedule was infeasible; a discount of -1 divided by zero and handed linprog a NaN objective;
    # +inf silently zeroed all future value; n_periods=0 surfaced as a shape complaint from scipy.
    if isinstance(n_periods, (bool, np.bool_)) or not isinstance(n_periods, (int, np.integer)) or n_periods < 1:
        raise ValueError(f"n_periods must be a positive integer number of periods, got {n_periods!r}")
    n_periods = int(n_periods)
    if isinstance(discount, (bool, np.bool_)) or not isinstance(discount, (int, float, np.integer, np.floating)):
        raise ValueError(f"discount must be a finite rate greater than -1, got {discount!r}")
    discount = float(discount)
    if not np.isfinite(discount) or discount <= -1.0:
        raise ValueError(f"discount must be a finite rate greater than -1, got {discount!r}")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("schedule_activities: every item value must be finite")
    capacity = np.asarray(capacity, dtype=np.float64)
    n = value.size
    if capacity.size != n_periods:
        raise ValueError(f"capacity must have length n_periods={n_periods}, got {capacity.size}")
    if not np.isfinite(capacity).all() or (capacity < 0.0).any():
        raise ValueError(f"capacity must be finite and non-negative in every period, got {capacity!r}")

    if mode == "exact":
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
