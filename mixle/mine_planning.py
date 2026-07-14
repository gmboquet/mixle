"""Production scheduling & block sequencing: ultimate pit limit and time-phased extraction.

A block model assigns each block ``b`` an economic value ``value_b`` (positive for ore net of
processing cost, negative for waste net of removal cost) and a *precedence* relation: block ``b``
cannot be extracted before its predecessor blocks (the material above it that a stable slope
requires be removed first). Two questions sit on top of that model:

* **Ultimate pit** (:func:`ultimate_pit`) -- ignoring time, which subset of blocks maximizes total
  value subject to precedence? This is exactly the maximum-weight closure of the precedence DAG
  (Lerchs-Grossmann), which Picard's construction reduces to a single min-cut: connect the source to
  every positive-value block with capacity ``value_b``, connect every negative-value block to the
  sink with capacity ``|value_b|``, and add an infinite-capacity arc ``b -> pred`` for every
  precedence pair -- an infinite arc can never be cut, so the source side of the min-cut can never
  contain a block without also containing its predecessors, which is exactly the closure property.
  Solved with :func:`mixle.relations.max_flow` / :func:`mixle.relations.min_cut` (Edmonds-Karp is a
  valid, if not asymptotically optimal, pseudoflow substitute for this fixed-size combinatorial
  problem).
* **Time-phased schedule** (:func:`schedule_extraction`) -- given a per-period mill/mining capacity
  and a discount rate, *when* (which period, if ever) should each block be mined to maximize net
  present value, honoring precedence at every period boundary and never exceeding capacity? This is
  a mixed-integer program (binary ``x[b, t]`` = block ``b`` mined in period ``t``) solved by
  :func:`mixle.relations.branch_and_bound_milp`.

    >>> import numpy as np
    >>> value = np.array([-1.0, 10.0])       # block 0 is waste, block 1 is ore beneath it
    >>> precedence = [(1, 0)]                # block 1 requires block 0 first
    >>> ultimate_pit(value, precedence)
    array([ True,  True])
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.relations import branch_and_bound_milp, min_cut

__all__ = ["ultimate_pit", "schedule_extraction"]


def ultimate_pit(block_value: Any, precedence: Sequence[tuple[int, int]]) -> np.ndarray:
    """Optimal ultimate pit limit: the maximum-weight closure of the precedence DAG.

    ``block_value`` is a length-``n`` array of per-block net value (positive = ore, negative =
    waste). ``precedence`` lists ``(b, pred)`` pairs meaning block ``b`` requires block ``pred`` be
    extracted first (typically the blocks a stable slope forces to be removed above ``b``). Returns
    a length-``n`` boolean mask, ``True`` for every block in the value-maximizing, precedence-closed
    pit (Picard's max-flow reduction of Lerchs-Grossmann): a super-source connects to every
    positive-value block with capacity equal to its value, every negative-value block connects to a
    super-sink with capacity equal to its magnitude, and each precedence pair becomes an
    effectively-infinite-capacity arc ``b -> pred`` -- such an arc is never cut, so the min-cut's
    source side can never hold ``b`` without also holding ``pred``, which is exactly the closure
    property. The source side of the minimum cut (:func:`mixle.relations.min_cut`) is the optimal
    pit. "Effectively infinite" is a finite big-M (the total absolute block value plus one) rather
    than a literal ``np.inf``: :func:`mixle.relations.max_flow` reconstructs flow as ``cap -
    residual``, and an infinite ``cap`` on an arc that ever carries flow leaves ``residual`` also
    infinite, so that subtraction is ``inf - inf = nan`` -- which then makes
    :func:`mixle.relations.min_cut`'s reachability BFS silently drop the arc (``nan > tol`` is
    false), breaking the closure guarantee. A finite big-M that no optimal cut could ever prefer to
    sever avoids the NaN while keeping the same combinatorial meaning.
    """
    value = np.asarray(block_value, dtype=np.float64)
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
            raise ValueError(f"precedence pair {(b, pred)} references a block outside 0..{n - 1}")
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
    """Rows encoding, for every precedence pair and every period ``t``: cumulative-mined(b) by ``t``
    ``<=`` cumulative-mined(pred) by ``t`` -- ``pred`` can never trail ``b``."""
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for b, pred in precedence:
        if not (0 <= b < n and 0 <= pred < n):
            raise ValueError(f"precedence pair {(b, pred)} references a block outside 0..{n - 1}")
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

    ``value``/``precedence`` are already restricted to the candidate blocks for this window (local
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
    for b in range(n):  # each block mined at most once across this window
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
        # mining nothing always satisfies every constraint, so this should be unreachable.
        raise ValueError("schedule_extraction: MILP came back infeasible on a window that admits the empty schedule")
    npv, x_flat = result
    x = x_flat.reshape(n, n_periods)
    period = np.full(n, -1, dtype=np.int64)
    for b in range(n):
        mined_at = np.where(x[b] > 0.5)[0]
        if mined_at.size:
            period[b] = period_offset + int(mined_at[0])
    return float(npv), period


# Above this many (block x period) binary variables, solve a rolling horizon instead of one MILP:
# branch-and-bound over the full horizon is exact but its worst case is exponential in n_vars, so
# large block models are chunked into windows that are each solved exactly and then committed.
_DIRECT_MILP_LIMIT = 400


def schedule_extraction(
    block_value: Any,
    precedence: Sequence[tuple[int, int]],
    mill_capacity: Any,
    n_periods: int,
    *,
    discount: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Time-phased extraction schedule maximizing discounted value under precedence and capacity.

    ``block_value`` is a length-``n`` array of per-block net value; ``precedence`` lists ``(b,
    pred)`` pairs as in :func:`ultimate_pit`; ``mill_capacity`` is a length-``n_periods`` array
    bounding how many blocks may be mined in each period (one block = one unit of capacity: the
    algorithm note's per-block "tonnage" collapses to 1 here since the frozen signature carries no
    separate tonnage array -- callers wanting heterogeneous tonnage can pre-scale ``mill_capacity``
    or split a heavy block into unit sub-blocks). Binary ``x[b, t]`` = block ``b`` mined in period
    ``t``; precedence is enforced at every period boundary (cumulative extraction of ``pred`` can
    never trail ``b``'s), each block is mined at most once (never, if it is never worth it), and the
    objective discounts each period's value by ``(1 + discount) ** t``. Solved exactly by
    :func:`mixle.relations.branch_and_bound_milp`; block models large enough to make the full
    horizon's MILP impractical are solved as a rolling horizon -- windows of periods solved exactly
    in sequence, each committing its blocks before the next window is built.

    Returns ``(npv, period)`` where ``period[b]`` is the 0-indexed period block ``b`` is mined in, or
    ``-1`` if it is never mined.
    """
    value = np.asarray(block_value, dtype=np.float64)
    capacity = np.asarray(mill_capacity, dtype=np.float64)
    n = value.size
    if capacity.size != n_periods:
        raise ValueError(f"mill_capacity must have length n_periods={n_periods}, got {capacity.size}")

    if n * n_periods <= _DIRECT_MILP_LIMIT:
        return _solve_schedule_window(value, precedence, capacity, n_periods, discount, period_offset=0)

    # Rolling horizon: repeatedly solve the exact MILP over every still-unmined block for a small
    # window of upcoming periods, honoring precedence against blocks already committed in earlier
    # windows, then fix that window's decisions and move on.
    window = max(1, _DIRECT_MILP_LIMIT // max(n, 1))
    pred_of: dict[int, list[int]] = defaultdict(list)
    for b, pred in precedence:
        pred_of[b].append(pred)
    period = np.full(n, -1, dtype=np.int64)
    mined = np.zeros(n, dtype=bool)
    npv_total = 0.0
    t = 0
    while t < n_periods:
        w = min(window, n_periods - t)
        candidates = [b for b in range(n) if not mined[b]]
        if not candidates:
            break
        local_of = {b: i for i, b in enumerate(candidates)}
        local_precedence = [
            (local_of[b], local_of[pred])
            for b in candidates
            for pred in pred_of.get(b, [])
            if not mined[pred] and pred in local_of
        ]
        npv_w, period_w = _solve_schedule_window(
            value[candidates], local_precedence, capacity[t : t + w], w, discount, period_offset=t
        )
        for i, b in enumerate(candidates):
            if period_w[i] != -1:
                mined[b] = True
                period[b] = period_w[i]
        npv_total += npv_w
        t += w
    return float(npv_total), period
