"""Scenario-based CVaR selection under uncertainty (work-plan §7-H, H4).

Bridges a calibrated `Posterior` over an uncertain per-item value (IC-1, `.samples(n, rng)`) into a
single-period binary selection decision. Rather than optimizing against one point-estimate value,
`two_stage_stochastic_plan` draws `k_scenarios` realizations and solves the sample-average program:
one shared selection decision `x_b in {0, 1}` per item, with per-scenario realized value
``v_k(x) = sum_b x_b * (price * g[k, b] - item_cost[b])``. The objective trades expected value against
downside risk via ``CVaR_alpha(-v_k(x))`` (Rockafellar–Uryasev), so an item whose *average* realization
looks profitable but whose scenario-conditional downside is large is priced correctly instead of
naively included on its mean alone. CVaR-penalized scenario selection is a general
finance/operations-research technique -- portfolio selection under uncertain returns and project
selection under uncertain payoffs are the same construction; this module's worked instantiation is
mine-planning block selection under grade uncertainty (an item is a block, price is ore price, and the
downside risk being priced is grade uncertainty / ore-waste misclassification risk).

**What "two-stage" does and does not mean here.** The implemented program has exactly one decision
vector, `x`, taken before any scenario is observed, plus the `eta`/`u_k` variables that linearize
CVaR. There is no scenario-indexed operational decision, no recourse constraint or recourse cost, no
second information stage, and therefore no nonanticipativity coupling to enforce -- a scenario's
"recourse value" above is simply the payoff of that same `x` evaluated under scenario `k`. This is a
*single-stage* sample-average risk-adjusted selection model (the deterministic equivalent of a
chance-constrained/CVaR selection), not a two-stage stochastic program with recourse. The function
name is kept for API compatibility with existing callers; a genuine second stage would require
per-scenario decision variables `y_k` with their own constraints and costs, which nothing here
builds. Read the outputs accordingly: `expected_value` and `cvar` describe the one committed
selection, not the value of an adaptive policy.

`cvar_epigraph` is the reusable LP-representable epigraph of CVaR: given ``losses[k] = L_k(x)`` as a
linear map of the (as yet undetermined) decision vector — an ``(K, n)`` coefficient matrix, not realized
numbers — it emits the extra ``eta`` (Value-at-Risk) / ``u_k`` (excess-loss) variable block plus the
``a_ub`` rows enforcing ``u_k >= L_k(x) - eta``, ``u_k >= 0``, ready to be concatenated onto any existing
MILP built on :func:`mixle.relations.branch_and_bound_milp`.

Repo-boundary note (see the PR body for the full explanation): H1 (`min_cost_flow` et al., IC-9) and H3
(`mixle.mine_planning`, since renamed `mixle.precedence_scheduling`) had not landed on ``release/0.8.0``
as of this PR, so neither this task's frozen Public API nor its Algorithm section was written to call
into either module directly — the scenario program here is built entirely on the already-landed
:mod:`mixle.reason.posterior_protocol` (IC-1) and :func:`mixle.relations.branch_and_bound_milp`. Both H1
and H3 have since landed, but this module still imports neither `mixle.relations`' flow surface nor
`mixle.precedence_scheduling`: its scenario program never needed them, so the boundary held on its own
merits rather than by landing-order accident.

J6 (the grand synthesis, work-plan §7-J) extends this scenario program with :func:`risk_adjusted_plan`:
the same grade-uncertainty CVaR objective, but net of the priced environmental/health/carbon liabilities
:func:`mixle.analysis.objective.priced_liabilities` assembles, and subject to the hard no-mine/exposure/
water constraints :func:`mixle.analysis.objective.hard_constraints` assembles — grade, cost, carbon, and
enviro/health terms all trading off on one objective, one MILP. It is a distinct symbol from
:func:`two_stage_stochastic_plan` above (never edited by J6), so the two land in different waves without
a same-file write conflict.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from mixle.reason.posterior_protocol import Posterior
from mixle.relations import branch_and_bound_milp

__all__ = ["StochasticPlan", "cvar_epigraph", "risk_adjusted_plan", "two_stage_stochastic_plan"]

# Risk-aversion weight lambda in ``maximize E[v] - _CVAR_LAMBDA * CVaR_alpha(-v)`` (work-plan §7-H step
# 3). Not exposed on the frozen public signature (only posterior/cost/price/k_scenarios/alpha/rng
# are); fixed at 1.0 so the expected-value term and the CVaR term are weighted equally by default.
_CVAR_LAMBDA = 1.0


def _finite_1d(value: Any, name: str) -> np.ndarray:
    """``value`` as a finite 1-D float array, or a ``ValueError`` naming what was wrong.

    Deliberately not ``.size``-flattening: a ``(2, 3)`` cost matrix has six entries, and quietly
    reading it as a six-item vector turns an alignment mistake into a different, plausible-looking
    problem instead of an error.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of per-item values, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite, got {value!r}")
    return arr


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return out


def _count(value: Any, name: str) -> int:
    """An exact positive integer count -- ``True`` is not one scenario, and ``50.5`` is not fifty."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    """Exact ``CVaR_alpha`` of a finite equal-weight loss sample.

    The Rockafellar-Uryasev objective ``eta + (1 / ((1 - alpha) K)) * sum_k max(L_k - eta, 0)`` is
    piecewise linear in ``eta`` with breakpoints at the sample losses, so its minimum is attained at
    one of them; evaluating all K breakpoints from a sorted cumulative sum is O(K log K) and exact.
    """
    ordered = np.sort(np.asarray(losses, dtype=np.float64))
    k_scenarios = ordered.size
    coef = 1.0 / ((1.0 - alpha) * k_scenarios)
    cumulative = np.cumsum(ordered)
    tail_excess = (cumulative[-1] - cumulative) - (k_scenarios - 1 - np.arange(k_scenarios)) * ordered
    return float(np.min(ordered + coef * tail_excess))


def _discrete_plan(
    x_full: np.ndarray, n_items: int, profit: np.ndarray, mean_profit: np.ndarray, alpha: float, name: str
) -> tuple[np.ndarray, float, float]:
    """The returned selection and the risk/value figures recomputed *from that selection*.

    The solver's own ``eta``/``u_k`` variables describe whatever relaxed point it stopped at; the
    plan a caller acts on is the rounded binary vector. Reading the CVaR out of the solver rather
    than recomputing it for the discrete artifact lets the two disagree -- an ``x`` of 0.49 rounds
    away to "select nothing", whose loss is identically zero and whose CVaR is therefore zero, while
    the solver block still reports the risk of the fractional point. Everything reported here is
    computed from the exact selection being returned, after checking the solver honored integrality.
    """
    relaxed = x_full[:n_items]
    if np.max(np.abs(relaxed - np.round(relaxed)), initial=0.0) > 1e-6:
        raise ValueError(f"{name}: solver returned a fractional selection for binary variables; refusing to certify it")
    extract = np.round(relaxed).astype(bool)
    selected = extract.astype(np.float64)
    return extract, float(mean_profit @ selected), _empirical_cvar(-(profit @ selected), alpha)


class StochasticPlan(NamedTuple):
    """A scenario-optimal selection plan: which items, and its risk profile.

    ``extract`` is the boolean per-item decision (named for this module's mine-planning instantiation;
    kept as-is rather than renamed to something like ``selected`` since existing callers already
    depend on this field name); ``expected_value`` is ``E_k[v_k(extract)]`` over the scenarios the plan
    was optimized against; ``cvar`` is ``CVaR_alpha(-v_k(extract))`` (a more negative value is safer —
    the tail is still profitable; a less negative or positive value is riskier); ``scenarios`` is the
    raw ``(K, n_items)`` value draws used for planning.

    Both figures are recomputed from ``extract`` itself, not read out of the solver's relaxation, so
    they always describe the selection actually being returned.
    """

    extract: np.ndarray
    expected_value: float
    cvar: float
    scenarios: np.ndarray


def cvar_epigraph(losses: Any, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Rockafellar–Uryasev LP epigraph of ``CVaR_alpha`` for a loss that is *linear in the decision*.

    ``losses`` is a ``(K, n)`` matrix such that scenario ``k``'s loss is ``losses[k] @ x`` for the
    not-yet-fixed length-``n`` decision vector ``x`` — e.g. ``losses[k, b] = -(price * g[k, b] -
    cost[b])``, the negated per-scenario recourse value. Returns the pieces of::

        CVaR_alpha(L(x)) = min_{eta, u >= 0}  eta + (1 / ((1 - alpha) * K)) * sum_k u_k
                            s.t.  u_k >= losses[k] @ x - eta

    embeddable alongside any existing constraints/variables on ``x``:

    - ``c_add``: length ``n + 1 + K`` objective row over ``[x, eta, u]`` giving the CVaR value itself
      (``eta``'s coefficient is 1, each ``u_k``'s coefficient is ``1 / ((1 - alpha) * K)``, zero on
      ``x``) — combine with an expected-value objective as ``c_ev_padded - lam * c_add`` for a
      ``sense="max"`` solve of ``E[v] - lam * CVaR_alpha(-v)``.
    - ``a_ub_rows``: ``(K, n + 1 + K)`` rows encoding ``losses[k] @ x - eta - u_k <= 0``.
    - ``b_ub``: length-``K`` zeros, the right-hand side of ``a_ub_rows``.
    - ``var_index``: the column index of ``eta`` in the ``[x, eta, u]`` layout (``= n``); ``u_k`` sits
      at ``var_index + 1 + k``. Give ``eta`` bounds ``(-inf, inf)`` and each ``u_k`` bounds ``(0, inf)``.
    """
    loss = np.asarray(losses, dtype=np.float64)
    if loss.ndim != 2:
        raise ValueError("losses must be a (K, n) matrix: scenario loss as a linear map of the decision")
    if not (0.0 < alpha < 1.0):  # NaN fails this comparison too, which is the intended answer
        raise ValueError("alpha must be in (0, 1)")
    # Every entry becomes a constraint coefficient. A NaN or infinite loss coefficient does not make
    # the program infeasible -- it makes an ordinary-looking row that no solver can interpret, and the
    # answer that comes back is not the answer to any stated problem.
    if not np.isfinite(loss).all():
        raise ValueError("losses must be finite: a NaN/infinite coefficient produces an uninterpretable CVaR row")
    k_scenarios, n = loss.shape
    if k_scenarios == 0:
        raise ValueError("losses must have at least one scenario (K >= 1); CVaR of zero scenarios is undefined")
    coef = 1.0 / ((1.0 - alpha) * k_scenarios)
    var_index = n
    width = n + 1 + k_scenarios

    c_add = np.zeros(width, dtype=np.float64)
    c_add[var_index] = 1.0
    c_add[var_index + 1 :] = coef

    a_ub_rows = np.zeros((k_scenarios, width), dtype=np.float64)
    a_ub_rows[:, :n] = loss
    a_ub_rows[:, var_index] = -1.0
    a_ub_rows[np.arange(k_scenarios), var_index + 1 + np.arange(k_scenarios)] = -1.0
    b_ub = np.zeros(k_scenarios, dtype=np.float64)
    return c_add, a_ub_rows, b_ub, var_index


def two_stage_stochastic_plan(
    posterior: Posterior,
    cost: Any,
    price: float,
    *,
    k_scenarios: int = 50,
    alpha: float = 0.9,
    rng: np.random.Generator,
) -> StochasticPlan:
    """Scenario program: select items to maximize ``E[v] - lambda * CVaR_alpha(-v)``.

    Draws ``g = posterior.samples(k_scenarios, rng)`` (IC-1) as the calibrated value scenarios, forms
    the per-scenario realized value ``v_k(x) = sum_b x_b * (price * g[k, b] - cost[b])``, and
    solves the joint MILP — binary ``x`` plus the free ``eta``/``u_k >= 0`` block from
    :func:`cvar_epigraph` — via :func:`mixle.relations.branch_and_bound_milp`.

    Despite the name (kept for API compatibility), the program has a single here-and-now decision and
    no second-stage recourse variables; see the module docstring for exactly what is and is not
    modeled. Every scenario/economic input is validated as a finite, aligned schema first: a NaN
    price, an infinite cost, a ``(2, 3)`` cost matrix read as six items, or a fractional/Boolean
    scenario count is refused rather than turned into a different, plausible-looking problem.
    """
    cost = _finite_1d(cost, "two_stage_stochastic_plan: cost")
    price = _finite_scalar(price, "two_stage_stochastic_plan: price")
    k_scenarios = _count(k_scenarios, "two_stage_stochastic_plan: k_scenarios")
    n_items = cost.size
    if n_items == 0:
        raise ValueError("two_stage_stochastic_plan: cost must describe at least one item")

    g = np.asarray(posterior.samples(k_scenarios, rng), dtype=np.float64)
    if g.shape != (k_scenarios, n_items):
        raise ValueError(f"posterior.samples returned shape {g.shape}, expected {(k_scenarios, n_items)}")
    if not np.isfinite(g).all():
        raise ValueError("two_stage_stochastic_plan: posterior.samples returned a non-finite scenario value")

    profit = price * g - cost[None, :]  # (K, n_items): v_k(x) = profit[k] @ x
    mean_profit = profit.mean(axis=0)
    losses = -profit  # L_k(x) = -v_k(x)

    c_add, a_ub_rows, b_ub, _var_index = cvar_epigraph(losses, alpha)
    width = a_ub_rows.shape[1]

    c_ev = np.zeros(width, dtype=np.float64)
    c_ev[:n_items] = mean_profit
    objective = c_ev - _CVAR_LAMBDA * c_add  # maximize E[v] - lambda * CVaR(-v)

    bounds = [(0.0, 1.0)] * n_items + [(-np.inf, np.inf)] + [(0.0, np.inf)] * k_scenarios
    integer = list(range(n_items))

    solved = branch_and_bound_milp(objective, a_ub_rows, b_ub, integer=integer, bounds=bounds, sense="max")
    if solved is None:
        raise ValueError("two_stage_stochastic_plan: MILP infeasible for the given items/scenarios")
    _, x_full = solved

    extract, expected_value, cvar = _discrete_plan(
        x_full, n_items, profit, mean_profit, alpha, "two_stage_stochastic_plan"
    )
    return StochasticPlan(extract=extract, expected_value=expected_value, cvar=cvar, scenarios=g)


def risk_adjusted_plan(
    posterior: Posterior,
    cost: Any,
    price: float,
    liabilities: dict,
    constraints: dict,
    *,
    k_scenarios: int = 50,
    alpha: float = 0.9,
    rng: np.random.Generator,
) -> StochasticPlan:
    """J6 — the risk-adjusted-NPV extension of :func:`two_stage_stochastic_plan`.

    Same two-stage scenario program (CVaR objective under scenario uncertainty), but:

    1. Per-scenario recourse value is net of ``liabilities["total"]`` — a length-``n_items`` per-item
       dollar array (typically :func:`mixle.analysis.objective.priced_liabilities`'s output) folding in
       remediation, health, and carbon-price terms: ``v_k(x) = sum_b x_b * (price * g[k, b] -
       cost[b] - liabilities["total"][b])``. An empty/absent ``liabilities`` (``{}`` or no
       ``"total"`` key) means zero liability, i.e. identical to :func:`two_stage_stochastic_plan`.
    2. ``constraints`` (typically :func:`mixle.analysis.objective.hard_constraints`'s output) adds hard
       constraints on top of the shared ``x_b in {0, 1}`` bounds:
       - ``"no_mine_mask"``: boolean array, ``True`` items are hard-fixed to ``x_b = 0`` (this module's
         mine-planning instantiation uses it for G9 no-mine/buffer zones; the mechanism itself is a
         general hard-exclusion mask) by tightening their variable bounds to ``(0, 0)`` — exact, not
         just penalized.
       - ``"caps"``: a list of ``{"coeffs": array, "bound": float}`` linear rows (already normalized to
         the solver's ``<=`` convention), each added as an extra ``a_ub`` row ``coeffs @ x <= bound``
         (K6 exposure budgets, L6 water budgets, or any other per-item activity cap).

    Reuses :func:`cvar_epigraph` for the CVaR epigraph and :func:`mixle.relations.branch_and_bound_milp`
    for the extended MILP — the same solver ``two_stage_stochastic_plan`` uses, so value, cost, and any
    other priced terms all trade off against each other on one objective.
    """
    cost = _finite_1d(cost, "risk_adjusted_plan: cost")
    price = _finite_scalar(price, "risk_adjusted_plan: price")
    k_scenarios = _count(k_scenarios, "risk_adjusted_plan: k_scenarios")
    n_items = cost.size
    if n_items == 0:
        raise ValueError("risk_adjusted_plan: cost must describe at least one item")

    g = np.asarray(posterior.samples(k_scenarios, rng), dtype=np.float64)
    if g.shape != (k_scenarios, n_items):
        raise ValueError(f"posterior.samples returned shape {g.shape}, expected {(k_scenarios, n_items)}")
    if not np.isfinite(g).all():
        raise ValueError("risk_adjusted_plan: posterior.samples returned a non-finite scenario value")

    liability_total = liabilities.get("total") if liabilities else None
    if liability_total is None:
        liability = np.zeros(n_items, dtype=np.float64)
    else:
        liability = _finite_1d(liability_total, "risk_adjusted_plan: liabilities['total']")
        if liability.shape != (n_items,):
            raise ValueError(f"risk_adjusted_plan: liabilities['total'] shape {liability.shape} != {(n_items,)}")

    profit = price * g - cost[None, :] - liability[None, :]  # (K, n_items): v_k(x) = profit[k] @ x
    mean_profit = profit.mean(axis=0)
    losses = -profit  # L_k(x) = -v_k(x)

    c_add, a_ub_rows, b_ub, _var_index = cvar_epigraph(losses, alpha)
    width = a_ub_rows.shape[1]

    c_ev = np.zeros(width, dtype=np.float64)
    c_ev[:n_items] = mean_profit
    objective = c_ev - _CVAR_LAMBDA * c_add  # maximize E[v] - lambda * CVaR(-v)

    bounds = [(0.0, 1.0)] * n_items + [(-np.inf, np.inf)] + [(0.0, np.inf)] * k_scenarios

    constraints = constraints or {}
    # A hard constraint is a policy decision, so the controls carrying it are a closed schema: an
    # unrecognized key is a miswired or misspelled control that would otherwise be dropped in
    # silence, leaving the caller believing a restriction is in force that never reached the solver.
    unknown = set(constraints) - {"no_mine_mask", "caps"}
    if unknown:
        raise ValueError(f"risk_adjusted_plan: unknown constraint key(s) {sorted(unknown)}")
    no_mine_mask = constraints.get("no_mine_mask")
    if no_mine_mask is not None:
        # NOT dtype=bool coercion: that maps every nonzero object to True, so the string "False",
        # a NaN (truthy -- NaN is only falsy under ==) or a stray sentinel becomes a permanent
        # exclusion, the exact opposite of what the data says. Only a genuine Boolean array decides
        # which items may never be selected.
        mask = np.asarray(no_mine_mask)
        if mask.dtype != np.bool_:
            raise ValueError(
                "risk_adjusted_plan: constraints['no_mine_mask'] must be an actual Boolean array "
                f"(dtype bool), got dtype {mask.dtype} ({no_mine_mask!r})"
            )
        if mask.shape != (n_items,):
            raise ValueError(f"risk_adjusted_plan: constraints['no_mine_mask'] shape {mask.shape} != {(n_items,)}")
        for b in np.flatnonzero(mask):
            bounds[b] = (0.0, 0.0)

    extra_rows: list[np.ndarray] = []
    extra_b: list[float] = []
    for cap in constraints.get("caps") or ():
        unknown_cap = set(cap) - {"coeffs", "bound", "sense"}
        if unknown_cap:
            raise ValueError(f"risk_adjusted_plan: unknown cap key(s) {sorted(unknown_cap)}")
        coeffs = _finite_1d(cap["coeffs"], "risk_adjusted_plan: constraints['caps'] coeffs")
        bound = _finite_scalar(cap["bound"], "risk_adjusted_plan: constraints['caps'] bound")
        sense = cap.get("sense", "<=")
        if sense == ">=":
            coeffs, bound = -coeffs, -bound
        elif sense != "<=":
            raise ValueError(f"risk_adjusted_plan: cap 'sense' must be '<=' or '>=', got {sense!r}")
        if coeffs.shape != (n_items,):
            raise ValueError(f"risk_adjusted_plan: constraints['caps'] coeffs shape {coeffs.shape} != {(n_items,)}")
        row = np.zeros(width, dtype=np.float64)
        row[:n_items] = coeffs
        extra_rows.append(row)
        extra_b.append(bound)

    if extra_rows:
        a_ub_rows = np.vstack([a_ub_rows, np.array(extra_rows)])
        b_ub = np.concatenate([b_ub, np.array(extra_b, dtype=np.float64)])

    integer = list(range(n_items))
    solved = branch_and_bound_milp(objective, a_ub_rows, b_ub, integer=integer, bounds=bounds, sense="max")
    if solved is None:
        raise ValueError("risk_adjusted_plan: MILP infeasible for the given items/scenarios/constraints")
    _, x_full = solved

    extract, expected_value, cvar = _discrete_plan(x_full, n_items, profit, mean_profit, alpha, "risk_adjusted_plan")
    return StochasticPlan(extract=extract, expected_value=expected_value, cvar=cvar, scenarios=g)
