"""Two-stage stochastic programming with CVaR risk under scenario uncertainty (work-plan §7-H, H4).

Bridges a calibrated `Posterior` over an uncertain per-item value (IC-1, `.samples(n, rng)`) into a
single-period binary selection decision. Rather than optimizing against one point-estimate value,
`two_stage_stochastic_plan` draws `k_scenarios` realizations and solves a two-stage scenario program:
one shared first-stage selection decision `x_b in {0, 1}` per item, with per-scenario recourse value
``v_k(x) = sum_b x_b * (price * g[k, b] - item_cost[b])``. The objective trades expected value against
downside risk via ``CVaR_alpha(-v_k(x))`` (Rockafellar–Uryasev), so an item whose *average* realization
looks profitable but whose scenario-conditional downside is large is priced correctly instead of
naively included on its mean alone. Two-stage CVaR-penalized scenario selection is a general
finance/operations-research technique -- portfolio selection under uncertain returns and project
selection under uncertain payoffs are the same construction; this module's worked instantiation is
mine-planning block selection under grade uncertainty (an item is a block, price is ore price, and the
downside risk being priced is grade uncertainty / ore-waste misclassification risk).

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


class StochasticPlan(NamedTuple):
    """A two-stage scenario-optimal selection plan: which items, and its risk profile.

    ``extract`` is the boolean per-item decision (named for this module's mine-planning instantiation;
    kept as-is rather than renamed to something like ``selected`` since existing callers already
    depend on this field name); ``expected_value`` is ``E_k[v_k(extract)]`` over the scenarios the plan
    was optimized against; ``cvar`` is ``CVaR_alpha(-v_k(extract))`` (a more negative value is safer —
    the tail is still profitable; a less negative or positive value is riskier); ``scenarios`` is the
    raw ``(K, n_items)`` value draws used for planning.
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
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
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
    """Two-stage scenario program: select items to maximize ``E[v] - lambda * CVaR_alpha(-v)``.

    Draws ``g = posterior.samples(k_scenarios, rng)`` (IC-1) as the calibrated value scenarios, forms
    the per-scenario recourse value ``v_k(x) = sum_b x_b * (price * g[k, b] - cost[b])``, and
    solves the joint MILP — binary ``x`` plus the free ``eta``/``u_k >= 0`` block from
    :func:`cvar_epigraph` — via :func:`mixle.relations.branch_and_bound_milp`.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_items = cost.size

    g = np.asarray(posterior.samples(k_scenarios, rng), dtype=np.float64)
    if g.shape != (k_scenarios, n_items):
        raise ValueError(f"posterior.samples returned shape {g.shape}, expected {(k_scenarios, n_items)}")

    profit = price * g - cost[None, :]  # (K, n_items): v_k(x) = profit[k] @ x
    mean_profit = profit.mean(axis=0)
    losses = -profit  # L_k(x) = -v_k(x)

    c_add, a_ub_rows, b_ub, var_index = cvar_epigraph(losses, alpha)
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

    extract = np.round(x_full[:n_items]).astype(bool)
    eta_star = float(x_full[var_index])
    u_star = x_full[var_index + 1 :]
    coef = 1.0 / ((1.0 - alpha) * k_scenarios)
    cvar = eta_star + coef * float(u_star.sum())
    expected_value = float(mean_profit @ extract.astype(np.float64))

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
    cost = np.asarray(cost, dtype=np.float64)
    n_items = cost.size

    g = np.asarray(posterior.samples(k_scenarios, rng), dtype=np.float64)
    if g.shape != (k_scenarios, n_items):
        raise ValueError(f"posterior.samples returned shape {g.shape}, expected {(k_scenarios, n_items)}")

    liability_total = liabilities.get("total") if liabilities else None
    if liability_total is None:
        liability = np.zeros(n_items, dtype=np.float64)
    else:
        liability = np.asarray(liability_total, dtype=np.float64)
        if liability.shape != (n_items,):
            raise ValueError(f"risk_adjusted_plan: liabilities['total'] shape {liability.shape} != {(n_items,)}")

    profit = price * g - cost[None, :] - liability[None, :]  # (K, n_items): v_k(x) = profit[k] @ x
    mean_profit = profit.mean(axis=0)
    losses = -profit  # L_k(x) = -v_k(x)

    c_add, a_ub_rows, b_ub, var_index = cvar_epigraph(losses, alpha)
    width = a_ub_rows.shape[1]

    c_ev = np.zeros(width, dtype=np.float64)
    c_ev[:n_items] = mean_profit
    objective = c_ev - _CVAR_LAMBDA * c_add  # maximize E[v] - lambda * CVaR(-v)

    bounds = [(0.0, 1.0)] * n_items + [(-np.inf, np.inf)] + [(0.0, np.inf)] * k_scenarios

    constraints = constraints or {}
    no_mine_mask = constraints.get("no_mine_mask")
    if no_mine_mask is not None:
        mask = np.asarray(no_mine_mask, dtype=bool)
        if mask.shape != (n_items,):
            raise ValueError(f"risk_adjusted_plan: constraints['no_mine_mask'] shape {mask.shape} != {(n_items,)}")
        for b in np.flatnonzero(mask):
            bounds[b] = (0.0, 0.0)

    extra_rows: list[np.ndarray] = []
    extra_b: list[float] = []
    for cap in constraints.get("caps") or ():
        coeffs = np.asarray(cap["coeffs"], dtype=np.float64)
        bound = float(cap["bound"])
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

    extract = np.round(x_full[:n_items]).astype(bool)
    eta_star = float(x_full[var_index])
    u_star = x_full[var_index + 1 :]
    coef = 1.0 / ((1.0 - alpha) * k_scenarios)
    cvar = eta_star + coef * float(u_star.sum())
    expected_value = float(mean_profit @ extract.astype(np.float64))

    return StochasticPlan(extract=extract, expected_value=expected_value, cvar=cvar, scenarios=g)
