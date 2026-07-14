"""H4 — stochastic / robust optimization under grade uncertainty (work-plan §7-H).

Bridges a calibrated grade `Posterior` (IC-1, `.samples(n, rng)`) into a single-period ore/waste
extraction decision. Rather than optimizing against one point-estimate grade, `two_stage_stochastic_plan`
draws `k_scenarios` grade realizations and solves a two-stage scenario program: one shared first-stage
extraction decision `x_b in {0, 1}` per block, with per-scenario recourse value
``v_k(x) = sum_b x_b * (price * g[k, b] - block_cost[b])``. The objective trades expected value against
downside risk via ``CVaR_alpha(-v_k(x))`` (Rockafellar–Uryasev), so a block whose *average* grade looks
profitable but whose scenario-conditional downside is large (grade uncertainty / ore-waste
misclassification risk) is priced correctly instead of naively included on its mean alone.

`cvar_epigraph` is the reusable LP-representable epigraph of CVaR: given ``losses[k] = L_k(x)`` as a
linear map of the (as yet undetermined) decision vector — an ``(K, n)`` coefficient matrix, not realized
numbers — it emits the extra ``eta`` (Value-at-Risk) / ``u_k`` (excess-loss) variable block plus the
``a_ub`` rows enforcing ``u_k >= L_k(x) - eta``, ``u_k >= 0``, ready to be concatenated onto any existing
MILP built on :func:`mixle.relations.branch_and_bound_milp`.

Repo-boundary note (see the PR body for the full explanation): H1 (`min_cost_flow` et al., IC-9) and H3
(`mixle.mine_planning`) had not landed on ``release/0.8.0`` as of this PR. Neither this task's frozen
Public API nor its Algorithm section actually calls into either module directly — the scenario program
here is built entirely on the already-landed :mod:`mixle.reason.posterior_protocol` (IC-1) and
:func:`mixle.relations.branch_and_bound_milp` — so this module imports neither `mixle.relations`' new
flow surface nor `mixle.mine_planning`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from mixle.reason.posterior_protocol import Posterior
from mixle.relations import branch_and_bound_milp

__all__ = ["StochasticPlan", "cvar_epigraph", "two_stage_stochastic_plan"]

# Risk-aversion weight lambda in ``maximize E[v] - _CVAR_LAMBDA * CVaR_alpha(-v)`` (work-plan §7-H step
# 3). Not exposed on the frozen public signature (only posterior/block_cost/price/k_scenarios/alpha/rng
# are); fixed at 1.0 so the expected-value term and the CVaR term are weighted equally by default.
_CVAR_LAMBDA = 1.0


class StochasticPlan(NamedTuple):
    """A two-stage scenario-optimal extraction plan: which blocks, and its risk profile.

    ``extract`` is the boolean per-block decision; ``expected_value`` is ``E_k[v_k(extract)]`` over the
    scenarios the plan was optimized against; ``cvar`` is ``CVaR_alpha(-v_k(extract))`` (a more negative
    value is safer — the tail is still profitable; a less negative or positive value is riskier);
    ``scenarios`` is the raw ``(K, n_blocks)`` grade draws used for planning.
    """

    extract: np.ndarray
    expected_value: float
    cvar: float
    scenarios: np.ndarray


def cvar_epigraph(losses: Any, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Rockafellar–Uryasev LP epigraph of ``CVaR_alpha`` for a loss that is *linear in the decision*.

    ``losses`` is a ``(K, n)`` matrix such that scenario ``k``'s loss is ``losses[k] @ x`` for the
    not-yet-fixed length-``n`` decision vector ``x`` — e.g. ``losses[k, b] = -(price * g[k, b] -
    block_cost[b])``, the negated per-scenario recourse value. Returns the pieces of::

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
    block_cost: Any,
    price: float,
    *,
    k_scenarios: int = 50,
    alpha: float = 0.9,
    rng: np.random.Generator,
) -> StochasticPlan:
    """Two-stage scenario program: extract blocks to maximize ``E[v] - lambda * CVaR_alpha(-v)``.

    Draws ``g = posterior.samples(k_scenarios, rng)`` (IC-1) as the calibrated grade scenarios, forms
    the per-scenario recourse value ``v_k(x) = sum_b x_b * (price * g[k, b] - block_cost[b])``, and
    solves the joint MILP — binary ``x`` plus the free ``eta``/``u_k >= 0`` block from
    :func:`cvar_epigraph` — via :func:`mixle.relations.branch_and_bound_milp`.
    """
    cost = np.asarray(block_cost, dtype=np.float64)
    n_blocks = cost.size

    g = np.asarray(posterior.samples(k_scenarios, rng), dtype=np.float64)
    if g.shape != (k_scenarios, n_blocks):
        raise ValueError(f"posterior.samples returned shape {g.shape}, expected {(k_scenarios, n_blocks)}")

    profit = price * g - cost[None, :]  # (K, n_blocks): v_k(x) = profit[k] @ x
    mean_profit = profit.mean(axis=0)
    losses = -profit  # L_k(x) = -v_k(x)

    c_add, a_ub_rows, b_ub, var_index = cvar_epigraph(losses, alpha)
    width = a_ub_rows.shape[1]

    c_ev = np.zeros(width, dtype=np.float64)
    c_ev[:n_blocks] = mean_profit
    objective = c_ev - _CVAR_LAMBDA * c_add  # maximize E[v] - lambda * CVaR(-v)

    bounds = [(0.0, 1.0)] * n_blocks + [(-np.inf, np.inf)] + [(0.0, np.inf)] * k_scenarios
    integer = list(range(n_blocks))

    solved = branch_and_bound_milp(objective, a_ub_rows, b_ub, integer=integer, bounds=bounds, sense="max")
    if solved is None:
        raise ValueError("two_stage_stochastic_plan: MILP infeasible for the given blocks/scenarios")
    _, x_full = solved

    extract = np.round(x_full[:n_blocks]).astype(bool)
    eta_star = float(x_full[var_index])
    u_star = x_full[var_index + 1 :]
    coef = 1.0 / ((1.0 - alpha) * k_scenarios)
    cvar = eta_star + coef * float(u_star.sum())
    expected_value = float(mean_profit @ extract.astype(np.float64))

    return StochasticPlan(extract=extract, expected_value=expected_value, cvar=cvar, scenarios=g)
