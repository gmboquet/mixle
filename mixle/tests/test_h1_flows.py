"""Min-cost & multi-commodity network flow + capacitated network design (H1: IC-9 implementation)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from mixle.relations import Design, Flow, min_cost_flow, multicommodity_flow, network_design

# Nodes: 0-2 mines, 3-4 plants, 5-8 customers.
N = 9
MINES = (0, 1, 2)
PLANTS = (3, 4)
CUSTOMERS = (5, 6, 7, 8)


def _base_network():
    """3-mine / 2-plant / 4-customer network. Plant 3 is capacity-bottlenecked: mines 0+1 push more
    supply into it (40) than its "native" customers 5+6 demand (30), so 10 units must detour through
    the expensive direct arc 3->7 in the absence of an inter-plant transfer arc."""
    cap = np.zeros((N, N))
    cost = np.zeros((N, N))

    def arc(u, v, c, w):
        cap[u, v] = c
        cost[u, v] = w

    arc(0, 3, 25, 2)  # mine0 -> plant3
    arc(1, 3, 25, 2)  # mine1 -> plant3
    arc(2, 4, 25, 2)  # mine2 -> plant4
    arc(3, 5, 20, 1)  # plant3 -> customer5
    arc(3, 6, 20, 1)  # plant3 -> customer6
    arc(3, 7, 20, 6)  # plant3 -> customer7 (expensive long-haul fallback)
    arc(4, 7, 20, 1)  # plant4 -> customer7
    arc(4, 8, 20, 1)  # plant4 -> customer8

    supply = np.zeros(N)
    supply[0], supply[1], supply[2] = 20.0, 20.0, 20.0
    supply[5], supply[6], supply[7], supply[8] = -15.0, -15.0, -15.0, -15.0
    return cap, cost, supply


def _reference_min_cost_flow(cap, cost, supply):
    """Hand-built linprog reference: one variable per existing arc, node-arc incidence for A_eq."""
    n = cap.shape[0]
    arcs = [(u, v) for u in range(n) for v in range(n) if cap[u, v] > 0.0]
    incidence = np.zeros((n, len(arcs)))
    c = np.zeros(len(arcs))
    bounds = []
    for j, (u, v) in enumerate(arcs):
        incidence[u, j] = 1.0
        incidence[v, j] = -1.0
        c[j] = cost[u, v]
        bounds.append((0.0, cap[u, v]))
    res = linprog(c, A_eq=incidence, b_eq=supply, bounds=bounds, method="highs")
    assert res.success
    return res.fun


def test_min_cost_flow_matches_linprog_reference():
    cap, cost, supply = _base_network()
    result = min_cost_flow(cap, cost, supply)
    assert isinstance(result, Flow)
    reference = _reference_min_cost_flow(cap, cost, supply)
    assert abs(result.value - reference) < 1.0e-6

    # flow conservation at every transship/customer node
    net = result.flow.sum(axis=1) - result.flow.sum(axis=0)
    assert np.allclose(net, supply, atol=1.0e-6)
    assert np.all(result.flow <= cap + 1.0e-6)
    assert np.all(result.flow >= -1.0e-9)


def test_inter_plant_transfer_arc_strictly_lowers_cost():
    cap, cost, supply = _base_network()
    base_value = min_cost_flow(cap, cost, supply).value

    cap2 = cap.copy()
    cost2 = cost.copy()
    cap2[3, 4] = 50.0  # open a cheap inter-plant transfer arc
    cost2[3, 4] = 0.2
    transfer_value = min_cost_flow(cap2, cost2, supply).value

    assert transfer_value < base_value - 1.0e-9

    # matches the hand-solved optimum: the transfer arc entirely displaces the expensive 3->7 fallback
    assert abs(transfer_value - 182.0) < 1.0e-6
    assert abs(base_value - 230.0) < 1.0e-6


def test_min_cost_flow_infeasible_raises():
    cap = np.array([[0.0, 1.0], [0.0, 0.0]])
    cost = np.array([[0.0, 1.0], [0.0, 0.0]])
    supply = np.array([5.0, -5.0])  # arc capacity (1) below required supply (5)
    try:
        min_cost_flow(cap, cost, supply)
        raise AssertionError("expected ValueError for an infeasible instance")
    except ValueError:
        pass


def test_min_cost_flow_unbalanced_supply_raises_value_error():
    # supply must sum to ~zero; this used to be a bare `assert` (an AssertionError, stripped
    # entirely under `python -O`) instead of the ValueError every other input-validation path here
    # raises.
    cap = np.array([[0.0, 1.0], [0.0, 0.0]])
    cost = np.array([[0.0, 1.0], [0.0, 0.0]])
    supply = np.array([5.0, -3.0])  # does not sum to zero
    try:
        min_cost_flow(cap, cost, supply)
        raise AssertionError("expected ValueError for unbalanced supply")
    except ValueError:
        pass


def test_min_cost_flow_cancels_a_negative_cycle_outside_the_supply_route():
    # A feasible unit flow 0 -> 1, satisfiable entirely by the direct 0->1 arc (cost 0), PLUS an
    # independent, disjoint capacity-1 cycle 1 -> 2 -> 1 whose net cost is -5 (1->2 costs -5, 2->1 costs
    # 0). Successive-shortest-path stops the instant the required supply is fully routed -- it never
    # revisits the residual graph to look for a leftover profitable cycle -- so a solver that is only
    # SSP would return 0 (just the 0->1 leg) instead of the true minimum, -5 (also push a unit around the
    # cycle to bank its cost). Nodes 1 and 2 form a genuine antiparallel arc pair (arcs run both 1->2 and
    # 2->1), which is the specific shape that makes canceling this cycle correctly non-trivial: a cancel
    # pass that treats `residual[u, v]` as a single scalar cost can find, "cancel", and then immediately
    # re-find the same cycle forever instead of converging.
    cap = np.zeros((3, 3))
    cost = np.zeros((3, 3))
    cap[0, 1], cost[0, 1] = 1.0, 0.0
    cap[1, 2], cost[1, 2] = 1.0, -5.0
    cap[2, 1], cost[2, 1] = 1.0, 0.0
    supply = np.array([1.0, -1.0, 0.0])

    result = min_cost_flow(cap, cost, supply)

    assert abs(result.value - (-5.0)) < 1.0e-9
    net = result.flow.sum(axis=1) - result.flow.sum(axis=0)
    assert np.allclose(net, supply, atol=1.0e-6)  # still a feasible flow, not just a lower number
    assert np.all(result.flow <= cap + 1.0e-6)
    assert np.all(result.flow >= -1.0e-9)


def test_min_cost_flow_leaves_a_positive_cost_cycle_untouched():
    # Same shape as the negative-cycle case above, but the 1->2->1 cycle nets +5 (2->1 now costs 10, not
    # 0): pushing flow around it would only raise cost, so the post-SSP flow (the required unit routed
    # 0->1 directly, nothing on the 1<->2 arcs) is already optimal and cycle-canceling must be a no-op.
    # Regression guard: confirms the fix does not change answers on instances SSP alone already solved
    # correctly (min_cost_flow is used elsewhere, e.g. mixle.fulfillment.route_distribution and
    # mixle.pipeline_twin's per-period re-solve).
    cap = np.zeros((3, 3))
    cost = np.zeros((3, 3))
    cap[0, 1], cost[0, 1] = 1.0, 0.0
    cap[1, 2], cost[1, 2] = 1.0, -5.0
    cap[2, 1], cost[2, 1] = 1.0, 10.0
    supply = np.array([1.0, -1.0, 0.0])

    result = min_cost_flow(cap, cost, supply)

    assert abs(result.value - 0.0) < 1.0e-9
    assert result.flow[1, 2] == 0.0
    assert result.flow[2, 1] == 0.0


def test_min_cost_flow_cancels_two_independent_negative_cycles():
    # Two disjoint negative cycles that don't touch the required supply route at all: -4 net on nodes
    # 2<->3 and -6 net on nodes 4<->5. Both must be found and canceled -- not just the first one the
    # search happens to hit -- so this guards the cancellation loop actually loops.
    n = 6
    cap = np.zeros((n, n))
    cost = np.zeros((n, n))
    cap[0, 1], cost[0, 1] = 1.0, 0.0
    cap[2, 3], cost[2, 3] = 1.0, -4.0
    cap[3, 2], cost[3, 2] = 1.0, 0.0
    cap[4, 5], cost[4, 5] = 1.0, -7.0
    cap[5, 4], cost[5, 4] = 1.0, 1.0
    supply = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    result = min_cost_flow(cap, cost, supply)

    assert abs(result.value - (-10.0)) < 1.0e-6  # base 0 + (-4) + (-6)


def test_min_cost_flow_unbounded_negative_cycle_raises():
    # An infinite-capacity negative cycle admits no finite minimum -- cost drops without bound as more
    # flow is pushed around it -- so this must raise a clear error rather than loop forever or silently
    # return some arbitrarily-negative flow.
    cap = np.zeros((3, 3))
    cost = np.zeros((3, 3))
    cap[0, 1], cost[0, 1] = 1.0, 0.0
    cap[1, 2], cost[1, 2] = np.inf, -1.0
    cap[2, 1], cost[2, 1] = np.inf, 0.0
    supply = np.array([1.0, -1.0, 0.0])

    try:
        min_cost_flow(cap, cost, supply)
        raise AssertionError("expected a ValueError for an unbounded negative cycle")
    except ValueError as e:
        assert "unbounded" in str(e)


def test_multicommodity_flow_respects_shared_capacity_and_cost():
    # nodes: 0 srcA, 1 srcB, 2 trunk-in, 3 trunk-out, 4 sinkA, 5 sinkB
    n = 6
    cap = np.zeros((n, n))
    cost = np.zeros((n, n))
    cap[0, 2], cost[0, 2] = 100.0, 1.0
    cap[1, 2], cost[1, 2] = 100.0, 1.0
    cap[2, 3], cost[2, 3] = 15.0, 0.0  # shared bottleneck trunk arc
    cap[3, 4], cost[3, 4] = 100.0, 1.0
    cap[3, 5], cost[3, 5] = 100.0, 1.0

    demands = np.array([[0, 4, 7.0], [1, 5, 7.0]])  # commodity A: 0->4 qty 7; commodity B: 1->5 qty 7
    result = multicommodity_flow(cap, cost, demands)
    assert isinstance(result, Flow)
    assert abs(result.value - 28.0) < 1.0e-6  # each commodity pays (1 + 0 + 1) * 7 = 14
    assert result.flow[2, 3] <= cap[2, 3] + 1.0e-6  # shared trunk capacity respected
    assert abs(result.flow[2, 3] - 14.0) < 1.0e-6


def test_multicommodity_flow_infeasible_raises():
    cap = np.array([[0.0, 1.0], [0.0, 0.0]])
    cost = np.array([[0.0, 1.0], [0.0, 0.0]])
    demands = np.array([[0, 1, 5.0]])  # needs 5 units through a capacity-1 arc
    try:
        multicommodity_flow(cap, cost, demands)
        raise AssertionError("expected ValueError for an infeasible instance")
    except ValueError:
        pass


def test_network_design_opens_cheaper_two_hop_path():
    nodes = [0, 1, 2]
    arcs = [(0, 1), (1, 2), (0, 2)]
    fixed_costs = np.array([5.0, 5.0, 100.0])  # direct arc is expensive to open; the two-hop route is cheap
    demands = np.array([10.0, 0.0, -10.0])

    result = network_design(nodes, arcs, fixed_costs, demands)
    assert isinstance(result, Design)
    assert abs(result.cost - 10.0) < 1.0e-6
    assert bool(result.open[0]) is True  # 0 -> 1 opened
    assert bool(result.open[1]) is True  # 1 -> 2 opened
    assert bool(result.open[2]) is False  # the expensive direct arc stays closed
    assert abs(result.flow[0, 1] - 10.0) < 1.0e-6
    assert abs(result.flow[1, 2] - 10.0) < 1.0e-6


def test_network_design_infeasible_raises():
    nodes = [0, 1, 2]
    arcs = [(0, 1)]  # node 2 has no arc at all, so its -10 demand can never be met
    fixed_costs = np.array([1.0])
    demands = np.array([10.0, 0.0, -10.0])
    try:
        network_design(nodes, arcs, fixed_costs, demands)
        raise AssertionError("expected ValueError for an infeasible instance")
    except ValueError:
        pass


def test_network_design_duplicate_candidate_arcs_accumulate_flow():
    # Two candidate arcs for the SAME (0, 1) node pair -- a legitimate use case (e.g. comparing a cheap
    # vs. premium design for one physical link), each getting its own opening-cost/open-decision
    # variable. The optimizer correctly opens only the cheaper candidate (cost 1, not cost 2) and
    # reports the correct total cost, but the returned flow matrix used to be built by *assigning* each
    # candidate's flow into flow_matrix[u, v] rather than accumulating it -- so the second (closed,
    # zero-flow) candidate silently overwrote the first candidate's real flow with 0, and the matrix's
    # own row/column balance stopped matching the requested demand entirely (flow was all zeros).
    nodes = [0, 1]
    arcs = [(0, 1), (0, 1)]
    fixed_costs = np.array([1.0, 2.0])
    demands = np.array([1.0, -1.0])

    result = network_design(nodes, arcs, fixed_costs, demands)

    assert abs(result.cost - 1.0) < 1.0e-6
    assert bool(result.open[0]) is True  # cheaper candidate opened
    assert bool(result.open[1]) is False  # pricier duplicate stays closed
    assert abs(result.flow[0, 1] - 1.0) < 1.0e-6

    # the returned matrix must actually reflect what the solver found, not just the objective/open mask
    net = result.flow.sum(axis=1) - result.flow.sum(axis=0)
    assert np.allclose(net, demands, atol=1.0e-6)


def test_network_design_three_duplicate_candidate_arcs_accumulate_flow():
    # Same idea with three candidates for one node pair, and -- deliberately -- the cheapest (the one
    # that opens and carries flow) is the MIDDLE entry, not the last. A last-write-wins bug would zero
    # the cell out because the last candidate (cost 3) stays closed; a fix that only accumulates
    # correctly for exactly two duplicates (e.g. an off-by-one special case) would also fail here. Guards
    # that accumulation sums over an arbitrary number of duplicates, not just a hardcoded pair.
    nodes = [0, 1]
    arcs = [(0, 1), (0, 1), (0, 1)]
    fixed_costs = np.array([2.0, 1.0, 3.0])
    demands = np.array([1.0, -1.0])

    result = network_design(nodes, arcs, fixed_costs, demands)

    assert abs(result.cost - 1.0) < 1.0e-6
    assert [bool(o) for o in result.open] == [False, True, False]
    assert abs(result.flow[0, 1] - 1.0) < 1.0e-6

    net = result.flow.sum(axis=1) - result.flow.sum(axis=0)
    assert np.allclose(net, demands, atol=1.0e-6)


def test_network_design_non_duplicate_arcs_still_balance_correctly():
    # Regression guard: two independent components, each with exactly one (non-duplicate) candidate
    # arc. Switching flow_matrix[u, v] from assignment to accumulation must be a no-op for the normal
    # case -- a zero-initialized matrix plus a single += is identical to a single "=" -- so both cells
    # must still land exactly on their own component's demand, with no cross-contamination between them.
    nodes = [0, 1, 2, 3]
    arcs = [(0, 1), (2, 3)]
    fixed_costs = np.array([1.0, 1.0])
    demands = np.array([5.0, -5.0, 3.0, -3.0])

    result = network_design(nodes, arcs, fixed_costs, demands)

    assert abs(result.cost - 2.0) < 1.0e-6
    assert bool(result.open[0]) is True
    assert bool(result.open[1]) is True
    assert abs(result.flow[0, 1] - 5.0) < 1.0e-6
    assert abs(result.flow[2, 3] - 3.0) < 1.0e-6

    net = result.flow.sum(axis=1) - result.flow.sum(axis=0)
    assert np.allclose(net, demands, atol=1.0e-6)
