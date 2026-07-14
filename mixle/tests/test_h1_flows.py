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
