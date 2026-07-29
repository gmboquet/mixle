"""Digital-twin simulation of a min-cost-flow network under stochastic arrivals (H8).

A :class:`PipelineTwin` is a period-stepped re-solve of a production network's flow: each period
it re-runs :func:`mixle.relations.min_cost_flow` (IC-9) under the current arc capacities/costs and a
draw of stochastic arrivals, then accumulates queue/bottleneck diagnostics as state. It is wrapped
as a :class:`mixle.inference.simulate.Simulator` so the per-period stochastic draws come from the
same ``.sampler(seed=...)`` surface every other simulated model in the codebase uses, and named
interventions are registered through the same :class:`~mixle.inference.simulate.Scenario` bookkeeping
the simulate module already exposes -- though, unlike a learned Bayesian network, a deterministic
flow network has no ``do``-operator to route through, so interventions here are applied directly as
arc/capacity/supply overrides (see :meth:`PipelineTwin._apply_interventions`).

Nothing about the twin itself is domain-specific: it is a general period-stepped supply/demand
network simulator with pluggable scenario interventions (new/changed arcs, node outages, supply-rate
shifts, demand shifts). This module's worked instantiation is a mine -> plant -> distribution
pipeline (source nodes are mines, sinks are customers, "plant-down" and "supply-rate shift" are the
named interventions below), but the network topology, capacities, and interventions are all caller-supplied.

Network capacities can be tightened by H5 (``mixle_pde.material_transport``) transport-physics
numbers -- those arrive as a plain ``{(u, v): capacity}`` mapping (no cross-plugin import) and are
combined with the network's nameplate capacities in :func:`build_twin`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.inference.simulate import Scenario, Simulator
from mixle.relations import min_cost_flow

__all__ = ["build_twin", "PipelineTwin"]

# Fallback/minimum penalty terms for the slack node that keeps every period's flow problem feasible
# even when the real subnetwork cannot fully satisfy conservation (a saturated arc, a plant outage,
# ...). These are only a floor: a fixed constant cannot universally keep the "slack capacity never
# binds, slack cost never beats a real route" promise for an arbitrary caller-supplied network whose
# own supply or arc costs exceed these numbers, so :func:`_augment_with_slack` derives the actual
# per-problem values from the real ``cap``/``cost``/``supply`` it is given and takes the larger of
# that derivation and these fallbacks -- these only matter for small/degenerate problems.
_SLACK_CAPACITY = 1.0e6
_SLACK_COST = 1.0e4


class _ArrivalModel:
    """Minimal ``.sampler(seed=...)`` surface -- the twin's stochastic-arrivals source, so the twin
    can be packaged as a :class:`mixle.inference.simulate.Simulator` like any other fitted model."""

    def __init__(self, n_supply: int, noise_scale: float) -> None:
        self.n_supply = n_supply
        self.noise_scale = noise_scale

    def sampler(self, seed: int = 0) -> _ArrivalSampler:
        return _ArrivalSampler(self.n_supply, self.noise_scale, seed)


class _ArrivalSampler:
    """Draws one perturbation vector (over supply nodes) per simulated period."""

    def __init__(self, n_supply: int, noise_scale: float, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_supply = n_supply
        self._noise_scale = noise_scale

    def sample(self, n_draws: int) -> list[np.ndarray]:
        if self._noise_scale <= 0.0:
            return [np.zeros(self._n_supply) for _ in range(int(n_draws))]
        return [self._rng.normal(0.0, self._noise_scale, size=self._n_supply) for _ in range(int(n_draws))]


def _augment_with_slack(
    cap: np.ndarray, cost: np.ndarray, supply: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add one universal slack node so ``min_cost_flow`` is always feasible for a period's draw.

    The slack node supplies whatever a deficit node is short, and absorbs whatever a surplus node
    cannot push through the real arcs, at a heavy per-unit penalty -- so real capacity is always
    preferred, and only the genuinely-unroutable remainder ever touches it.

    Both the slack capacity and slack cost are derived from THIS call's actual ``cap``/``cost``/
    ``supply`` (falling back to the module-level ``_SLACK_CAPACITY``/``_SLACK_COST`` only as a floor
    for small/degenerate problems), rather than trusting a fixed constant to dominate an arbitrary
    caller-supplied network:

    * Capacity: no single node's own edge to the slack node ever needs to carry more than the
      period's total supply/demand magnitude (conservatively, no node can source or sink more flow
      than exists in the whole system), so ``sum(|supply|)`` -- plus a margin -- is a safe bound that
      keeps slack capacity from ever itself being the binding constraint, regardless of how large
      this period's supply is relative to the fixed default.
    * Cost: a real simple path uses each arc at most once, so its cost magnitude can never exceed the
      sum of ``|cost|`` over every real (``cap > 0``) arc -- summing magnitudes bounds it regardless
      of arc-cost sign. Pricing the slack round-trip (one edge in, one out, ``2 * slack_cost``) above
      that ceiling guarantees a real route is always strictly preferred whenever one can satisfy
      demand at all; slack only ever carries the genuinely-unroutable remainder.
    """
    n = cap.shape[0]
    real_arcs = cap > 0.0
    total_supply_magnitude = float(np.abs(supply).sum())
    slack_capacity = max(_SLACK_CAPACITY, total_supply_magnitude + 1.0)
    real_cost_ceiling = float(np.abs(cost[real_arcs]).sum()) if real_arcs.any() else 0.0
    slack_cost = max(_SLACK_COST, real_cost_ceiling + 1.0)

    cap_ext = np.zeros((n + 1, n + 1))
    cost_ext = np.zeros((n + 1, n + 1))
    cap_ext[:n, :n] = cap
    cost_ext[:n, :n] = cost
    cap_ext[:n, n] = slack_capacity
    cap_ext[n, :n] = slack_capacity
    cost_ext[:n, n] = slack_cost
    cost_ext[n, :n] = slack_cost
    supply_ext = np.append(supply, -float(supply.sum()))
    return cap_ext, cost_ext, supply_ext


class PipelineTwin:
    """A period-stepped digital twin of the production network's flow.

    Each :meth:`run` re-solves ``min_cost_flow`` per period under the twin's (possibly
    scenario-modified) capacities/costs and a draw of stochastic arrivals on the supply nodes,
    tracking per-period throughput/queue/bottleneck-arc diagnostics.
    """

    def __init__(
        self,
        cap: np.ndarray,
        cost: np.ndarray,
        supply: np.ndarray,
        *,
        supply_nodes: list[int],
        demand_nodes: list[int],
        seed: int = 0,
        arrival_noise: float = 0.0,
        interventions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._cap = np.array(cap, dtype=float)
        self._cost = np.array(cost, dtype=float)
        self._supply = np.array(supply, dtype=float)
        self._supply_nodes = list(supply_nodes)
        self._demand_nodes = list(demand_nodes)
        self._seed = seed
        # arrival_noise is a standard deviation, so it has no meaning below zero and a NaN silently
        # turns every period's supply into NaN. Both used to construct fine and only show up as an
        # unexplained run much later.
        if not np.isfinite(arrival_noise) or float(arrival_noise) < 0.0:
            raise ValueError(
                f"arrival_noise is a standard deviation: it must be finite and >= 0, got {arrival_noise!r}"
            )
        self._arrival_noise = float(arrival_noise)
        self._interventions: dict[str, dict[str, Any]] = dict(interventions or {})
        self._simulator = Simulator(_ArrivalModel(len(self._supply_nodes), arrival_noise))
        for name, iv in self._interventions.items():
            self._simulator.scenarios[name] = Scenario(name, {})

    def scenario(self, name: str, interventions: dict[str, Any]) -> PipelineTwin:
        """Register a named intervention (node outage / supply-rate shift / demand-spike / new arc / ...).

        Returns a new twin sharing this one's network and seed but with ``name`` available to
        :meth:`run` via its ``scenario=`` argument -- the base twin (and any other scenario already
        registered on it) is left untouched.
        """
        merged = dict(self._interventions)
        merged[name] = dict(interventions)
        return PipelineTwin(
            self._cap,
            self._cost,
            self._supply,
            supply_nodes=self._supply_nodes,
            demand_nodes=self._demand_nodes,
            seed=self._seed,
            arrival_noise=self._arrival_noise,
            interventions=merged,
        )

    def _apply_interventions(
        self, cap: np.ndarray, cost: np.ndarray, supply: np.ndarray, interventions: dict[str, Any]
    ) -> None:
        """Apply a scenario's interventions in place onto this period's cap/cost/supply arrays."""
        for kind, spec in interventions.items():
            if kind == "add_arc":
                # {(u, v): (capacity, cost)} -- e.g. an inter-plant transfer arc.
                for (u, v), (c, w) in spec.items():
                    cap[u, v] = float(c)
                    cost[u, v] = float(w)
            elif kind == "zero_capacity_node":
                # plant-down: knock out every arc touching the given node(s).
                nodes = spec if isinstance(spec, (list, tuple, set)) else [spec]
                for node in nodes:
                    cap[node, :] = 0.0
                    cap[:, node] = 0.0
            elif kind == "supply_multiplier":
                # {node: multiplier} -- perturb a supply node's rate (this module's mine-planning
                # instantiation uses it for an effective usable feed grade/tonnage shift).
                for node, mult in spec.items():
                    supply[node] *= float(mult)
            elif kind == "demand_delta":
                # {node: additive change in DEMAND, positive = more demand} -- a demand spike (or,
                # negative, relief) at a demand node (a customer, in this module's mine-planning
                # instantiation).
                #
                # The delta is stated in demand, NOT in the signed supply vector this network stores
                # it in. Those have opposite signs: the network convention is positive supply /
                # negative demand, so a documented positive spike of 5 added onto a demand node's
                # supply of -10 produced -5 -- half the demand, the exact opposite of the requested
                # scenario, and delivered throughput fell while the run reported nothing unmet.
                # Subtracting the delta from the signed supply is what "a spike of 5 means 5 more
                # units demanded" actually requires.
                for node, delta in spec.items():
                    if int(node) not in self._demand_nodes:
                        raise ValueError(
                            f"demand_delta targets node {node!r}, which is not one of this twin's demand "
                            f"nodes {self._demand_nodes!r}; a demand change has no meaning at a supply node"
                        )
                    supply[node] -= float(delta)
            else:
                raise KeyError(f"unknown intervention kind {kind!r}")

    def run(self, n_periods: int, *, scenario: str | None = None) -> dict:
        """Step the twin ``n_periods`` periods, re-solving the flow each period.

        Returns a dict of per-period diagnostics: ``throughput`` (delivered per period),
        ``lost_sales`` (demand unmet in that period), ``lost_sales_cumulative`` (the running total),
        ``demand`` (the effective per-demand-node demand actually solved against, after any scenario
        interventions), ``utilization`` (per-period ``(n, n)`` arc flow/capacity ratios),
        ``bottleneck_arc`` (the ``(u, v)`` with highest utilization in the final period), and
        ``bottleneck_utilization`` (that arc's utilization series).

        ``lost_sales`` is LOST demand, not a backlog. Every period solves against the same nominal
        demand; unmet demand is counted and dropped, never added to a later period's demand or carried
        as state, so it can never be served late. This series used to be reported as ``queue``, which
        read as a backlog that would eventually clear: a capacity-five network facing demand ten
        reported a growing "queue" of 5 then 10 while the second solve still requested only ten rather
        than the outstanding fifteen. Modelling an actual backlog needs per-node carried state in the
        conservation constraints, with aging, service and terminal treatment defined; until that
        exists, the honest name for this number is lost sales.

        Raises :class:`ValueError` for ``n_periods < 1`` -- matching this class's other invalid-input
        conventions (an unregistered ``scenario`` name or intervention ``kind`` raises rather than
        returning a degraded result), a zero/negative period count has no meaningful per-period
        diagnostics to report, and silently returning empty series would only move today's crash (an
        unconditional ``bottleneck_arcs[-1]`` index) to whatever code reads the result instead of
        surfacing it here at the call that actually got the invalid count.
        """
        # `int(n_periods) < 1` accepted 2.7 and then silently ran two periods: the truncation
        # happened inside the guard that was supposed to be validating the value, so a caller asking
        # for a fractional horizon got a shorter run with no indication the request had been changed.
        if isinstance(n_periods, (bool, np.bool_)) or not isinstance(n_periods, (int, np.integer)):
            raise ValueError(f"n_periods must be an exact integer, got {n_periods!r}")
        n_periods = int(n_periods)
        if n_periods < 1:
            raise ValueError(f"n_periods must be >= 1, got {n_periods!r}")

        cap = self._cap.copy()
        cost = self._cost.copy()
        supply = self._supply.copy()
        if scenario is not None:
            if scenario not in self._interventions:
                raise KeyError(f"no scenario named {scenario!r}; register it with .scenario(...) first")
            self._apply_interventions(cap, cost, supply, self._interventions[scenario])

        n = cap.shape[0]
        arc_mask = cap > 0.0
        noises = self._simulator.run(int(n_periods), seed=self._seed)

        nominal_demand = -supply[self._demand_nodes]

        throughput_series: list[float] = []
        lost_sales_series: list[float] = []
        lost_sales_cumulative: list[float] = []
        utilization_series: list[np.ndarray] = []
        bottleneck_arcs: list[tuple[int, int] | None] = []
        bottleneck_utils: list[float] = []
        lost_total = 0.0

        for t in range(int(n_periods)):
            period_supply = supply.copy()
            noise = noises[t]
            for i, node in enumerate(self._supply_nodes):
                period_supply[node] += float(noise[i])

            cap_ext, cost_ext, supply_ext = _augment_with_slack(cap, cost, period_supply)
            result = min_cost_flow(cap_ext, cost_ext, supply_ext)
            flow = result.flow[:n, :n]

            # Delivery is what a demand node *keeps*: inflow minus whatever it forwards on. Counting
            # gross inflow credited a demand node with every unit that merely passed through it, so a
            # distribution centre that consumes 4 and forwards 6 reported 10 delivered. That is the
            # twin's headline throughput number, and it overstated by the entire transit volume --
            # while the matching lost_sales, clipped at zero, hid any real shortfall at that node
            # behind the same transit. Identical to the old value for a pure sink, which has no
            # outgoing flow to subtract.
            inflow = flow[:, self._demand_nodes].sum(axis=0)
            outflow = flow[self._demand_nodes, :].sum(axis=1)
            delivered = inflow - outflow
            shortfall = float(np.clip(nominal_demand - delivered, 0.0, None).sum())
            lost_total += shortfall

            util = np.zeros_like(cap)
            util[arc_mask] = flow[arc_mask] / cap[arc_mask]

            if arc_mask.any():
                masked = np.where(arc_mask, util, -np.inf)
                u, v = (int(x) for x in np.unravel_index(np.argmax(masked), masked.shape))
                bottleneck_arcs.append((u, v))
                bottleneck_utils.append(float(util[u, v]))
            else:
                bottleneck_arcs.append(None)
                bottleneck_utils.append(0.0)

            throughput_series.append(float(delivered.sum()))
            lost_sales_series.append(shortfall)
            lost_sales_cumulative.append(lost_total)
            utilization_series.append(util)

        return {
            "throughput": np.array(throughput_series),
            "lost_sales": np.array(lost_sales_series),
            "lost_sales_cumulative": np.array(lost_sales_cumulative),
            "demand": np.array(nominal_demand, dtype=float),
            "utilization": np.array(utilization_series),
            "bottleneck_arc": bottleneck_arcs[-1],
            "bottleneck_utilization": np.array(bottleneck_utils),
        }


def build_twin(network: dict, transport: dict, *, seed: int = 0) -> PipelineTwin:
    """Build a :class:`PipelineTwin` from a network spec and (optionally empty) H5 transport caps.

    ``network`` keys: ``cap``/``cost`` (``(n, n)`` arc matrices), ``supply`` (length ``n``, positive
    = source (a mine, in this module's worked instantiation), negative = demand (a customer)),
    ``supply_nodes``/``demand_nodes`` (node-index lists; inferred from the sign of ``supply`` if
    omitted), and an optional ``arrival_noise`` std-dev for the per-period stochastic-arrivals draw
    (default 0, deterministic).

    ``transport`` is a plain ``{(u, v): capacity}`` mapping of H5 (``mixle_pde.material_transport``)
    derived real-world throughput ceilings (slurry line / conveyor limits, etc.) -- these are combined
    with ``network``'s nameplate arc capacities via ``min`` (no cross-plugin import: the twin only
    ever sees the resulting plain numbers).
    """
    cap = np.array(network["cap"], dtype=float)
    cost = np.array(network["cost"], dtype=float)
    supply = np.array(network["supply"], dtype=float)

    supply_nodes = network.get("supply_nodes")
    if supply_nodes is None:
        supply_nodes = [i for i, s in enumerate(supply) if s > 0.0]
    demand_nodes = network.get("demand_nodes")
    if demand_nodes is None:
        demand_nodes = [i for i, s in enumerate(supply) if s < 0.0]

    for (u, v), capacity in (transport or {}).items():
        cap[u, v] = min(cap[u, v], float(capacity)) if cap[u, v] > 0.0 else float(capacity)

    arrival_noise = float(network.get("arrival_noise", 0.0))

    return PipelineTwin(
        cap,
        cost,
        supply,
        supply_nodes=list(supply_nodes),
        demand_nodes=list(demand_nodes),
        seed=seed,
        arrival_noise=arrival_noise,
    )
