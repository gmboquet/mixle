"""Digital-twin simulation of the mine -> plant -> distribution pipeline (H8).

A :class:`PipelineTwin` is a period-stepped re-solve of the production network's flow: each period
it re-runs :func:`mixle.relations.min_cost_flow` (IC-9) under the current arc capacities/costs and a
draw of stochastic arrivals, then accumulates queue/bottleneck diagnostics as state. It is wrapped
as a :class:`mixle.inference.simulate.Simulator` so the per-period stochastic draws come from the
same ``.sampler(seed=...)`` surface every other simulated model in the codebase uses, and named
interventions are registered through the same :class:`~mixle.inference.simulate.Scenario` bookkeeping
the simulate module already exposes -- though, unlike a learned Bayesian network, a deterministic
flow network has no ``do``-operator to route through, so interventions here are applied directly as
arc/capacity/supply overrides (see :meth:`PipelineTwin._apply_interventions`).

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

# Penalty terms for the slack node that keeps every period's flow problem feasible even when the
# real subnetwork cannot fully satisfy conservation (a saturated arc, a plant outage, ...). Slack
# capacity is large enough to never itself bind; slack cost is large enough to never be preferred
# over any real route, so it only carries what the real network genuinely could not.
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
    """
    n = cap.shape[0]
    cap_ext = np.zeros((n + 1, n + 1))
    cost_ext = np.zeros((n + 1, n + 1))
    cap_ext[:n, :n] = cap
    cost_ext[:n, :n] = cost
    cap_ext[:n, n] = _SLACK_CAPACITY
    cap_ext[n, :n] = _SLACK_CAPACITY
    cost_ext[:n, n] = _SLACK_COST
    cost_ext[n, :n] = _SLACK_COST
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
        self._arrival_noise = arrival_noise
        self._interventions: dict[str, dict[str, Any]] = dict(interventions or {})
        self._simulator = Simulator(_ArrivalModel(len(self._supply_nodes), arrival_noise))
        for name, iv in self._interventions.items():
            self._simulator.scenarios[name] = Scenario(name, {})

    def scenario(self, name: str, interventions: dict[str, Any]) -> PipelineTwin:
        """Register a named intervention (plant-down / grade-shift / demand-spike / new arc / ...).

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
            elif kind == "grade_shift":
                # {node: multiplier} -- perturb a mine's effective usable feed grade/tonnage.
                for node, mult in spec.items():
                    supply[node] *= float(mult)
            elif kind == "demand_delta":
                # {node: additive delta} -- demand-spike (or relief) at a customer node.
                for node, delta in spec.items():
                    supply[node] += float(delta)
            else:
                raise KeyError(f"unknown intervention kind {kind!r}")

    def run(self, n_periods: int, *, scenario: str | None = None) -> dict:
        """Step the twin ``n_periods`` periods, re-solving the flow each period.

        Returns a dict of per-period diagnostics: ``throughput`` (delivered per period),
        ``queue`` (cumulative unmet demand), ``utilization`` (per-period ``(n, n)`` arc
        flow/capacity ratios), ``bottleneck_arc`` (the ``(u, v)`` with highest utilization in the
        final period), and ``bottleneck_utilization`` (that arc's utilization series).
        """
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
        queue_series: list[float] = []
        utilization_series: list[np.ndarray] = []
        bottleneck_arcs: list[tuple[int, int] | None] = []
        bottleneck_utils: list[float] = []
        queue_total = 0.0

        for t in range(int(n_periods)):
            period_supply = supply.copy()
            noise = noises[t]
            for i, node in enumerate(self._supply_nodes):
                period_supply[node] += float(noise[i])

            cap_ext, cost_ext, supply_ext = _augment_with_slack(cap, cost, period_supply)
            result = min_cost_flow(cap_ext, cost_ext, supply_ext)
            flow = result.flow[:n, :n]

            delivered = flow[:, self._demand_nodes].sum(axis=0)
            shortfall = float(np.clip(nominal_demand - delivered, 0.0, None).sum())
            queue_total += shortfall

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
            queue_series.append(queue_total)
            utilization_series.append(util)

        return {
            "throughput": np.array(throughput_series),
            "queue": np.array(queue_series),
            "utilization": np.array(utilization_series),
            "bottleneck_arc": bottleneck_arcs[-1],
            "bottleneck_utilization": np.array(bottleneck_utils),
        }


def build_twin(network: dict, transport: dict, *, seed: int = 0) -> PipelineTwin:
    """Build a :class:`PipelineTwin` from a network spec and (optionally empty) H5 transport caps.

    ``network`` keys: ``cap``/``cost`` (``(n, n)`` arc matrices), ``supply`` (length ``n``, positive
    = mine/source, negative = customer/demand), ``supply_nodes``/``demand_nodes`` (node-index lists;
    inferred from the sign of ``supply`` if omitted), and an optional ``arrival_noise`` std-dev for
    the per-period stochastic-arrivals draw (default 0, deterministic).

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
