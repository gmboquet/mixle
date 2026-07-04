"""Placement planning -- the 99/1 topology axis of the estimation plan (A4).

The certificate (:mod:`mixle.inference.planning`) says which method solves each block and how strong
the guarantee is. Placement adds the second axis: WHERE each block runs. The rule, enforced here:

    Everything runs local by default. A block is OFFLOADED to the pool only when it is genuinely
    heavy AND pool-eligible (a gradient residual the certificate already flagged) AND a pool is
    configured AND the economics price the round-trip as worth it.

So closed-form / conjugate / convex / EM blocks -- which are exactly what makes 99% of the work fit
on a laptop -- stay local, always. Only the gradient blocks the certificate isolated are candidates
for the small GPU pool. The result is a :class:`PlacementPlan`: per-block placement + a priced
reason, plus the pool jobs to submit. With no pool configured it degrades to all-local, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.inference.planning import BlockPlan, EstimationCertificate

__all__ = ["PoolSpec", "BlockPlacement", "PlacementPlan", "plan_placement"]


@dataclass
class PoolSpec:
    """What the shared GPU pool offers, and when offloading is worth it.

    ``available``: is a pool configured at all (else everything is local). ``cost_per_hour``: the
    pool's price. ``flop_threshold``: the estimated work (in TFLOP) below which the round-trip is not
    worth it -- small gradient blocks stay local. ``local_tflops`` / pool speedup drive the estimate.
    """

    available: bool = False
    cost_per_hour: float = 0.5
    flop_threshold_tflop: float = 1.0  # below this, keep it local (round-trip not worth it)
    pool_speedup: float = 10.0  # how much faster the pool is than local for an eligible block


@dataclass
class BlockPlacement:
    """Where one block runs and why: 'local' or 'pool', with the priced justification."""

    name: str
    kind: str
    placement: str  # 'local' | 'pool'
    reason: str
    est_tflop: float = 0.0
    est_cost: float = 0.0

    def __str__(self) -> str:
        c = f", ~${self.est_cost:.2f}" if self.placement == "pool" else ""
        return f"{self.name} [{self.kind}] -> {self.placement}{c}  ({self.reason})"


@dataclass
class PlacementPlan:
    """The full local-vs-pool plan for a model's blocks, derived from its certificate + a PoolSpec."""

    placements: list[BlockPlacement] = field(default_factory=list)
    pool_spec: PoolSpec | None = None

    @property
    def pool_blocks(self) -> list[BlockPlacement]:
        return [p for p in self.placements if p.placement == "pool"]

    @property
    def local_blocks(self) -> list[BlockPlacement]:
        return [p for p in self.placements if p.placement == "local"]

    @property
    def est_pool_cost(self) -> float:
        return float(sum(p.est_cost for p in self.pool_blocks))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_blocks": len(self.placements),
            "n_pool": len(self.pool_blocks),
            "est_pool_cost": self.est_pool_cost,
            "placements": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "placement": p.placement,
                    "reason": p.reason,
                    "est_tflop": p.est_tflop,
                    "est_cost": p.est_cost,
                }
                for p in self.placements
            ],
        }

    def report(self) -> str:
        head = (
            f"PlacementPlan: {len(self.local_blocks)} local, {len(self.pool_blocks)} pool"
            f" (~${self.est_pool_cost:.2f} pool spend)"
        )
        return "\n".join([head] + [f"  {p}" for p in self.placements])

    def __str__(self) -> str:
        return self.report()


def _est_tflop(block: BlockPlan) -> float:
    """A coarse TFLOP estimate for a gradient block from its resource profile, if any.

    v1: reads an ``est_tflop`` hint off the block's reason/metadata when present, else a conservative
    default that keeps small blocks local. Real profiles arrive with the capability-schema resource
    fields (workstream A1); this is the seam they plug into.
    """
    toks = block.reason.replace("(", " ").replace(")", " ").replace("~", " ").split()
    for i, token in enumerate(toks):
        low = token.lower()
        if low.endswith("tflop") and low != "tflop":  # glued, e.g. "8.0tflop"
            try:
                return float(token[:-5])
            except ValueError:
                pass
        if low == "tflop" and i > 0:  # separate, e.g. "8.0 TFLOP"
            try:
                return float(toks[i - 1])
            except ValueError:
                pass
    return 2.0  # a gradient block with no profile: assume it clears a 1-TFLOP threshold by default


def plan_placement(
    certificate: EstimationCertificate,
    pool: PoolSpec | None = None,
    *,
    telemetry: Any = None,
) -> PlacementPlan:
    """Decide local vs pool for every block of a certified fit (see module docstring).

    Closed-form / convex / EM blocks always stay local. A gradient (``pool_eligible``) block goes to
    the pool only when ``pool.available`` and its estimated work clears ``pool.flop_threshold_tflop``;
    otherwise it stays local too. Each decision carries a priced reason, and (with a pool) a
    ``placement`` telemetry event is emitted per block.
    """
    pool = pool or PoolSpec(available=False)
    placements: list[BlockPlacement] = []
    for b in certificate.blocks:
        if b.placement != "pool_eligible":
            placements.append(
                BlockPlacement(b.name, b.kind, "local", "closed-form / convex / EM -- runs local by design")
            )
            continue
        tflop = _est_tflop(b)
        if not pool.available:
            placements.append(
                BlockPlacement(b.name, b.kind, "local", "gradient block, but no pool configured -- runs local", tflop)
            )
        elif tflop < pool.flop_threshold_tflop:
            placements.append(
                BlockPlacement(
                    b.name,
                    b.kind,
                    "local",
                    f"gradient block ~{tflop:.1f} TFLOP below the {pool.flop_threshold_tflop} threshold -- local",
                    tflop,
                )
            )
        else:
            hours = tflop / max(pool.pool_speedup, 1e-9) / 3600.0 * 1000.0  # coarse: TFLOP -> pool-hours
            cost = hours * pool.cost_per_hour
            placements.append(
                BlockPlacement(
                    b.name,
                    b.kind,
                    "pool",
                    f"gradient residual ~{tflop:.1f} TFLOP -- offload (pool {pool.pool_speedup:.0f}x faster)",
                    tflop,
                    round(cost, 4),
                )
            )
    plan = PlacementPlan(placements=placements, pool_spec=pool)
    _emit(telemetry, plan)
    return plan


def _emit(telemetry: Any, plan: PlacementPlan) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        for p in plan.placements:
            rec(
                "placement",
                features={"kind": p.kind, "est_tflop": p.est_tflop, "has_pool": plan.pool_spec.available},
                choice=p.placement,
                outcome={"est_cost": p.est_cost},
            )
    except Exception:  # noqa: BLE001 - telemetry must never break planning
        pass
