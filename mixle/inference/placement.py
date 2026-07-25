"""Placement planning for certified estimation blocks.

The estimation certificate identifies how each block is solved and whether it
is eligible for offloading. Placement adds the execution decision: local or
pool. Blocks stay local by default. A block is assigned to the pool only when it
is pool-eligible, a pool is configured, and the estimated work clears the
round-trip cost threshold.

The result is a :class:`PlacementPlan` with per-block placement, an estimated
cost, and a human-readable reason. With no pool configured, the plan is
all-local.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Real
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

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("pool available must be a boolean")
        for name, value, allow_zero in (
            ("cost_per_hour", self.cost_per_hour, True),
            ("flop_threshold_tflop", self.flop_threshold_tflop, True),
            ("pool_speedup", self.pool_speedup, False),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)) or (float(value) < 0.0 if allow_zero else float(value) <= 0.0):
                domain = "non-negative" if allow_zero else "strictly positive"
                raise ValueError(f"{name} must be finite and {domain}")


@dataclass
class BlockPlacement:
    """Where one block runs and why: 'local' or 'pool', with the priced justification."""

    name: str
    kind: str
    placement: str  # 'local' | 'pool'
    reason: str
    est_tflop: float = 0.0
    est_cost: float = 0.0
    estimate_source: str = "none"
    execution_status: str = "not_executed"
    observed_placement: str | None = None

    def __str__(self) -> str:
        c = f", ~${self.est_cost:.2f}" if self.placement == "pool" else ""
        return f"{self.name} [{self.kind}] -> {self.placement}{c}  ({self.reason})"


@dataclass
class PlacementPlan:
    """An unexecuted local-vs-pool plan, never an execution receipt."""

    placements: list[BlockPlacement] = field(default_factory=list)
    pool_spec: PoolSpec | None = None
    context_digest: str | None = None
    bindings: dict[str, str | None] = field(default_factory=dict)
    execution_status: str = "not_executed"

    @property
    def pool_blocks(self) -> list[BlockPlacement]:
        """Return blocks assigned to pool execution."""
        return [p for p in self.placements if p.placement == "pool"]

    @property
    def local_blocks(self) -> list[BlockPlacement]:
        """Return blocks assigned to local execution."""
        return [p for p in self.placements if p.placement == "local"]

    @property
    def est_pool_cost(self) -> float:
        """Return estimated total pool spend."""
        return float(sum(p.est_cost for p in self.pool_blocks))

    def as_dict(self) -> dict[str, Any]:
        """Return the placement plan as JSON-compatible data."""
        return {
            "n_blocks": len(self.placements),
            "n_pool": len(self.pool_blocks),
            "est_pool_cost": self.est_pool_cost,
            "context_digest": self.context_digest,
            "bindings": dict(self.bindings),
            "execution_status": self.execution_status,
            "placements": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "placement": p.placement,
                    "reason": p.reason,
                    "est_tflop": p.est_tflop,
                    "est_cost": p.est_cost,
                    "estimate_source": p.estimate_source,
                    "execution_status": p.execution_status,
                    "observed_placement": p.observed_placement,
                }
                for p in self.placements
            ],
        }

    def report(self) -> str:
        """Render a human-readable placement report."""
        head = (
            f"PlacementPlan: {len(self.local_blocks)} local, {len(self.pool_blocks)} pool"
            f" (~${self.est_pool_cost:.2f} pool spend)"
        )
        return "\n".join([head] + [f"  {p}" for p in self.placements])

    def __str__(self) -> str:
        return self.report()


def _est_tflop(block: BlockPlan) -> float | None:
    """A coarse TFLOP estimate for a gradient block from its resource profile, if any.

    Reads an ``est_tflop`` hint from the block reason or metadata when present;
    otherwise returns a conservative default that keeps small blocks local.
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
    return None


def plan_placement(
    certificate: EstimationCertificate,
    pool: PoolSpec | None = None,
    *,
    telemetry: Any = None,
    model_digest: str | None = None,
    data_digest: str | None = None,
    version_digest: str | None = None,
) -> PlacementPlan:
    """Decide local vs pool for every block of a certified fit (see module docstring).

    Closed-form / convex / EM blocks always stay local. A gradient (``pool_eligible``) block goes to
    the pool only when ``pool.available`` and its estimated work clears ``pool.flop_threshold_tflop``;
    otherwise it stays local too. Each decision carries a priced reason, and (with a pool) a
    ``placement`` telemetry event is emitted per block.
    """
    pool = pool or PoolSpec(available=False)
    bindings = {
        "model": model_digest,
        "data": data_digest,
        "version": version_digest,
    }
    digest_payload = {
        "blocks": [
            {
                "name": block.name,
                "kind": block.kind,
                "method": block.method,
                "placement": block.placement,
                "reason": block.reason,
            }
            for block in certificate.blocks
        ],
        "pool": {
            "available": pool.available,
            "cost_per_hour": pool.cost_per_hour,
            "flop_threshold_tflop": pool.flop_threshold_tflop,
            "pool_speedup": pool.pool_speedup,
        },
        "bindings": bindings,
    }
    context_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    placements: list[BlockPlacement] = []
    for b in certificate.blocks:
        if b.placement != "pool_eligible":
            placements.append(
                BlockPlacement(b.name, b.kind, "local", "closed-form / convex / EM -- runs local by design")
            )
            continue
        tflop = _est_tflop(b)
        if tflop is None:
            placements.append(
                BlockPlacement(
                    b.name,
                    b.kind,
                    "local",
                    "gradient block has no measured work estimate -- keep local",
                    estimate_source="unavailable",
                )
            )
            continue
        if not pool.available:
            placements.append(
                BlockPlacement(
                    b.name,
                    b.kind,
                    "local",
                    "gradient block, but no pool configured -- plan local",
                    tflop,
                    estimate_source="block_reason_hint",
                )
            )
        elif tflop < pool.flop_threshold_tflop:
            placements.append(
                BlockPlacement(
                    b.name,
                    b.kind,
                    "local",
                    f"gradient block ~{tflop:.1f} TFLOP below the {pool.flop_threshold_tflop} threshold -- local",
                    tflop,
                    estimate_source="block_reason_hint",
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
                    f"gradient residual ~{tflop:.1f} TFLOP -- plan pool "
                    f"(assumed {pool.pool_speedup:.0f}x speedup)",
                    tflop,
                    round(cost, 4),
                    estimate_source="block_reason_hint",
                )
            )
    plan = PlacementPlan(
        placements=placements,
        pool_spec=pool,
        context_digest=context_digest,
        bindings=bindings,
    )
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
                outcome={
                    "est_cost": p.est_cost,
                    "execution_status": "not_executed",
                    "context_digest": plan.context_digest,
                },
            )
    except Exception:  # noqa: BLE001 - telemetry must never break planning
        pass
