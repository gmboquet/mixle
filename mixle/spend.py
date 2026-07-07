"""``Spend`` -- the unified budget ledger every subsystem accumulates into (workstream J: SPEND-a).

One cost total, summed across every card that spends something on a query: workstream B's cascade/router
tiers (``frontier_calls``), workstream C6/I's oracle-scored search loops (``oracle_calls``), and wall-clock /
dollar cost wherever those are tracked. ``System.answer`` carries the incremental ``Spend`` of *this* call plus
the running ``total_spend`` on every receipt, and treats ``budget`` as a hard ceiling: a request that cannot
afford even the cheapest answer path is refused -- with the shortfall named on the receipt -- rather than
silently served over budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Spend:
    """A summable cost total. ``total_units()`` is the scalar figure ``budget=`` is checked against."""

    frontier_calls: int = 0
    oracle_calls: int = 0
    wall_ms: float = 0.0
    dollars: float = 0.0

    def __add__(self, other: Spend) -> Spend:
        return Spend(
            frontier_calls=self.frontier_calls + other.frontier_calls,
            oracle_calls=self.oracle_calls + other.oracle_calls,
            wall_ms=self.wall_ms + other.wall_ms,
            dollars=self.dollars + other.dollars,
        )

    def total_units(self) -> float:
        """The scalar cost a ``budget=`` integer is measured against.

        Currently ``frontier_calls + oracle_calls`` -- the two countable, per-call costs every existing
        card (B's cascade tiers, I's oracle budget) already measures budget in. ``wall_ms``/``dollars`` are
        carried and reported on every receipt but not yet priced into the hard-ceiling check; extend this
        method (not the call sites) when a real dollar cost model lands.
        """
        return float(self.frontier_calls + self.oracle_calls)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "frontier_calls": self.frontier_calls,
            "oracle_calls": self.oracle_calls,
            "wall_ms": self.wall_ms,
            "dollars": self.dollars,
        }
