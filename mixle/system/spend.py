"""Unified budget ledger for subsystem and query-level spending.

One cost total is summed across query paths: cascade or router tiers
(``frontier_calls``), oracle-scored search loops (``oracle_calls``), and
wall-clock or dollar cost wherever those are tracked. ``System.answer`` carries
the incremental ``Spend`` of the current call plus the running ``total_spend``
on every receipt, and treats ``budget`` as a hard ceiling.
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

        Currently ``frontier_calls + oracle_calls`` -- the two countable,
        per-call costs the existing routes measure budget in. ``wall_ms`` and
        ``dollars`` are carried and reported on every receipt but are not yet
        priced into the hard-ceiling check; extend this method when a concrete
        dollar cost model is introduced.
        """
        return float(self.frontier_calls + self.oracle_calls)

    def to_dict(self) -> dict[str, float | int]:
        """Serialize the spend ledger into primitive numeric fields."""
        return {
            "frontier_calls": self.frontier_calls,
            "oracle_calls": self.oracle_calls,
            "wall_ms": self.wall_ms,
            "dollars": self.dollars,
        }
