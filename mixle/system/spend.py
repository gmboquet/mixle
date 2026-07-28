"""Unified budget ledger for subsystem and query-level spending.

One cost total is summed across query paths: cascade or router tiers
(``frontier_calls``), oracle-scored search loops (``oracle_calls``), and
wall-clock or dollar cost wherever those are tracked. ``System.answer`` carries
the incremental ``Spend`` of the current call plus the running ``total_spend``
on every receipt, and treats ``budget`` as a hard ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _count(name: str, value: object) -> int:
    """An exact non-Boolean nonnegative call count, or ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an exact nonnegative integer call count, got {value!r}")
    return value


def _measure(name: str, value: object) -> float:
    """A finite nonnegative measured cost, or ``ValueError``."""
    try:
        measured = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real measured cost, got {value!r}") from exc
    if not math.isfinite(measured) or measured < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return measured


@dataclass(frozen=True)
class Spend:
    """A summable cost total. ``total_units()`` is the scalar figure ``budget=`` is checked against.

    Every dimension is checked on construction, and therefore again after each :meth:`__add__` and
    after deserialization, because a receipt is a ledger: work only ever *consumes* budget. Without
    these invariants ``Spend(frontier_calls=-5).total_units()`` returned ``-5``, so doing work could
    hand budget back, and a NaN ``wall_ms`` serialized as an ordinary receipt field while making every
    hard-ceiling comparison against it false -- failing open on the check meant to stop overspending.
    """

    frontier_calls: int = 0
    oracle_calls: int = 0
    wall_ms: float = 0.0
    dollars: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "frontier_calls", _count("frontier_calls", self.frontier_calls))
        object.__setattr__(self, "oracle_calls", _count("oracle_calls", self.oracle_calls))
        object.__setattr__(self, "wall_ms", _measure("wall_ms", self.wall_ms))
        object.__setattr__(self, "dollars", _measure("dollars", self.dollars))

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
