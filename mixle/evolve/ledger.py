"""The evolution ledger: an in-memory, JSON-serializable record of every improvement attempt.

The library layer does no I/O, so the ledger is just an in-process list of rows -- one per proposed
challenger -- that an orchestrator can persist (e.g. into a registry version's metadata). Each row is a
plain dict so it round-trips through ``json.dumps`` without custom encoders: the model objects
themselves are never stored, only their operator, measured delta, verdict, cost, and parent hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionLedger:
    """An ordered, JSON-serializable log of improvement attempts."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        operator: str,
        delta: float,
        verdict: dict | None,
        cost: float,
        parent_hash: str | None,
        meta: dict | None = None,
    ) -> dict[str, Any]:
        """Append one attempt row and return it."""
        row: dict[str, Any] = {
            "operator": operator,
            "delta": float(delta),
            "verdict": verdict,
            "cost": float(cost),
            "parent_hash": parent_hash,
            "meta": dict(meta) if meta else {},
        }
        self.rows.append(row)
        return row

    def to_json(self, **dumps_kwargs: Any) -> str:
        """Serialize the full ledger to a JSON string (rows are already plain dicts)."""
        return json.dumps(self.rows, default=_json_default, **dumps_kwargs)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def _json_default(obj: Any) -> Any:
    """Best-effort fallback so stray numpy scalars / dataclasses don't break serialization."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)


__all__ = ["EvolutionLedger"]
