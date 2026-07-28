"""The evolution ledger: an in-memory, JSON-serializable record of every improvement attempt.

The library layer does no I/O, so the ledger is just an in-process list of rows -- one per proposed
challenger -- that an orchestrator can persist (e.g. into a registry version's metadata). Each row is a
plain dict so it round-trips through ``json.dumps`` without custom encoders: the model objects
themselves are never stored, only their operator, measured delta, verdict, cost, and parent hash.

The ledger is the *evidence* for a claimed trail of autonomous improvement attempts, so it owns its
rows rather than handing out the live ones. Every row carries a sequence number, a schema version, and
a digest chained to its predecessor (:func:`_row_hash`); :meth:`EvolutionLedger.verify` re-derives the
chain, so a changed delta, a rewritten verdict, a deleted attempt, or a reordering is detectable
rather than silent. :meth:`EvolutionLedger.record` and the ``rows`` view both return deep copies:
mutating what you were handed cannot rewrite what was recorded.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

LEDGER_SCHEMA_VERSION = 1
"""Version of the row schema, stamped on every row so a later reader can detect an older layout."""

_GENESIS_HASH = "0" * 64
"""``prev_hash`` of the first row: a fixed anchor, so truncating the ledger from the front is visible."""

_HASHED_KEYS = ("seq", "schema_version", "operator", "delta", "verdict", "cost", "parent_hash", "meta", "prev_hash")


def _row_hash(row: dict[str, Any]) -> str:
    """Digest a row's canonical content -- every field except the digest itself, plus ``prev_hash``.

    Including ``prev_hash`` makes the rows a chain: altering row ``i`` invalidates every row after it,
    so no localized edit can be made to verify.
    """
    payload = json.dumps({key: row[key] for key in _HASHED_KEYS}, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EvolutionLedger:
    """An ordered, JSON-serializable, integrity-chained log of improvement attempts."""

    _rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        """A read-only view: a tuple of deep copies of the recorded rows.

        The backing list used to be the public attribute holding the live row dicts, so a caller could
        edit a recorded delta or verdict in place, or ``clear()`` the whole trail, with nothing to show
        it happened. Reading, indexing, iterating and ``json.dumps``-ing the returned rows all work as
        before; writing to them simply does not reach the ledger.
        """
        return tuple(copy.deepcopy(row) for row in self._rows)

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
        """Append one attempt row and return a copy of it.

        The returned dict is a deep copy, not the stored object: the previous behaviour handed back the
        exact row still held by the ledger, so a caller that adjusted a field on "its" result silently
        rewrote the recorded attempt.
        """
        row: dict[str, Any] = {
            "seq": len(self._rows),
            "schema_version": LEDGER_SCHEMA_VERSION,
            "operator": operator,
            "delta": float(delta),
            "verdict": copy.deepcopy(verdict),
            "cost": float(cost),
            "parent_hash": parent_hash,
            "meta": copy.deepcopy(meta) if meta else {},
            "prev_hash": self._rows[-1]["row_hash"] if self._rows else _GENESIS_HASH,
        }
        row["row_hash"] = _row_hash(row)
        self._rows.append(row)
        return copy.deepcopy(row)

    def verify(self) -> bool:
        """Whether every row still matches its digest and its place in the chain.

        Checks, per row: the sequence number matches its position, ``prev_hash`` matches the previous
        row's ``row_hash`` (:data:`_GENESIS_HASH` for the first), and ``row_hash`` still matches the
        row's canonical content. A rewritten delta or verdict, a deleted or reordered attempt, and a
        front-truncated ledger all fail.
        """
        expected_prev = _GENESIS_HASH
        for index, row in enumerate(self._rows):
            if row.get("seq") != index or row.get("prev_hash") != expected_prev:
                return False
            if _row_hash(row) != row.get("row_hash"):
                return False
            expected_prev = row["row_hash"]
        return True

    def to_json(self, **dumps_kwargs: Any) -> str:
        """Serialize the full ledger to a JSON string (rows are already plain dicts)."""
        return json.dumps(self._rows, default=_json_default, **dumps_kwargs)

    @classmethod
    def from_json(cls, s: str) -> EvolutionLedger:
        """Rebuild a ledger from :meth:`to_json` output. Call :meth:`verify` to check its integrity."""
        return cls(list(json.loads(s)))

    def __len__(self) -> int:
        return len(self._rows)

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


__all__ = ["EvolutionLedger", "LEDGER_SCHEMA_VERSION"]
