"""``EpistemicJournal`` -- an append-only, replayable decision log for a sequence of loop steps.

The program plan's decision journal (§3.4: "timestamp, model version, belief snapshot hash, options
considered with their EIG/cost/risk scores, chosen action, rationale claims..."), following
:class:`mixle.evolve.ledger.EvolutionLedger`'s existing shape (flat, JSON-serializable, append-only,
no model objects stored directly) rather than inventing a third bespoke logging pattern.

One deliberate refinement over a hash-only ledger: each record carries the *full* serialized belief
snapshot (``portfolio_snapshot``, from :meth:`~mixle.epistemic.portfolio.HypothesisPortfolio.to_dict`)
alongside its content-address (``belief_snapshot_hash``). A hash alone cannot be reversed back into a
portfolio, so "an auditor can reconstruct the full belief trajectory from the ledger alone" (program
plan §2) requires the snapshot content to actually be there; the hash is what lets :meth:`verify`
catch tampering/corruption of that stored content, which is the property a bare append-only list
doesn't get for free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from mixle.epistemic.loop import EpistemicStep
from mixle.epistemic.portfolio import HypothesisPortfolio


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _hash_snapshot(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionRecord:
    """One journaled decision: what was believed, what was considered, what was chosen, and why."""

    step_index: int
    belief_snapshot_hash: str
    portfolio_snapshot: dict
    surprise: float
    action_considered: list[Any] = field(default_factory=list)
    action_chosen: Any | None = None
    action_eig: float | None = None
    timestamp: float | None = None
    rationale: str | None = None


class EpistemicJournal:
    """An ordered, JSON-serializable, replayable log of :class:`~mixle.epistemic.loop.EpistemicStep`\\ s."""

    def __init__(self, records: list[DecisionRecord] | None = None) -> None:
        self.records: list[DecisionRecord] = list(records) if records else []

    def append(
        self,
        step: EpistemicStep,
        *,
        action_considered: list[Any] = (),
        rationale: str | None = None,
        timestamp: float | None = None,
    ) -> DecisionRecord:
        """Append one record for ``step`` and return it. ``timestamp`` is caller-supplied, never sampled here."""
        snapshot = step.portfolio_after.to_dict()
        record = DecisionRecord(
            step_index=len(self.records),
            belief_snapshot_hash=_hash_snapshot(snapshot),
            portfolio_snapshot=snapshot,
            surprise=step.surprise,
            action_considered=list(action_considered),
            action_chosen=step.next_action,
            action_eig=step.next_action_eig,
            timestamp=timestamp,
            rationale=rationale,
        )
        self.records.append(record)
        return record

    def replay(self, portfolio0: HypothesisPortfolio | None = None) -> list[HypothesisPortfolio]:
        """Reconstruct the belief trajectory from the journal's stored snapshots alone.

        ``portfolio0`` is accepted for interface symmetry with the loop's own ``step(portfolio, ...)``
        signature but is not required for reconstruction here: every record already carries its own
        full ``portfolio_snapshot``, so replay is deserialization, not re-simulation (re-simulation
        would additionally need the original observations and likelihood callables, which are
        deliberately not journaled -- they may not be JSON-serializable, and the snapshot is the thing
        an audit actually needs). If given, ``portfolio0`` is prepended to the returned trajectory.
        """
        trajectory = [HypothesisPortfolio.from_dict(r.portfolio_snapshot) for r in self.records]
        return ([portfolio0] + trajectory) if portfolio0 is not None else trajectory

    def verify(self) -> bool:
        """Return whether every record's stored snapshot still matches its recorded content-address."""
        return all(_hash_snapshot(r.portfolio_snapshot) == r.belief_snapshot_hash for r in self.records)

    def to_json(self, **dumps_kwargs: Any) -> str:
        return json.dumps([asdict(r) for r in self.records], default=_json_default, **dumps_kwargs)

    @classmethod
    def from_json(cls, s: str) -> EpistemicJournal:
        rows = json.loads(s)
        return cls([DecisionRecord(**row) for row in rows])

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


__all__ = ["DecisionRecord", "EpistemicJournal"]
