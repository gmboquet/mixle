"""``Registry`` -- a local directory + index of fitted task models, queryable by capability and fingerprint.

The registry is the local library catalog that orchestrators, routers, capture
flows, and accumulation workflows can read from or write into. Deliberately a
directory plus a JSON index, not a server: every entry is a saved
:class:`~mixle.task.model.TaskModel` or :class:`~mixle.task.calibrate.CalibratedTaskModel` artifact directory
(see :mod:`mixle.task.artifact`) plus a small index record naming its capabilities, task fingerprint
(:func:`~mixle.task.edge.task_fingerprint`), and capture profile. ``find_for`` answers "do I already have
something for this task"; ``tier_stack`` turns a matching capability into an ascending-cost tier list -- the
shape :class:`~mixle.task.router.Router` consumes directly (``Router(tiers=stack)``), with the frontier
appended last as the router's own fallback tier.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.model import TaskModel

_INDEX_NAME = "index.json"


@dataclass
class RegistryEntry:
    """One catalog record: where the artifact lives, what it's registered under, and how much it costs to run."""

    entry_id: str
    path: str
    kind: str  # "task" or "calibrated" -- which class reloads the artifact at ``path``
    capabilities: list[str] = field(default_factory=list)
    fingerprint: list[float] | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry entry into JSON-compatible fields."""
        return {
            "entry_id": self.entry_id,
            "path": self.path,
            "kind": self.kind,
            "capabilities": list(self.capabilities),
            "fingerprint": list(self.fingerprint) if self.fingerprint is not None else None,
            "profile": self.profile,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegistryEntry:
        """Create a registry entry from a JSON index record."""
        return cls(
            entry_id=d["entry_id"],
            path=d["path"],
            kind=d["kind"],
            capabilities=list(d.get("capabilities", [])),
            fingerprint=d.get("fingerprint"),
            profile=d.get("profile", {}),
            cost=float(d.get("cost", 0.0)),
        )


class Registry:
    """A ``dir``-backed catalog of registered models: ``register`` writes an artifact + index entry;
    ``find_for``/``tier_stack`` query it. Re-opening the same ``dir`` in a fresh process sees every entry."""

    def __init__(self, dir: str) -> None:
        self.dir = dir
        os.makedirs(dir, exist_ok=True)
        self._entries: list[RegistryEntry] = self._read_index()

    def _index_path(self) -> str:
        return os.path.join(self.dir, _INDEX_NAME)

    def _read_index(self) -> list[RegistryEntry]:
        if not os.path.exists(self._index_path()):
            return []
        with open(self._index_path()) as f:
            return [RegistryEntry.from_dict(d) for d in json.load(f)]

    def _write_index(self) -> None:
        with open(self._index_path(), "w") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2, sort_keys=True)

    def register(
        self,
        model: TaskModel | CalibratedTaskModel,
        *,
        capabilities: Sequence[str],
        fingerprint: Sequence[float] | None = None,
        profile: dict[str, Any] | None = None,
        cost: float = 0.0,
        entry_id: str | None = None,
    ) -> RegistryEntry:
        """Save ``model``'s artifact under ``dir`` and add its index entry; return the entry.

        ``model`` is a fitted :class:`~mixle.task.model.TaskModel` or
        :class:`~mixle.task.calibrate.CalibratedTaskModel` -- the two artifact-saveable task model kinds.
        ``capabilities`` names what this model answers (matched by :meth:`find_for`); ``fingerprint`` is
        typically :func:`~mixle.task.edge.task_fingerprint`'s vector for the training data; ``profile`` is
        free-form (e.g. a :func:`~mixle.task.capability.capture_profile` dict); ``cost`` is the per-request
        cost used to order :meth:`tier_stack`.
        """
        if isinstance(model, CalibratedTaskModel):
            kind = "calibrated"
        elif isinstance(model, TaskModel):
            kind = "task"
        else:
            raise TypeError(f"Registry only stores TaskModel/CalibratedTaskModel, got {type(model)!r}")
        entry_id = entry_id or f"entry_{len(self._entries):04d}"
        path = os.path.join(self.dir, entry_id)
        model.save(path)
        entry = RegistryEntry(
            entry_id=entry_id,
            path=path,
            kind=kind,
            capabilities=list(capabilities),
            fingerprint=list(fingerprint) if fingerprint is not None else None,
            profile=dict(profile or {}),
            cost=float(cost),
        )
        self._entries.append(entry)
        self._write_index()
        return entry

    def load(self, entry_id: str) -> TaskModel | CalibratedTaskModel:
        """Reload a registered model by ``entry_id`` (round-trips through the artifact on disk)."""
        entry = self._get(entry_id)
        cls = CalibratedTaskModel if entry.kind == "calibrated" else TaskModel
        return cls.load(entry.path)

    def _get(self, entry_id: str) -> RegistryEntry:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        raise KeyError(f"no registry entry {entry_id!r}")

    def find_for(self, query: str | Sequence[float], *, top_k: int | None = None) -> list[RegistryEntry]:
        """Entries matching ``query``: a capability name (``str``, containment match) or a task fingerprint
        vector (array-like of floats, nearest-neighbor match). ``top_k`` caps how many are returned -- every
        capability match by default, or the single nearest fingerprint match by default."""
        if isinstance(query, str):
            matches = [e for e in self._entries if query in e.capabilities]
            return matches[:top_k] if top_k is not None else matches
        q: np.ndarray = np.asarray(query, dtype=np.float64)
        scored = sorted(
            (
                (float(np.linalg.norm(np.asarray(e.fingerprint, dtype=np.float64) - q)), e)
                for e in self._entries
                if e.fingerprint is not None
            ),
            key=lambda t: t[0],
        )
        k = top_k if top_k is not None else 1
        return [e for _, e in scored[:k]]

    def tier_stack(
        self,
        task: str,
        *,
        frontier: Any,
        costs: Sequence[float] | None = None,
        names: Sequence[str] | None = None,
    ) -> list[tuple[str, Any, float]]:
        """Ascending-cost ``(name, model, cost)`` tiers for capability ``task``, ``frontier`` appended last.

        Matching entries are loaded (:meth:`load`) and ordered by their registered ``cost``. The result is
        exactly the shape :class:`~mixle.task.router.Router` takes as ``tiers=``: each non-final tier exposes
        ``decide(x)``, the final tier is the callable ``frontier`` fallback. ``costs`` (one entry per matching
        solution plus one for ``frontier``, mirroring :meth:`~mixle.task.router.Router.from_solutions`) overrides
        the registered per-entry costs when given.
        """
        pool = sorted(self.find_for(task), key=lambda e: e.cost)
        if costs is not None and len(costs) != len(pool) + 1:
            raise ValueError("costs needs one entry per matching solution plus one for the frontier")
        tier_costs = [float(c) for c in costs] if costs is not None else [e.cost for e in pool] + [1.0]
        tier_names = list(names) if names is not None else [e.entry_id for e in pool] + ["frontier"]
        tiers = [(tier_names[i], self.load(e.entry_id), tier_costs[i]) for i, e in enumerate(pool)]
        tiers.append((tier_names[-1], frontier, tier_costs[-1]))
        return tiers
