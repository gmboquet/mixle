"""``Registry`` -- a local directory + index of fitted task models, queryable by capability and fingerprint.

The registry is the local library catalog that orchestrators, routers, capture
flows, and accumulation workflows can read from or write into. Deliberately a
directory plus a JSON index, not a server: every entry is a saved
:class:`~mixle.task.model.TaskModel`, :class:`~mixle.task.calibrate.CalibratedTaskModel`, or an IC-1
``Posterior``-conforming field-posterior artifact directory (see :mod:`mixle.task.artifact` for the
first two; the third is written/read through ``mixle_pde.io.artifacts`` -- IC-2 -- lazily imported so
this module never hard-depends on the ``mixle_pde`` plugin). A small index record names each entry's
capabilities, task fingerprint (:func:`~mixle.task.edge.task_fingerprint`), and capture profile.
``find_for`` answers "do I already have something for this task"; ``tier_stack`` turns a matching
capability into an ascending-cost tier list -- the shape :class:`~mixle.task.router.Router` consumes
directly (``Router(tiers=stack)``), with the frontier appended last as the router's own fallback tier.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.model import TaskModel

__all__ = ["Registry", "RegistryEntry"]

_INDEX_NAME = "index.json"
_LOCK_NAME = ".registry.lock"


def _safe_entry_id(entry_id: str) -> str:
    """Reject an ``entry_id`` that is not a single, safe path component; return it unchanged otherwise.

    ``register`` joins ``entry_id`` onto :attr:`Registry.dir` to build the artifact path it writes to and
    later reads back from (``os.path.join(self.dir, entry_id)``). An unvalidated caller-supplied id
    containing path separators or a ``..`` component -- ``"../escaped"``, ``"../../etc/cron.d/x"``, an
    absolute path -- would write OUTSIDE the registry root instead of inside it: a path-traversal write,
    not merely a naming quirk. Blocking only the literal substring ``".."`` is not enough (a bare
    OS-specific separator, or an id that is already an absolute path, escapes without ever containing
    that substring), so this instead requires the id to equal its own ``os.path.basename`` -- true only
    for a plain single path component -- mirroring
    :func:`mixle.inference.production.registry._safe_segment`, the same guard that sibling registry uses
    for its ``name``/``version``/``alias`` segments.
    """
    if (
        not isinstance(entry_id, str)
        or not entry_id
        or entry_id in (os.curdir, os.pardir)
        or os.sep in entry_id
        or (os.altsep and os.altsep in entry_id)
        or "\x00" in entry_id
        or os.path.isabs(entry_id)
        or os.path.basename(entry_id) != entry_id
    ):
        raise ValueError(f"unsafe entry_id {entry_id!r}: must be a single path component (no separators or '..')")
    return entry_id


def _is_field_posterior(model: Any) -> bool:
    """True when ``model`` is a ``mixle_pde`` field posterior (the "field_posterior" `Registry` kind, IC-2).

    Checks the concrete ``mixle_pde.latent.PosteriorField3D`` type rather than the abstract IC-1
    ``Posterior`` protocol so this keeps working whether or not the field posterior has picked up the
    ``samples``-method rename the protocol freezes -- this module only cares which artifact I/O to call,
    not full protocol conformance. Returns ``False`` (never raises) when ``mixle_pde`` isn't installed, so
    a bare `mixle` checkout never fails ``isinstance``/``TypeError`` dispatch for a class it can't see.
    """
    try:
        from mixle_pde.latent import PosteriorField3D
    except ImportError:
        return False
    return isinstance(model, PosteriorField3D)


def _save_field_posterior(model: Any, path: str) -> None:
    """Delegate to ``mixle_pde.io.artifacts.save_posterior`` (IC-2), imported lazily.

    ``mixle_pde`` depends on ``mixle``, never the reverse (see ``mixle_pde``'s package docstring), so this
    module never imports it at module scope -- only here, when a caller actually registers a field
    posterior.
    """
    try:
        from mixle_pde.io.artifacts import save_posterior
    except ImportError as exc:
        raise ImportError(
            "registering a field_posterior kind requires the mixle_pde package "
            "(install mixle_pde, or add its checkout to PYTHONPATH)"
        ) from exc
    save_posterior(model, path)


def _load_field_posterior(path: str) -> Any:
    """Delegate to ``mixle_pde.io.artifacts.load_posterior`` (IC-2), imported lazily (see `_save_field_posterior`)."""
    try:
        from mixle_pde.io.artifacts import load_posterior
    except ImportError as exc:
        raise ImportError(
            "loading a field_posterior kind requires the mixle_pde package "
            "(install mixle_pde, or add its checkout to PYTHONPATH)"
        ) from exc
    return load_posterior(path)


@dataclass
class RegistryEntry:
    """One catalog record: where the artifact lives, what it's registered under, and how much it costs to run."""

    entry_id: str
    path: str
    kind: str  # "task", "calibrated", or "field_posterior" -- which loader reloads the artifact at ``path``
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
        """Persist ``self._entries`` to the index file, atomically (temp file + ``os.replace``).

        A plain ``open(path, "w")`` truncates the index before the new content is written, so a crash
        (or a concurrent reader constructing a fresh ``Registry`` against this same ``dir``) mid-write
        could observe a truncated, unparseable ``index.json`` instead of the old or the new content.
        Mirrors :func:`mixle.task.artifact._atomic_json_dump`'s temp-file-plus-``os.replace`` pattern: the
        write is all-or-nothing, so a reader always sees either the pre- or the post-write index, never a
        torn one.
        """
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp-index-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._index_path())
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def register(
        self,
        model: TaskModel | CalibratedTaskModel | Any,
        *,
        capabilities: Sequence[str],
        fingerprint: Sequence[float] | None = None,
        profile: dict[str, Any] | None = None,
        cost: float = 0.0,
        entry_id: str | None = None,
    ) -> RegistryEntry:
        """Save ``model``'s artifact under ``dir`` and add its index entry; return the entry.

        ``model`` is a fitted :class:`~mixle.task.model.TaskModel`, a
        :class:`~mixle.task.calibrate.CalibratedTaskModel`, or a ``mixle_pde`` field posterior (a
        "field_posterior" kind, IC-2) -- the three artifact-saveable model kinds. ``capabilities`` names
        what this model answers (matched by :meth:`find_for`); ``fingerprint`` is typically
        :func:`~mixle.task.edge.task_fingerprint`'s vector for the training data; ``profile`` is
        free-form (e.g. a :func:`~mixle.task.capability.capture_profile` dict); ``cost`` is the per-request
        cost used to order :meth:`tier_stack`. An explicit ``entry_id`` that already exists (in the index
        or as an artifact directory) raises rather than duplicating the index row and silently
        overwriting the artifact; auto-generated ids scan past taken ones.

        ``entry_id`` must be a single safe path component (:func:`_safe_entry_id`): an unvalidated value
        such as ``"../escaped"`` would otherwise write the artifact outside ``dir`` instead of inside it.

        Allocating an id and persisting the index is a read-modify-write over the on-disk index: two
        ``Registry`` instances opened on the same ``dir`` (or two threads/processes sharing one) each
        cache their own ``self._entries`` snapshot from construction time and otherwise never refresh it,
        so one instance's ``register`` can silently overwrite another's already-persisted index row when
        it writes back its own stale-plus-one view -- even when the two calls pick different,
        non-colliding entry ids. Fixed by serializing id-allocation-through-index-write with an
        ``fcntl.flock`` on a lock file at the registry root, re-reading the on-disk index fresh under the
        lock rather than trusting the cached snapshot. An independent claim-file guard
        (``O_CREAT | O_EXCL``) additionally protects the artifact write itself, belt-and-suspenders
        alongside the lock, mirroring
        :meth:`mixle.inference.production.registry.Registry.register`'s fix for the same class of bug.
        """
        if isinstance(model, CalibratedTaskModel):
            kind = "calibrated"
        elif isinstance(model, TaskModel):
            kind = "task"
        elif _is_field_posterior(model):
            kind = "field_posterior"
        else:
            raise TypeError(
                f"Registry only stores TaskModel/CalibratedTaskModel/field-posterior artifacts, got {type(model)!r}"
            )
        if entry_id is not None:
            _safe_entry_id(entry_id)

        lock_path = os.path.join(self.dir, _LOCK_NAME)
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # a concurrent writer (another Registry instance, or another thread/process sharing this
                # one) may have persisted entries since this instance last read the index -- re-read it
                # fresh under the lock rather than trusting self._entries, which is otherwise populated
                # once at __init__ and never refreshed.
                self._entries = self._read_index()
                taken = {e.entry_id for e in self._entries}
                if entry_id is not None:
                    if entry_id in taken or os.path.exists(os.path.join(self.dir, entry_id)):
                        raise ValueError(f"registry already has an entry {entry_id!r}; entry ids must be unique")
                else:
                    # a len()-based id collides after manual index edits, explicit ids, or another
                    # writer's artifacts -- scan forward until the id is free in BOTH the index and the
                    # directory
                    i = len(self._entries)
                    while f"entry_{i:04d}" in taken or os.path.exists(os.path.join(self.dir, f"entry_{i:04d}")):
                        i += 1
                    entry_id = f"entry_{i:04d}"
                path = os.path.join(self.dir, entry_id)
                root_real = os.path.realpath(self.dir)
                path_real = os.path.realpath(path)
                if path_real != root_real and not path_real.startswith(root_real + os.sep):
                    raise ValueError(f"unsafe entry_id {entry_id!r}: resolves outside the registry root")
                # independent conflict-detection guard, belt-and-suspenders alongside the lock: claim
                # entry_id with an atomically-created marker before writing its artifact, so even a stale
                # read that somehow slips past the lock (e.g. a filesystem where flock does not actually
                # exclude) raises instead of silently sharing or overwriting another writer's artifact. A
                # dedicated marker rather than O_CREAT | O_EXCL directly on `path` itself, because `path`
                # is a directory for "task"/"calibrated" kinds but a bare file-prefix
                # (``path + ".npz"``/``path + ".json"``) for "field_posterior" -- the marker is the one
                # thing every kind can claim identically.
                claim_path = os.path.join(self.dir, f".{entry_id}.claim")
                try:
                    claim_fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    raise RuntimeError(
                        f"registry conflict: entry {entry_id!r} already exists -- another writer claimed "
                        "it concurrently; retry register()"
                    ) from None
                os.close(claim_fd)
                try:
                    if kind == "field_posterior":
                        _save_field_posterior(model, path)
                    else:
                        model.save(path)
                except BaseException:
                    # the claim succeeded but the artifact write did not -- drop the claim so a retry
                    # with the same entry_id is not spuriously rejected as "already claimed" by a failed
                    # attempt's leftover marker.
                    try:
                        os.unlink(claim_path)
                    except OSError:
                        pass
                    raise
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
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        return entry

    def load(self, entry_id: str) -> TaskModel | CalibratedTaskModel | Any:
        """Reload a registered model by ``entry_id`` (round-trips through the artifact on disk)."""
        entry = self._get(entry_id)
        if entry.kind == "field_posterior":
            return _load_field_posterior(entry.path)
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

        Matching entries are loaded (:meth:`load`) and ordered by their *effective* cost -- the ``costs``
        override when given, else the registered per-entry cost. The result is exactly the shape
        :class:`~mixle.task.router.Router` takes as ``tiers=``: each non-final tier exposes ``decide(x)``,
        the final tier is the callable ``frontier`` fallback. ``costs`` (one entry per matching solution in
        registered-cost order, plus one for ``frontier``, mirroring
        :meth:`~mixle.task.router.Router.from_solutions`) overrides the registered per-entry costs when given.
        """
        pool = sorted(self.find_for(task), key=lambda e: e.cost)
        if costs is not None and len(costs) != len(pool) + 1:
            raise ValueError("costs needs one entry per matching solution plus one for the frontier")
        tier_costs = [float(c) for c in costs] if costs is not None else [e.cost for e in pool] + [1.0]
        tier_names = list(names) if names is not None else [e.entry_id for e in pool] + ["frontier"]
        tiers = [(tier_names[i], self.load(e.entry_id), tier_costs[i]) for i, e in enumerate(pool)]
        tiers.sort(key=lambda t: t[2])  # a costs= override can reorder the pool; Router assumes ascending tiers
        tiers.append((tier_names[-1], frontier, tier_costs[-1]))
        return tiers
