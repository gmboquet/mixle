"""A versioned model registry: store fitted models + their provenance, list versions, promote/swap.

A filesystem-backed store so a production system can register every fitted model (with its
:class:`~mixle.inference.production.provenance.Header`), list and load any version, and promote a chosen version
to an alias (e.g. ``"production"``) -- the swap point a serving layer reads from. Models serialize through
``mixle.utils.serialization`` (the safe registry-keyed JSON); headers are plain JSON dicts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable, to_serializable


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_segment(seg: str, kind: str = "name") -> str:
    """Reject a model name / version / alias that is not a single path component under the registry root.

    The registry is a filesystem store that may be fed names/aliases from an API (e.g. via
    ``Service.from_registry``); joining a raw ``../escape`` (or an absolute path, or one with separators)
    onto the root would read or write outside it. Constrain each segment to a plain basename."""
    if not isinstance(seg, str) or not seg:
        raise ValueError(f"registry {kind} must be a non-empty string, got {seg!r}")
    if (
        seg in (os.curdir, os.pardir)
        or os.sep in seg
        or (os.altsep and os.altsep in seg)
        or "\x00" in seg
        or os.path.isabs(seg)
        or os.path.basename(seg) != seg
    ):
        raise ValueError(f"unsafe registry {kind} {seg!r}: must be a single path component (no separators or '..')")
    return seg


class Registry:
    """A directory of named models, each with numbered versions and movable aliases."""

    def __init__(self, root: str) -> None:
        ensure_pysp_serialization_registry()
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _model_dir(self, name: str, *, create: bool) -> str:
        """Resolve the on-disk directory for model ``name``, refusing any path that escapes the store root.

        ``_safe_segment`` blocks traversal *inside* the name string, but a symlink pre-placed in the root
        (from an untrusted or restored registry, or a tar extraction) named like a model would still let a
        read or write follow it outside the root. Reject an entry that is a symlink, or whose real path is
        not contained in the root's real path, before any ``open`` / ``makedirs`` follows it.
        """
        d = os.path.join(self.root, _safe_segment(name))
        if os.path.lexists(d):
            root_real = os.path.realpath(self.root)
            real = os.path.realpath(d)
            if os.path.islink(d) or (real != root_real and not real.startswith(root_real + os.sep)):
                raise ValueError(f"unsafe registry name {name!r}: entry resolves outside the store root")
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    def _dir(self, name: str) -> str:
        return self._model_dir(name, create=True)

    def names(self) -> list[str]:
        """Registered model names."""
        return sorted(n for n in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, n)))

    def versions(self, name: str) -> list[str]:
        """Version ids for ``name`` in registration order (``v1``, ``v2``, ...)."""
        d = self._model_dir(name, create=False)
        if not os.path.isdir(d):
            return []
        vs = [f[:-5] for f in os.listdir(d) if f.endswith(".json")]
        return sorted(vs, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)

    def register(self, model: Any, name: str, *, header: Any = None, metadata: dict | None = None) -> str:
        """Store ``model`` under ``name`` as a new version; return its version id.

        ``header`` defaults to ``model.header`` if present. The model is serialized with the safe mixle
        registry; the header (a :class:`Header` or dict) and ``metadata`` are stored alongside."""
        d = self._dir(name)
        # the NEXT version number, not the current COUNT: a deleted version (v2 removed from v1,v2,v3)
        # must not free up its number for reuse, or the next register() overwrites the surviving v3.
        existing = self.versions(name)
        next_n = max((int(v[1:]) for v in existing if v[1:].isdigit()), default=0) + 1
        ver = f"v{next_n}"
        attached = getattr(model, "header", None)
        if header is None:
            header = attached
        hdr = header.to_dict() if hasattr(header, "to_dict") else header
        # the header is stored separately; detach it so it is not serialized as part of the model state
        # (a Header is not a registered serializable class).
        had_attr = hasattr(model, "__dict__") and "header" in vars(model)
        if had_attr:
            del model.header
        try:
            model_ser = to_serializable(model)
        finally:
            if had_attr:
                model.header = attached
        payload = {
            "version": ver,
            "registered_at": _now(),
            "model": model_ser,
            "header": hdr,
            "metadata": metadata or {},
        }
        with open(os.path.join(d, ver + ".json"), "w") as f:
            json.dump(payload, f)
        return ver

    def checkpointer(self, name: str, *, every: int = 1) -> Callable[[Any], None]:
        """Return an ``optimize(on_step=...)`` callback that snapshots the model under ``name`` every
        ``every`` iterations (recording the iteration + log-density in the version metadata).

        Each checkpoint records its model ``model_hash`` and the previous checkpoint's ``parent_hash``,
        so the saved snapshots form a verifiable chain (see :meth:`verify_chain`). Resume an interrupted
        run from the latest checkpoint::

            reg = Registry("ckpts")
            optimize(data, est, on_step=reg.checkpointer("run", every=5))
            model, _ = reg.get("run")              # latest checkpoint
            optimize(data, est, prev_estimate=model)   # continue training
        """
        from mixle.data.hashing import model_hash

        parent: str | None = None

        def _save(step: Any) -> None:
            nonlocal parent
            if every <= 1 or step.iter % every == 0:
                h = model_hash(step.model)
                self.register(
                    step.model,
                    name,
                    metadata={
                        "checkpoint_iter": step.iter,
                        "log_density": step.log_density,
                        "model_hash": h,
                        "parent_hash": parent,
                    },
                )
                parent = h

        return _save

    def _resolve_version(self, name: str, version: str) -> str:
        """Resolve ``"latest"`` to the highest version and raise a clear KeyError for an unknown name or
        version -- rather than a bare IndexError on an unregistered name or a raw FileNotFoundError (which
        leaks the store path) on a missing version. Mirrors the guard get() already had."""
        _safe_segment(name)
        vs = self.versions(name)
        if not vs:
            raise KeyError(f"no versions registered for model {name!r}")
        if version == "latest":
            return vs[-1]
        if version not in vs:  # returned value is therefore always a known-safe version id
            raise KeyError(f"{name!r} has no version {version!r}")
        return version

    def get(self, name: str, version: str = "latest", *, trust_code: bool = False) -> tuple[Any, dict | None]:
        """Load ``(model, header)`` for a version (``"latest"`` = highest-numbered).

        A registered model containing a NeuralLeaf-family component embeds its weights as a pickle
        blob (see :mod:`mixle.models._neural_serial`); deserializing that executes code, so ``get``
        requires ``trust_code=True`` for such an entry -- trust the registry root, not just the JSON
        extension. A pure-statistical entry loads either way.
        """
        version = self._resolve_version(name, version)
        with open(os.path.join(self._model_dir(name, create=False), version + ".json")) as f:
            payload = json.load(f)
        if trust_code:
            from mixle.utils.serialization import trusted_deserialization

            with trusted_deserialization():
                return from_serializable(payload["model"]), payload.get("header")
        return from_serializable(payload["model"]), payload.get("header")

    def header(self, name: str, version: str = "latest") -> dict | None:
        """Just the provenance header of a version (no model deserialization)."""
        version = self._resolve_version(name, version)
        with open(os.path.join(self._model_dir(name, create=False), version + ".json")) as f:
            return json.load(f).get("header")

    def metadata(self, name: str, version: str = "latest") -> dict:
        """Just the ``metadata`` of a version (no model deserialization) -- e.g. a checkpoint's iteration."""
        version = self._resolve_version(name, version)
        with open(os.path.join(self._model_dir(name, create=False), version + ".json")) as f:
            return json.load(f).get("metadata") or {}

    def promote(self, name: str, version: str, alias: str = "production") -> None:
        """Point ``alias`` (e.g. ``"production"``) at ``version`` -- the atomic model swap.

        Written via a temp file + ``os.replace`` in the same directory (same filesystem, so the
        rename is atomic): a concurrent reader of :meth:`current` either sees the old alias target or
        the new one, never a truncated/partial write, and a crash mid-write leaves the old alias
        file untouched rather than corrupted.
        """
        if version not in self.versions(name):
            raise KeyError(f"{name!r} has no version {version!r}")
        d = self._dir(name)
        target = os.path.join(d, _safe_segment(alias, "alias") + ".alias")
        tmp = os.path.join(d, f".{_safe_segment(alias, 'alias')}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            f.write(version)
        os.replace(tmp, target)

    def current(self, name: str, alias: str = "production", *, trust_code: bool = False) -> tuple[Any, dict | None]:
        """Load the model an ``alias`` points at (falls back to ``latest`` if the alias is unset).

        See :meth:`get` -- ``trust_code`` is required in the same way and for the same reason.
        """
        p = os.path.join(self._model_dir(name, create=False), _safe_segment(alias, "alias") + ".alias")
        # the version READ FROM the alias file is still resolved against the known version list by get(),
        # so a tampered alias file cannot traverse either.
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                version = f.read().strip()
        else:
            version = "latest"
        return self.get(name, version, trust_code=trust_code)

    def verify_chain(self, name: str) -> bool:
        """Verify the persisted checkpoint lineage for ``name`` (see :meth:`checkpointer`).

        For each version carrying a ``model_hash``, checks that its ``parent_hash`` matches the previous
        such version's hash *and* that re-hashing the loaded model reproduces the stored hash (catching
        corruption or tampering). Returns True when every link holds, or vacuously when no version carries
        lineage metadata."""
        from mixle.data.hashing import model_hash

        prev: str | None = None
        for ver in self.versions(name):
            stored = self.metadata(name, ver).get("model_hash")
            if stored is None:
                continue
            if self.metadata(name, ver).get("parent_hash") != prev:
                return False
            model, _ = self.get(name, ver)
            if model_hash(model) != stored:
                return False
            prev = stored
        return True
