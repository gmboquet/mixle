"""A versioned model registry: store fitted models + their provenance, list versions, promote/swap.

A filesystem-backed store so a production system can register every fitted model (with its
:class:`~pysp.inference.provenance.ModelHeader`), list and load any version, and promote a chosen version
to an alias (e.g. ``"production"``) -- the swap point a serving layer reads from. Models serialize through
``pysp.utils.serialization`` (the safe registry-keyed JSON); headers are plain JSON dicts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pysp.utils.serialization import ensure_pysp_serialization_registry, from_serializable, to_serializable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """A directory of named models, each with numbered versions and movable aliases."""

    def __init__(self, root: str) -> None:
        ensure_pysp_serialization_registry()
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _dir(self, name: str) -> str:
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        return d

    def names(self) -> list[str]:
        """Registered model names."""
        return sorted(n for n in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, n)))

    def versions(self, name: str) -> list[str]:
        """Version ids for ``name`` in registration order (``v1``, ``v2``, ...)."""
        d = os.path.join(self.root, name)
        if not os.path.isdir(d):
            return []
        vs = [f[:-5] for f in os.listdir(d) if f.endswith(".json")]
        return sorted(vs, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)

    def register(self, model: Any, name: str, *, header: Any = None, metadata: dict | None = None) -> str:
        """Store ``model`` under ``name`` as a new version; return its version id.

        ``header`` defaults to ``model.header`` if present. The model is serialized with the safe pysp
        registry; the header (a :class:`ModelHeader` or dict) and ``metadata`` are stored alongside."""
        d = self._dir(name)
        ver = f"v{len(self.versions(name)) + 1}"
        attached = getattr(model, "header", None)
        if header is None:
            header = attached
        hdr = header.to_dict() if hasattr(header, "to_dict") else header
        # the header is stored separately; detach it so it is not serialized as part of the model state
        # (a ModelHeader is not a registered serializable class).
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

            reg = ModelRegistry("ckpts")
            optimize(data, est, on_step=reg.checkpointer("run", every=5))
            model, _ = reg.get("run")              # latest checkpoint
            optimize(data, est, prev_estimate=model)   # continue training
        """
        from pysp.data.hashing import model_hash

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

    def get(self, name: str, version: str = "latest") -> tuple[Any, dict | None]:
        """Load ``(model, header)`` for a version (``"latest"`` = highest-numbered)."""
        vs = self.versions(name)
        if not vs:
            raise KeyError(f"no versions registered for model {name!r}")
        if version == "latest":
            version = vs[-1]
        with open(os.path.join(self.root, name, version + ".json")) as f:
            payload = json.load(f)
        return from_serializable(payload["model"]), payload.get("header")

    def header(self, name: str, version: str = "latest") -> dict | None:
        """Just the provenance header of a version (no model deserialization)."""
        vs = self.versions(name)
        if version == "latest":
            version = vs[-1]
        with open(os.path.join(self.root, name, version + ".json")) as f:
            return json.load(f).get("header")

    def metadata(self, name: str, version: str = "latest") -> dict:
        """Just the ``metadata`` of a version (no model deserialization) -- e.g. a checkpoint's iteration."""
        vs = self.versions(name)
        if version == "latest":
            version = vs[-1]
        with open(os.path.join(self.root, name, version + ".json")) as f:
            return json.load(f).get("metadata") or {}

    def promote(self, name: str, version: str, alias: str = "production") -> None:
        """Point ``alias`` (e.g. ``"production"``) at ``version`` -- the atomic model swap."""
        if version not in self.versions(name):
            raise KeyError(f"{name!r} has no version {version!r}")
        with open(os.path.join(self._dir(name), alias + ".alias"), "w") as f:
            f.write(version)

    def current(self, name: str, alias: str = "production") -> tuple[Any, dict | None]:
        """Load the model an ``alias`` points at (falls back to ``latest`` if the alias is unset)."""
        p = os.path.join(self.root, name, alias + ".alias")
        version = open(p).read().strip() if os.path.exists(p) else "latest"
        return self.get(name, version)

    def verify_chain(self, name: str) -> bool:
        """Verify the persisted checkpoint lineage for ``name`` (see :meth:`checkpointer`).

        For each version carrying a ``model_hash``, checks that its ``parent_hash`` matches the previous
        such version's hash *and* that re-hashing the loaded model reproduces the stored hash (catching
        corruption or tampering). Returns True when every link holds, or vacuously when no version carries
        lineage metadata."""
        from pysp.data.hashing import model_hash

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
