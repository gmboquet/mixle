"""Durable, portable artifacts for local task models.

``mixle.utils.serialization`` round-trips pure probabilistic models as registry-keyed JSON, but a task model
is usually torch-backed (a distilled Transformer, an MLP head), and its parameters are *weights*, not a
JSON-serializable state. Worse, the causal LM ties ``head.weight = tok.weight``; a naive tensor dump rejects the
shared storage. This module is the missing piece: a self-describing **directory** that pairs

  * ``manifest.json`` -- how to *rebuild* the module (a registered builder name + its config) plus task I/O and
    free-form metadata, and
  * ``weights.safetensors`` -- the parameters, written through ``safetensors.torch.save_model`` so tied weights
    survive,

so a fitted model survives the process that made it. ``save_module``/``load_module`` are the torch path;
``save_json``/``load_json`` are the fallback for a pure mixle distribution. A builder is any
``(**config) -> nn.Module`` callable registered by name (``register_builder``); the two native architectures
(``mixle.causal_lm``, ``mixle.mlp``) self-register on first use, and a caller can register its own.

The acceptance bar is a fresh-process round trip: save here, load in a new interpreter from the manifest alone,
get bit-identical outputs. ``mixle.task.model.TaskModel`` builds the callable task surface on top of this.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1"
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.safetensors"
JSON_MODEL_NAME = "model.json"
ARRAYS_NAME = "arrays.npz"


def _atomic_json_dump(dst: str, obj: Any, **dump_kwargs: Any) -> None:
    """Serialize ``obj`` as JSON to ``dst`` atomically: write a sibling temp file, fsync, then ``os.replace``.

    A plain ``open(dst, "w")`` truncates ``dst`` *before* serialization runs, so a non-serializable model (or
    a crash mid-``json.dump``) leaves a truncated, unloadable artifact -- or destroys the previous good one.
    Writing to a temp file in the same directory and swapping it in with ``os.replace`` (atomic on POSIX and
    Windows) makes the write all-or-nothing: on any failure the temp file is removed and ``dst`` is untouched.
    """
    directory = os.path.dirname(dst) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-artifact-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    except BaseException:
        # Serialization failed or was interrupted: drop the temp file, leave dst as it was.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- builder registry: name -> (**config) -> nn.Module ------------------------------------------------------

_BUILDERS: dict[str, Callable[..., Any]] = {}


def register_builder(name: str, builder: Callable[..., Any]) -> None:
    """Register ``builder`` under ``name`` so an artifact carrying ``builder=name`` can reconstruct its module.

    ``builder(**config)`` must return a fresh (untrained) ``nn.Module`` whose parameter shapes match the saved
    weights. Re-registering the same name with the same callable is a no-op; a conflicting one raises.
    """
    existing = _BUILDERS.get(name)
    if existing is not None and existing is not builder:
        raise ValueError(f"builder {name!r} already registered to a different callable")
    _BUILDERS[name] = builder


def get_builder(name: str) -> Callable[..., Any]:
    """Look up a registered builder, triggering native-builder self-registration on first call."""
    if name not in _BUILDERS:
        _register_native_builders()
    if name not in _BUILDERS:
        raise KeyError(f"no builder registered as {name!r}; call register_builder({name!r}, ...) first")
    return _BUILDERS[name]


def _register_native_builders() -> None:
    """Self-register mixle's own architectures (lazy: avoids importing torch at module import time)."""
    if "mixle.causal_lm" not in _BUILDERS:
        from mixle.models.transformer import build_causal_lm

        register_builder("mixle.causal_lm", build_causal_lm)
    if "mixle.mlp" not in _BUILDERS:
        from mixle.models.neural import make_mlp

        register_builder("mixle.mlp", make_mlp)
    if "mixle.seq_tagger" not in _BUILDERS:
        from mixle.task.extract import build_seq_tagger

        register_builder("mixle.seq_tagger", build_seq_tagger)


# --- manifest ------------------------------------------------------------------------------------------------


@dataclass
class TaskManifest:
    """The self-describing header of a task artifact: enough to rebuild and call the model, plus provenance."""

    payload: str  # "torch" (weights.safetensors + builder/config) or "json" (model.json)
    builder: str | None = None  # registered builder name (torch payload)
    config: dict[str, Any] = field(default_factory=dict)  # builder kwargs (torch payload)
    task: str = ""  # one-line description of what this model does
    io: dict[str, Any] = field(default_factory=dict)  # how raw input/output map to the model (TaskModel uses this)
    meta: dict[str, Any] = field(default_factory=dict)  # free-form provenance (teacher, data hash, eval, ...)
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the strict-JSON manifest representation written to ``manifest.json``."""
        d = {
            "artifact_type": "mixle.task",
            "schema_version": self.schema_version,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "payload": self.payload,
            "task": self.task,
            "io": self.io,
            "meta": self.meta,
        }
        if self.payload in ("torch", "arrays"):  # payloads reconstructed through a registered builder
            d["builder"] = self.builder
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskManifest:
        """Parse a manifest dictionary into a :class:`TaskManifest`."""
        return cls(
            payload=d["payload"],
            builder=d.get("builder"),
            config=d.get("config", {}),
            task=d.get("task", ""),
            io=d.get("io", {}),
            meta=d.get("meta", {}),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            created_at=d.get("created_at", ""),
        )


def read_manifest(path: str) -> TaskManifest:
    """Read only the manifest of an artifact directory without loading weights."""
    with open(os.path.join(path, MANIFEST_NAME)) as f:
        return TaskManifest.from_dict(json.load(f))


def _write_manifest(path: str, manifest: TaskManifest) -> None:
    _atomic_json_dump(os.path.join(path, MANIFEST_NAME), manifest.to_dict(), indent=2, sort_keys=True)


# --- torch payload: builder + config + tied-safe weights ----------------------------------------------------


def save_module(
    path: str,
    module: Any,
    builder: str,
    config: dict[str, Any],
    *,
    task: str = "",
    io: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Persist a torch ``module`` as an artifact directory and return ``path``.

    ``builder``/``config`` must reconstruct an architecturally identical module (``get_builder(builder)(**config)``);
    weights go through ``safetensors.torch.save_model`` so tied parameters (e.g. the LM's tied head) round-trip.
    """
    from safetensors.torch import save_model

    os.makedirs(path, exist_ok=True)
    get_builder(builder)  # fail fast if the builder is unknown -- before writing anything
    save_model(module, os.path.join(path, WEIGHTS_NAME))
    _write_manifest(
        path,
        TaskManifest(payload="torch", builder=builder, config=dict(config), task=task, io=io or {}, meta=meta or {}),
    )
    return path


def load_module(path: str, *, device: str = "cpu") -> tuple[Any, TaskManifest]:
    """Rebuild a torch module from its manifest alone and load weights; return ``(module, manifest)``."""
    from safetensors.torch import load_model

    manifest = read_manifest(path)
    if manifest.payload != "torch":
        raise ValueError(f"artifact at {path!r} is a {manifest.payload!r} payload, not torch")
    module = get_builder(manifest.builder)(**manifest.config)
    load_model(module, os.path.join(path, WEIGHTS_NAME), device=device)
    return module, manifest


# --- json payload: a pure mixle distribution ----------------------------------------------------------------


def save_json(
    path: str,
    model: Any,
    *,
    task: str = "",
    io: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Persist a pure (torch-free) mixle distribution via the safe serialization registry; return ``path``."""
    from mixle.utils.serialization import ensure_pysp_serialization_registry, to_serializable

    ensure_pysp_serialization_registry()
    os.makedirs(path, exist_ok=True)
    _atomic_json_dump(os.path.join(path, JSON_MODEL_NAME), to_serializable(model))
    _write_manifest(path, TaskManifest(payload="json", task=task, io=io or {}, meta=meta or {}))
    return path


def load_json(path: str) -> tuple[Any, TaskManifest]:
    """Rebuild a pure mixle distribution from a json-payload artifact; return ``(model, manifest)``."""
    from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable

    manifest = read_manifest(path)
    if manifest.payload != "json":
        raise ValueError(f"artifact at {path!r} is a {manifest.payload!r} payload, not json")
    ensure_pysp_serialization_registry()
    with open(os.path.join(path, JSON_MODEL_NAME)) as f:
        return from_serializable(json.load(f)), manifest


# --- arrays payload: a dict of numpy arrays + a registered reconstructor (torch-free students) ---------------

_ARRAYS_BUILDERS: dict[str, Callable[..., Any]] = {}


def register_arrays_builder(name: str, builder: Callable[..., Any]) -> None:
    """Register ``builder(arrays: dict[str, ndarray], **config) -> model`` for arrays-payload artifacts.

    The arrays payload is for torch-free numeric students (e.g. an int8-quantized MLP): weights live in
    one ``.npz``, and the builder reconstructs the runnable model from them in a fresh process.
    """
    existing = _ARRAYS_BUILDERS.get(name)
    if existing is not None and existing is not builder:
        raise ValueError(f"arrays builder {name!r} is already registered to a different callable")
    _ARRAYS_BUILDERS[name] = builder


def get_arrays_builder(name: str | None) -> Callable[..., Any]:
    """Look up a registered arrays builder, triggering native self-registration on first call."""
    if name is None:
        raise KeyError("arrays artifact has no builder recorded; it cannot be reconstructed")
    if name not in _ARRAYS_BUILDERS and name.startswith("mixle."):
        import mixle.task.quantize  # noqa: F401  (registers mixle.quantized_mlp)
    if name not in _ARRAYS_BUILDERS:
        raise KeyError(f"no arrays builder registered as {name!r}; call register_arrays_builder({name!r}, ...) first")
    return _ARRAYS_BUILDERS[name]


def save_arrays(
    path: str,
    arrays: dict[str, Any],
    builder: str,
    config: dict[str, Any] | None = None,
    *,
    task: str = "",
    io: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Persist a dict of numpy arrays as an artifact directory (``arrays.npz``); return ``path``."""
    import numpy as np

    get_arrays_builder(builder)  # fail fast before writing anything
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, ARRAYS_NAME), **arrays)
    _write_manifest(
        path,
        TaskManifest(
            payload="arrays", builder=builder, config=dict(config or {}), task=task, io=io or {}, meta=meta or {}
        ),
    )
    return path


def load_arrays(path: str) -> tuple[Any, TaskManifest]:
    """Rebuild a torch-free model from an arrays-payload artifact; return ``(model, manifest)``."""
    import numpy as np

    manifest = read_manifest(path)
    if manifest.payload != "arrays":
        raise ValueError(f"artifact at {path!r} is a {manifest.payload!r} payload, not arrays")
    with np.load(os.path.join(path, ARRAYS_NAME)) as z:
        arrays = {k: z[k] for k in z.files}
    return get_arrays_builder(manifest.builder)(arrays, **manifest.config), manifest
