"""Durable, portable artifacts for small task models -- the contract mixle's JSON serialization can't carry.

``mixle.utils.serialization`` round-trips pure probabilistic models as registry-keyed JSON, but a task model
is usually torch-backed (a distilled tiny transformer, an MLP head), and its parameters are *weights*, not a
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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1"
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.safetensors"
JSON_MODEL_NAME = "model.json"


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
        d = {
            "artifact_type": "mixle.task",
            "schema_version": self.schema_version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "payload": self.payload,
            "task": self.task,
            "io": self.io,
            "meta": self.meta,
        }
        if self.payload == "torch":
            d["builder"] = self.builder
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskManifest:
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
    """Read just the manifest of an artifact directory (cheap: no weights loaded)."""
    with open(os.path.join(path, MANIFEST_NAME)) as f:
        return TaskManifest.from_dict(json.load(f))


def _write_manifest(path: str, manifest: TaskManifest) -> None:
    with open(os.path.join(path, MANIFEST_NAME), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


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
    with open(os.path.join(path, JSON_MODEL_NAME), "w") as f:
        json.dump(to_serializable(model), f)
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
