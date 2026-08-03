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

import hashlib
import hmac
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "2"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION,)  # every schema_version this code currently knows how to read
ARTIFACT_TYPE = "mixle.task"
KNOWN_PAYLOADS = ("torch", "json", "arrays")
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.safetensors"
JSON_MODEL_NAME = "model.json"
ARRAYS_NAME = "arrays.npz"
_PAYLOAD_FILES = {
    "torch": WEIGHTS_NAME,
    "json": JSON_MODEL_NAME,
    "arrays": ARRAYS_NAME,
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_integrity(data: dict[str, Any]) -> str:
    covered = dict(data)
    covered.pop("integrity_sha256", None)
    encoded = json.dumps(
        covered,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: str) -> None:
    """Durably publish directory-entry changes where the platform supports it."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _recover_artifact(path: str) -> None:
    """Restore an interrupted replacement when the canonical path is absent."""

    target = os.path.abspath(path)
    if os.path.lexists(target):
        return
    parent = os.path.dirname(target) or "."
    prefix = f".{os.path.basename(target)}.prev-"
    try:
        candidates = [
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.startswith(prefix) and os.path.isdir(os.path.join(parent, name))
        ]
    except FileNotFoundError:
        return
    if not candidates:
        return
    # A normal replacement can create only one live backup. If an earlier process left more
    # than one, recover the most recently published generation and retain the others for diagnosis.
    selected = max(candidates, key=lambda candidate: os.stat(candidate).st_mtime_ns)
    os.replace(selected, target)
    _fsync_directory(parent)


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


def _atomic_artifact_write(path: str, write_fn: Callable[[str], None]) -> None:
    """Publish a full artifact generation (payload file(s) + manifest) to ``path`` as a single atomic step.

    ``save_module``/``save_json``/``save_arrays`` each write more than one file that must move together as a
    set -- a payload (weights/model/arrays) plus the manifest describing it. Writing them as separate in-place
    steps -- payload first, then manifest -- means a crash or exception between the two leaves ``path`` holding
    a NEW payload paired with the OLD manifest: a mismatched pairing that a subsequent ``load_*`` cannot detect
    as corrupt when the builder/config didn't change -- it silently loads the new weights next to stale
    provenance (task/io/meta) describing the run that produced the *previous* weights.

    ``write_fn(staging_dir)`` must write every file the new generation needs into a fresh sibling staging
    directory that nothing else can see; ``path`` itself is never touched while it runs, so a failure anywhere
    inside ``write_fn`` (including ``_atomic_json_dump`` itself failing) leaves whatever was already at
    ``path`` -- the previous, still-consistent artifact, or nothing at all on a first save -- completely
    untouched. Once staging holds a complete new generation, publishing it is two directory renames, each
    atomic on a POSIX filesystem (same directory, so same filesystem): the existing ``path`` (if any) is moved
    aside to a backup name, then staging is moved into ``path``. A crash between those two renames leaves
    ``path`` momentarily absent -- never a mix of old and new files -- with the old generation fully
    recoverable from its backup suffix; if the second rename itself fails, the backup is moved straight back
    so ``path`` is never left missing. The backup is removed only once the new generation is already live.
    """
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    _recover_artifact(target)
    staging = tempfile.mkdtemp(dir=parent, prefix=".tmp-artifact-")
    # mkdtemp defaults to 0o700; match the world-readable dirs os.makedirs would make. 0o755 grants no
    # group or world WRITE bit, so no other local user can inject a file into the generation being
    # staged here -- it only makes the published directory as readable as the one it replaces.
    os.chmod(staging, 0o755)  # nosec B103 # 0o755 is read/execute for others, never writable by them
    published = False
    try:
        write_fn(staging)  # if this raises, `path` has not been touched at all
        if os.path.lexists(target):
            backup = os.path.join(parent, f".{os.path.basename(target)}.prev-{uuid.uuid4().hex}")
            os.replace(target, backup)
            _fsync_directory(parent)
            try:
                os.replace(staging, target)
                _fsync_directory(parent)
            except BaseException:
                os.replace(backup, target)
                _fsync_directory(parent)
                raise
            published = True
            shutil.rmtree(backup, ignore_errors=True)  # best-effort: new generation is already live either way
        else:
            os.replace(staging, target)
            _fsync_directory(parent)
            published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


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
    payload_sha256: str = ""
    integrity_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the strict-JSON manifest representation written to ``manifest.json``."""
        d = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "payload": self.payload,
            "task": self.task,
            "io": self.io,
            "meta": self.meta,
            "payload_sha256": self.payload_sha256,
            "integrity_sha256": self.integrity_sha256,
        }
        if self.payload in ("torch", "arrays"):  # payloads reconstructed through a registered builder
            d["builder"] = self.builder
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskManifest:
        """Parse a manifest dictionary into a :class:`TaskManifest`, rejecting anything this code can't safely read.

        A ``manifest.json`` is data a fresh process trusts blindly to decide how to rebuild and load a model, so
        this validates it before trusting any field: ``artifact_type`` must mark it as a mixle task artifact
        (not some other format that happens to share a directory layout), ``schema_version`` must be one this
        code was actually written against -- an unrecognized version (e.g. from a newer or older mixle) is
        rejected outright rather than silently misread as today's schema, ``payload`` must be a payload kind
        this module knows how to load, and a payload that needs a builder (``torch``/``arrays``) must name one.
        """
        if not isinstance(d, dict):
            raise ValueError(f"task-artifact manifest must be a JSON object, got {type(d).__name__}")

        artifact_type = d.get("artifact_type")
        if artifact_type != ARTIFACT_TYPE:
            raise ValueError(
                f"not a mixle task-artifact manifest: artifact_type={artifact_type!r}, expected {ARTIFACT_TYPE!r}"
            )

        schema_version = d.get("schema_version", SCHEMA_VERSION)
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported task-artifact schema_version {schema_version!r}; this code only understands "
                f"{SUPPORTED_SCHEMA_VERSIONS!r} -- likely written by an incompatible (newer or older) mixle"
            )

        if "payload" not in d:
            raise ValueError("task-artifact manifest is missing required field 'payload'")
        payload = d["payload"]
        if payload not in KNOWN_PAYLOADS:
            raise ValueError(f"unknown task-artifact payload {payload!r}; expected one of {KNOWN_PAYLOADS}")

        builder = d.get("builder")
        if payload in ("torch", "arrays") and (not isinstance(builder, str) or not builder):
            raise ValueError(f"{payload!r} payload requires a non-empty string 'builder', got {builder!r}")

        config = d.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"manifest 'config' must be an object, got {type(config).__name__}")
        io = d.get("io", {})
        if not isinstance(io, dict):
            raise ValueError(f"manifest 'io' must be an object, got {type(io).__name__}")
        meta = d.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"manifest 'meta' must be an object, got {type(meta).__name__}")

        return cls(
            payload=payload,
            builder=builder,
            config=config,
            task=d.get("task", ""),
            io=io,
            meta=meta,
            schema_version=schema_version,
            created_at=d.get("created_at", ""),
            payload_sha256=d.get("payload_sha256", ""),
            integrity_sha256=d.get("integrity_sha256", ""),
        )


def read_manifest(path: str) -> TaskManifest:
    """Read only the manifest of an artifact directory without loading weights."""
    _recover_artifact(path)
    with open(os.path.join(path, MANIFEST_NAME)) as f:
        data = json.load(f)
    manifest = TaskManifest.from_dict(data)
    if not isinstance(manifest.payload_sha256, str) or len(manifest.payload_sha256) != 64:
        raise ValueError("task-artifact manifest has no valid payload_sha256 integrity binding.")
    if not isinstance(manifest.integrity_sha256, str) or len(manifest.integrity_sha256) != 64:
        raise ValueError("task-artifact manifest has no valid integrity_sha256 binding.")
    payload_path = os.path.join(path, _PAYLOAD_FILES[manifest.payload])
    actual_payload = _sha256_file(payload_path)
    if not hmac.compare_digest(actual_payload, manifest.payload_sha256):
        raise ValueError("task-artifact payload digest does not match its manifest.")
    actual_integrity = _manifest_integrity(data)
    if not hmac.compare_digest(actual_integrity, manifest.integrity_sha256):
        raise ValueError("task-artifact manifest integrity digest does not match its contents.")
    return manifest


def _write_manifest(path: str, manifest: TaskManifest) -> None:
    manifest.created_at = manifest.created_at or datetime.now(UTC).isoformat()
    manifest.payload_sha256 = _sha256_file(os.path.join(path, _PAYLOAD_FILES[manifest.payload]))
    data = manifest.to_dict()
    data["integrity_sha256"] = _manifest_integrity(data)
    manifest.integrity_sha256 = data["integrity_sha256"]
    _atomic_json_dump(
        os.path.join(path, MANIFEST_NAME),
        data,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


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

    Weights and manifest are staged together and published to ``path`` in one atomic step (see
    ``_atomic_artifact_write``): calling this on an existing artifact -- an update -- either fully replaces it
    or, on any failure, leaves the previous model/manifest/provenance fully intact. It never leaves new weights
    paired with the old manifest, or the reverse.
    """
    from safetensors.torch import save_model

    get_builder(builder)  # fail fast if the builder is unknown -- before writing anything

    def _write(staging: str) -> None:
        save_model(module, os.path.join(staging, WEIGHTS_NAME))
        _write_manifest(
            staging,
            TaskManifest(
                payload="torch", builder=builder, config=dict(config), task=task, io=io or {}, meta=meta or {}
            ),
        )

    _atomic_artifact_write(path, _write)
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
    """Persist a pure (torch-free) mixle distribution via the safe serialization registry; return ``path``.

    ``model.json`` and the manifest are staged together and published to ``path`` in one atomic step (see
    ``_atomic_artifact_write``), so an update that fails partway -- e.g. a non-serializable replacement model --
    never leaves ``path`` with a new model paired with the old manifest, or a truncated file of either kind.
    """
    from mixle.utils.serialization import ensure_pysp_serialization_registry, to_serializable

    ensure_pysp_serialization_registry()

    def _write(staging: str) -> None:
        _atomic_json_dump(os.path.join(staging, JSON_MODEL_NAME), to_serializable(model))
        _write_manifest(staging, TaskManifest(payload="json", task=task, io=io or {}, meta=meta or {}))

    _atomic_artifact_write(path, _write)
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
    """Persist a dict of numpy arrays as an artifact directory (``arrays.npz``); return ``path``.

    Arrays and manifest are staged together and published to ``path`` in one atomic step (see
    ``_atomic_artifact_write``), so a failed update never leaves new arrays paired with the old manifest.

    Written compressed. This payload holds quantized edge students whose whole point is bytes on a
    device, and every member costs a fixed ~128-byte ``.npy`` header plus zip framing -- on a small
    student the per-member framing outweighs the weights themselves. ``np.load`` reads compressed and
    stored archives identically, so this is transparent to :func:`load_arrays` and to any artifact
    written by an earlier version.
    """
    import numpy as np

    get_arrays_builder(builder)  # fail fast before writing anything

    def _write(staging: str) -> None:
        np.savez_compressed(os.path.join(staging, ARRAYS_NAME), **arrays)
        _write_manifest(
            staging,
            TaskManifest(
                payload="arrays", builder=builder, config=dict(config or {}), task=task, io=io or {}, meta=meta or {}
            ),
        )

    _atomic_artifact_write(path, _write)
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
