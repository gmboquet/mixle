"""Sharded distributed checkpoints via ``torch.distributed.checkpoint`` (DCP) -- replaces pickle-broadcast at scale.

.. note::

   This is a provisional API with experimental multi-node behavior.
   Complete-state and failure semantics are covered on CPU; resharding and
   throughput at multi-node GPU scale still require retained hardware receipts.

The gather-to-root + ``pickle``-broadcast that :class:`TorchRunEncodedData` uses to move a model cannot save a
model that does not fit (and is not folded on) one rank. DCP saves each rank's shard of the (FSDP2-sharded) model
+ optimizer state in parallel to a checkpoint directory, and loads it back sharded -- the standard frontier
checkpoint, and the resume hook for :class:`~mixle.utils.parallel.torch_neural.StreamingTokenEncodedData`.

CUDA / multi-GPU path: implemented against the torch 2.4+ DCP and
distributed-state-dict APIs, but not verified without retained hardware receipts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT_VERSION = 2
_GLOBAL_METADATA_FIELDS = ("step", "scheduler", "scaler", "parallel_plan", "extra")
_RANK_LOCAL_METADATA_FIELDS = ("loader_state", "typed_scheduler_state", "rng")


def _rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sidecar_payload(
    *,
    step: int,
    scheduler: Any,
    scaler: Any,
    loader_state: Any,
    parallel_plan: Any,
    typed_scheduler_state: Any,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": _FORMAT_VERSION,
        "step": int(step),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "loader_state": loader_state.to_dict() if hasattr(loader_state, "to_dict") else loader_state,
        "parallel_plan": parallel_plan.as_dict() if hasattr(parallel_plan, "as_dict") else parallel_plan,
        "typed_scheduler_state": typed_scheduler_state,
        "rng": _rng_state(),
        "extra": dict(extra or {}),
    }


def _safe_encode(value: Any) -> Any:
    """Encode metadata without pickle or executable object reconstruction."""
    import torch

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"__type__": "float", "value": repr(value)}
    if isinstance(value, np.generic):
        return _safe_encode(value.item())
    if isinstance(value, bytes):
        return {"__type__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__type__": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {
            "__type__": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_safe_encode(item) for item in value]}
    if isinstance(value, list):
        return [_safe_encode(item) for item in value]
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [[_safe_encode(key), _safe_encode(item)] for key, item in value.items()],
        }
    raise TypeError(
        "checkpoint metadata type %s.%s is not safely serializable"
        % (type(value).__module__, type(value).__qualname__)
    )


def _safe_decode(value: Any) -> Any:
    """Decode only the closed typed metadata schema emitted by ``_safe_encode``."""
    import torch

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_decode(item) for item in value]
    if not isinstance(value, dict) or "__type__" not in value:
        raise RuntimeError("checkpoint metadata contains an invalid typed value")
    kind = value["__type__"]
    if kind == "float":
        values = {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}
        if value.get("value") not in values:
            raise RuntimeError("checkpoint metadata contains an invalid non-finite float")
        return values[value["value"]]
    if kind == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if kind == "ndarray":
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(size) for size in value["shape"])
        raw = base64.b64decode(value["data"], validate=True)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise RuntimeError("checkpoint ndarray metadata has an invalid byte length")
        return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
    if kind == "tensor":
        dtype_name = str(value["dtype"])
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError("checkpoint tensor metadata has an unsupported dtype")
        shape = tuple(int(size) for size in value["shape"])
        raw = base64.b64decode(value["data"], validate=True)
        itemsize = torch.empty((), dtype=dtype).element_size()
        expected = int(np.prod(shape, dtype=np.int64)) * itemsize
        if len(raw) != expected:
            raise RuntimeError("checkpoint tensor metadata has an invalid byte length")
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return byte_tensor.view(dtype).clone().reshape(shape)
    if kind == "tuple":
        return tuple(_safe_decode(item) for item in value["items"])
    if kind == "dict":
        return {_safe_decode(key): _safe_decode(item) for key, item in value["items"]}
    raise RuntimeError("checkpoint metadata contains unknown typed value %r" % kind)


def _atomic_write(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding=encoding) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _generation_path(destination: Path, rank: int) -> Path:
    """Create one fresh generation name shared by every distributed rank."""
    import torch.distributed as dist

    token = uuid.uuid4().hex if rank == 0 else None
    if dist.is_available() and dist.is_initialized():
        box = [token]
        dist.broadcast_object_list(box, src=0)
        token = box[0]
    generation = destination / ".generations" / ("generation-" + str(token))
    if rank == 0:
        generation.mkdir(parents=True, exist_ok=False)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    elif rank != 0:
        raise RuntimeError("non-root checkpoint rank requires an initialized process group")
    return generation


def _write_sidecar(path: Path, rank: int, payload: dict[str, Any]) -> None:
    target = path / ("rank-%05d.json" % rank)
    _atomic_write(target, json.dumps(_safe_encode(payload), sort_keys=True, separators=(",", ":")))


def _finalize_checkpoint(path: Path, *, rank: int, world_size: int) -> str:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        files = []
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file() or candidate.name in ("manifest.json", "_SUCCESS"):
                continue
            relative = candidate.relative_to(path).as_posix()
            files.append({"path": relative, "size": candidate.stat().st_size, "sha256": _sha256(candidate)})
        manifest = {
            "format_version": _FORMAT_VERSION,
            "world_size": world_size,
            "rank_sidecars": ["rank-%05d.json" % index for index in range(world_size)],
            "files": files,
        }
        manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        _atomic_write(path / "manifest.json", manifest_text)
        _atomic_write(path / "_SUCCESS", hashlib.sha256(manifest_text.encode("utf-8")).hexdigest() + "\n")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return path.name


def _publish_generation(destination: Path, generation: Path, *, rank: int) -> None:
    """Atomically publish a completed fresh generation through ``CURRENT``."""
    import torch.distributed as dist

    if rank == 0:
        relative = generation.relative_to(destination).as_posix()
        manifest_text = (generation / "manifest.json").read_text(encoding="utf-8")
        pointer = json.dumps(
            {
                "format_version": _FORMAT_VERSION,
                "generation": relative,
                "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        # Root-level files are compatibility summaries only. CURRENT is the
        # single atomic publication point used by the loader.
        _atomic_write(destination / "manifest.json", manifest_text)
        _atomic_write(destination / "_SUCCESS", pointer + "\n")
        _atomic_write(destination / "CURRENT", pointer + "\n")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _resolve_and_validate_generation(destination: Path) -> tuple[Path, dict[str, Any]]:
    current = destination / "CURRENT"
    if not current.is_file():
        raise RuntimeError("checkpoint is incomplete: %s has no CURRENT generation pointer." % destination)
    try:
        pointer = json.loads(current.read_text(encoding="utf-8"))
        if int(pointer["format_version"]) != _FORMAT_VERSION:
            raise RuntimeError("unsupported training checkpoint pointer version")
        relative = Path(pointer["generation"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("checkpoint generation pointer escapes its destination")
        generation = (destination / relative).resolve()
        root = destination.resolve()
        if generation.parent.parent != root or generation.parent.name != ".generations":
            raise RuntimeError("checkpoint generation pointer has an invalid location")
        manifest_path = generation / "manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        if digest != pointer["manifest_sha256"]:
            raise RuntimeError("checkpoint manifest does not match CURRENT")
        if (generation / "_SUCCESS").read_text(encoding="ascii").strip() != digest:
            raise RuntimeError("checkpoint generation completion marker is invalid")
        manifest = json.loads(manifest_text)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("checkpoint generation metadata is invalid") from exc
    if int(manifest.get("format_version", -1)) != _FORMAT_VERSION:
        raise RuntimeError("unsupported training checkpoint format version")
    saved_world_size = manifest.get("world_size")
    if isinstance(saved_world_size, bool) or not isinstance(saved_world_size, int) or saved_world_size <= 0:
        raise RuntimeError("checkpoint manifest has an invalid world size")
    declared_sidecars = manifest.get("rank_sidecars")
    required_sidecars = ["rank-%05d.json" % rank for rank in range(saved_world_size)]
    if declared_sidecars != required_sidecars:
        raise RuntimeError("checkpoint manifest rank sidecars do not match its saved world size")
    listed: set[str] = set()
    for record in manifest.get("files", ()):
        relative_name = str(record.get("path", ""))
        relative_path = Path(relative_name)
        if not relative_name or relative_path.is_absolute() or ".." in relative_path.parts or relative_name in listed:
            raise RuntimeError("checkpoint manifest contains an invalid file path")
        listed.add(relative_name)
        candidate = generation / relative_path
        if not candidate.is_file():
            raise RuntimeError("checkpoint file is missing: %s" % relative_name)
        if candidate.stat().st_size != int(record["size"]) or _sha256(candidate) != record["sha256"]:
            raise RuntimeError("checkpoint file failed integrity validation: %s" % relative_name)
    expected_sidecars = set(declared_sidecars)
    if not expected_sidecars or not expected_sidecars.issubset(listed):
        raise RuntimeError("checkpoint manifest does not cover every rank sidecar")
    actual = {
        candidate.relative_to(generation).as_posix()
        for candidate in generation.rglob("*")
        if candidate.is_file() and candidate.name not in ("manifest.json", "_SUCCESS")
    }
    if actual != listed:
        raise RuntimeError("checkpoint generation contains files outside its integrity manifest")
    return generation, manifest


def _dcp_state(module: Any, optimizer: Any) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(module, optimizer)
    return {"model": model_state, "optimizer": optimizer_state}


def _save_generation(state: dict[str, Any], destination: Path, payload: dict[str, Any]) -> None:
    """Write, verify, and publish one complete immutable generation."""
    import torch.distributed.checkpoint as dcp

    rank, world_size = _rank_world()
    destination.mkdir(parents=True, exist_ok=True)
    generation = _generation_path(destination, rank)
    dcp.save(state, checkpoint_id=str(generation))
    _write_sidecar(generation, rank, payload)
    _finalize_checkpoint(generation, rank=rank, world_size=world_size)
    _publish_generation(destination, generation, rank=rank)


def save_sharded(module: Any, optimizer: Any, path: str) -> None:
    """Save a sharded ``(model, optimizer)`` checkpoint to ``path`` -- every rank writes its own shard in parallel."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_sd, optim_sd = get_state_dict(module, optimizer)
    dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(path))


def load_sharded(module: Any, optimizer: Any, path: str) -> None:
    """Load a sharded checkpoint from ``path`` into ``module`` + ``optimizer`` in place (resumable training)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_sd, optim_sd = get_state_dict(module, optimizer)  # templates with the right sharded shapes
    dcp.load({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(path))
    set_state_dict(module, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)


def save_training_state(
    module: Any,
    optimizer: Any,
    path: str,
    *,
    step: int,
    scheduler: Any = None,
    scaler: Any = None,
    loader_state: Any = None,
    parallel_plan: Any = None,
    typed_scheduler_state: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Synchronously save complete, reshardable training state."""

    destination = Path(path)
    payload = _sidecar_payload(
        step=step,
        scheduler=scheduler,
        scaler=scaler,
        loader_state=loader_state,
        parallel_plan=parallel_plan,
        typed_scheduler_state=typed_scheduler_state,
        extra=extra,
    )
    _save_generation(_dcp_state(module, optimizer), destination, payload)


def save_frozen_training_state(
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    path: str,
    *,
    step: int,
    loader_state: Any = None,
    parallel_plan: Any = None,
    typed_scheduler_state: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Transactionally save already-frozen model and optimizer state dictionaries."""
    payload = _sidecar_payload(
        step=step,
        scheduler=None,
        scaler=None,
        loader_state=loader_state,
        parallel_plan=parallel_plan,
        typed_scheduler_state=typed_scheduler_state,
        extra=extra,
    )
    _save_generation(
        {"model": model_state, "optimizer": optimizer_state},
        Path(path),
        payload,
    )


def _decode_rank_metadata(source: Path, manifest: dict[str, Any], rank: int) -> dict[str, Any]:
    """Decode one explicitly selected rank sidecar from a validated generation."""
    sidecar_name = "rank-%05d.json" % rank
    if sidecar_name not in set(manifest["rank_sidecars"]):
        raise RuntimeError("checkpoint has no metadata sidecar for rank %d" % rank)
    try:
        encoded_payload = json.loads((source / sidecar_name).read_text(encoding="utf-8"))
        payload = _safe_decode(encoded_payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("checkpoint rank metadata is invalid") from exc
    if not isinstance(payload, dict) or int(payload.get("format_version", -1)) != _FORMAT_VERSION:
        raise RuntimeError("checkpoint rank metadata has an unsupported schema")
    required = (*_GLOBAL_METADATA_FIELDS, *_RANK_LOCAL_METADATA_FIELDS)
    for field in required:
        if field not in payload:
            raise RuntimeError("checkpoint rank metadata is missing %s" % field)
    return payload


def _split_rank_metadata(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate topology-invariant metadata from rank-owned resume state."""
    global_metadata = {field: payload[field] for field in _GLOBAL_METADATA_FIELDS}
    rank_local_metadata = {field: payload[field] for field in _RANK_LOCAL_METADATA_FIELDS}
    return global_metadata, rank_local_metadata


def _load_metadata_from_generation(
    source: Path,
    manifest: dict[str, Any],
    *,
    rank_local_transform: Callable[
        [tuple[dict[str, Any], ...], int, int, int], dict[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Load global metadata and only topology-compatible or explicitly transformed local state."""
    rank, world_size = _rank_world()
    saved_world_size = int(manifest["world_size"])
    changed = world_size != saved_world_size
    if not changed:
        payload = dict(_decode_rank_metadata(source, manifest, rank))
        if rank_local_transform is not None:
            raise ValueError("rank_local_transform is only valid when checkpoint world size changes")
        payload["rank_local_state_status"] = "native"
        payload["requires_rank_local_transform"] = False
    else:
        saved_payloads = tuple(
            _decode_rank_metadata(source, manifest, old_rank) for old_rank in range(saved_world_size)
        )
        global_metadata, _ = _split_rank_metadata(saved_payloads[0])
        canonical_global = json.dumps(_safe_encode(global_metadata), sort_keys=True, separators=(",", ":"))
        rank_local_states = []
        for old_rank, saved_payload in enumerate(saved_payloads):
            candidate_global, candidate_local = _split_rank_metadata(saved_payload)
            if json.dumps(_safe_encode(candidate_global), sort_keys=True, separators=(",", ":")) != canonical_global:
                raise RuntimeError(
                    "checkpoint global metadata differs across saved ranks; cannot choose an arbitrary copy"
                )
            rank_local_states.append(candidate_local)
        if rank_local_transform is None:
            rank_local_metadata = {field: None for field in _RANK_LOCAL_METADATA_FIELDS}
            status = "transform_required"
        else:
            rank_local_metadata = rank_local_transform(
                tuple(rank_local_states), saved_world_size, rank, world_size
            )
            if not isinstance(rank_local_metadata, dict):
                raise TypeError("rank_local_transform must return a metadata dictionary")
            missing = set(_RANK_LOCAL_METADATA_FIELDS) - set(rank_local_metadata)
            unknown = set(rank_local_metadata) - set(_RANK_LOCAL_METADATA_FIELDS)
            if missing or unknown:
                raise ValueError(
                    "rank_local_transform must return exactly loader_state, typed_scheduler_state, and rng"
                )
            rank_local_metadata = dict(rank_local_metadata)
            status = "transformed"
        payload = {
            "format_version": _FORMAT_VERSION,
            **global_metadata,
            **rank_local_metadata,
            "rank_local_state_status": status,
            "requires_rank_local_transform": rank_local_transform is None,
        }
    payload["world_size_changed"] = changed
    payload["saved_world_size"] = saved_world_size
    return payload


def load_training_metadata(
    path: str,
    *,
    rank_local_transform: Callable[
        [tuple[dict[str, Any], ...], int, int, int], dict[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Decode metadata without ever assigning an old rank's local state to a new rank."""
    source, manifest = _resolve_and_validate_generation(Path(path))
    return _load_metadata_from_generation(
        source,
        manifest,
        rank_local_transform=rank_local_transform,
    )


def load_training_state(
    module: Any,
    optimizer: Any,
    path: str,
    *,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
    allow_world_size_change: bool = False,
    rank_local_transform: Callable[
        [tuple[dict[str, Any], ...], int, int, int], dict[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Restore complete training state and return its non-tensor metadata."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    destination = Path(path)
    source, manifest = _resolve_and_validate_generation(destination)
    _, world_size = _rank_world()
    saved_world_size = int(manifest["world_size"])
    changed = world_size != saved_world_size
    if changed and not allow_world_size_change:
        raise RuntimeError(
            "checkpoint world_size changed from %d to %d; pass allow_world_size_change=True to reshard "
            "model state and rebuild rank-local loader state." % (saved_world_size, world_size)
        )
    if changed and rank_local_transform is None:
        raise RuntimeError(
            "world-size-changing restore requires rank_local_transform to explicitly repartition or "
            "rebuild loader, typed scheduler, and RNG state for each new rank"
        )
    payload = _load_metadata_from_generation(
        source,
        manifest,
        rank_local_transform=rank_local_transform,
    )

    # All files and typed metadata have been authenticated and decoded before
    # any live model, optimizer, scheduler, scaler, or RNG state is mutated.
    model_state, optimizer_state = get_state_dict(module, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(source))
    set_state_dict(
        module,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng") is not None:
        _restore_rng_state(payload["rng"])
    return payload


@dataclass
class AsyncTrainingCheckpoint:
    """DCP asynchronous save plus atomic sidecar/manifest finalization."""

    future: Any
    path: Path
    generation: Path
    rank: int
    world_size: int
    payload: dict[str, Any]
    _complete: bool = False

    def wait(self) -> None:
        if self._complete:
            return
        wait = getattr(self.future, "wait", None)
        if callable(wait):
            wait()
        else:
            self.future.result()
        _write_sidecar(self.generation, self.rank, self.payload)
        _finalize_checkpoint(self.generation, rank=self.rank, world_size=self.world_size)
        _publish_generation(self.path, self.generation, rank=self.rank)
        self._complete = True

    @property
    def done(self) -> bool:
        # I/O completion is not checkpoint completion until sidecars and the
        # manifest marker have been committed by ``wait``.
        return self._complete


def async_save_training_state(
    module: Any,
    optimizer: Any,
    path: str,
    *,
    step: int,
    scheduler: Any = None,
    scaler: Any = None,
    loader_state: Any = None,
    parallel_plan: Any = None,
    typed_scheduler_state: Any = None,
    extra: dict[str, Any] | None = None,
) -> AsyncTrainingCheckpoint:
    """Start DCP's native async save; :meth:`wait` surfaces write failures."""

    import torch.distributed.checkpoint as dcp

    rank, world_size = _rank_world()
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    generation = _generation_path(destination, rank)
    payload = _sidecar_payload(
        step=step,
        scheduler=scheduler,
        scaler=scaler,
        loader_state=loader_state,
        parallel_plan=parallel_plan,
        typed_scheduler_state=typed_scheduler_state,
        extra=extra,
    )
    future = dcp.async_save(_dcp_state(module, optimizer), checkpoint_id=str(generation))
    return AsyncTrainingCheckpoint(future, destination, generation, rank, world_size, payload)


__all__ = [
    "AsyncTrainingCheckpoint",
    "async_save_training_state",
    "load_sharded",
    "load_training_metadata",
    "load_training_state",
    "save_frozen_training_state",
    "save_sharded",
    "save_training_state",
]
