"""Resource and placement planning helpers.

The planner is advisory: it estimates memory pressure from an encoder/model
pair and produces a printable, editable placement.  Orchestrators still own
actual data movement and sufficient-statistic folding.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.engines import NUMPY_ENGINE, NumpyEngine, auto_precision, engine_with_precision, precision_name
from mixle.stats import ResidentEncodedPayload, move_encoded_payload
from mixle.stats.compute.pdist import DataSequenceEncoder, encoded_nbytes

__all__ = [
    "CalibrationCatalog",
    "CalibrationRecord",
    "DeviceSpec",
    "DaskEncodedData",
    "EncodedDataHandle",
    "EncodedFold",
    "LocalEncodedData",
    "ModelShard",
    "Placement",
    "PlacementShard",
    "Resources",
    "SparkEncodedData",
    "available_encoded_data_backends",
    "calibrate_resources",
    "encoded_data",
    "estimate_estimator_stat_nbytes",
    "estimate_model_nbytes",
    "is_encoded_data_handle",
    "model_sharding_plan",
    "plan",
    "register_encoded_data_backend",
]


@dataclass(frozen=True)
class DeviceSpec:
    """Description of one compute placement target."""

    name: str
    kind: str = "cpu"
    memory_bytes: int | None = None
    engine: str = "numpy"
    throughput: float = 1.0
    precision: str | None = None

    @property
    def is_gpu(self) -> bool:
        """Return true for CUDA/MPS/GPU-like devices."""
        return self.kind in ("cuda", "mps", "gpu")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable device description."""
        return {
            "name": self.name,
            "kind": self.kind,
            "memory_bytes": self.memory_bytes,
            "engine": self.engine,
            "throughput": self.throughput,
            "precision": self.precision,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DeviceSpec:
        """Build a device description from ``to_dict`` output."""
        return cls(
            name=str(payload["name"]),
            kind=str(payload.get("kind", "cpu")),
            memory_bytes=None if payload.get("memory_bytes") is None else int(payload["memory_bytes"]),
            engine=str(payload.get("engine", "numpy")),
            throughput=float(payload.get("throughput", 1.0)),
            precision=payload.get("precision"),
        )


@dataclass
class Resources:
    """A collection of placement targets."""

    devices: tuple[DeviceSpec, ...]

    def __post_init__(self) -> None:
        if len(self.devices) == 0:
            raise ValueError("Resources requires at least one device.")
        for device in self.devices:
            if device.throughput <= 0.0:
                raise ValueError("device throughput must be positive.")
            if device.memory_bytes is not None and device.memory_bytes <= 0:
                raise ValueError("device memory_bytes must be positive when supplied.")

    @classmethod
    def single_cpu(
        cls, memory_bytes: int | None = None, throughput: float = 1.0, precision: Any | None = None
    ) -> Resources:
        """Return a one-device CPU resource description."""
        return cls(
            (
                DeviceSpec(
                    name="cpu:0",
                    kind="cpu",
                    memory_bytes=memory_bytes,
                    engine="numpy",
                    throughput=float(throughput),
                    precision=None if precision is None else precision_name(precision),
                ),
            )
        )

    @classmethod
    def local(
        cls, num_cpus: int | None = None, memory_bytes: int | None = None, precision: Any | None = None
    ) -> Resources:
        """Return local CPU resources split into logical worker slots."""
        count = os.cpu_count() if num_cpus is None else int(num_cpus)
        count = max(1, count)
        per_device_memory = None
        if memory_bytes is not None:
            per_device_memory = max(1, int(memory_bytes) // count)
        return cls(
            tuple(
                DeviceSpec(
                    name="cpu:%d" % i,
                    kind="cpu",
                    memory_bytes=per_device_memory,
                    engine="numpy",
                    throughput=1.0,
                    precision=None if precision is None else precision_name(precision),
                )
                for i in range(count)
            )
        )

    @classmethod
    def discover(
        cls, include_torch: bool = True, cpu_workers: int | None = None, precision: Any | None = None
    ) -> Resources:
        """Best-effort local resource discovery with no required extras."""
        devices: list[DeviceSpec] = list(cls.local(cpu_workers or 1, precision=precision).devices)
        if include_torch:
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                dtype = None if precision is None else precision_name(precision)
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        devices.append(
                            DeviceSpec(
                                name="cuda:%d" % i,
                                kind="cuda",
                                memory_bytes=int(props.total_memory),
                                engine="torch",
                                throughput=10.0,
                                precision=dtype,
                            )
                        )
                if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    devices.append(
                        DeviceSpec(
                            name="mps:0", kind="mps", memory_bytes=None, engine="torch", throughput=3.0, precision=dtype
                        )
                    )
        return cls(tuple(devices))

    @classmethod
    def from_specs(cls, specs: Iterable[DeviceSpec]) -> Resources:
        """Build resources from an iterable of device specifications."""
        return cls(tuple(specs))

    @classmethod
    def from_mpi(
        cls,
        comm: Any | None = None,
        memory_bytes: int | None = None,
        throughput: float = 1.0,
        precision: Any | None = None,
    ) -> Resources:
        """Return CPU resource slots for an MPI world without importing mpi4py.

        ``comm`` may be an mpi4py-style communicator exposing ``Get_size``.
        When it is omitted, common MPI launcher environment variables are used
        as a best-effort size hint, falling back to one slot.
        """
        if comm is not None and callable(getattr(comm, "Get_size", None)):
            size = int(comm.Get_size())
        else:
            size = int(
                os.environ.get("OMPI_COMM_WORLD_SIZE")
                or os.environ.get("PMI_SIZE")
                or os.environ.get("PMIX_SIZE")
                or os.environ.get("MPI_LOCALNRANKS")
                or 1
            )
        size = max(1, size)
        per_device_memory = None if memory_bytes is None else max(1, int(memory_bytes) // size)
        dtype = None if precision is None else precision_name(precision)
        return cls(
            tuple(
                DeviceSpec(
                    name="mpi:%d" % i,
                    kind="cpu",
                    memory_bytes=per_device_memory,
                    engine="numpy",
                    throughput=float(throughput),
                    precision=dtype,
                )
                for i in range(size)
            )
        )

    @classmethod
    def from_dask(cls, client: Any, precision: Any | None = None) -> Resources:
        """Return resource slots from a dask.distributed-like client.

        The method relies only on ``client.scheduler_info()`` and therefore
        does not introduce a dask dependency.
        """
        info_fn = getattr(client, "scheduler_info", None)
        if not callable(info_fn):
            raise TypeError("from_dask requires a client with scheduler_info().")
        info = info_fn()
        workers = info.get("workers", {}) if isinstance(info, dict) else {}
        if not workers:
            raise ValueError("dask scheduler_info did not report any workers.")
        dtype = None if precision is None else precision_name(precision)
        devices = []
        for i, (name, worker) in enumerate(sorted(workers.items(), key=lambda item: str(item[0]))):
            nthreads = int(worker.get("nthreads", 1)) if isinstance(worker, dict) else 1
            memory_limit = worker.get("memory_limit") if isinstance(worker, dict) else None
            devices.append(
                DeviceSpec(
                    name="dask:%s" % name,
                    kind="cpu",
                    memory_bytes=None if memory_limit is None else int(memory_limit),
                    engine="numpy",
                    throughput=max(1.0, float(nthreads)),
                    precision=dtype,
                )
            )
        return cls(tuple(devices))

    @classmethod
    def from_spark(cls, spark_context: Any, memory_bytes: int | None = None, precision: Any | None = None) -> Resources:
        """Return CPU resource slots from a SparkContext-like object."""
        workers = int(getattr(spark_context, "defaultParallelism", 1) or 1)
        workers = max(1, workers)
        per_device_memory = None if memory_bytes is None else max(1, int(memory_bytes) // workers)
        dtype = None if precision is None else precision_name(precision)
        return cls(
            tuple(
                DeviceSpec(
                    name="spark:%d" % i,
                    kind="cpu",
                    memory_bytes=per_device_memory,
                    engine="numpy",
                    throughput=1.0,
                    precision=dtype,
                )
                for i in range(workers)
            )
        )

    @classmethod
    def from_torchrun(cls, memory_bytes: int | None = None, precision: Any | None = None) -> Resources:
        """Return torchrun rank/device slots from launcher environment hints."""
        world = int(os.environ.get("WORLD_SIZE") or 1)
        world = max(1, world)
        dtype = None if precision is None else precision_name(precision)
        try:
            import torch
        except ImportError:
            torch = None
        cuda_count = int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0
        per_device_memory = None if memory_bytes is None else max(1, int(memory_bytes) // world)
        devices = []
        for rank in range(world):
            if cuda_count:
                dev_idx = rank % cuda_count
                try:
                    mem = int(torch.cuda.get_device_properties(dev_idx).total_memory)
                except Exception:  # noqa: BLE001
                    mem = per_device_memory
                devices.append(
                    DeviceSpec(
                        name="cuda:%d" % dev_idx,
                        kind="cuda",
                        memory_bytes=mem,
                        engine="torch",
                        throughput=10.0,
                        precision=dtype,
                    )
                )
            else:
                devices.append(
                    DeviceSpec(
                        name="torchrun:%d" % rank,
                        kind="cpu",
                        memory_bytes=per_device_memory,
                        engine="torch",
                        throughput=1.0,
                        precision=dtype,
                    )
                )
        return cls(tuple(devices))

    def fastest(self) -> DeviceSpec:
        """Return the device with the largest advisory throughput weight."""
        return max(self.devices, key=lambda d: d.throughput)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resource description."""
        return {"devices": [device.to_dict() for device in self.devices]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Resources:
        """Build resources from ``to_dict`` output."""
        devices = payload.get("devices")
        if devices is None:
            raise ValueError("resource payload requires a devices field.")
        return cls(tuple(DeviceSpec.from_dict(device) for device in devices))

    def to_json(self, **kwargs: Any) -> str:
        """Serialize resources, including calibrated throughput, to JSON."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> Resources:
        """Deserialize resources from JSON produced by ``to_json``."""
        return cls.from_dict(json.loads(text))

    def save(self, path: Any, **kwargs: Any) -> None:
        """Persist resources to a JSON file for reuse by later plans."""
        with open(path, "w") as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def load(cls, path: Any) -> Resources:
        """Load resources from a JSON file created by ``save``."""
        with open(path) as f:
            return cls.from_json(f.read())


@dataclass(frozen=True)
class CalibrationRecord:
    """One persisted calibration measurement for a model/workload/resource set."""

    model_type: str
    workload: str
    resources: Resources
    sample_size: int
    repeats: int
    precision: str | None = None
    estimator_type: str | None = None
    row_count: int | None = None
    model_bytes: int | None = None
    statistic_bytes: int | None = None
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration record."""
        return {
            "model_type": self.model_type,
            "estimator_type": self.estimator_type,
            "workload": self.workload,
            "sample_size": int(self.sample_size),
            "repeats": int(self.repeats),
            "precision": self.precision,
            "row_count": self.row_count,
            "model_bytes": self.model_bytes,
            "statistic_bytes": self.statistic_bytes,
            "timestamp": float(self.timestamp),
            "resources": self.resources.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationRecord:
        """Build a calibration record from ``to_dict`` output."""
        return cls(
            model_type=str(payload["model_type"]),
            estimator_type=payload.get("estimator_type"),
            workload=str(payload["workload"]),
            sample_size=int(payload.get("sample_size", 0)),
            repeats=int(payload.get("repeats", 0)),
            precision=payload.get("precision"),
            row_count=None if payload.get("row_count") is None else int(payload["row_count"]),
            model_bytes=None if payload.get("model_bytes") is None else int(payload["model_bytes"]),
            statistic_bytes=None if payload.get("statistic_bytes") is None else int(payload["statistic_bytes"]),
            timestamp=float(payload.get("timestamp", 0.0)),
            resources=Resources.from_dict(payload["resources"]),
        )


@dataclass
class CalibrationCatalog:
    """Append-only catalog of resource calibration measurements."""

    records: tuple[CalibrationRecord, ...] = ()

    def add(self, record: CalibrationRecord) -> CalibrationRecord:
        """Append ``record`` and return it."""
        self.records = tuple(self.records) + (record,)
        return record

    def latest(
        self, model_type: str | None = None, workload: str | None = None, precision: str | None = None
    ) -> CalibrationRecord | None:
        """Return the newest record matching the supplied filters."""
        workload_name = None if workload is None else str(workload).lower()
        for record in sorted(self.records, key=lambda r: r.timestamp, reverse=True):
            if model_type is not None and record.model_type != model_type:
                continue
            if workload_name is not None and record.workload != workload_name:
                continue
            if precision is not None and record.precision != precision_name(precision):
                continue
            return record
        return None

    def resources_for(
        self, model_type: str | None = None, workload: str | None = None, precision: str | None = None
    ) -> Resources | None:
        """Return calibrated resources from the newest matching record."""
        record = self.latest(model_type=model_type, workload=workload, precision=precision)
        return None if record is None else record.resources

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration catalog."""
        return {"records": [record.to_dict() for record in self.records]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationCatalog:
        """Build a catalog from ``to_dict`` output."""
        return cls(tuple(CalibrationRecord.from_dict(record) for record in payload.get("records", ())))

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the calibration catalog to JSON."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> CalibrationCatalog:
        """Deserialize a calibration catalog from JSON."""
        return cls.from_dict(json.loads(text))

    def save(self, path: Any, **kwargs: Any) -> None:
        """Persist the catalog to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def load(cls, path: Any) -> CalibrationCatalog:
        """Load a calibration catalog from disk."""
        with open(path) as f:
            return cls.from_json(f.read())


@dataclass(frozen=True)
class PlacementShard:
    """A contiguous row range assigned to one device."""

    device: DeviceSpec
    start: int
    stop: int
    sub_chunks: int = 1
    encoded_bytes: int = 0
    transient_bytes: int = 0

    @property
    def size(self) -> int:
        """Return the number of rows assigned to this shard."""
        return max(0, int(self.stop) - int(self.start))

    @property
    def total_bytes(self) -> int:
        """Return estimated encoded plus transient bytes for this shard."""
        return int(self.encoded_bytes) + int(self.transient_bytes)


@dataclass
class Placement:
    """Printable placement returned by ``plan``."""

    shards: tuple[PlacementShard, ...]
    total_rows: int
    encoded_row_bytes: float
    transient_row_bytes: float
    model_bytes: int
    statistic_bytes: int
    dtype_bytes: int

    def __str__(self) -> str:
        parts = [
            "Placement(total_rows=%d, shards=%d, row_bytes=%.1f, transient_row_bytes=%.1f, "
            "model_bytes=%d, statistic_bytes=%d, dtype_bytes=%d)"
            % (
                self.total_rows,
                len(self.shards),
                self.encoded_row_bytes,
                self.transient_row_bytes,
                self.model_bytes,
                self.statistic_bytes,
                self.dtype_bytes,
            )
        ]
        for shard in self.shards:
            parts.append(
                "  %s[%d:%d] rows=%d sub_chunks=%d estimated_bytes=%d"
                % (shard.device.name, shard.start, shard.stop, shard.size, shard.sub_chunks, shard.total_bytes)
            )
        return "\n".join(parts)

    def for_device(self, name: str) -> tuple[PlacementShard, ...]:
        """Return all placement shards assigned to a named device."""
        return tuple(shard for shard in self.shards if shard.device.name == name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly placement summary."""
        return {
            "total_rows": self.total_rows,
            "encoded_row_bytes": self.encoded_row_bytes,
            "transient_row_bytes": self.transient_row_bytes,
            "model_bytes": self.model_bytes,
            "statistic_bytes": self.statistic_bytes,
            "dtype_bytes": self.dtype_bytes,
            "shards": [
                {
                    "device": shard.device.name,
                    "kind": shard.device.kind,
                    "engine": shard.device.engine,
                    "start": shard.start,
                    "stop": shard.stop,
                    "sub_chunks": shard.sub_chunks,
                    "encoded_bytes": shard.encoded_bytes,
                    "transient_bytes": shard.transient_bytes,
                }
                for shard in self.shards
            ],
        }


@dataclass(frozen=True)
class ModelShard:
    """Advisory component-axis shard for a large model."""

    device: DeviceSpec
    component_start: int
    component_stop: int
    parameter_bytes: int = 0
    statistic_bytes: int = 0

    @property
    def num_components(self) -> int:
        """Return the number of mixture components in this model shard."""
        return max(0, int(self.component_stop) - int(self.component_start))

    @property
    def total_bytes(self) -> int:
        """Return estimated parameter plus statistic bytes for this shard."""
        return int(self.parameter_bytes) + int(self.statistic_bytes)


@dataclass
class _LocalShard:
    device: DeviceSpec
    engine: Any
    chunks: tuple[tuple[int, Any], ...]


class EncodedDataHandle:
    """Duck-typed orchestrator contract consumed by ``mixle.stats``.

    Local, multiprocessing, MPI, Spark, dask, or additional worker handles can
    implement these methods without sharing inheritance.  The base class exists
    to document the contract and to give local code a common type to return.
    """

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Return ``(num_observations, summed_log_density)`` for ``estimate``."""
        raise NotImplementedError

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Run one distributed/local sufficient-statistic fold and M-step."""
        raise NotImplementedError

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Initialize a model through the handle's resident encoded data."""
        raise NotImplementedError

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Return folded sufficient statistics for streaming/incremental EM."""
        raise NotImplementedError

    def close(self) -> None:
        """Release worker resources owned by this handle, if any."""
        return None

    def __enter__(self) -> EncodedDataHandle:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@runtime_checkable
class EncodedFold(Protocol):
    """Structural contract for the parallel/distributed fold consumed by ``mixle.stats``.

    This formalizes the duck-typed orchestrator contract that the concrete
    :class:`EncodedDataHandle` base class documents.  Local, multiprocessing,
    MPI, Spark, dask, Ray, Lightning, or additional worker handles satisfy this
    Protocol structurally without sharing inheritance.  Membership is decided by
    :func:`isinstance` against the four ``pysp_seq_*``/``pysp_stream_*`` methods.
    """

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Return total observation count and summed log-density for the encoded data."""
        ...

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Estimate a model from encoded data and a previous estimate."""
        ...

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Initialize an estimate from encoded data using Bernoulli subsampling."""
        ...

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Accumulate streaming sufficient statistics for encoded data."""
        ...


def is_encoded_data_handle(obj: Any) -> bool:
    """Return true when ``obj`` exposes the sequence-orchestrator contract."""
    return isinstance(obj, EncodedFold)


def encoded_data(
    data: Any,
    estimator: Any | None = None,
    model: Any | None = None,
    encoder: DataSequenceEncoder | None = None,
    placement: Placement | None = None,
    resources: Resources | None = None,
    engine: Any | None = None,
    precision: Any | None = None,
    num_chunks: int | None = None,
    sub_chunks: int = 1,
    backend: str = "local",
    num_workers: int | None = None,
    client: Any | None = None,
    comm: Any | None = None,
    root: int = 0,
    root_only: bool = False,
    parallel_chunks: bool = False,
    chunk_workers: int | None = None,
) -> EncodedDataHandle:
    """Return an encoded-data handle, preserving existing compatible handles.

    Backend dispatch goes through a registry (see :func:`register_encoded_data_backend`)
    rather than a hard-coded branch, so a new distributed framework (Lightning, Ray, JAX,
    ...) plugs in by registering a factory -- the same "register, don't branch" pattern the
    compute engines use -- without editing this function.
    """
    if is_encoded_data_handle(data):
        return data
    backend_name = str(backend or "local").lower()
    factory = _ENCODED_DATA_BACKENDS.get(backend_name)
    if factory is None:
        raise ValueError(
            "unknown encoded-data backend %r; registered backends: %s"
            % (backend, ", ".join(available_encoded_data_backends()))
        )
    return factory(
        data,
        estimator=estimator,
        model=model,
        encoder=encoder,
        placement=placement,
        resources=resources,
        engine=engine,
        precision=precision,
        num_chunks=num_chunks,
        sub_chunks=sub_chunks,
        num_workers=num_workers,
        client=client,
        comm=comm,
        root=root,
        root_only=root_only,
        parallel_chunks=parallel_chunks,
        chunk_workers=chunk_workers,
    )


# --- encoded-data backend registry ("register, don't branch") ----------------------------
# A backend factory has signature ``factory(data, **params) -> EncodedDataHandle`` where
# ``params`` are the keyword arguments of :func:`encoded_data`. Built-in backends are
# registered at import time below; third-party frameworks register their own.
_ENCODED_DATA_BACKENDS: dict[str, Any] = {}


def register_encoded_data_backend(
    name: str, factory: Any, aliases: tuple[str, ...] = (), *, override: bool = False
) -> None:
    """Register an encoded-data backend factory under ``name`` (and any ``aliases``).

    ``factory`` is called as ``factory(data, **params)`` with the keyword arguments of
    :func:`encoded_data`; it should accept ``**_`` for the parameters it ignores and return
    an :class:`EncodedDataHandle`. This is the extension point for new parallel/distributed
    frameworks -- registering is all that is needed, no core edits.

    Raises ``ValueError`` if ``name`` or any ``alias`` is already registered to a DIFFERENT
    factory, rather than silently shadowing it -- a third-party package registering under a
    name that collides with a built-in (e.g. ``'local'``) or another package's backend would
    otherwise hijack every subsequent ``backend=name`` call with no error, no warning, at
    whatever time that package happened to import. Pass ``override=True`` to replace an
    existing registration deliberately (e.g. hot-swapping a backend in a test).
    """
    if not callable(factory):
        raise TypeError("backend factory must be callable.")
    for key in (name, *aliases):
        existing = _ENCODED_DATA_BACKENDS.get(key.lower())
        if existing is not None and existing is not factory and not override:
            raise ValueError(
                f"encoded-data backend {key.lower()!r} is already registered to {existing!r}; "
                "pass override=True to replace it deliberately."
            )
    _ENCODED_DATA_BACKENDS[name.lower()] = factory
    for alias in aliases:
        _ENCODED_DATA_BACKENDS[alias.lower()] = factory


def available_encoded_data_backends() -> list[str]:
    """Return the sorted names of all registered encoded-data backends."""
    return sorted(_ENCODED_DATA_BACKENDS)


def _local_backend(
    data,
    *,
    estimator,
    model,
    encoder,
    placement,
    resources,
    engine,
    precision,
    num_chunks,
    sub_chunks,
    parallel_chunks=False,
    chunk_workers=None,
    **_,
):
    return LocalEncodedData(
        data,
        estimator=estimator,
        model=model,
        encoder=encoder,
        placement=placement,
        resources=resources,
        engine=engine,
        precision=precision,
        num_chunks=num_chunks,
        sub_chunks=sub_chunks,
        parallel_chunks=parallel_chunks,
        chunk_workers=chunk_workers,
    )


def _mp_backend(data, *, estimator, encoder, num_workers, sub_chunks, **_):
    from mixle.utils.parallel.multiprocessing import MPEncodedData

    return MPEncodedData(data, estimator=estimator, encoder=encoder, num_workers=num_workers, sub_chunks=sub_chunks)


def _mpi_backend(data, *, estimator, encoder, sub_chunks, comm, root, root_only, **_):
    from mixle.utils.parallel.mpi import MPIEncodedData

    return MPIEncodedData(
        data, estimator=estimator, encoder=encoder, sub_chunks=sub_chunks, comm=comm, root=root, root_only=root_only
    )


def _spark_backend(data, *, estimator, model, encoder, **_):
    return SparkEncodedData(data, estimator=estimator, model=model, encoder=encoder)


def _dask_backend(data, *, estimator, model, encoder, client, num_chunks, num_workers, sub_chunks, **_):
    return DaskEncodedData(
        data,
        estimator=estimator,
        model=model,
        encoder=encoder,
        client=client,
        num_partitions=num_chunks or num_workers,
        sub_chunks=sub_chunks,
    )


def _torchrun_backend(data, *, estimator, model, encoder, sub_chunks, comm, root, root_only, **_):
    from mixle.utils.parallel.torchrun import TorchRunEncodedData

    return TorchRunEncodedData(
        data,
        estimator=estimator,
        model=model,
        encoder=encoder,
        sub_chunks=sub_chunks,
        group=comm,
        root=root,
        root_only=root_only,
    )


def _lightning_backend(data, *, estimator, model, encoder, sub_chunks, **_):
    from mixle.utils.parallel.lightning_data import LightningEncodedData

    return LightningEncodedData(data, estimator=estimator, model=model, encoder=encoder, sub_chunks=sub_chunks)


def _ray_backend(data, *, estimator, model, encoder, num_chunks, num_workers, client, **_):
    from mixle.utils.parallel.ray_data import RayEncodedData

    return RayEncodedData(
        data,
        estimator=estimator,
        model=model,
        encoder=encoder,
        num_partitions=num_chunks,
        num_workers=num_workers,
        address=client,
    )


register_encoded_data_backend("local", _local_backend)
register_encoded_data_backend("mp", _mp_backend, aliases=("multiprocessing",))
register_encoded_data_backend("mpi", _mpi_backend)
register_encoded_data_backend("spark", _spark_backend)
register_encoded_data_backend("dask", _dask_backend)
register_encoded_data_backend("torchrun", _torchrun_backend)
register_encoded_data_backend("lightning", _lightning_backend, aliases=("pl",))
register_encoded_data_backend("ray", _ray_backend)


class LocalEncodedData(EncodedDataHandle):
    """In-process encoded-data handle implementing the orchestrator protocol.

    The handle encodes raw data once according to an advisory ``Placement`` and
    then exposes the ``pysp_seq_*`` methods recognized by ``mixle.stats``.  It is
    intentionally small: distributions/estimators own math, kernels own
    backend scoring/accumulation, and this object owns only local movement and
    global sufficient-statistic folding.
    """

    def __init__(
        self,
        data: Sequence[Any],
        estimator: Any | None = None,
        model: Any | None = None,
        encoder: DataSequenceEncoder | None = None,
        placement: Placement | None = None,
        resources: Resources | None = None,
        engine: Any | None = None,
        precision: Any | None = None,
        num_chunks: int | None = None,
        sub_chunks: int = 1,
        parallel_chunks: bool = False,
        chunk_workers: int | None = None,
    ) -> None:
        if len(data) == 0:
            raise ValueError("LocalEncodedData requires non-empty data.")
        self.parallel_chunks = bool(parallel_chunks)
        self._chunk_workers = chunk_workers
        if encoder is None:
            if model is not None and callable(getattr(model, "dist_to_encoder", None)):
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
        if encoder is None:
            raise ValueError("LocalEncodedData requires an encoder, model, or estimator.")
        if placement is None:
            placement = plan(
                data=data,
                model=model,
                estimator=estimator,
                encoder=encoder,
                resources=resources,
                engine=engine,
                precision=precision,
                num_chunks=num_chunks,
                sub_chunks=sub_chunks,
            )
        self.placement = placement
        self.encoder = encoder
        self.size = int(len(data))
        self.shards: tuple[_LocalShard, ...] = tuple(
            self._encode_shard(data, shard, engine=engine, precision=precision)
            for shard in placement.shards
            if shard.size > 0
        )

    def _encode_shard(
        self, data: Sequence[Any], shard: PlacementShard, engine: Any | None, precision: Any | None
    ) -> _LocalShard:
        shard_engine = engine if engine is not None else _engine_for_device(shard.device, precision)
        chunks = []
        part_count = max(1, int(shard.sub_chunks))
        for start, stop in _split_range(shard.start, shard.stop, part_count):
            if stop <= start:
                continue
            raw = [data[i] for i in range(start, stop)]
            host_payload = self.encoder.seq_encode(raw)
            engine_payload = move_encoded_payload(host_payload, shard_engine)
            chunks.append((len(raw), ResidentEncodedPayload(host_payload, engine_payload)))
        return _LocalShard(shard.device, shard_engine, tuple(chunks))

    def _chunk_pool_map(self, work: Any, items: list) -> list:
        """Apply ``work`` to each chunk task, preserving input order.

        When ``parallel_chunks`` is enabled (and there is more than one chunk) the work
        runs on a thread pool: per-chunk numpy accumulation/scoring releases the GIL inside
        vectorized ``seq_update``/``seq_log_density`` calls, so threads give real parallel
        speedup. Results are returned in input order, so the caller's fold stays
        bit-identical to the serial path regardless of completion order.
        """
        if not (self.parallel_chunks and len(items) > 1):
            return [work(it) for it in items]
        from concurrent.futures import ThreadPoolExecutor

        workers = self._chunk_workers or min(len(items), (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            return list(pool.map(work, items))

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Return total count and summed log density over resident chunks."""
        plans = [(shard, _kernel_or_none(estimate, shard.engine)) for shard in self.shards]
        count = float(sum(sz for shard, _ in plans for sz, _ in shard.chunks))
        if self.parallel_chunks and all(kernel is None for _, kernel in plans):
            tasks = [(sz, enc) for shard, _ in plans for sz, enc in shard.chunks]

            def _score(task: tuple) -> float:
                _, enc = task
                scores = estimate.seq_log_density(getattr(enc, "host_payload", enc))
                return float(np.asarray(scores, dtype=np.float64).sum())

            total = sum(self._chunk_pool_map(_score, tasks))
            return count, float(total)
        total = 0.0
        for shard, kernel in plans:
            for sz, enc in shard.chunks:
                if kernel is None:
                    scores = estimate.seq_log_density(getattr(enc, "host_payload", enc))
                    total += float(np.asarray(scores, dtype=np.float64).sum())
                else:
                    scores = kernel.score(enc)
                    total += float(np.asarray(shard.engine.to_numpy(scores), dtype=np.float64).sum())
        return count, total

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Run one local EM E-step/M-step through the unified handle contract."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        accumulator = estimator.accumulator_factory().make()
        plans = [(shard, _kernel_or_none(prev_estimate, shard.engine, estimator=estimator)) for shard in self.shards]
        nobs = float(sum(sz for shard, _ in plans for sz, _ in shard.chunks))
        if self.parallel_chunks and all(kernel is None for _, kernel in plans):
            tasks = [(sz, enc) for shard, _ in plans for sz, enc in shard.chunks]

            def _accumulate(task: tuple) -> Any:
                sz, enc = task
                local_acc = estimator.accumulator_factory().make()
                local_acc.seq_update(getattr(enc, "host_payload", enc), np.ones(sz, dtype=np.float64), prev_estimate)
                return local_acc.value()

            for value in self._chunk_pool_map(_accumulate, tasks):
                accumulator.combine(value)
        else:
            for shard, kernel in plans:
                for sz, enc in shard.chunks:
                    if kernel is None:
                        local_acc = estimator.accumulator_factory().make()
                        local_acc.seq_update(
                            getattr(enc, "host_payload", enc), np.ones(sz, dtype=np.float64), prev_estimate
                        )
                        accumulator.combine(local_acc.value())
                    else:
                        weights = shard.engine.asarray(np.ones(sz, dtype=np.float64))
                        accumulator.combine(kernel.accumulate(enc, weights))
        _global_key_merge(accumulator)
        return estimator.estimate(nobs, accumulator.value())

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Randomized initialization over resident encoded chunks."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        seeds = rng.randint(2**31, size=max(1, self.num_chunks))
        seed_idx = 0
        for shard in self.shards:
            for sz, enc in shard.chunks:
                rng_loc = np.random.RandomState(int(seeds[seed_idx % len(seeds)]))
                rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))
                seed_idx += 1
                weights = np.zeros(sz, dtype=np.float64)
                weights[rng_w.rand(sz) <= p] = 1.0
                nobs += float(weights.sum())
                local_acc = estimator.accumulator_factory().make()
                local_acc.seq_initialize(getattr(enc, "host_payload", enc), weights, rng_loc)
                accumulator.combine(local_acc.value())
        _global_key_merge(accumulator)
        return estimator.estimate(nobs, accumulator.value())

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Return globally tied batch sufficient statistics for streaming EM."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        accumulator = estimator.accumulator_factory().make()
        plans = [(shard, _kernel_or_none(model, shard.engine, estimator=estimator)) for shard in self.shards]
        nobs = float(sum(sz for shard, _ in plans for sz, _ in shard.chunks))
        if self.parallel_chunks and all(kernel is None for _, kernel in plans):
            tasks = [(sz, enc) for shard, _ in plans for sz, enc in shard.chunks]

            def _accumulate(task: tuple) -> Any:
                sz, enc = task
                local_acc = estimator.accumulator_factory().make()
                local_acc.seq_update(getattr(enc, "host_payload", enc), np.ones(sz, dtype=np.float64), model)
                return local_acc.value()

            for value in self._chunk_pool_map(_accumulate, tasks):
                accumulator.combine(value)
        else:
            for shard, kernel in plans:
                for sz, enc in shard.chunks:
                    if kernel is None:
                        local_acc = estimator.accumulator_factory().make()
                        local_acc.seq_update(getattr(enc, "host_payload", enc), np.ones(sz, dtype=np.float64), model)
                        accumulator.combine(local_acc.value())
                    else:
                        weights = shard.engine.asarray(np.ones(sz, dtype=np.float64))
                        accumulator.combine(kernel.accumulate(enc, weights))
        _global_key_merge(accumulator)
        return nobs, accumulator.value()

    @property
    def num_chunks(self) -> int:
        """Return the number of local encoded chunks across all shards."""
        return sum(len(shard.chunks) for shard in self.shards)

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        for shard in self.shards:
            for sz, enc in shard.chunks:
                yield sz, getattr(enc, "host_payload", enc)

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        """No-op for parity with process-backed handles."""
        return None


class SparkEncodedData(EncodedDataHandle):
    """Spark RDD encoded-data handle implementing the orchestrator protocol."""

    def __init__(
        self,
        data: Any,
        estimator: Any | None = None,
        model: Any | None = None,
        encoder: DataSequenceEncoder | None = None,
        materialize: bool = True,
    ) -> None:
        if encoder is None:
            if model is not None and callable(getattr(model, "dist_to_encoder", None)):
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
        if encoder is None:
            raise ValueError("SparkEncodedData requires an encoder, model, or estimator.")
        if not hasattr(data, "context"):
            raise TypeError("SparkEncodedData requires a Spark RDD-like object.")
        from mixle.stats import seq_encode

        self.encoder = encoder
        self.enc_rdd = seq_encode(data, encoder=encoder).cache()
        self.size = None
        if materialize:
            self.size = int(self.enc_rdd.map(lambda item: item[0]).sum())

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Return Spark-folded ``(count, summed_log_density)`` for ``estimate``."""
        from mixle.stats import seq_log_density_sum

        return seq_log_density_sum(self.enc_rdd, estimate)

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Run one Spark distributed E-step fold and estimator M-step."""
        from mixle.inference import seq_estimate

        return seq_estimate(self.enc_rdd, estimator, prev_estimate)

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Initialize a model over the resident Spark encoded RDD."""
        from mixle.inference import seq_initialize

        return seq_initialize(self.enc_rdd, estimator, rng, p)

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Return Spark-folded sufficient statistics for streaming EM."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        sc = self.enc_rdd.context
        estimator_b = sc.broadcast(estimator)
        model_b = sc.broadcast(pickle.dumps(model, protocol=0))

        def acc(split_index, itr):
            del split_index
            estimator_loc = estimator_b.value
            model_loc = pickle.loads(model_b.value)
            accumulator = estimator_loc.accumulator_factory().make()
            count = 0.0
            for sz, enc in itr:
                count += sz
                accumulator.seq_update(enc, np.ones(sz), model_loc)
            return [pickle.dumps((count, accumulator.value()), protocol=0)]

        temp = self.enc_rdd.mapPartitionsWithIndex(acc, True).cache()
        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()
        for raw in temp.collect():
            count, value = pickle.loads(raw)
            nobs += count
            accumulator.combine(value)
        _global_key_merge(accumulator)
        estimator_b.destroy()
        model_b.destroy()
        temp.unpersist()
        return nobs, accumulator.value()

    @property
    def num_chunks(self) -> int:
        """Return the number of Spark partitions in the encoded RDD."""
        return int(self.enc_rdd.getNumPartitions())

    def __len__(self) -> int:
        if self.size is None:
            self.size = int(self.enc_rdd.map(lambda item: item[0]).sum())
        return self.size

    def close(self) -> None:
        """Unpersist the encoded RDD from Spark storage."""
        self.enc_rdd.unpersist()


class DaskEncodedData(EncodedDataHandle):
    """dask.distributed encoded-data handle implementing the orchestrator protocol."""

    def __init__(
        self,
        data: Any,
        estimator: Any | None = None,
        model: Any | None = None,
        encoder: DataSequenceEncoder | None = None,
        client: Any | None = None,
        num_partitions: int | None = None,
        sub_chunks: int = 1,
        materialize: bool = True,
    ) -> None:
        if encoder is None:
            if model is not None and callable(getattr(model, "dist_to_encoder", None)):
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
        if encoder is None:
            raise ValueError("DaskEncodedData requires an encoder, model, or estimator.")
        self.encoder = encoder
        self.client, self._owns_client = _dask_client(client)
        self._closed = False
        self._partitions = tuple(self._submit_partitions(data, encoder, num_partitions, sub_chunks))
        if not self._partitions:
            raise ValueError("DaskEncodedData requires non-empty data.")
        self.size = None
        if materialize:
            self.size = int(
                sum(
                    self.client.gather(
                        [self.client.submit(_dask_payload_size, part, pure=False) for part in self._partitions]
                    )
                )
            )

    def _submit_partitions(
        self, data: Any, encoder: DataSequenceEncoder, num_partitions: int | None, sub_chunks: int
    ) -> list[Any]:
        encoder_b = pickle.dumps(encoder, protocol=pickle.HIGHEST_PROTOCOL)
        if hasattr(data, "to_delayed") and callable(data.to_delayed):
            try:
                from dask import delayed
            except ImportError as e:
                raise ImportError("DaskEncodedData requires dask for collection ingestion.") from e
            delayed_parts = list(data.to_delayed())
            delayed_payloads = [delayed(_dask_encode_partition)(encoder_b, part, sub_chunks) for part in delayed_parts]
            return list(self.client.compute(delayed_payloads))
        if not hasattr(data, "__len__"):
            data = list(data)
        nobs = len(data)
        if nobs == 0:
            return []
        if num_partitions is None:
            num_partitions = _dask_worker_count(self.client)
        num_partitions = max(1, min(int(num_partitions), nobs))
        futures = []
        for start, stop in _split_range(0, nobs, num_partitions):
            shard = [data[i] for i in range(start, stop)]
            futures.append(self.client.submit(_dask_encode_shard, encoder_b, shard, sub_chunks, pure=False))
        return futures

    def _fold_stats(self, estimator: Any, payloads: Iterable[bytes]) -> tuple[float, Any]:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        for raw in payloads:
            count, value = pickle.loads(raw)
            nobs += count
            accumulator.combine(value)
        _global_key_merge(accumulator)
        return nobs, accumulator.value()

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Return dask-folded ``(count, summed_log_density)`` for ``estimate``."""
        model_b = pickle.dumps(estimate, protocol=pickle.HIGHEST_PROTOCOL)
        futures = [self.client.submit(_dask_log_density_sum, part, model_b, pure=False) for part in self._partitions]
        results = self.client.gather(futures)
        return (sum(r[0] for r in results), sum(r[1] for r in results))

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Run one dask distributed E-step fold and estimator M-step."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        estimator_b = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        model_b = pickle.dumps(prev_estimate, protocol=pickle.HIGHEST_PROTOCOL)
        futures = [
            self.client.submit(_dask_update_partition, part, estimator_b, model_b, pure=False)
            for part in self._partitions
        ]
        nobs, value = self._fold_stats(estimator, self.client.gather(futures))
        return estimator.estimate(nobs, value)

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Initialize a model over persisted dask encoded partitions."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        estimator_b = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        seeds = rng.randint(2**31, size=max(1, len(self._partitions)))
        futures = [
            self.client.submit(_dask_initialize_partition, part, estimator_b, int(seed), float(p), pure=False)
            for part, seed in zip(self._partitions, seeds)
        ]
        nobs, value = self._fold_stats(estimator, self.client.gather(futures))
        return estimator.estimate(nobs, value)

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Return dask-folded sufficient statistics for streaming EM."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        estimator_b = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        model_b = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        futures = [
            self.client.submit(_dask_update_partition, part, estimator_b, model_b, pure=False)
            for part in self._partitions
        ]
        return self._fold_stats(estimator, self.client.gather(futures))

    @property
    def num_chunks(self) -> int:
        """Return the number of persisted dask encoded partitions."""
        return len(self._partitions)

    def __len__(self) -> int:
        if self.size is None:
            self.size = int(
                sum(
                    self.client.gather(
                        [self.client.submit(_dask_payload_size, part, pure=False) for part in self._partitions]
                    )
                )
            )
        return self.size

    def close(self) -> None:
        """Cancel persisted partitions and close an owned dask client."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._partitions:
                self.client.cancel(list(self._partitions), force=True)
        finally:
            if self._owns_client:
                self.client.close()


def _dask_client(client: Any | None) -> tuple[Any, bool]:
    if client is not None:
        return client, False
    try:
        from distributed import Client, get_client
    except ImportError as e:
        raise ImportError("DaskEncodedData requires dask.distributed.") from e
    try:
        return get_client(), False
    except ValueError:
        return Client(processes=False), True


def _dask_worker_count(client: Any) -> int:
    try:
        info = client.scheduler_info()
        workers = info.get("workers", {}) if isinstance(info, dict) else {}
        return max(1, len(workers))
    except Exception:  # noqa: BLE001
        return 1


def _dask_rows(partition: Any) -> list[Any]:
    if hasattr(partition, "itertuples"):
        return list(partition.itertuples(index=False, name=None))
    if hasattr(partition, "tolist"):
        values = partition.tolist()
        return values if isinstance(values, list) else list(values)
    return list(partition)


def _dask_encode_partition(
    encoder_b: bytes, partition: Any, sub_chunks: int
) -> tuple[int, tuple[tuple[int, Any], ...]]:
    return _dask_encode_shard(encoder_b, _dask_rows(partition), sub_chunks)


def _dask_encode_shard(
    encoder_b: bytes, shard: Sequence[Any], sub_chunks: int
) -> tuple[int, tuple[tuple[int, Any], ...]]:
    encoder = pickle.loads(encoder_b)
    nobs = len(shard)
    chunks = []
    part_count = max(1, min(int(sub_chunks), nobs)) if nobs else 1
    for start, stop in _split_range(0, nobs, part_count):
        part = [shard[i] for i in range(start, stop)]
        if part:
            chunks.append((len(part), encoder.seq_encode(part)))
    return nobs, tuple(chunks)


def _dask_payload_size(payload: tuple[int, tuple[tuple[int, Any], ...]]) -> int:
    return int(payload[0])


def _dask_update_partition(
    payload: tuple[int, tuple[tuple[int, Any], ...]], estimator_b: bytes, model_b: bytes
) -> bytes:
    estimator = pickle.loads(estimator_b)
    model = pickle.loads(model_b)
    accumulator = estimator.accumulator_factory().make()
    count = 0.0
    for sz, enc in payload[1]:
        count += sz
        accumulator.seq_update(enc, np.ones(sz, dtype=np.float64), model)
    return pickle.dumps((count, accumulator.value()), protocol=pickle.HIGHEST_PROTOCOL)


def _dask_initialize_partition(
    payload: tuple[int, tuple[tuple[int, Any], ...]], estimator_b: bytes, seed: int, p: float
) -> bytes:
    estimator = pickle.loads(estimator_b)
    accumulator = estimator.accumulator_factory().make()
    rng_loc = np.random.RandomState(seed)
    rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))
    count = 0.0
    for sz, enc in payload[1]:
        weights = np.zeros(sz, dtype=np.float64)
        weights[rng_w.rand(sz) <= p] = 1.0
        count += float(weights.sum())
        accumulator.seq_initialize(enc, weights, rng_loc)
    return pickle.dumps((count, accumulator.value()), protocol=pickle.HIGHEST_PROTOCOL)


def _dask_log_density_sum(payload: tuple[int, tuple[tuple[int, Any], ...]], model_b: bytes) -> tuple[float, float]:
    model = pickle.loads(model_b)
    count = 0.0
    total = 0.0
    for sz, enc in payload[1]:
        count += sz
        total += float(np.asarray(model.seq_log_density(enc), dtype=np.float64).sum())
    return count, total


def plan(
    data: Sequence[Any] | None = None,
    model: Any | None = None,
    estimator: Any | None = None,
    encoder: DataSequenceEncoder | None = None,
    resources: Resources | None = None,
    engine: Any | None = None,
    precision: Any | None = None,
    num_chunks: int | None = None,
    sub_chunks: int = 1,
    sample_size: int = 256,
    safety_factor: float = 1.25,
) -> Placement:
    """Return an advisory placement for local or distributed orchestration."""
    if precision == "auto":
        precision = auto_precision(data, engine=engine, sample_size=sample_size)
    if resources is None:
        resources = Resources.single_cpu(precision=precision)
    if sub_chunks <= 0:
        raise ValueError("sub_chunks must be positive.")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive.")
    if data is None and num_chunks is None:
        raise ValueError("plan requires data or an explicit num_chunks.")

    nobs = 0 if data is None else len(data)
    if nobs < 0:
        raise ValueError("data length must be non-negative.")
    if encoder is None:
        if model is not None and callable(getattr(model, "dist_to_encoder", None)):
            encoder = model.dist_to_encoder()
        elif estimator is not None:
            encoder = estimator.accumulator_factory().make().acc_to_encoder()
    if encoder is None and data is not None:
        raise ValueError("plan requires an encoder, model, or estimator when data are supplied.")

    engine = engine_with_precision(engine or NUMPY_ENGINE, precision)
    dtype_bytes = _dtype_bytes(getattr(engine, "dtype", None), precision)
    encoded_row_bytes = _estimate_encoded_row_bytes(data, encoder, sample_size) if data is not None else 0.0
    model_bytes = estimate_model_nbytes(model) if model is not None else 0
    statistic_bytes = estimate_estimator_stat_nbytes(estimator) if estimator is not None else model_bytes
    transient_row_bytes = _transient_row_bytes(model, dtype_bytes)

    chunks = _chunk_ranges(nobs, resources, num_chunks)
    shards: list[PlacementShard] = []
    for device, start, stop in chunks:
        size = stop - start
        row_bytes = encoded_row_bytes + transient_row_bytes
        estimated_payload = int(np.ceil(size * row_bytes * safety_factor))
        fixed = int(np.ceil((model_bytes + statistic_bytes) * safety_factor))
        max_rows = _max_rows_for_device(device, row_bytes, fixed, safety_factor)
        shard_count = max(1, int(np.ceil(size / max_rows))) if max_rows is not None and size > max_rows else 1
        for a, b in _split_range(start, stop, shard_count):
            part_size = b - a
            shards.append(
                PlacementShard(
                    device=device,
                    start=a,
                    stop=b,
                    sub_chunks=max(1, int(sub_chunks)),
                    encoded_bytes=int(np.ceil(part_size * encoded_row_bytes * safety_factor)),
                    transient_bytes=int(np.ceil(part_size * transient_row_bytes * safety_factor + fixed)),
                )
            )
    return Placement(
        shards=tuple(shards),
        total_rows=nobs,
        encoded_row_bytes=float(encoded_row_bytes),
        transient_row_bytes=float(transient_row_bytes),
        model_bytes=int(model_bytes),
        statistic_bytes=int(statistic_bytes),
        dtype_bytes=int(dtype_bytes),
    )


def model_sharding_plan(
    model: Any,
    resources: Resources,
    estimator: Any | None = None,
    axis: str = "components",
    min_components_per_shard: int = 1,
) -> tuple[ModelShard, ...]:
    """Return an advisory component-axis sharding plan.

    This is the planning half of model parallelism.  It does not create DTensor
    placements or move tensors; it answers which component range each device
    should own when a stacked/generated component kernel is available.
    """
    if axis != "components":
        raise ValueError("only axis='components' is currently supported.")
    if min_components_per_shard <= 0:
        raise ValueError("min_components_per_shard must be positive.")
    k = int(getattr(model, "num_components", 0) or 0)
    if k <= 0:
        raise ValueError("model does not expose a positive num_components attribute.")
    devices = tuple(resources.devices)
    max_shards = max(1, min(len(devices), int(np.ceil(k / float(min_components_per_shard)))))
    weights = np.asarray([d.throughput for d in devices[:max_shards]], dtype=np.float64)
    weights /= weights.sum()
    counts = np.floor(weights * k).astype(int)
    counts[counts < min_components_per_shard] = min_components_per_shard
    while counts.sum() > k:
        idx = int(np.argmax(counts))
        if counts[idx] > min_components_per_shard:
            counts[idx] -= 1
        else:
            break
    while counts.sum() < k:
        idx = int(np.argmax(weights - counts / float(k)))
        counts[idx] += 1
    counts = counts[:max_shards]

    total_param_bytes = estimate_model_nbytes(model)
    total_stat_bytes = estimate_estimator_stat_nbytes(estimator) if estimator is not None else total_param_bytes
    shards: list[ModelShard] = []
    start = 0
    for device, count in zip(devices[:max_shards], counts):
        stop = min(k, start + int(count))
        if stop <= start:
            continue
        frac = float(stop - start) / float(k)
        shards.append(
            ModelShard(
                device=device,
                component_start=start,
                component_stop=stop,
                parameter_bytes=int(np.ceil(total_param_bytes * frac)),
                statistic_bytes=int(np.ceil(total_stat_bytes * frac)),
            )
        )
        start = stop
    if start < k:
        last = shards[-1]
        shards[-1] = ModelShard(
            device=last.device,
            component_start=last.component_start,
            component_stop=k,
            parameter_bytes=last.parameter_bytes + int(np.ceil(total_param_bytes * (k - start) / float(k))),
            statistic_bytes=last.statistic_bytes + int(np.ceil(total_stat_bytes * (k - start) / float(k))),
        )
    return tuple(shards)


def calibrate_resources(
    data: Sequence[Any],
    model: Any,
    resources: Resources | None = None,
    estimator: Any | None = None,
    encoder: DataSequenceEncoder | None = None,
    sample_size: int = 512,
    repeats: int = 3,
    workload: str = "score",
    precision: Any | None = None,
    catalog: CalibrationCatalog | None = None,
    catalog_path: Any | None = None,
) -> Resources:
    """Time model scoring on each resource and return updated throughputs.

    Calibration is deliberately advisory and local.  It runs a small scoring
    or E-step pass through ``model.kernel(engine=...)`` where possible and
    leaves a device's previous throughput unchanged if that device cannot run
    the requested workload.  ``workload`` may be ``score``, ``estep`` /
    ``accumulate``, or ``em``.  The ``em`` workload includes the estimator's
    M-step on the sampled sufficient statistics.  Pass ``catalog`` and/or
    ``catalog_path`` to append a persisted model/workload calibration record.
    """
    if len(data) == 0:
        raise ValueError("calibrate_resources requires non-empty data.")
    workload_name = str(workload).lower()
    if workload_name == "accumulate":
        workload_name = "estep"
    if workload_name not in ("score", "estep", "em"):
        raise ValueError("unknown calibration workload %r" % workload)
    resources = Resources.single_cpu(precision=precision) if resources is None else resources
    encoder = model.dist_to_encoder() if encoder is None else encoder
    if workload_name in ("estep", "em") and estimator is None and callable(getattr(model, "estimator", None)):
        estimator = model.estimator()
    if workload_name in ("estep", "em") and estimator is None:
        raise ValueError("calibrate_resources workload=%r requires an estimator or model.estimator()." % workload)
    m = min(max(1, int(sample_size)), len(data))
    sample = [data[i] for i in range(m)]
    enc = encoder.seq_encode(sample)
    updated = []
    for device in resources.devices:
        try:
            engine = _engine_for_device(device, precision)
            enc_loc = move_encoded_payload(enc, engine)
            kernel = model.kernel(engine=engine, estimator=estimator)
            best = None
            for _ in range(max(1, int(repeats))):
                _synchronize(device)
                start = time.perf_counter()
                if workload_name == "score":
                    scores = kernel.score(enc_loc)
                    engine.to_numpy(scores)
                else:
                    weights = engine.asarray(np.ones(m, dtype=np.float64))
                    stats = kernel.accumulate(ResidentEncodedPayload(enc, enc_loc), weights)
                    if workload_name == "em":
                        estimator.estimate(float(m), stats)
                _synchronize(device)
                elapsed = max(time.perf_counter() - start, 1.0e-12)
                best = elapsed if best is None else min(best, elapsed)
            throughput = float(m) / float(best)
            updated.append(
                DeviceSpec(
                    name=device.name,
                    kind=device.kind,
                    memory_bytes=device.memory_bytes,
                    engine=device.engine,
                    throughput=throughput,
                    precision=device.precision if precision is None else precision_name(precision),
                )
            )
        except Exception:  # noqa: BLE001
            updated.append(device)
    calibrated = Resources(tuple(updated))
    _record_calibration(
        catalog=catalog,
        catalog_path=catalog_path,
        model=model,
        estimator=estimator,
        workload=workload_name,
        sample_size=m,
        repeats=max(1, int(repeats)),
        precision=precision,
        resources=calibrated,
        row_count=len(data),
    )
    return calibrated


def _record_calibration(
    catalog: CalibrationCatalog | None,
    catalog_path: Any | None,
    model: Any,
    estimator: Any | None,
    workload: str,
    sample_size: int,
    repeats: int,
    precision: Any | None,
    resources: Resources,
    row_count: int,
) -> None:
    if catalog is None and catalog_path is None:
        return
    target = catalog
    if target is None:
        if os.path.exists(catalog_path) and os.path.getsize(catalog_path) > 0:
            target = CalibrationCatalog.load(catalog_path)
        else:
            target = CalibrationCatalog()
    record = CalibrationRecord(
        model_type=type(model).__name__,
        estimator_type=None if estimator is None else type(estimator).__name__,
        workload=str(workload).lower(),
        sample_size=int(sample_size),
        repeats=int(repeats),
        precision=None if precision is None else precision_name(precision),
        row_count=int(row_count),
        model_bytes=estimate_model_nbytes(model),
        statistic_bytes=estimate_estimator_stat_nbytes(estimator) if estimator is not None else None,
        timestamp=time.time(),
        resources=resources,
    )
    target.add(record)
    if catalog_path is not None:
        target.save(catalog_path, sort_keys=True)


def estimate_model_nbytes(model: Any) -> int:
    """Approximate bytes held by a model's public parameter payload."""
    if model is None:
        return 0
    return _object_nbytes(model, set(), depth=0, max_depth=8)


def estimate_estimator_stat_nbytes(estimator: Any) -> int:
    """Approximate bytes in a zero accumulator value for an estimator."""
    if estimator is None:
        return 0
    try:
        acc = estimator.accumulator_factory().make()
        return _object_nbytes(acc.value(), set(), depth=0, max_depth=8)
    except Exception:  # noqa: BLE001
        return 0


def _estimate_encoded_row_bytes(data: Sequence[Any], encoder: DataSequenceEncoder, sample_size: int) -> float:
    if len(data) == 0:
        return 0.0
    m = min(max(1, int(sample_size)), len(data))
    sample = [data[i] for i in range(m)]
    payload = encoder.seq_encode(sample)
    return max(1.0, float(encoder.nbytes(payload)) / float(m))


def _engine_for_device(device: DeviceSpec, precision: Any | None) -> Any:
    dtype = precision if precision is not None else device.precision
    if device.engine == "torch":
        from mixle.engines import TorchEngine

        if device.kind == "cpu":
            device_name = "cpu"
        elif device.kind == "mps":
            device_name = "mps"
        else:
            device_name = device.name
        return TorchEngine(device=device_name, dtype=dtype)
    return NumpyEngine(dtype=dtype)


def _synchronize(device: DeviceSpec) -> None:
    if device.engine != "torch":
        return
    try:
        import torch
    except ImportError:
        return
    if device.kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device.name)


def _chunk_ranges(nobs: int, resources: Resources, num_chunks: int | None) -> list[tuple[DeviceSpec, int, int]]:
    if nobs == 0:
        return [(resources.fastest(), 0, 0)]
    weights = np.asarray([device.throughput for device in resources.devices], dtype=np.float64)
    weights /= weights.sum()
    if num_chunks is None:
        desired = np.floor(weights * nobs).astype(int)
        while desired.sum() < nobs:
            desired[int(np.argmax(weights - desired / float(nobs)))] += 1
        while desired.sum() > nobs:
            idx = int(np.argmax(desired))
            desired[idx] -= 1
        chunks = []
        start = 0
        for device, count in zip(resources.devices, desired):
            stop = start + int(count)
            if stop > start:
                chunks.append((device, start, stop))
            start = stop
        return chunks

    num_chunks = max(1, int(num_chunks))
    ranges = list(_split_range(0, nobs, num_chunks))
    devices = _weighted_device_cycle(resources.devices, weights, num_chunks)
    return [(device, start, stop) for device, (start, stop) in zip(devices, ranges)]


def _weighted_device_cycle(devices: tuple[DeviceSpec, ...], weights: np.ndarray, n: int) -> list[DeviceSpec]:
    counts = np.floor(weights * n).astype(int)
    while counts.sum() < n:
        counts[int(np.argmax(weights - counts / float(max(1, n))))] += 1
    rv = []
    for device, count in zip(devices, counts):
        rv.extend([device] * int(count))
    return rv[:n]


def _split_range(start: int, stop: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, int(parts))
    n = max(0, stop - start)
    rv = []
    for i in range(parts):
        a = start + (i * n) // parts
        b = start + ((i + 1) * n) // parts
        if b > a or n == 0:
            rv.append((a, b))
    return rv


def _max_rows_for_device(device: DeviceSpec, row_bytes: float, fixed_bytes: int, safety_factor: float) -> int | None:
    if device.memory_bytes is None or row_bytes <= 0.0:
        return None
    available = int(device.memory_bytes) - int(fixed_bytes)
    if available <= 0:
        return 1
    return max(1, int(available / max(1.0, row_bytes * safety_factor)))


def _transient_row_bytes(model: Any, dtype_bytes: int) -> float:
    k = int(getattr(model, "num_components", 1) or 1)
    states = int(getattr(model, "num_states", k) or k)
    return float(max(k, states, 1) * max(1, dtype_bytes) * 3)


def _dtype_bytes(dtype: Any, precision: Any | None) -> int:
    if precision is not None:
        name = precision_name(precision)
        if name in ("float16", "bfloat16"):
            return 2
        if name == "float32":
            return 4
        if name == "float64":
            return 8
    if dtype is None:
        return 8
    try:
        return int(np.dtype(dtype).itemsize)
    except TypeError:
        text = str(dtype)
        if "16" in text:
            return 2
        if "32" in text:
            return 4
        return 8


def _object_nbytes(x: Any, seen: set, depth: int, max_depth: int) -> int:
    if x is None or depth > max_depth:
        return 0
    oid = id(x)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if isinstance(x, (bytes, bytearray)):
        return len(x)
    if isinstance(x, str):
        return len(x.encode("utf-8"))
    if isinstance(x, (bool, np.bool_)):
        return 1
    if isinstance(x, (int, float, complex, np.number)):
        return sys.getsizeof(x)
    nbytes = getattr(x, "nbytes", None)
    if isinstance(nbytes, (int, np.integer)):
        return int(nbytes)
    if isinstance(x, dict):
        return sum(
            _object_nbytes(k, seen, depth + 1, max_depth) + _object_nbytes(v, seen, depth + 1, max_depth)
            for k, v in x.items()
        )
    if isinstance(x, (list, tuple)):
        return sum(_object_nbytes(v, seen, depth + 1, max_depth) for v in x)
    if hasattr(x, "__dict__"):
        total = 0
        for key, value in vars(x).items():
            if key.startswith("_") or callable(value):
                continue
            total += _object_nbytes(value, seen, depth + 1, max_depth)
        return total
    try:
        return encoded_nbytes(x)
    except Exception:  # noqa: BLE001
        return 0


def _kernel_or_none(model: Any, engine: Any, estimator: Any | None = None) -> Any:
    # Only a genuinely kernel-less model falls back to the legacy seq path;
    # a real failure inside a kernel factory must surface, not silently
    # degrade to the slow path.
    from mixle.stats.compute.backend import BackendScoringError

    if not hasattr(model, "kernel"):
        return None
    try:
        return model.kernel(engine=engine, estimator=estimator)
    except (NotImplementedError, BackendScoringError):
        return None


def _global_key_merge(accumulator: Any) -> None:
    stats_dict: dict[str, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
