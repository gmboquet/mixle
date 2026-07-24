"""Scientific-array data sources -- zarr / HDF5 / numpy-memmap volumes, plus a ``PatchSampler``.

mixle's data model is records/tuples/sequences; a 40 GB array on disk has no first-class answer without
this module. Each array-store connector is *lazy*: constructing it opens the store's metadata (shape,
dtype, chunking) but never reads the full array into memory -- only requested slices are pulled off disk.

``PatchSampler`` wraps any of these (or any N-D array-like object exposing ``.shape``/``__getitem__``)
and yields ``(patch, coords)`` records -- fixed-size N-D patches at deterministic, seeded locations --
without ever materializing the underlying volume. Because it also implements ``__len__``/``__getitem__``
by index, it plugs directly into :class:`~mixle.utils.parallel.multiprocessing.MPEncodedData`: the driver
shards by index (``data[j] for j in range(i, n, num_workers)``) and only *those* patches are read and
pickled to worker ``i`` -- the full volume is never touched by the driver process.

Optional: zarr and h5py are guarded behind ``mixle.utils.optional_deps`` (``pip install mixle[arrays]``);
numpy-memmap needs no extra dependency. Every connector still constructs even when its dependency is
uninstalled -- it defers the ``require(...)`` error to first use (matching ``sql_source``/``mongo_source``),
except memmap, which never has a missing dependency to guard.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from mixle.data.partition import num_chunks_for
from mixle.data.schema import Schema
from mixle.data.structure import EXCHANGEABLE, SampleStructure
from mixle.utils.optional_deps import HAS_H5PY, HAS_ZARR, require
from mixle.utils.optional_deps import h5py as _h5py
from mixle.utils.optional_deps import zarr as _zarr

__all__ = [
    "ZarrArraySource",
    "HDF5ArraySource",
    "MemmapArraySource",
    "PatchSampler",
    "read_zarr",
    "read_hdf5",
    "read_memmap",
]


# Bounds how many records `.encode()` buffers per partition before flushing them through the schema
# and encoder. The memory ceiling for the incremental encode path below is `num_partitions *
# _ENCODE_BATCH_ROWS` records, never the source's total length -- callers that need a different bound
# (larger records want a smaller batch; tests want a tiny one to make bounded reads observable) can
# pass `batch_size` to `encode()` directly.
_ENCODE_BATCH_ROWS = 1024


def _encode_indexable(
    obj: Any,
    encoder: Any,
    structure: SampleStructure,
    schema: Schema | None,
    num_chunks: int,
    chunk_size: int | None,
    batch_size: int,
) -> list[tuple[int, Any]]:
    """Partition and ``encoder.seq_encode`` an ``obj`` supporting ``len``/``__getitem__(i)``, incrementally.

    Shared by :meth:`_ArrayVolumeSource.encode` and :meth:`PatchSampler.encode` (MXR-080-0056,
    MXR-080-0057). Never builds ``list(obj[i] for i in range(len(obj)))``: records are read one at a
    time, off disk/memmap, into small per-partition buffers that are schema-conformed
    (:meth:`~mixle.data.schema.Schema.conform`, applied per bounded batch rather than once over the
    whole source) and handed to ``encoder.seq_encode`` as soon as a buffer reaches ``batch_size``
    records -- so resident memory is bounded by ``num_partitions * batch_size`` records, independent
    of ``len(obj)``. The returned list is consumed exactly like ``encode_partitions``'s
    ``[(count, payload)]`` output by every existing caller (``seq_log_density_sum``, ``seq_estimate``,
    ``seq_log_density`` + ``np.concatenate`` for order-preserving ``num_chunks=1`` reads, ...), all of
    which sum/iterate/concatenate over however many entries the list contains rather than assuming
    exactly ``num_chunks`` of them -- so a partition legitimately arriving as several flushed batches
    (in read order) instead of one big list is not a behavior change from their point of view.

    Strideable structures (``structure.strides_records`` -- everything except partially-exchangeable)
    bucket record ``i`` into partition ``i % num_partitions`` as it is read, matching the
    ``records[k::n]`` striding :func:`~mixle.data.partition.partition_records` uses, computed without
    ever holding ``records`` itself.

    Partially-exchangeable (grouped) structures need a whole group in one partition, which needs every
    record's group key before any partition assignment is final. Rather than holding the group-keyed
    records in memory (the original whole-volume-materializing bug), this makes one pass to build an
    index of ``group key -> [row index, ...]`` -- integers only, negligible next to the row data --
    assigns whole groups to partitions round-robin exactly as ``partition_records`` does, then makes a
    second pass that re-reads each partition's rows by index, off disk, in the same bounded batches.
    This is the explicit spill/index strategy MXR-080-0056 calls for: row data is spilled (never
    retained) between the two passes and only the lightweight index survives between them, at the cost
    of reading each grouped record twice instead of holding all of them at once.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    n_total = len(obj)
    n_parts = num_chunks_for(n_total, num_chunks, chunk_size)
    results: list[tuple[int, Any]] = []

    def flush(buf: list[Any]) -> None:
        if not buf:
            return
        conformed = schema.conform(buf) if schema is not None else list(buf)
        results.append((len(conformed), encoder.seq_encode(conformed)))

    if structure.strides_records:
        buffers: list[list[Any]] = [[] for _ in range(n_parts)]
        for i in range(n_total):
            part = i % n_parts
            buffers[part].append(obj[i])
            if len(buffers[part]) >= batch_size:
                flush(buffers[part])
                buffers[part] = []
        for buf in buffers:
            flush(buf)
        return results

    # Grouped: pass 1 builds a row-index (not row-data) per group key, reading each row exactly once
    # and discarding it immediately after computing its key.
    index: dict[Any, list[int]] = {}
    order: list[Any] = []
    for i in range(n_total):
        key = structure.group_key(obj[i])
        bucket = index.get(key)
        if bucket is None:
            index[key] = bucket = []
            order.append(key)
        bucket.append(i)

    # Pass 2: assign whole groups to partitions round-robin (mirrors partition_records exactly).
    part_row_indices: list[list[int]] = [[] for _ in range(n_parts)]
    for g, key in enumerate(order):
        part_row_indices[g % n_parts].extend(index[key])

    # Pass 3: re-read each partition's rows by index, off disk, in bounded batches.
    for row_indices in part_row_indices:
        buf: list[Any] = []
        for i in row_indices:
            buf.append(obj[i])
            if len(buf) >= batch_size:
                flush(buf)
                buf = []
        flush(buf)
    return results


class _ArrayVolumeSource:
    """Shared base for the array-store connectors: a lazy, index-along-axis-0 ``DataSource``.

    ``records()``/``encode()`` satisfy the ordinary :class:`~mixle.data.core.DataSource` protocol
    (iterating row-by-row along the leading axis, each row read fresh off disk); ``.array`` exposes the
    underlying lazy N-D object (a zarr array / h5py dataset / memmap) for :class:`PatchSampler`, which
    needs full N-D slicing rather than row iteration.
    """

    array: Any

    def __init__(self, structure: SampleStructure, schema: Schema | None) -> None:
        self.structure = structure
        self.schema = schema

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> Any:
        return self.array.dtype

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.array[i])

    def records(self) -> Iterable[Any]:
        return (self[i] for i in range(len(self)))

    def encode(
        self,
        encoder: Any,
        num_chunks: int = 1,
        chunk_size: int | None = None,
        batch_size: int = _ENCODE_BATCH_ROWS,
    ) -> list[tuple[int, Any]]:
        """Partition and encode in bounded batches -- never reads the full volume into memory.

        See :func:`_encode_indexable` (MXR-080-0056, MXR-080-0057): rows are read off disk and
        schema-conformed ``batch_size`` at a time, not all up front via ``list(self.records())``.
        """
        return _encode_indexable(self, encoder, self.structure, self.schema, num_chunks, chunk_size, batch_size)


class ZarrArraySource(_ArrayVolumeSource):
    """Lazy connector over a zarr array (or a named array within a zarr group/store).

    Optional: requires ``zarr`` (``pip install mixle[arrays]``). Opening a zarr store reads only its
    metadata; ``self.array`` is the live zarr ``Array`` -- slicing it (row iteration here, or arbitrary
    N-D slices via :class:`PatchSampler`) reads and decompresses only the requested chunks.
    """

    def __init__(
        self,
        path: str,
        component: str | None = None,
        *,
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
    ) -> None:
        super().__init__(structure, schema)
        if not HAS_ZARR:
            require("zarr", "arrays")
        store = _zarr.open(path, mode="r")
        self.array = store[component] if component is not None else store


class HDF5ArraySource(_ArrayVolumeSource):
    """Lazy connector over an HDF5 dataset within a file.

    Optional: requires ``h5py`` (``pip install mixle[arrays]``). The file handle is opened read-only
    and kept resident (h5py datasets are themselves lazy: indexing reads only the requested slice); call
    :meth:`close` (or use as a context manager) to release it.
    """

    def __init__(
        self,
        path: str,
        dataset: str,
        *,
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
    ) -> None:
        super().__init__(structure, schema)
        if not HAS_H5PY:
            require("h5py", "arrays")
        self._file = _h5py.File(path, "r")
        self.array = self._file[dataset]

    def close(self) -> None:
        """Close the underlying HDF5 file handle. Idempotent."""
        try:
            self._file.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> HDF5ArraySource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class MemmapArraySource(_ArrayVolumeSource):
    """Lazy connector over a numpy ``memmap`` volume (no optional dependency: pure numpy).

    ``np.memmap`` maps the file into the process's address space and only pages in the slices that are
    actually indexed, so this never reads the full volume into resident memory either.
    """

    def __init__(
        self,
        path: str,
        dtype: Any,
        shape: tuple[int, ...],
        *,
        mode: str = "r",
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
    ) -> None:
        super().__init__(structure, schema)
        self.array = np.memmap(path, dtype=dtype, mode=mode, shape=tuple(shape))


def read_zarr(
    path: str,
    component: str | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> ZarrArraySource:
    """Open a zarr array/store lazily as a :class:`~mixle.data.core.DataSource`."""
    return ZarrArraySource(path, component, structure=structure, schema=schema)


def read_hdf5(
    path: str,
    dataset: str,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> HDF5ArraySource:
    """Open an HDF5 dataset lazily as a :class:`~mixle.data.core.DataSource`."""
    return HDF5ArraySource(path, dataset, structure=structure, schema=schema)


def read_memmap(
    path: str,
    dtype: Any,
    shape: tuple[int, ...],
    *,
    mode: str = "r",
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> MemmapArraySource:
    """Open a numpy-memmap volume lazily as a :class:`~mixle.data.core.DataSource`."""
    return MemmapArraySource(path, dtype, shape, mode=mode, structure=structure, schema=schema)


class PatchSampler:
    """Yield ``(patch, coords)`` records -- fixed-size N-D patches sampled from an array -- lazily.

    Wraps any N-D array-like object with a ``.shape`` and N-D ``__getitem__`` (a
    :class:`ZarrArraySource`/``HDF5ArraySource``/``MemmapArraySource``'s ``.array``, a raw zarr/h5py/
    memmap object, or a plain ``numpy.ndarray``). Patch placement is a deterministic function of
    ``seed``: the top-left corner of each patch is drawn from a seeded ``numpy.random.Generator`` once,
    up front (cheap -- it is only ``num_patches`` integer tuples), and never touches the underlying
    array until a patch is actually indexed. Iterating/indexing therefore reads only the requested
    patches off disk, never the full volume.

    Args:
        array: N-D array-like source (anything supporting ``.shape`` and N-D ``__getitem__``).
        patch_size: the N-D patch extent; must have the same rank as ``array.shape``.
        num_patches: how many ``(patch, coords)`` records to sample.
        seed: seeds the corner-placement RNG; identical ``seed`` -> identical patch sequence.
        stride: if given, corners are snapped to this stride along every axis (useful for aligning
            patches to a chunk grid); default ``None`` samples arbitrary integer corners.
        structure: the :class:`~mixle.data.structure.SampleStructure` of the patch stream (patches are
            i.i.d. draws from the volume by default, hence ``EXCHANGEABLE``).
    """

    def __init__(
        self,
        array: Any,
        patch_size: Sequence[int],
        num_patches: int,
        *,
        seed: int = 0,
        stride: int | None = None,
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
    ) -> None:
        self._array = array
        self.shape = tuple(int(s) for s in array.shape)
        self.patch_size = tuple(int(p) for p in patch_size)
        if len(self.patch_size) != len(self.shape):
            raise ValueError(
                "patch_size rank %d does not match array rank %d" % (len(self.patch_size), len(self.shape))
            )
        for s, p in zip(self.shape, self.patch_size):
            if p <= 0 or p > s:
                raise ValueError("patch_size %r does not fit within array shape %r" % (self.patch_size, self.shape))
        # Require exact nonnegative cardinality and exact positive stride at construction (MXR-080-0058):
        # a blind int(num_patches) truncated fractional input and let negatives through to make __len__
        # return a negative number (a Python contract violation), and a blind stride reached a
        # ZeroDivisionError for 0 or silently produced out-of-range corners for negative values instead
        # of a clear error here. isinstance(..., bool) is excluded explicitly since bool is an int
        # subclass in Python (True/False are not meaningful cardinalities or strides).
        if isinstance(num_patches, bool) or not isinstance(num_patches, (int, np.integer)) or num_patches < 0:
            raise ValueError(f"num_patches must be a non-negative integer, got {num_patches!r}")
        if stride is not None and (
            isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride <= 0
        ):
            raise ValueError(f"stride must be a positive integer or None, got {stride!r}")
        self.num_patches = int(num_patches)
        self.seed = seed
        self.stride = None if stride is None else int(stride)
        self.structure = structure
        self.schema = schema
        self._coords = self._sample_coords()

    def _sample_coords(self) -> list[tuple[int, ...]]:
        rng = np.random.default_rng(self.seed)
        coords: list[tuple[int, ...]] = []
        highs = [s - p + 1 for s, p in zip(self.shape, self.patch_size)]
        for _ in range(self.num_patches):
            corner = tuple(int(rng.integers(0, h)) for h in highs)
            if self.stride is not None:
                corner = tuple((c // self.stride) * self.stride for c in corner)
            coords.append(corner)
        return coords

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, i: int) -> tuple[np.ndarray, tuple[int, ...]]:
        corner = self._coords[i]
        slices = tuple(slice(c, c + p) for c, p in zip(corner, self.patch_size))
        patch = np.asarray(self._array[slices])
        return patch, corner

    def coords(self) -> list[tuple[int, ...]]:
        """Return the full, deterministic list of sampled patch corners (cheap -- no array I/O)."""
        return list(self._coords)

    def records(self) -> Iterable[tuple[np.ndarray, tuple[int, ...]]]:
        return (self[i] for i in range(len(self)))

    def encode(
        self,
        encoder: Any,
        num_chunks: int = 1,
        chunk_size: int | None = None,
        batch_size: int = _ENCODE_BATCH_ROWS,
    ) -> list[tuple[int, Any]]:
        """Partition and encode in bounded batches -- never materializes every sampled patch at once.

        See :func:`_encode_indexable` (MXR-080-0056, MXR-080-0057): patches are read off disk and
        schema-conformed ``batch_size`` at a time, not all up front via ``list(self.records())``.
        """
        return _encode_indexable(self, encoder, self.structure, self.schema, num_chunks, chunk_size, batch_size)
