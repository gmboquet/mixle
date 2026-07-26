"""P5 (experimental) -- commitment-backed exact unlearning for audited closed-form leaves.

This module separates two phases that a deletion certificate must not conflate:

1. While raw shards are available, :func:`prepare_unlearning` computes each shard's additive sufficient
   statistic and seals the records in a cryptographic manifest.  The caller must retain the manifest digest
   in an independent append-only log or similarly trusted store.
2. After deletion, :func:`certify_unlearning` receives only the retained statistic records, the manifest,
   the excluded shard IDs, and that externally retained digest.  It verifies record integrity and re-reduces
   only retained statistics in canonical ID order.  It never receives or rereads raw shards or excluded
   statistics.

Threat model
------------
The certificate detects accidental corruption or post-ingestion modification of retained statistic records,
provided the expected manifest digest is held outside the mutable record store.  It attests that the fitted
model is the deterministic result of the committed retained sufficient statistics for one of the explicitly
audited single-step estimator classes.  It does *not* prove physical erasure, protect an anchor that an attacker
can also rewrite, prove that ingestion computed truthful statistics, or certify absence of information in
other artifacts (logs, checkpoints, backups, or downstream models).  Source digests can enable dictionary
attacks on low-entropy shards and therefore belong in a protected audit store.

Iterative or latent estimators are refused.  Their sufficient statistics generally depend on an earlier
parameter trajectory, so re-reducing those statistics is not an exact never-saw-it refit.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_SCHEMA_VERSION = "commitment-rereduce-v1"
_SUPPORTED_ESTIMATORS = frozenset(
    {
        "mixle.stats.univariate.continuous.gaussian.GaussianEstimator",
        "mixle.stats.univariate.discrete.categorical.CategoricalEstimator",
        "mixle.stats.univariate.discrete.poisson.PoissonEstimator",
    }
)


def _type_id(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _frame(tag: bytes, payload: bytes = b"") -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload


def _canonical_bytes(value: Any, active: set[int] | None = None) -> bytes:
    """Encode supported state with type, shape, dtype, and exact scalar/array bits."""
    if active is None:
        active = set()
    if value is None:
        return _frame(b"N")
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        payload = _frame(b"D", scalar.dtype.str.encode("ascii"))
        if scalar.dtype.hasobject:
            payload += _canonical_bytes(scalar.item(), active)
        else:
            payload += _frame(b"V", scalar.tobytes())
        return _frame(b"G", payload)
    if isinstance(value, bool):
        return _frame(b"B", b"\x01" if bool(value) else b"\x00")
    if isinstance(value, int):
        return _frame(b"I", str(int(value)).encode("ascii"))
    if isinstance(value, float):
        return _frame(b"F", struct.pack(">d", float(value)))
    if isinstance(value, complex):
        number = complex(value)
        return _frame(b"C", struct.pack(">dd", number.real, number.imag))
    if isinstance(value, str):
        return _frame(b"S", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _frame(b"Y", value)
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic state is not supported by the unlearning commitment format")
        active.add(identity)
        try:
            descriptor = _frame(b"D", value.dtype.str.encode("ascii")) + _canonical_bytes(tuple(value.shape), active)
            if value.dtype.hasobject:
                payload = b"".join(_canonical_bytes(item, active) for item in value.reshape(-1).tolist())
            else:
                payload = np.ascontiguousarray(value).tobytes(order="C")
            return _frame(b"A", descriptor + _frame(b"V", payload))
        finally:
            active.remove(identity)

    identity = id(value)
    if identity in active:
        raise TypeError("cyclic state is not supported by the unlearning commitment format")
    active.add(identity)
    try:
        if isinstance(value, tuple):
            return _frame(b"T", b"".join(_canonical_bytes(item, active) for item in value))
        if isinstance(value, list):
            return _frame(b"L", b"".join(_canonical_bytes(item, active) for item in value))
        if isinstance(value, (set, frozenset)):
            items = sorted(_canonical_bytes(item, active) for item in value)
            return _frame(b"E", b"".join(items))
        if isinstance(value, Mapping):
            items = sorted(
                (_canonical_bytes(key, active), _canonical_bytes(item, active)) for key, item in value.items()
            )
            return _frame(b"M", b"".join(_frame(b"K", key) + _frame(b"V", item) for key, item in items))
        if hasattr(value, "__dict__"):
            payload = _frame(b"Q", _type_id(value).encode("utf-8")) + _canonical_bytes(vars(value), active)
            return _frame(b"O", payload)
    finally:
        active.remove(identity)
    raise TypeError(f"unsupported value in unlearning commitment: {_type_id(value)}")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_shard_id(shard_id: Any) -> str:
    if not isinstance(shard_id, str) or not shard_id:
        raise ValueError("shard IDs must be non-empty strings")
    return shard_id


def _estimator_id(estimator: Any) -> str:
    estimator_id = _type_id(estimator)
    if estimator_id not in _SUPPORTED_ESTIMATORS:
        raise TypeError(
            f"{estimator_id} is not an audited additive single-step estimator; "
            "iterative, latent, subclassed, and unregistered estimators cannot receive this certificate"
        )
    return estimator_id


def _estimator_config_digest(estimator: Any) -> str:
    return _digest((_type_id(estimator), vars(estimator)))


@dataclass(frozen=True)
class StoredShard:
    """Committed sufficient statistics for one raw shard.

    ``source_digest`` binds this record to the exact raw values seen during trusted ingestion without retaining
    those values here. ``commitment`` binds the shard ID, count, source digest, and full typed statistic state.
    """

    shard_id: str
    n: float
    value: Any
    source_digest: str
    commitment: str


@dataclass(frozen=True)
class ShardCommitment:
    """Manifest entry retained after the corresponding statistic record may be deleted."""

    shard_id: str
    commitment: str


@dataclass(frozen=True)
class UnlearningManifest:
    """Immutable ingestion manifest whose digest is anchored outside the mutable statistic store."""

    schema_version: str
    estimator_id: str
    estimator_config_digest: str
    shards: tuple[ShardCommitment, ...]
    digest: str


@dataclass(frozen=True)
class UnlearningCertificate:
    """Receipt for a commitment-verified, retained-statistics-only unlearning operation."""

    bitwise_exact: bool
    method: str
    estimator_id: str
    manifest_digest: str
    retained_state_digest: str
    model_state_digest: str
    excluded_ids: tuple[str, ...]
    n_excluded: int
    n_retained_shards: int
    n_shards_total: int
    raw_data_accessed: bool
    excluded_statistics_accessed: bool
    guarantee: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bitwise_exact": self.bitwise_exact,
            "method": self.method,
            "estimator_id": self.estimator_id,
            "manifest_digest": self.manifest_digest,
            "retained_state_digest": self.retained_state_digest,
            "model_state_digest": self.model_state_digest,
            "excluded_ids": list(self.excluded_ids),
            "n_excluded": self.n_excluded,
            "n_retained_shards": self.n_retained_shards,
            "n_shards_total": self.n_shards_total,
            "raw_data_accessed": self.raw_data_accessed,
            "excluded_statistics_accessed": self.excluded_statistics_accessed,
            "guarantee": self.guarantee,
            "note": self.note,
        }


def _record_commitment(shard_id: str, n: float, value: Any, source_digest: str) -> str:
    return _digest((_SCHEMA_VERSION, shard_id, n, source_digest, value))


def shard_statistic(estimator: Any, shard: Any, *, shard_id: str) -> StoredShard:
    """Ingest one raw shard into a committed statistic record.

    This is the only public operation in this module that accepts raw data.
    """
    _estimator_id(estimator)
    shard_id = _validate_shard_id(shard_id)
    rows = list(shard)
    if not rows:
        raise ValueError("raw shards must contain at least one observation")
    encoder = estimator.accumulator_factory().make().acc_to_encoder()
    accumulator = estimator.accumulator_factory().make()
    accumulator.seq_update(encoder.seq_encode(rows), np.ones(len(rows)), None)
    value = copy.deepcopy(accumulator.value())
    n = float(len(rows))
    source_digest = _digest(rows)
    return StoredShard(
        shard_id=shard_id,
        n=n,
        value=value,
        source_digest=source_digest,
        commitment=_record_commitment(shard_id, n, value, source_digest),
    )


def _manifest_digest(
    estimator_id: str,
    estimator_config_digest: str,
    shards: tuple[ShardCommitment, ...],
) -> str:
    entries = tuple((entry.shard_id, entry.commitment) for entry in shards)
    return _digest((_SCHEMA_VERSION, estimator_id, estimator_config_digest, entries))


def prepare_unlearning(
    estimator: Any,
    shards: Mapping[str, Any] | Sequence[Any],
) -> tuple[dict[str, StoredShard], UnlearningManifest]:
    """Create committed statistic records and an externally anchorable manifest while raw data exists."""
    estimator_id = _estimator_id(estimator)
    if isinstance(shards, Mapping):
        items = list(shards.items())
    else:
        items = [(f"shard-{index:08d}", shard) for index, shard in enumerate(shards)]
    if not items:
        raise ValueError("prepare_unlearning requires at least one shard")
    ids = [_validate_shard_id(shard_id) for shard_id, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError("shard IDs must be unique")

    records = {
        shard_id: shard_statistic(estimator, shard, shard_id=shard_id)
        for shard_id, shard in sorted(items, key=lambda item: item[0])
    }
    commitments = tuple(
        ShardCommitment(shard_id=shard_id, commitment=record.commitment) for shard_id, record in records.items()
    )
    config_digest = _estimator_config_digest(estimator)
    manifest = UnlearningManifest(
        schema_version=_SCHEMA_VERSION,
        estimator_id=estimator_id,
        estimator_config_digest=config_digest,
        shards=commitments,
        digest=_manifest_digest(estimator_id, config_digest, commitments),
    )
    return records, manifest


def retained_records(
    records: Mapping[str, StoredShard],
    *,
    exclude: Any,
) -> dict[str, StoredShard]:
    """Return a new store without excluded statistic records; invalid IDs are rejected."""
    excluded = _normalize_excluded(exclude)
    unknown = excluded.difference(records)
    if unknown:
        raise ValueError(f"unknown exclusion IDs: {sorted(unknown)!r}")
    return {shard_id: record for shard_id, record in records.items() if shard_id not in excluded}


def _normalize_excluded(exclude: Any) -> set[str]:
    if isinstance(exclude, (str, bytes)):
        raise TypeError("exclude must be an iterable of shard IDs, not one string")
    try:
        excluded = set(exclude)
    except TypeError as exc:
        raise TypeError("exclude must be an iterable of shard IDs") from exc
    if not excluded:
        raise ValueError("exclude must identify at least one shard")
    for shard_id in excluded:
        _validate_shard_id(shard_id)
    return excluded


def _validate_manifest(
    estimator: Any,
    retained: Mapping[str, StoredShard],
    manifest: UnlearningManifest,
    excluded: set[str],
    expected_manifest_digest: str,
) -> tuple[str, ...]:
    estimator_id = _estimator_id(estimator)
    if not isinstance(manifest, UnlearningManifest):
        raise TypeError("manifest must be an UnlearningManifest")
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported unlearning manifest schema {manifest.schema_version!r}")
    if manifest.estimator_id != estimator_id:
        raise ValueError("manifest estimator type does not match the requested estimator")
    if manifest.estimator_config_digest != _estimator_config_digest(estimator):
        raise ValueError("manifest estimator configuration does not match the requested estimator")
    if not _is_digest(expected_manifest_digest):
        raise ValueError("expected_manifest_digest must be the externally retained SHA-256 hex digest")

    if any(not isinstance(entry, ShardCommitment) for entry in manifest.shards):
        raise TypeError("manifest shards must be ShardCommitment entries")
    canonical_shards = tuple(sorted(manifest.shards, key=lambda entry: entry.shard_id))
    ids = tuple(entry.shard_id for entry in canonical_shards)
    if (
        len(set(ids)) != len(ids)
        or any(not isinstance(shard_id, str) or not shard_id for shard_id in ids)
        or any(not _is_digest(entry.commitment) for entry in canonical_shards)
    ):
        raise ValueError("manifest shard IDs must be unique non-empty strings")
    computed_manifest_digest = _manifest_digest(
        manifest.estimator_id,
        manifest.estimator_config_digest,
        canonical_shards,
    )
    if manifest.digest != computed_manifest_digest or manifest.digest != expected_manifest_digest:
        raise ValueError("manifest integrity check failed against the external digest anchor")

    unknown = excluded.difference(ids)
    if unknown:
        raise ValueError(f"unknown exclusion IDs: {sorted(unknown)!r}")
    if not isinstance(retained, Mapping):
        raise TypeError("retained must be a mapping from shard ID to StoredShard")
    if any(not isinstance(shard_id, str) for shard_id in retained):
        raise ValueError("retained statistic store keys must be shard ID strings")
    expected_retained = set(ids).difference(excluded)
    if set(retained) != expected_retained:
        missing = sorted(expected_retained.difference(retained))
        unexpected = sorted(set(retained).difference(expected_retained))
        raise ValueError(
            "retained statistic store must contain exactly the non-excluded manifest IDs; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if not retained:
        raise ValueError("at least one committed shard must remain after deletion")

    commitments = {entry.shard_id: entry.commitment for entry in canonical_shards}
    for shard_id in sorted(retained):
        record = retained[shard_id]
        if not isinstance(record, StoredShard) or record.shard_id != shard_id:
            raise ValueError(f"retained record {shard_id!r} has an invalid identity")
        if (
            isinstance(record.n, bool)
            or not isinstance(record.n, (int, float))
            or not math.isfinite(record.n)
            or record.n <= 0
        ):
            raise ValueError(f"retained record {shard_id!r} has an invalid observation count")
        if not _is_digest(record.source_digest) or not _is_digest(record.commitment):
            raise ValueError(f"retained record {shard_id!r} has invalid digest fields")
        computed = _record_commitment(record.shard_id, record.n, record.value, record.source_digest)
        if record.commitment != computed or commitments[shard_id] != computed:
            raise ValueError(f"retained record {shard_id!r} failed its commitment check")
    return tuple(sorted(expected_retained))


def _reduce(estimator: Any, retained: Mapping[str, StoredShard], ordered_ids: tuple[str, ...]) -> tuple[float, Any]:
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0
    for shard_id in ordered_ids:
        record = retained[shard_id]
        accumulator.combine(copy.deepcopy(record.value))
        nobs += record.n
    return nobs, accumulator.value()


def _fit_retained(
    estimator: Any,
    retained: Mapping[str, StoredShard],
    ordered_ids: tuple[str, ...],
) -> tuple[Any, str]:
    nobs, value = _reduce(estimator, retained, ordered_ids)
    model = estimator.estimate(nobs, copy.deepcopy(value))
    return model, _digest((nobs, value))


def _model_state_digest(model: Any) -> str:
    """Hash the exact typed bits of the fitted model's complete instance state."""
    if not hasattr(model, "__dict__"):
        raise TypeError("certified fitted models must expose instance state")
    return _digest((_type_id(model), vars(model)))


def unlearn(
    estimator: Any,
    retained: Mapping[str, StoredShard],
    *,
    manifest: UnlearningManifest,
    exclude: Any,
    expected_manifest_digest: str,
) -> Any:
    """Fit from commitment-verified retained statistics without reading deleted data or statistics."""
    excluded = _normalize_excluded(exclude)
    ordered_ids = _validate_manifest(estimator, retained, manifest, excluded, expected_manifest_digest)
    model, _ = _fit_retained(estimator, retained, ordered_ids)
    return model


def certify_unlearning(
    estimator: Any,
    retained: Mapping[str, StoredShard],
    *,
    manifest: UnlearningManifest,
    exclude: Any,
    expected_manifest_digest: str,
) -> tuple[Any, UnlearningCertificate]:
    """Certify deterministic re-reduction from committed retained statistics only.

    The estimator is executed twice from the same verified retained state.  Equality compares typed binary
    sufficient-statistic state and complete fitted-model instance state, not JSON or ``repr`` output.
    """
    excluded = _normalize_excluded(exclude)
    ordered_ids = _validate_manifest(estimator, retained, manifest, excluded, expected_manifest_digest)
    estimator_digest_before = _estimator_config_digest(estimator)
    first_model, first_retained_digest = _fit_retained(estimator, retained, ordered_ids)
    first_model_digest = _model_state_digest(first_model)
    second_model, second_retained_digest = _fit_retained(estimator, retained, ordered_ids)
    second_model_digest = _model_state_digest(second_model)
    estimator_stable = estimator_digest_before == _estimator_config_digest(estimator)
    exact = (
        first_retained_digest == second_retained_digest
        and first_model_digest == second_model_digest
        and estimator_stable
    )
    guarantee = (
        "The model is the deterministic single-step fit of exactly the committed retained sufficient "
        "statistics in canonical shard-ID order."
    )
    note = (
        "Commitments and repeated typed-state digests agree; excluded raw data and excluded statistics were "
        "not inputs to certification."
        if exact
        else "Repeated retained-state reduction or model-state estimation was not bitwise deterministic."
    )
    certificate = UnlearningCertificate(
        bitwise_exact=exact,
        method=_SCHEMA_VERSION,
        estimator_id=manifest.estimator_id,
        manifest_digest=manifest.digest,
        retained_state_digest=first_retained_digest,
        model_state_digest=first_model_digest,
        excluded_ids=tuple(sorted(excluded)),
        n_excluded=len(excluded),
        n_retained_shards=len(ordered_ids),
        n_shards_total=len(manifest.shards),
        raw_data_accessed=False,
        excluded_statistics_accessed=False,
        guarantee=guarantee,
        note=note,
    )
    return first_model, certificate
