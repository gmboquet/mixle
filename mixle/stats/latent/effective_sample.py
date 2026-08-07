"""Shared effective-sample and sufficient-statistic contracts for latent models.

An outer observation with weight ``w_i`` contributes finite non-negative effective
mass. A latent responsibility vector may distribute that mass among children, but
must never create mass. Child estimators receive their assigned posterior mass,
not the outer row count. Positive-weight observations impossible under every
latent branch may remain unassigned only when the family explicitly opts into
that support policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EffectiveSampleReceipt:
    """Validated relationship between declared and assigned effective mass."""

    declared_mass: float | None
    assigned_mass: float
    unassigned_mass: float
    allows_unassigned: bool
    schema_version: int = 1
    contract: str = "mixle.latent_effective_sample/v1"


def validated_positive_integer(value: Any, label: str) -> int:
    """Return a positive exact non-boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive exact integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def validated_observation_weight(value: Any, label: str = "observation weight") -> float:
    """Return one finite non-negative, non-boolean observation weight."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real non-boolean scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def validated_observation_weights(
    values: Any,
    rows: int,
    label: str = "observation weights",
) -> np.ndarray:
    """Return an owned finite non-negative weight vector with exact row geometry."""
    if isinstance(rows, (bool, np.bool_)) or not isinstance(rows, (int, np.integer)):
        raise TypeError("rows must be a non-negative exact integer")
    rows = int(rows)
    if rows < 0:
        raise ValueError("rows must be non-negative")
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{label} must contain real non-boolean values")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a real vector") from exc
    if result.shape != (rows,):
        raise ValueError(f"{label} must have shape ({rows},)")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return result.copy()


def validated_count_array(
    values: Any,
    shape: tuple[int, ...],
    label: str = "sufficient-statistic counts",
) -> np.ndarray:
    """Return an owned finite non-negative count array with exact geometry."""
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{label} must contain real non-boolean values")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a real array") from exc
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return result.copy()


def validated_statistic_tuple(values: Any, arity: int, label: str) -> tuple[Any, ...]:
    """Return a statistic tuple only when its public schema arity is exact."""
    if not isinstance(values, (tuple, list)) or len(values) != arity:
        raise ValueError(f"{label} must be a {arity}-item tuple")
    return tuple(values)


def validate_effective_sample_mass(
    declared_mass: Any,
    assigned_mass: Any,
    *,
    label: str,
    allow_unassigned: bool = False,
) -> EffectiveSampleReceipt:
    """Validate declared outer mass against mass assigned to latent children.

    ``declared_mass=None`` means the caller did not provide outer-mass metadata.
    Otherwise exact agreement is required, except that families whose documented
    impossible-evidence policy assigns zero responsibility may set
    ``allow_unassigned=True``. Assigned mass may never exceed declared mass.
    """
    assigned = validated_observation_weight(assigned_mass, f"{label} assigned mass")
    declared = None if declared_mass is None else validated_observation_weight(declared_mass, f"{label} declared mass")
    if declared is None:
        return EffectiveSampleReceipt(None, assigned, 0.0, allow_unassigned)

    # Relative tolerance. The 1e-9 floor is a float64 figure: responsibilities summed on a float32
    # engine (the torch/MPS EM path) cannot agree that closely -- float32 eps alone is ~1.2e-7, and
    # accumulating n of them loses more still, so a float64-calibrated bound rejected sufficient
    # statistics that were as exact as their arithmetic allows. Scale the floor to float32 eps with
    # headroom for accumulation; a genuine mass violation is O(1) relative, orders of magnitude above.
    tolerance = 64.0 * float(np.finfo(np.float32).eps) * max(1.0, declared, assigned)
    difference = declared - assigned
    if difference < -tolerance:
        raise ValueError(f"{label} assigned mass cannot exceed declared mass")
    unassigned = max(0.0, difference)
    if not allow_unassigned and unassigned > tolerance:
        raise ValueError(f"{label} assigned mass must equal declared mass")
    return EffectiveSampleReceipt(declared, assigned, unassigned, allow_unassigned)


def validated_weighted_responsibilities(
    values: Any,
    weights: Any,
    components: int,
    *,
    label: str,
    allow_unassigned: bool = False,
) -> np.ndarray:
    """Validate an owned ``(rows, components)`` matrix of assigned posterior mass."""
    if isinstance(components, (bool, np.bool_)) or not isinstance(components, (int, np.integer)):
        raise TypeError("components must be a positive exact integer")
    components = int(components)
    if components <= 0:
        raise ValueError("components must be positive")
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional matrix")
    outer = validated_observation_weights(weights, raw.shape[0], f"{label} outer weights")
    assigned = validated_count_array(raw, (raw.shape[0], components), label)
    row_mass = assigned.sum(axis=1)
    tolerance = 1.0e-9 * np.maximum(1.0, outer)
    if np.any(row_mass - outer > tolerance):
        raise ValueError(f"{label} rows cannot assign more than their outer weight")
    if not allow_unassigned and np.any(outer - row_mass > tolerance):
        raise ValueError(f"{label} rows must assign all outer weight")
    return assigned


def require_finite_count_totals(named_arrays: Any, *, label: str) -> None:
    """Every element AND every aggregate must be finite.

    Validating only what ARRIVES is not enough: two individually valid statistics whose elements
    are each 4.6e307 combine to finite elements with an infinite total, a valid statistic scaled by
    a valid factor of 3.0 overflows outright, and keyed pooling reaches the same state through
    addition (STAT-RR8-1). Finiteness has to be a postcondition of every public mutator, not a
    precondition of ingestion, so this runs on the RESULT and each mutator rolls back if it fails.
    This is the canonical home; the chain/tree HMM implementation aliases it, and the latent-family
    mutator audit applies it family-wide.
    """
    for name, array in named_arrays:
        values = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} {name} must contain only finite values")
        if not np.isfinite(float(values.sum())):
            raise ValueError(f"{label} {name} must aggregate to a finite total")


def snapshot_accumulator_statistics(
    acc: Any,
    *,
    count_attrs: Any = (),
    child_attrs: Any = (),
    single_child_attrs: Any = (),
) -> dict[str, Any]:
    """Snapshot the named count arrays, child-accumulator lists, and single children of ``acc``.

    Taken by a mutator BEFORE it changes anything, so a failure at any later point can hand the
    snapshot to :func:`restore_accumulator_statistics` and put the accumulator back exactly as it
    was -- counts, children, and the identity of the child containers included. Child lists are
    captured as (container reference, child references, serialized values): the container
    reference matters because keyed replacement can REBIND the attribute to an adopted list, and
    rollback must restore the original object, not a copy.

    Attributes the object does not carry are skipped: the keyed-pooling protocol tests build
    accumulators via ``__new__`` with only the fields under test, and an attribute an object
    lacks cannot need restoring either.

    Count arrays are captured as (original object, value copy) pairs, not value copies alone:
    mutators change counts IN PLACE (``+=``, ``*=``), so a rollback that merely rebinds the
    attribute to a pristine copy abandons the mutated original -- an external alias to the
    count array then keeps the doubled values while the attribute looks restored, and the
    attribute's identity changes (STAT-RR11-3). Restoration writes the values back INTO the
    original object and rebinds that original.
    """
    counts = {}
    for name in count_attrs:
        if not hasattr(acc, name):
            continue
        original = getattr(acc, name)
        counts[name] = (original, np.asarray(original).copy())
    children = {}
    for name in child_attrs:
        if not hasattr(acc, name):
            continue
        container = getattr(acc, name)
        members = list(container)
        children[name] = (container, members, [child.value() for child in members])
    singles = {}
    for name in single_child_attrs:
        if not hasattr(acc, name):
            continue
        child = getattr(acc, name)
        singles[name] = (child, None if child is None else child.value())
    return {"counts": counts, "children": children, "singles": singles}


def restore_accumulator_statistics(acc: Any, snapshot: dict[str, Any]) -> None:
    """Best-effort rollback of a failed mutator from a prior snapshot.

    Count attributes are healed by writing the snapshotted values back INTO the original array
    object and rebinding that original -- both an external alias and the attribute itself then
    observe the rollback, with identity preserved (STAT-RR11-3: rebinding a copy left a
    caller-held alias doubled after a rejected in-place ``*=``/``+=``). Child containers are
    rebound to their ORIGINAL objects and every child restored through ``from_value``; single
    children (length accumulators and kin) likewise. Restoration errors are deliberately
    suppressed: the original rejection is the signal, and a failure while rolling back must
    not mask it.
    """
    for name, (original, values) in snapshot["counts"].items():
        restored = original
        try:
            if isinstance(original, np.ndarray) and original.shape == values.shape:
                original[...] = values
            else:
                restored = values
        except Exception:  # noqa: BLE001, S110 - see docstring
            restored = values
        setattr(acc, name, restored)
    for name, (container, members, values) in snapshot["children"].items():
        setattr(acc, name, container)
        for child, value in zip(members, values):
            try:
                child.from_value(value)
            except Exception:  # noqa: BLE001, S110 - see docstring
                pass
    for name, (child, value) in snapshot["singles"].items():
        setattr(acc, name, child)
        if child is not None and value is not None:
            try:
                child.from_value(value)
            except Exception:  # noqa: BLE001, S110 - see docstring
                pass


def _healed_in_place(current: Any, snap_value: Any) -> Any:
    """Write ``snap_value`` back INTO ``current`` where possible, else return the snapshot.

    Arrays are restored element-wise, lists element-by-element, and accumulator objects through
    ``from_value`` of the snapshot's ``value()`` -- each preserves the identity of the object a
    caller may already hold a reference to. Healing errors fall back to returning the snapshot
    object itself (mapping-level restoration), and are suppressed so a rollback failure cannot
    mask the original rejection.
    """
    try:
        if isinstance(current, np.ndarray) and isinstance(snap_value, np.ndarray) and current.shape == snap_value.shape:
            current[...] = snap_value
            return current
        if isinstance(current, list) and isinstance(snap_value, list) and len(current) == len(snap_value):
            for i in range(len(current)):
                current[i] = _healed_in_place(current[i], snap_value[i])
            return current
        if isinstance(current, tuple) and isinstance(snap_value, tuple) and len(current) == len(snap_value):
            return tuple(_healed_in_place(c, s) for c, s in zip(current, snap_value))
        if hasattr(current, "from_value") and hasattr(snap_value, "value"):
            current.from_value(snap_value.value())
            return current
    except Exception:  # noqa: BLE001, S110 - see docstring
        pass
    return snap_value


def heal_pooled_statistics(stats_dict: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Restore a keyed-statistics mapping after a failed mutator, healing in-place damage.

    Swapping restored COPIES into the mapping is not enough: merges mutate pooled arrays with
    ``+=`` and pooled emission accumulators with ``combine``, so state reachable through a
    pre-existing reference -- a caller's alias to a pooled array, the shared child list a tied
    site adopted -- stayed corrupted while the mapping itself looked restored (STAT-RR10-1).
    Entries the failed mutator added are removed; surviving entries are healed in place via
    :func:`_healed_in_place` so external aliases observe the rollback too. Take the snapshot
    with ``copy.deepcopy(stats_dict)`` before the first mutation.
    """
    for key in [k for k in stats_dict if k not in snapshot]:
        del stats_dict[key]
    for key, snap_value in snapshot.items():
        stats_dict[key] = _healed_in_place(stats_dict.get(key), snap_value)


__all__ = [
    "EffectiveSampleReceipt",
    "validated_positive_integer",
    "validated_observation_weight",
    "validated_observation_weights",
    "validated_count_array",
    "validated_statistic_tuple",
    "validate_effective_sample_mass",
    "validated_weighted_responsibilities",
    "require_finite_count_totals",
    "snapshot_accumulator_statistics",
    "restore_accumulator_statistics",
    "heal_pooled_statistics",
]
