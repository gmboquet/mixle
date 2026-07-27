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

    tolerance = 1.0e-9 * max(1.0, declared, assigned)
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


__all__ = [
    "EffectiveSampleReceipt",
    "validated_positive_integer",
    "validated_observation_weight",
    "validated_observation_weights",
    "validated_count_array",
    "validated_statistic_tuple",
    "validate_effective_sample_mass",
    "validated_weighted_responsibilities",
]
