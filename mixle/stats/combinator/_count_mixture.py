"""Shared validation for hurdle and zero-inflated count combinators."""

import math
import operator
from typing import Any

import numpy as np

from mixle.capability import Discrete, supports
from mixle.stats.compute.declarations import declaration_for

_COUNT_SUPPORTS = frozenset(
    {
        "boolean",
        "bounded_integer",
        "bounded_integer_spike",
        "non_negative_integer",
        "positive_integer",
    }
)


def count_value_kind(value: Any) -> int:
    """Return ``0`` for zero, ``1`` for a positive integer, and ``-1`` otherwise."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -1
    if not math.isfinite(numeric) or numeric < 0.0 or math.floor(numeric) != numeric:
        return -1
    return 0 if numeric == 0.0 else 1


def require_count_value(value: Any, *, path: str) -> int:
    """Classify one non-negative integer observation or raise a typed contract error."""
    kind = count_value_kind(value)
    if kind < 0:
        raise ValueError("%s must be a finite non-negative integer." % path)
    return kind


def require_count_base(
    base: Any,
    *,
    model: str,
    require_zero_atom: bool,
    require_positive_mass: bool,
) -> float:
    """Return the declared base log-mass at zero after proving count/atomic semantics."""
    if not supports(base, Discrete):
        raise TypeError("%s requires a discrete atomic base distribution." % model)
    declaration = declaration_for(base)
    support = None if declaration is None else declaration.support
    if support == "fixed_atom":
        if count_value_kind(getattr(base, "value", None)) < 0:
            raise TypeError("%s requires a base supported on non-negative integer counts." % model)
    elif support not in _COUNT_SUPPORTS:
        raise TypeError(
            "%s requires a base declaring non-negative integer count support; got %r."
            % (model, support)
        )
    try:
        log_p0 = float(base.log_density(0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s base must report a scalar log-probability at zero." % model) from exc
    if math.isnan(log_p0) or log_p0 > 0.0:
        raise ValueError("%s base zero log-probability must be in [-inf, 0]." % model)
    if require_zero_atom and log_p0 == -math.inf:
        raise ValueError("%s base must assign positive probability mass to zero." % model)
    if require_positive_mass and log_p0 == 0.0:
        raise ValueError("%s base must retain positive probability mass above zero." % model)
    return log_p0


def require_probability(value: Any, *, name: str) -> float:
    """Return a finite probability in the closed unit interval."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a finite real probability." % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a finite real probability." % name) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("%s must be in [0, 1]." % name)
    return result


def require_nonnegative(value: Any, *, name: str) -> float:
    """Return a finite non-negative scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a finite non-negative real number." % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a finite non-negative real number." % name) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be a finite non-negative real number." % name)
    return result


def require_positive_integer(value: Any, *, name: str) -> int:
    """Return a positive integer control value."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive integer." % name)
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("%s must be a positive integer." % name) from exc
    if result <= 0:
        raise ValueError("%s must be a positive integer." % name)
    return int(result)


def require_positive(value: Any, *, name: str) -> float:
    """Return a finite strictly positive scalar control value."""
    result = require_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError("%s must be strictly positive." % name)
    return result


def require_weights(weights: Any, size: int, *, path: str) -> np.ndarray:
    """Return a one-dimensional finite non-negative weight array of the expected size."""
    result = np.asarray(weights, dtype=np.float64)
    if result.ndim != 1 or result.shape[0] != size:
        raise ValueError("%s must be a one-dimensional array with %d entries." % (path, size))
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must contain only finite non-negative values." % path)
    return result


def require_component_counts(component: Any, total: Any, *, name: str) -> tuple[float, float]:
    """Validate one latent/structural component count against its outer total."""
    checked_component = require_nonnegative(component, name=name)
    checked_total = require_nonnegative(total, name="total weight")
    if checked_component > checked_total:
        raise ValueError("%s cannot exceed total weight." % name)
    return checked_component, checked_total


def boundary_preserving_rate(component: float, total: float, pseudo_count: float | None) -> float:
    """Estimate a component rate, shrinking only non-deterministic empirical laws."""
    if total == 0.0:
        return 0.5 if pseudo_count else 0.0
    if component == 0.0:
        return 0.0
    if component == total:
        return 1.0
    if pseudo_count:
        return (component + 0.5 * pseudo_count) / (total + pseudo_count)
    return component / total
