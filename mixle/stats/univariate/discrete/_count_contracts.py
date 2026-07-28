"""Shared validation for count observations, weights, and cached law parameters."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


class CachedParameterLaw:
    """Keep derived parameter caches consistent with the parameters they came from.

    Several count families validate a parameter in ``__init__`` and then cache
    derived quantities (logs, normalizers, gamma terms) that every scorer reads.
    Assigning the parameter afterwards used to leave those caches stale, so the
    sampler drew from the new parameter while scalar, sequence, and backend
    scoring all kept using the old cache.

    A subclass lists its parameters in ``_cached_parameters`` and performs the
    validation plus the whole cache computation in ``_rebuild_parameter_caches``.
    Constructors set the raw parameters, call the rebuild once, and then set
    ``_parameter_caches_ready``; assigning any listed parameter afterwards re-runs
    that same validated rebuild, so the parameters and their caches can never
    disagree. Writes that bypass ``__setattr__`` entirely -- pickling, ``deepcopy``
    and the inference-transaction rollback all restore ``__dict__`` directly --
    are untouched because they replay a consistent snapshot.
    """

    _cached_parameters: tuple[str, ...] = ()

    def _rebuild_parameter_caches(self) -> None:
        """Validate this law's parameters and rebuild every derived cache."""
        raise NotImplementedError("cached parameter laws must implement _rebuild_parameter_caches.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, rebuilding derived caches when a parameter changes."""
        if not (name in self._cached_parameters and self.__dict__.get("_parameter_caches_ready", False)):
            object.__setattr__(self, name, value)
            return
        previous = self.__dict__.copy()
        object.__setattr__(self, name, value)
        try:
            self._rebuild_parameter_caches()
        except BaseException:
            # A rejected parameter must leave the law exactly as it was rather than
            # stranding it with a new parameter beside its old caches.
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise


def exact_integer_observations(
    values: Sequence[Any] | np.ndarray,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    """Return an exact int64 array after validating every observation."""
    raw = np.asarray(values, dtype=object)
    encoded = np.empty(raw.shape, dtype=np.int64)
    for index, value in np.ndenumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be exact integers, not booleans.")
        if isinstance(value, numbers.Integral):
            integer = int(value)
        elif isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{label} must be finite exact integers.")
            integer = int(numeric)
        else:
            raise ValueError(f"{label} must be finite exact integers.")
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise ValueError(f"{label} must fit in signed 64-bit integer storage.")
        if minimum is not None and integer < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        if maximum is not None and integer > maximum:
            raise ValueError(f"{label} must be at most {maximum}.")
        encoded[index] = integer
    return encoded


def nonnegative_weights(values: Any, *, shape: tuple[int, ...], label: str = "weights") -> np.ndarray:
    """Return finite, non-negative float64 weights aligned to an observation array."""
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != shape:
        raise ValueError(f"{label} must have the same shape as the encoded observations.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{label} must be finite and non-negative.")
    return weights
