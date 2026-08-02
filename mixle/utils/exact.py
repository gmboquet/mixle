"""Exact scalar validation for public option flags -- no truthiness coercion.

``bool("false")`` is ``True``. Any public boundary that reads a caller-supplied flag with
``bool(value)`` or a bare ``if value:`` therefore ENABLES the feature when handed the string that
names its opposite, which is exactly the form a flag arrives in when it comes from serialized
configuration, an environment variable, a CLI argument, or a JSON payload (MXR-080-1907).

Two levels, because two different things are at stake:

* :func:`require_exact_bool` -- for ordinary semantic flags (graph directedness, maximize/minimize,
  normalization, approximation permission). A non-Boolean is a caller error, not something to coerce,
  because coercion silently changes what the computation means.
* :func:`require_explicit_true` -- for a gate whose "yes" authorizes code execution or another
  irreversible act. Here even ``True``-like is not enough: the caller must pass the ``True`` singleton
  itself, so that no value which merely *evaluates* truthy can open the gate. This is the contract
  :func:`mixle.data.encoded_io.load_encoded` uses for its unpickle gate (MXR-080-1873/1881).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["require_exact_bool", "require_explicit_true"]


def require_exact_bool(value: Any, name: str) -> bool:
    """Return ``value`` as a ``bool``, or raise ``TypeError`` if it is not actually Boolean.

    ``np.bool_`` is accepted and canonicalized -- it is a genuine Boolean, and rejecting it would
    refuse values the library's own array paths produce. Everything else, including ``0``/``1`` and
    any non-empty string, is refused.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be an actual Boolean (True/False), got {type(value).__name__} ({value!r}). "
            "It is not coerced: bool('false') is True, so accepting a non-Boolean here would let a "
            "value that names the opposite of what it does enable the behaviour."
        )
    return bool(value)


def require_explicit_true(value: Any, name: str, *, because: str) -> None:
    """Raise unless ``value`` is the ``True`` singleton.

    For gates that authorize code execution or another act that cannot be taken back. Truthiness is
    not enough and neither is equality: ``1 == True`` and ``np.True_ == True``, so an ``== True``
    test still admits a value the caller never meant as consent. ``because`` states what is being
    authorized and appears in the error, so a caller who hits it learns what they would be agreeing
    to rather than only that a flag was rejected.
    """
    if value is not True:
        raise ValueError(
            f"{name} must be exactly True to proceed. {because} Got {type(value).__name__} "
            f"({value!r}); a truthy value is deliberately not enough, because bool('false') is True "
            "and a flag that arrives from configuration text must not be able to authorize this by "
            "accident."
        )
