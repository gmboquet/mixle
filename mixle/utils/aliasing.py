"""Backward-compatible keyword-argument aliasing for distribution and estimator constructors.

Some distribution constructors keep legacy positional names while also accepting descriptive public
keyword aliases. To keep existing code working while preferred spellings become canonical,
constructors accept both the legacy name and the preferred name and reconcile them with
:func:`coalesce_alias`.

Usage::

    def __init__(self, components, w=None, name=None, *, weights=None):
        w = coalesce_alias('w', w, 'weights', weights)

The preferred (alias) argument is keyword-only so it never shadows a positional argument, and the
legacy argument keeps its position. Passing both raises ``TypeError``; passing neither raises
``TypeError`` for required arguments.
"""

from typing import Any

__all__ = ["coalesce_alias", "require", "MISSING"]


class _Missing:
    """Sentinel type for "argument not supplied" so that ``None`` stays a valid explicit value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<MISSING>"


MISSING = _Missing()


def require(name: str, value: Any, *, default: Any = MISSING) -> Any:
    """Return ``value`` unless it is the not-supplied sentinel, in which case raise ``TypeError``.

    Used for required positional arguments that were given a sentinel default so an aliased
    earlier argument could become optional (which would otherwise force a non-default argument to
    follow a defaulted one).
    """
    if value is default:
        raise TypeError("missing required argument %r" % name)
    return value


def coalesce_alias(
    canonical_name: str,
    canonical_value: Any,
    alias_name: str,
    alias_value: Any,
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    """Reconcile a canonical (legacy) argument with its preferred alias.

    Args:
        canonical_name (str): Name of the legacy argument, used in error messages.
        canonical_value (Any): Value bound to the legacy argument.
        alias_name (str): Name of the preferred argument, used in error messages.
        alias_value (Any): Value bound to the preferred argument.
        required (bool): If True, raise when neither argument was supplied.
        default (Any): Sentinel marking "not supplied" for both arguments. Compared by identity,
            so the legacy argument's declared default must match this value.

    Returns:
        The supplied value, preferring the alias when both resolve to non-default (which is only
        reachable when exactly one was supplied).

    Raises:
        TypeError: If both arguments are supplied, or if neither is supplied and ``required``.

    """
    alias_given = alias_value is not default
    canonical_given = canonical_value is not default

    if alias_given and canonical_given:
        raise TypeError("%r and its alias %r are mutually exclusive; pass only one" % (canonical_name, alias_name))

    if alias_given:
        return alias_value

    if canonical_given:
        return canonical_value

    if required:
        raise TypeError("missing required argument %r (or its alias %r)" % (canonical_name, alias_name))

    return default
