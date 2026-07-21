"""Introspection helpers for calling a caller-supplied callable whose exact signature is unknown."""

import inspect
from collections.abc import Callable
from typing import Any


def accepts_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """True iff fn's signature can accept being called as ``fn(*args, **kwargs)``, checked without
    actually calling fn.

    Use this to choose between an optional richer call and a reduced fallback call BEFORE invoking
    either -- not by trying the richer call and catching the TypeError it might raise. A TypeError
    raised by fn's own implementation (as opposed to a genuine signature mismatch) looks identical
    to "the richer call isn't supported" from the outside, so catching it and silently retrying
    with the reduced call duplicates whatever fn's call does (a remote request, a random draw, an
    expensive computation) and masks the real error.

    Some callables (certain builtins, C-extension callables, some `functools.partial` chains) do
    not support signature introspection at all; ``inspect.signature`` raises ``ValueError`` for
    those. This conservatively returns True in that case (assume the richer call is supported) so
    behavior for a callable this cannot actually check falls through to attempting it, matching
    what unconditionally trying the richer call first would have done anyway.

    Args:
        fn: The callable to check.
        *args, **kwargs: The arguments the richer call would pass.

    Returns:
        True if the signature accepts this call shape, else False.

    """
    try:
        inspect.signature(fn).bind(*args, **kwargs)
        return True
    except TypeError:
        return False
    except ValueError:
        return True
