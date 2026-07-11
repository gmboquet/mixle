"""One mechanism for deprecating public names, implementing the policy in ``docs/deprecation-policy.rst``.

A deprecated stable name must keep working *and* tell the caller where to go, in a category tools can act
on. This module is the single place that emits that signal, so every deprecation in the library speaks
with one warning category (``DeprecationWarning``) and one message format::

    <old> is deprecated since mixle <since>; use <new> instead. It will be removed in mixle <removed_in>.

Use :func:`deprecated_alias` to wrap a thin forwarding method or function; use :func:`warn_deprecated`
directly for finer-grained cases (a deprecated argument value, a deprecated code path). The ``stacklevel``
is set so the warning points at the *caller's* line, not at this module.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["warn_deprecated", "deprecated_alias", "deprecation_message"]

F = TypeVar("F", bound=Callable[..., Any])


def deprecation_message(old: str, new: str, *, since: str, removed_in: str | None = None) -> str:
    """Return the canonical deprecation message (the one format every mixle deprecation uses)."""
    tail = f" It will be removed in mixle {removed_in}." if removed_in else ""
    return f"{old} is deprecated since mixle {since}; use {new} instead.{tail}"


def warn_deprecated(old: str, new: str, *, since: str, removed_in: str | None = None, stacklevel: int = 1) -> None:
    """Emit the standard ``DeprecationWarning`` for using ``old`` instead of ``new``.

    ``stacklevel`` counts caller frames above this function: the default (1) attributes the warning to the
    immediate caller. :func:`deprecated_alias` passes 2 to skip its own wrapper frame so the warning still
    lands on the user's call site.
    """
    warnings.warn(
        deprecation_message(old, new, since=since, removed_in=removed_in),
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def deprecated_alias(
    new: str, *, since: str, removed_in: str | None = None, old: str | None = None
) -> Callable[[F], F]:
    """Mark a thin forwarding method/function as a deprecated alias for ``new``.

    The wrapped callable's body is unchanged (it should just forward to the canonical name); calling it
    emits the standard warning first, attributed to the caller. ``old`` defaults to the wrapped callable's
    qualified name (e.g. ``FooEstimator.accumulatorFactory``).
    """

    def decorate(func: F) -> F:
        label = old if old is not None else func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(label, new, since=since, removed_in=removed_in, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
