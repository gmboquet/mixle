"""Degradation policy with named failure modes for receipts.

Subsystem failures should not silently serve degraded answers as ordinary
successes. :func:`with_fallback` runs a primary path and, on exception, runs a
named fallback. The result records whether the primary path succeeded or which
degradation mode produced the fallback value. :func:`abstain_on_timeout` and
:func:`route_past` apply the same receipt discipline to timeout abstention and
multi-tier routing.

The named modes are ``teacher_down`` (fall back to captured or store-only
reasoning; see :meth:`mixle.system.core.System.answer`), ``store_down`` (reason
without accumulated knowledge; see :meth:`mixle.system.core.System.ingest`),
``oracle_timeout`` (abstain or escalate rather than guess), and ``model_error``
(route past the failing tier to the next one). The timeout and routing helpers
are reusable fault-boundary primitives.

A fault boundary is for *recoverable* failures -- a provider outage, a saturated
endpoint, an unreachable store. It is not a general exception swallower.
:data:`NON_RECOVERABLE_FAULTS` names the classes that must never be degraded
into an ordinary-looking answer: a programming error (``TypeError``,
``AttributeError``, ...) means the fallback answer is built on a defect nobody
will see, and an authorization/policy failure (``PermissionError``) routed past
a fail-closed boundary is a security hole, not a degraded mode. Pass an explicit
``recoverable=`` allowlist to narrow a particular route further.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Failure classes a fault boundary must never convert into a degraded answer. Programmer errors
#: (a defect in this process, not a failing dependency), integrity/resource failures the process
#: cannot honestly continue past, and authorization/policy denials -- which a route-past would turn
#: into "the next tier answered it", silently bypassing a fail-closed boundary.
NON_RECOVERABLE_FAULTS: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    SyntaxError,
    AssertionError,
    RecursionError,
    SystemError,
    MemoryError,
    PermissionError,
)


def _is_recoverable(exc: BaseException, recoverable: Sequence[type[BaseException]] | None) -> bool:
    """Whether ``exc`` may be degraded: an explicit allowlist when given, else the default policy."""
    if recoverable is not None:
        return isinstance(exc, tuple(recoverable))
    return not isinstance(exc, NON_RECOVERABLE_FAULTS)


@dataclass(frozen=True)
class DegradedResult:
    """The outcome of a fault-boundary call: a value, whether it came from a fallback, and, if so, why.

    ``attempts`` carries the structured causal evidence for a multi-tier route: one
    ``(tier_name, exception_repr)`` pair per tier that was tried and failed, so a receipt records
    what actually went wrong rather than only that something did.
    """

    value: Any
    degraded: bool
    mode: str | None = None
    reason: str | None = None
    attempts: tuple[tuple[str, str], ...] = field(default=())

    def to_receipt_fields(self) -> dict[str, Any]:
        """Return ``degraded_mode`` and ``degraded_reason`` receipt fields."""
        return {"degraded_mode": self.mode, "degraded_reason": self.reason}


def with_fallback(
    fn: Callable[[], Any],
    fallback: Callable[[Exception], Any],
    *,
    mode: str,
    recoverable: Sequence[type[BaseException]] | None = None,
) -> DegradedResult:
    """Run ``fn()``; on a *recoverable* exception, run ``fallback(exc)`` instead and flag it under ``mode``.

    ``recoverable`` is an explicit allowlist of failure classes this route may degrade past; omit it
    for the default policy (anything except :data:`NON_RECOVERABLE_FAULTS`). A non-recoverable failure
    propagates unchanged: an internal ``TypeError`` handed back as a fallback answer hides a defect,
    and a ``PermissionError`` degraded into a successful-looking reply bypasses a fail-closed boundary
    rather than reporting that access was denied.

    If ``fallback`` itself raises, that exception propagates. A fallback that
    cannot produce a value is a real failure, not a second implicit fallback.
    """
    try:
        return DegradedResult(value=fn(), degraded=False)
    except Exception as exc:  # noqa: BLE001 -- recoverability is decided by policy, just below
        if not _is_recoverable(exc, recoverable):
            raise
        return DegradedResult(
            value=fallback(exc),
            degraded=True,
            mode=mode,
            reason=str(exc),
            attempts=((mode, repr(exc)),),
        )


def abstain_on_timeout(fn: Callable[[], Any], *, timeout_error: type[BaseException] = TimeoutError) -> DegradedResult:
    """``oracle_timeout`` mode: run ``fn()``; if it raises ``timeout_error``, abstain (``value=None``) rather
    than guess. Other exceptions propagate."""
    try:
        return DegradedResult(value=fn(), degraded=False)
    except timeout_error as exc:
        return DegradedResult(value=None, degraded=True, mode="oracle_timeout", reason=str(exc))


def route_past(
    tiers: Sequence[Callable[[], Any]],
    *,
    names: Sequence[str] | None = None,
    recoverable: Sequence[type[BaseException]] | None = None,
) -> DegradedResult:
    """``model_error`` mode: try each tier in order; a tier that fails *recoverably* is skipped (not
    fatal) in favor of the next. The result is degraded unless the first tier answers cleanly. Raises
    the last tier's exception if every tier fails.

    ``recoverable`` is an explicit allowlist of failure classes this route may skip a tier for; omit it
    for the default policy (anything except :data:`NON_RECOVERABLE_FAULTS`). A non-recoverable failure
    propagates immediately instead of being reclassified as ``model_error``: an internal ``TypeError``
    routed past becomes the next tier's ordinary answer and the defect is never seen, and skipping a
    ``PermissionError`` turns a denial into a served result from a tier the caller was not authorized
    to reach.

    Every skipped tier is recorded on :attr:`DegradedResult.attempts` as a
    ``(tier_name, exception_repr)`` pair -- the causal evidence for why the answer came from where it
    did, not merely that it was degraded."""
    if len(tiers) == 0:
        raise ValueError("route_past needs at least one tier")
    names = list(names) if names is not None else [f"tier{i}" for i in range(len(tiers))]
    if len(names) != len(tiers):
        raise ValueError(f"names has {len(names)} entries for {len(tiers)} tiers; they must match one-to-one")
    attempts: list[tuple[str, str]] = []
    last_exc: Exception | None = None
    for name, tier in zip(names, tiers, strict=True):
        try:
            value = tier()
        except Exception as exc:  # noqa: BLE001 -- recoverability is decided by policy, just below
            if not _is_recoverable(exc, recoverable):
                raise
            attempts.append((name, repr(exc)))
            last_exc = exc
            continue
        if not attempts:
            return DegradedResult(value=value, degraded=False)
        return DegradedResult(
            value=value,
            degraded=True,
            mode="model_error",
            reason=f"routed past {[n for n, _ in attempts]}",
            attempts=tuple(attempts),
        )
    raise last_exc  # every tier failed
