"""Degradation policy with named failure modes for receipts.

Subsystem failures should not silently serve degraded answers as ordinary
successes. :func:`with_fallback` runs a primary path and, on exception, runs a
named fallback. The result records whether the primary path succeeded or which
degradation mode produced the fallback value. :func:`abstain_on_timeout` and
:func:`route_past` apply the same receipt discipline to timeout abstention and
multi-tier routing.

The named modes are ``teacher_down`` (fall back to captured or store-only
reasoning; see :meth:`mixle.system.System.answer`), ``store_down`` (reason
without accumulated knowledge; see :meth:`mixle.system.System.ingest`),
``oracle_timeout`` (abstain or escalate rather than guess), and ``model_error``
(route past the failing tier to the next one). The timeout and routing helpers
are reusable fault-boundary primitives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DegradedResult:
    """The outcome of a fault-boundary call: a value, whether it came from a fallback, and, if so, why."""

    value: Any
    degraded: bool
    mode: str | None = None
    reason: str | None = None

    def to_receipt_fields(self) -> dict[str, Any]:
        """Return ``degraded_mode`` and ``degraded_reason`` receipt fields."""
        return {"degraded_mode": self.mode, "degraded_reason": self.reason}


def with_fallback(fn: Callable[[], Any], fallback: Callable[[Exception], Any], *, mode: str) -> DegradedResult:
    """Run ``fn()``; on any exception, run ``fallback(exc)`` instead and flag the result under ``mode``.

    If ``fallback`` itself raises, that exception propagates. A fallback that
    cannot produce a value is a real failure, not a second implicit fallback.
    """
    try:
        return DegradedResult(value=fn(), degraded=False)
    except Exception as exc:  # noqa: BLE001 -- catch-any is the point: flag whatever the primary path raised
        return DegradedResult(value=fallback(exc), degraded=True, mode=mode, reason=str(exc))


def abstain_on_timeout(fn: Callable[[], Any], *, timeout_error: type[BaseException] = TimeoutError) -> DegradedResult:
    """``oracle_timeout`` mode: run ``fn()``; if it raises ``timeout_error``, abstain (``value=None``) rather
    than guess. Other exceptions propagate."""
    try:
        return DegradedResult(value=fn(), degraded=False)
    except timeout_error as exc:
        return DegradedResult(value=None, degraded=True, mode="oracle_timeout", reason=str(exc))


def route_past(tiers: Sequence[Callable[[], Any]], *, names: Sequence[str] | None = None) -> DegradedResult:
    """``model_error`` mode: try each tier in order; a raising tier is skipped (not fatal) in favor of the
    next. The result is degraded unless the first tier answers cleanly. Raises
    the last tier's exception if every tier fails."""
    if len(tiers) == 0:
        raise ValueError("route_past needs at least one tier")
    names = list(names) if names is not None else [f"tier{i}" for i in range(len(tiers))]
    if len(names) != len(tiers):
        raise ValueError(f"names has {len(names)} entries for {len(tiers)} tiers; they must match one-to-one")
    failed: list[str] = []
    last_exc: Exception | None = None
    for name, tier in zip(names, tiers, strict=True):
        try:
            value = tier()
        except Exception as exc:  # noqa: BLE001 -- route past this tier to the next, whatever it raised
            failed.append(name)
            last_exc = exc
            continue
        if not failed:
            return DegradedResult(value=value, degraded=False)
        return DegradedResult(value=value, degraded=True, mode="model_error", reason=f"routed past {failed}")
    raise last_exc  # every tier failed
