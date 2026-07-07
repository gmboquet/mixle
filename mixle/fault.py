"""Degradation policy (workstream J: FAULT-a) -- named failure modes, each flagged on the receipt.

A subsystem failure must never silently serve a worse answer as if nothing happened. This module is the
shared fault boundary every degraded path goes through: :func:`with_fallback` runs the primary path and, on
any exception, runs a named fallback instead -- the result is either NOT degraded (primary succeeded) or
degraded under an explicit, named mode with the triggering reason attached. :func:`abstain_on_timeout` and
:func:`route_past` are the same discipline specialized to two more named modes.

The four named modes (per the card): ``teacher_down`` (fall back to captured+store-only reasoning -- see
:meth:`mixle.system.System.answer`), ``store_down`` (reason without accumulated knowledge -- see
:meth:`mixle.system.System.ingest`), ``oracle_timeout`` (abstain/escalate rather than guess), and
``model_error`` (route past the failing tier to the next one). The last two are validated here as standalone
primitives: neither an oracle nor a multi-tier local-model router is wired into :class:`~mixle.system.System`
yet, so there is nothing live to attach them to without inventing an integration this card doesn't own.
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
        """The ``degraded_mode``/``degraded_reason`` fields a caller merges onto its own receipt dict."""
        return {"degraded_mode": self.mode, "degraded_reason": self.reason}


def with_fallback(fn: Callable[[], Any], fallback: Callable[[Exception], Any], *, mode: str) -> DegradedResult:
    """Run ``fn()``; on any exception, run ``fallback(exc)`` instead and flag the result under ``mode``.

    Either ``fn()`` succeeds and the result is NOT degraded, or the fallback runs and the result IS flagged
    with ``mode`` and the triggering exception's message as ``reason`` -- never a silent, unflagged
    "somehow it still worked" path. If ``fallback`` itself raises, that exception propagates: a fallback that
    cannot produce anything is a real failure to report, not something to paper over with a second guess.
    """
    try:
        return DegradedResult(value=fn(), degraded=False)
    except Exception as exc:  # noqa: BLE001 -- catch-any is the point: flag whatever the primary path raised
        return DegradedResult(value=fallback(exc), degraded=True, mode=mode, reason=str(exc))


def abstain_on_timeout(fn: Callable[[], Any], *, timeout_error: type[BaseException] = TimeoutError) -> DegradedResult:
    """``oracle_timeout`` mode: run ``fn()``; if it raises ``timeout_error``, abstain (``value=None``) rather
    than guess. Any OTHER exception propagates -- only a timeout is a license to abstain, not any failure."""
    try:
        return DegradedResult(value=fn(), degraded=False)
    except timeout_error as exc:
        return DegradedResult(value=None, degraded=True, mode="oracle_timeout", reason=str(exc))


def route_past(tiers: Sequence[Callable[[], Any]], *, names: Sequence[str] | None = None) -> DegradedResult:
    """``model_error`` mode: try each tier in order; a raising tier is skipped (not fatal) in favor of the
    next. Degraded (flagged with which tier(s) failed) unless the FIRST tier answers cleanly. Raises the
    last tier's exception if every tier fails -- there is no further fallback to route past."""
    names = list(names) if names is not None else [f"tier{i}" for i in range(len(tiers))]
    failed: list[str] = []
    last_exc: Exception | None = None
    for name, tier in zip(names, tiers):
        try:
            value = tier()
        except Exception as exc:  # noqa: BLE001 -- route past this tier to the next, whatever it raised
            failed.append(name)
            last_exc = exc
            continue
        if not failed:
            return DegradedResult(value=value, degraded=False)
        return DegradedResult(value=value, degraded=True, mode="model_error", reason=f"routed past {failed}")
    raise last_exc  # every tier failed -- nothing left to route past to
