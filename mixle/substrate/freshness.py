"""Freshness checks for substrate knowledge items.

Models drift; so does knowledge. An item can go stale three ways, each independently checkable:

* **moved data** -- the item references a file (``payload.ref`` / ``provenance.source``/``path``) that no
  longer exists, or whose content hash no longer matches the one recorded at ingest;
* **superseded** -- a newer item shares the same lineage parent or declares ``supersedes`` on this one,
  so this item is no longer the current version of its knowledge;
* **aged out** -- older than the caller's ``max_age_s`` policy for its kind (a soft signal: age alone is
  a review trigger, not proof of wrongness -- the finding says so).

:func:`check_freshness` audits one item and returns a three-state :class:`FreshnessState` verdict
(MXR-080-0267, mirrors :class:`~mixle.substrate.trust.LineageState`/MXR-080-0260/0261) rather than a
bare boolean: FRESH (every check ran, nothing fired), STALE (a confirmed signal fired), or
UNVERIFIABLE (a referenced path exists and carries a recorded ingest-time hash, but its CURRENT
content could not be read to check against that hash -- a directory, a permission error, or any other
read failure). Before this fix, an unreadable path's hash comparison was silently skipped as "no
signal," so a directory with a deliberately wrong recorded hash was reported fresh; unreadable content
is now always UNVERIFIABLE, never silently promoted to FRESH. :func:`content_hash` records a full,
algorithm-labelled digest (``"sha256:<64-hex>"``), not a bare hex string truncated to 128 bits with no
algorithm on it. ``now`` and ``max_age_s`` are validated (finite; ``max_age_s`` also non-negative)
before use, and a NaN or future-dated ``item.created_at`` is itself a named STALE signal rather than a
silently clamped ``age_s=0.0``.

:func:`freshness_report` sweeps a store. These checks help monitoring workflows
surface stale citations, unverifiable artifacts, and moved data before they affect downstream
answers.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


class FreshnessState(StrEnum):
    """The three-state verdict of one item's freshness check (MXR-080-0267) -- a closed vocabulary,
    the same discipline :class:`~mixle.substrate.trust.LineageState` applies to lineage verification
    and :class:`~mixle.substrate.spaces.MergeStrategy` applies to merge strategies. Never a bare
    boolean: a bare ``fresh=True`` cannot distinguish "every check ran, nothing fired" from "couldn't
    check some of it, so I don't actually know" -- exactly the fail-open overclaim this finding fixes.

    FRESH: every staleness check ran to completion and no signal fired.

    STALE: at least one CONFIRMED signal fired -- moved, changed, superseded, aged out past policy, or
    an invalid (NaN / future-dated) recorded timestamp.

    UNVERIFIABLE: a referenced path exists and an ingest-time hash was recorded for it, but the
    CURRENT content could not be read to compare against that hash (a directory, a permission
    error, or any other read failure). Never silently promoted to FRESH just because nothing
    contradicted the recorded hash -- there was no comparison to contradict it WITH.
    """

    FRESH = "fresh"
    STALE = "stale"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Freshness:
    """One item's freshness verdict: a three-state :attr:`state`, never a bare boolean (MXR-080-0267;
    mirrors :class:`~mixle.substrate.trust.LineageReport`). Every signal -- confirmed or unverifiable
    -- is named in :attr:`signals`.
    """

    item_id: str
    state: FreshnessState
    signals: list[str] = field(default_factory=list)
    age_s: float = 0.0

    @property
    def fresh(self) -> bool:
        """Read-only convenience view: ``True`` iff :attr:`state` is :attr:`FreshnessState.FRESH`.

        Never itself the source of truth -- :attr:`state` is (MXR-080-0267): both
        :attr:`FreshnessState.STALE` and :attr:`FreshnessState.UNVERIFIABLE` read ``False`` here,
        where the pre-fix code could read ``True`` for content it had never actually been able to
        check against its recorded hash -- a directory (or any other unreadable path) with a
        deliberately wrong recorded hash was reported fresh.
        """
        return self.state is FreshnessState.FRESH

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable freshness verdict."""
        return {
            "item_id": self.item_id,
            "state": self.state.value,
            "fresh": self.fresh,
            "signals": self.signals,
            "age_s": round(self.age_s, 1),
        }


def _referenced_path(item: SubstrateItem) -> str | None:
    for key in ("ref", "path"):
        v = item.payload.get(key)
        if isinstance(v, str) and v:
            return v
    for key in ("path", "source"):
        v = item.provenance.get(key)
        if isinstance(v, str) and ("/" in v or "\\" in v):
            return v
    return None


_HASH_ALGORITHM = "sha256"


def content_hash(path: str) -> str | None:
    """The full, algorithm-labelled digest of a file's bytes (``"sha256:<64-hex>"``), or ``None`` if
    unreadable -- record this verbatim at ingest, and compare it verbatim at audit time.

    MXR-080-0267: previously a bare hex string truncated to the first 32 characters (128 bits), with
    no algorithm recorded -- collision resistance silently weaker than sha256's own, and no way for a
    future reader to know what algorithm produced a recorded value (or to add a second scheme without
    guessing which one an old value used). The label makes the digest self-describing and
    algorithm-agile; the digest itself is never truncated.
    """
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
    return f"{_HASH_ALGORITHM}:{digest}"


def _validate_clock(value: float, *, label: str) -> float:
    """A finite float, or raise ``ValueError`` -- a caller-supplied timestamp argument is a caller-
    input correctness question, never audited data (MXR-080-0267). Pre-fix, ``max(0.0, now -
    created_at)`` silently turned a NaN delta into a falsely-fresh ``age_s=0.0`` instead of surfacing
    the invalid input (``max``'s left-argument-wins-when-NaN-compares-False quirk)."""
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"check_freshness: {label} must be a finite timestamp (seconds since epoch), got {value!r}")
    return v


def _validate_age_policy(max_age_s: float) -> float:
    """A finite, non-negative ``max_age_s``, or raise ``ValueError`` (MXR-080-0267): NaN silently
    disabled the age check forever (NaN compares False against everything, so ``age > max_age_s``
    never fires); +inf has the identical silent-disable effect; a negative bound is not a coherent
    "review after this many seconds" policy."""
    v = float(max_age_s)
    if math.isnan(v) or math.isinf(v) or v < 0:
        raise ValueError(
            f"check_freshness: max_age_s must be a finite, non-negative number of seconds, got {max_age_s!r}"
        )
    return v


def check_freshness(
    substrate: Substrate,
    item_id: str,
    *,
    max_age_s: float | None = None,
    now: float | None = None,
) -> Freshness:
    """Audit one item for the staleness signals (see module docstring), returning a three-state
    :class:`FreshnessState` verdict (MXR-080-0267) -- never a bare boolean. Missing item -> stale.

    ``now`` (default: the current wall clock, injectable for deterministic tests) and ``max_age_s``
    are caller-supplied ARGUMENTS, validated up front: NaN or infinite (and, for ``max_age_s``,
    negative) values are always a caller bug, so an invalid one raises :class:`ValueError`
    immediately rather than being silently clamped or ignored. By contrast, a NaN or future-dated
    ``item.created_at`` is substrate DATA being audited, exactly like a moved file or a changed hash:
    it is named in the result (state :attr:`FreshnessState.STALE`) rather than raised, so one
    corrupted record can never abort a :func:`freshness_report` sweep over everything else.

    Precedence when more than one kind of finding applies to the same item (mirrors
    :func:`~mixle.substrate.trust.verify_lineage`'s BROKEN-outranks-UNVERIFIED rule):
    :attr:`FreshnessState.STALE` (a confirmed signal) outranks :attr:`FreshnessState.UNVERIFIABLE`
    (an open question) outranks :attr:`FreshnessState.FRESH`.
    """
    now = time.time() if now is None else _validate_clock(now, label="now")
    if max_age_s is not None:
        max_age_s = _validate_age_policy(max_age_s)

    item = substrate.get(item_id)
    if item is None:
        return Freshness(item_id=item_id, state=FreshnessState.STALE, signals=["missing: item no longer exists"])

    stale_signals: list[str] = []
    unverifiable_signals: list[str] = []

    created_at = float(item.created_at)
    if math.isnan(created_at) or math.isinf(created_at):
        stale_signals.append(f"invalid: created_at={item.created_at!r} is not a finite timestamp")
        age = 0.0
    else:
        age = now - created_at
        if age < 0:
            stale_signals.append(
                f"invalid: created_at={created_at!r} is {-age:.0f}s after now={now!r} (a future-dated item)"
            )

    # moved / changed / unverifiable data
    ref = _referenced_path(item)
    if ref is not None:
        if not Path(ref).exists():
            stale_signals.append(f"moved: referenced path {ref!r} no longer exists")
        else:
            recorded = item.provenance.get("content_hash")
            if recorded:
                digest = content_hash(ref)
                if digest is None:
                    # Content exists but could not be read/hashed (a directory, a permission error, or
                    # any other read failure): the comparison against `recorded` cannot be performed,
                    # so this is an open question -- never silently "no signal" / certified fresh. This
                    # was MXR-080-0267's core fail-open bug: a directory with a deliberately wrong
                    # recorded hash used to fall straight through here with no signal at all.
                    unverifiable_signals.append(
                        f"unverifiable: {ref!r} exists but its content could not be read to check "
                        "against the recorded content_hash"
                    )
                elif digest != recorded:
                    stale_signals.append(f"changed: content hash of {ref!r} no longer matches ingest")

    # superseded by a newer item
    for other in substrate.all():
        if other.id == item.id:
            continue
        declares = other.provenance.get("supersedes")
        if declares == item.id:
            stale_signals.append(f"superseded: item {other.id!r} declares supersedes={item.id!r}")
        elif (
            other.kind == item.kind
            and other.created_at > item.created_at
            and item.id in other.links
            and other.provenance.get("replaces") == item.id
        ):
            stale_signals.append(f"superseded: newer linked item {other.id!r} replaces this one")

    # aged out (a review trigger, not proof of wrongness)
    if max_age_s is not None and age > max_age_s:
        stale_signals.append(f"aged: {age:.0f}s old exceeds the {max_age_s:.0f}s policy (review, not proof)")

    if stale_signals:
        state = FreshnessState.STALE
    elif unverifiable_signals:
        state = FreshnessState.UNVERIFIABLE
    else:
        state = FreshnessState.FRESH

    return Freshness(item_id=item_id, state=state, signals=[*stale_signals, *unverifiable_signals], age_s=age)


def freshness_report(
    substrate: Substrate,
    *,
    max_age_s: float | None = None,
    scope: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Sweep a store for stale/unverifiable knowledge (MXR-080-0267 three-state; mirrors
    :func:`~mixle.substrate.trust.audit_substrate`'s broken/unverified split): ``{n_items, n_fresh,
    n_stale, n_unverifiable, stale: [...], unverifiable: [...]}`` -- the monitor feed. ``stale`` names
    every item with a confirmed signal (moved/changed/superseded/aged/invalid timestamp);
    ``unverifiable`` names every item whose referenced content exists and carries a recorded hash but
    could not be read to check against it -- kept separate from ``stale`` because "could not verify"
    is never the same claim as "confirmed defective."
    """
    items = substrate.all(scope=scope)
    stale: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    for it in items:
        f = check_freshness(substrate, it.id, max_age_s=max_age_s, now=now)
        if f.state is FreshnessState.STALE:
            stale.append(f.as_dict())
        elif f.state is FreshnessState.UNVERIFIABLE:
            unverifiable.append(f.as_dict())
    return {
        "n_items": len(items),
        "n_fresh": len(items) - len(stale) - len(unverifiable),
        "n_stale": len(stale),
        "n_unverifiable": len(unverifiable),
        "stale": stale,
        "unverifiable": unverifiable,
    }
