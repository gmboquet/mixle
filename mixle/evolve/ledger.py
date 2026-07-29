"""The evolution ledger: an in-memory, JSON-serializable record of every improvement attempt.

The library layer does no I/O, so the ledger is just an in-process list of rows -- one per proposed
challenger -- that an orchestrator can persist (e.g. into a registry version's metadata). Each row is a
plain dict so it round-trips through ``json.dumps`` without custom encoders: the model objects
themselves are never stored, only their operator, measured delta, verdict, cost, and parent hash.

The ledger is the *evidence* for a claimed trail of autonomous improvement attempts, so it owns its
rows rather than handing out the live ones. Every row carries a sequence number, a schema version, and
a digest chained to its predecessor (:func:`_row_hash`); :meth:`EvolutionLedger.verify` re-derives the
chain, so a changed delta, a rewritten verdict, a deleted attempt, or a reordering is detectable
rather than silent. :meth:`EvolutionLedger.record` and the ``rows`` view both return deep copies:
mutating what you were handed cannot rewrite what was recorded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

LEDGER_SCHEMA_VERSION = 1
"""Version of the row schema, stamped on every row so a later reader can detect an older layout."""

FLOATENVELOPE_VERSION = 1
"""Version of the non-finite-float encoding :meth:`EvolutionLedger.to_json` writes (see :data:`_FLOAT_TAG`)."""

_FLOAT_TAG = "__mixle_float__"
"""Key marking a JSON object that stands for a non-finite float.

``json.dumps`` writes bare ``NaN`` / ``Infinity`` / ``-Infinity`` tokens by default. Those are a
Python extension, not JSON: a conforming parser (JavaScript's ``JSON.parse``, Go, Rust's serde,
PostgreSQL's ``jsonb``) rejects the document outright, so a ledger holding one was durable evidence
nothing else could read. A ledger nonetheless has to be ABLE to hold one -- a scalar-only
:class:`~mixle.evolve.verify.Verdict` carries ``p_value = nan`` as its documented "no paired test was
run" sentinel -- so non-finite values are encoded exactly and reversibly rather than refused or
silently rewritten: ``{"__mixle_float__": "NaN", "codec_version": 1}``.
"""


def _finite_number(value: Any, name: str) -> float:
    """A finite real ``float``, or a :class:`ValueError` naming the field.

    ``delta`` and ``cost`` are the two measurements a row asserts. ``float(delta)`` alone accepted
    ``nan``/``inf``, so a ledger could record "this attempt improved the objective by NaN at a cost of
    Infinity" and :meth:`EvolutionLedger.verify` would confirm the trail as intact. A non-measurement
    is not evidence; it is refused where it enters, not carried as though it were a number.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _nonnegative_cost(value: Any) -> float:
    """A finite, non-negative cost. Spend is never a refund, and never an unmeasured NaN."""
    cost = _finite_number(value, "cost")
    if cost < 0.0:
        raise ValueError(f"cost must be non-negative, got {value!r}")
    return cost


def _encode_floats(obj: Any) -> Any:
    """Rewrite non-finite floats into the tagged form above; everything else passes through unchanged."""
    if isinstance(obj, float) and not math.isfinite(obj):
        token = "NaN" if math.isnan(obj) else ("Infinity" if obj > 0 else "-Infinity")
        return {_FLOAT_TAG: token, "codec_version": FLOATENVELOPE_VERSION}
    if isinstance(obj, dict):
        return {k: _encode_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_floats(v) for v in obj]
    return obj


def _decode_floats(obj: Any) -> Any:
    """Inverse of :func:`_encode_floats`; an unrecognized tag payload is corruption, not a value."""
    if isinstance(obj, dict):
        if _FLOAT_TAG in obj:
            version = obj.get("codec_version")
            if version != FLOATENVELOPE_VERSION:
                raise ValueError(
                    f"EvolutionLedger: {_FLOAT_TAG} has codec_version {version!r}, expected "
                    f"{FLOATENVELOPE_VERSION!r} -- written by an incompatible codec version"
                )
            token = obj[_FLOAT_TAG]
            if token in ("nan", "NaN"):
                return float("nan")
            if token in ("inf", "Infinity"):
                return float("inf")
            if token in ("-inf", "-Infinity"):
                return float("-inf")
            raise ValueError(f"EvolutionLedger: unrecognized {_FLOAT_TAG} payload {token!r} -- corrupted receipt")
        return {k: _decode_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_floats(v) for v in obj]
    return obj


ENVELOPE_VERSION = 1
"""Version of the anchored document :meth:`EvolutionLedger.to_json` writes."""

_GENESIS_HASH = "0" * 64
"""``prev_hash`` of the first row: a fixed anchor, so truncating the ledger from the front is visible."""

_HASHED_KEYS = ("seq", "schema_version", "operator", "delta", "verdict", "cost", "parent_hash", "meta", "prev_hash")


def _row_hash(row: dict[str, Any]) -> str:
    """Digest a row's canonical content -- every field except the digest itself, plus ``prev_hash``.

    Including ``prev_hash`` makes the rows a chain: altering row ``i`` invalidates every row after it,
    so no localized edit can be made to verify.
    """
    payload = json.dumps({key: row[key] for key in _HASHED_KEYS}, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EvolutionLedger:
    """An ordered, JSON-serializable, integrity-chained log of improvement attempts."""

    _rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        """A read-only view: a tuple of deep copies of the recorded rows.

        The backing list used to be the public attribute holding the live row dicts, so a caller could
        edit a recorded delta or verdict in place, or ``clear()`` the whole trail, with nothing to show
        it happened. Reading, indexing, iterating and ``json.dumps``-ing the returned rows all work as
        before; writing to them simply does not reach the ledger.
        """
        return tuple(copy.deepcopy(row) for row in self._rows)

    def record(
        self,
        *,
        operator: str,
        delta: float,
        verdict: dict | None,
        cost: float,
        parent_hash: str | None,
        meta: dict | None = None,
    ) -> dict[str, Any]:
        """Append one attempt row and return a copy of it.

        The returned dict is a deep copy, not the stored object: the previous behaviour handed back the
        exact row still held by the ledger, so a caller that adjusted a field on "its" result silently
        rewrote the recorded attempt.

        ``delta`` and ``cost`` are the row's two measurements and must be finite; ``cost`` must also be
        non-negative, since it is spend, not a refund. See :func:`_finite_number`. (A ``nan`` p-value
        inside ``verdict`` is a different thing entirely -- a documented "no paired test was run"
        sentinel -- and is preserved, exactly, by the serialization codec.)
        """
        row: dict[str, Any] = {
            "seq": len(self._rows),
            "schema_version": LEDGER_SCHEMA_VERSION,
            "operator": operator,
            "delta": _finite_number(delta, "delta"),
            "verdict": copy.deepcopy(verdict),
            "cost": _nonnegative_cost(cost),
            "parent_hash": parent_hash,
            "meta": copy.deepcopy(meta) if meta else {},
            "prev_hash": self._rows[-1]["row_hash"] if self._rows else _GENESIS_HASH,
        }
        row["row_hash"] = _row_hash(row)
        self._rows.append(row)
        return copy.deepcopy(row)

    def head(self) -> str:
        """The chain's current tip: the last row's ``row_hash``, or :data:`_GENESIS_HASH` if empty.

        This is the anchor :meth:`verify` needs to detect truncation. Record it somewhere the ledger
        's own storage cannot reach -- a run report, a build artifact, a separate durable store --
        each time you finish appending; passing it back to ``verify(expect_head=...)`` is what turns
        "these rows are self-consistent" into "these are the rows that were written".
        """
        return self._rows[-1]["row_hash"] if self._rows else _GENESIS_HASH

    def verify(self, *, expect_length: int | None = None, expect_head: str | None = None) -> bool:
        """Whether every row still matches its digest and its place in the chain.

        Checks, per row: the sequence number matches its position, ``prev_hash`` matches the previous
        row's ``row_hash`` (:data:`_GENESIS_HASH` for the first), and ``row_hash`` still matches the
        row's canonical content. A rewritten delta or verdict, a deleted or reordered attempt, and a
        front-truncated ledger all fail.

        **Truncation from the end cannot be detected from the rows alone, and no hash chain can do
        it.** Deleting the last row leaves a shorter chain that is perfectly self-consistent -- every
        remaining ``prev_hash`` still matches, every ``row_hash`` still verifies -- because a prefix
        of a valid chain is a valid chain. Detecting it needs one fact the ledger does not hold: how
        long the chain is supposed to be, or what its tip was. Supply either and the check becomes
        real:

            head = ledger.head()          # store this outside the ledger
            ...
            ledger.verify(expect_head=head, expect_length=n)

        ``expect_head`` is the stronger of the two: it pins the exact final row, so it also rejects a
        ledger truncated and then re-extended to the original length. ``expect_length`` alone catches
        only a change in count. Omitting both is still meaningful -- it proves internal consistency --
        but it is not proof that nothing was removed from the end.
        """
        if expect_length is not None and len(self._rows) != expect_length:
            return False
        expected_prev = _GENESIS_HASH
        for index, row in enumerate(self._rows):
            if row.get("seq") != index or row.get("prev_hash") != expected_prev:
                return False
            if _row_hash(row) != row.get("row_hash"):
                return False
            expected_prev = row["row_hash"]
        return expect_head is None or expected_prev == expect_head

    def to_json(self, **dumps_kwargs: Any) -> str:
        """Serialize the full ledger to a **strict** JSON string.

        Non-finite floats anywhere in a row (a scalar-only verdict's ``nan`` p-value, say) are encoded
        into the reversible tagged form :data:`_FLOAT_TAG` documents, and ``allow_nan`` is then forced
        off so anything the encoder missed fails loudly here rather than emitting a bare ``NaN`` /
        ``Infinity`` token a conforming JSON parser refuses to read back. Round-trips exactly through
        :meth:`from_json`, digests included.
        """
        dumps_kwargs.pop("allow_nan", None)
        # The document carries its own anchor. A hash chain cannot detect truncation from the end --
        # a prefix of a valid chain is a valid chain -- so head()/verify(expect_head=) exist to close
        # that with an external fact. An anchor nobody supplies closes nothing, and serialization is
        # the moment the records leave this process's control, so the length and tip travel with
        # them and :meth:`from_json` checks both. Truncating the array now fails on load.
        payload = {
            "codec_version": ENVELOPE_VERSION,
            "length": len(self._rows),
            "head": self.head(),
            "rows": _encode_floats(self._rows),
        }
        return json.dumps(payload, default=_json_default, allow_nan=False, **dumps_kwargs)

    @classmethod
    def from_json(cls, s: str) -> EvolutionLedger:
        """Rebuild a ledger from :meth:`to_json` output, checking the anchor it carries.

        A document written by :meth:`to_json` is an envelope holding the rows plus the length and
        chain tip they had when written; both are verified here, so a row deleted from the end --
        which :meth:`verify` alone cannot see -- is caught on load.

        A bare JSON array is still accepted for documents written before the envelope existed. Those
        carry no anchor, so loading one proves only that the rows are internally consistent; pass a
        separately stored ``expect_head`` to :meth:`verify` for the stronger check.
        """
        doc = json.loads(s)
        if isinstance(doc, list):
            return cls(list(_decode_floats(doc)))
        version = doc.get("codec_version")
        if version != ENVELOPE_VERSION:
            raise ValueError(f"ledger document has codec_version {version!r}, expected {ENVELOPE_VERSION!r}")
        led = cls(list(_decode_floats(doc["rows"])))
        # Only the ANCHOR is checked here -- the count and tip the document declares against what was
        # read back. That is the one fact loading can establish and verify() cannot, since a
        # truncated chain is still a valid chain. Chain validity stays verify()'s job and the
        # caller's decision: a ledger whose rows are internally inconsistent must still load so it
        # can be inspected, not become unreadable exactly when someone needs to see what happened.
        length, head = int(doc["length"]), str(doc["head"])
        if len(led._rows) != length or led.head() != head:
            raise ValueError(
                f"ledger does not match the anchor it was written with: the document declares "
                f"{length} row(s) ending at {head[:12]}..., and {len(led._rows)} row(s) ending at "
                f"{led.head()[:12]}... were read back -- rows were added or removed after it was written"
            )
        return led

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self.rows)


def _json_default(obj: Any) -> Any:
    """Fallback so stray numpy scalars / dataclasses don't break serialization.

    The first three routes are exact and replayable. The last is not: ``str(obj)`` keeps the receipt
    human-readable at the price of its type and any hope of reconstructing the value, so it now warns
    -- naming the type that was flattened -- instead of substituting silently. (Same discipline as
    :class:`mixle.epistemic.journal.EpistemicJournal`, which warns on the identical fallback.) A
    caller that needs the value back must put a JSON-native or ``as_dict()``-bearing object in ``meta``.
    """
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    warnings.warn(
        f"EvolutionLedger: {type(obj).__name__} has no exact JSON encoding and was stored as its "
        "str() -- the receipt stays readable but this value cannot be replayed from it",
        UserWarning,
        stacklevel=2,
    )
    return str(obj)


__all__ = ["EvolutionLedger", "LEDGER_SCHEMA_VERSION", "FLOATENVELOPE_VERSION"]
