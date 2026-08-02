"""``EpistemicJournal`` -- an append-only, replayable decision log for a sequence of loop steps.

The program plan's decision journal (§3.4: "timestamp, model version, belief snapshot hash, options
considered with their EIG/cost/risk scores, chosen action, rationale claims..."), following
:class:`mixle.evolve.ledger.EvolutionLedger`'s existing shape (flat, JSON-serializable, append-only,
no model objects stored directly) rather than inventing a third bespoke logging pattern.

One deliberate refinement over a hash-only ledger: each record carries the *full* serialized belief
snapshot (``portfolio_snapshot``, from :meth:`~mixle.epistemic.portfolio.HypothesisPortfolio.to_dict`)
alongside its content-address (``belief_snapshot_hash``). A hash alone cannot be reversed back into a
portfolio, so "an auditor can reconstruct the full belief trajectory from the ledger alone" (program
plan §2) requires the snapshot content to actually be there; the hash is what lets :meth:`verify`
catch tampering/corruption of that stored content, which is the property a bare append-only list
doesn't get for free.

Type fidelity of the *payload* content within a snapshot (``Hypothesis.payload``, and the journal's own
``action_chosen``/``action_considered`` fields, both declared ``Any``) is bounded by what JSON can
actually represent. Plain ``json.dumps``/``json.loads`` silently turns a ``tuple`` into a ``list`` --
a different Python value (``(1, 2) != [1, 2]``) that happens to render as identical JSON text, so a
hash computed over that JSON text cannot tell the two apart either. :func:`_tag_encode`/
:func:`_tag_decode` close this for the types it is small and safe to do so for: JSON-native values plus
``tuple``, numpy scalars (``numpy.generic``), and numpy arrays round-trip as themselves (see
:func:`_tag_encode`'s docstring for the exact list and the versioned tag format), and :meth:`verify`
hashes this type-faithful encoding rather than JSON's lossy shadow of it -- a tuple silently replayed
as a list now actually fails verification instead of passing it.

Anything else (an arbitrary custom object -- e.g. a fitted model used directly as a hypothesis payload,
which real callers such as :func:`mixle.task.discrepancy_invention_loop.run_discrepancy_invention_loop`
do) has no small, general, always-correct JSON encoding of its own; building one is a substantially
larger undertaking (already partially addressed, for the types that opt into it, by
:mod:`mixle.utils.serialization`) and out of scope for this journal. Such a payload is instead stored
as a one-way ``str()`` snapshot, with a ``UserWarning`` raised at journal time so the caller knows:
:meth:`EpistemicJournal.replay` hands back that string, not the original object, and :meth:`verify` for
such a record only attests that the string is unchanged -- not that the original object would be
recovered. This keeps the journal usable for arbitrary hypothesis/action payloads without ever
silently overclaiming a fidelity it can't deliver.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from mixle.epistemic.loop import EpistemicStep
from mixle.epistemic.portfolio import HypothesisPortfolio
from mixle.utils.immutable import detach_receipt_container

_CODEC_VERSION = 1
_ENVELOPE_VERSION = 1
"""Version of the anchored document :meth:`EpistemicJournal.to_json` writes."""
_TAG_KEY = "__type__"
_TUPLE_TAG = "tuple"
_NDARRAY_TAG = "ndarray"
_NP_SCALAR_TAG = "np_scalar"
_NONFINITE_TAG = "nonfinite_float"
_JSON_SCALAR_TYPES = (str, int, float, bool)
_NONFINITE_BY_NAME = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _nonfinite_encoded(x: float) -> dict:
    """Tagged form of a non-finite float.

    JSON has no NaN or Infinity. Python's ``json`` emits the bare tokens ``NaN``/``Infinity``
    anyway, which every strict parser rejects -- so a journal holding one non-finite surprise or
    EIG serialized to text that says it is JSON and is not, and only an external reader would ever
    find out. Round-tripping through the same versioned tag scheme the other non-JSON-native types
    use keeps the value exact and the document valid.
    """
    return {
        _TAG_KEY: _NONFINITE_TAG,
        "codec_version": _CODEC_VERSION,
        "value": ("nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")),
    }


def _encode_nonfinite(value: Any) -> Any:
    """Rewrite non-finite floats inside an already-plain-Python nested list structure."""
    if isinstance(value, list):
        return [_encode_nonfinite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return _nonfinite_encoded(value)
    return value


def _all_json_native(value: Any) -> bool:
    """True if ``value`` (arbitrarily nested lists) is built only from JSON-native leaves.

    Used to decide whether a numpy array/scalar's plain-Python content (post ``.tolist()``/``.item()``)
    is safe to tag-and-restore exactly, or must fall back to :func:`_encode_opaque` -- e.g. a complex-
    or object-dtype array whose elements are not themselves JSON-native.
    """
    if isinstance(value, list):
        return all(_all_json_native(v) for v in value)
    return value is None or isinstance(value, _JSON_SCALAR_TYPES)


def _encode_opaque(obj: Any) -> str:
    """One-way fallback for a payload type the codec has no exact-round-trip representation for.

    Returns ``str(obj)`` and warns rather than silently pretending the value survives a round trip:
    :meth:`EpistemicJournal.replay` hands back this string (not ``obj``), and :meth:`EpistemicJournal.
    verify` for a record holding it only attests that the string is unchanged, not that ``obj`` itself
    would be recovered. See :func:`_tag_encode` for the (small, exact) set of types that do round-trip.
    """
    warnings.warn(
        f"EpistemicJournal: {type(obj).__name__!r} payload has no exact-round-trip codec; storing a "
        "one-way str() snapshot instead. replay() will hand back this string, not the original object, "
        "and verify() only certifies that the string is unchanged -- not that the original object/type "
        "would be recovered. Types that DO round-trip exactly: JSON-native values (dict/list/str/int/"
        "float/bool/None), tuple, numpy scalars, and numpy arrays.",
        UserWarning,
        stacklevel=3,
    )
    return str(obj)


def _tag_encode(obj: Any) -> Any:
    """Recursively rewrite ``obj`` into a JSON-native structure that :func:`_tag_decode` can reverse.

    A small, explicit, versioned type registry (``codec_version`` on every tag, so a future encoding
    change can be detected on load rather than silently mis-decoded): ``tuple``, numpy scalars
    (``np.generic``), and numpy arrays are wrapped as ``{"__type__": ..., "codec_version": ...,
    "value": ...}`` so they come back as themselves, not as JSON's nearest native stand-in (a tuple
    would otherwise silently become a list -- a different, non-tuple-equal Python value). A dict that
    already uses the reserved ``"__type__"`` key, or has any non-``str`` key (JSON object keys are
    always strings, so e.g. an int key would otherwise be silently string-coerced), would be ambiguous
    with this scheme on decode, so it is routed through :func:`_encode_opaque` instead of being
    misread. Everything else JSON already represents natively (``dict``, ``list``, ``str``, ``int``,
    ``float``, ``bool``, ``None``) passes through unchanged. Anything not covered above -- an arbitrary
    custom object, e.g. a fitted model used as a hypothesis payload -- falls back to
    :func:`_encode_opaque`'s one-way ``str()`` capture: a full general-purpose codec for arbitrary
    classes is out of scope here (see the module docstring), so this is deliberately narrow rather than
    silently claiming a fidelity it can't deliver. This function never raises.
    """
    import numpy as np

    if isinstance(obj, tuple):
        return {_TAG_KEY: _TUPLE_TAG, "codec_version": _CODEC_VERSION, "value": [_tag_encode(x) for x in obj]}
    if isinstance(obj, np.ndarray):
        values = obj.tolist()
        if _all_json_native(values):
            # the vectorized finiteness test keeps the common all-finite array on its original
            # zero-copy path; only an array that really holds a NaN/inf pays for the rewrite
            if obj.dtype.kind in "fc" and not np.isfinite(obj).all():
                values = _encode_nonfinite(values)
            return {_TAG_KEY: _NDARRAY_TAG, "codec_version": _CODEC_VERSION, "dtype": str(obj.dtype), "value": values}
        return _encode_opaque(obj)
    if isinstance(obj, np.generic):
        item = obj.item()
        if _all_json_native(item):
            item = _encode_nonfinite(item)
            return {_TAG_KEY: _NP_SCALAR_TAG, "codec_version": _CODEC_VERSION, "dtype": str(obj.dtype), "value": item}
        return _encode_opaque(obj)
    # below the numpy branches on purpose: np.float64 subclasses float, and it has to keep reaching
    # the np_scalar tag or a numpy nan would come back as a plain Python float
    if isinstance(obj, float) and not math.isfinite(obj):
        return _nonfinite_encoded(obj)
    if isinstance(obj, dict):
        if _TAG_KEY in obj or not all(isinstance(k, str) for k in obj):
            return _encode_opaque(obj)
        return {k: _tag_encode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tag_encode(x) for x in obj]
    if obj is None or isinstance(obj, _JSON_SCALAR_TYPES):
        return obj
    return _encode_opaque(obj)


def _tag_decode(obj: Any) -> Any:
    """Inverse of :func:`_tag_encode`: restore tuples/numpy scalars/numpy arrays from their tagged form.

    Any dict containing the reserved ``"__type__"`` key is guaranteed (by :func:`_tag_encode` routing
    real collisions through :func:`_encode_opaque` instead) to have been produced by this codec, so an
    unrecognized tag or a ``codec_version`` this function doesn't understand is genuinely anomalous
    data -- corrupted, hand-edited, or written by an incompatible codec version -- and raises rather
    than risk a silent mis-decode.
    """
    import numpy as np

    if isinstance(obj, dict):
        tag = obj.get(_TAG_KEY)
        if tag is None:
            return {k: _tag_decode(v) for k, v in obj.items()}
        version = obj.get("codec_version")
        if version != _CODEC_VERSION:
            raise ValueError(
                f"EpistemicJournal: tag {tag!r} has codec_version {version!r}, expected {_CODEC_VERSION!r} "
                "-- written by an incompatible codec version"
            )
        if tag == _TUPLE_TAG:
            return tuple(_tag_decode(x) for x in obj["value"])
        if tag == _NONFINITE_TAG:
            name = obj["value"]
            if name not in _NONFINITE_BY_NAME:
                raise ValueError(f"EpistemicJournal: unrecognized non-finite float {name!r} -- corrupted data")
            return _NONFINITE_BY_NAME[name]
        if tag == _NDARRAY_TAG:
            return np.array(_tag_decode(obj["value"]), dtype=obj["dtype"])
        if tag == _NP_SCALAR_TAG:
            return np.dtype(obj["dtype"]).type(_tag_decode(obj["value"]))
        raise ValueError(f"EpistemicJournal: unrecognized codec tag {tag!r} -- corrupted or foreign data")
    if isinstance(obj, list):
        return [_tag_decode(x) for x in obj]
    return obj


def _hash_snapshot(snapshot: dict) -> str:
    payload = json.dumps(_tag_encode(snapshot), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_GENESIS_HASH = "0" * 64
"""``prev_hash`` of the first record: a fixed anchor so "this is record 0" is itself attested.

Without it, ``prev_hash=None`` on record 0 would be indistinguishable from "the records before me
were removed", and truncating a journal from the front would still verify.
"""


@dataclass(frozen=True)
class DecisionRecord:
    """One journaled decision: what was believed, what was considered, what was chosen, and why.

    ``belief_snapshot_hash`` content-addresses ``portfolio_snapshot`` alone. ``record_hash`` covers
    the *whole* record -- surprise, actions considered, action chosen, EIG, timestamp, rationale,
    step index, and ``prev_hash`` -- and because it includes ``prev_hash`` it is a hash chain: the
    hash of record ``i`` depends on every record before it, so removing, reordering, or splicing
    records breaks the chain rather than going unnoticed.
    """

    step_index: int
    belief_snapshot_hash: str
    portfolio_snapshot: dict
    surprise: float
    action_considered: list[Any] = field(default_factory=list)
    action_chosen: Any | None = None
    action_eig: float | None = None
    timestamp: float | None = None
    rationale: str | None = None
    prev_hash: str = _GENESIS_HASH
    record_hash: str = ""

    def __post_init__(self) -> None:
        # detach, not freeze: this record is content-addressed and JSON round-tripped with type
        # fidelity -- the journal deliberately distinguishes a tuple from an equal-valued list --
        # and mappingproxy cannot be deep-copied or pickled. Converting the containers broke the
        # hash chain (MXR-080-1876).
        object.__setattr__(self, "portfolio_snapshot", detach_receipt_container(self.portfolio_snapshot))
        object.__setattr__(self, "action_considered", detach_receipt_container(self.action_considered))


def _record_payload(record: DecisionRecord) -> dict:
    """The canonical, hashable content of ``record`` -- everything except ``record_hash`` itself.

    Built field-by-field rather than with :func:`~dataclasses.asdict`, which deep-copies: a
    hypothesis payload or chosen action can be an arbitrary object (a fitted model, say), and this
    runs on every append and every verify.
    """
    return {
        "step_index": record.step_index,
        "belief_snapshot_hash": record.belief_snapshot_hash,
        "portfolio_snapshot": record.portfolio_snapshot,
        "surprise": record.surprise,
        "action_considered": list(record.action_considered),
        "action_chosen": record.action_chosen,
        "action_eig": record.action_eig,
        "timestamp": record.timestamp,
        "rationale": record.rationale,
        "prev_hash": record.prev_hash,
    }


def _hash_record(record: DecisionRecord) -> str:
    """Content-address the complete record, over the same type-faithful encoding as the snapshot."""
    payload = json.dumps(_tag_encode(_record_payload(record)), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EpistemicJournal:
    """An ordered, JSON-serializable, replayable log of :class:`~mixle.epistemic.loop.EpistemicStep`\\ s.

    See the module docstring for the precise type-fidelity contract: JSON-native values, ``tuple``,
    numpy scalars, and numpy arrays round-trip through :meth:`to_json`/:meth:`from_json` (and
    :meth:`replay`) exactly; any other payload type is stored as a one-way ``str()`` snapshot (with a
    ``UserWarning`` at journal time) and does not.
    """

    def __init__(self, records: list[DecisionRecord] | None = None) -> None:
        self._records: list[DecisionRecord] = list(records) if records else []

    @property
    def records(self) -> tuple[DecisionRecord, ...]:
        """The journaled records, as a read-only snapshot.

        The backing list used to be the public attribute, so a caller could ``clear()``, ``del``, or
        reorder the journal in place -- an append-only decision log that anything holding a reference
        could rewrite. Indexing, iteration, and ``len()`` are unchanged; use :meth:`append` to add.
        (Record *content* is still reachable through the returned records, which is why
        :meth:`verify` exists and now covers the whole record and its position in the chain.)
        """
        return tuple(self._records)

    def append(
        self,
        step: EpistemicStep,
        *,
        action_considered: list[Any] = (),
        rationale: str | None = None,
        timestamp: float | None = None,
    ) -> DecisionRecord:
        """Append one record for ``step`` and return it. ``timestamp`` is caller-supplied, never sampled here.

        ``belief_snapshot_hash`` is computed over ``step.portfolio_after``'s type-faithful encoding (see
        :func:`_tag_encode`): a hypothesis payload outside the supported type set is stored anyway (as a
        one-way ``str()`` snapshot) rather than rejected, since real callers journal arbitrary typed
        hypothesis payloads (e.g. fitted model objects) by design -- but a ``UserWarning`` is raised so
        this is never silent.
        """
        snapshot = step.portfolio_after.to_dict()
        record = DecisionRecord(
            step_index=len(self._records),
            belief_snapshot_hash=_hash_snapshot(snapshot),
            portfolio_snapshot=snapshot,
            surprise=step.surprise,
            action_considered=list(action_considered),
            action_chosen=step.next_action,
            action_eig=step.next_action_eig,
            timestamp=timestamp,
            rationale=rationale,
            prev_hash=self._records[-1].record_hash if self._records else _GENESIS_HASH,
        )
        record = replace(record, record_hash=_hash_record(record))
        self._records.append(record)
        return record

    def replay(self, portfolio0: HypothesisPortfolio | None = None) -> list[HypothesisPortfolio]:
        """Reconstruct the belief trajectory from the journal's stored snapshots alone.

        ``portfolio0`` is accepted for interface symmetry with the loop's own ``step(portfolio, ...)``
        signature but is not required for reconstruction here: every record already carries its own
        full ``portfolio_snapshot``, so replay is deserialization, not re-simulation (re-simulation
        would additionally need the original observations and likelihood callables, which are
        deliberately not journaled -- they may not be JSON-serializable, and the snapshot is the thing
        an audit actually needs). If given, ``portfolio0`` is prepended to the returned trajectory.

        A hypothesis payload that round-tripped through :meth:`to_json`/:meth:`from_json` comes back as
        its original type for the supported set (JSON-native, tuple, numpy scalar/array); otherwise it
        comes back as the one-way ``str()`` snapshot that was actually stored (see the module
        docstring). Replaying a journal that was never serialized (no :meth:`to_json`/:meth:`from_json`
        round trip) always returns the original in-memory payload objects unchanged, since nothing was
        ever encoded.
        """
        trajectory = [HypothesisPortfolio.from_dict(r.portfolio_snapshot) for r in self._records]
        return ([portfolio0] + trajectory) if portfolio0 is not None else trajectory

    def head(self) -> str:
        """The chain's current tip: the last record's ``record_hash``, or the genesis anchor if empty.

        This is what :meth:`verify` needs in order to detect truncation. Store it outside the
        journal's own storage each time you finish appending, and hand it back to
        ``verify(expect_head=...)``.
        """
        return self._records[-1].record_hash if self._records else _GENESIS_HASH

    def verify(self, *, expect_length: int | None = None, expect_head: str | None = None) -> bool:
        """Return whether the journal is intact: every record's content AND its place in the chain.

        This used to compare only ``portfolio_snapshot`` against ``belief_snapshot_hash``, so
        everything the log exists to attest -- the surprise, the actions considered, the action
        actually chosen, its EIG, the timestamp, the rationale, the step index -- could be rewritten
        and this still returned ``True``. So could deleting or reordering records, since nothing tied
        a record to its position or its predecessor.

        Four things are now checked per record, in order: its ``step_index`` matches its position;
        its ``prev_hash`` matches the previous record's ``record_hash`` (:data:`_GENESIS_HASH` for
        the first); its snapshot still matches ``belief_snapshot_hash``; and its ``record_hash``
        still matches the whole canonical record. Because each ``record_hash`` covers ``prev_hash``,
        the records form a chain: altering record ``i`` invalidates every record after it, so no
        localized edit can be made to verify.

        Hashing is over the type-faithful encoding (:func:`_tag_encode`), not a lossy JSON shadow of
        it: two snapshots differing only in payload *type* (a tuple silently replaced by an
        equal-valued list) hash differently and correctly fail. For a payload stored as a one-way
        ``str()`` snapshot (see the module docstring), this only proves that string is unchanged, not
        that the original object would be recovered -- there is no stronger claim to make once the
        exact object is gone.

        **Truncation from the end cannot be detected from the records alone, and no hash chain can
        do it.** Deleting the final record leaves a shorter chain that is entirely self-consistent:
        every remaining ``prev_hash`` still matches its predecessor and every ``record_hash`` still
        verifies, because a prefix of a valid chain is a valid chain. Catching it requires one fact
        the journal does not itself hold -- how many records there should be, or what the tip was:

            head = journal.head()          # store this outside the journal
            ...
            journal.verify(expect_head=head, expect_length=n)

        ``expect_head`` is the stronger anchor: it pins the exact final record, so it also rejects a
        journal truncated and then re-extended back to its original length. ``expect_length`` alone
        only notices a change in count. With neither, this still proves internal consistency -- but
        it is not evidence that nothing was dropped off the end, and an auditor should not read it
        as such.
        """
        if expect_length is not None and len(self._records) != expect_length:
            return False
        expected_prev = _GENESIS_HASH
        for index, record in enumerate(self._records):
            if record.step_index != index or record.prev_hash != expected_prev:
                return False
            if _hash_snapshot(record.portfolio_snapshot) != record.belief_snapshot_hash:
                return False
            if _hash_record(record) != record.record_hash:
                return False
            expected_prev = record.record_hash
        return expect_head is None or expected_prev == expect_head

    def to_json(self, **dumps_kwargs: Any) -> str:
        """Serialize the journal as JSON that a strict parser actually accepts.

        ``allow_nan`` is forced off rather than left at Python's default. json.dumps otherwise
        writes bare ``NaN``/``Infinity``/``-Infinity`` tokens, which no JSON grammar admits: the
        journal round-tripped fine inside Python (json.loads reads those tokens back) while being
        unreadable to every other consumer -- and an audit trail nobody else can read is the one
        thing this class exists to prevent. :func:`_tag_encode` now carries non-finite floats
        through the versioned tag scheme, so this switch should never fire; it is here so that a
        value that escapes the codec fails loudly instead of writing a document that lies about
        being JSON.
        """
        if "allow_nan" in dumps_kwargs:
            raise TypeError("EpistemicJournal.to_json does not accept allow_nan: the output must be valid JSON")
        # The document carries its own anchor. A hash chain cannot detect truncation from the end --
        # a prefix of a valid chain is a valid chain -- which is why head()/verify(expect_head=)
        # exist. An anchor nobody supplies closes nothing, and serialization is the moment the
        # records leave this process's control, so the length and chain tip travel with them and
        # :meth:`from_json` checks both. Deleting a record from the end now fails on load.
        payload = {
            "envelope_version": _ENVELOPE_VERSION,
            "length": len(self._records),
            "head": self.head(),
            "records": [_tag_encode(asdict(r)) for r in self._records],
        }
        return json.dumps(payload, allow_nan=False, **dumps_kwargs)

    @classmethod
    def from_json(cls, s: str) -> EpistemicJournal:
        """Rebuild a journal from :meth:`to_json` output, checking the anchor it carries.

        A document written by :meth:`to_json` is an envelope holding the records plus the length and
        chain tip they had when written; both are verified here, so a record deleted from the end --
        which :meth:`verify` alone cannot see -- is caught on load.

        A bare JSON array is still accepted for documents written before the envelope existed. Those
        carry no anchor, so loading one proves only that the records are internally consistent; pass
        a separately stored ``expect_head`` to :meth:`verify` for the stronger check.
        """
        doc = json.loads(s)
        if isinstance(doc, list):
            return cls([DecisionRecord(**_tag_decode(row)) for row in doc])
        version = doc.get("envelope_version")
        if version != _ENVELOPE_VERSION:
            raise ValueError(f"journal document has envelope_version {version!r}, expected {_ENVELOPE_VERSION!r}")
        journal = cls([DecisionRecord(**_tag_decode(row)) for row in doc["records"]])
        # Only the ANCHOR is checked here -- the count and tip the document declares against what was
        # actually read back. That is the one fact loading can establish and verify() cannot, because
        # a truncated chain is still a valid chain. Chain validity itself stays verify()'s job and
        # the caller's decision: a journal whose records are internally inconsistent must still load
        # so it can be inspected and reported on, rather than becoming unreadable at the moment
        # someone is trying to find out what went wrong with it.
        length, head = int(doc["length"]), str(doc["head"])
        if len(journal) != length or journal.head() != head:
            raise ValueError(
                f"journal does not match the anchor it was written with: the document declares "
                f"{length} record(s) ending at {head[:12]}..., and {len(journal)} record(s) ending "
                f"at {journal.head()[:12]}... were read back -- records were added or removed after "
                "it was written"
            )
        return journal

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)


__all__ = ["DecisionRecord", "EpistemicJournal"]
