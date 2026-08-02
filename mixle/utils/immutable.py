"""An immutable stand-in for a value a receipt must record but cannot convert.

Several receipt/snapshot paths recursively convert a payload into immutable form: lists become
tuples, dicts become ``MappingProxyType``, arrays are copied and sealed. A *custom object* has no
immutable counterpart to convert into, and those paths used to fall back to ``copy.deepcopy`` plus a
warning when the copy failed or returned the original. That fallback fails the contract three ways
at once (MXR-080-1872):

* ``copy.deepcopy`` invokes ``__deepcopy__``/``__reduce__``, which is caller-defined code. Running it
  is a side effect, and a forensic snapshot is exactly the wrong place to have one.
* a ``__deepcopy__`` that returns ``self`` restores the alias the snapshot exists to sever, and a
  warning does not un-alias it.
* even a successful deep copy is still a *mutable* object, so what the receipt retains can be edited
  through it.

:func:`opaque_snapshot` replaces the copy with a record: the object's type and its own recorded
attribute state, deeply frozen by the caller's freezer. Reading ``__dict__`` and ``__slots__`` runs
no dunder hook, the result shares no mutable structure with the caller, and it is immutable. What is
lost is the object's *behavior* -- the snapshot is evidence about the value, not a substitute for it.
That is the honest trade: a payload with no immutable counterpart was already out of contract for the
paths that call this.
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from types import MappingProxyType
from typing import Any

__all__ = [
    "OpaqueSnapshot",
    "detach_receipt_container",
    "freeze_receipt_container",
    "is_immutable_atom",
    "opaque_snapshot",
    "plain_receipt_container",
]

_IN_PROGRESS = threading.local()

_IMMUTABLE_ATOMS: tuple[type, ...] = (
    bool,
    str,
    bytes,
    Number,  # int, float, complex, Decimal, Fraction, and numpy scalars -- all immutable
    Enum,
    type,
    range,
    slice,
    datetime.date,  # also covers datetime.datetime
    datetime.time,
    datetime.timedelta,
)


def is_immutable_atom(value: Any) -> bool:
    """Whether ``value`` is already immutable and can be retained as itself.

    A freezer must ask this before falling back to :func:`opaque_snapshot`. The recursive freezers
    that call it enumerate the *containers* they convert and reach the fallback for everything else,
    which previously routed through ``copy.deepcopy`` -- and deep-copying an ``int`` returns the
    ``int``. Replacing that fallback without this predicate turned every scalar the container branches
    did not name into a snapshot of itself.
    """
    return value is None or isinstance(value, _IMMUTABLE_ATOMS)


@dataclass(frozen=True)
class OpaqueSnapshot:
    """An immutable, non-aliased record of an object with no immutable counterpart.

    ``state`` is ``None`` when the object exposed no readable attribute state, or when it was reached
    through a reference cycle and has already been recorded higher up this walk.
    """

    type_name: str
    module: str
    state: Mapping[str, Any] | None = None

    def __repr__(self) -> str:
        names = "" if not self.state else " " + " ".join(sorted(self.state))
        return f"<OpaqueSnapshot {self.module}.{self.type_name}{names}>"


def _recorded_state(value: Any) -> dict[str, Any]:
    """Return the object's own attribute state without invoking any caller-defined hook."""
    state: dict[str, Any] = {}
    try:
        instance_dict = object.__getattribute__(value, "__dict__")
    except AttributeError:
        instance_dict = None
    if isinstance(instance_dict, Mapping):
        state.update({name: item for name, item in instance_dict.items() if isinstance(name, str)})
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if not isinstance(name, str) or name in ("__dict__", "__weakref__") or name in state:
                continue
            try:
                state[name] = object.__getattribute__(value, name)
            except AttributeError:  # a declared slot that was never assigned
                continue
    return state


def opaque_snapshot(value: Any, freeze: Callable[[Any], Any]) -> OpaqueSnapshot:
    """Record ``value`` as an immutable snapshot, freezing its attribute state with ``freeze``.

    ``freeze`` is the caller's own recursive freezer, so nested containers are converted by the same
    rules the rest of that payload was, and a nested custom object lands back here.
    """
    cls = type(value)
    active = getattr(_IN_PROGRESS, "ids", None)
    if active is None:
        active = set()
        _IN_PROGRESS.ids = active
    key = id(value)
    if key in active:
        # A reference cycle. The object is already being recorded further up this walk, so a
        # state-less marker terminates the recursion without losing the fact that it was here.
        return OpaqueSnapshot(type_name=cls.__name__, module=getattr(cls, "__module__", "?"))
    active.add(key)
    try:
        state = _recorded_state(value)
        frozen = {name: freeze(item) for name, item in state.items()}
    finally:
        active.discard(key)
    return OpaqueSnapshot(
        type_name=cls.__name__,
        module=getattr(cls, "__module__", "?"),
        state=MappingProxyType(frozen) if frozen else None,
    )


def freeze_receipt_container(value: Any) -> Any:
    """Detach and seal the *containers* in a receipt field, preserving element identity.

    This is the structural half of receipt integrity, and it is the one the aliasing defect actually
    needs: a caller hands a ``dict`` or ``list`` to a frozen dataclass, the dataclass stores the
    reference, and the caller mutates its own copy afterwards -- rewriting a decision that had
    already been recorded. Copying severs that alias; the read-only view stops anyone editing the
    receipt through the field itself.

    Elements are passed through by identity rather than judged. A receipt may legitimately hold a
    frozen dataclass, a distribution, or an array, and refusing those would reject the receipts the
    library actually produces. Where a receipt additionally advertises JSON-serializable output, the
    stricter per-module freezers keep that separate contract -- this one does not claim it.

    ``Mapping -> MappingProxyType`` is deliberately chosen over a plain copy because it still
    compares equal to the original dict, so consumers that compare receipts keep working. Sequences
    become tuples, which does NOT compare equal to a list; call sites converted to this helper were
    checked for consumers that mutate or type-test the field (MXR-080-1876).
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_receipt_container(item) for key, item in value.items()})
    if isinstance(value, (str, bytes, bytearray)):
        return bytes(value) if isinstance(value, bytearray) else value
    if isinstance(value, (list, tuple)):
        return tuple(freeze_receipt_container(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_receipt_container(item) for item in value)
    return value


def plain_receipt_container(value: Any) -> Any:
    """Undo :func:`freeze_receipt_container` for an ``as_dict``-style JSON-compatible view."""
    if isinstance(value, Mapping):
        return {key: plain_receipt_container(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (tuple, frozenset)):
        return [plain_receipt_container(item) for item in value]
    return value


def detach_receipt_container(value: Any) -> Any:
    """Copy a receipt's containers, preserving their concrete types.

    The weaker sibling of :func:`freeze_receipt_container`, for fields whose *type* is load-bearing.
    It severs the caller's alias -- the half of the defect that lets a mutation after construction
    rewrite a recorded decision -- without converting ``dict`` to ``mappingproxy`` or ``list`` to
    ``tuple``.

    Use it where the concrete type is observable: a field that is content-addressed or serialized
    with type fidelity (``mixle.epistemic.journal`` deliberately distinguishes a tuple from an
    equal-valued list, and hashes what it stores), a field a consumer type-tests with
    ``isinstance(x, dict)``, or a field on a record that gets pickled -- ``mappingproxy`` cannot be
    pickled or deep-copied. Applying the stronger freezer to those fields broke the journal's hash
    chain and a training receipt's own validation (MXR-080-1876).

    The receipt's own field stays writable, which is a real and stated limitation: this defends
    against the caller, not against the holder.
    """
    if isinstance(value, Mapping):
        return {key: detach_receipt_container(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, list):
        return [detach_receipt_container(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_receipt_container(item) for item in value)
    if isinstance(value, set):
        return {detach_receipt_container(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(detach_receipt_container(item) for item in value)
    return value
