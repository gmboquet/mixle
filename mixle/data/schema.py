"""Schema and logical types -- the bridge between external data and the Python types encoders expect.

A deliberately small, *closed* logical type system (not an open type algebra). Each :class:`FieldType`
knows its canonical NumPy dtype and how to coerce a raw value into the Python object the existing
``DataSequenceEncoder`` already consumes (a label for ``Categorical``, an ``np.ndarray`` for ``Vector``,
...). A :class:`Schema` is an ordered tuple of named :class:`Field` s; it can be *derived from a model*
(formalizing the ``fields``/``sources`` duck-probe in the DataFrame adapter) and used to *conform* raw
records (coerce + validate) before encoding -- the thing connectors silently get wrong today.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class FieldType:
    """Base logical type: a canonical NumPy dtype plus a coercion to the encoder-ready Python value."""

    numpy_dtype: Any = np.float64

    def coerce(self, value: Any) -> Any:
        """Return ``value`` in the Python representation expected by the corresponding encoder."""
        return value

    def __repr__(self) -> str:
        return type(self).__name__


class Real(FieldType):
    """A real-valued scalar."""

    numpy_dtype = np.float64

    def coerce(self, value: Any) -> float:
        """Coerce ``value`` to a Python ``float``."""
        return float(value)


class Count(FieldType):
    """A non-negative integer count logical type."""

    numpy_dtype = np.int64

    def coerce(self, value: Any) -> int:
        """Coerce ``value`` to a non-negative Python ``int``, validating exactness first.

        Bare ``int(value)`` used to silently truncate (``1.9`` -> ``1``), silently accept ``True``/
        ``False`` (bool is an ``int`` subclass in Python) as ``1``/``0``, and silently accept negative
        values -- turning a data problem into a wrong-but-plausible-looking observation instead of an
        error. A numeric string (``"42"``) is still accepted, since that is common, legitimate
        connector input (CSV/SQL text columns); a bool, a non-integral value, a negative value, or a
        non-finite (NaN/Inf) value is rejected instead of silently coerced.
        """
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("cannot coerce bool %r to Count (not a genuine count)" % (value,))
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ValueError("cannot coerce %r to Count" % (value,)) from None
        if not np.isfinite(f):
            raise ValueError("cannot coerce non-finite value %r to Count" % (value,))
        if f != int(f):
            raise ValueError("cannot coerce non-integral value %r to Count" % (value,))
        n = int(f)
        if n < 0:
            raise ValueError("cannot coerce negative value %r to Count" % (value,))
        return n


_BOOLEAN_FALSE_STRINGS = frozenset({"false", "f", "no", "n", "0"})
_BOOLEAN_TRUE_STRINGS = frozenset({"true", "t", "yes", "y", "1"})


class Boolean(FieldType):
    """A boolean flag."""

    numpy_dtype = np.bool_

    def coerce(self, value: Any) -> bool:
        """Coerce ``value`` to a Python ``bool``."""
        # bool(value) alone is wrong for the primary use case this module exists for (CSV/SQL/Mongo
        # sources hand back string-typed booleans): bool("False") is True, since any non-empty string
        # is truthy. Parse the common string spellings explicitly; a string that isn't one of them is
        # a real data problem, so it raises rather than guessing.
        if isinstance(value, str):
            s = value.strip().lower()
            if s in _BOOLEAN_FALSE_STRINGS:
                return False
            if s in _BOOLEAN_TRUE_STRINGS:
                return True
            raise ValueError("cannot coerce string %r to Boolean" % value)
        return bool(value)


class Text(FieldType):
    """A free-text string."""

    numpy_dtype = np.object_

    def coerce(self, value: Any) -> str:
        """Coerce ``value`` to ``str``."""
        return str(value)


@dataclass(frozen=True)
class Categorical(FieldType):
    """A categorical label, optionally over a fixed set of ``categories``."""

    categories: tuple[Any, ...] | None = None
    numpy_dtype: Any = np.object_

    def coerce(self, value: Any) -> Any:
        """Return ``value`` after validating membership in ``categories`` when categories are fixed."""
        if self.categories is not None and value not in self.categories:
            raise ValueError("value %r is not one of the categories %r" % (value, self.categories))
        return value


@dataclass(frozen=True)
class Vector(FieldType):
    """A fixed- or free-length real vector."""

    dim: int | None = None
    numpy_dtype: Any = np.float64

    def coerce(self, value: Any) -> np.ndarray:
        """Coerce ``value`` to a one-dimensional ``float64`` array and validate ``dim`` when set."""
        arr = np.asarray(value, dtype=np.float64)
        if self.dim is not None and arr.shape != (self.dim,):
            raise ValueError("expected a length-%d vector, got shape %s" % (self.dim, arr.shape))
        return arr


class Timestamp(FieldType):
    """A point in time (datetime / numpy datetime64 / ISO string / POSIX seconds)."""

    numpy_dtype = np.dtype("datetime64[ns]")

    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` to ``numpy.datetime64`` unless it already has that representation."""
        return np.datetime64(value) if not isinstance(value, np.datetime64) else value


@dataclass(frozen=True)
class Optional(FieldType):
    """A value that may be missing (``None`` passes through; otherwise the inner type coerces)."""

    inner: FieldType = field(default_factory=Real)

    @property
    def numpy_dtype(self) -> Any:
        """Return the NumPy dtype of the wrapped field type."""
        return self.inner.numpy_dtype

    def coerce(self, value: Any) -> Any:
        """Pass missing ``None`` through; otherwise delegate coercion to ``inner``."""
        return None if value is None else self.inner.coerce(value)


@dataclass(frozen=True)
class Nested(FieldType):
    """A sub-record with its own :class:`Schema`."""

    schema: Schema
    numpy_dtype: Any = np.object_

    def coerce(self, value: Any) -> Any:
        """Conform a nested record with this field's child schema."""
        return self.schema.conform_record(value)


@dataclass(frozen=True)
class Field:
    """A named, typed column."""

    name: str
    type: FieldType


@dataclass(frozen=True)
class Schema:
    """An ordered set of typed fields."""

    fields: tuple[Field, ...]
    # True iff this schema has exactly one field whose type is ITSELF container-shaped (Vector/Nested):
    # a raw list/tuple record is then unambiguously that one field's whole value (e.g. a length-k list
    # for a multivariate model's k-dim observation), never "k scalar values for k fields" -- there is
    # only one field. Schema.for_model sets this for schemas it derives from such a model. Defaults to
    # False so a hand-built single-field Vector/Nested Schema keeps conform_record's stricter
    # raise-on-ambiguity behavior (see ConformRecordLengthTest in mixle/tests/data_schema_test.py).
    container_value: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        """Return field names in schema order."""
        return tuple(f.name for f in self.fields)

    def conform_record(self, record: Any) -> Any:
        """Coerce one record (a scalar for a 1-field schema, else a tuple/dict) to the schema's types."""
        if len(self.fields) == 1 and not isinstance(record, (tuple, list, dict)):
            return self.fields[0].type.coerce(record)
        if len(self.fields) == 1 and self.container_value and isinstance(record, (tuple, list)):
            # this schema's one field IS itself container-shaped -- the whole record is unambiguously
            # that field's raw value, so skip the "N values for N fields" interpretation entirely
            # (there is only one field; disambiguation on length is not needed here).
            return self.fields[0].type.coerce(record)
        if isinstance(record, dict):
            return {f.name: f.type.coerce(record[f.name]) for f in self.fields}
        # zip() silently stops at the shorter side, so an unchecked zip here would silently drop
        # trailing values (or fields) instead of raising -- the exact "connectors silently get wrong"
        # failure mode this module exists to prevent. Materialize once and check length explicitly.
        values = record if isinstance(record, (tuple, list)) else list(record)
        if len(values) != len(self.fields):
            raise ValueError(
                "record has %d value(s) but schema %r has %d field(s)" % (len(values), self.names, len(self.fields))
            )
        return tuple(f.type.coerce(v) for f, v in zip(self.fields, values))

    def conform(self, records: Any) -> list[Any]:
        """Coerce every record in ``records`` to this schema (raising clear errors on mismatch)."""
        return [self.conform_record(r) for r in records]

    @staticmethod
    def for_model(model: Any) -> Schema:
        """Best-effort schema a model expects, from its ``fields``/``sources`` + child distributions.

        Formalizes the duck-probe in the DataFrame adapter: a record/composite model exposes ``fields``
        and ``sources`` plus child distributions whose support fixes each field's logical type; a bare
        leaf yields a single field typed by its support (discrete -> Count, continuous -> Real, ...).
        """
        names = getattr(model, "fields", None)
        dists = getattr(model, "dists", None)  # the independent factors of a composite/record (not mixture comps)
        if dists is not None:
            labels = (
                [str(n) for n in names]
                if (names is not None and len(names) == len(dists))
                else [f"field_{i}" for i in range(len(dists))]
            )
            fields = tuple(Field(n, _type_for(c)) for n, c in zip(labels, dists))
        else:
            fields = (Field(getattr(model, "name", None) or "value", _type_for(model)),)
        # a derived schema of exactly one Vector/Nested field expects its raw record to BE that
        # field's whole value (e.g. a multivariate model's raw list/ndarray observation) -- see
        # conform_record's container_value branch.
        container_value = len(fields) == 1 and isinstance(fields[0].type, (Vector, Nested))
        return Schema(fields, container_value=container_value)


def _type_for(dist: Any) -> FieldType:
    """Infer a logical type from a distribution's support (capability-based, with safe fallbacks).

    A ``Discrete`` capability alone is not enough to conclude ``Count``: a categorical label set and a
    genuine integer count are BOTH ``Discrete`` (finite support), so this module's actual job is
    telling them apart. When the capability check says a distribution is discrete, the distribution's
    OWN declared support values are inspected (via its enumerator -- the codebase's uniform "what is
    my support" surface) and classified by what they actually are, before ever defaulting to ``Count``
    -- see :func:`_type_for_discrete_support`. That ordering is the fix: a ``CategoricalDistribution``
    over string labels (e.g. ``{"red": .5, "blue": .5}``) is ``Discrete`` (finite support) but its
    support values are not integers, so it must resolve to ``Categorical``, never ``Count``.
    """
    try:
        from mixle.capability import Continuous, Discrete, supports

        if supports(dist, Discrete):
            return _type_for_discrete_support(dist)
        if supports(dist, Continuous):
            return Real()
    except Exception:  # noqa: BLE001
        pass
    cls = type(dist).__name__.lower()
    if "categor" in cls:
        return Categorical()
    if "multivariate" in cls or "gaussian" in cls and "diag" in cls:
        return Vector()
    return Real()


def _type_for_discrete_support(dist: Any, *, peek: int = 16) -> FieldType:
    """Classify a ``Discrete`` distribution's logical type from its own declared support values.

    Draws up to ``peek`` values from ``dist.enumerator()`` (lazily -- ``itertools.islice`` never pulls
    more than ``peek`` items, so this is safe even for the unbounded-support enumerators, e.g. Poisson
    or negative-binomial, that stream in descending-probability order rather than materializing their
    whole support). All-``bool`` support -> :class:`Boolean`; all non-boolean-integer support ->
    :class:`Count`; anything else (strings, tuples, mixed types, ...) -> :class:`Categorical`. Falls
    back to :class:`Count` -- the prior behaviour for every ``Discrete`` distribution -- only when the
    support cannot be inspected at all (no working enumerator), so this never regresses a distribution
    this module previously handled correctly.
    """
    enumerator = getattr(dist, "enumerator", None)
    values: list[Any] = []
    if callable(enumerator):
        try:
            values = [v for v, _ in itertools.islice(enumerator(), peek)]
        except Exception:  # noqa: BLE001
            values = []
    if not values:
        return Count()
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return Boolean()
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        return Count()
    return Categorical()
