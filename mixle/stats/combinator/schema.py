"""Typed structured schema -- named, type-validated fields over a CompositeDistribution.

mixle's identity is composable models of *heterogeneous* data; ``CompositeDistribution`` already models a
product of differently-typed fields, but only positionally (a bare tuple, no names, no validation). A
:class:`Schema` puts a typed record front-end on it: declare ``{name: (type, distribution)}``, score/sample
with ordinary dicts, and have observations validated against their declared types before they reach the
model. The field types reuse the same friendly specs as ``SelectDistribution.by_type`` ('str', 'int',
'float', 'number', ...), numpy-scalar aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.select import _normalize_type_spec, _resolve_alias_group


@dataclass(frozen=True)
class Field:
    """One schema field: a name, the type(s) its value must be, and the distribution that models it."""

    name: str
    type_spec: Any
    dist: Any


class Schema:
    """An ordered set of named, typed fields backed by a :class:`CompositeDistribution`."""

    def __init__(self, fields: list[Field]) -> None:
        if not fields:
            raise ValueError("a schema needs at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate field names: %r" % names)
        self.fields = list(fields)
        self.names = names
        self._types = [_normalize_type_spec(f.type_spec) for f in fields]
        self.composite = CompositeDistribution(tuple(f.dist for f in fields))

    @classmethod
    def from_fields(cls, specs: list[tuple[str, Any, Any]]) -> Schema:
        """Build from ``[(name, type_spec, distribution), ...]``."""
        return cls([Field(n, t, d) for n, t, d in specs])

    def _index(self, name: str) -> int:
        if name not in self.names:
            raise KeyError("no field %r in schema %r" % (name, self.names))
        return self.names.index(name)

    def validate(self, record: dict[str, Any]) -> None:
        """Raise if ``record`` is missing/has extra fields or a value's type violates its declaration."""
        keys = set(record)
        expected = set(self.names)
        if keys != expected:
            raise ValueError("record fields %r do not match schema %r" % (sorted(keys), self.names))
        for field, names in zip(self.fields, self._types):
            value = record[field.name]
            if not isinstance(value, _resolve_alias_group(names)):
                raise TypeError(
                    "field %r expected %s, got %r (%s)" % (field.name, "|".join(names), value, type(value).__name__)
                )

    def to_tuple(self, record: dict[str, Any]) -> tuple[Any, ...]:
        """Validate ``record`` and order its values into the composite's positional tuple."""
        self.validate(record)
        return tuple(record[n] for n in self.names)

    def from_tuple(self, values: tuple[Any, ...]) -> dict[str, Any]:
        """Convert positional composite values back to a named record."""
        return dict(zip(self.names, values))

    def log_density(self, record: dict[str, Any]) -> float:
        """Validated, named log-density: delegates to the backing composite after the type check."""
        return self.composite.log_density(self.to_tuple(record))

    def density(self, record: dict[str, Any]) -> float:
        """Validated, named density delegated to the backing composite."""
        return self.composite.density(self.to_tuple(record))

    def sample(self, seed: int | None = None, size: int | None = None) -> Any:
        """Draw record dict(s) from the model."""
        drawn = self.composite.sampler(seed).sample(size=size)
        if size is None:
            return self.from_tuple(tuple(drawn))
        return [self.from_tuple(tuple(t)) for t in drawn]

    def marginal(self, names: list[str]) -> Schema:
        """The sub-schema over a subset of fields (a marginal of the joint, in the given order)."""
        sub = [self.fields[self._index(n)] for n in names]
        return Schema(sub)

    def field_distribution(self, name: str) -> Any:
        """Return the distribution associated with a named schema field."""
        return self.fields[self._index(name)].dist

    def __str__(self) -> str:
        return "Schema(" + ", ".join("%s:%s" % (f.name, "|".join(t)) for f, t in zip(self.fields, self._types)) + ")"
