"""Shared bases for the single-child combinators (censoring/truncation/survival/tilt/...).

These combinators all wrap a single child distribution and are, mechanically, thin delegators:
their accumulators forward sufficient-statistic bookkeeping to one child accumulator, and their
data encoders forward to one child encoder plus a small per-row "mask"/statistic column. The two
bases here factor out that boilerplate so each combinator only states what is genuinely its own.

* :class:`SingleChildAccumulator` (finding B7) owns the delegation trio
  (``combine``/``value``/``from_value``/``scale``) and the ``key_merge``/``key_replace`` pooling for
  the common case where ``value()`` is the *bare* child value. The child accumulator lives on the
  attribute named by ``_child_attr`` (``"base_accumulator"`` by default). Combinators that carry
  extra scalar sufficient statistics alongside the child value (hurdle, zero-inflated, tilt) override
  the statistic-bearing methods but still inherit the key-pooling delegation.

* :class:`MaskedBaseEncoder` (finding B8) owns the identical ``__init__``/``__str__``/``__eq__`` of
  the per-combinator data encoders; each subclass overrides only :meth:`_extra_columns`, the per-row
  extra column(s) appended to the base encoding.
"""

from collections.abc import Sequence
from typing import Any

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    SequenceEncodableStatisticAccumulator,
)


class SingleChildAccumulator(SequenceEncodableStatisticAccumulator):
    """Delegate sufficient-statistic bookkeeping to a single child accumulator.

    Subclasses set ``_child_attr`` (the attribute name of the child accumulator) and implement the
    statistic-gathering methods (``update``/``seq_update``/``initialize``/``seq_initialize``/
    ``acc_to_encoder``). The delegation trio and key-pooling below assume ``value()`` is the bare
    child value; combinators whose ``value()`` bundles extra scalars override the relevant methods.
    """

    _child_attr: str = "base_accumulator"

    @property
    def _child(self) -> SequenceEncodableStatisticAccumulator:
        return getattr(self, self._child_attr)

    def combine(self, suff_stat: Any) -> "SingleChildAccumulator":
        self._child.combine(suff_stat)
        return self

    def value(self) -> Any:
        return self._child.value()

    def from_value(self, x: Any) -> "SingleChildAccumulator":
        self._child.from_value(x)
        return self

    def scale(self, c: float) -> "SingleChildAccumulator":
        self._child.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        self._child.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        self._child.key_replace(stats_dict)


class MaskedBaseEncoder(DataSequenceEncoder):
    """Encode observations via a child encoder plus per-row mask/statistic column(s).

    Subclasses set the child encoder on ``self.base_encoder`` (the default ``__init__`` accepts it
    directly) and override :meth:`_extra_columns` to return the per-row extra column(s) appended to
    the base encoding. ``__str__``/``__eq__`` derive their label from the concrete class name.
    """

    def __init__(self, base_encoder: DataSequenceEncoder) -> None:
        self.base_encoder = base_encoder

    def __str__(self) -> str:
        return "%s(%s)" % (type(self).__name__, str(self.base_encoder))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and other.base_encoder == self.base_encoder

    def _extra_columns(self, x: Sequence[Any]) -> tuple[Any, ...]:
        """Return the per-row extra column(s) appended to the base encoding (override)."""
        return ()

    def seq_encode(self, x: Sequence[Any]) -> tuple[Any, ...]:
        return (self.base_encoder.seq_encode(list(x)), *self._extra_columns(x))
