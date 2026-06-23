"""Back-compat facade for the smart-enumeration machinery.

The implementation was split by concern (it had grown into one ~1100-line module mixing three
distinct things):

* generic stream primitives        -> :mod:`pysp.enumeration.streams`
* best-first / product search      -> :mod:`pysp.enumeration.best_first`
* the quantized seek / unrank index -> :mod:`pysp.enumeration.quantization.seek`

This module re-exports them so ``from pysp.enumeration.algorithms import X`` keeps working unchanged.
New code can import from the modules above directly. See
:class:`pysp.stats.compute.pdist.DistributionEnumerator` for the enumeration contract.
"""

from pysp.enumeration.best_first import (
    LengthFrontierMerge,
    ProductEnumerator,
    best_first_union,
    best_first_union_max,
    bounded_best_first_union_index,
    frontier_merge,
    sound_top_k,
)
from pysp.enumeration.quantization.seek import (
    LazyQuantizedEnumerationIndex,
    QuantizedCrossIndex,
    QuantizedEnumerationIndex,
    quantized_index,
)
from pysp.enumeration.streams import (
    BufferedStream,
    freeze,
    merge_enumerators,
    supports_enumeration,
)

__all__ = [
    "BufferedStream",
    "freeze",
    "merge_enumerators",
    "supports_enumeration",
    "ProductEnumerator",
    "LengthFrontierMerge",
    "frontier_merge",
    "best_first_union",
    "best_first_union_max",
    "bounded_best_first_union_index",
    "sound_top_k",
    "QuantizedEnumerationIndex",
    "LazyQuantizedEnumerationIndex",
    "QuantizedCrossIndex",
    "quantized_index",
]
