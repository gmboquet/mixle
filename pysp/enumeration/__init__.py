"""The enumeration concern — one home for "what can be enumerated, and how".

Enumeration is not a property of distributions; it is a capability shared by distributions,
relations, quantized objects, and any combinatorial model. Anything that can iterate its support in
**descending probability** implements :class:`DistributionEnumerator` (the contract) and reports the
:class:`Enumerable` capability; anything finite-or-structural can additionally be **unranked by integer
rank** (:class:`RankableByIndex`) through the count-budget seek index.

This module gathers that concern in one place — the contract, the capability lens, the k-best
algorithms, the count-budget unranking, and the count semiring — so you can read off *what
enumeration is* and *who participates* without hunting through ``utils``/``stats``. The implementations
live here, split by concern: generic stream primitives in :mod:`~pysp.enumeration.streams`, best-first
/ product search in :mod:`~pysp.enumeration.best_first`, the quantized seek/unrank index and the
structural count-budget index in :mod:`~pysp.enumeration.quantization`, and the k-best combinatorial
enumerators in :mod:`~pysp.enumeration.assignment` / :mod:`~pysp.enumeration.spanning`. Only the
*contract* (:class:`DistributionEnumerator`) lives in the compute layer and is re-exported here.
``pysp.enumeration.algorithms`` remains as a back-compat re-export of the stream / best-first / index
names. Layout per ``docs/ARCHITECTURE.md``.

Who plugs in: every finite/countable leaf (Categorical, Poisson, …), the combinators over enumerable
children (Sequence, Composite, Mixture, …), the graph/ranking families (Markov chains, Mallows,
Chow-Liu trees, spanning trees), and ``pysp.relations.Relation`` — all via the same ``enumerator()``.
"""

from __future__ import annotations

# --- the capability lens (detect + dispatch) ---
from pysp.capability import (
    Enumerable,
    FiniteSupport,
    RankableByIndex,
    supports,
    top_k,
)

# --- the k-best / descending-probability algorithms ---
from pysp.enumeration.algorithms import (
    LazyQuantizedEnumerationIndex,
    ProductEnumerator,
    QuantizedEnumerationIndex,
    best_first_union,
    merge_enumerators,
    quantized_index,
    sound_top_k,
    supports_enumeration,
)
from pysp.enumeration.density_rank import DensityRankResult, density_rank

# --- exact descending-probability enumeration of HMM state paths (list-Viterbi / A*) ---
from pysp.enumeration.hmm_paths import hmm_best_paths
from pysp.enumeration.model_enumeration import quantized_best_first_decode

# --- the count-budget seek / unrank index + the count semiring (rank-by-index machinery) ---
from pysp.enumeration.quantization.core import count_budget_index
from pysp.enumeration.quantization.semiring import CountSemiring, DecomposableSemiring, TropicalSemiring

# --- the contract (implemented by distributions AND relations) ---
from pysp.stats.compute.pdist import (
    DistributionEnumerator,
    EnumerationError,
    child_enumerator,
)

__all__ = [
    # capability lens
    "Enumerable",
    "FiniteSupport",
    "RankableByIndex",
    "supports",
    "top_k",
    # contract
    "DistributionEnumerator",
    "EnumerationError",
    "child_enumerator",
    "supports_enumeration",
    "DensityRankResult",
    "density_rank",
    # rank-by-index (count-budget unranking)
    "count_budget_index",
    "quantized_index",
    "QuantizedEnumerationIndex",
    "LazyQuantizedEnumerationIndex",
    "CountSemiring",
    "DecomposableSemiring",
    "TropicalSemiring",
    # k-best algorithms
    "best_first_union",
    "merge_enumerators",
    "ProductEnumerator",
    "sound_top_k",
    "quantized_best_first_decode",
    # HMM path enumeration (non-decomposable family, served by A* not the count index)
    "hmm_best_paths",
]
