"""The enumeration concern — one home for "what can be enumerated, and how".

Enumeration is not a property of distributions; it is a capability shared by distributions,
relations, quantized objects, and any combinatorial model. Anything that can iterate its support in
**descending probability** implements :class:`DistributionEnumerator` (the contract) and reports the
:class:`Enumerable` capability; anything finite-or-structural can additionally be **unranked by integer
rank** (:class:`RankableByIndex`) through the count-budget seek index.

This module gathers that concern in one place — the contract, the capability lens, the k-best
algorithms, the count-budget unranking, and the count semiring — so you can read off *what
enumeration is* and *who participates* without hunting through ``utils``/``stats``. The implementations
live here, split by concern: generic stream primitives in :mod:`~mixle.enumeration.streams`, best-first
/ product search in :mod:`~mixle.enumeration.best_first`, the quantized seek/unrank index and the
structural count-budget index in :mod:`~mixle.enumeration.quantization`, and the k-best combinatorial
enumerators in :mod:`~mixle.enumeration.assignment` / :mod:`~mixle.enumeration.spanning`. Only the
*contract* (:class:`DistributionEnumerator`) lives in the compute layer and is re-exported here.
``mixle.enumeration.algorithms`` remains as a back-compat re-export of the stream / best-first / index
names. Layout per ``docs/ARCHITECTURE.md``.

Who plugs in: every finite/countable leaf (Categorical, Poisson, …), the combinators over enumerable
children (Sequence, Composite, Mixture, …), the graph/ranking families (Markov chains, Mallows,
Chow-Liu trees, spanning trees), and ``mixle.relations.Relation`` — all via the same ``enumerator()``.
"""

from __future__ import annotations

# --- the capability lens (detect + dispatch) ---
from mixle.capability import (
    Enumerable,
    FiniteSupport,
    RankableByIndex,
    supports,
    top_k,
)

# --- the k-best / descending-probability algorithms ---
from mixle.enumeration.algorithms import (
    LazyQuantizedEnumerationIndex,
    ProductEnumerator,
    QuantizedEnumerationIndex,
    best_first_union,
    merge_enumerators,
    quantized_index,
    sound_top_k,
    supports_enumeration,
)

# --- count / threshold / unrank for arbitrary autoregressive (next_logprobs) models ---
from mixle.enumeration.autoregressive import AutoregressiveEnumerable, autoregressive_count_index
from mixle.enumeration.density_rank import DensityRankResult, density_rank
from mixle.enumeration.envelope import AREnvelopeIndex, LatticeEnvelopeIndex

# --- HMM state paths: exact A* enumeration + the quantized random-access path index ---
from mixle.enumeration.hmm_paths import HMMPathIndex, hmm_best_paths
from mixle.enumeration.model_enumeration import (
    beam_search,
    best_first_decode,
    quantized_best_first_decode,
    top_k_scored,
)

# NOTE: model_enumeration.best_first (the generic engine function) is deliberately NOT re-exported
# here despite being one of the module's own documented "four entry points" -- this package also has
# a *submodule* named mixle.enumeration.best_first (best_first.py), and binding a same-named function
# in this __init__ shadows `import mixle.enumeration.best_first` for anyone reaching the submodule
# directly (verified: it silently resolves to the function instead, breaking that import). Reach the
# generic engine via `from mixle.enumeration.model_enumeration import best_first` instead.
# --- the count-budget seek / unrank index + the count semiring (rank-by-index machinery) ---
from mixle.enumeration.quantization.core import count_budget_index, logit_error_bucket_slack
from mixle.enumeration.quantization.semiring import CountSemiring, DecomposableSemiring, TropicalSemiring
from mixle.enumeration.rescore import RescoredIndex
from mixle.enumeration.seek_index import SeekIndex

# --- the contract (implemented by distributions AND relations) ---
from mixle.stats.compute.pdist import (
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
    "SeekIndex",
    "logit_error_bucket_slack",
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
    "best_first_decode",
    "beam_search",
    "top_k_scored",
    # autoregressive (next_logprobs) models: count / threshold / unrank
    "AutoregressiveEnumerable",
    "autoregressive_count_index",
    "AREnvelopeIndex",
    "LatticeEnvelopeIndex",
    "RescoredIndex",
    # HMM path enumeration (non-decomposable family): exact A* head + quantized random-access index
    "hmm_best_paths",
    "HMMPathIndex",
]
