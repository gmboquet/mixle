"""Information-gain retrieval over substrate items.

Score substrate items by how much they would actually move a belief, not by how
textually similar they are to the query.

:func:`~mixle.substrate.retrieve.retrieve` ranks by cosine/lexical similarity -- a sound default, but many
similar-looking items can carry the *same* evidence (redundant), while a single differently-worded item can
be decisive. :func:`eig_retrieve` instead scores each candidate by the entropy it would actually remove from a
given :class:`~mixle.inference.belief.BeliefState` if assimilated, and greedily picks the highest-gain item
each round, updating the running belief before scoring what remains -- so a second item redundant with the
first correctly scores near zero the next round. Experiment-design workflows can reuse the same greedy EIG
scorer; it is written once, here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.inference.belief import BeliefState
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.retrieve import Retrieval


def eig_retrieve(
    substrate: Substrate,
    belief: BeliefState,
    evidence_fn: Callable[[SubstrateItem], Any],
    *,
    k: int = 8,
    kind: str | None = None,
    scope: str | None = None,
) -> Retrieval:
    """Greedily pick up to ``k`` substrate items by expected posterior-entropy reduction against ``belief``.

    ``evidence_fn(item)`` turns a candidate item into whatever ``belief.update(...)`` expects (e.g. a
    per-hypothesis log-likelihood vector for a :class:`~mixle.inference.belief.CategoricalBelief`). Each
    round, every remaining candidate is scored by ``current_belief.entropy() - updated_belief.entropy()``;
    the best-scoring item is taken, the running belief moves to its post-update state, and scoring repeats
    against the shrunk pool -- so an item whose evidence is redundant with an already-picked item scores
    near zero on its next look, the direct fix for similarity retrieval pulling in near-duplicates. Items
    whose ``evidence_fn`` raises (no usable evidence) are skipped, not fatal. Returned as a
    :class:`~mixle.substrate.retrieve.Retrieval` (``query`` is a fixed marker, not a text query) so it
    composes with the same ``to_context``/``by_kind`` surface as cosine retrieval.
    """
    pool = [it for it in substrate.all(scope=scope) if kind is None or it.kind == kind]
    chosen: list[SubstrateItem] = []
    scores: list[float] = []
    current = belief
    remaining = list(pool)
    while remaining and len(chosen) < k:
        best_idx: int | None = None
        best_gain = -np.inf
        best_next: BeliefState | None = None
        for idx, item in enumerate(remaining):
            try:
                nxt = current.update(evidence_fn(item))
            except Exception:  # noqa: BLE001 -- an item with no usable evidence is skipped, not fatal
                continue
            gain = float(current.entropy() - nxt.entropy())
            if gain > best_gain:
                best_idx, best_gain, best_next = idx, gain, nxt
        if best_idx is None:
            break
        chosen.append(remaining.pop(best_idx))
        scores.append(best_gain)
        current = best_next
    return Retrieval(query="<information-gain>", items=chosen, scores=scores)
