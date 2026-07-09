"""Persistent what-works prior over structural design families.

:class:`~mixle.task.edge.DesignModel` already ledgers every evaluated design point with an arbitrary
tag dict, so a caller can tag each accepted structural recipe with which FAMILY it belongs to (a
quotient leaf vs a plain head, a richer factorization vs a simpler one, or
capability-conditioned recipes). This module is the thin
query layered on that existing ledger: rank the families seen so far by their mean recorded quality,
so the NEXT round's structural-search proposal starts from a sharper prior instead of from scratch --
"what has actually worked" persisted across rounds and design searches, not re-derived each time.

    record_accepted_recipe(design, point, quality, violations, family="quotient_leaf")
    rank_design_families(design)                    # [("quotient_leaf", 0.91), ("plain_head", 0.62)]
    best_family(design)                              # "quotient_leaf"

An untried candidate family (never recorded) has no evidence and ranks below every family that has
been recorded, via an explicit, named ``default_score`` -- never silently tied with a proven winner.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.task.edge import DesignModel


def record_accepted_recipe(
    design: DesignModel,
    point: Any,
    quality: float,
    violations: Sequence[float],
    *,
    family: str,
    fingerprint: Sequence[float] | None = None,
    **tag: Any,
) -> None:
    """Record an accepted structural recipe under its ``family`` tag -- the training signal for
    :func:`rank_design_families`. A thin, named wrapper over ``DesignModel.add`` so callers do not
    have to remember which tag key the prior reads."""
    design.add(point, quality, violations, fingerprint=fingerprint, family=family, **tag)


def rank_design_families(
    design: DesignModel,
    *,
    tag_key: str = "family",
    candidates: Sequence[str] | None = None,
    default_score: float = float("-inf"),
) -> list[tuple[str, float]]:
    """Rank every family tag recorded in ``design`` by its mean quality, best first.

    ``candidates``, if given, are ALSO included in the ranking even if never recorded -- an untried
    family gets ``default_score`` (``-inf`` by default: no evidence ranks strictly below any recorded
    family, however weak, rather than being silently omitted or tied with a proven winner).
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for tag, q in zip(design.tags, design.quality):
        fam = tag.get(tag_key)
        if fam is not None:
            buckets[str(fam)].append(float(q))

    families = set(buckets) | (set(str(c) for c in candidates) if candidates else set())
    scored = [(fam, float(np.mean(buckets[fam])) if fam in buckets else default_score) for fam in families]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def best_family(design: DesignModel, *, tag_key: str = "family") -> str | None:
    """The single top-ranked recorded family, or ``None`` if nothing has been recorded yet."""
    ranked = rank_design_families(design, tag_key=tag_key)
    return ranked[0][0] if ranked else None


__all__ = ["best_family", "rank_design_families", "record_accepted_recipe"]
