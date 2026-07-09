"""Posterior retrieval by fitted-model affinity rather than raw-feature cosine.

Fit a mixture to heterogeneous records and retrieval similarity becomes
*posterior affinity*: two records are close when the model's
field-restricted latent posteriors agree. The implementation uses the per-field
Bhattacharyya-style ``balanced`` affinity from :mod:`mixle.utils.hvis`, with an
evidence cap so one inconsistent field can contribute negative evidence without
dominating every other field. Raw-feature cosine has neither property: it
weights fields by numeric scale, and one high-variance field can dominate the
dot product::

    m = mixle.propose(records, fit=True)
    r = PosteriorRetriever(m.fitted, records)          # any mixture over the records works
    r.retrieve(query, k=5)                             # [(corpus index, log-affinity), ...]

Cost note: affinities are computed jointly over ``corpus + queries`` through
the model's per-field likelihoods. Model passes are linear in rows, while the
affinity block is quadratic, so this is intended for moderate corpora. For
large-corpus first-stage recall, use :func:`mixle.represent.fit_embedder` and
rerank the shortlist here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.utils.hvis.affinity import balanced_factors, model_log_affinity


class PosteriorRetriever:
    """Retrieve over raw heterogeneous records by the fitted mixture's posterior affinity."""

    def __init__(
        self,
        model: Any,
        corpus: Any,
        *,
        evidence_cap: float | None = 1.0,
        field_weights: Any = None,
    ) -> None:
        if not (hasattr(model, "components") and hasattr(model, "log_w")):
            raise TypeError("PosteriorRetriever needs a fitted mixture (components + log_w)")
        self.model = model
        self.corpus = list(corpus)
        if len(self.corpus) < 2:
            raise ValueError("PosteriorRetriever needs a corpus of at least 2 records")
        self.evidence_cap = evidence_cap
        self.field_weights = field_weights

    def _log_affinity(self, rows: list) -> np.ndarray:
        factors = balanced_factors(self.model, rows, field_weights=self.field_weights)
        # Pre-built per-field factors use the affinity slot; posterior_mat is unused in that path.
        return model_log_affinity(None, None, affinity=factors, evidence_cap=self.evidence_cap)

    def affinity_matrix(self) -> np.ndarray:
        """The corpus's dense ``(n, n)`` log-affinity matrix (diagonal ``-inf``)."""
        return self._log_affinity(self.corpus)

    def retrieve(self, query: Any, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` corpus records for one query: ``[(corpus_index, log_affinity), ...]`` best first."""
        return self.retrieve_batch([query], k=k)[0]

    def retrieve_batch(self, queries: Any, k: int = 5) -> list[list[tuple[int, float]]]:
        """Top-``k`` per query, computed in one joint pass over ``corpus + queries``."""
        qs = list(queries)
        n = len(self.corpus)
        log_aff = self._log_affinity([*self.corpus, *qs])
        out: list[list[tuple[int, float]]] = []
        for j in range(len(qs)):
            row = log_aff[n + j, :n]  # query row against corpus columns only
            top = np.argsort(-row)[: int(k)]
            out.append([(int(i), float(row[i])) for i in top])
        return out
