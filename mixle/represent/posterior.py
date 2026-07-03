"""Posterior retrieval -- nearest neighbours by what the MODEL believes, not by raw-feature cosine.

The differentiated half of model-based RAG: fit a mixture to heterogeneous records
(:func:`mixle.propose` will happily do it), and retrieval similarity becomes *posterior affinity* --
two records are close when the model's field-restricted latent posteriors agree (per-field
Bhattacharyya, the ``balanced`` affinity from :mod:`mixle.utils.hvis`), with the 1-nat **evidence cap**
so a single wildly-different field can testify "these differ" but can never single-handedly veto a
pair that every other field matches. Raw-feature cosine has neither property: it weights fields by
their numeric scale, and one hot field dominates the dot product::

    m = mixle.propose(records, fit=True)
    r = PosteriorRetriever(m.fitted, records)          # any mixture over the records works
    r.retrieve(query, k=5)                             # [(corpus index, log-affinity), ...]

Honest cost note: affinities are computed jointly over ``corpus + queries`` through the model's
per-field likelihoods -- linear in rows for the model passes but quadratic for the affinity block, so
this is built for corpora in the thousands, not millions. For big-corpus first-stage recall use
:func:`mixle.represent.fit_embedder` and re-rank the shortlist here.
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
        # pre-built per-field factors ride through the affinity= slot (posterior_mat is unused then)
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
