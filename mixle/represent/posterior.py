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

from mixle.represent.identity import RetrievalIdentity, exact_count, records_digest
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
        self._model = model
        # A tuple, not a list: `self.corpus` was public and mutable, so `r.corpus.append(rec)` after
        # construction silently changed what every previously returned index MEANT, with no error and
        # no way for a holder of an earlier result to notice (MXR-080-1906). Indices are only
        # interpretable against a fixed corpus, so the corpus is fixed.
        self._corpus = tuple(corpus)
        if len(self._corpus) < 2:
            raise ValueError("PosteriorRetriever needs a corpus of at least 2 records")
        self._evidence_cap = evidence_cap
        # Copied and frozen for the same reason: `field_weights` was stored by reference and fed
        # straight into `balanced_factors`, so a caller who kept their array could reweight the
        # similarity function of an already-built retriever.
        self._field_weights = _own_field_weights(field_weights)
        self._identity = RetrievalIdentity(
            model=f"{type(model).__module__}.{type(model).__name__}",
            corpus_size=len(self._corpus),
            corpus_digest=records_digest(self._corpus),
        )

    @property
    def model(self) -> Any:
        """The fitted mixture whose posterior defines affinity (read-only; see :attr:`identity`)."""
        return self._model

    @property
    def corpus(self) -> tuple:
        """The fixed corpus every returned index refers to, as an immutable tuple.

        The *elements* are the caller's own record objects and are deliberately NOT deep-copied:
        records here are arbitrary heterogeneous payloads, copying them could be arbitrarily
        expensive, and a mixture's records are routinely shared with the model that was fitted on
        them. What is owned is the sequence -- its length and order, which is what an index means.
        """
        return self._corpus

    @property
    def evidence_cap(self) -> float | None:
        """The per-field negative-evidence cap (read-only)."""
        return self._evidence_cap

    @property
    def field_weights(self) -> np.ndarray | None:
        """The per-field affinity weights as a privately owned read-only array, or ``None``."""
        return self._field_weights

    @property
    def identity(self) -> RetrievalIdentity:
        """What this retriever's ``(index, score)`` pairs are relative to (MXR-080-1906).

        A bare corpus index is meaningless without the corpus it indexes and the model that scored
        it, and neither used to be recorded anywhere. This is stable for the retriever's lifetime --
        model, corpus and weights are all fixed at construction -- so it identifies every result the
        instance returns, and two retrievers can be compared for whether they were even ranking the
        same thing.
        """
        return self._identity

    def _log_affinity(self, rows: list) -> np.ndarray:
        factors = balanced_factors(self._model, rows, field_weights=self._field_weights)
        # Pre-built per-field factors use the affinity slot; posterior_mat is unused in that path.
        return model_log_affinity(None, None, affinity=factors, evidence_cap=self._evidence_cap)

    def affinity_matrix(self) -> np.ndarray:
        """The corpus's dense ``(n, n)`` log-affinity matrix (diagonal ``-inf``)."""
        return self._log_affinity(list(self._corpus))

    def retrieve(self, query: Any, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` corpus records for one query: ``[(corpus_index, log_affinity), ...]`` best first."""
        return self.retrieve_batch([query], k=k)[0]

    def retrieve_batch(self, queries: Any, k: int = 5) -> list[list[tuple[int, float]]]:
        """Top-``k`` per query, computed in one joint pass over ``corpus + queries``.

        Indices are positions in :attr:`corpus`; :attr:`identity` records which corpus that is.

        Raises:
            ValueError: If ``k`` is not an exact non-negative integer. A negative ``k`` is never
                silently accepted here: Python's ``[:k]`` slicing treats a negative ``k`` as "all
                but the last ``|k|`` items", not empty/error, which is never what a caller passing
                a negative retrieval count intended. A Boolean is refused for the same reason
                ``num_codes=True`` is: ``float(True) == 1.0``, so ``k=True`` used to be accepted and
                silently return exactly one neighbour (MXR-080-1906).
        """
        count = exact_count(k, "k")
        qs = list(queries)
        n = len(self._corpus)
        log_aff = self._log_affinity([*self._corpus, *qs])
        out: list[list[tuple[int, float]]] = []
        for j in range(len(qs)):
            row = log_aff[n + j, :n]  # query row against corpus columns only
            top = np.argsort(-row)[:count]
            out.append([(int(i), float(row[i])) for i in top])
        return out


def _own_field_weights(field_weights: Any) -> np.ndarray | None:
    """Return a private, finite, read-only copy of ``field_weights``, or ``None``.

    Deliberately NOT checked: the LENGTH against the model's field count. ``balanced_factors``
    resolves how weights map onto fields, heterogeneous models reach it with several different field
    groupings, and refusing a length here would reject shapes the library legitimately produces.
    What is enforced is only what is unambiguously broken for a weight vector -- non-numeric, or
    non-finite, which would silently poison every affinity into NaN.
    """
    if field_weights is None:
        return None
    try:
        weights = np.array(field_weights, dtype=np.float64)  # np.array copies; never alias the caller's
    except (TypeError, ValueError) as exc:
        raise ValueError("field_weights must be a numeric array of per-field weights") from exc
    if not np.isfinite(weights).all():
        raise ValueError("field_weights must be finite; a non-finite weight makes every affinity NaN")
    weights.setflags(write=False)
    return weights
