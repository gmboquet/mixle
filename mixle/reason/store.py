"""Cross-modal RAG with a raw-data fallback: retrieve by embedding, condition on raw evidence.

Embedding-only retrieval (embed everything, fetch top-k, stuff the vectors in) loses information
twice for modalities too lossy to compress -- a full spectrum, a thin-section image, a seismic
sub-volume. This store treats retrieval as **evidence selection for Bayesian assimilation** instead:

1. index the corpus by a cheap embedding key (an approximate router, not the answer);
2. for a query, retrieve the nearest items by embedding;
3. for each, run a **sufficiency test** -- would the raw payload reduce the *query's* uncertainty
   materially more than its lossy embedding? If not, use the cheap embedding evidence; if so,
   **fetch the raw payload** and condition the belief on it through its full (precise) evidence;
4. fuse each choice into the belief (a product-of-experts update), recording provenance;
5. optionally **retrieve actively** -- fetch the corpus item that most reduces the query entropy.

Domain-neutral: the store knows nothing about seismic or spectra. The application supplies two
callables -- ``coarse(payload) -> Evidence`` (embedding fidelity) and ``fine(payload) -> Evidence``
(raw fidelity) -- so the same machinery serves a document corpus or one spatially-indexed volume
(see the ``mixle_pde`` spatial-store application).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.belief import GaussianBelief
from mixle.reason.core import LinearGaussianEvidence


@dataclass(frozen=True)
class RetrievalStep:
    """Provenance for one assimilated item: which corpus index, at what fidelity, and the nats it removed."""

    index: int
    fidelity: str  # "raw" or "embedding"
    gain: float


def _query_entropy(belief: GaussianBelief, query: Any) -> float:
    return belief.entropy() if query is None else belief.marginal(query).entropy()


def _apply(belief: GaussianBelief, ev: LinearGaussianEvidence) -> GaussianBelief:
    return belief.update(ev.H, ev.y, ev.R)


class CrossModalStore:
    """A corpus indexed by embedding keys, with raw payloads conditioned on when embeddings fall short.

    Args:
        keys: ``(N, d_key)`` embedding vectors -- the retrieval index (routers, not answers).
        payloads: length-``N`` sequence of raw items (arbitrary; passed to ``coarse``/``fine``).
        coarse: ``payload -> LinearGaussianEvidence`` at *embedding* fidelity (cheap, lossy).
        fine: ``payload -> LinearGaussianEvidence`` at *raw* fidelity (precise, "expensive").
        metric: ``"euclidean"`` (default) or ``"cosine"`` for retrieval.
    """

    def __init__(
        self,
        keys: Any,
        payloads: Sequence[Any],
        *,
        coarse: Callable[[Any], LinearGaussianEvidence],
        fine: Callable[[Any], LinearGaussianEvidence],
        metric: str = "euclidean",
    ) -> None:
        self.keys = np.atleast_2d(np.asarray(keys, dtype=float))
        self.payloads = list(payloads)
        if self.keys.shape[0] != len(self.payloads):
            raise ValueError(f"{self.keys.shape[0]} keys but {len(self.payloads)} payloads")
        if metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.coarse = coarse
        self.fine = fine
        self.metric = metric

    def __len__(self) -> int:
        return len(self.payloads)

    def retrieve(self, query_key: Any, k: int = 8) -> list[int]:
        """Indices of the ``k`` corpus items whose embedding keys are nearest ``query_key``."""
        q = np.asarray(query_key, dtype=float).reshape(-1)
        if self.metric == "cosine":
            kn = self.keys / (np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-12)
            qn = q / (np.linalg.norm(q) + 1e-12)
            dist = 1.0 - kn @ qn
        else:
            dist = np.linalg.norm(self.keys - q, axis=1)
        return list(np.argsort(dist)[: int(k)])

    def assimilate(
        self,
        belief: GaussianBelief,
        query_key: Any,
        *,
        k: int = 8,
        query: Any = None,
        epsilon: float = 0.0,
    ) -> tuple[GaussianBelief, list[RetrievalStep]]:
        """Retrieve ``k`` neighbors and fold each into ``belief``, fetching raw payloads when the
        embedding is too lossy for the ``query``.

        For each retrieved item the sufficiency test compares how much the *raw* evidence would
        reduce the query entropy versus the *embedding* evidence; if the surplus exceeds
        ``epsilon`` the raw payload is used, else the cheap embedding is. Returns the updated belief
        and a per-item provenance trail.
        """
        steps: list[RetrievalStep] = []
        for idx in self.retrieve(query_key, k):
            payload = self.payloads[idx]
            before = _query_entropy(belief, query)
            emb_ev = self.coarse(payload)
            raw_ev = self.fine(payload)
            gain_emb = before - _query_entropy(_apply(belief, emb_ev), query)
            gain_raw = before - _query_entropy(_apply(belief, raw_ev), query)
            use_raw = (gain_raw - gain_emb) > epsilon
            chosen = raw_ev if use_raw else emb_ev
            belief = _apply(belief, chosen)
            steps.append(
                RetrievalStep(
                    index=idx, fidelity="raw" if use_raw else "embedding", gain=gain_raw if use_raw else gain_emb
                )
            )
        return belief, steps

    def next_evidence(
        self,
        belief: GaussianBelief,
        *,
        query: Any = None,
        candidates: Sequence[int] | None = None,
        fidelity: str = "fine",
    ) -> tuple[int, float]:
        """Active retrieval: the corpus item whose evidence most reduces the query entropy (EIG).

        Returns ``(index, expected_gain_nats)``. ``fidelity`` selects the ``fine`` (raw) or
        ``coarse`` (embedding) evidence builder for the look-ahead.
        """
        build = self.fine if fidelity == "fine" else self.coarse
        pool = list(range(len(self.payloads))) if candidates is None else list(candidates)
        before = _query_entropy(belief, query)
        best_idx, best_gain = -1, -np.inf
        for idx in pool:
            gain = before - _query_entropy(_apply(belief, build(self.payloads[idx])), query)
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        return best_idx, float(best_gain)
