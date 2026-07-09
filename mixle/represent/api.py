"""``fit_embedder`` -- point it at raw heterogeneous data, get back vectors and retrieval.

The one-call product surface over the representation layer: text or records (dicts / tuples of mixed
fields) featurize through the task layer's deterministic hashers, and a generatively-trained autoencoder
(:func:`mixle.represent.generative.fit_autoencoder`) compresses them into a learned ``dim``-space. The
returned :class:`Embedder` transforms new items into that space and retrieves nearest neighbours over the
fitted corpus -- model-based retrieval over *raw* data, no upstream tokenizer or external embedding API::

    emb = fit_embedder(tickets, dim=16)          # dict/tuple records or strings
    emb.transform(new_items)                     # (N, dim)
    emb.retrieve(query, k=5)                     # [(corpus index, similarity), ...]
    emb.save(path); Embedder.load(path)          # durable artifact

Deterministic given ``seed``. Needs torch only to FIT; a saved Embedder reloads and transforms anywhere.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from mixle.represent.generative import AutoencoderResult, fit_autoencoder


def _featurizer(kind: str, dim: int, seed: int) -> Any:
    from mixle.task.model import HashedNGram, HashedRecord

    return HashedNGram(n=3, dim=dim, seed=seed) if kind == "text" else HashedRecord(dim=dim, seed=seed)


def _kind_of(x: Any) -> str:
    if isinstance(x, str):
        return "text"
    if isinstance(x, (dict, tuple, list)):
        return "record"
    raise TypeError(
        "fit_embedder handles text or record (dict/tuple) items; got %r. Pass kind='text'|'record'." % type(x).__name__
    )


class Embedder:
    """A fitted embedding of raw heterogeneous items: ``transform`` to vectors, ``retrieve`` neighbours."""

    def __init__(self, featurizer: Any, result: AutoencoderResult, kind: str, corpus_vectors: np.ndarray) -> None:
        self.featurizer = featurizer
        self.result = result
        self.kind = kind
        self.corpus_vectors = corpus_vectors  # (N, dim) unit-normalized embeddings of the fitted data

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        return int(self.corpus_vectors.shape[1])

    def _units(self, items: list) -> np.ndarray:
        coerced = [str(x) for x in items] if self.kind == "text" else list(items)
        return np.asarray(self.featurizer.transform(coerced), dtype=np.float32)

    def transform(self, items: Any) -> np.ndarray:
        """Embed items into the learned space, unit-normalized (so dot = cosine similarity)."""
        one = not isinstance(items, (list, np.ndarray))
        vec = self.result.encode(self._units([items] if one else list(items)))
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
        return vec[0] if one else vec

    def retrieve(self, query: Any, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` fitted-corpus neighbours of ``query`` as ``(corpus index, cosine similarity)``."""
        q = self.transform(query)
        sims = self.corpus_vectors @ q
        order = np.argsort(-sims)[: int(k)]
        return [(int(i), float(sims[i])) for i in order]

    def save(self, path: str) -> str:
        """Persist the embedder and fitted corpus vectors."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "embedder.pkl", "wb") as f:
            pickle.dump(
                {
                    "featurizer": self.featurizer,
                    "result": self.result,
                    "kind": self.kind,
                    "corpus_vectors": self.corpus_vectors,
                },
                f,
            )
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "mixle_artifact": "represent.Embedder/v1",
                    "kind": self.kind,
                    "dim": self.dim,
                    "n_corpus": int(self.corpus_vectors.shape[0]),
                    "created_at": time.time(),
                },
                indent=2,
            )
        )
        return str(out)

    @classmethod
    def load(cls, path: str) -> Embedder:
        """Load an embedder previously saved with :meth:`save`."""
        with open(Path(path) / "embedder.pkl", "rb") as f:
            d = pickle.load(f)
        return cls(d["featurizer"], d["result"], d["kind"], d["corpus_vectors"])


def fit_embedder(
    data: Any,
    dim: int = 32,
    *,
    kind: str | None = None,
    feature_dim: int = 256,
    hidden: tuple[int, ...] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
) -> Embedder:
    """Fit a learned embedding of raw text or record items and return an :class:`Embedder`.

    Items featurize deterministically (hashing trick; no fitted vocabulary), then an autoencoder learns a
    ``dim``-dimensional generative representation of the corpus. ``retrieve`` works out of the box over
    the fitted data; ``transform`` embeds anything of the same kind.
    """
    items = list(data)
    if len(items) < 4:
        raise ValueError("fit_embedder needs at least 4 items")
    k = kind or _kind_of(items[0])
    feat = _featurizer(k, feature_dim, seed)
    units = np.asarray(feat.transform([str(x) for x in items] if k == "text" else items), dtype=np.float32)
    result = fit_autoencoder(units, dim, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
    vecs = result.encode(units)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return Embedder(feat, result, k, vecs)
