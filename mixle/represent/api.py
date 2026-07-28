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


def _own_corpus_vectors(corpus_vectors: Any, result: AutoencoderResult) -> np.ndarray:
    """Return a private, immutable, finite, unit-normalized ``(N, dim)`` copy bound to ``result``.

    The array is copied (never aliased to the caller's), checked for rank/cardinality/finiteness,
    checked to be the width the fitted encoder actually produces, re-normalized with the same
    zero-safe clamp the fitter uses (so a legitimately all-zero row survives instead of being
    rejected), and finally frozen with ``writeable = False``.
    """
    vectors = np.array(corpus_vectors, dtype=np.float32)  # np.array copies; never alias the caller's buffer
    if vectors.ndim != 2:
        raise ValueError(f"corpus_vectors must be a 2-D (n_corpus, dim) array, got shape {vectors.shape}")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(f"corpus_vectors must be non-empty in both dimensions, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("corpus_vectors must contain only finite values")
    encoder_dim = getattr(getattr(result, "encoder", None), "dim", None)
    if encoder_dim is not None and int(encoder_dim) != vectors.shape[1]:
        raise ValueError(
            f"corpus_vectors width {vectors.shape[1]} does not match the fitted encoder's dim {int(encoder_dim)}"
        )
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    vectors.flags.writeable = False
    return vectors


class Embedder:
    """A fitted embedding of raw heterogeneous items: ``transform`` to vectors, ``retrieve`` neighbours."""

    def __init__(self, featurizer: Any, result: AutoencoderResult, kind: str, corpus_vectors: np.ndarray) -> None:
        if kind not in ("text", "record"):
            raise ValueError(f"kind must be 'text' or 'record', got {kind!r}")
        self.featurizer = featurizer
        self.result = result
        self.kind = kind
        self._corpus_vectors = _own_corpus_vectors(corpus_vectors, result)

    @property
    def corpus_vectors(self) -> np.ndarray:
        """The fitted corpus's ``(N, dim)`` unit embeddings, as a privately owned read-only array.

        Retrieval ranks against these vectors, so they are the embedder's evidence, not a scratch
        buffer: they are validated, normalized and frozen at construction, and exposed only through
        this property. Handing back the caller's own writable array (or letting the attribute be
        rebound) meant a post-construction ``corpus_vectors[0, 0] = nan`` turned ``retrieve()`` into
        an ordinary-looking ranked result whose similarities were all NaN (MXR-080-1785).
        """
        return self._corpus_vectors

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        return int(self._corpus_vectors.shape[1])

    def _units(self, items: list) -> np.ndarray:
        coerced = [str(x) for x in items] if self.kind == "text" else list(items)
        return np.asarray(self.featurizer.transform(coerced), dtype=np.float32)

    def _embed(self, rows: list) -> np.ndarray:
        """Embed an explicit list of whole records into unit-normalized ``(n, dim)`` rows.

        The encoded block is checked for shape and finiteness before it leaves: an embedding the
        fitted encoder could not actually produce (wrong width, NaN/inf) is a broken result, not a
        usable coordinate, and must never reach a similarity ranking as an ordinary vector.
        """
        vec = np.asarray(self.result.encode(self._units(rows)), dtype=np.float32)
        if vec.ndim != 2 or vec.shape[0] != len(rows) or vec.shape[1] != self.dim:
            raise ValueError(f"encoder returned shape {vec.shape}, expected ({len(rows)}, {self.dim})")
        if not np.isfinite(vec).all():
            raise ValueError("encoder produced a non-finite embedding; the fitted state is unusable")
        return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)

    def transform_one(self, item: Any) -> np.ndarray:
        """Embed exactly ONE item; always returns a single ``(dim,)`` vector.

        The unambiguous single-record contract. ``item`` is one whole record no matter which
        container it happens to be, so a list-valued record (``[1, 2]``) embeds to one vector here
        exactly as the equivalent tuple or dict does -- see :meth:`transform` for why the container
        alone cannot decide that.
        """
        return self._embed([item])[0]

    def transform_batch(self, items: Any) -> np.ndarray:
        """Embed a SEQUENCE of items; always returns an ``(n, dim)`` block.

        The unambiguous batch contract: every element of ``items`` is one whole record, never one
        record's field.
        """
        return self._embed(list(items))

    def transform(self, items: Any) -> np.ndarray:
        """Embed one item or a batch of them, unit-normalized (so dot = cosine similarity).

        A convenience wrapper over :meth:`transform_one`/:meth:`transform_batch` whose one-vs-batch
        decision is a guess from the outer container. That guess is only sound while a record
        cannot itself be list-shaped, and for ``kind="record"`` it can: ``_kind_of`` accepts a list
        as one record at fit time, so ``[1, 2]`` is a legitimate single record while ``[{...},
        {...}]`` is just as legitimately a batch of two. Rather than silently returning two
        embedding rows for a single declared record -- training and serving disagreeing about the
        same record type -- the genuinely ambiguous shape raises and names the two explicit methods
        that carry no ambiguity at all.

        Raises:
            ValueError: If ``items`` is a ``kind="record"`` sequence whose own elements are not
                themselves record containers, so it is indistinguishable from one list-shaped
                record. Call :meth:`transform_one` or :meth:`transform_batch` instead.
        """
        if not isinstance(items, (list, np.ndarray)):
            return self.transform_one(items)
        rows = list(items)
        if self.kind == "record" and rows and not all(isinstance(r, (dict, tuple, list, np.ndarray)) for r in rows):
            raise ValueError(
                "transform() cannot tell whether this list is one record or a batch of records: its "
                "elements are not themselves records. Call transform_one(item) for a single "
                "list-shaped record or transform_batch(items) for a sequence of records."
            )
        return self.transform_batch(rows)

    def retrieve(self, query: Any, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` fitted-corpus neighbours of ``query`` as ``(corpus index, cosine similarity)``.

        Raises:
            ValueError: If ``k`` is negative or not an integer value. A negative ``k`` is never
                silently accepted here: Python's ``[:k]`` slicing treats a negative ``k`` as "all
                but the last ``|k|`` items", not empty/error, which is never what a caller passing
                a negative retrieval count intended.
        """
        kf = float(k)
        if kf < 0 or kf != round(kf):
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        q = self.transform_one(query)  # a query is exactly one item; never guess batch-ness here
        sims = self._corpus_vectors @ q
        if not np.isfinite(sims).all():
            raise ValueError("retrieval similarities are not finite; refusing to rank invalid evidence")
        order = np.argsort(-sims)[: int(kf)]
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
    def load(cls, path: str, *, trust_code: bool = False) -> Embedder:
        """Load an embedder previously saved with :meth:`save`.

        The saved artifact is a pickle of the fitted state, and ``result`` can embed a live torch
        module (see :class:`~mixle.represent.generative.AutoencoderResult`) -- unpickling it executes
        arbitrary code from the file, exactly like ``pickle.load`` on an untrusted source. Refuses by
        default; pass ``trust_code=True`` for a path whose source you trust (or call from inside
        :func:`mixle.utils.serialization.trusted_deserialization`), matching
        :meth:`mixle.inference.production.registry.Registry.get`.
        """
        from mixle.utils.serialization import SerializationError, deserialization_is_trusted

        if not (trust_code or deserialization_is_trusted()):
            raise SerializationError(
                "refusing to unpickle an Embedder artifact: this executes arbitrary code from the "
                "file, the same as pickle.load on an untrusted source. Only load a path whose source "
                "you trust, and pass trust_code=True (or call inside "
                "mixle.utils.serialization.trusted_deserialization())."
            )
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
