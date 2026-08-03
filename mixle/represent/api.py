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

import contextlib
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from mixle.data.hashing import _canonical
from mixle.represent.generative import AutoencoderResult, fit_autoencoder
from mixle.represent.identity import RetrievalIdentity, exact_count, vectors_digest
from mixle.utils.exact import require_explicit_true

_ARTIFACT_ID = "represent.Embedder/v2"
_ARTIFACT_NAME = "embedder.mixle"
_ARTIFACT_MAGIC = b"MIXLEEMB2\n"


def _envelope_digest(manifest: dict[str, Any], body: bytes) -> str:
    """SHA-256 over the whole artifact: every manifest field (except ``digest``, which cannot cover
    its own value) plus the serialized body.

    The manifest is hashed through :func:`mixle.data.hashing._canonical` rather than its raw JSON
    bytes -- the same helper :mod:`mixle.data.encoded_io` uses for the identical job -- so the
    digest is independent of on-disk key order and whitespace, and ``_canonical``'s length-prefixed,
    separator-free encoding makes the manifest/body concatenation unambiguous: no other split can
    land on the same bytes. Covering the manifest is the point: hashing the body alone would let the
    recorded kind, dim, or corpus count be edited on disk without invalidating anything.
    """
    return hashlib.sha256(_canonical(manifest) + body).hexdigest()


_KINDS = ("text", "record")


def _featurizer(kind: str, dim: int, seed: int) -> Any:
    """Return the featurizer for ``kind``, which must name one of the two supported input schemas.

    The kind is checked against the closed :data:`_KINDS` enum here rather than being treated as
    "text, or else record": routing every unrecognized value to :class:`HashedRecord` meant a typo
    such as ``kind="txet"`` selected the record featurizer and drove a whole autoencoder fit before
    anything noticed (MXR-080-1784).
    """
    from mixle.task.model import HashedNGram, HashedRecord

    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {list(_KINDS)}, got {kind!r}")
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
        if kind not in _KINDS:
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

    @property
    def identity(self) -> RetrievalIdentity:
        """What this embedder's ``(index, similarity)`` pairs are relative to (MXR-080-1906).

        A bare corpus index names nothing without the corpus it indexes, and a cosine similarity is
        only interpretable against the model that produced both sides of it. Neither was recorded:
        the save manifest carried ``kind``/``dim``/``n_corpus`` and no fitted-state identity at all,
        so two embedders over different corpora returned indistinguishable results.

        The digest covers the fitted corpus vectors, which are frozen at construction, so this is
        stable for the instance's lifetime and identifies every result it returns. It is computed on
        each access rather than cached because ``result`` and ``featurizer`` remain rebindable
        attributes -- see the note in :meth:`retrieve` about what that does and does not cover.
        """
        return RetrievalIdentity(
            model=f"{type(self.result).__module__}.{type(self.result).__name__}",
            corpus_size=int(self._corpus_vectors.shape[0]),
            corpus_digest=vectors_digest(self._corpus_vectors),
        )

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

        Indices are positions in :attr:`corpus_vectors`; :attr:`identity` records which corpus that
        is. Note what identity does NOT cover: ``result`` and ``featurizer`` are still plain
        rebindable attributes, so replacing ``emb.result`` re-points the query encoder while the
        frozen corpus vectors still come from the old one. Fixing that is a constructor-ownership
        change to a class that is pickled by :meth:`save` and rebuilt field-by-field by
        :meth:`load`, which is a wider change than this finding, and it is left deliberately
        undone rather than half-done.

        Raises:
            ValueError: If ``k`` is not an exact non-negative integer -- see
                :func:`mixle.represent.identity.exact_count` for why a Boolean is refused and an
                integral float is not.
        """
        count = exact_count(k, "k")
        q = self.transform_one(query)  # a query is exactly one item; never guess batch-ness here
        sims = self._corpus_vectors @ q
        if not np.isfinite(sims).all():
            raise ValueError("retrieval similarities are not finite; refusing to rank invalid evidence")
        order = np.argsort(-sims)[:count]
        return [(int(i), float(sims[i])) for i in order]

    def save(self, path: str) -> str:
        """Persist the embedder and fitted corpus vectors as one atomic, digest-bound envelope.

        The artifact is a single file: magic, a canonical JSON manifest, then the pickled state.
        The manifest's digest covers the manifest fields AND the body together, and the file is
        written to a temporary sibling and ``os.replace``-d into position, so a reader can never see
        a half-written artifact or a manifest describing a different payload than the one beside it.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        body = pickle.dumps(
            {
                "featurizer": self.featurizer,
                "result": self.result,
                "kind": self.kind,
                "corpus_vectors": self._corpus_vectors,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        manifest: dict[str, Any] = {
            "mixle_artifact": _ARTIFACT_ID,
            "kind": self.kind,
            "dim": self.dim,
            "n_corpus": int(self._corpus_vectors.shape[0]),
            "created_at": time.time(),
        }
        meta = {"digest": _envelope_digest(manifest, body), **manifest}
        destination = out / _ARTIFACT_NAME
        fd, temporary = tempfile.mkstemp(prefix=f".{_ARTIFACT_NAME}.", suffix=".tmp", dir=str(out))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(_ARTIFACT_MAGIC)
                f.write(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                f.write(b"\n")
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        return str(out)

    @classmethod
    def load(cls, path: str, *, trust_code: bool = False) -> Embedder:
        """Load an embedder previously saved with :meth:`save`, verifying its manifest.

        The saved artifact is a pickle of the fitted state, and ``result`` can embed a live torch
        module (see :class:`~mixle.represent.generative.AutoencoderResult`) -- unpickling it executes
        arbitrary code from the file, exactly like ``pickle.load`` on an untrusted source. Refuses by
        default; pass ``trust_code=True`` for a path whose source you trust (or call from inside
        :func:`mixle.utils.serialization.trusted_deserialization`), matching
        :meth:`mixle.inference.production.registry.Registry.get`.

        Beyond that trust gate, the manifest is now actually enforced rather than ignored
        (MXR-080-1786). The envelope digest -- manifest fields and body together -- is checked BEFORE
        the body is unpickled, so a truncated, corrupted, swapped or manifest-tampered artifact is
        rejected before any deserialization runs; and the decoded payload's own type, kind, and
        corpus shape are checked against what the manifest declared, so state that does not match
        its description can never load under it. The digest is corruption/mismatch detection, not
        authentication: it lives in the same file it covers, so it cannot prove the artifact was not
        replaced wholesale by whoever could already write to ``path``.
        """
        from mixle.utils.serialization import SerializationError, deserialization_is_trusted

        # `or` on a raw flag is truthiness, and this gate authorizes code execution: trust_code="false"
        # -- the string, straight out of a config file or CLI argument -- opened it (MXR-080-1881).
        # The flag must be the True singleton, matching load_encoded's contract; the ambient
        # trusted_deserialization() scope remains the other way in, and is an explicit context manager
        # rather than a caller-supplied value.
        if trust_code is not False:
            require_explicit_true(
                trust_code,
                "Embedder.load trust_code",
                because="Unpickling this artifact executes arbitrary code from the file.",
            )
        if not (trust_code or deserialization_is_trusted()):
            raise SerializationError(
                "refusing to unpickle an Embedder artifact: this executes arbitrary code from the "
                "file, the same as pickle.load on an untrusted source. Only load a path whose source "
                "you trust, and pass trust_code=True (or call inside "
                "mixle.utils.serialization.trusted_deserialization())."
            )
        root = Path(path)
        envelope = root / _ARTIFACT_NAME
        if not envelope.exists() and (root / "embedder.pkl").exists():
            raise ValueError(
                f"{path!r} holds a legacy unbound Embedder artifact (an undigested pickle beside a "
                f"manifest nothing verified); re-save it with the current mixle to produce a "
                f"{_ARTIFACT_NAME} envelope."
            )
        with open(envelope, "rb") as f:
            if f.read(len(_ARTIFACT_MAGIC)) != _ARTIFACT_MAGIC:
                raise ValueError(f"{path!r} is not a mixle Embedder artifact")
            meta_line = bytearray()
            while True:
                c = f.read(1)
                if c in (b"\n", b""):
                    break
                meta_line += c
            try:
                meta = json.loads(bytes(meta_line).decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"{path!r} has a corrupt Embedder manifest") from exc
            body = f.read()
        manifest = {k: v for k, v in meta.items() if k != "digest"}
        if _envelope_digest(manifest, body) != meta.get("digest"):
            raise ValueError(f"{path!r} failed its integrity check (corrupt, truncated, or a tampered manifest)")
        if meta.get("mixle_artifact") != _ARTIFACT_ID:
            raise ValueError(f"{path!r} declares artifact {meta.get('mixle_artifact')!r}, expected {_ARTIFACT_ID!r}")
        payload = pickle.loads(body)  # noqa: S301 - trust-gated above and digest-verified; see the docstring  # nosec B301 # MXR-080-1881: Embedder.load requires trust_code to be the True singleton (or an already-open trusted_deserialization scope) before reaching this envelope
        expected = {"featurizer", "result", "kind", "corpus_vectors"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(f"{path!r} does not contain an Embedder state payload")
        vectors = payload["corpus_vectors"]
        declared = (meta.get("n_corpus"), meta.get("dim"))
        if not isinstance(vectors, np.ndarray) or vectors.shape != declared:
            raise ValueError(
                f"{path!r} carries corpus vectors of shape {getattr(vectors, 'shape', type(vectors).__name__)!r}, "
                f"but its manifest declares {declared!r}"
            )
        if payload["kind"] != meta.get("kind"):
            raise ValueError(
                f"{path!r} carries kind {payload['kind']!r}, but its manifest declares {meta.get('kind')!r}"
            )
        return cls(payload["featurizer"], payload["result"], payload["kind"], vectors)


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

    ``kind`` is one explicit input schema for the whole corpus, resolved and validated BEFORE any
    featurizing or fitting happens (MXR-080-1784). An explicit ``kind`` must name one of the two
    supported schemas; an unrecognized value used to select the record featurizer and only surface
    at the very end of an otherwise complete autoencoder fit. When ``kind`` is left to inference,
    EVERY item is inspected rather than only the first: a corpus mixing text and record items has no
    single schema to infer, and silently featurizing dict records through ``str()`` as text (or the
    reverse) is a wrong representation, not a usable one. Declare ``kind`` explicitly if that
    coercion is what you want.

    Raises:
        ValueError: If ``kind`` names no supported schema, or if it is left to inference over a
            corpus whose items are not all the same kind.
    """
    items = list(data)
    if len(items) < 4:
        raise ValueError("fit_embedder needs at least 4 items")
    if kind is None:
        observed = {_kind_of(x) for x in items}
        if len(observed) > 1:
            raise ValueError(
                f"cannot infer one input kind from a corpus mixing {sorted(observed)} items; pass "
                "kind='text' or kind='record' to declare the schema every item is featurized under."
            )
        k = observed.pop()
    elif kind not in _KINDS:
        raise ValueError(f"kind must be one of {list(_KINDS)} or None, got {kind!r}")
    else:
        k = kind
    feat = _featurizer(k, feature_dim, seed)
    units = np.asarray(feat.transform([str(x) for x in items] if k == "text" else items), dtype=np.float32)
    result = fit_autoencoder(units, dim, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
    vecs = result.encode(units)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return Embedder(feat, result, k, vecs)
