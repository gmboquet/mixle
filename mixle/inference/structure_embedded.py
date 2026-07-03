"""Structure learning over embedded heterogeneous fields -- text becomes a first-class graph node.

:func:`mixle.inference.learn_bayesian_network` handles flat records of categoricals, counts, and
reals -- but a free-text field (a description, a log line, a title) has no place in that record: it
is not discrete (unbounded distinct values) and ``float()`` of it is meaningless, so today such a
field cannot participate in structure discovery at all. This module couples the structure learner to
:mod:`mixle.represent`: each text field is embedded (:func:`mixle.represent.fit_embedder`), the
embeddings are clustered (seeded Lloyd k-means on the unit sphere), and the field enters the record
as its CLUSTER CODE -- a plain categorical the graph machinery already knows how to relate to every
other field in both directions (table factors, GLM nodes, CLG one-hots). The learned edges answer
"does what the text says relate to the price / the label / the count?", with per-cluster
representative examples for reading the discovered structure.

Honest scope: the returned model scores the PROXY record (text collapsed to its cluster code), so its
``log_density`` is a joint over (codes, other fields) -- a coarse but calibrated view; it is exactly
as fine as ``n_clusters``. Sampling returns proxy codes, not text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.structure import _is_discrete


def _lloyd(vectors: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    """Seeded k-means centroids on unit vectors (Euclidean Lloyd == cosine ordering on the sphere)."""
    rng = np.random.RandomState(seed)
    k = min(int(k), len(vectors))
    centroids = vectors[rng.choice(len(vectors), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(vectors @ centroids.T, axis=1)
        for j in range(k):
            members = vectors[assign == j]
            if len(members):
                c = members.mean(axis=0)
                centroids[j] = c / max(float(np.linalg.norm(c)), 1e-12)
            else:  # a starved centroid restarts on the point farthest from its cluster
                centroids[j] = vectors[int(np.argmin(np.max(vectors @ centroids.T, axis=1)))]
    return centroids


@dataclass
class EmbeddedFieldCodec:
    """One text field's bridge into the graph: embed, then nearest-centroid cluster code."""

    field: int
    embedder: Any
    centroids: np.ndarray
    representatives: list[str]  # one training example per cluster, nearest its centroid

    def code(self, value: Any) -> str:
        vec = self.embedder.transform(str(value))
        return f"c{int(np.argmax(self.centroids @ vec))}"

    def codes(self, values: Sequence[Any]) -> list[str]:
        vecs = self.embedder.transform([str(v) for v in values])
        return [f"c{int(j)}" for j in np.argmax(vecs @ self.centroids.T, axis=1)]


class EmbeddedStructureModel:
    """A discovered dependency graph over records whose text fields ride in as cluster codes."""

    def __init__(self, net: Any, codecs: dict[int, EmbeddedFieldCodec]) -> None:
        self.net = net
        self.codecs = codecs

    def __str__(self) -> str:
        e = ", ".join(f"{p}->{c}" for p, c in self.net.edges())
        return f"EmbeddedStructureModel(text_fields={sorted(self.codecs)}, edges=[{e or 'none'}])"

    def encode_record(self, x: tuple) -> tuple:
        """The proxy record: text fields replaced by their cluster codes."""
        vals = list(x)
        for i, codec in self.codecs.items():
            vals[i] = codec.code(vals[i])
        return tuple(vals)

    def encode_records(self, rows: Sequence[tuple]) -> list[tuple]:
        rows = [list(r) for r in rows]
        for i, codec in self.codecs.items():
            for r, code in zip(rows, codec.codes([r[i] for r in rows])):
                r[i] = code
        return [tuple(r) for r in rows]

    def edges(self) -> list[tuple[int, int]]:
        return self.net.edges()

    def log_density(self, x: tuple) -> float:
        return float(self.net.log_density(self.encode_record(x)))

    def seq_log_density(self, rows: Sequence[tuple]) -> np.ndarray:
        proxies = self.encode_records(list(rows))
        return np.asarray(self.net.seq_log_density(self.net.dist_to_encoder().seq_encode(proxies)))

    def describe(self) -> dict[str, Any]:
        """The discovered structure with per-cluster representative examples for each text field."""
        return {
            "edges": self.net.edges(),
            "text_fields": {
                i: {f"c{j}": rep for j, rep in enumerate(codec.representatives)} for i, codec in self.codecs.items()
            },
        }


def learn_structure_embedded(
    data: Sequence[tuple],
    *,
    text_fields: Sequence[int] | str = "auto",
    n_clusters: int = 8,
    embed_dim: int = 16,
    seed: int = 0,
    max_parents: int = 2,
    **embed_kw: Any,
) -> EmbeddedStructureModel:
    """Discover cross-field structure where some fields are free text (see module docstring).

    Args:
        data: flat tuple records; text fields may hold arbitrary strings.
        text_fields: which field indices to embed, or ``"auto"`` -- every string-valued field with too
            many distinct values to be a categorical (the exact fields the plain structure learner
            cannot accept today).
        n_clusters: cluster-code vocabulary per text field (the proxy's resolution).
        embed_dim: embedding dimension for :func:`mixle.represent.fit_embedder`.
        **embed_kw: forwarded to ``fit_embedder`` (``epochs``, ``hidden``, ``feature_dim``, ...).
    """
    from mixle.inference.bayesian_network import learn_bayesian_network
    from mixle.represent import fit_embedder

    rows = [tuple(r) for r in data]
    if len(rows) < 40:
        raise ValueError("learn_structure_embedded needs at least 40 records")
    n_fields = len(rows[0])

    if text_fields == "auto":
        text_fields = [
            i
            for i in range(n_fields)
            if all(isinstance(r[i], str) for r in rows) and not _is_discrete([r[i] for r in rows])
        ]
    fields = sorted(int(i) for i in text_fields)
    if not fields:
        raise ValueError(
            "no text fields to embed: pass text_fields= explicitly, or use learn_bayesian_network "
            "directly for records without free-text fields"
        )

    codecs: dict[int, EmbeddedFieldCodec] = {}
    for i in fields:
        values = [str(r[i]) for r in rows]
        emb = fit_embedder(values, dim=int(embed_dim), kind="text", seed=seed, **embed_kw)
        vecs = emb.corpus_vectors
        centroids = _lloyd(vecs, int(n_clusters), seed)
        sims = vecs @ centroids.T
        reps = [values[int(np.argmax(sims[:, j]))] for j in range(len(centroids))]
        codecs[i] = EmbeddedFieldCodec(field=i, embedder=emb, centroids=centroids, representatives=reps)

    model = EmbeddedStructureModel(net=None, codecs=codecs)  # encode_records needs only the codecs
    proxies = model.encode_records(rows)
    model.net = learn_bayesian_network(proxies, max_parents=max_parents)
    return model
