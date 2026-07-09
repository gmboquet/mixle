"""Structure learning over embedded heterogeneous fields.

:func:`mixle.inference.learn_bayesian_network` can model flat records containing
categorical, count, real, and fixed-length vector fields. This module adapts
free-text fields into that workflow by embedding each text value with
:func:`mixle.represent.fit_embedder` and representing the field as a
multivariate vector node.

The learned graph relates the embedding vector to other fields through
multivariate conditional-linear-Gaussian factors or by using vector components
as parent design features. Representative examples per embedding cluster remain
available for interpretation, but the model itself uses the full vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.structure import _is_discrete


def _lloyd(vectors: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    """Seeded k-means centroids on unit vectors (for readable per-cluster representatives only)."""
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
            else:
                centroids[j] = vectors[int(np.argmin(np.max(vectors @ centroids.T, axis=1)))]
    return centroids


@dataclass
class EmbeddedFieldCodec:
    """One text field's bridge into the graph: embed to a vector; centroids are for interpretability."""

    field: int
    embedder: Any
    centroids: np.ndarray
    representatives: list[str]  # one training example per cluster, nearest its centroid (for describe())

    def vector(self, value: Any) -> np.ndarray:
        """Embed one field value as a numeric vector."""
        return np.asarray(self.embedder.transform(str(value)), dtype=np.float64)

    def vectors(self, values: Sequence[Any]) -> np.ndarray:
        """Embed a sequence of field values as a matrix."""
        return np.asarray(self.embedder.transform([str(v) for v in values]), dtype=np.float64)


class EmbeddedStructureModel:
    """A discovered dependency graph over records whose text fields ride in as embedding VECTORS."""

    def __init__(self, net: Any, codecs: dict[int, EmbeddedFieldCodec]) -> None:
        self.net = net
        self.codecs = codecs

    def __str__(self) -> str:
        e = ", ".join(f"{p}->{c}" for p, c in self.net.edges())
        return f"EmbeddedStructureModel(text_fields={sorted(self.codecs)}, edges=[{e or 'none'}])"

    def encode_record(self, x: tuple) -> tuple:
        """The record with each text field replaced by its embedding vector."""
        vals = list(x)
        for i, codec in self.codecs.items():
            vals[i] = codec.vector(vals[i])
        return tuple(vals)

    def encode_records(self, rows: Sequence[tuple]) -> list[tuple]:
        """Replace configured text fields with embedding vectors for each record."""
        rows = [list(r) for r in rows]
        for i, codec in self.codecs.items():
            vecs = codec.vectors([r[i] for r in rows])
            for r, v in zip(rows, vecs):
                r[i] = v
        return [tuple(r) for r in rows]

    def edges(self) -> list[tuple[int, int]]:
        """Return discovered graph edges."""
        return self.net.edges()

    def log_density(self, x: tuple) -> float:
        """Evaluate log density after embedding text fields in one record."""
        return float(self.net.log_density(self.encode_record(x)))

    def seq_log_density(self, rows: Sequence[tuple]) -> np.ndarray:
        """Evaluate log density for records after embedding text fields."""
        embedded = self.encode_records(list(rows))
        return np.asarray(self.net.seq_log_density(self.net.dist_to_encoder().seq_encode(embedded)))

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
            many distinct values to be a categorical.
        n_clusters: number of representative clusters surfaced by ``describe()`` (interpretability only;
            the model uses the full embedding vector, not a cluster code).
        embed_dim: embedding dimension for :func:`mixle.represent.fit_embedder` -- the vector node's dim.
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
    embedded = model.encode_records(rows)
    model.net = learn_bayesian_network(embedded, max_parents=max_parents)
    return model
