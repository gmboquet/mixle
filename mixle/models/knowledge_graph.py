"""Dependency-free knowledge-graph embedding helpers based on TransE.

The module provides a small NumPy implementation for scoring triples, generating
negative samples, and fitting entity and relation embeddings with a margin
ranking objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.models._result import FitResult

Triple = tuple[Any, Any, Any]


@dataclass
class KnowledgeGraphFitResult(FitResult["TransEKnowledgeGraphModel"]):
    """Result from TransE margin fitting."""


class TransEKnowledgeGraphModel:
    """Dependency-free TransE model with a NumPy margin objective."""

    def __init__(
        self,
        entity_embeddings: Any,
        relation_embeddings: Any,
        entity_names: Sequence[Any] | None = None,
        relation_names: Sequence[Any] | None = None,
        name: str | None = None,
    ) -> None:
        self.entity_embeddings = np.asarray(entity_embeddings, dtype=np.float64)
        self.relation_embeddings = np.asarray(relation_embeddings, dtype=np.float64)
        if self.entity_embeddings.ndim != 2 or self.relation_embeddings.ndim != 2:
            raise ValueError("embeddings must be two-dimensional arrays.")
        if self.entity_embeddings.shape[1] != self.relation_embeddings.shape[1]:
            raise ValueError("entity and relation embeddings must share a dimension.")
        self.num_entities = int(self.entity_embeddings.shape[0])
        self.num_relations = int(self.relation_embeddings.shape[0])
        self.embedding_dim = int(self.entity_embeddings.shape[1])
        self.entity_names = list(range(self.num_entities)) if entity_names is None else list(entity_names)
        self.relation_names = list(range(self.num_relations)) if relation_names is None else list(relation_names)
        if len(self.entity_names) != self.num_entities:
            raise ValueError("entity_names length must match entity_embeddings.")
        if len(self.relation_names) != self.num_relations:
            raise ValueError("relation_names length must match relation_embeddings.")
        self.entity_index: dict[Any, int] = {v: i for i, v in enumerate(self.entity_names)}
        self.relation_index: dict[Any, int] = {v: i for i, v in enumerate(self.relation_names)}
        self.name = name

    @classmethod
    def random(
        cls,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 16,
        seed: int | None = None,
        scale: float = 0.01,
        entity_names: Sequence[Any] | None = None,
        relation_names: Sequence[Any] | None = None,
        name: str | None = None,
    ) -> TransEKnowledgeGraphModel:
        """Create a randomly initialized model."""
        if num_entities <= 0 or num_relations <= 0 or embedding_dim <= 0:
            raise ValueError("num_entities, num_relations, and embedding_dim must be positive.")
        rng = np.random.RandomState(seed)
        ent = rng.normal(scale=scale, size=(num_entities, embedding_dim))
        rel = rng.normal(scale=scale, size=(num_relations, embedding_dim))
        return cls(ent, rel, entity_names=entity_names, relation_names=relation_names, name=name)

    def __str__(self) -> str:
        return "TransEKnowledgeGraphModel(num_entities=%d, num_relations=%d, dim=%d, name=%r)" % (
            self.num_entities,
            self.num_relations,
            self.embedding_dim,
            self.name,
        )

    def distance_triples(self, triples: Sequence[Triple]) -> np.ndarray:
        """Return squared TransE distances ||h + r - t||^2."""
        idx = self._triple_indices(triples)
        h = self.entity_embeddings[idx[:, 0]]
        r = self.relation_embeddings[idx[:, 1]]
        t = self.entity_embeddings[idx[:, 2]]
        diff = h + r - t
        return np.sum(diff * diff, axis=1)

    def score_triples(self, triples: Sequence[Triple]) -> np.ndarray:
        """Return TransE scores; higher is more plausible."""
        return -self.distance_triples(triples)

    def margin_loss(
        self, positive_triples: Sequence[Triple], negative_triples: Sequence[Triple], margin: float = 1.0
    ) -> float:
        """Return the pairwise TransE ranking loss."""
        if len(positive_triples) != len(negative_triples):
            raise ValueError("positive and negative triples must have the same length.")
        pos = self.distance_triples(positive_triples)
        neg = self.distance_triples(negative_triples)
        return float(np.maximum(0.0, float(margin) + pos - neg).sum())

    def negative_sample(
        self, triples: Sequence[Triple], seed: int | None = None, corrupt: str = "tail"
    ) -> list[Triple]:
        """Corrupt heads or tails to produce negative triples."""
        if corrupt not in ("head", "tail", "both"):
            raise ValueError("corrupt must be 'head', 'tail', or 'both'.")
        rng = np.random.RandomState(seed)
        rv: list[Triple] = []
        for h, r, t in triples:
            mode = corrupt
            if corrupt == "both":
                mode = "head" if rng.rand() < 0.5 else "tail"
            if mode == "head":
                new_h = self.entity_names[int(rng.randint(0, self.num_entities))]
                rv.append((new_h, r, t))
            else:
                new_t = self.entity_names[int(rng.randint(0, self.num_entities))]
                rv.append((h, r, new_t))
        return rv

    def fit_margin(
        self,
        positive_triples: Sequence[Triple],
        negative_triples: Sequence[Triple] | None = None,
        margin: float = 1.0,
        lr: float = 0.01,
        max_its: int = 100,
        seed: int | None = None,
        normalize_entities: bool = True,
    ) -> KnowledgeGraphFitResult:
        """Fit embeddings with simple stochastic subgradient descent."""
        if len(positive_triples) == 0:
            raise ValueError("positive_triples must not be empty.")
        if lr <= 0.0:
            raise ValueError("lr must be positive.")
        rng = np.random.RandomState(seed)
        history: list[float] = []
        positives = list(positive_triples)
        for _ in range(max(1, int(max_its))):
            negatives = (
                list(negative_triples)
                if negative_triples is not None
                else self.negative_sample(positives, seed=int(rng.randint(0, 2**31 - 1)), corrupt="both")
            )
            if len(negatives) != len(positives):
                raise ValueError("negative_triples length must match positive_triples.")
            order = rng.permutation(len(positives))
            for idx in order:
                pos = self._triple_indices([positives[int(idx)]])[0]
                neg = self._triple_indices([negatives[int(idx)]])[0]
                pos_dist = self._distance_indexed(pos)
                neg_dist = self._distance_indexed(neg)
                if margin + pos_dist - neg_dist > 0.0:
                    self._apply_distance_gradient(pos, scale=1.0, lr=lr)
                    self._apply_distance_gradient(neg, scale=-1.0, lr=lr)
            if normalize_entities:
                self.normalize_entity_embeddings()
            history.append(self.margin_loss(positives, negatives, margin=margin))
        return KnowledgeGraphFitResult(self, history)

    def normalize_entity_embeddings(self, max_norm: float = 1.0) -> None:
        """Project entity embeddings into an L2 ball."""
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive.")
        norms = np.linalg.norm(self.entity_embeddings, axis=1)
        scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1.0e-300))
        self.entity_embeddings *= scale[:, None]

    def _distance_indexed(self, triple: np.ndarray) -> float:
        h, r, t = [int(x) for x in triple]
        diff = self.entity_embeddings[h] + self.relation_embeddings[r] - self.entity_embeddings[t]
        return float(np.dot(diff, diff))

    def _apply_distance_gradient(self, triple: np.ndarray, scale: float, lr: float) -> None:
        h, r, t = [int(x) for x in triple]
        diff = self.entity_embeddings[h] + self.relation_embeddings[r] - self.entity_embeddings[t]
        grad = 2.0 * float(scale) * diff
        self.entity_embeddings[h] -= lr * grad
        self.relation_embeddings[r] -= lr * grad
        self.entity_embeddings[t] += lr * grad

    def _triple_indices(self, triples: Sequence[Triple]) -> np.ndarray:
        arr = np.empty((len(triples), 3), dtype=np.int64)
        for i, (h, r, t) in enumerate(triples):
            arr[i, 0] = _lookup(h, self.entity_index, self.num_entities, "entity")
            arr[i, 1] = _lookup(r, self.relation_index, self.num_relations, "relation")
            arr[i, 2] = _lookup(t, self.entity_index, self.num_entities, "entity")
        return arr


def _lookup(value: Any, mapping: dict[Any, int], size: int, kind: str) -> int:
    if value in mapping:
        return mapping[value]
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < size:
        return int(value)
    raise ValueError("unknown %s %r." % (kind, value))
