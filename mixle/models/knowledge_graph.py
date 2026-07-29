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
        self.entity_embeddings = np.asarray(entity_embeddings, dtype=np.float64).copy()
        self.relation_embeddings = np.asarray(relation_embeddings, dtype=np.float64).copy()
        if self.entity_embeddings.ndim != 2 or self.relation_embeddings.ndim != 2:
            raise ValueError("embeddings must be two-dimensional arrays.")
        if (
            self.entity_embeddings.shape[0] == 0
            or self.relation_embeddings.shape[0] == 0
            or self.entity_embeddings.shape[1] == 0
            or self.relation_embeddings.shape[1] == 0
        ):
            raise ValueError("entity and relation embeddings must have non-empty rows and dimensions.")
        if self.entity_embeddings.shape[1] != self.relation_embeddings.shape[1]:
            raise ValueError("entity and relation embeddings must share a dimension.")
        if not np.all(np.isfinite(self.entity_embeddings)) or not np.all(np.isfinite(self.relation_embeddings)):
            raise ValueError("entity and relation embeddings must contain only finite values.")
        self.num_entities = int(self.entity_embeddings.shape[0])
        self.num_relations = int(self.relation_embeddings.shape[0])
        self.embedding_dim = int(self.entity_embeddings.shape[1])
        self.entity_names = list(range(self.num_entities)) if entity_names is None else list(entity_names)
        self.relation_names = list(range(self.num_relations)) if relation_names is None else list(relation_names)
        if len(self.entity_names) != self.num_entities:
            raise ValueError("entity_names length must match entity_embeddings.")
        if len(self.relation_names) != self.num_relations:
            raise ValueError("relation_names length must match relation_embeddings.")
        self.entity_index = _unique_index(self.entity_names, "entity_names")
        self.relation_index = _unique_index(self.relation_names, "relation_names")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
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
        num_entities = _positive_int(num_entities, "num_entities")
        num_relations = _positive_int(num_relations, "num_relations")
        embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        scale = _positive_finite(scale, "scale")
        seed = _validated_seed(seed)
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
        margin = _positive_finite(margin, "margin")
        pos = self.distance_triples(positive_triples)
        neg = self.distance_triples(negative_triples)
        value = float(np.maximum(0.0, margin + pos - neg).sum())
        if not np.isfinite(value):
            raise RuntimeError("margin loss is non-finite.")
        return value

    def negative_sample(
        self,
        triples: Sequence[Triple],
        seed: int | None = None,
        corrupt: str = "tail",
        *,
        known_triples: Sequence[Triple] | None = None,
    ) -> list[Triple]:
        """Produce filtered corruptions that are not any supplied known true triple."""
        if corrupt not in ("head", "tail", "both"):
            raise ValueError("corrupt must be 'head', 'tail', or 'both'.")
        seed = _validated_seed(seed)
        indexed = self._triple_indices(triples)
        known_source = triples if known_triples is None else known_triples
        known = {tuple(int(value) for value in row) for row in self._triple_indices(known_source)}
        known.update(tuple(int(value) for value in row) for row in indexed)
        rng = np.random.RandomState(seed)
        rv: list[Triple] = []
        for h, r, t in indexed:
            candidates = []
            if corrupt in {"head", "both"}:
                candidates.extend(
                    (candidate, int(r), int(t))
                    for candidate in range(self.num_entities)
                    if candidate != h and (candidate, int(r), int(t)) not in known
                )
            if corrupt in {"tail", "both"}:
                candidates.extend(
                    (int(h), int(r), candidate)
                    for candidate in range(self.num_entities)
                    if candidate != t and (int(h), int(r), candidate) not in known
                )
            if not candidates:
                triple = (
                    self.entity_names[int(h)],
                    self.relation_names[int(r)],
                    self.entity_names[int(t)],
                )
                raise ValueError(f"no filtered {corrupt} corruption is available for true triple {triple!r}")
            sampled = candidates[int(rng.randint(len(candidates)))]
            rv.append(
                (
                    self.entity_names[sampled[0]],
                    self.relation_names[sampled[1]],
                    self.entity_names[sampled[2]],
                )
            )
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
        margin = _positive_finite(margin, "margin")
        lr = _positive_finite(lr, "lr")
        max_its = _positive_int(max_its, "max_its")
        seed = _validated_seed(seed)
        if not isinstance(normalize_entities, (bool, np.bool_)):
            raise ValueError("normalize_entities must be a boolean.")
        rng = np.random.RandomState(seed)
        history: list[float] = []
        positives = list(positive_triples)
        positive_indices = self._triple_indices(positives)
        known = {tuple(int(value) for value in row) for row in positive_indices}
        fixed_negatives = None
        if negative_triples is not None:
            fixed_negatives = list(negative_triples)
            if len(fixed_negatives) != len(positives):
                raise ValueError("negative_triples length must match positive_triples.")
            negative_indices = self._triple_indices(fixed_negatives)
            for index, (positive, negative) in enumerate(zip(positive_indices, negative_indices, strict=True)):
                negative_key = tuple(int(value) for value in negative)
                if negative_key == tuple(int(value) for value in positive):
                    raise ValueError(f"negative_triples[{index}] is identical to its positive triple.")
                if negative_key in known:
                    raise ValueError(f"negative_triples[{index}] is a known positive triple.")

        for _ in range(max_its):
            negatives = fixed_negatives or self.negative_sample(
                positives,
                seed=int(rng.randint(0, 2**31 - 1)),
                corrupt="both",
                known_triples=positives,
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
            if not np.all(np.isfinite(self.entity_embeddings)) or not np.all(np.isfinite(self.relation_embeddings)):
                raise RuntimeError("knowledge-graph fitting produced non-finite embeddings.")
            history.append(self.margin_loss(positives, negatives, margin=margin))
        return KnowledgeGraphFitResult(self, history)

    def normalize_entity_embeddings(self, max_norm: float = 1.0) -> None:
        """Project entity embeddings into an L2 ball."""
        max_norm = _positive_finite(max_norm, "max_norm")
        if not np.all(np.isfinite(self.entity_embeddings)):
            raise ValueError("entity embeddings must be finite before normalization.")
        norms = np.linalg.norm(self.entity_embeddings, axis=1)
        scale = np.minimum(1.0, max_norm / np.maximum(norms, 1.0e-300))
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
        if not isinstance(triples, Sequence) or isinstance(triples, (str, bytes)):
            raise ValueError("triples must be a sequence of three-item triples.")
        arr = np.empty((len(triples), 3), dtype=np.int64)
        for i, triple in enumerate(triples):
            if not isinstance(triple, Sequence) or isinstance(triple, (str, bytes)) or len(triple) != 3:
                raise ValueError(f"triples[{i}] must contain exactly (head, relation, tail).")
            h, r, t = triple
            arr[i, 0] = _lookup(h, self.entity_index, self.num_entities, "entity")
            arr[i, 1] = _lookup(r, self.relation_index, self.num_relations, "relation")
            arr[i, 2] = _lookup(t, self.entity_index, self.num_entities, "entity")
        return arr


def _lookup(value: Any, mapping: dict[Any, int], size: int, kind: str) -> int:
    try:
        if value in mapping:
            return mapping[value]
    except TypeError as exc:
        raise ValueError(f"{kind} identity {value!r} must be hashable.") from exc
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < size:
        return int(value)
    raise ValueError("unknown %s %r." % (kind, value))


def _unique_index(names: Sequence[Any], name: str) -> dict[Any, int]:
    index = {}
    for position, value in enumerate(names):
        try:
            if value in index:
                raise ValueError(f"{name} must contain unique identities; duplicate {value!r}.")
            index[value] = position
        except TypeError as exc:
            raise ValueError(f"{name}[{position}] must be hashable.") from exc
    return index


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a strictly positive finite scalar.")
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a strictly positive finite scalar.")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a strictly positive finite scalar.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a strictly positive finite scalar.")
    return result


def _validated_seed(seed: Any) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be None or an integer from 0 through 2**32 - 1.")
    result = int(seed)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be None or an integer from 0 through 2**32 - 1.")
    return result
