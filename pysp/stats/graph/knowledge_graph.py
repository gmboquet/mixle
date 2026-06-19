"""Create, estimate, and sample from a knowledge-graph embedding distribution.

Defines KnowledgeGraphDistribution, KnowledgeGraphSampler, KnowledgeGraphAccumulatorFactory,
KnowledgeGraphAccumulator, KnowledgeGraphEstimator, and KnowledgeGraphDataEncoder for use with the
regular pysparkplug estimation framework (``optimize`` / ``seq_log_density`` / the PPL surface), like
the other ``pysp.stats.graph`` distributions.

Data type: a triple ``(h, r, t)`` of integer indices -- head entity, relation, tail entity. The model
embeds each entity and relation in ``dim`` dimensions and scores a triple by the DistMult bilinear form

    score(h, r, t) = sum_k E[h, k] * R[r, k] * E[t, k] = (E[h] * R[r]) . E[t],

and defines the conditional tail distribution by a softmax over all entities,

    p(t | h, r) = softmax_t score(h, r, t),     log p(h, r, t) = score(h, r, t) - logsumexp_a score(h, r, a).

This is the standard tail-prediction likelihood; maximizing it over observed triples is the model's MLE.
It has no closed form, so -- exactly like the Plackett-Luce minorization-maximization estimator in this
package -- each ``fit`` / ``optimize`` iteration performs one full-batch gradient-ascent step on the
embeddings, evaluated at the previous estimate (a random seeded init seeds the first pass). The threaded
``estimate`` carries the embeddings between passes, so no parameter state lives outside the framework.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _tail_log_posterior(entity: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Log softmax over all entities of the DistMult scores ``entity @ v`` for one query vector ``v``."""
    scores = entity @ v
    return scores - (scores.max() + np.log(np.sum(np.exp(scores - scores.max()))))


class KnowledgeGraphDistribution(SequenceEncodableProbabilityDistribution):
    """DistMult knowledge-graph embedding distribution over triples ``(h, r, t)``.

    ``entity_embeddings`` is ``(num_entities, dim)`` and ``relation_embeddings`` is
    ``(num_relations, dim)``. ``log_density((h, r, t))`` is the conditional tail log-probability
    ``log p(t | h, r)`` under the entity softmax.
    """

    def __init__(
        self,
        entity_embeddings: Any,
        relation_embeddings: Any,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.entity = np.asarray(entity_embeddings, dtype=float)
        self.relation = np.asarray(relation_embeddings, dtype=float)
        if self.entity.ndim != 2 or self.relation.ndim != 2 or self.entity.shape[1] != self.relation.shape[1]:
            raise ValueError("entity and relation embeddings must be 2-D and share the embedding dimension.")
        self.num_entities = int(self.entity.shape[0])
        self.num_relations = int(self.relation.shape[0])
        self.dim = int(self.entity.shape[1])
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "KnowledgeGraphDistribution(num_entities=%d, num_relations=%d, dim=%d, name=%s, keys=%s)" % (
            self.num_entities,
            self.num_relations,
            self.dim,
            repr(self.name),
            repr(self.keys),
        )

    def score(self, h: int, r: int, t: int) -> float:
        """DistMult score of a single triple (higher is more plausible)."""
        return float(np.sum(self.entity[h] * self.relation[r] * self.entity[t]))

    def tail_log_posterior(self, h: int, r: int) -> np.ndarray:
        """Length-``num_entities`` vector of ``log p(t | h, r)`` over all tail candidates."""
        return _tail_log_posterior(self.entity, self.entity[h] * self.relation[r])

    def log_density(self, x: Sequence[int]) -> float:
        h, r, t = int(x[0]), int(x[1]), int(x[2])
        return float(self.tail_log_posterior(h, r)[t])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=int)
        out = np.empty(x.shape[0], dtype=float)
        for n in range(x.shape[0]):
            out[n] = self.tail_log_posterior(x[n, 0], x[n, 1])[x[n, 2]]
        return out

    def sampler(self, seed: int | None = None) -> "KnowledgeGraphSampler":
        return KnowledgeGraphSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "KnowledgeGraphEstimator":
        return KnowledgeGraphEstimator(
            self.num_entities, self.num_relations, dim=self.dim, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "KnowledgeGraphDataEncoder":
        return KnowledgeGraphDataEncoder()


class KnowledgeGraphSampler(DistributionSampler):
    """Draw triples: head and relation uniformly, tail from the conditional softmax ``p(t | h, r)``."""

    def __init__(self, dist: KnowledgeGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None) -> Any:
        sz = 1 if size is None else size
        out = []
        for _ in range(sz):
            h = int(self.rng.randint(self.dist.num_entities))
            r = int(self.rng.randint(self.dist.num_relations))
            p = np.exp(self.dist.tail_log_posterior(h, r))
            t = int(self.rng.choice(self.dist.num_entities, p=p / p.sum()))
            out.append((h, r, t))
        return out[0] if size is None else out


class KnowledgeGraphAccumulator(SequenceEncodableStatisticAccumulator):
    """Collect the observed triples (and weights) for the estimator to train on.

    A DistMult embedding model has no finite sufficient statistic, so -- like other
    non-exponential-family models in this package -- the accumulator retains the data: it concatenates
    the ``(h, r, t)`` triples seen across the (possibly distributed) partitions.  The estimator then
    runs the gradient training in :meth:`KnowledgeGraphEstimator.estimate`.
    """

    def __init__(self, keys: str | None = None) -> None:
        self.key = keys
        self.triples: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        self.count = 0.0

    def update(self, x: Sequence[int], weight: float, estimate: KnowledgeGraphDistribution | None) -> None:
        self.seq_update(np.asarray([x], dtype=int), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: KnowledgeGraphDistribution | None) -> None:
        self.triples.append(np.asarray(x, dtype=int))
        self.weights.append(np.asarray(weights, dtype=float))
        self.count += float(np.sum(weights))

    def _stacked(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.triples:
            return np.zeros((0, 3), dtype=int), np.zeros(0)
        return np.concatenate(self.triples, axis=0), np.concatenate(self.weights)

    def combine(self, suff_stat: tuple) -> "KnowledgeGraphAccumulator":
        count, triples, weights = suff_stat
        self.count += count
        if len(triples):
            self.triples.append(np.asarray(triples, dtype=int))
            self.weights.append(np.asarray(weights, dtype=float))
        return self

    def value(self) -> tuple:
        triples, weights = self._stacked()
        return self.count, triples, weights

    def from_value(self, x: tuple) -> "KnowledgeGraphAccumulator":
        self.count = x[0]
        self.triples = [np.asarray(x[1], dtype=int)] if len(x[1]) else []
        self.weights = [np.asarray(x[2], dtype=float)] if len(x[1]) else []
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "KnowledgeGraphDataEncoder":
        return KnowledgeGraphDataEncoder()


class KnowledgeGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for KnowledgeGraphAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> KnowledgeGraphAccumulator:
        return KnowledgeGraphAccumulator(keys=self.keys)


class KnowledgeGraphEstimator(ParameterEstimator):
    """Train DistMult knowledge-graph embeddings by maximizing the tail-softmax log-likelihood.

    ``estimate`` runs vectorized mini-batch gradient ascent (``epochs`` passes, batch size
    ``batch_size``, step ``lr`` with L2 ``weight_decay``) from a deterministic seeded init, projecting
    each entity embedding back to the unit ball every epoch so the scale -- hence the step size -- stays
    well behaved.  One ``optimize`` / ``fit`` iteration (``max_its=1``) trains the model; the data is
    supplied through the accumulator like any other estimator.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 16,
        lr: float = 0.5,
        epochs: int = 100,
        batch_size: int = 256,
        weight_decay: float = 1.0e-4,
        init_scale: float = 0.3,
        max_norm: float = 1.0,
        seed: int = 1,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if num_entities < 2 or num_relations < 1 or dim < 1:
            raise ValueError("KnowledgeGraphEstimator requires num_entities>=2, num_relations>=1, dim>=1.")
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.init_scale = float(init_scale)
        self.max_norm = float(max_norm)
        self.seed = int(seed)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> KnowledgeGraphAccumulatorFactory:
        return KnowledgeGraphAccumulatorFactory(keys=self.keys)

    def _project(self, entity: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(entity, axis=1, keepdims=True)
        return entity * np.minimum(1.0, self.max_norm / np.maximum(norms, 1e-12))

    def estimate(self, nobs: float | None, suff_stat: tuple) -> KnowledgeGraphDistribution:
        _count, triples, weights = suff_stat
        rng = RandomState(self.seed)
        nE, nR, d = self.num_entities, self.num_relations, self.dim
        E = self._project(rng.normal(0.0, self.init_scale, (nE, d)))
        R = rng.normal(0.0, self.init_scale, (nR, d))
        triples = np.asarray(triples, dtype=int)
        if triples.shape[0] == 0:
            return KnowledgeGraphDistribution(E, R, name=self.name, keys=self.keys)
        weights = np.asarray(weights, dtype=float)
        n = triples.shape[0]
        bs = min(self.batch_size, n)
        ent_index = np.arange(nE)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                h, r, t = triples[idx, 0], triples[idx, 1], triples[idx, 2]
                w = weights[idx][:, None]
                v = E[h] * R[r]  # (B, d) DistMult query vectors
                scores = v @ E.T  # (B, nE)
                scores -= scores.max(axis=1, keepdims=True)
                p = np.exp(scores)
                p /= p.sum(axis=1, keepdims=True)
                onehot = (ent_index[None, :] == t[:, None]).astype(float)
                resid = (onehot - p) * w  # (B, nE)
                ebar = p @ E  # (B, d) expected tail embedding
                ge = resid.T @ v  # tail-role gradient over all entities, (nE, d)
                gr = np.zeros_like(R)
                head_grad = w * R[r] * (E[t] - ebar)
                rel_grad = w * E[h] * (E[t] - ebar)
                np.add.at(ge, h, head_grad)
                np.add.at(gr, r, rel_grad)
                m = len(idx)
                E = E + self.lr * (ge / m - self.weight_decay * E)
                R = R + self.lr * (gr / m - self.weight_decay * R)
            E = self._project(E)
        return KnowledgeGraphDistribution(E, R, name=self.name, keys=self.keys)


class KnowledgeGraphDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(h, r, t)`` triples into an ``(N, 3)`` integer array."""

    def __str__(self) -> str:
        return "KnowledgeGraphDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KnowledgeGraphDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        rv = np.asarray([list(row) for row in x], dtype=int)
        if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] == 0:
            raise ValueError("KnowledgeGraphDistribution requires a non-empty sequence of (h, r, t) triples.")
        return rv
