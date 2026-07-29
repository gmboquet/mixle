"""Knowledge-graph embedding distributions for integer triple observations.

Data type: a triple ``(h, r, t)`` of integer indices -- head entity, relation, tail entity. The model
embeds each entity and relation in ``dim`` dimensions and scores a triple by the DistMult bilinear form

    score(h, r, t) = sum_k E[h, k] * R[r, k] * E[t, k] = (E[h] * R[r]) . E[t],

and defines the conditional tail distribution by a softmax over all entities,

    p(t | h, r) = softmax_t score(h, r, t).

The modeled joint law uses explicit uniform context laws,
``p(h,r,t)=p(h)p(r)p(t|h,r)``. The sampler and ``log_density`` therefore
describe the same normalized random variable; tail/head/relation posterior
helpers remain conditional query APIs. Maximizing the joint law is equivalent
to maximizing its tail-conditional term because the context factors are
fixed.
It has no closed form, so -- exactly like the Plackett-Luce minorization-maximization estimator in this
package -- each ``fit`` / ``optimize`` iteration performs one full-batch gradient-ascent step on the
embeddings, evaluated at the previous estimate (a random seeded init seeds the first pass). The threaded
``estimate`` carries the embeddings between passes, so no parameter state lives outside the framework.
"""

import itertools
import math
import operator
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

# Canonical guarded row-wise softmax. The DistMult scores fed here are finite dot-products of
# embeddings, so the all-(-inf)-row guard never triggers and results are identical to the previous
# local implementation; the guard is a harmless safety net.
from mixle.utils.special import softmax_rows as _softmax_rows


class KnowledgeGraphStatistics(NamedTuple):
    """Versioned weighted triples plus an optional immutable warm start."""

    count: float
    triples: np.ndarray
    weights: np.ndarray
    num_entities: int
    num_relations: int
    dim: int
    warm_entity: np.ndarray | None
    warm_relation: np.ndarray | None

    @property
    def schema_version(self) -> int:
        """Return the serialized-statistic schema version."""
        return 1


def _exact_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an exact integer") from exc


def _finite_nonnegative_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _validated_id(value: Any, *, size: int, label: str) -> int:
    result = _exact_integer(value, label=label)
    if result < 0 or result >= size:
        raise ValueError(f"{label} must lie in [0, {size})")
    return result


def _validated_triples(
    value: Any,
    *,
    num_entities: int,
    num_relations: int,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("knowledge-graph triples must be array-like") from exc
    if raw.size == 0:
        if not ((raw.ndim == 1 and raw.shape == (0,)) or (raw.ndim == 2 and raw.shape[1] == 3)):
            raise ValueError("knowledge-graph triples must have exact shape (N, 3)")
        result = np.empty((0, 3), dtype=np.int64)
        result.setflags(write=False)
        return result
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("knowledge-graph triples must have exact shape (N, 3)")
    if raw.dtype == np.bool_:
        raise TypeError("knowledge-graph identifiers must be exact integers")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("knowledge-graph identifiers must be exact integers") from exc
    if np.any(~np.isfinite(numeric)) or not np.array_equal(numeric, np.round(numeric)):
        raise TypeError("knowledge-graph identifiers must be exact integers")
    triples = numeric.astype(np.int64)
    if (
        np.any(triples[:, 0] < 0)
        or np.any(triples[:, 0] >= num_entities)
        or np.any(triples[:, 1] < 0)
        or np.any(triples[:, 1] >= num_relations)
        or np.any(triples[:, 2] < 0)
        or np.any(triples[:, 2] >= num_entities)
    ):
        raise ValueError("knowledge-graph identifier is outside model bounds")
    triples.setflags(write=False)
    return triples


def _validated_single_triple(
    value: Any,
    *,
    num_entities: int,
    num_relations: int,
) -> tuple[int, int, int]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("knowledge-graph triple must be array-like") from exc
    if raw.ndim != 1 or raw.size != 3:
        raise ValueError("knowledge-graph triple must contain exactly (h, r, t)")
    triple = _validated_triples(
        raw.reshape(1, 3),
        num_entities=num_entities,
        num_relations=num_relations,
    )[0]
    return int(triple[0]), int(triple[1]), int(triple[2])


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("knowledge-graph weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"knowledge-graph weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("knowledge-graph weights must be finite and non-negative")
    return weights


def _owned_embeddings(
    entity: Any,
    relation: Any,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        owned_entity = np.asarray(entity, dtype=np.float64)
        owned_relation = np.asarray(relation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("knowledge-graph embeddings must be numeric") from exc
    if (
        owned_entity.ndim != 2
        or owned_relation.ndim != 2
        or owned_entity.shape[0] == 0
        or owned_relation.shape[0] == 0
        or owned_entity.shape[1] == 0
        or owned_entity.shape[1] != owned_relation.shape[1]
        or np.any(~np.isfinite(owned_entity))
        or np.any(~np.isfinite(owned_relation))
    ):
        raise ValueError(
            "knowledge-graph embeddings must be nonempty finite 2-D arrays with a shared embedding dimension"
        )
    owned_entity = owned_entity.copy()
    owned_relation = owned_relation.copy()
    owned_entity.setflags(write=False)
    owned_relation.setflags(write=False)
    return owned_entity, owned_relation


def _validated_statistics(
    value: Any,
    *,
    num_entities: int,
    num_relations: int,
    dim: int,
) -> KnowledgeGraphStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 8:
        raise ValueError(
            "knowledge-graph statistics must contain count, triples, weights, "
            "dimensions, and optional warm-start embeddings"
        )
    (
        raw_count,
        raw_triples,
        raw_weights,
        raw_entities,
        raw_relations,
        raw_dim,
        raw_warm_entity,
        raw_warm_relation,
    ) = value
    statistic_entities = _exact_integer(
        raw_entities,
        label="knowledge-graph statistic entity count",
    )
    statistic_relations = _exact_integer(
        raw_relations,
        label="knowledge-graph statistic relation count",
    )
    statistic_dim = _exact_integer(
        raw_dim,
        label="knowledge-graph statistic dimension",
    )
    if statistic_entities != num_entities or statistic_relations != num_relations or statistic_dim != dim:
        raise ValueError("knowledge-graph statistic dimensions do not match estimator")
    triples = _validated_triples(
        raw_triples,
        num_entities=num_entities,
        num_relations=num_relations,
    )
    weights = _validated_weights(raw_weights, len(triples)).copy()
    weights.setflags(write=False)
    count = _finite_nonnegative_scalar(
        raw_count,
        label="knowledge-graph total weight",
    )
    if not math.isclose(
        count,
        float(np.sum(weights)),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("knowledge-graph total weight contradicts row weights")
    if (raw_warm_entity is None) != (raw_warm_relation is None):
        raise ValueError("knowledge-graph warm start requires both embedding arrays")
    warm_entity = None
    warm_relation = None
    if raw_warm_entity is not None:
        warm_entity, warm_relation = _owned_embeddings(
            raw_warm_entity,
            raw_warm_relation,
        )
        if warm_entity.shape != (num_entities, dim) or warm_relation.shape != (num_relations, dim):
            raise ValueError("knowledge-graph warm-start embedding shapes do not match estimator")
    return KnowledgeGraphStatistics(
        count,
        triples,
        weights,
        num_entities,
        num_relations,
        dim,
        warm_entity,
        warm_relation,
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
        fit_receipt: dict[str, Any] | None = None,
    ) -> None:
        self.entity, self.relation = _owned_embeddings(
            entity_embeddings,
            relation_embeddings,
        )
        self.num_entities = int(self.entity.shape[0])
        self.num_relations = int(self.relation.shape[0])
        self.dim = int(self.entity.shape[1])
        self.name = name
        self.keys = keys
        self.fit_receipt = dict(fit_receipt or {})

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
        h = _validated_id(h, size=self.num_entities, label="head id")
        r = _validated_id(r, size=self.num_relations, label="relation id")
        t = _validated_id(t, size=self.num_entities, label="tail id")
        return float(np.sum(self.entity[h] * self.relation[r] * self.entity[t]))

    def tail_log_posterior(self, h: int, r: int) -> np.ndarray:
        """Length-``num_entities`` vector of ``log p(t | h, r)`` over all tail candidates."""
        h = _validated_id(h, size=self.num_entities, label="head id")
        r = _validated_id(r, size=self.num_relations, label="relation id")
        return _tail_log_posterior(self.entity, self.entity[h] * self.relation[r])

    def head_log_posterior(self, r: int, t: int) -> np.ndarray:
        """Length-``num_entities`` vector of ``log p(h | r, t)`` over all head candidates."""
        r = _validated_id(r, size=self.num_relations, label="relation id")
        t = _validated_id(t, size=self.num_entities, label="tail id")
        return _tail_log_posterior(self.entity, self.relation[r] * self.entity[t])

    def relation_log_posterior(self, h: int, t: int) -> np.ndarray:
        """Length-``num_relations`` vector of ``log p(r | h, t)`` over all relation candidates."""
        h = _validated_id(h, size=self.num_entities, label="head id")
        t = _validated_id(t, size=self.num_entities, label="tail id")
        return _tail_log_posterior(self.relation, self.entity[h] * self.entity[t])

    def complete(self, h: int | None = None, r: int | None = None, t: int | None = None) -> np.ndarray:
        """Log-posterior over candidates for the single missing slot of a query.

        Exactly one of ``h``, ``r``, ``t`` must be ``None``; the returned vector is over entities (for a
        missing head or tail) or relations (for a missing relation).
        """
        missing = [name for name, v in (("h", h), ("r", r), ("t", t)) if v is None]
        if len(missing) != 1:
            raise ValueError("complete() needs exactly one of h, r, t to be None (the slot to fill).")
        if t is None:
            return self.tail_log_posterior(h, r)
        if h is None:
            return self.head_log_posterior(r, t)
        return self.relation_log_posterior(h, t)

    def rank(
        self,
        h: int | None = None,
        r: int | None = None,
        t: int | None = None,
        exclude: Any = (),
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rank candidates for the missing slot by log-probability, dropping ``exclude`` candidates.

        Returns ``[(candidate, log_prob), ...]`` highest first (the most plausible completions).
        """
        logp = self.complete(h=h, r=r, t=t)
        order = np.argsort(-logp)
        candidate_size = len(logp)
        excl = (
            {
                _validated_id(
                    item,
                    size=candidate_size,
                    label="excluded candidate id",
                )
                for item in exclude
            }
            if len(exclude)
            else set()
        )
        ranked = [(int(c), float(logp[c])) for c in order if int(c) not in excl]
        if top_n is None:
            return ranked
        checked_top = _exact_integer(top_n, label="top_n")
        if checked_top < 0:
            raise ValueError("top_n must be non-negative")
        return ranked[:checked_top]

    def recommend(self, known: Any, top_n: int = 10) -> list[tuple[int, int, int, float]]:
        """Recommend the most plausible missing tail facts for the ``(h, r)`` contexts in ``known``.

        ``known`` is a sequence of observed ``(h, r, t)`` triples; for each distinct ``(h, r)`` the
        already-present tails are excluded, the remaining tails are ranked by ``log p(t | h, r)``, and
        the global top ``top_n`` new facts are returned as ``[(h, r, t, log_prob), ...]``.
        """
        known = _validated_triples(
            list(known),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )
        seen: dict[tuple[int, int], set] = {}
        for h, r, t in known:
            seen.setdefault((int(h), int(r)), set()).add(int(t))
        out: list[tuple[int, int, int, float]] = []
        for (h, r), tails in seen.items():
            for t, lp in self.rank(h=h, r=r, exclude=tails):
                out.append((h, r, t, lp))
        out.sort(key=lambda u: -u[3])
        checked_top = _exact_integer(top_n, label="top_n")
        if checked_top < 0:
            raise ValueError("top_n must be non-negative")
        return out[:checked_top]

    def recommend_subgraph(self, node: int, known: Any, top_n: int = 5) -> list[tuple[int, int, int, float]]:
        """Recommend plausible new edges incident to ``node`` (both ``(node, r, ?)`` and ``(?, r, node)``).

        Excludes edges already in ``known`` and returns the top ``top_n`` by log-probability as
        ``[(h, r, t, log_prob), ...]``, the suggested missing subgraph around the node.
        """
        node = _validated_id(
            node,
            size=self.num_entities,
            label="node id",
        )
        known_array = _validated_triples(
            list(known),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )
        known_set = {(int(h), int(r), int(t)) for h, r, t in known_array}
        cand: list[tuple[int, int, int, float]] = []
        for r in range(self.num_relations):
            for t, lp in self.rank(h=node, r=r):
                if (node, r, t) not in known_set:
                    cand.append((node, r, t, lp))
            for h, lp in self.rank(r=r, t=node):
                if (h, r, node) not in known_set:
                    cand.append((h, r, node, lp))
        cand.sort(key=lambda u: -u[3])
        checked_top = _exact_integer(top_n, label="top_n")
        if checked_top < 0:
            raise ValueError("top_n must be non-negative")
        return cand[:checked_top]

    def pattern(
        self, pattern: Any, candidates: Any = None, known: Any = None, beam: int = 64
    ) -> "KnowledgeGraphPattern":
        """A subgraph-pattern query over this model for flexible enumeration of missing parts.

        ``pattern`` is a list of triples whose slots are either fixed integer ids or named variables
        (strings starting with ``'?'``), variables shared across edges (e.g.
        ``[(alice, friend, '?x'), ('?x', lives_in, '?c')]``).  The returned
        :class:`KnowledgeGraphPattern` enumerates the variable bindings (completed subgraphs) in
        descending joint plausibility, restricts variables to ``candidates`` if given, drops groundings
        that add nothing new when ``known`` is given, and plugs into
        :class:`~mixle.ppl.ConformalStructure` for a calibrated set of completed subgraphs.
        """
        return KnowledgeGraphPattern(self, pattern, candidates=candidates, known=known, beam=beam)

    def log_density(self, x: Sequence[int]) -> float:
        """Return normalized joint ``log p(h,r,t)`` for one triple."""
        h, r, t = _validated_single_triple(
            x,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )
        context_log_prob = -math.log(self.num_entities) - math.log(self.num_relations)
        return float(self.tail_log_posterior(h, r)[t] + context_log_prob)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized tail log-probabilities for encoded triples."""
        triples = _validated_triples(
            x,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )
        out = np.empty(triples.shape[0], dtype=float)
        context_log_prob = -math.log(self.num_entities) - math.log(self.num_relations)
        for n in range(triples.shape[0]):
            out[n] = (
                self.tail_log_posterior(
                    triples[n, 0],
                    triples[n, 1],
                )[triples[n, 2]]
                + context_log_prob
            )
        return out

    def sampler(self, seed: int | None = None) -> "KnowledgeGraphSampler":
        """Return a sampler for observed triples."""
        return KnowledgeGraphSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "KnowledgeGraphEstimator":
        """Return a DistMult embedding estimator for this entity/relation shape."""
        if pseudo_count is not None:
            checked = _finite_nonnegative_scalar(
                pseudo_count,
                label="knowledge-graph pseudo-count",
            )
            if checked != 0.0:
                raise NotImplementedError(
                    "KnowledgeGraph does not define an implicit pseudo-count "
                    "prior; configure explicit regularization instead"
                )
        return KnowledgeGraphEstimator(
            self.num_entities,
            self.num_relations,
            dim=self.dim,
            name=self.name,
            keys=self.keys,
            initial_model=self,
        )

    def dist_to_encoder(self) -> "KnowledgeGraphDataEncoder":
        """Return the triple encoder used by vectorized methods."""
        return KnowledgeGraphDataEncoder(
            self.num_entities,
            self.num_relations,
        )


class KnowledgeGraphSampler(DistributionSampler):
    """Draw triples: head and relation uniformly, tail from the conditional softmax ``p(t | h, r)``."""

    def __init__(self, dist: KnowledgeGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one triple or ``size`` iid triples."""
        sz = 1 if size is None else _exact_integer(size, label="knowledge-graph sample size")
        if sz < 0:
            raise ValueError("knowledge-graph sample size must be non-negative")
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

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        keys: str | None = None,
    ) -> None:
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.keys = keys
        self.triples: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        self.count = 0.0
        self.warm_entity: np.ndarray | None = None
        self.warm_relation: np.ndarray | None = None

    def _set_warm(
        self,
        estimate: KnowledgeGraphDistribution | None,
    ) -> None:
        if estimate is None:
            return
        if estimate.entity.shape != (self.num_entities, self.dim) or estimate.relation.shape != (
            self.num_relations,
            self.dim,
        ):
            raise ValueError("knowledge-graph warm-start model shape does not match accumulator")
        entity, relation = _owned_embeddings(
            estimate.entity,
            estimate.relation,
        )
        if self.warm_entity is not None and (
            not np.array_equal(self.warm_entity, entity) or not np.array_equal(self.warm_relation, relation)
        ):
            raise ValueError("knowledge-graph accumulator received conflicting warm starts")
        self.warm_entity = entity
        self.warm_relation = relation

    def update(self, x: Sequence[int], weight: float, estimate: KnowledgeGraphDistribution | None) -> None:
        """Store one weighted triple for embedding training."""
        triple = _validated_single_triple(
            x,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )
        self.seq_update(
            np.asarray([triple], dtype=np.int64),
            np.asarray([weight], dtype=float),
            estimate,
        )

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Store one weighted triple during initialization."""
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Store encoded triples during initialization."""
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: KnowledgeGraphDistribution | None) -> None:
        """Store encoded triples and weights for embedding training."""
        triples = _validated_triples(
            x,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )
        checked_weights = _validated_weights(weights, len(triples)).copy()
        checked_weights.setflags(write=False)
        self._set_warm(estimate)
        self.triples.append(triples)
        self.weights.append(checked_weights)
        self.count += float(np.sum(checked_weights))

    def _stacked(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.triples:
            return np.zeros((0, 3), dtype=int), np.zeros(0)
        return np.concatenate(self.triples, axis=0), np.concatenate(self.weights)

    def combine(self, suff_stat: Any) -> "KnowledgeGraphAccumulator":
        """Merge stored triples and weights from another accumulator value."""
        checked = _validated_statistics(
            suff_stat,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            dim=self.dim,
        )
        self.count += checked.count
        if len(checked.triples):
            self.triples.append(checked.triples)
            self.weights.append(checked.weights)
        if checked.warm_entity is not None:
            model = KnowledgeGraphDistribution(
                checked.warm_entity,
                checked.warm_relation,
            )
            self._set_warm(model)
        return self

    def value(self) -> KnowledgeGraphStatistics:
        """Return total weight, stacked triples, and stacked weights."""
        triples, weights = self._stacked()
        triples = triples.copy()
        weights = weights.copy()
        triples.setflags(write=False)
        weights.setflags(write=False)
        return KnowledgeGraphStatistics(
            self.count,
            triples,
            weights,
            self.num_entities,
            self.num_relations,
            self.dim,
            self.warm_entity,
            self.warm_relation,
        )

    def from_value(self, x: Any) -> "KnowledgeGraphAccumulator":
        """Restore stored triples and weights from ``value`` output."""
        checked = _validated_statistics(
            x,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            dim=self.dim,
        )
        self.count = checked.count
        self.triples = [checked.triples] if len(checked.triples) else []
        self.weights = [checked.weights] if len(checked.triples) else []
        self.warm_entity = checked.warm_entity
        self.warm_relation = checked.warm_relation
        return self

    def scale(self, c: float) -> "KnowledgeGraphAccumulator":
        """Scale the observation weights without changing triples or warm start."""
        checked = _finite_nonnegative_scalar(
            c,
            label="knowledge-graph scale",
        )
        self.weights = [weights * checked for weights in self.weights]
        self.count *= checked
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge evidence under the configured shared key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace evidence from the configured shared key."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "KnowledgeGraphDataEncoder":
        """Return the encoder compatible with stored triples."""
        return KnowledgeGraphDataEncoder(
            self.num_entities,
            self.num_relations,
        )


class KnowledgeGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for KnowledgeGraphAccumulator."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        keys: str | None = None,
    ) -> None:
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.keys = keys

    def make(self) -> KnowledgeGraphAccumulator:
        """Create an empty knowledge-graph accumulator."""
        return KnowledgeGraphAccumulator(
            self.num_entities,
            self.num_relations,
            self.dim,
            keys=self.keys,
        )


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
        weight_decay: float = 0.0,
        init_scale: float = 0.3,
        max_norm: float = 1.0,
        directions: tuple = ("tail",),
        negatives: int | None = None,
        seed: int = 1,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        objective: str | None = None,
        initial_model: KnowledgeGraphDistribution | None = None,
    ) -> None:
        self.num_entities = _exact_integer(
            num_entities,
            label="knowledge-graph entity count",
        )
        self.num_relations = _exact_integer(
            num_relations,
            label="knowledge-graph relation count",
        )
        self.dim = _exact_integer(dim, label="knowledge-graph dimension")
        if self.num_entities < 2 or self.num_relations < 1 or self.dim < 1:
            raise ValueError("KnowledgeGraphEstimator requires num_entities>=2, num_relations>=1, dim>=1.")
        self.lr = float(lr)
        self.epochs = _exact_integer(epochs, label="knowledge-graph epochs")
        self.batch_size = _exact_integer(
            batch_size,
            label="knowledge-graph batch size",
        )
        self.weight_decay = float(weight_decay)
        self.init_scale = float(init_scale)
        self.max_norm = float(max_norm)
        self.directions = tuple(directions)
        if (
            not np.isfinite(self.lr)
            or self.lr <= 0.0
            or self.epochs <= 0
            or self.batch_size <= 0
            or not np.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not np.isfinite(self.init_scale)
            or self.init_scale <= 0.0
            or not np.isfinite(self.max_norm)
            or self.max_norm <= 0.0
        ):
            raise ValueError(
                "knowledge-graph training requires lr>0, epochs>0, "
                "batch_size>0, weight_decay>=0, init_scale>0, and max_norm>0"
            )
        allowed_directions = {"tail", "head", "relation"}
        if (
            not self.directions
            or len(set(self.directions)) != len(self.directions)
            or not set(self.directions) <= allowed_directions
        ):
            raise ValueError("knowledge-graph directions must be unique members of {'tail', 'head', 'relation'}")
        if objective is None:
            objective = "joint_likelihood" if self.directions == ("tail",) else "pseudo_likelihood"
        if objective not in {"joint_likelihood", "pseudo_likelihood"}:
            raise ValueError("knowledge-graph objective must be 'joint_likelihood' or 'pseudo_likelihood'")
        if objective == "joint_likelihood" and self.directions != ("tail",):
            raise ValueError(
                "joint_likelihood trains only the tail conditional because head/relation context laws are fixed uniform"
            )
        self.objective = objective
        if negatives is None:
            self.negatives = None
        else:
            self.negatives = _exact_integer(
                negatives,
                label="knowledge-graph negative count",
            )
            if self.negatives <= 0 or self.negatives >= self.num_entities:
                raise ValueError("knowledge-graph negatives must lie in [1, num_entities)")
        self.seed = _exact_integer(seed, label="knowledge-graph seed")
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("knowledge-graph seed must lie in [0, 2**32)")
        if pseudo_count is not None:
            checked_pseudo = _finite_nonnegative_scalar(
                pseudo_count,
                label="knowledge-graph pseudo-count",
            )
            if checked_pseudo != 0.0:
                raise NotImplementedError(
                    "KnowledgeGraphEstimator does not define an implicit "
                    "pseudo-count prior; use weight_decay or an explicit prior"
                )
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys
        self.initial_entity = None
        self.initial_relation = None
        if initial_model is not None:
            if initial_model.entity.shape != (self.num_entities, self.dim) or initial_model.relation.shape != (
                self.num_relations,
                self.dim,
            ):
                raise ValueError("knowledge-graph initial model shape does not match estimator")
            self.initial_entity, self.initial_relation = _owned_embeddings(
                initial_model.entity,
                initial_model.relation,
            )
        self.outer_objective_compatible = (
            self.objective == "joint_likelihood" and self.negatives is None and self.weight_decay == 0.0
        )

    def accumulator_factory(self) -> KnowledgeGraphAccumulatorFactory:
        """Return a factory for stored-triple accumulators."""
        return KnowledgeGraphAccumulatorFactory(
            self.num_entities,
            self.num_relations,
            self.dim,
            keys=self.keys,
        )

    def _project(self, entity: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(entity, axis=1, keepdims=True)
        return entity * np.minimum(1.0, self.max_norm / np.maximum(norms, 1e-12))

    def _entity_direction_grad(self, E, R, q, target, other, r, w, ge, gr, rng):
        """Accumulate the gradient of ``log p(target_entity | q)`` (q = R[r] * E[other]) into ge, gr.

        Full softmax over all entities by default, or sampled softmax against ``self.negatives`` uniform
        negatives per row (so the per-row cost is O(K d) instead of O(num_entities d), the key to scaling
        to large graphs). The context-role gradient flows to E[other] and R[r] identically either way.
        """
        if self.negatives is None:
            p = _softmax_rows(q @ E.T)
            ebar = p @ E
            resid = ((np.arange(E.shape[0])[None, :] == target[:, None]) - p) * w
            ge += resid.T @ q
        else:
            k = int(self.negatives)
            cand = np.empty((q.shape[0], k + 1), dtype=np.int64)
            cand[:, 0] = target
            all_entities = np.arange(E.shape[0])
            for row, positive in enumerate(target):
                pool = all_entities[all_entities != positive]
                cand[row, 1:] = rng.choice(
                    pool,
                    size=k,
                    replace=False,
                )
            cand_emb = E[cand]  # (m, 1+k, d); column 0 is the positive
            logits = np.einsum("bkd,bd->bk", cand_emb, q)
            # Exact collision semantics: the positive appears once at column
            # zero and negatives are unique samples without replacement from
            # the remaining entities. Correct negative logits by their
            # inclusion probability k/(N-1); this is an explicitly named
            # sampled-softmax approximation, not the public exact likelihood.
            logits[:, 1:] -= math.log(k / (E.shape[0] - 1.0))
            p = _softmax_rows(logits)
            ebar = np.einsum("bk,bkd->bd", p, cand_emb)
            onehot = np.zeros_like(p)
            onehot[:, 0] = 1.0
            resid = (onehot - p) * w
            np.add.at(ge, cand.reshape(-1), (resid[:, :, None] * q[:, None, :]).reshape(-1, q.shape[1]))
        np.add.at(ge, other, w * R[r] * (E[target] - ebar))
        np.add.at(gr, r, w * E[other] * (E[target] - ebar))

    def estimate(self, nobs: float | None, suff_stat: tuple) -> KnowledgeGraphDistribution:
        """Fit DistMult embeddings from stored triples and weights."""
        checked = _validated_statistics(
            suff_stat,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            dim=self.dim,
        )
        if checked.count <= 0.0 or checked.triples.shape[0] == 0:
            raise ValueError("knowledge-graph fitting requires positively weighted triples")
        rng = RandomState(self.seed)
        nE, nR, d = self.num_entities, self.num_relations, self.dim
        warm_entity = checked.warm_entity if checked.warm_entity is not None else self.initial_entity
        warm_relation = checked.warm_relation if checked.warm_relation is not None else self.initial_relation
        used_warm_start = warm_entity is not None
        if used_warm_start:
            E = np.asarray(warm_entity, dtype=np.float64).copy()
            R = np.asarray(warm_relation, dtype=np.float64).copy()
        else:
            E = self._project(rng.normal(0.0, self.init_scale, (nE, d)))
            R = rng.normal(0.0, self.init_scale, (nR, d))
        triples = checked.triples
        weights = checked.weights
        n = triples.shape[0]
        bs = min(self.batch_size, n)
        rel_index = np.arange(nR)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                h, r, t = triples[idx, 0], triples[idx, 1], triples[idx, 2]
                w = weights[idx][:, None]
                m = len(idx)
                ge = np.zeros_like(E)
                gr = np.zeros_like(R)
                if "tail" in self.directions:  # maximize log p(t | h, r)
                    self._entity_direction_grad(E, R, E[h] * R[r], t, h, r, w, ge, gr, rng)
                if "head" in self.directions:  # maximize log p(h | r, t)
                    self._entity_direction_grad(E, R, R[r] * E[t], h, t, r, w, ge, gr, rng)
                if "relation" in self.directions:  # maximize log p(r | h, t)  (relations are few; full softmax)
                    q = E[h] * E[t]
                    pr = _softmax_rows(q @ R.T)
                    rbar = pr @ R
                    resid = ((rel_index[None, :] == r[:, None]) - pr) * w
                    gr += resid.T @ q
                    np.add.at(ge, h, w * E[t] * (R[r] - rbar))
                    np.add.at(ge, t, w * E[h] * (R[r] - rbar))
                E = E + self.lr * (ge / m - self.weight_decay * E)
                R = R + self.lr * (gr / m - self.weight_decay * R)
            E = self._project(E)
            if np.any(~np.isfinite(E)) or np.any(~np.isfinite(R)):
                raise RuntimeError("knowledge-graph training produced non-finite embeddings")
        training_objective = self.objective
        if self.negatives is not None:
            training_objective = "corrected_sampled_" + training_objective
        if self.weight_decay:
            training_objective = "l2_penalized_" + training_objective
        return KnowledgeGraphDistribution(
            E,
            R,
            name=self.name,
            keys=self.keys,
            fit_receipt={
                "objective": training_objective,
                "directions": self.directions,
                "negative_count": self.negatives,
                "warm_start_used": used_warm_start,
                "epochs": self.epochs,
                "complete": True,
            },
        )


class KnowledgeGraphDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(h, r, t)`` triples into an ``(N, 3)`` integer array."""

    def __init__(self, num_entities: int, num_relations: int) -> None:
        self.num_entities = _exact_integer(
            num_entities,
            label="knowledge-graph entity count",
        )
        self.num_relations = _exact_integer(
            num_relations,
            label="knowledge-graph relation count",
        )
        if self.num_entities <= 0 or self.num_relations <= 0:
            raise ValueError("knowledge-graph encoder dimensions must be positive")

    def __str__(self) -> str:
        return "KnowledgeGraphDataEncoder(%s, %s)" % (
            repr(self.num_entities),
            repr(self.num_relations),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, KnowledgeGraphDataEncoder)
            and self.num_entities == other.num_entities
            and self.num_relations == other.num_relations
        )

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode triples as an ``(N, 3)`` integer array."""
        return _validated_triples(
            list(x),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
        )

    def row_count(self, x: Any) -> int:
        """Return the number of validated encoded triples."""
        return int(
            _validated_triples(
                x,
                num_entities=self.num_entities,
                num_relations=self.num_relations,
            ).shape[0]
        )


class KnowledgeGraphEnsemble:
    """An ensemble of independently fit :class:`KnowledgeGraphDistribution` models, for epistemic
    (model) uncertainty over completions.

    The members share the entity and relation index spaces but are fit from different random seeds, so
    where the data pins the answer down they agree and where it does not they disagree.  The mean tail
    posterior averages ``p(t | h, r)`` across members; the epistemic uncertainty is the mutual
    information (BALD) ``H(mean) - mean_m H(member_m)`` -- the part of the predictive entropy that comes
    from disagreement among members rather than from genuine ambiguity.
    """

    def __init__(self, members: list[KnowledgeGraphDistribution]) -> None:
        if len(members) < 2:
            raise ValueError("a KnowledgeGraphEnsemble needs at least two members.")
        self.members = list(members)

    def _tail_probs(self, h: int, r: int) -> np.ndarray:
        return np.array([np.exp(m.tail_log_posterior(int(h), int(r))) for m in self.members])

    def mean_tail_posterior(self, h: int, r: int) -> np.ndarray:
        """The ensemble-averaged ``p(t | h, r)`` over all tail candidates."""
        return self._tail_probs(h, r).mean(axis=0)

    def epistemic_tail_uncertainty(self, h: int, r: int) -> float:
        """Mutual-information (BALD) epistemic uncertainty of the tail completion (nats); 0 if members agree.

        Thin wrapper over the general :func:`mixle.inference.uncertainty.decompose_entropy` -- the
        tail posteriors ``p(t | h, r)`` per member are exactly the categorical predictives it splits.
        """
        from mixle.inference.uncertainty import decompose_entropy

        return float(decompose_entropy(self._tail_probs(h, r)).epistemic)


def fit_knowledge_graph_ensemble(
    triples: Sequence[Sequence[int]],
    num_entities: int,
    num_relations: int,
    dim: int = 16,
    members: int = 5,
    bootstrap: bool = False,
    rng: Any = None,
    **estimator_kwargs: Any,
) -> KnowledgeGraphEnsemble:
    """Fit ``members`` knowledge-graph models and wrap them in an ensemble.

    Members differ by their random seed; with ``bootstrap=True`` each is also fit on a bootstrap
    resample of the triples (bagging), which spreads the members further apart where the data is thin
    and so sharpens the epistemic-uncertainty estimate.
    """
    from mixle.inference.estimation import optimize

    base = RandomState() if rng is None else rng
    triples = list(triples)
    mods = []
    for k in range(int(members)):
        data = triples
        if bootstrap:
            idx = base.randint(len(triples), size=len(triples))
            data = [triples[i] for i in idx]
        est = KnowledgeGraphEstimator(num_entities, num_relations, dim=dim, seed=1 + k, **estimator_kwargs)
        mods.append(optimize(data, est, max_its=1, rng=RandomState(base.randint(2**31)), print_iter=10**9))
    return KnowledgeGraphEnsemble(mods)


class KnowledgeGraphPattern:
    """A subgraph-pattern query over a fitted :class:`KnowledgeGraphDistribution`.

    A pattern is a list of triples whose slots are fixed integer ids or named variables (strings
    starting with ``'?'``); a variable may recur across edges (shared join), and a variable in the
    relation slot ranges over relations, otherwise over entities.  A *binding* assigns every variable a
    value. Edgewise conditional factors define an unnormalized binding
    potential; this class computes the finite binding-space partition function
    and ``log_density`` returns the normalized probability over bindings.

    ``enumerate`` returns the most plausible completed subgraphs, and ``enumerator`` yields them lazily
    in descending score (a best-first beam of width ``beam``), so the object also satisfies the
    structure-distribution interface (``log_density`` + ``enumerator``) and can be handed to
    :class:`~mixle.ppl.ConformalStructure` for a calibrated set of completed subgraphs.  A binding is
    represented as a tuple of values in the canonical (sorted) variable order; :meth:`binding` builds one
    from a dict and :meth:`triples` grounds it to edges.
    """

    def __init__(
        self, kg: "KnowledgeGraphDistribution", pattern: Any, candidates: Any = None, known: Any = None, beam: int = 64
    ) -> None:
        self.kg = kg
        self.edges = []
        for raw_edge in pattern:
            edge = tuple(raw_edge)
            if len(edge) != 3:
                raise ValueError("knowledge-graph pattern edges must contain exactly (head, relation, tail)")
            self.edges.append(edge)
        if not self.edges:
            raise ValueError("knowledge-graph pattern cannot be empty")
        kind: dict[str, str] = {}
        for edge in self.edges:
            for slot, val in enumerate(edge):
                if isinstance(val, str) and val.startswith("?"):
                    k = "relation" if slot == 1 else "entity"
                    if kind.get(val, k) != k:
                        raise ValueError(f"variable {val!r} is used as both an entity and a relation.")
                    kind[val] = k
                elif slot == 1:
                    _validated_id(
                        val,
                        size=kg.num_relations,
                        label="fixed relation id",
                    )
                else:
                    _validated_id(
                        val,
                        size=kg.num_entities,
                        label="fixed entity id",
                    )
        self.variables = sorted(kind)
        self.kind = kind
        cand = dict(candidates or {})
        unknown_candidates = set(cand) - set(self.variables)
        if unknown_candidates:
            raise ValueError(f"candidate domains provided for unknown variables: {sorted(unknown_candidates)}")
        self.domain = {v: self._validated_domain(v, cand.get(v)) for v in self.variables}
        self.known = (
            None
            if known is None
            else {
                tuple(row)
                for row in _validated_triples(
                    list(known),
                    num_entities=kg.num_entities,
                    num_relations=kg.num_relations,
                ).tolist()
            }
        )
        self.beam = _exact_integer(beam, label="pattern beam width")
        if self.beam <= 0:
            raise ValueError("pattern beam width must be strictly positive")
        binding_count = math.prod(len(self.domain[variable]) for variable in self.variables)
        if binding_count > 1_000_000:
            raise ValueError("knowledge-graph binding space exceeds the exact normalization budget")
        raw_scores = [
            self._raw_score(tuple(values))
            for values in itertools.product(*(self.domain[variable] for variable in self.variables))
        ]
        maximum = max(raw_scores)
        self._log_normalizer = maximum + math.log(sum(math.exp(score - maximum) for score in raw_scores))

    def _validated_domain(
        self,
        variable: str,
        supplied: Any,
    ) -> list[int]:
        size = self.kg.num_relations if self.kind[variable] == "relation" else self.kg.num_entities
        values = range(size) if supplied is None else supplied
        result = [
            _validated_id(
                value,
                size=size,
                label=f"candidate for {variable}",
            )
            for value in values
        ]
        if not result or len(set(result)) != len(result):
            raise ValueError("knowledge-graph candidate domains must be nonempty and contain unique ids")
        return result

    @staticmethod
    def _edge_vars(edge: tuple) -> set:
        return {s for s in edge if isinstance(s, str) and s.startswith("?")}

    def _ground_edge(self, edge: tuple, b: dict) -> tuple:
        grounded = tuple(b[s] if isinstance(s, str) and s.startswith("?") else s for s in edge)
        return _validated_single_triple(
            grounded,
            num_entities=self.kg.num_entities,
            num_relations=self.kg.num_relations,
        )

    def binding(self, assignment: dict) -> tuple:
        """Canonical binding tuple (sorted-variable order) from a ``{variable: value}`` dict."""
        if set(assignment) != set(self.variables):
            raise ValueError("binding assignment must contain exactly the pattern variables")
        binding = tuple(
            _validated_id(
                assignment[variable],
                size=(self.kg.num_relations if self.kind[variable] == "relation" else self.kg.num_entities),
                label=f"binding for {variable}",
            )
            for variable in self.variables
        )
        if any(value not in self.domain[variable] for variable, value in zip(self.variables, binding)):
            raise ValueError("binding lies outside the candidate domain")
        return binding

    def triples(self, binding: tuple) -> list[tuple]:
        """Ground a binding tuple to the list of completed ``(h, r, t)`` edges."""
        if not isinstance(binding, tuple) or len(binding) != len(self.variables):
            raise ValueError("binding must be a tuple aligned with pattern variables")
        b = dict(zip(self.variables, binding))
        return [self._ground_edge(e, b) for e in self.edges]

    def _edge_logprob(self, h: int, r: int, t: int) -> float:
        return float(self.kg.tail_log_posterior(h, r)[t])

    def _raw_score(self, binding: tuple) -> float:
        return float(sum(self._edge_logprob(*edge) for edge in self.triples(binding)))

    def log_density(self, binding: tuple) -> float:
        """Normalized log-probability of a complete binding."""
        checked = self.binding(dict(zip(self.variables, binding)))
        return self._raw_score(checked) - self._log_normalizer

    def enumerator(self):
        """Yield ``(binding, joint_log_prob)`` over completed subgraphs in descending score (beam-limited)."""
        beam: list[tuple[dict, float]] = [({}, 0.0)]
        bound: set = set()
        for v in self.variables:
            bound.add(v)
            ready = [e for e in self.edges if self._edge_vars(e) <= bound and v in self._edge_vars(e)]
            nxt: list[tuple[dict, float]] = []
            for b, sc in beam:
                for val in self.domain[v]:
                    nb = dict(b)
                    nb[v] = val
                    inc = sum(self._edge_logprob(*self._ground_edge(e, nb)) for e in ready)
                    nxt.append((nb, sc + inc))
            nxt.sort(key=lambda u: -u[1])
            beam = nxt[: self.beam]
        fixed = sum(self._edge_logprob(*self._ground_edge(e, {})) for e in self.edges if not self._edge_vars(e))
        results = [
            (
                tuple(b[v] for v in self.variables),
                sc + fixed - self._log_normalizer,
            )
            for b, sc in beam
        ]
        if self.known is not None:  # keep only groundings that add at least one new edge
            results = [
                (bt, sc)
                for bt, sc in results
                if any(self._ground_edge(e, dict(zip(self.variables, bt))) not in self.known for e in self.edges)
            ]
        results.sort(key=lambda u: -u[1])
        yield from results

    def enumerate(self, top_n: int | None = 10) -> list[tuple[dict, list[tuple], float]]:
        """Top completed subgraphs as ``[({variable: value}, [edges], joint_log_prob), ...]``."""
        out = []
        for binding, score in self.enumerator():
            out.append((dict(zip(self.variables, binding)), self.triples(binding), score))
            if top_n is not None and len(out) >= top_n:
                break
        return out
