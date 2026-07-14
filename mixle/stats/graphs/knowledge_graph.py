"""Knowledge-graph embedding distributions for integer triple observations.

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

    def head_log_posterior(self, r: int, t: int) -> np.ndarray:
        """Length-``num_entities`` vector of ``log p(h | r, t)`` over all head candidates."""
        return _tail_log_posterior(self.entity, self.relation[r] * self.entity[t])

    def relation_log_posterior(self, h: int, t: int) -> np.ndarray:
        """Length-``num_relations`` vector of ``log p(r | h, t)`` over all relation candidates."""
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
            return self.tail_log_posterior(int(h), int(r))
        if h is None:
            return self.head_log_posterior(int(r), int(t))
        return self.relation_log_posterior(int(h), int(t))

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
        excl = set(int(e) for e in np.atleast_1d(np.asarray(list(exclude), dtype=int))) if len(exclude) else set()
        ranked = [(int(c), float(logp[c])) for c in order if int(c) not in excl]
        return ranked if top_n is None else ranked[:top_n]

    def recommend(self, known: Any, top_n: int = 10) -> list[tuple[int, int, int, float]]:
        """Recommend the most plausible missing tail facts for the ``(h, r)`` contexts in ``known``.

        ``known`` is a sequence of observed ``(h, r, t)`` triples; for each distinct ``(h, r)`` the
        already-present tails are excluded, the remaining tails are ranked by ``log p(t | h, r)``, and
        the global top ``top_n`` new facts are returned as ``[(h, r, t, log_prob), ...]``.
        """
        known = np.asarray(list(known), dtype=int).reshape(-1, 3)
        seen: dict[tuple[int, int], set] = {}
        for h, r, t in known:
            seen.setdefault((int(h), int(r)), set()).add(int(t))
        out: list[tuple[int, int, int, float]] = []
        for (h, r), tails in seen.items():
            for t, lp in self.rank(h=h, r=r, exclude=tails):
                out.append((h, r, t, lp))
        out.sort(key=lambda u: -u[3])
        return out[:top_n]

    def recommend_subgraph(self, node: int, known: Any, top_n: int = 5) -> list[tuple[int, int, int, float]]:
        """Recommend plausible new edges incident to ``node`` (both ``(node, r, ?)`` and ``(?, r, node)``).

        Excludes edges already in ``known`` and returns the top ``top_n`` by log-probability as
        ``[(h, r, t, log_prob), ...]``, the suggested missing subgraph around the node.
        """
        node = int(node)
        known_set = {(int(h), int(r), int(t)) for h, r, t in np.asarray(list(known), dtype=int).reshape(-1, 3)}
        cand: list[tuple[int, int, int, float]] = []
        for r in range(self.num_relations):
            for t, lp in self.rank(h=node, r=r):
                if (node, r, t) not in known_set:
                    cand.append((node, r, t, lp))
            for h, lp in self.rank(r=r, t=node):
                if (h, r, node) not in known_set:
                    cand.append((h, r, node, lp))
        cand.sort(key=lambda u: -u[3])
        return cand[:top_n]

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
        """Return ``log p(t | h, r)`` for one integer triple."""
        h, r, t = int(x[0]), int(x[1]), int(x[2])
        return float(self.tail_log_posterior(h, r)[t])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized tail log-probabilities for encoded triples."""
        x = np.asarray(x, dtype=int)
        out = np.empty(x.shape[0], dtype=float)
        for n in range(x.shape[0]):
            out[n] = self.tail_log_posterior(x[n, 0], x[n, 1])[x[n, 2]]
        return out

    def sampler(self, seed: int | None = None) -> "KnowledgeGraphSampler":
        """Return a sampler for observed triples."""
        return KnowledgeGraphSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "KnowledgeGraphEstimator":
        """Return a DistMult embedding estimator for this entity/relation shape."""
        return KnowledgeGraphEstimator(
            self.num_entities, self.num_relations, dim=self.dim, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "KnowledgeGraphDataEncoder":
        """Return the triple encoder used by vectorized methods."""
        return KnowledgeGraphDataEncoder()


class KnowledgeGraphSampler(DistributionSampler):
    """Draw triples: head and relation uniformly, tail from the conditional softmax ``p(t | h, r)``."""

    def __init__(self, dist: KnowledgeGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None) -> Any:
        """Draw one triple or ``size`` iid triples."""
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
        self.keys = keys
        self.triples: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        self.count = 0.0

    def update(self, x: Sequence[int], weight: float, estimate: KnowledgeGraphDistribution | None) -> None:
        """Store one weighted triple for embedding training."""
        self.seq_update(np.asarray([x], dtype=int), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Store one weighted triple during initialization."""
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Store encoded triples during initialization."""
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: KnowledgeGraphDistribution | None) -> None:
        """Store encoded triples and weights for embedding training."""
        self.triples.append(np.asarray(x, dtype=int))
        self.weights.append(np.asarray(weights, dtype=float))
        self.count += float(np.sum(weights))

    def _stacked(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.triples:
            return np.zeros((0, 3), dtype=int), np.zeros(0)
        return np.concatenate(self.triples, axis=0), np.concatenate(self.weights)

    def combine(self, suff_stat: tuple) -> "KnowledgeGraphAccumulator":
        """Merge stored triples and weights from another accumulator value."""
        count, triples, weights = suff_stat
        self.count += count
        if len(triples):
            self.triples.append(np.asarray(triples, dtype=int))
            self.weights.append(np.asarray(weights, dtype=float))
        return self

    def value(self) -> tuple:
        """Return total weight, stacked triples, and stacked weights."""
        triples, weights = self._stacked()
        return self.count, triples, weights

    def from_value(self, x: tuple) -> "KnowledgeGraphAccumulator":
        """Restore stored triples and weights from ``value`` output."""
        self.count = x[0]
        self.triples = [np.asarray(x[1], dtype=int)] if len(x[1]) else []
        self.weights = [np.asarray(x[2], dtype=float)] if len(x[1]) else []
        return self

    def acc_to_encoder(self) -> "KnowledgeGraphDataEncoder":
        """Return the encoder compatible with stored triples."""
        return KnowledgeGraphDataEncoder()


class KnowledgeGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for KnowledgeGraphAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> KnowledgeGraphAccumulator:
        """Create an empty knowledge-graph accumulator."""
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
        directions: tuple = ("tail", "head", "relation"),
        negatives: int | None = None,
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
        self.directions = tuple(directions)
        self.negatives = (
            None if negatives is None else int(negatives)
        )  # sampled-softmax negatives (scales to large KGs)
        self.seed = int(seed)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> KnowledgeGraphAccumulatorFactory:
        """Return a factory for stored-triple accumulators."""
        return KnowledgeGraphAccumulatorFactory(keys=self.keys)

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
            cand = np.concatenate([target[:, None], rng.randint(E.shape[0], size=(q.shape[0], k))], axis=1)
            cand_emb = E[cand]  # (m, 1+k, d); column 0 is the positive
            p = _softmax_rows(np.einsum("bkd,bd->bk", cand_emb, q))
            ebar = np.einsum("bk,bkd->bd", p, cand_emb)
            onehot = np.zeros_like(p)
            onehot[:, 0] = 1.0
            resid = (onehot - p) * w
            np.add.at(ge, cand.reshape(-1), (resid[:, :, None] * q[:, None, :]).reshape(-1, q.shape[1]))
        np.add.at(ge, other, w * R[r] * (E[target] - ebar))
        np.add.at(gr, r, w * E[other] * (E[target] - ebar))

    def estimate(self, nobs: float | None, suff_stat: tuple) -> KnowledgeGraphDistribution:
        """Fit DistMult embeddings from stored triples and weights."""
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
        return KnowledgeGraphDistribution(E, R, name=self.name, keys=self.keys)


class KnowledgeGraphDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(h, r, t)`` triples into an ``(N, 3)`` integer array."""

    def __str__(self) -> str:
        return "KnowledgeGraphDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KnowledgeGraphDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode triples as an ``(N, 3)`` integer array."""
        rv = np.asarray([list(row) for row in x], dtype=int)
        if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] == 0:
            raise ValueError("KnowledgeGraphDistribution requires a non-empty sequence of (h, r, t) triples.")
        return rv


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
    value; its joint score is the sum over edges of ``log p(tail | head, relation)``.

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
        self.edges = [tuple(e) for e in pattern]
        kind: dict[str, str] = {}
        for edge in self.edges:
            for slot, val in enumerate(edge):
                if isinstance(val, str) and val.startswith("?"):
                    k = "relation" if slot == 1 else "entity"
                    if kind.get(val, k) != k:
                        raise ValueError(f"variable {val!r} is used as both an entity and a relation.")
                    kind[val] = k
        self.variables = sorted(kind)
        self.kind = kind
        cand = dict(candidates or {})
        self.domain = {
            v: list(cand[v])
            if v in cand
            else list(range(kg.num_relations if kind[v] == "relation" else kg.num_entities))
            for v in self.variables
        }
        self.known = None if known is None else {tuple(int(x) for x in e) for e in known}
        self.beam = int(beam)

    @staticmethod
    def _edge_vars(edge: tuple) -> set:
        return {s for s in edge if isinstance(s, str) and s.startswith("?")}

    def _ground_edge(self, edge: tuple, b: dict) -> tuple:
        return tuple(int(b[s]) if isinstance(s, str) and s.startswith("?") else int(s) for s in edge)

    def binding(self, assignment: dict) -> tuple:
        """Canonical binding tuple (sorted-variable order) from a ``{variable: value}`` dict."""
        return tuple(int(assignment[v]) for v in self.variables)

    def triples(self, binding: tuple) -> list[tuple]:
        """Ground a binding tuple to the list of completed ``(h, r, t)`` edges."""
        b = dict(zip(self.variables, binding))
        return [self._ground_edge(e, b) for e in self.edges]

    def _edge_logprob(self, h: int, r: int, t: int) -> float:
        return float(self.kg.tail_log_posterior(h, r)[t])

    def log_density(self, binding: tuple) -> float:
        """Joint log-probability of a complete binding (sum of edge tail-conditional log-probs)."""
        return float(sum(self._edge_logprob(*e) for e in self.triples(binding)))

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
        results = [(tuple(b[v] for v in self.variables), sc + fixed) for b, sc in beam]
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
