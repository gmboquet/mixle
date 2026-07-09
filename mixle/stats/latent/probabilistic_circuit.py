"""Probabilistic circuits (sum-product networks) -- a tractable deep model that scores in integer log-space.

A probabilistic circuit is a DAG of **sum** nodes (mixtures, with weights), **product** nodes (independent
factorizations over disjoint variable scopes), and **leaf** distributions over a scope. When it is
*decomposable* (a product's children have pairwise-disjoint scopes) and *smooth* (a sum's children share
one scope) the density is exact and inference is **linear in the circuit size** -- the appeal over an
intractable deep net. Every node is a sum or a product of probabilities, so the whole forward pass runs in
mixle's logarithmic number system: products become integer ADDs, sums become integer ``logsumexp`` (the
compiled max+LUT kernel), and leaf log-densities are quantized -- a transcendental-free deep forward pass.

This is the model class where the LNS is a *complete* fit (not just the normalizer). v1 takes a
user-supplied structure (build it with :func:`leaf` / :func:`prod` / :func:`summ`) and learns the per-leaf
parameters and per-sum log-weights by EM; structure learning is a later phase that emits the same DAG.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.vector import log_sum

# --- structure builder ----------------------------------------------------------------------------


class _Node:
    """A circuit node before flattening; identity-hashable so a child can be shared across parents."""

    __slots__ = ("kind", "children", "log_w", "dist", "scope")

    def __init__(self, kind: str, children=None, log_w=None, dist=None, scope=None) -> None:
        self.kind = kind
        self.children = children or []
        self.log_w = log_w
        self.dist = dist
        self.scope = scope


def leaf(scope: Any, dist: Any) -> _Node:
    """A leaf node: an existing mixle ``dist`` over the variable indices ``scope`` (an int or a tuple)."""
    sc = (int(scope),) if np.isscalar(scope) else tuple(int(v) for v in scope)
    return _Node("leaf", dist=dist, scope=sc)


def prod(children: list[_Node]) -> _Node:
    """A product node over children with PAIRWISE-DISJOINT scopes (the decomposability requirement)."""
    return _Node("product", children=list(children))


def summ(children: list[_Node], w: Any = None) -> _Node:
    """A sum (mixture) node over children that share ONE scope (smoothness); ``w`` are mixing weights."""
    return _Node("sum", children=list(children), log_w=w)


def _flatten(root: _Node) -> tuple[list[tuple], dict[int, Any], dict[int, tuple]]:
    """DFS the DAG into a topologically ordered node list (children before parents) + a leaf side table."""
    order: list[_Node] = []
    index: dict[int, int] = {}
    leaf_dists: dict[int, Any] = {}
    leaf_scope: dict[int, tuple] = {}

    def visit(node: _Node) -> int:
        if id(node) in index:
            return index[id(node)]
        for c in node.children:
            visit(c)
        i = len(order)
        index[id(node)] = i
        order.append(node)
        return i

    visit(root)
    nodes: list[tuple] = []
    for node in order:
        if node.kind == "leaf":
            lid = len(leaf_dists)
            leaf_dists[lid] = node.dist
            leaf_scope[lid] = node.scope
            nodes.append(("leaf", lid))
        elif node.kind == "product":
            nodes.append(("product", [index[id(c)] for c in node.children]))
        else:  # sum
            ch = [index[id(c)] for c in node.children]
            k = len(ch)
            w = np.full(k, 1.0 / k) if node.log_w is None else np.asarray(node.log_w, dtype=np.float64)
            w = w / w.sum()
            nodes.append(("sum", ch, list(np.log(w))))
    return nodes, leaf_dists, leaf_scope


# --- distribution ---------------------------------------------------------------------------------


class ProbabilisticCircuitDistribution(SequenceEncodableProbabilityDistribution):
    """A sum-product network density; build with :func:`leaf`/:func:`prod`/:func:`summ` then pass the root."""

    def __init__(self, root: _Node, num_vars: int, lns_step: float | None = None) -> None:
        """``root`` is the built DAG, ``num_vars`` the observation length; ``lns_step`` (e.g. 0.01) scores in
        the integer log number system at that precision instead of float64."""
        nodes, leaf_dists, leaf_scope = root if isinstance(root, tuple) else _flatten(root)
        self.nodes = nodes
        self.leaf_dists = leaf_dists
        self.leaf_scope = leaf_scope
        self.num_vars = int(num_vars)
        self.lns_step = lns_step
        self.scopes = self._validate_scopes()

    def _validate_scopes(self) -> list[frozenset]:
        """Compute every node scope and ENFORCE decomposability (disjoint products) + smoothness (equal sums)."""
        scopes: list[frozenset] = [frozenset()] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                scopes[i] = frozenset(self.leaf_scope[node[1]])
            elif node[0] == "product":
                acc: frozenset = frozenset()
                for c in node[1]:
                    if acc & scopes[c]:
                        raise ValueError("product node %d violates decomposability: child scopes overlap" % i)
                    acc = acc | scopes[c]
                scopes[i] = acc
            else:  # sum
                first = scopes[node[1][0]]
                for c in node[1][1:]:
                    if scopes[c] != first:
                        raise ValueError("sum node %d violates smoothness: child scopes differ" % i)
                scopes[i] = first
        if scopes[-1] != frozenset(range(self.num_vars)):
            raise ValueError("root scope %s must cover all %d variables" % (set(scopes[-1]), self.num_vars))
        return scopes

    def _project(self, x: Any, leaf_id: int) -> Any:
        sc = self.leaf_scope[leaf_id]
        return x[sc[0]] if len(sc) == 1 else tuple(x[v] for v in sc)

    def log_density(self, x: Any) -> float:
        """Return the log-density of one full observation by an upward circuit pass."""
        vals: list[float] = [0.0] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                vals[i] = float(self.leaf_dists[node[1]].log_density(self._project(x, node[1])))
            elif node[0] == "product":
                vals[i] = float(sum(vals[c] for c in node[1]))
            else:  # sum
                vals[i] = float(log_sum(np.array([vals[c] + lw for c, lw in zip(node[1], node[2])])))
        return vals[-1]

    def _node_values(self, enc: dict[int, Any]) -> list[np.ndarray]:
        """Per-node ``(n,)`` log-value vectors -- one cached pass over the DAG (linear in circuit size)."""
        vals: list[Any] = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                vals[i] = np.asarray(self.leaf_dists[node[1]].seq_log_density(enc[node[1]]), dtype=np.float64)
            elif node[0] == "product":
                acc = vals[node[1][0]].copy()
                for c in node[1][1:]:
                    acc = acc + vals[c]
                vals[i] = acc
            else:  # sum -- stable row logsumexp of weighted children (lifts the mixture masking)
                stack = np.stack([vals[c] + lw for c, lw in zip(node[1], node[2])], axis=0)
                m = stack.max(axis=0)
                finite = m > -np.inf
                out = np.full(stack.shape[1], -np.inf)
                out[finite] = m[finite] + np.log(np.exp(stack[:, finite] - m[finite]).sum(axis=0))
                vals[i] = out
        return vals

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return vectorized log-densities for encoded observations."""
        if self.lns_step is not None:
            return self._seq_log_density_lns(x)
        return self._node_values(x)[-1]

    def _seq_log_density_lns(self, enc: dict[int, Any]) -> np.ndarray:
        """Score the whole forward pass in the integer log number system (products=add, sums=logsumexp)."""
        from mixle.engines.lns import LogNumberSystem

        lns = LogNumberSystem(step=self.lns_step)
        vals: list[Any] = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                vals[i] = lns.quantize(np.asarray(self.leaf_dists[node[1]].seq_log_density(enc[node[1]])))
            elif node[0] == "product":
                acc = vals[node[1][0]].copy()
                for c in node[1][1:]:
                    acc = acc + vals[c]
                vals[i] = acc
            else:  # sum
                wk = lns.quantize(np.asarray(node[2]))
                stack = np.stack([vals[c] + wk[j] for j, c in enumerate(node[1])], axis=0)
                vals[i] = lns.logsumexp(stack, axis=0)
        return lns.dequantize(vals[-1])

    def dist_to_encoder(self) -> ProbabilisticCircuitEncoder:
        """Return the encoder that projects observations into each leaf scope."""
        return ProbabilisticCircuitEncoder(self.leaf_dists, self.leaf_scope)

    def sampler(self, seed: int | None = None) -> ProbabilisticCircuitSampler:
        """Return an ancestral sampler for this fixed circuit."""
        return ProbabilisticCircuitSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> Any:
        """Return an EM estimator for this fixed circuit structure."""
        return ProbabilisticCircuitEstimator(self, pseudo_count=pseudo_count)

    def with_params(self, new_nodes: list[tuple], new_leaf_dists: dict[int, Any]) -> ProbabilisticCircuitDistribution:
        """A new circuit with the same structure but re-estimated sum-weights / leaf parameters (M-step output)."""
        pc = ProbabilisticCircuitDistribution.__new__(ProbabilisticCircuitDistribution)
        pc.nodes = new_nodes
        pc.leaf_dists = new_leaf_dists
        pc.leaf_scope = self.leaf_scope
        pc.num_vars = self.num_vars
        pc.lns_step = self.lns_step
        pc.scopes = self.scopes
        return pc


class ProbabilisticCircuitEncoder(DataSequenceEncoder):
    """Encode each leaf's projected columns once with the leaf's own encoder (shared across EM iterations)."""

    def __init__(self, leaf_dists: dict[int, Any], leaf_scope: dict[int, tuple]) -> None:
        self.leaf_dists = leaf_dists
        self.leaf_scope = leaf_scope

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProbabilisticCircuitEncoder) and other.leaf_scope == self.leaf_scope

    def seq_encode(self, x: Any) -> dict[int, Any]:
        """Encode a batch for every leaf distribution using its projected scope."""
        enc: dict[int, Any] = {}
        for lid, sc in self.leaf_scope.items():
            if len(sc) == 1:
                col = [row[sc[0]] for row in x]
            else:
                col = [tuple(row[v] for v in sc) for row in x]
            enc[lid] = self.leaf_dists[lid].dist_to_encoder().seq_encode(col)
        return enc


class ProbabilisticCircuitSampler(DistributionSampler):
    """Ancestral top-down sampling: a sum draws one child by its weights, a product recurses into all."""

    def __init__(self, dist: ProbabilisticCircuitDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.leaf_samplers = {lid: d.sampler(self.rng.randint(0, 2**31 - 1)) for lid, d in dist.leaf_dists.items()}

    def _sample_one(self) -> list:
        out: list = [None] * self.dist.num_vars

        def descend(i: int) -> None:
            node = self.dist.nodes[i]
            if node[0] == "leaf":
                sc = self.dist.leaf_scope[node[1]]
                v = self.leaf_samplers[node[1]].sample()
                if len(sc) == 1:
                    out[sc[0]] = v
                else:
                    for j, var in enumerate(sc):
                        out[var] = v[j]
            elif node[0] == "product":
                for c in node[1]:
                    descend(c)
            else:  # sum
                w = np.exp(np.asarray(node[2]))
                descend(node[1][int(self.rng.choice(len(node[1]), p=w / w.sum()))])

        descend(len(self.dist.nodes) - 1)
        return out

    def sample(self, size: int | None = None) -> Any:
        """Draw one observation or ``size`` iid observations from the circuit."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


# --- EM estimation: circuit-flow soft-count E-step, weight + leaf M-step --------------------------

from mixle.stats.compute.pdist import (  # noqa: E402
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class ProbabilisticCircuitAccumulator(SequenceEncodableStatisticAccumulator):
    """E-step sufficient statistics: per-sum-node expected child counts + per-leaf weighted statistics.

    The E-step is the circuit FLOW (Peharz et al. EM-for-SPNs): an upward forward gives each node's
    log-value, a downward pass gives each node's log-context ``lc`` = derivative of the root log-density
    w.r.t. that node (the posterior the node is active). A sum node's per-child responsibility is then
    ``exp(lc[sum] + value[child] + log_w - value[sum])`` (its expected count), and a leaf's responsibility
    is ``exp(lc[leaf])`` (the weight for its sufficient statistic).
    """

    def __init__(self, nodes: list[tuple], leaf_scope: dict[int, tuple], leaf_estimators: dict[int, Any]) -> None:
        self.nodes = nodes
        self.leaf_scope = leaf_scope
        self.leaf_estimators = leaf_estimators
        self.sum_counts = {i: np.zeros(len(node[1])) for i, node in enumerate(nodes) if node[0] == "sum"}
        self.leaf_accs = {lid: e.accumulator_factory().make() for lid, e in leaf_estimators.items()}

    def seq_update(self, enc: dict[int, Any], weights: Any, estimate: ProbabilisticCircuitDistribution) -> None:
        """Update circuit-flow responsibilities and leaf sufficient statistics."""
        weights = np.asarray(weights, dtype=np.float64)
        n = weights.shape[0]
        node_vals = estimate._node_values(enc)
        lc = [np.full(n, -np.inf) for _ in self.nodes]
        lc[-1] = np.zeros(n)  # root context = 1
        for i in range(len(self.nodes) - 1, -1, -1):
            node = self.nodes[i]
            if node[0] == "leaf":
                continue
            lci = lc[i]
            if node[0] == "product":
                for c in node[1]:
                    lc[c] = np.logaddexp(lc[c], lci)
            else:  # sum
                vi = node_vals[i]
                for j, c in enumerate(node[1]):
                    edge_log = lci + (node_vals[c] + node[2][j] - vi)  # log responsibility through edge (i->c)
                    resp = np.where(np.isfinite(edge_log), np.exp(edge_log), 0.0)
                    self.sum_counts[i][j] += float(np.sum(weights * resp))
                    lc[c] = np.logaddexp(lc[c], edge_log)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                resp = np.where(np.isfinite(lc[i]), np.exp(lc[i]), 0.0)
                self.leaf_accs[node[1]].seq_update(enc[node[1]], weights * resp, estimate.leaf_dists[node[1]])

    def update(self, x: Any, weight: float, estimate: ProbabilisticCircuitDistribution) -> None:
        """Update from one weighted observation."""
        enc = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc, np.array([weight], dtype=np.float64), estimate)

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize sum counts and leaf statistics from one weighted observation."""
        for i, cnt in self.sum_counts.items():
            self.sum_counts[i] = cnt + float(weight) * rng.dirichlet(np.ones(len(cnt)))
        for lid, acc in self.leaf_accs.items():
            sc = self.leaf_scope[lid]
            acc.initialize(x[sc[0]] if len(sc) == 1 else tuple(x[v] for v in sc), weight, rng)

    def seq_initialize(self, enc: dict[int, Any], weights: Any, rng: RandomState) -> None:
        """Initialize sum counts and leaf statistics from encoded observations."""
        weights = np.asarray(weights, dtype=np.float64)
        for i, cnt in self.sum_counts.items():
            r = rng.dirichlet(np.ones(len(cnt)))  # random initial responsibilities break symmetry
            self.sum_counts[i] = cnt + float(np.sum(weights)) * r
        for lid, acc in self.leaf_accs.items():
            acc.seq_initialize(enc[lid], weights, rng)

    def combine(self, suff_stat: Any) -> ProbabilisticCircuitAccumulator:
        """Merge sum-node expected counts and leaf accumulator values."""
        sc, lv = suff_stat
        for i in self.sum_counts:
            self.sum_counts[i] += sc[i]
        for lid in self.leaf_accs:
            self.leaf_accs[lid].combine(lv[lid])
        return self

    def value(self) -> Any:
        """Return sum-node expected counts and leaf sufficient statistics."""
        return (
            {i: c.copy() for i, c in self.sum_counts.items()},
            {lid: a.value() for lid, a in self.leaf_accs.items()},
        )

    def from_value(self, x: Any) -> ProbabilisticCircuitAccumulator:
        """Restore sum-node and leaf sufficient statistics from ``value`` output."""
        sc, lv = x
        self.sum_counts = {i: np.asarray(c, dtype=np.float64) for i, c in sc.items()}
        for lid, v in lv.items():
            self.leaf_accs[lid].from_value(v)
        return self

    def scale(self, c: float) -> ProbabilisticCircuitAccumulator:
        """Scale sum-node and leaf sufficient statistics by a constant."""
        for i in self.sum_counts:
            self.sum_counts[i] *= c
        for lid in self.leaf_accs:
            self.leaf_accs[lid].scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed merges to the leaf accumulators."""
        for acc in self.leaf_accs.values():
            acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed replacements to the leaf accumulators."""
        for acc in self.leaf_accs.values():
            acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> ProbabilisticCircuitEncoder:
        """Return an encoder based on the current leaf accumulator estimates."""
        leaf_dists = {
            lid: self.leaf_estimators[lid].estimate(None, self.leaf_accs[lid].value()) for lid in self.leaf_accs
        }
        return ProbabilisticCircuitEncoder(leaf_dists, self.leaf_scope)


class ProbabilisticCircuitAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for fixed-structure probabilistic-circuit EM."""

    def __init__(self, nodes: list[tuple], leaf_scope: dict[int, tuple], leaf_estimators: dict[int, Any]) -> None:
        self.nodes = nodes
        self.leaf_scope = leaf_scope
        self.leaf_estimators = leaf_estimators

    def make(self) -> ProbabilisticCircuitAccumulator:
        """Create an empty probabilistic-circuit accumulator."""
        return ProbabilisticCircuitAccumulator(self.nodes, self.leaf_scope, self.leaf_estimators)


class ProbabilisticCircuitEstimator(ParameterEstimator):
    """Fits a fixed-structure circuit by EM: renormalize each sum node's weights to its expected child
    counts, and re-estimate each leaf from its responsibility-weighted sufficient statistic."""

    def __init__(self, dist: ProbabilisticCircuitDistribution, pseudo_count: float | None = None) -> None:
        self.dist = dist
        self.pseudo_count = 0.0 if pseudo_count is None else float(pseudo_count)
        self.leaf_estimators = {lid: d.estimator() for lid, d in dist.leaf_dists.items()}

    def accumulator_factory(self) -> ProbabilisticCircuitAccumulatorFactory:
        """Return a factory for circuit-flow sufficient-statistic accumulators."""
        return ProbabilisticCircuitAccumulatorFactory(self.dist.nodes, self.dist.leaf_scope, self.leaf_estimators)

    def estimate(self, nobs: float | None, suff_stat: Any) -> ProbabilisticCircuitDistribution:
        """Estimate sum-node weights and leaf distributions from accumulated circuit flows."""
        sum_counts, leaf_values = suff_stat
        new_nodes: list[tuple] = []
        for i, node in enumerate(self.dist.nodes):
            if node[0] == "sum":
                cnt = sum_counts[i] + self.pseudo_count
                total = float(cnt.sum())
                w = cnt / total if total > 0 else np.full(len(cnt), 1.0 / len(cnt))
                new_nodes.append(("sum", node[1], list(np.log(w))))
            else:
                new_nodes.append(node)
        new_leaf_dists = {lid: self.leaf_estimators[lid].estimate(None, leaf_values[lid]) for lid in leaf_values}
        return self.dist.with_params(new_nodes, new_leaf_dists)
