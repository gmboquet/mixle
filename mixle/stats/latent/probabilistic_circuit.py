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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.mixture_evidence import validated_probability_vector
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_observation_weight,
)
from mixle.utils.vector import log_sum, require_possible_log_evidence


@dataclass(frozen=True)
class ProbabilisticCircuitStatistics(Sequence[Any]):
    """Mass-aware circuit statistics with backward two-slot iteration."""

    sum_counts: dict[int, np.ndarray]
    leaf_statistics: dict[int, Any]
    leaf_counts: dict[int, float]
    observation_mass: float
    schema_version: int = 1

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        return (self.sum_counts, self.leaf_statistics)[index]


def _unpack_circuit_statistics(values, leaf_ids):
    if isinstance(values, ProbabilisticCircuitStatistics):
        if set(values.leaf_counts) != set(leaf_ids):
            raise ValueError("probabilistic circuit leaf-count statistics do not match circuit leaves")
        leaf_counts = {
            leaf_id: validated_observation_weight(
                values.leaf_counts[leaf_id],
                f"probabilistic circuit leaf {leaf_id} mass",
            )
            for leaf_id in leaf_ids
        }
        observation_mass = validated_observation_weight(
            values.observation_mass,
            "probabilistic circuit observation mass",
        )
        return values.sum_counts, values.leaf_statistics, leaf_counts, observation_mass
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError("legacy probabilistic circuit statistics must contain sum and leaf values")
    return values[0], values[1], None, None


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


def _exact_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def _finite_nonnegative_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _finite_nonnegative_weights(values: Any, size: int, label: str) -> np.ndarray:
    try:
        weights = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric vector.") from exc
    if weights.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},).")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{label} must contain finite non-negative values.")
    return weights.copy()


def _exact_scope(scope: Any, label: str) -> tuple[int, ...]:
    if isinstance(scope, (bool, np.bool_)):
        raise TypeError(f"{label} must contain exact integer variable IDs.")
    try:
        raw = [scope] if isinstance(scope, (int, np.integer)) else list(scope)
    except TypeError as exc:
        raise TypeError(f"{label} must contain exact integer variable IDs.") from exc
    if not raw:
        raise ValueError(f"{label} must not be empty.")
    normalized: list[int] = []
    for index, variable in enumerate(raw):
        if isinstance(variable, (bool, np.bool_)) or not isinstance(variable, (int, np.integer)):
            raise TypeError(f"{label}[{index}] must be an integer.")
        normalized.append(int(variable))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} variable IDs must be unique.")
    return tuple(normalized)


def _exact_child_ids(values: Any, node_index: int) -> tuple[int, ...]:
    try:
        raw = list(values)
    except TypeError as exc:
        raise TypeError(f"circuit node {node_index} children must be an iterable of node IDs.") from exc
    if not raw:
        raise ValueError(f"circuit node {node_index} must have at least one child.")
    children: list[int] = []
    for position, child in enumerate(raw):
        if isinstance(child, (bool, np.bool_)) or not isinstance(child, (int, np.integer)):
            raise TypeError(f"circuit node {node_index} child {position} must be an integer.")
        child_id = int(child)
        if child_id < 0 or child_id >= node_index:
            raise ValueError(f"circuit node {node_index} child {position}={child_id} must reference an earlier node.")
        children.append(child_id)
    if len(set(children)) != len(children):
        raise ValueError(f"circuit node {node_index} child IDs must be unique.")
    return tuple(children)


def leaf(scope: Any, dist: Any) -> _Node:
    """A leaf node: an existing mixle ``dist`` over the variable indices ``scope`` (an int or a tuple)."""
    if dist is None:
        raise TypeError("circuit leaf distribution must not be None.")
    return _Node("leaf", dist=dist, scope=_exact_scope(scope, "circuit leaf scope"))


def prod(children: list[_Node]) -> _Node:
    """A product node over children with PAIRWISE-DISJOINT scopes (the decomposability requirement)."""
    try:
        child_list = list(children)
    except TypeError as exc:
        raise TypeError("product children must be an iterable of circuit nodes.") from exc
    if not child_list:
        raise ValueError("product node must have at least one child.")
    return _Node("product", children=child_list)


def summ(children: list[_Node], w: Any = None) -> _Node:
    """A sum (mixture) node over children that share ONE scope (smoothness); ``w`` are mixing weights."""
    try:
        child_list = list(children)
    except TypeError as exc:
        raise TypeError("sum children must be an iterable of circuit nodes.") from exc
    if not child_list:
        raise ValueError("sum node must have at least one child.")
    if w is not None:
        validated_probability_vector(w, "sum node weights", size=len(child_list))
    return _Node("sum", children=child_list, log_w=w)


def _flatten(root: _Node) -> tuple[list[tuple], dict[int, Any], dict[int, tuple]]:
    """DFS the DAG into a topologically ordered node list (children before parents) + a leaf side table."""
    if not isinstance(root, _Node):
        raise TypeError("circuit root must be a circuit node or a validated flattened circuit tuple.")
    order: list[_Node] = []
    index: dict[int, int] = {}
    visiting: set[int] = set()
    leaf_dists: dict[int, Any] = {}
    leaf_scope: dict[int, tuple] = {}

    def visit(node: _Node) -> int:
        if not isinstance(node, _Node):
            raise TypeError("circuit children must all be circuit nodes.")
        node_identity = id(node)
        if node_identity in index:
            return index[node_identity]
        if node_identity in visiting:
            raise ValueError("circuit graph must be acyclic.")
        if node.kind not in {"leaf", "product", "sum"}:
            raise ValueError(f"unknown circuit node kind {node.kind!r}.")
        visiting.add(node_identity)
        if node.kind == "leaf":
            if node.children:
                raise ValueError("circuit leaf nodes cannot have children.")
            if node.dist is None:
                raise TypeError("circuit leaf distribution must not be None.")
            node.scope = _exact_scope(node.scope, "circuit leaf scope")
        else:
            if not node.children:
                raise ValueError(f"circuit {node.kind} node must have at least one child.")
            if len({id(child) for child in node.children}) != len(node.children):
                raise ValueError(f"circuit {node.kind} node children must be unique.")
            for child in node.children:
                visit(child)
        i = len(order)
        visiting.remove(node_identity)
        index[node_identity] = i
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
            nodes.append(("product", tuple(index[id(c)] for c in node.children)))
        else:
            ch = tuple(index[id(c)] for c in node.children)
            k = len(ch)
            w = (
                np.full(k, 1.0 / k)
                if node.log_w is None
                else validated_probability_vector(node.log_w, "sum node weights", size=k)
            )
            with np.errstate(divide="ignore"):
                nodes.append(("sum", ch, tuple(np.log(w))))
    return nodes, leaf_dists, leaf_scope


def _canonical_circuit(
    flattened: Any,
    num_vars: int,
) -> tuple[list[tuple], dict[int, Any], dict[int, tuple[int, ...]]]:
    """Validate and own a flattened topological circuit representation."""
    if not isinstance(flattened, tuple) or len(flattened) != 3:
        raise TypeError("flattened circuit must be a (nodes, leaf_dists, leaf_scope) tuple.")
    raw_nodes, raw_leaf_dists, raw_leaf_scope = flattened
    try:
        nodes_input = list(raw_nodes)
    except TypeError as exc:
        raise TypeError("flattened circuit nodes must be an iterable.") from exc
    if not nodes_input:
        raise ValueError("probabilistic circuit must contain at least one node.")
    if not isinstance(raw_leaf_dists, dict) or not isinstance(raw_leaf_scope, dict):
        raise TypeError("flattened circuit leaf distributions and scopes must be dictionaries.")

    leaf_ids = set(raw_leaf_dists)
    if leaf_ids != set(raw_leaf_scope):
        raise ValueError("flattened circuit leaf distribution and scope IDs must match.")
    expected_leaf_ids = set(range(len(leaf_ids)))
    if (
        any(isinstance(leaf_id, (bool, np.bool_)) or not isinstance(leaf_id, (int, np.integer)) for leaf_id in leaf_ids)
        or {int(leaf_id) for leaf_id in leaf_ids} != expected_leaf_ids
    ):
        raise ValueError("flattened circuit leaf IDs must be the dense integer range 0..L-1.")
    leaf_dists = {int(leaf_id): raw_leaf_dists[leaf_id] for leaf_id in sorted(leaf_ids)}
    if any(distribution is None for distribution in leaf_dists.values()):
        raise TypeError("flattened circuit leaf distributions must not be None.")
    leaf_scope = {
        int(leaf_id): _exact_scope(raw_leaf_scope[leaf_id], f"circuit leaf {leaf_id} scope")
        for leaf_id in sorted(leaf_ids)
    }
    for leaf_id, scope in leaf_scope.items():
        invalid = [variable for variable in scope if variable < 0 or variable >= num_vars]
        if invalid:
            raise ValueError(f"circuit leaf {leaf_id} scope contains out-of-range variables {invalid}.")

    nodes: list[tuple] = []
    referenced_leaf_ids: list[int] = []
    for node_index, raw_node in enumerate(nodes_input):
        if not isinstance(raw_node, (tuple, list)) or not raw_node:
            raise TypeError(f"circuit node {node_index} must be a non-empty tuple.")
        kind = raw_node[0]
        if kind == "leaf":
            if len(raw_node) != 2:
                raise ValueError(f"circuit leaf node {node_index} must have arity two.")
            leaf_id = raw_node[1]
            if isinstance(leaf_id, (bool, np.bool_)) or not isinstance(leaf_id, (int, np.integer)):
                raise TypeError(f"circuit leaf node {node_index} ID must be an integer.")
            leaf_id = int(leaf_id)
            if leaf_id not in leaf_dists:
                raise ValueError(f"circuit leaf node {node_index} references unknown leaf ID {leaf_id}.")
            referenced_leaf_ids.append(leaf_id)
            nodes.append(("leaf", leaf_id))
        elif kind == "product":
            if len(raw_node) != 2:
                raise ValueError(f"circuit product node {node_index} must have arity two.")
            nodes.append(("product", _exact_child_ids(raw_node[1], node_index)))
        elif kind == "sum":
            if len(raw_node) != 3:
                raise ValueError(f"circuit sum node {node_index} must have arity three.")
            children = _exact_child_ids(raw_node[1], node_index)
            try:
                log_weights = np.asarray(raw_node[2], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"circuit sum node {node_index} log weights must be numeric.") from exc
            if log_weights.shape != (len(children),):
                raise ValueError(f"circuit sum node {node_index} log weights must have shape ({len(children)},).")
            if np.any(np.isnan(log_weights) | np.isposinf(log_weights)):
                raise ValueError(f"circuit sum node {node_index} log weights must be finite or -inf.")
            probabilities = np.exp(log_weights)
            total = float(probabilities.sum())
            if not np.isclose(total, 1.0, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError(f"circuit sum node {node_index} weights must form a probability simplex.")
            nodes.append(("sum", children, tuple(float(value) for value in log_weights)))
        else:
            raise ValueError(f"unknown circuit node kind {kind!r} at node {node_index}.")

    if sorted(referenced_leaf_ids) != list(range(len(leaf_dists))):
        raise ValueError("each flattened circuit leaf ID must be referenced by exactly one leaf node.")
    reachable: set[int] = set()
    pending = [len(nodes) - 1]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        node = nodes[node_id]
        if node[0] != "leaf":
            pending.extend(node[1])
    if reachable != set(range(len(nodes))):
        disconnected = sorted(set(range(len(nodes))) - reachable)
        raise ValueError(f"circuit contains nodes unreachable from the root: {disconnected}.")
    return nodes, leaf_dists, leaf_scope


# --- distribution ---------------------------------------------------------------------------------


class ProbabilisticCircuitDistribution(SequenceEncodableProbabilityDistribution):
    """A sum-product network density; build with :func:`leaf`/:func:`prod`/:func:`summ` then pass the root."""

    def __init__(self, root: _Node, num_vars: int, lns_step: float | None = None) -> None:
        """``root`` is the built DAG, ``num_vars`` the observation length; ``lns_step`` (e.g. 0.01) scores in
        the integer log number system at that precision instead of float64."""
        self.num_vars = _exact_positive_integer(num_vars, "probabilistic circuit num_vars")
        flattened = root if isinstance(root, tuple) else _flatten(root)
        nodes, leaf_dists, leaf_scope = _canonical_circuit(flattened, self.num_vars)
        self.nodes = nodes
        self.leaf_dists = leaf_dists
        self.leaf_scope = leaf_scope
        if lns_step is None:
            self.lns_step = None
        else:
            if isinstance(lns_step, (bool, np.bool_)) or not isinstance(
                lns_step,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError("probabilistic circuit lns_step must be a real number.")
            self.lns_step = float(lns_step)
            if not np.isfinite(self.lns_step) or self.lns_step <= 0.0:
                raise ValueError("probabilistic circuit lns_step must be finite and positive.")
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
        if not isinstance(enc, dict) or set(enc) != set(self.leaf_dists):
            raise ValueError("probabilistic circuit encoding must contain exactly one payload per leaf.")
        vals: list[Any] = [None] * len(self.nodes)
        row_count = None
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                values = np.asarray(self.leaf_dists[node[1]].seq_log_density(enc[node[1]]), dtype=np.float64)
                if values.ndim != 1:
                    raise ValueError(
                        f"probabilistic circuit leaf {node[1]} must return a one-dimensional score vector."
                    )
                if np.any(np.isnan(values) | np.isposinf(values)):
                    raise ValueError(f"probabilistic circuit leaf {node[1]} returned invalid log densities.")
                if row_count is None:
                    row_count = len(values)
                elif len(values) != row_count:
                    raise ValueError("probabilistic circuit leaves returned different row counts.")
                vals[i] = values
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

        if not isinstance(enc, dict) or set(enc) != set(self.leaf_dists):
            raise ValueError("probabilistic circuit encoding must contain exactly one payload per leaf.")
        lns = LogNumberSystem(step=self.lns_step)
        vals: list[Any] = [None] * len(self.nodes)
        row_count = None
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                values = np.asarray(self.leaf_dists[node[1]].seq_log_density(enc[node[1]]), dtype=np.float64)
                if values.ndim != 1 or np.any(np.isnan(values) | np.isposinf(values)):
                    raise ValueError(f"probabilistic circuit leaf {node[1]} returned invalid log densities.")
                if row_count is None:
                    row_count = len(values)
                elif len(values) != row_count:
                    raise ValueError("probabilistic circuit leaves returned different row counts.")
                vals[i] = lns.quantize(values)
            elif node[0] == "product":
                acc = vals[node[1][0]].copy()
                for c in node[1][1:]:
                    acc = lns.multiply(acc, vals[c])  # MXR-080-0138: safe LNS product (was raw `+`)
                vals[i] = acc
            else:  # sum
                wk = lns.quantize(np.asarray(node[2]))
                # each term is log(child_prob * weight) = child_code (+) weight_code -- an LNS product,
                # so it must go through the same overflow-safe multiply() (MXR-080-0138), not raw `+`:
                # a child that quantized to LOG_ZERO_CODE, or a component weighted exactly 0 (log(0) =
                # -inf -> LOG_ZERO_CODE), would otherwise overflow int64 like the product node above.
                stack = np.stack([lns.multiply(vals[c], wk[j]) for j, c in enumerate(node[1])], axis=0)
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
        return ProbabilisticCircuitDistribution(
            (new_nodes, new_leaf_dists, self.leaf_scope),
            self.num_vars,
            self.lns_step,
        )


class ProbabilisticCircuitEncoder(DataSequenceEncoder):
    """Encode each leaf's projected columns once with the leaf's own encoder (shared across EM iterations)."""

    def __init__(self, leaf_dists: dict[int, Any], leaf_scope: dict[int, tuple]) -> None:
        self.leaf_scope = {leaf_id: tuple(scope) for leaf_id, scope in leaf_scope.items()}
        self.leaf_encoders = {leaf_id: leaf_dists[leaf_id].dist_to_encoder() for leaf_id in sorted(leaf_dists)}
        self.leaf_semantics = {
            leaf_id: (
                type(leaf_dists[leaf_id]),
                None
                if not hasattr(leaf_dists[leaf_id], "pmap")
                else tuple(
                    sorted(
                        (
                            type(value).__module__,
                            type(value).__qualname__,
                            repr(value),
                        )
                        for value in leaf_dists[leaf_id].pmap
                    )
                ),
            )
            for leaf_id in sorted(leaf_dists)
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ProbabilisticCircuitEncoder)
            and other.leaf_scope == self.leaf_scope
            and other.leaf_semantics == self.leaf_semantics
            and tuple(other.leaf_encoders) == tuple(self.leaf_encoders)
            and all(other.leaf_encoders[leaf_id] == encoder for leaf_id, encoder in self.leaf_encoders.items())
        )

    def seq_encode(self, x: Any) -> dict[int, Any]:
        """Encode a batch for every leaf distribution using its projected scope."""
        rows = list(x)
        enc: dict[int, Any] = {}
        for lid, sc in self.leaf_scope.items():
            if len(sc) == 1:
                col = [row[sc[0]] for row in rows]
            else:
                col = [tuple(row[v] for v in sc) for row in rows]
            enc[lid] = self.leaf_encoders[lid].seq_encode(col)
        return enc

    def row_count(self, x: Any) -> int:
        """Return the common number of encoded circuit rows."""
        if not isinstance(x, dict) or set(x) != set(self.leaf_encoders):
            raise ValueError("probabilistic circuit encoding must contain exactly one payload per leaf.")
        counts = {leaf_id: self.leaf_encoders[leaf_id].row_count(payload) for leaf_id, payload in x.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"probabilistic circuit leaf encodings disagree on row count: {counts}.")
        return next(iter(counts.values()))


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

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
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
        self.leaf_counts = {lid: 0.0 for lid in leaf_estimators}
        self.observation_mass = 0.0

    def _validated_payload(self, values):
        sc, lv, leaf_counts, observation_mass = _unpack_circuit_statistics(values, self.leaf_accs)
        if not isinstance(sc, dict) or set(sc) != set(self.sum_counts):
            raise ValueError("probabilistic circuit sum statistics do not match circuit sum nodes")
        if not isinstance(lv, dict) or set(lv) != set(self.leaf_accs):
            raise ValueError("probabilistic circuit leaf statistics do not match circuit leaves")
        checked_sum_counts = {}
        for node_id, expected in self.sum_counts.items():
            try:
                counts = np.asarray(sc[node_id], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"probabilistic circuit sum node {node_id} counts must be numeric") from exc
            if counts.shape != expected.shape or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
                raise ValueError(
                    f"probabilistic circuit sum node {node_id} counts must be finite, non-negative, "
                    f"and have shape {expected.shape}"
                )
            checked_sum_counts[node_id] = counts.copy()
        return checked_sum_counts, lv, leaf_counts, observation_mass

    def seq_update(self, enc: dict[int, Any], weights: Any, estimate: ProbabilisticCircuitDistribution) -> None:
        """Update circuit-flow responsibilities and leaf sufficient statistics."""
        node_vals = estimate._node_values(enc)
        n = len(node_vals[-1])
        weights = _finite_nonnegative_weights(weights, n, "probabilistic circuit accumulator weights")
        active = weights > 0.0
        if not np.any(active):
            return
        require_possible_log_evidence(
            node_vals[-1][active],
            context="ProbabilisticCircuitAccumulator.seq_update",
        )
        lc = [np.full(n, -np.inf) for _ in self.nodes]
        lc[-1] = np.where(active, 0.0, -np.inf)  # root context = 1 for effective rows
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
                    with np.errstate(invalid="ignore"):
                        edge_log = lci + (node_vals[c] + node[2][j] - vi)  # log responsibility through edge (i->c)
                    edge_log = np.where(active, edge_log, -np.inf)
                    resp = np.where(np.isfinite(edge_log), np.exp(edge_log), 0.0)
                    self.sum_counts[i][j] += float(np.sum(weights * resp))
                    lc[c] = np.logaddexp(lc[c], edge_log)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                resp = np.where(np.isfinite(lc[i]), np.exp(lc[i]), 0.0)
                assigned = weights * resp
                self.leaf_accs[node[1]].seq_update(enc[node[1]], assigned, estimate.leaf_dists[node[1]])
                self.leaf_counts[node[1]] += float(assigned.sum())
        self.observation_mass += float(weights.sum())

    def update(self, x: Any, weight: float, estimate: ProbabilisticCircuitDistribution) -> None:
        """Update from one weighted observation."""
        weight = validated_observation_weight(weight, "probabilistic circuit update weight")
        enc = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc, np.array([weight], dtype=np.float64), estimate)

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initialize sum counts and leaf statistics from one weighted observation."""
        weight = _finite_nonnegative_scalar(weight, "probabilistic circuit initialization weight")
        if weight == 0.0:
            return
        for i, cnt in self.sum_counts.items():
            self.sum_counts[i] = cnt + weight * rng.dirichlet(np.ones(len(cnt)))
        for lid, acc in self.leaf_accs.items():
            sc = self.leaf_scope[lid]
            acc.initialize(x[sc[0]] if len(sc) == 1 else tuple(x[v] for v in sc), weight, rng)
            self.leaf_counts[lid] += weight
        self.observation_mass += weight

    def seq_initialize(self, enc: dict[int, Any], weights: Any, rng: RandomState) -> None:
        """Initialize sum counts and leaf statistics from encoded observations."""
        if not isinstance(enc, dict) or set(enc) != set(self.leaf_accs):
            raise ValueError("probabilistic circuit encoding must contain exactly one payload per leaf.")
        row_counts = {
            leaf_id: accumulator.acc_to_encoder().row_count(enc[leaf_id])
            for leaf_id, accumulator in self.leaf_accs.items()
        }
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"probabilistic circuit leaf encodings disagree on row count: {row_counts}.")
        row_count = next(iter(row_counts.values()))
        weights = _finite_nonnegative_weights(
            weights,
            row_count,
            "probabilistic circuit initialization weights",
        )
        if float(weights.sum()) == 0.0:
            return
        for i, cnt in self.sum_counts.items():
            r = rng.dirichlet(np.ones(len(cnt)))  # random initial responsibilities break symmetry
            self.sum_counts[i] = cnt + float(np.sum(weights)) * r
        for lid, acc in self.leaf_accs.items():
            acc.seq_initialize(enc[lid], weights, rng)
            self.leaf_counts[lid] += float(weights.sum())
        self.observation_mass += float(weights.sum())

    def combine(self, suff_stat: Any) -> ProbabilisticCircuitAccumulator:
        """Merge sum-node expected counts and leaf accumulator values."""
        sc, lv, leaf_counts, observation_mass = self._validated_payload(suff_stat)
        for i in self.sum_counts:
            self.sum_counts[i] += sc[i]
        for lid in self.leaf_accs:
            self.leaf_accs[lid].combine(lv[lid])
            if leaf_counts is not None:
                self.leaf_counts[lid] += leaf_counts[lid]
        if observation_mass is not None:
            self.observation_mass += observation_mass
        return self

    def value(self) -> ProbabilisticCircuitStatistics:
        """Return sum-node expected counts and leaf sufficient statistics."""
        return ProbabilisticCircuitStatistics(
            {i: c.copy() for i, c in self.sum_counts.items()},
            {lid: a.value() for lid, a in self.leaf_accs.items()},
            dict(self.leaf_counts),
            self.observation_mass,
        )

    def from_value(self, x: Any) -> ProbabilisticCircuitAccumulator:
        """Restore sum-node and leaf sufficient statistics from ``value`` output."""
        sc, lv, leaf_counts, observation_mass = self._validated_payload(x)
        self.sum_counts = sc
        for lid, v in lv.items():
            self.leaf_accs[lid].from_value(v)
        self.leaf_counts = {lid: 0.0 for lid in self.leaf_accs} if leaf_counts is None else dict(leaf_counts)
        self.observation_mass = 0.0 if observation_mass is None else observation_mass
        return self

    def scale(self, c: float) -> ProbabilisticCircuitAccumulator:
        """Scale sum-node and leaf sufficient statistics by a constant."""
        c = _finite_nonnegative_scalar(c, "probabilistic circuit statistic scale")
        for i in self.sum_counts:
            self.sum_counts[i] *= c
        for lid in self.leaf_accs:
            self.leaf_accs[lid].scale(c)
            self.leaf_counts[lid] *= c
        self.observation_mass *= c
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
            lid: self.leaf_estimators[lid].estimate(self.leaf_counts[lid], self.leaf_accs[lid].value())
            for lid in self.leaf_accs
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
        self.pseudo_count = (
            0.0
            if pseudo_count is None
            else _finite_nonnegative_scalar(pseudo_count, "probabilistic circuit pseudo_count")
        )
        self.leaf_estimators = {lid: d.estimator() for lid, d in dist.leaf_dists.items()}

    def accumulator_factory(self) -> ProbabilisticCircuitAccumulatorFactory:
        """Return a factory for circuit-flow sufficient-statistic accumulators."""
        return ProbabilisticCircuitAccumulatorFactory(self.dist.nodes, self.dist.leaf_scope, self.leaf_estimators)

    def estimate(self, nobs: float | None, suff_stat: Any) -> ProbabilisticCircuitDistribution:
        """Estimate sum-node weights and leaf distributions from accumulated circuit flows."""
        sum_counts, leaf_values, leaf_counts, observation_mass = _unpack_circuit_statistics(
            suff_stat,
            self.leaf_estimators,
        )
        if observation_mass is not None:
            validate_effective_sample_mass(
                nobs,
                observation_mass,
                label="probabilistic circuit effective sample",
            )
        expected_sum_ids = {index for index, node in enumerate(self.dist.nodes) if node[0] == "sum"}
        if not isinstance(sum_counts, dict) or set(sum_counts) != expected_sum_ids:
            raise ValueError("probabilistic circuit sum statistics do not match the circuit sum nodes.")
        if not isinstance(leaf_values, dict) or set(leaf_values) != set(self.leaf_estimators):
            raise ValueError("probabilistic circuit leaf statistics do not match the circuit leaves.")
        new_nodes: list[tuple] = []
        for i, node in enumerate(self.dist.nodes):
            if node[0] == "sum":
                try:
                    cnt = np.asarray(sum_counts[i], dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TypeError(f"probabilistic circuit sum node {i} counts must be numeric.") from exc
                if cnt.shape != (len(node[1]),):
                    raise ValueError(f"probabilistic circuit sum node {i} counts must have shape ({len(node[1])},).")
                if np.any(~np.isfinite(cnt)) or np.any(cnt < 0.0):
                    raise ValueError(f"probabilistic circuit sum node {i} counts must be finite and non-negative.")
                cnt = cnt + self.pseudo_count
                total = float(cnt.sum())
                if np.any(~np.isfinite(cnt)) or not np.isfinite(total):
                    raise ValueError(f"probabilistic circuit sum node {i} effective counts overflowed.")
                w = cnt / total if total > 0 else np.exp(np.asarray(node[2], dtype=np.float64))
                with np.errstate(divide="ignore"):
                    new_nodes.append(("sum", node[1], tuple(np.log(w))))
            else:
                new_nodes.append(node)
        new_leaf_dists = {
            lid: self.leaf_estimators[lid].estimate(
                None if leaf_counts is None else leaf_counts[lid],
                leaf_values[lid],
            )
            for lid in leaf_values
        }
        return self.dist.with_params(new_nodes, new_leaf_dists)
