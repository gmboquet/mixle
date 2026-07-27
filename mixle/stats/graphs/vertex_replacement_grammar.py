"""Vertex-replacement (NLC) graph grammar -- a distribution over networks you can score, fit, and sample.

A node-label-controlled (NLC) vertex-replacement grammar: each rule rewrites a single nonterminal NODE
with a right-hand-side graph and reconnects it to the replaced node's former neighbours via an NLC
embedding relation (pairs of ``(neighbour_label, rhs_node_label)``). This is one kind of graph grammar;
the other main kind -- hyperedge replacement -- lives in ``hyperedge_replacement_grammar``.

Observations are GRAPHS (networkx graphs); the model is parameterised by a ``VertexReplacementGrammar``.

- ``log_density(graph)`` is the grammar's MARGINAL likelihood: the graph is parsed (reduced back to the
  start symbol along the productions) and the score is the log-sum over ALL derivations that yield it
  (the inside / sum-product recursion, ``marginal_log_prob``). It is exact when the parse forest is
  fully explored, a certified partial-mass lower bound if the budget truncates it, and ``-inf`` if the
  grammar cannot derive the graph. ``best_derivation`` gives the single best (Viterbi) parse.
- ``sample()`` runs a real vertex-replacement derivation, so sampling and scoring share one space.
- the estimator learns rule FREQUENCIES from graphs by Viterbi parse-counting (the rule structure is
  given; inducing the structure from graphs is a separate problem, out of scope).

Defines ``VertexReplacementRule``, ``VertexReplacementGrammar``, and the
``VertexReplacementGrammar{Distribution,Sampler,Estimator,Accumulator,AccumulatorFactory,DataEncoder}``
classes. Pre-0.4 generic ``Grammar*`` spellings remain as aliases.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

try:
    import networkx as nx
    import networkx.algorithms.isomorphism as iso
    from networkx.readwrite import json_graph
except ImportError:  # networkx is an optional extra; the module stays importable (serialization walks it)
    nx = iso = json_graph = None


def _require_networkx() -> None:
    if nx is None:
        raise ImportError("The graph-grammar models require networkx. Install it with `pip install mixle[grammar]`.")


from mixle.engines.arithmetic import *
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.deprecation import deprecated_alias

#: Node attribute marking a nonterminal: its value is the left-hand-side symbol to rewrite during
#: a derivation. A node is rewritable iff this attribute is present and indexes a rule in the grammar.
_NONTERMINAL = "nonterminal"


class VertexReplacementRule:
    """A node-replacement rule: rewrite a nonterminal node with ``graph``, then reconnect via ``embedding``.

    The right-hand side ``graph`` is a networkx graph whose nodes are terminals (carrying ``label`` /
    ``node_color``) or nonterminals (carrying a ``nonterminal`` attribute equal to some rule's
    left-hand side, enabling recursive derivation). ``embedding`` is an NLC-style connection relation:
    an iterable of ``(neighbour_label, rhs_node_label)`` pairs. When this rule replaces a node v, each
    former neighbour u of v is reconnected to every right-hand-side node w with
    ``(label(u), label(w))`` in the relation (the original edge data is preserved). ``embedding=None``
    means "no relation given": each former neighbour is connected to the right-hand side's canonical
    connector (its first node), which keeps derivations connected.
    """

    __pysp_serializable__ = True

    def __init__(self, lhs, graph, frequency=1.0, embedding=None) -> None:
        _require_networkx()
        if lhs is None:
            raise ValueError("vertex-replacement rule lhs cannot be None.")
        try:
            hash(lhs)
        except TypeError as exc:
            raise ValueError("vertex-replacement rule lhs must be hashable.") from exc
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise ValueError("vertex-replacement rule RHS must be an undirected simple networkx Graph.")
        if graph.number_of_nodes() == 0:
            raise ValueError("empty vertex-replacement right-hand sides are not supported.")
        checked_frequency = float(frequency)
        if not np.isfinite(checked_frequency) or checked_frequency < 0.0:
            raise ValueError("vertex-replacement rule frequency must be finite and non-negative.")
        for _, attrs in graph.nodes(data=True):
            is_nonterminal = _NONTERMINAL in attrs
            if is_nonterminal and attrs[_NONTERMINAL] is None:
                raise ValueError("RHS nonterminal symbols cannot be None.")
            if is_nonterminal and set(attrs) != {_NONTERMINAL}:
                raise ValueError("RHS nonterminal nodes may carry only the nonterminal symbol.")
            if not is_nonterminal and "label" not in attrs:
                raise ValueError("every terminal RHS node must define a label.")
            if not is_nonterminal:
                try:
                    hash(attrs["label"])
                except TypeError as exc:
                    raise ValueError("terminal RHS labels must be hashable.") from exc
        for _, _, attrs in graph.edges(data=True):
            try:
                finite_weight = np.isfinite(float(attrs.get("weight", 1.0)))
            except (TypeError, ValueError):
                finite_weight = False
            if not finite_weight:
                raise ValueError("RHS edge weights must be finite numeric values.")
        checked_embedding = None
        if embedding is not None:
            checked_embedding = []
            rhs_labels = {attrs.get("label") for _, attrs in graph.nodes(data=True) if _NONTERMINAL not in attrs}
            for pair in embedding:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("embedding entries must be (neighbour_label, rhs_node_label) pairs.")
                checked = tuple(pair)
                if checked[1] not in rhs_labels:
                    raise ValueError("embedding rhs labels must name terminal nodes in the rule RHS.")
                checked_embedding.append(checked)
            if len(set(checked_embedding)) != len(checked_embedding):
                raise ValueError("embedding relation cannot contain duplicate pairs.")
        self.lhs = lhs
        self.graph = graph.copy()
        self.frequency = checked_frequency
        self.embedding = None if checked_embedding is None else tuple(checked_embedding)

    @property
    def embedding_relation(self):
        """The embedding as a set of ``(neighbour_label, rhs_node_label)`` tuples (empty if ``None``)."""
        return set() if self.embedding is None else set(self.embedding)

    def __pysp_getstate__(self):
        return {
            "lhs": self.lhs,
            "graph": json_graph.node_link_data(self.graph, edges="edges"),
            "frequency": self.frequency,
            "embedding": None if self.embedding is None else [list(pair) for pair in self.embedding],
        }

    def __pysp_setstate__(self, state):
        restored = VertexReplacementRule(
            state["lhs"],
            json_graph.node_link_graph(state["graph"], edges="edges"),
            state["frequency"],
            embedding=state.get("embedding"),
        )
        self.__dict__.update(restored.__dict__)

    def __str__(self) -> str:
        return "VertexReplacementRule(lhs=%s, frequency=%s, nodes=%s, edges=%s, embedding=%s)" % (
            repr(self.lhs),
            repr(self.frequency),
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            "default" if self.embedding is None else "%d pair(s)" % len(self.embedding),
        )


class VertexReplacementGrammar:
    """Small in-tree node-replacement grammar container."""

    __pysp_serializable__ = True

    def __init__(self, grammar_type="mu_level_dl", clustering="leiden", name="", mu=4) -> None:
        _require_networkx()
        if not isinstance(grammar_type, str) or not grammar_type:
            raise ValueError("grammar_type must be a non-empty string.")
        if not isinstance(clustering, str) or not clustering:
            raise ValueError("clustering must be a non-empty string.")
        if not isinstance(name, str):
            raise ValueError("grammar name must be a string.")
        if isinstance(mu, (bool, np.bool_)) or not isinstance(mu, (int, np.integer)):
            raise ValueError("mu must be an exact non-Boolean integer.")
        checked_mu = int(mu)
        if checked_mu < 1:
            raise ValueError("mu must be positive.")
        self.type = grammar_type
        self.clustering = clustering
        self.name = name
        self.mu = checked_mu
        self.rule_dict = {}
        self.rule_list = []
        self.cost = 0.0
        self.num_rules = 0

    def add_rule(self, rule: VertexReplacementRule) -> None:
        """Add a replacement rule and refresh derived rule lists."""
        if not isinstance(rule, VertexReplacementRule):
            raise ValueError("VertexReplacementGrammar accepts only VertexReplacementRule instances.")
        self.rule_dict.setdefault(rule.lhs, []).append(rule)
        self.refresh_rules()

    def refresh_rules(self) -> None:
        """Refresh cached flat rule lists and rule counts."""
        self.rule_list = [rule for rules in self.rule_dict.values() for rule in rules]
        self.num_rules = len(self.rule_list)

    def __pysp_getstate__(self):
        return {
            "type": self.type,
            "clustering": self.clustering,
            "name": self.name,
            "mu": self.mu,
            "rule_dict": self.rule_dict,
            "cost": self.cost,
            "num_rules": self.num_rules,
        }

    def __pysp_setstate__(self, state):
        restored = VertexReplacementGrammar(state["type"], state["clustering"], state["name"], state["mu"])
        for symbol, rules in state["rule_dict"].items():
            for rule in rules:
                if rule.lhs != symbol:
                    raise ValueError("serialized grammar rule_dict key does not match rule lhs.")
                restored.add_rule(_copy_rule(rule))
        restored.cost = float(state["cost"])
        if not np.isfinite(restored.cost):
            raise ValueError("grammar cost must be finite.")
        if state.get("num_rules", restored.num_rules) != restored.num_rules:
            raise ValueError("serialized grammar num_rules does not match rule_dict.")
        self.__dict__.update(restored.__dict__)

    def __str__(self) -> str:
        return "VertexReplacementGrammar(name=%s, num_rules=%s)" % (repr(self.name), self.num_rules)


def _copy_rule(rule):
    return VertexReplacementRule(rule.lhs, rule.graph, rule.frequency, embedding=rule.embedding)


def _copy_grammar(grammar):
    if not isinstance(grammar, VertexReplacementGrammar):
        raise ValueError("grammar must be a VertexReplacementGrammar.")
    copied = VertexReplacementGrammar(grammar.type, grammar.clustering, grammar.name, grammar.mu)
    copied.cost = float(grammar.cost)
    for symbol, rules in grammar.rule_dict.items():
        for rule in rules:
            if rule.lhs != symbol:
                raise ValueError("grammar rule_dict keys must agree with each rule lhs.")
            copied.add_rule(_copy_rule(rule))
    return copied


def _validate_grammar(grammar, *, start_symbol=None):
    copied = _copy_grammar(grammar)
    if not copied.rule_dict:
        raise ValueError("vertex-replacement grammar must contain at least one rule.")
    for symbol, rules in copied.rule_dict.items():
        total = float(sum(rule.frequency for rule in rules))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"rules for symbol {symbol!r} must have positive finite total frequency.")
        for rule in rules:
            for _, attrs in rule.graph.nodes(data=True):
                child = attrs.get(_NONTERMINAL)
                if child is not None and child not in copied.rule_dict:
                    raise ValueError(f"RHS references nonterminal {child!r} with no production rules.")
    if start_symbol is not None and start_symbol not in copied.rule_dict:
        raise ValueError("start_symbol must identify a grammar left-hand side.")
    return copied


def _exact_positive_int(value, *, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact non-Boolean integer.")
    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be positive.")
    return checked


def _exact_nonnegative_int(value, *, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact non-Boolean integer.")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative.")
    return checked


@dataclass(frozen=True)
class GrammarSamplingReceipt:
    """Completion evidence for one budgeted grammar derivation."""

    completed: bool
    steps: int
    max_steps: int
    active_sites: int
    node_count: int


class GrammarSamplingTruncated(RuntimeError):
    """Raised when a grammar sample cannot finish within its declared work budget."""

    def __init__(self, receipt: GrammarSamplingReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            f"vertex-replacement sample truncated after {receipt.steps}/{receipt.max_steps} steps "
            f"with {receipt.active_sites} active sites"
        )


@dataclass(frozen=True)
class GrammarParseReceipt:
    """Exactness and support evidence for one grammar parse."""

    exact: bool
    derivable: bool
    expansions: int
    budget: int


class GrammarParseTruncated(RuntimeError):
    """Raised when exact fitting cannot complete a grammar parse."""

    def __init__(self, receipt: GrammarParseReceipt) -> None:
        self.receipt = receipt
        super().__init__(f"vertex-replacement parse exhausted its {receipt.budget}-expansion budget")


#: Default cap on reduction-step expansions while parsing a single graph (graph-grammar parsing is
#: NP-hard, so the search is bounded; a graph the grammar cannot derive scores -inf).
_PARSE_BUDGET = 50_000


def _grammar_node_match(g_attrs, p_attrs):
    """Match a host-graph node to a right-hand-side node: terminals by label, nonterminals by symbol."""
    return dict(g_attrs) == dict(p_attrs)


def _grammar_edge_match(g_attrs, p_attrs):
    return dict(g_attrs) == dict(p_attrs)


def _reduce_occurrence(graph, mapping, rule, symbol):
    """Reverse one rule application: collapse a matched right-hand-side occurrence to one ``symbol`` node.

    ``mapping`` is a host-node -> rhs-node induced-subgraph isomorphism. The reduction is valid only if
    the occurrence's external edges are exactly what ``rule``'s embedding would have produced when the
    rule was applied (otherwise this occurrence could not have come from this rule). Returns the reduced
    graph, or None if the embedding check fails.
    """
    rhs = rule.graph
    occurrence = set(mapping)
    inv = {p: g for g, p in mapping.items()}  # rhs node -> host node
    relation = rule.embedding_relation
    connector = None if relation else inv[next(iter(rhs.nodes()))]  # forward's canonical connector

    external = {}  # external neighbour -> [host nodes in the occurrence it touches], + representative edge data
    edge_data = {}
    for g_node in occurrence:
        for u, data in graph[g_node].items():
            if u in occurrence:
                continue
            external.setdefault(u, set()).add(g_node)
            candidate = dict(data)
            if u in edge_data and edge_data[u] != candidate:
                return None
            edge_data[u] = candidate

    for u, touched in external.items():
        if relation:
            u_label = graph.nodes[u].get("label")
            expected = {inv[w] for w in rhs.nodes if (u_label, rhs.nodes[w].get("label")) in relation}
        else:
            expected = {connector}
        if touched != expected:
            return None  # external connectivity inconsistent with the embedding -> not this rule application

    reduced = graph.copy()
    reduced.remove_nodes_from(occurrence)
    new_node = object()  # unique, transient id for the reinstated nonterminal
    reduced.add_node(new_node, **{_NONTERMINAL: symbol})
    for u, data in edge_data.items():
        reduced.add_edge(u, new_node, **dict(data))
    return reduced


def _reductions(graph, grammar):
    """Yield ``(reduced_graph, rule, symbol_total_frequency)`` for every valid single reverse step.

    Reductions are deduplicated by (rule, occurrence node-set): an occurrence's internal automorphisms
    yield several isomorphism mappings, but they are the *same* derivation step (one rule applied at one
    location). Counting them once is required for the marginal likelihood (summing over derivations) and
    harmless for the Viterbi maximum.
    """
    totals = {s: float(sum(r.frequency for r in rules)) for s, rules in grammar.rule_dict.items()}
    for symbol, rules in grammar.rule_dict.items():
        if totals[symbol] <= 0.0:
            continue
        for rule in rules:
            if rule.frequency <= 0.0 or rule.graph.number_of_nodes() == 0:
                continue
            matcher = iso.GraphMatcher(
                graph, rule.graph, node_match=_grammar_node_match, edge_match=_grammar_edge_match
            )
            seen = set()
            for mapping in matcher.subgraph_isomorphisms_iter():
                occurrence = frozenset(mapping)
                if occurrence in seen:
                    continue
                reduced = _reduce_occurrence(graph, mapping, rule, symbol)
                if reduced is not None:
                    seen.add(occurrence)  # one step per (rule, occurrence); skip automorphic duplicates
                    yield reduced, rule, totals[symbol]


def _active_site_count(graph, grammar):
    return sum(graph.nodes[node].get(_NONTERMINAL) in grammar.rule_dict for node in graph.nodes)


def best_derivation(graph, grammar, start_symbol, budget=_PARSE_BUDGET, with_status=False):
    """Best (Viterbi) derivation of a graph under the grammar: parse by reducing to the start symbol.

    Repeatedly un-applies rules (``_reductions``) until a single ``start_symbol`` node remains, searching
    for the reduction sequence of highest probability ``prod freq(rule)/total(lhs)``. Returns
    ``(log_probability, [rules applied in derivation order])``; ``(-inf, None)`` if the graph cannot be
    reduced to the start symbol (the grammar does not generate it) or the search budget is exhausted.

    This is the max over derivations, a tractable lower bound on the exact likelihood (sum over all
    derivations), which is intractable -- general graph-grammar parsing is NP-hard.
    """
    checked_budget = _exact_positive_int(budget, name="budget")
    remaining = [checked_budget]
    truncated = [False]

    def solve(h, depth):
        if h.number_of_nodes() == 1 and h.number_of_edges() == 0:
            (only,) = h.nodes
            if h.nodes[only].get(_NONTERMINAL) == start_symbol:
                return 0.0, []  # reached the start symbol -- a complete derivation
            # otherwise (e.g. a lone terminal from a single-node rule) keep reducing below
        if depth <= 0 or remaining[0] <= 0:
            truncated[0] = True
            return float("-inf"), None
        best_lp, best_seq = float("-inf"), None
        for reduced, rule, total in _reductions(h, grammar):
            remaining[0] -= 1
            if remaining[0] <= 0:
                truncated[0] = True
                break
            sub_lp, sub_seq = solve(reduced, depth - 1)
            if sub_seq is not None:
                active_sites = _active_site_count(reduced, grammar)
                if active_sites < 1:
                    continue
                lp = float(np.log(rule.frequency / total)) - float(np.log(active_sites)) + sub_lp
                if lp > best_lp:
                    best_lp, best_seq = lp, [rule, *sub_seq]
        return best_lp, best_seq

    if graph.number_of_nodes() == 0:
        result = (float("-inf"), None)
    else:
        result = solve(graph, 3 * graph.number_of_nodes() + 10)
    receipt = GrammarParseReceipt(
        exact=not truncated[0],
        derivable=result[1] is not None,
        expansions=checked_budget - remaining[0],
        budget=checked_budget,
    )
    return (*result, receipt) if with_status else result


def marginal_log_prob(graph, grammar, start_symbol, budget=_PARSE_BUDGET, with_status=False):
    """Marginal log-likelihood of a graph: log-sum over ALL derivations that yield it.

    This is the inside (sum-product) recursion over the reduction state graph -- identical to
    ``best_derivation`` but combining a state's children with ``logsumexp`` instead of ``max``, so it
    sums ``prod freq(rule)/total(lhs)`` over every parse rather than taking the single best one. It is
    therefore >= the Viterbi value and equals the EXACT marginal when the whole parse forest is explored.

    The search is bounded by ``budget`` (reduction expansions) and a recursion depth of ``3n+10``. If
    either cap is reached the forest is truncated and the result is the log-sum over the *explored*
    parses -- a partial-mass lower bound, still >= Viterbi.
    For acyclic grammars on graphs that fit the budget, neither cap is hit and the value is exact.

    Args:
        with_status: if True, return ``(value, exact)`` where ``exact`` is False iff a cap was reached
            (so the value may be a lower bound); if False, return just ``value``.

    Returns -inf if the grammar cannot derive the graph at all.
    """
    checked_budget = _exact_positive_int(budget, name="budget")
    remaining = [checked_budget]
    truncated = [False]

    def inside(h, depth):
        if h.number_of_nodes() == 1 and h.number_of_edges() == 0:
            (only,) = h.nodes
            if h.nodes[only].get(_NONTERMINAL) == start_symbol:
                return 0.0  # the start symbol: one (empty) completion, probability 1
        if depth <= 0 or remaining[0] <= 0:
            truncated[0] = True  # a cap was reached, so the explored forest is partial
            return float("-inf")
        terms = []
        for reduced, rule, total in _reductions(h, grammar):
            remaining[0] -= 1
            if remaining[0] <= 0:
                truncated[0] = True
                break
            sub = inside(reduced, depth - 1)
            if sub != float("-inf"):
                active_sites = _active_site_count(reduced, grammar)
                if active_sites > 0:
                    terms.append(float(np.log(rule.frequency / total)) - float(np.log(active_sites)) + sub)
        if not terms:
            return float("-inf")
        high = max(terms)
        return high + float(np.log(sum(np.exp(t - high) for t in terms)))  # logsumexp

    value = float("-inf") if graph.number_of_nodes() == 0 else inside(graph, 3 * graph.number_of_nodes() + 10)
    return (value, not truncated[0]) if with_status else value


def _isomorphic_rule_graph(g1, g2):
    g1i = nx.convert_node_labels_to_integers(g1)
    g2i = nx.convert_node_labels_to_integers(g2)
    node_match = iso.categorical_node_match(["label", "node_color"], ["", ""])
    color_match = iso.categorical_edge_match("edge_color", "")
    weight_match = iso.numerical_edge_match("weight", 1.0)
    return nx.is_isomorphic(g1i, g2i, edge_match=color_match, node_match=node_match) and nx.is_isomorphic(
        g1i, g2i, edge_match=weight_match, node_match=node_match
    )


def decomp_pair(sub_rule, method="connected"):
    """Decompose a sub-rule graph into connected components.

    This conservative fallback leaves connected graphs unchanged and produces one sub-rule per connected component
    for disconnected graphs.
    """
    lhs, graph = sub_rule
    if graph.number_of_nodes() == 0:
        return []
    components = list(nx.connected_components(graph.to_undirected()))
    if len(components) <= 1:
        return []
    return [(len(component), graph.subgraph(component).copy()) for component in components]


def _rhs_has_nonterminal(graph, rule_dict):
    """True if any node of ``graph`` is a nonterminal that some rule can rewrite."""
    return any(graph.nodes[n].get(_NONTERMINAL) in rule_dict for n in graph.nodes)


def _choose_rule(rules, rng):
    """Pick one rule from the declared law, proportional to frequency."""
    candidates = [r for r in rules if r.frequency > 0.0]
    if not candidates:
        return None
    weights = np.asarray([r.frequency for r in candidates], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _apply_rule(graph, node, rule, next_id, rng):
    """Replace ``node`` with a fresh copy of ``rule``'s right-hand side and embed it.

    The replaced node's incident edges are reconnected to the right-hand side according to the rule's
    NLC embedding relation -- each former neighbour u joins every right-hand-side node w with
    ``(label(u), label(w))`` in the relation -- or, when the rule has no relation, to the canonical
    connector (the first right-hand-side node). Returns the next free integer node id.
    """
    rhs = rule.graph
    mapping = {n: next_id + i for i, n in enumerate(rhs.nodes())}
    next_id += len(mapping)
    if not mapping:  # empty right-hand side: just delete the nonterminal
        graph.remove_node(node)
        return next_id
    for n in rhs.nodes:
        graph.add_node(mapping[n], **dict(rhs.nodes[n]))
    for a, b, data in rhs.edges(data=True):
        graph.add_edge(mapping[a], mapping[b], **dict(data))

    neighbours = [(u, dict(graph.get_edge_data(u, node))) for u in graph.neighbors(node) if u != node]
    relation = rule.embedding_relation
    if not relation:
        connector = mapping[next(iter(rhs.nodes()))]
        for u, edge_data in neighbours:
            graph.add_edge(u, connector, **edge_data)
    else:
        for u, edge_data in neighbours:
            u_label = graph.nodes[u].get("label")
            for n in rhs.nodes:
                if (u_label, rhs.nodes[n].get("label")) in relation:
                    graph.add_edge(u, mapping[n], **edge_data)
    graph.remove_node(node)
    return next_id


def generate_graph(rule_dict, target_n=100, rng=None, start_symbol=None, *, with_receipt=False):
    """Generate a graph by a node-label-controlled (NLC) vertex-replacement derivation.

    Starts from a single nonterminal node carrying ``start_symbol`` (default: the left-hand side with
    the most total rule frequency). Repeatedly picks a nonterminal node, chooses one of its symbol's
    rules with probability proportional to frequency, deletes the node, splices in a fresh copy of the
    rule's right-hand side, and reconnects the deleted node's former neighbours via the rule's embedding
    relation. Derivation is recursive: right-hand sides may themselves carry nonterminal nodes.

    ``target_n`` controls only the work budget (``10 * target_n + 100``
    rewrites); it never changes the rule law. If the derivation has not
    terminated at that boundary, :class:`GrammarSamplingTruncated` is raised
    with a receipt instead of returning a graph from a different conditional
    process. Returns ``(graph, symbols)`` or ``(graph, symbols, receipt)``.
    """
    rng = np.random.RandomState() if rng is None else rng
    if not rule_dict:
        raise ValueError("cannot sample from an empty vertex-replacement grammar.")
    if start_symbol is None:
        start_symbol = max(rule_dict, key=lambda s: sum(r.frequency for r in rule_dict[s]))
    if start_symbol not in rule_dict:
        raise ValueError("start_symbol must identify a grammar left-hand side.")

    target_n = _exact_positive_int(target_n, name="target_n")
    graph = nx.Graph()
    graph.add_node(0, **{_NONTERMINAL: start_symbol})
    next_id = 1
    rule_ordering = []
    max_steps = 10 * target_n + 100

    for _ in range(max_steps):
        nonterminals = [v for v in graph.nodes if graph.nodes[v].get(_NONTERMINAL) in rule_dict]
        if not nonterminals:
            break
        node = nonterminals[rng.randint(len(nonterminals))]
        symbol = graph.nodes[node][_NONTERMINAL]
        rule = _choose_rule(rule_dict[symbol], rng)
        if rule is None:
            raise ValueError(f"nonterminal {symbol!r} has no positive-frequency production.")
        next_id = _apply_rule(graph, node, rule, next_id, rng)
        rule_ordering.append(symbol)
    active = [v for v in graph.nodes if graph.nodes[v].get(_NONTERMINAL) in rule_dict]
    receipt = GrammarSamplingReceipt(
        completed=not active,
        steps=len(rule_ordering),
        max_steps=max_steps,
        active_sites=len(active),
        node_count=graph.number_of_nodes(),
    )
    if active:
        raise GrammarSamplingTruncated(receipt)
    return (graph, rule_ordering, receipt) if with_receipt else (graph, rule_ordering)


def get_degree_dist(rule_list):
    """Node-degree histogram over the graphs of a list of grammar rules.

    Args:
        rule_list: List of rule objects, each with a networkx graph attribute.

    Returns:
        Dict mapping an observed node degree to its count, plus an ``'inf'`` bucket of count 1 that
        reserves smoothing mass for degrees not seen in the model.

    """
    dist = {}
    for rule in rule_list:
        for _, degree in rule.graph.degree():
            dist[degree] = dist.get(degree, 0) + 1
    dist["inf"] = 1
    return dist


def _validate_terminal_graph(graph):
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError("vertex-grammar observations must be undirected simple networkx Graphs.")
    for _, attrs in graph.nodes(data=True):
        if _NONTERMINAL in attrs:
            raise ValueError("observed vertex-grammar graphs must be fully terminal.")
        if "label" not in attrs:
            raise ValueError("every observed graph node must define a label.")
    for _, _, attrs in graph.edges(data=True):
        weight = attrs.get("weight", 1.0)
        try:
            finite = np.isfinite(float(weight))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError("observed graph edge weights must be finite numeric values.")
    return graph


class VertexReplacementGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """VertexReplacementGrammarDistribution: a distribution over GRAPHS parameterised by a node-replacement grammar.

    Observations are terminal networkx graphs. ``log_density`` sums complete
    derivation probabilities (including active-site choice), ``sample`` emits
    exact completed derivations, and the estimator fits rule frequencies from
    exact Viterbi parses.
    """

    def __init__(
        self,
        grammar,
        mix_p=0.0,
        decomp_level=0,
        lhs_delta=0,
        name=None,
        orig_n=100,
        start_symbol=None,
        keys=None,
    ):
        """Create a vertex-replacement grammar distribution.

        Args:
            grammar: VertexReplacementGrammar object serving as the model grammar.
            mix_p (float): Removed legacy control; must be zero.
            decomp_level (int): Removed legacy control; must be zero.
            lhs_delta (int): Removed legacy control; must be zero.
            name (Optional[str]): Optional distribution name.
            orig_n (int): Scale for the sampling work budget; never changes rule probabilities.
            start_symbol: Left-hand side to begin a derivation from (default: the most frequent one).

        Attributes:
            grammar: VertexReplacementGrammar object serving as the model grammar.
            mix_p (float): Always zero.
            decomp_level (int): Always zero.
            lhs_delta (int): Always zero.
            name (Optional[str]): Optional distribution name.
            orig_n (int): Scale for the sampling work budget.
            start_symbol: Left-hand side to begin a derivation from (default: the most frequent one).

        """
        _require_networkx()
        checked_mix = float(mix_p)
        if not np.isfinite(checked_mix) or checked_mix != 0.0:
            raise ValueError("mix_p was an inert legacy control; only mix_p=0 is supported.")
        if decomp_level != 0 or lhs_delta != 0:
            raise ValueError("decomp_level and lhs_delta were inert legacy controls; only zero is supported.")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        if keys is not None and not isinstance(keys, str):
            raise ValueError("keys must be a string or None.")
        self.name = name
        self._grammar = _validate_grammar(grammar, start_symbol=start_symbol)
        self.mix_p = 0.0
        self.decomp_level = 0
        self.lhs_delta = 0
        self.orig_n = _exact_positive_int(orig_n, name="orig_n")
        self.start_symbol = start_symbol
        self.keys = keys

    @property
    def grammar(self):
        """Return a defensive copy of the immutable grammar snapshot."""
        return _copy_grammar(self._grammar)

    def __pysp_getstate__(self):
        return {
            "grammar": self._grammar,
            "mix_p": self.mix_p,
            "decomp_level": self.decomp_level,
            "lhs_delta": self.lhs_delta,
            "name": self.name,
            "orig_n": self.orig_n,
            "start_symbol": self.start_symbol,
            "keys": self.keys,
        }

    def __pysp_setstate__(self, state):
        restored = VertexReplacementGrammarDistribution(**state)
        self.__dict__.update(restored.__dict__)

    def __str__(self):
        return (
            "VertexReplacementGrammarDistribution("
            f"grammar={self._grammar}, mix_p={self.mix_p!r}, decomp_level={self.decomp_level!r}, "
            f"lhs_delta={self.lhs_delta!r}, name={self.name!r}, orig_n={self.orig_n!r}, "
            f"start_symbol={self.start_symbol!r}, keys={self.keys!r})"
        )

    def density(self, x):
        """Density of the grammar distribution at observation x.

        See log_density() for details.

        Args:
            x: Observed graph (a networkx graph).

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def _resolve_start(self):
        """The derivation start symbol: ``self.start_symbol`` or, if None, the most frequent left-hand side."""
        if self.start_symbol is not None:
            return self.start_symbol
        if not self._grammar.rule_dict:
            return None
        return max(self._grammar.rule_dict, key=lambda s: sum(r.frequency for r in self._grammar.rule_dict[s]))

    def density_semantics(self):
        """Return density semantics for budgeted grammar parsing."""
        # exact (the inside sum) unless the parse budget/depth truncates the forest -> conservatively a
        # lower bound. marginal_log_prob(..., with_status=True) certifies whether a given call was exact.
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.LOWER_BOUND

    def log_density(self, x, with_status=False):
        """Log-density of the grammar distribution at an observed GRAPH x -- the marginal likelihood.

        ``x`` is parsed (reduced back to the start symbol along the grammar's productions) and the score
        is the log-sum over ALL derivations that yield it, ``log sum_D prod_i freq(r_i)/total(lhs_i)``,
        computed by the inside (sum-product) recursion (``marginal_log_prob``). A graph the grammar
        cannot generate scores ``-inf``.

        This is the true marginal, not the Viterbi (single best-derivation) lower bound. The parse search
        is budget-bounded; if the budget truncates the parse forest the result is the probability mass of
        explored derivations, a certified lower bound. ``best_derivation`` exposes the MAP parse.

        Args:
            x: Observed graph (a networkx graph).

        Args (cont.):
            with_status: if True, return ``(value, exact)`` where ``exact`` is False iff the parse
                forest was truncated (so ``value`` may be a lower bound); if False, return just ``value``.

        Returns:
            Log-density at observation x (<= 0, or -inf if the grammar cannot derive x).

        """
        start = self._resolve_start()
        if start is None:
            return (float("-inf"), True) if with_status else float("-inf")
        _validate_terminal_graph(x)
        return marginal_log_prob(x, self._grammar, start, with_status=with_status)

    # combine list of grammars into singular grammar? need to take multiple sample outputs as input
    def seq_encode(self, x):
        """Encode a sequence of observed graphs for vectorized calls (identity encoding).

        Args:
            x: Sequence of observed graphs (networkx graphs).

        Returns:
            The input sequence unchanged.

        """
        encoded = tuple(x)
        for graph in encoded:
            _validate_terminal_graph(graph)
        return encoded

    def seq_log_density(self, x, with_status=False):
        """Evaluate log_density() at each encoded observation.

        Args:
            x: Sequence of observed graphs (from seq_encode).
            with_status: if True, also return a boolean mask that is True for rows whose marginal was
                computed exactly (the parse forest was not truncated) and False where it is a bound.

        Returns:
            A numpy array of log-densities, or ``(values, exact_mask)`` when ``with_status`` is True.

        """
        if not with_status:
            return np.asarray([self.log_density(xx) for xx in x])
        pairs = [self.log_density(xx, with_status=True) for xx in x]
        values = np.asarray([v for v, _ in pairs])
        exact = np.asarray([e for _, e in pairs], dtype=bool)
        return values, exact

    def sampler(self, seed=None):
        """Create a sampler from this grammar distribution.

        Args:
            seed (Optional[int]): Seed for the sampler random generator.

        Returns:
            VertexReplacementGrammarSampler object.

        """
        return VertexReplacementGrammarSampler(
            self._grammar, orig_n=self.orig_n, seed=seed, start_symbol=self.start_symbol
        )

    def estimator(self, pseudo_count=None):
        """Create an estimator for this grammar distribution.

        Args:
            pseudo_count (Optional[float]): Added to rule frequencies when estimating.

        Returns:
            VertexReplacementGrammarEstimator object.

        """
        return VertexReplacementGrammarEstimator(
            grammar=self._grammar,
            start_symbol=self.start_symbol,
            pseudo_count=pseudo_count,
            name=self.name,
            mix_p=self.mix_p,
            decomp_level=self.decomp_level,
            lhs_delta=self.lhs_delta,
            orig_n=self.orig_n,
            keys=self.keys,
        )

    def dist_to_encoder(self):
        """Return the encoder for grammar observations."""
        return VertexReplacementGrammarDataEncoder()


class VertexReplacementGrammarSampler(DistributionSampler):
    """Sampler for graphs generated from a node-replacement grammar."""

    def __init__(self, grammar, orig_n=100, seed=None, start_symbol=None):
        """Create a sampler for a vertex-replacement grammar distribution.

        Args:
            grammar: VertexReplacementGrammar object to generate graphs from.
            orig_n (int): Soft node budget for generated graphs (see generate_graph).
            seed (Optional[int]): Seed for the local random generator.
            start_symbol: Left-hand side to begin each derivation from (default: the most frequent one).

        Attributes:
            grammar: VertexReplacementGrammar object to generate graphs from.
            orig_n (int): Soft node budget for generated graphs.
            start_symbol: Left-hand side to begin each derivation from.

        """
        self._grammar = _validate_grammar(grammar, start_symbol=start_symbol)
        self.orig_n = _exact_positive_int(orig_n, name="orig_n")
        self.start_symbol = start_symbol
        self.rng = np.random.RandomState(seed)

    @property
    def grammar(self):
        """Return a defensive copy of the sampler's grammar snapshot."""
        return _copy_grammar(self._grammar)

    def _sample_one(self):
        g, _ = generate_graph(
            rule_dict=self._grammar.rule_dict, target_n=self.orig_n, rng=self.rng, start_symbol=self.start_symbol
        )
        return g

    def sample_with_receipt(self):
        """Generate one exact sample and return its completion receipt."""
        graph, _, receipt = generate_graph(
            rule_dict=self._grammar.rule_dict,
            target_n=self.orig_n,
            rng=self.rng,
            start_symbol=self.start_symbol,
            with_receipt=True,
        )
        return graph, receipt

    def sample(self, size=None, *, batched=True):
        """Generate graphs from the grammar by NLC vertex-replacement derivation.

        Args:
            size (Optional[int]): Number of graphs to draw; ``None`` returns a single graph (honouring
                the DistributionSampler contract). Each graph uses the sampler's ``orig_n`` node budget.
            batched (bool): Accepted for interface compatibility; results are returned as a list.

        Returns:
            A single networkx graph when ``size`` is None, else a list of ``size`` graphs.

        """
        if size is None:
            return self._sample_one()
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self._sample_one() for _ in range(sample_size)]

    def sample_seq(self, size_arr):
        """Generate one graph per entry of size_arr, each with that node budget.

        Args:
            size_arr: Sequence of node budgets.

        Returns:
            List of networkx graphs, one per requested budget.

        """
        rv = []
        for size in size_arr:
            g, _ = generate_graph(
                rule_dict=self._grammar.rule_dict,
                target_n=_exact_positive_int(size, name="size_arr entry"),
                rng=self.rng,
                start_symbol=self.start_symbol,
            )
            rv.append(g)
        return rv


def _zeroed_counts(grammar):
    """A copy of ``grammar``'s rule structure with every frequency set to 0 (a counts accumulator)."""
    counts = VertexReplacementGrammar(grammar.type, grammar.clustering, grammar.name, grammar.mu)
    for symbol, rules in grammar.rule_dict.items():
        counts.rule_dict[symbol] = [VertexReplacementRule(r.lhs, r.graph, 0.0, embedding=r.embedding) for r in rules]
    counts.refresh_rules()
    return counts


class VertexReplacementGrammarStatistics(NamedTuple):
    """Versioned, owned sufficient statistics for grammar frequency fitting."""

    schema_version: int
    counts: VertexReplacementGrammar
    accepted_weight: float
    rejected_weight: float
    truncated_weight: float


@dataclass(frozen=True)
class GrammarFitReceipt:
    """Weight accounting for exact grammar fitting."""

    accepted_weight: float
    rejected_weight: float
    truncated_weight: float


def _grammars_share_structure(left, right):
    if not isinstance(left, VertexReplacementGrammar) or not isinstance(right, VertexReplacementGrammar):
        return False
    if (
        left.type != right.type
        or left.clustering != right.clustering
        or left.mu != right.mu
        or list(left.rule_dict) != list(right.rule_dict)
    ):
        return False
    for symbol in left.rule_dict:
        left_rules = left.rule_dict[symbol]
        right_rules = right.rule_dict[symbol]
        if len(left_rules) != len(right_rules):
            return False
        for left_rule, right_rule in zip(left_rules, right_rules):
            if (
                left_rule.lhs != right_rule.lhs
                or left_rule.embedding != right_rule.embedding
                or not iso.is_isomorphic(
                    left_rule.graph,
                    right_rule.graph,
                    node_match=_grammar_node_match,
                    edge_match=_grammar_edge_match,
                )
            ):
                return False
    return True


def _validate_vertex_statistics(value, structure):
    if not isinstance(value, VertexReplacementGrammarStatistics) or value.schema_version != 1:
        raise ValueError("vertex-grammar statistics must use schema version 1.")
    if not _grammars_share_structure(value.counts, structure):
        raise ValueError("vertex-grammar statistics do not match the estimator's rule structure.")
    weights = tuple(float(v) for v in (value.accepted_weight, value.rejected_weight, value.truncated_weight))
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("vertex-grammar receipt weights must be finite and non-negative.")
    counts = _copy_grammar(value.counts)
    for rules in counts.rule_dict.values():
        for rule in rules:
            if not np.isfinite(rule.frequency) or rule.frequency < 0.0:
                raise ValueError("vertex-grammar rule counts must be finite and non-negative.")
    return VertexReplacementGrammarStatistics(1, counts, *weights)


class VertexReplacementGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate Viterbi rule-firing counts: parse each observed graph and tally how often each rule fires.

    This estimates rule FREQUENCIES only. The rule STRUCTURE (which right-hand sides / embeddings exist)
    is supplied via the estimator's ``grammar`` argument; inducing the structure from graphs is a separate
    problem and out of scope here. Counting aligns rules by (symbol, index), which is stable because every
    model in the EM loop is built from the same structure.
    """

    def __init__(self, grammar, start_symbol=None, keys=None):
        self.structure = _validate_grammar(grammar, start_symbol=start_symbol)
        self.start_symbol = start_symbol
        self.keys = keys
        self.counts = _zeroed_counts(self.structure)
        self.accepted_weight = 0.0
        self.rejected_weight = 0.0
        self.truncated_weight = 0.0

    def _parse_model(self, estimate):
        """The (grammar, start_symbol) to parse against: the previous estimate, else the given structure."""
        if estimate is not None:
            return estimate._grammar, estimate._resolve_start()
        start = self.start_symbol
        if start is None:
            start = max(self.structure.rule_dict, key=lambda s: sum(r.frequency for r in self.structure.rule_dict[s]))
        return self.structure, start

    def update(self, x, weight, estimate):
        """Parse graph ``x`` with the current model and add ``weight`` to every rule its derivation fires."""
        model_grammar, start = self._parse_model(estimate)
        _validate_terminal_graph(x)
        checked_weight = float(weight)
        if not np.isfinite(checked_weight) or checked_weight < 0.0:
            raise ValueError("vertex-grammar observation weight must be finite and non-negative.")
        if checked_weight == 0.0:
            return
        _, derivation, receipt = best_derivation(x, model_grammar, start, with_status=True)
        if not receipt.exact:
            self.truncated_weight += checked_weight
            raise GrammarParseTruncated(receipt)
        if derivation is None:
            self.rejected_weight += checked_weight
            raise ValueError("observed graph is outside the vertex grammar's support.")
        position = {id(r): (s, i) for s, rules in model_grammar.rule_dict.items() for i, r in enumerate(rules)}
        for rule in derivation:
            symbol, index = position[id(rule)]
            self.counts.rule_dict[symbol][index].frequency += checked_weight
        self.accepted_weight += checked_weight
        self.counts.refresh_rules()

    def initialize(self, x, weight, rng):
        """Initialize from one weighted observed graph (parse with the structure's current frequencies)."""
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Initialize from a sequence of weighted observed graphs."""
        self.seq_update(x, weights, None)

    def seq_update(self, x, weights, estimate):
        """Parse-and-count a sequence of weighted observed graphs against the previous estimate."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the graph batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = VertexReplacementGrammarAccumulator(self.structure, self.start_symbol)
        for graph, weight in zip(x, checked_weights):
            pending.update(graph, float(weight), estimate)
        self.combine(pending.value())

    def combine(self, suff_stat):
        """Add another accumulator's rule-firing counts (same structure) position-wise."""
        checked = _validate_vertex_statistics(suff_stat, self.structure)
        for symbol, rules in checked.counts.rule_dict.items():
            for index, rule in enumerate(rules):
                self.counts.rule_dict[symbol][index].frequency += rule.frequency
        self.accepted_weight += checked.accepted_weight
        self.rejected_weight += checked.rejected_weight
        self.truncated_weight += checked.truncated_weight
        self.counts.refresh_rules()
        return self

    def value(self):
        """Returns the accumulated rule-firing counts as a VertexReplacementGrammar."""
        return VertexReplacementGrammarStatistics(
            1,
            _copy_grammar(self.counts),
            self.accepted_weight,
            self.rejected_weight,
            self.truncated_weight,
        )

    def from_value(self, x):
        """Set accumulated counts from a vertex-replacement grammar distribution."""
        checked = _validate_vertex_statistics(x, self.structure)
        self.counts = checked.counts
        self.accepted_weight = checked.accepted_weight
        self.rejected_weight = checked.rejected_weight
        self.truncated_weight = checked.truncated_weight
        return self

    def receipt(self):
        """Return accepted/rejected/truncated weight accounting."""
        return GrammarFitReceipt(self.accepted_weight, self.rejected_weight, self.truncated_weight)

    def key_merge(self, stats_dict):
        """Merge keyed sufficient statistics into stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of keyed sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict):
        """Replace keyed sufficient statistics from stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary of keyed sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return VertexReplacementGrammarDataEncoder()


class VertexReplacementGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for grammar accumulators carrying the rule structure needed to estimate frequencies."""

    def __init__(self, grammar, start_symbol=None, keys=None):
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.keys = keys

    def make(self):
        """Return a fresh grammar accumulator."""
        return VertexReplacementGrammarAccumulator(grammar=self.grammar, start_symbol=self.start_symbol, keys=self.keys)


class VertexReplacementGrammarEstimator(ParameterEstimator):
    """Estimate a VertexReplacementGrammarDistribution's rule FREQUENCIES from graphs by Viterbi parse-counting.

    The rule structure is supplied via ``grammar`` (e.g. from ``dist.estimator()``): each training graph is
    parsed with the current model and the rules its best derivation fires are counted; frequencies are the
    accumulated counts. Inducing the structure (the right-hand sides / embeddings) from graphs is a separate
    problem and out of scope.
    """

    def __init__(
        self,
        grammar,
        start_symbol=None,
        pseudo_count=None,
        name=None,
        keys=None,
        mix_p=0.0,
        decomp_level=0,
        lhs_delta=0,
        orig_n=100,
    ):
        """Create an estimator for vertex-replacement grammar distributions.

        Args:
            grammar: VertexReplacementGrammar giving the rule structure whose frequencies are estimated.
            start_symbol: Symbol to start derivations from (default: the most frequent left-hand side).
            pseudo_count (Optional[float]): Added to each rule's counted frequency before normalising.
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Key for merging sufficient statistics with matching key'd objects.
        """
        _require_networkx()
        self._grammar = _validate_grammar(grammar, start_symbol=start_symbol)
        self.start_symbol = start_symbol
        self.pseudo_count = None if pseudo_count is None else float(pseudo_count)
        if self.pseudo_count is not None and (not np.isfinite(self.pseudo_count) or self.pseudo_count < 0.0):
            raise ValueError("pseudo_count must be finite and non-negative.")
        if float(mix_p) != 0.0 or decomp_level != 0 or lhs_delta != 0:
            raise ValueError("legacy vertex-grammar controls mix_p, decomp_level, and lhs_delta must be zero.")
        self.mix_p = 0.0
        self.decomp_level = 0
        self.lhs_delta = 0
        self.orig_n = _exact_positive_int(orig_n, name="orig_n")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        if keys is not None and not isinstance(keys, str):
            raise ValueError("keys must be a string or None.")
        self.name = name
        self.keys = keys

    @property
    def grammar(self):
        """Return a defensive copy of the fitted rule structure."""
        return _copy_grammar(self._grammar)

    def __pysp_getstate__(self):
        return {
            "grammar": self._grammar,
            "start_symbol": self.start_symbol,
            "pseudo_count": self.pseudo_count,
            "name": self.name,
            "keys": self.keys,
            "mix_p": self.mix_p,
            "decomp_level": self.decomp_level,
            "lhs_delta": self.lhs_delta,
            "orig_n": self.orig_n,
        }

    def __pysp_setstate__(self, state):
        restored = VertexReplacementGrammarEstimator(**state)
        self.__dict__.update(restored.__dict__)

    def accumulator_factory(self):
        """Returns a VertexReplacementGrammarAccumulatorFactory carrying the rule structure."""
        return VertexReplacementGrammarAccumulatorFactory(
            grammar=self._grammar, start_symbol=self.start_symbol, keys=self.keys
        )

    @deprecated_alias("accumulator_factory", since="0.8.0", removed_in="0.10.0")
    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        """Build a VertexReplacementGrammarDistribution from accumulated rule-firing counts (frequencies).

        Args:
            nobs (Optional[float]): Weighted number of observations (unused).
            suff_stat: Versioned VertexReplacementGrammarStatistics.

        Returns:
            VertexReplacementGrammarDistribution object.

        """
        checked = _validate_vertex_statistics(suff_stat, self._grammar)
        if checked.rejected_weight > 0.0 or checked.truncated_weight > 0.0:
            raise ValueError("exact vertex-grammar estimation rejects statistics with failed or truncated parses.")
        if checked.accepted_weight <= 0.0 and not (self.pseudo_count is not None and self.pseudo_count > 0.0):
            raise ValueError("cannot estimate vertex-grammar frequencies without accepted evidence or pseudo-count.")
        grammar = checked.counts
        if self.pseudo_count is not None:
            for rlist in grammar.rule_dict.values():
                for rule in rlist:
                    rule.frequency += self.pseudo_count
        for symbol, rules in grammar.rule_dict.items():
            if sum(rule.frequency for rule in rules) <= 0.0:
                source = self._grammar.rule_dict[symbol]
                for rule, original in zip(rules, source):
                    rule.frequency = original.frequency
        return VertexReplacementGrammarDistribution(
            grammar,
            self.mix_p,
            decomp_level=self.decomp_level,
            lhs_delta=self.lhs_delta,
            start_symbol=self.start_symbol,
            name=self.name,
            orig_n=self.orig_n,
            keys=self.keys,
        )


class VertexReplacementGrammarDataEncoder(DataSequenceEncoder):
    """Data encoder for observed vertex-replacement grammar graphs."""

    def __str__(self):
        """Return a constructor-style representation of the grammar encoder."""
        return "VertexReplacementGrammarDataEncoder"

    def __eq__(self, other):
        """Encoders are interchangeable iff other is also a VertexReplacementGrammarDataEncoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a VertexReplacementGrammarDataEncoder instance.

        """
        return isinstance(other, VertexReplacementGrammarDataEncoder)

    def seq_encode(self, x):
        """Encode a sequence of observed graphs for vectorized calls (identity encoding).

        Args:
            x: Sequence of observed graphs (networkx graphs).

        Returns:
            The input sequence unchanged.

        """
        encoded = tuple(x)
        for graph in encoded:
            _validate_terminal_graph(graph)
        return encoded

    def row_count(self, x):
        """Return the number of graph observations in a validated encoded payload."""
        if not isinstance(x, tuple):
            raise ValueError("encoded vertex-grammar payload must be a tuple.")
        for graph in x:
            _validate_terminal_graph(graph)
        return len(x)
