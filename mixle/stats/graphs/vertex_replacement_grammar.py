"""Vertex-replacement (NLC) graph grammar -- a distribution over networks you can score, fit, and sample.

A node-label-controlled (NLC) vertex-replacement grammar: each rule rewrites a single nonterminal NODE
with a right-hand-side graph and reconnects it to the replaced node's former neighbours via an NLC
embedding relation (pairs of ``(neighbour_label, rhs_node_label)``). This is one kind of graph grammar;
the other main kind -- hyperedge replacement -- lives in ``hyperedge_replacement_grammar``.

Observations are GRAPHS (networkx graphs); the model is parameterised by a ``VertexReplacementGrammar``.

- ``log_density(graph)`` is the grammar's MARGINAL likelihood: the graph is parsed (reduced back to the
  start symbol along the productions) and the score is the log-sum over ALL derivations that yield it
  (the inside / sum-product recursion, ``marginal_log_prob``). It is exact when the parse forest is
  fully explored, a variational lower bound (ELBO) if the budget truncates it, and ``-inf`` if the
  grammar cannot derive the graph. ``best_derivation`` gives the single best (Viterbi) parse.
- ``sample()`` runs a real vertex-replacement derivation, so sampling and scoring share one space.
- the estimator learns rule FREQUENCIES from graphs by Viterbi parse-counting (the rule structure is
  given; inducing the structure from graphs is a separate problem, out of scope).

Defines ``VertexReplacementRule``, ``VertexReplacementGrammar``, and the
``VertexReplacementGrammar{Distribution,Sampler,Estimator,Accumulator,AccumulatorFactory,DataEncoder}``
classes. Pre-0.4 generic ``Grammar*`` spellings remain as aliases.
"""

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
        self.lhs = lhs
        self.graph = graph.copy()
        self.frequency = float(frequency)
        self.embedding = None if embedding is None else [tuple(pair) for pair in embedding]

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
        self.lhs = state["lhs"]
        self.graph = json_graph.node_link_graph(state["graph"], edges="edges")
        self.frequency = float(state["frequency"])
        emb = state.get("embedding")
        self.embedding = None if emb is None else [tuple(pair) for pair in emb]

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
        self.type = grammar_type
        self.clustering = clustering
        self.name = name
        self.mu = mu
        self.rule_dict = {}
        self.rule_list = []
        self.cost = 0.0
        self.num_rules = 0

    def add_rule(self, rule: VertexReplacementRule) -> None:
        self.rule_dict.setdefault(rule.lhs, []).append(rule)
        self.refresh_rules()

    def refresh_rules(self) -> None:
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
        self.type = state["type"]
        self.clustering = state["clustering"]
        self.name = state["name"]
        self.mu = state["mu"]
        self.rule_dict = state["rule_dict"]
        self.cost = state["cost"]
        self.refresh_rules()
        self.num_rules = state.get("num_rules", self.num_rules)

    def __str__(self) -> str:
        return "VertexReplacementGrammar(name=%s, num_rules=%s)" % (repr(self.name), self.num_rules)


def _copy_rule(rule):
    return VertexReplacementRule(rule.lhs, rule.graph, rule.frequency, embedding=rule.embedding)


#: Default cap on reduction-step expansions while parsing a single graph (graph-grammar parsing is
#: NP-hard, so the search is bounded; a graph the grammar cannot derive scores -inf).
_PARSE_BUDGET = 50_000


def _grammar_node_match(g_attrs, p_attrs):
    """Match a host-graph node to a right-hand-side node: terminals by label, nonterminals by symbol."""
    g_nt = g_attrs.get(_NONTERMINAL)
    p_nt = p_attrs.get(_NONTERMINAL)
    if (g_nt is None) != (p_nt is None):
        return False
    if g_nt is not None:
        return g_nt == p_nt
    return g_attrs.get("label") == p_attrs.get("label") and g_attrs.get("node_color", "") == p_attrs.get(
        "node_color", ""
    )


def _grammar_edge_match(g_attrs, p_attrs):
    return g_attrs.get("edge_color", "") == p_attrs.get("edge_color", "") and g_attrs.get("weight", 1.0) == p_attrs.get(
        "weight", 1.0
    )


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
            edge_data[u] = data

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


def best_derivation(graph, grammar, start_symbol, budget=_PARSE_BUDGET):
    """Best (Viterbi) derivation of a graph under the grammar: parse by reducing to the start symbol.

    Repeatedly un-applies rules (``_reductions``) until a single ``start_symbol`` node remains, searching
    for the reduction sequence of highest probability ``prod freq(rule)/total(lhs)``. Returns
    ``(log_probability, [rules applied in derivation order])``; ``(-inf, None)`` if the graph cannot be
    reduced to the start symbol (the grammar does not generate it) or the search budget is exhausted.

    This is the max over derivations, a tractable lower bound on the exact likelihood (sum over all
    derivations), which is intractable -- general graph-grammar parsing is NP-hard.
    """
    remaining = [budget]

    def solve(h, depth):
        if h.number_of_nodes() == 1 and h.number_of_edges() == 0:
            (only,) = h.nodes
            if h.nodes[only].get(_NONTERMINAL) == start_symbol:
                return 0.0, []  # reached the start symbol -- a complete derivation
            # otherwise (e.g. a lone terminal from a single-node rule) keep reducing below
        if depth <= 0 or remaining[0] <= 0:
            return float("-inf"), None
        best_lp, best_seq = float("-inf"), None
        for reduced, rule, total in _reductions(h, grammar):
            remaining[0] -= 1
            if remaining[0] <= 0:
                break
            sub_lp, sub_seq = solve(reduced, depth - 1)
            if sub_seq is not None:
                lp = float(np.log(rule.frequency / total)) + sub_lp
                if lp > best_lp:
                    best_lp, best_seq = lp, [rule, *sub_seq]
        return best_lp, best_seq

    if graph.number_of_nodes() == 0:
        return float("-inf"), None  # the start symbol always derives at least one node
    return solve(graph, 3 * graph.number_of_nodes() + 10)


def marginal_log_prob(graph, grammar, start_symbol, budget=_PARSE_BUDGET, with_status=False):
    """Marginal log-likelihood of a graph: log-sum over ALL derivations that yield it.

    This is the inside (sum-product) recursion over the reduction state graph -- identical to
    ``best_derivation`` but combining a state's children with ``logsumexp`` instead of ``max``, so it
    sums ``prod freq(rule)/total(lhs)`` over every parse rather than taking the single best one. It is
    therefore >= the Viterbi value and equals the EXACT marginal when the whole parse forest is explored.

    The search is bounded by ``budget`` (reduction expansions) and a recursion depth of ``3n+10``. If
    either cap is reached the forest is truncated and the result is the log-sum over the *explored*
    parses -- a variational ELBO (the tightest bound for a posterior on that set), still >= Viterbi.
    For acyclic grammars on graphs that fit the budget, neither cap is hit and the value is exact.

    Args:
        with_status: if True, return ``(value, exact)`` where ``exact`` is False iff a cap was reached
            (so the value may be a lower bound); if False, return just ``value``.

    Returns -inf if the grammar cannot derive the graph at all.
    """
    remaining = [budget]
    truncated = [False]

    def inside(h, depth):
        if h.number_of_nodes() == 1 and h.number_of_edges() == 0:
            (only,) = h.nodes
            if h.nodes[only].get(_NONTERMINAL) == start_symbol:
                return 0.0  # the start symbol: one (empty) completion, probability 1
        if depth <= 0 or remaining[0] <= 0:
            truncated[0] = True  # a cap was reached -> the explored forest may be incomplete
            return float("-inf")
        terms = []
        for reduced, rule, total in _reductions(h, grammar):
            remaining[0] -= 1
            if remaining[0] <= 0:
                truncated[0] = True
                break
            sub = inside(reduced, depth - 1)
            if sub != float("-inf"):
                terms.append(float(np.log(rule.frequency / total)) + sub)
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


def _choose_rule(rules, rng, rule_dict, prefer_terminal):
    """Pick one rule with probability proportional to frequency.

    When ``prefer_terminal`` is set, restrict to rules whose right-hand side has no nonterminals (so the
    derivation can terminate); fall back to all rules if the symbol has no terminal-only rule.
    """
    candidates = [r for r in rules if r.frequency > 0.0]
    if not candidates:
        return None
    if prefer_terminal:
        terminal = [r for r in candidates if not _rhs_has_nonterminal(r.graph, rule_dict)]
        if terminal:
            candidates = terminal
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


def generate_graph(rule_dict, target_n=100, rng=None, start_symbol=None):
    """Generate a graph by a node-label-controlled (NLC) vertex-replacement derivation.

    Starts from a single nonterminal node carrying ``start_symbol`` (default: the left-hand side with
    the most total rule frequency). Repeatedly picks a nonterminal node, chooses one of its symbol's
    rules with probability proportional to frequency, deletes the node, splices in a fresh copy of the
    rule's right-hand side, and reconnects the deleted node's former neighbours via the rule's embedding
    relation. Derivation is recursive: right-hand sides may themselves carry nonterminal nodes.

    ``target_n`` is a soft node budget, not an exact size: once it is reached the derivation prefers
    terminal-only rules so it can finish, and any nonterminals still left after the step cap are demoted
    to terminals. A non-recursive grammar therefore yields exactly its right-hand side, while a
    recursive one grows until the budget. Returns (networkx graph, list of symbols rewritten in order).
    """
    rng = np.random.RandomState() if rng is None else rng
    if not rule_dict:
        return nx.Graph(), []
    if start_symbol is None:
        start_symbol = max(rule_dict, key=lambda s: sum(r.frequency for r in rule_dict[s]))
    if start_symbol not in rule_dict:
        return nx.Graph(), []

    target_n = max(1, int(target_n))
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
        over_budget = graph.number_of_nodes() >= target_n
        rule = _choose_rule(rule_dict[symbol], rng, rule_dict, prefer_terminal=over_budget)
        if rule is None:
            graph.nodes[node].pop(_NONTERMINAL, None)  # no usable rule -> treat as terminal
            continue
        next_id = _apply_rule(graph, node, rule, next_id, rng)
        rule_ordering.append(symbol)

    for v in graph.nodes:  # demote any nonterminals left after the budget/step cap
        graph.nodes[v].pop(_NONTERMINAL, None)
    return graph, rule_ordering


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


class VertexReplacementGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """VertexReplacementGrammarDistribution: a distribution over GRAPHS parameterised by a node-replacement grammar.

    Observations are networkx graphs. ``log_density`` scores a graph by the product over its nodes of
    the model probability of each node's ego pattern, ``sample`` emits graphs by derivation, and the
    estimator learns the model grammar from graphs -- so all three share the graph sample space.
    """

    def __init__(self, grammar, mix_p, decomp_level=0, lhs_delta=0, name=None, orig_n=100, start_symbol=None):
        """VertexReplacementGrammarDistribution object defined by a model grammar and mixing parameters.

        Args:
            grammar: VertexReplacementGrammar object serving as the model grammar.
            mix_p (float): Weight in [0, 1] on the degree-distribution background model.
            decomp_level (int): Maximum recursion depth for decomposing unmatched rules.
            lhs_delta (int): Allowed slack when matching rule left-hand sides.
            name (Optional[str]): String name of object instance.
            orig_n (int): Soft node budget used when sampling graphs by derivation.
            start_symbol: Left-hand side to begin a derivation from (default: the most frequent one).

        Attributes:
            grammar: VertexReplacementGrammar object serving as the model grammar.
            mix_p (float): Weight in [0, 1] on the degree-distribution background model.
            decomp_level (int): Maximum recursion depth for decomposing unmatched rules.
            lhs_delta (int): Allowed slack when matching rule left-hand sides.
            name (Optional[str]): String name of object instance.
            orig_n (int): Soft node budget used when sampling graphs by derivation.
            start_symbol: Left-hand side to begin a derivation from (default: the most frequent one).

        """
        _require_networkx()
        self.name = name
        self.grammar = grammar
        self.mix_p = mix_p
        self.decomp_level = decomp_level
        self.lhs_delta = lhs_delta
        self.orig_n = orig_n
        self.start_symbol = start_symbol

    def __str__(self):

        return (
            "VertexReplacementGrammarDistribution("
            + str(self.grammar)
            + ","
            + str(self.mix_p)
            + ","
            + str(self.decomp_level)
            + ","
            + str(self.lhs_delta)
            + ","
            + str(self.name)
            + ")"
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
        if not self.grammar.rule_dict:
            return None
        return max(self.grammar.rule_dict, key=lambda s: sum(r.frequency for r in self.grammar.rule_dict[s]))

    def density_semantics(self):
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
        is budget-bounded; if the budget truncates the parse forest the result is a variational ELBO over
        the explored derivations -- still >= the Viterbi value. ``best_derivation`` exposes the MAP parse.

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
        return marginal_log_prob(x, self.grammar, start, with_status=with_status)

    # combine list of grammars into singular grammar? need to take multiple sample outputs as input
    def seq_encode(self, x):
        """Encode a sequence of observed graphs for vectorized calls (identity encoding).

        Args:
            x: Sequence of observed graphs (networkx graphs).

        Returns:
            The input sequence unchanged.

        """
        return x

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
        """Create a VertexReplacementGrammarSampler object from the model grammar of this instance.

        Args:
            seed (Optional[int]): Seed for the sampler random generator.

        Returns:
            VertexReplacementGrammarSampler object.

        """
        return VertexReplacementGrammarSampler(
            self.grammar, orig_n=self.orig_n, seed=seed, start_symbol=self.start_symbol
        )

    def estimator(self, pseudo_count=None):
        """Create a VertexReplacementGrammarEstimator object.

        Args:
            pseudo_count (Optional[float]): Added to rule frequencies when estimating.

        Returns:
            VertexReplacementGrammarEstimator object.

        """
        return VertexReplacementGrammarEstimator(
            grammar=self.grammar, start_symbol=self.start_symbol, pseudo_count=pseudo_count, name=self.name
        )

    def dist_to_encoder(self):
        """Returns a VertexReplacementGrammarDataEncoder object for encoding sequences of data."""
        return VertexReplacementGrammarDataEncoder()


class VertexReplacementGrammarSampler(DistributionSampler):
    """VertexReplacementGrammarSampler object for sampling graphs generated from a node-replacement grammar."""

    def __init__(self, grammar, orig_n=100, seed=None, start_symbol=None):
        """VertexReplacementGrammarSampler object.

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
        self.grammar = grammar
        self.orig_n = orig_n
        self.start_symbol = start_symbol
        self.rng = np.random.RandomState(seed)

    def _sample_one(self):
        g, _ = generate_graph(
            rule_dict=self.grammar.rule_dict, target_n=self.orig_n, rng=self.rng, start_symbol=self.start_symbol
        )
        return g

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
        return [self._sample_one() for _ in range(int(size))]

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
                rule_dict=self.grammar.rule_dict, target_n=size, rng=self.rng, start_symbol=self.start_symbol
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


class VertexReplacementGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate Viterbi rule-firing counts: parse each observed graph and tally how often each rule fires.

    This estimates rule FREQUENCIES only. The rule STRUCTURE (which right-hand sides / embeddings exist)
    is supplied via the estimator's ``grammar`` argument; inducing the structure from graphs is a separate
    problem and out of scope here. Counting aligns rules by (symbol, index), which is stable because every
    model in the EM loop is built from the same structure.
    """

    def __init__(self, grammar=None, start_symbol=None, keys=None):
        self.structure = grammar  # rule structure whose frequencies are being estimated
        self.start_symbol = start_symbol
        self.keys = keys
        self.counts = _zeroed_counts(grammar) if grammar is not None else None

    def _parse_model(self, estimate):
        """The (grammar, start_symbol) to parse against: the previous estimate, else the given structure."""
        if estimate is not None:
            return estimate.grammar, estimate._resolve_start()
        if self.structure is None or not self.structure.rule_dict:
            return None, None
        start = self.start_symbol
        if start is None:
            start = max(self.structure.rule_dict, key=lambda s: sum(r.frequency for r in self.structure.rule_dict[s]))
        return self.structure, start

    def update(self, x, weight, estimate):
        """Parse graph ``x`` with the current model and add ``weight`` to every rule its derivation fires."""
        model_grammar, start = self._parse_model(estimate)
        if model_grammar is None or start is None:
            return  # no rule structure to count against
        if self.counts is None:
            self.counts = _zeroed_counts(model_grammar)
        _, derivation = best_derivation(x, model_grammar, start)
        if derivation is None:
            return  # the current model cannot derive x -> it contributes no counts
        position = {id(r): (s, i) for s, rules in model_grammar.rule_dict.items() for i, r in enumerate(rules)}
        for rule in derivation:
            symbol, index = position[id(rule)]
            self.counts.rule_dict[symbol][index].frequency += weight
        self.counts.refresh_rules()

    def initialize(self, x, weight, rng):
        """Initialize from one weighted observed graph (parse with the structure's current frequencies)."""
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Initialize from a sequence of weighted observed graphs."""
        for i in range(len(x)):
            self.initialize(x[i], weights[i], rng)

    def seq_update(self, x, weights, estimate):
        """Parse-and-count a sequence of weighted observed graphs against the previous estimate."""
        for i in range(len(x)):
            self.update(x[i], weights[i], estimate)

    def combine(self, suff_stat):
        """Add another accumulator's rule-firing counts (same structure) position-wise."""
        if suff_stat is None:
            return self
        if self.counts is None:
            self.counts = _zeroed_counts(suff_stat)
        for symbol, rules in suff_stat.rule_dict.items():
            for index, rule in enumerate(rules):
                self.counts.rule_dict[symbol][index].frequency += rule.frequency
        self.counts.refresh_rules()
        return self

    def value(self):
        """Returns the accumulated rule-firing counts as a VertexReplacementGrammar."""
        return self.counts

    def from_value(self, x):
        """Set the accumulated counts from a VertexReplacementGrammar object."""
        self.counts = x
        return self

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
        """Returns a VertexReplacementGrammarDataEncoder object for encoding sequences of data."""
        return VertexReplacementGrammarDataEncoder()


class VertexReplacementGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Creates VertexReplacementGrammarAccumulator objects carrying the rule structure to estimate frequencies for."""

    def __init__(self, grammar=None, start_symbol=None, keys=None):
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.keys = keys

    def make(self):
        """Returns a new VertexReplacementGrammarAccumulator object."""
        return VertexReplacementGrammarAccumulator(grammar=self.grammar, start_symbol=self.start_symbol, keys=self.keys)


class VertexReplacementGrammarEstimator(ParameterEstimator):
    """Estimate a VertexReplacementGrammarDistribution's rule FREQUENCIES from graphs by Viterbi parse-counting.

    The rule structure is supplied via ``grammar`` (e.g. from ``dist.estimator()``): each training graph is
    parsed with the current model and the rules its best derivation fires are counted; frequencies are the
    accumulated counts. Inducing the structure (the right-hand sides / embeddings) from graphs is a separate
    problem and out of scope.
    """

    def __init__(self, grammar=None, start_symbol=None, pseudo_count=None, name=None, keys=None):
        """VertexReplacementGrammarEstimator object.

        Args:
            grammar: VertexReplacementGrammar giving the rule structure whose frequencies are estimated.
            start_symbol: Symbol to start derivations from (default: the most frequent left-hand side).
            pseudo_count (Optional[float]): Added to each rule's counted frequency before normalising.
            name (Optional[str]): String name of object instance.
            keys (Optional[str]): Key for merging sufficient statistics with matching key'd objects.
        """
        _require_networkx()
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self):
        """Returns a VertexReplacementGrammarAccumulatorFactory carrying the rule structure."""
        return VertexReplacementGrammarAccumulatorFactory(
            grammar=self.grammar, start_symbol=self.start_symbol, keys=self.keys
        )

    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        """Build a VertexReplacementGrammarDistribution from accumulated rule-firing counts (frequencies).

        Args:
            nobs (Optional[float]): Weighted number of observations (unused).
            suff_stat: VertexReplacementGrammar of accumulated rule-firing counts.

        Returns:
            VertexReplacementGrammarDistribution object.

        """
        grammar = suff_stat if suff_stat is not None else self.grammar
        if grammar is None:
            raise ValueError(
                "VertexReplacementGrammarEstimator needs a rule structure (grammar=...) to estimate frequencies."
            )
        if self.pseudo_count is not None:
            for rlist in grammar.rule_dict.values():
                for rule in rlist:
                    rule.frequency += self.pseudo_count
        return VertexReplacementGrammarDistribution(grammar, 0.01, start_symbol=self.start_symbol, name=self.name)


class VertexReplacementGrammarDataEncoder(DataSequenceEncoder):
    """VertexReplacementGrammarDataEncoder object for encoding sequences of observed graphs (identity encoding)."""

    def __str__(self):
        """Returns string representation of VertexReplacementGrammarDataEncoder object."""
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
        return x
