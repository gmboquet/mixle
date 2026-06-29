"""Hyperedge-replacement graph grammar (HRG) -- a distribution over networks you can score, fit, and sample.

The second main kind of graph grammar (the other is vertex replacement; see
``vertex_replacement_grammar``). A production ``A -> R`` rewrites a nonterminal HYPEREDGE labelled ``A``
with a ranked tuple of attachment nodes (its tentacles) by a right-hand-side hypergraph ``R`` carrying
an ordered tuple of ``rank(A)`` *external* nodes; the rewrite **fuses** ``R``'s external nodes with the
hyperedge's tentacles (so the gluing is intrinsic -- no embedding relation, unlike NLC). HRGs are
context-free and confluent, with cleaner parsing theory.

Observations are GRAPHS (networkx graphs, all-terminal); the start symbol has rank 0 by default, so a
derivation generates a graph with no boundary. The distribution mirrors ``vertex_replacement_grammar``:

- ``log_density(graph)`` is the MARGINAL likelihood -- the graph is parsed (reduced back to the start
  symbol by un-applying productions) and scored as the log-sum over all derivations (the inside /
  sum-product recursion). Exact when the parse forest is fully explored, a lower bound if the budget
  truncates it, ``-inf`` if the grammar cannot derive the graph. ``best_derivation`` gives the Viterbi parse.
- ``sample()`` runs a real hyperedge-replacement derivation.
- the estimator learns rule FREQUENCIES by Viterbi parse-counting (structure given; induction is out of scope).
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


from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DensitySemantics,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

#: Cap on reduction-step expansions while parsing one graph (HR parsing is NP-hard in general).
_PARSE_BUDGET = 50_000


class Hypergraph:
    """A hypergraph: a networkx graph of terminal (rank-2) edges plus a list of nonterminal hyperedges.

    ``graph`` holds the nodes and terminal edges (with ``label`` / ``node_color`` / ``weight`` /
    ``edge_color`` attributes, as for vertex replacement). ``hyperedges`` is a list of
    ``(label, tuple_of_attachment_nodes)`` -- the nonterminal hyperedges still to be rewritten.
    """

    def __init__(self, graph=None, hyperedges=()):
        _require_networkx()
        self.graph = nx.Graph() if graph is None else graph
        self.hyperedges = [(label, tuple(att)) for label, att in hyperedges]

    def copy(self):
        return Hypergraph(self.graph.copy(), list(self.hyperedges))


class HyperedgeReplacementRule:
    """A production ``lhs -> rhs``: replace a rank-k nonterminal hyperedge by ``rhs``, fusing externals.

    ``external`` is the ordered tuple of ``rhs`` nodes (length = rank of ``lhs``) fused, in order, with
    the rewritten hyperedge's tentacles. ``frequency`` weights the production within its left-hand side.
    """

    __pysp_serializable__ = True

    def __init__(self, lhs, rhs, external, frequency=1.0) -> None:
        _require_networkx()
        self.lhs = lhs
        self.rhs = rhs if isinstance(rhs, Hypergraph) else Hypergraph(rhs, ())
        self.external = tuple(external)
        self.frequency = float(frequency)

    @property
    def rank(self) -> int:
        return len(self.external)

    def __pysp_getstate__(self):
        return {
            "lhs": self.lhs,
            "graph": json_graph.node_link_data(self.rhs.graph, edges="edges"),
            "hyperedges": [[label, list(att)] for label, att in self.rhs.hyperedges],
            "external": list(self.external),
            "frequency": self.frequency,
        }

    def __pysp_setstate__(self, state):
        self.lhs = state["lhs"]
        graph = json_graph.node_link_graph(state["graph"], edges="edges")
        self.rhs = Hypergraph(graph, [(label, tuple(att)) for label, att in state["hyperedges"]])
        self.external = tuple(state["external"])
        self.frequency = float(state["frequency"])

    def __str__(self) -> str:
        return "HyperedgeReplacementRule(lhs=%s, rank=%d, frequency=%s, nodes=%s, hyperedges=%s)" % (
            repr(self.lhs),
            self.rank,
            repr(self.frequency),
            self.rhs.graph.number_of_nodes(),
            len(self.rhs.hyperedges),
        )


class HyperedgeReplacementGrammar:
    """A container of HyperedgeReplacementRule objects keyed by left-hand-side symbol."""

    __pysp_serializable__ = True

    def __init__(self, name="") -> None:
        _require_networkx()
        self.name = name
        self.rule_dict = {}
        self.rule_list = []

    def add_rule(self, rule: HyperedgeReplacementRule) -> None:
        self.rule_dict.setdefault(rule.lhs, []).append(rule)
        self.refresh_rules()

    def refresh_rules(self) -> None:
        self.rule_list = [rule for rules in self.rule_dict.values() for rule in rules]
        self.num_rules = len(self.rule_list)

    def __pysp_getstate__(self):
        return {"name": self.name, "rule_dict": self.rule_dict}

    def __pysp_setstate__(self, state):
        self.name = state["name"]
        self.rule_dict = state["rule_dict"]
        self.refresh_rules()

    def __str__(self) -> str:
        return "HyperedgeReplacementGrammar(name=%s, num_rules=%s)" % (repr(self.name), len(self.rule_list))


# --- derivation (sampling) -------------------------------------------------------------------------
def _rhs_has_nonterminal(rule, rule_dict):
    return any(label in rule_dict for label, _ in rule.rhs.hyperedges)


def _choose_rule(rules, rng, rule_dict, prefer_terminal):
    candidates = [r for r in rules if r.frequency > 0.0]
    if not candidates:
        return None
    if prefer_terminal:
        terminal = [r for r in candidates if not _rhs_has_nonterminal(r, rule_dict)]
        if terminal:
            candidates = terminal
    weights = np.asarray([r.frequency for r in candidates], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def generate_graph(grammar, start_symbol, target_n=100, rng=None, start_rank=0):
    """Generate a graph by a hyperedge-replacement derivation.

    Begins with a single nonterminal hyperedge ``start_symbol`` on ``start_rank`` fresh boundary nodes
    (default 0 -> no boundary). Repeatedly rewrites a nonterminal hyperedge by one of its symbol's rules
    (probability proportional to frequency), fusing the rule's external nodes onto the hyperedge's
    tentacles. ``target_n`` is a soft node budget: once reached the derivation prefers terminal-only
    rules, and any hyperedges left after the step cap are dropped. Returns a networkx graph.
    """
    rng = np.random.RandomState() if rng is None else rng
    if start_symbol not in grammar.rule_dict:
        return nx.Graph()
    target_n = max(1, int(target_n))
    g = nx.Graph()
    counter = [0]

    def fresh():
        counter[0] += 1
        return counter[0] - 1

    boundary = tuple(fresh() for _ in range(start_rank))
    g.add_nodes_from(boundary)
    hyperedges = [(start_symbol, boundary)]
    max_steps = 10 * target_n + 100

    for _ in range(max_steps):
        active = [he for he in hyperedges if he[0] in grammar.rule_dict]
        if not active:
            break
        label, tentacles = active[rng.randint(len(active))]
        rule = _choose_rule(
            grammar.rule_dict[label], rng, grammar.rule_dict, prefer_terminal=g.number_of_nodes() >= target_n
        )
        hyperedges.remove((label, tentacles))
        if rule is None:
            continue
        # map rhs nodes: external -> the hyperedge's tentacles (fusion), internal -> fresh ids
        node_map = {ext: tentacles[i] for i, ext in enumerate(rule.external)}
        for n in rule.rhs.graph.nodes:
            if n not in node_map:
                node_map[n] = fresh()
                g.add_node(node_map[n], **dict(rule.rhs.graph.nodes[n]))
        for a, b, data in rule.rhs.graph.edges(data=True):
            g.add_edge(node_map[a], node_map[b], **dict(data))
        for hl, hatt in rule.rhs.hyperedges:
            hyperedges.append((hl, tuple(node_map[x] for x in hatt)))

    return g


# --- parsing (reduction) ---------------------------------------------------------------------------
def _hr_node_match(host_attrs, pat_attrs):
    # an external right-hand-side node matches any host node (it is just an attachment point); an
    # internal node must match the host terminal node's label/color.
    if pat_attrs.get("_external"):
        return True
    return host_attrs.get("label") == pat_attrs.get("label") and host_attrs.get("node_color", "") == pat_attrs.get(
        "node_color", ""
    )


def _hr_edge_match(host_attrs, pat_attrs):
    return host_attrs.get("edge_color", "") == pat_attrs.get("edge_color", "") and host_attrs.get(
        "weight", 1.0
    ) == pat_attrs.get("weight", 1.0)


def _match_hyperedges(rule, inv, host_hyperedges):
    """Assign each right-hand-side nonterminal hyperedge to a distinct host hyperedge with the same
    label and mapped tentacles. Returns the set of matched host indices, or None."""
    used = set()
    for label, att in rule.rhs.hyperedges:
        target = (label, tuple(inv[x] for x in att))
        found = None
        for i, he in enumerate(host_hyperedges):
            if i not in used and he == target:
                found = i
                break
        if found is None:
            return None
        used.add(found)
    return used


def _try_reduce_hr(hg, rule, inv, external_set):
    """Reverse one production: collapse a matched right-hand-side occurrence to a single nonterminal
    hyperedge. Returns the reduced Hypergraph, or None if the occurrence is not a valid reverse step."""
    host = hg.graph
    rhs = rule.rhs.graph
    internal_host = {inv[n] for n in rhs.nodes if n not in external_set}
    image = {inv[n] for n in rhs.nodes}
    # privacy: internal host nodes carry no terminal edge leaving the occurrence
    for hi in internal_host:
        if any(nb not in image for nb in host.neighbors(hi)):
            return None
    matched = _match_hyperedges(rule, inv, hg.hyperedges)
    if matched is None:
        return None
    # privacy: internal host nodes are tentacles of no UNMATCHED host hyperedge
    for i, (_, att) in enumerate(hg.hyperedges):
        if i not in matched and any(t in internal_host for t in att):
            return None
    remaining_hyperedges = [he for i, he in enumerate(hg.hyperedges) if i not in matched]
    new_hyperedge = (rule.lhs, tuple(inv[e] for e in rule.external))
    if new_hyperedge in remaining_hyperedges:
        # would create a duplicate hyperedge; reject. A rule whose right-hand side has no terminal
        # content (e.g. an external-only "stop" rule) does not reduce the graph when reversed, so it
        # could otherwise be re-applied without end -- forbidding duplicates prunes those spirals.
        return None
    reduced = host.copy()
    for a, b in rhs.edges:  # remove the occurrence's terminal edges (external-external edges included)
        if reduced.has_edge(inv[a], inv[b]):
            reduced.remove_edge(inv[a], inv[b])
    reduced.remove_nodes_from(internal_host)
    return Hypergraph(reduced, [*remaining_hyperedges, new_hyperedge])


def _reductions(hg, grammar):
    """Yield (reduced_hypergraph, rule, symbol_total_frequency) for each valid single reverse step."""
    totals = {s: float(sum(r.frequency for r in rs)) for s, rs in grammar.rule_dict.items()}
    for symbol, rules in grammar.rule_dict.items():
        if totals[symbol] <= 0.0:
            continue
        for rule in rules:
            if rule.frequency <= 0.0:
                continue
            ext = set(rule.external)
            pattern = rule.rhs.graph.copy()
            for n in pattern.nodes:
                pattern.nodes[n]["_external"] = n in ext
            if pattern.number_of_nodes() == 0:
                continue  # empty right-hand side (would need a hyperedge-only match); unsupported
            matcher = iso.GraphMatcher(hg.graph, pattern, node_match=_hr_node_match, edge_match=_hr_edge_match)
            seen = set()
            for mapping in matcher.subgraph_monomorphisms_iter():
                inv = {r: h for h, r in mapping.items()}
                key = (
                    rule.lhs,
                    frozenset(inv[n] for n in rule.rhs.graph.nodes if n not in ext),
                    tuple(inv[e] for e in rule.external),
                )
                if key in seen:
                    continue
                reduced = _try_reduce_hr(hg, rule, inv, ext)
                if reduced is not None:
                    seen.add(key)
                    yield reduced, rule, totals[symbol]


def _is_start(hg, start_symbol):
    return hg.graph.number_of_nodes() == 0 and hg.hyperedges == [(start_symbol, ())]


def best_derivation(graph, grammar, start_symbol, budget=_PARSE_BUDGET):
    """Best (Viterbi) hyperedge-replacement derivation of a graph: (log_prob, [rules]) or (-inf, None)."""
    remaining = [budget]

    def solve(hg, depth):
        if _is_start(hg, start_symbol):
            return 0.0, []
        if depth <= 0 or remaining[0] <= 0:
            return float("-inf"), None
        best_lp, best_seq = float("-inf"), None
        for reduced, rule, total in _reductions(hg, grammar):
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
        return float("-inf"), None
    return solve(Hypergraph(graph.copy(), []), 3 * graph.number_of_nodes() + 10)


def marginal_log_prob(graph, grammar, start_symbol, budget=_PARSE_BUDGET, with_status=False):
    """Marginal log-likelihood: log-sum over ALL hyperedge-replacement derivations that yield the graph.

    Exact when the parse forest is fully explored; a variational lower bound (ELBO) if the budget/depth
    cap truncates it. ``with_status`` returns ``(value, exact)`` with ``exact`` False iff a cap was hit.
    """
    remaining = [budget]
    truncated = [False]

    def inside(hg, depth):
        if _is_start(hg, start_symbol):
            return 0.0
        if depth <= 0 or remaining[0] <= 0:
            truncated[0] = True
            return float("-inf")
        terms = []
        for reduced, rule, total in _reductions(hg, grammar):
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
        return high + float(np.log(sum(np.exp(t - high) for t in terms)))

    value = (
        float("-inf")
        if graph.number_of_nodes() == 0
        else inside(Hypergraph(graph.copy(), []), 3 * graph.number_of_nodes() + 10)
    )
    return (value, not truncated[0]) if with_status else value


def _zeroed_counts(grammar):
    """A copy of ``grammar``'s rule structure with every frequency set to 0 (a counts accumulator)."""
    counts = HyperedgeReplacementGrammar(grammar.name)
    for symbol, rules in grammar.rule_dict.items():
        counts.rule_dict[symbol] = [HyperedgeReplacementRule(r.lhs, r.rhs.copy(), r.external, 0.0) for r in rules]
    counts.refresh_rules()
    return counts


# --- distribution / sampler / estimator (mirrors vertex_replacement_grammar) -----------------------
class HyperedgeReplacementGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A distribution over GRAPHS parameterised by a hyperedge-replacement grammar.

    ``log_density(graph)`` is the marginal likelihood (sum over derivations, by parsing); ``sample()``
    emits graphs by derivation; the estimator learns rule frequencies by Viterbi parse-counting.
    """

    def __init__(self, grammar, start_symbol=None, orig_n=100, name=None):
        _require_networkx()
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.orig_n = orig_n
        self.name = name

    def __str__(self):
        return "HyperedgeReplacementGrammarDistribution(%s, start_symbol=%s)" % (self.grammar, repr(self.start_symbol))

    def _resolve_start(self):
        if self.start_symbol is not None:
            return self.start_symbol
        if not self.grammar.rule_dict:
            return None
        return max(self.grammar.rule_dict, key=lambda s: sum(r.frequency for r in self.grammar.rule_dict[s]))

    def density_semantics(self):
        return DensitySemantics.LOWER_BOUND  # exact unless the parse budget truncates; see log_density(with_status)

    def density(self, x):
        return float(np.exp(self.log_density(x)))

    def log_density(self, x, with_status=False):
        """Marginal log-likelihood of graph ``x`` (see ``marginal_log_prob``). ``with_status`` -> (value, exact)."""
        start = self._resolve_start()
        if start is None:
            return (float("-inf"), True) if with_status else float("-inf")
        return marginal_log_prob(x, self.grammar, start, with_status=with_status)

    def seq_encode(self, x):
        return x

    def seq_log_density(self, x, with_status=False):
        if not with_status:
            return np.asarray([self.log_density(xx) for xx in x])
        pairs = [self.log_density(xx, with_status=True) for xx in x]
        return np.asarray([v for v, _ in pairs]), np.asarray([e for _, e in pairs], dtype=bool)

    def sampler(self, seed=None):
        return HyperedgeReplacementGrammarSampler(self.grammar, self.start_symbol, self.orig_n, seed)

    def estimator(self, pseudo_count=None):
        return HyperedgeReplacementGrammarEstimator(
            grammar=self.grammar, start_symbol=self.start_symbol, pseudo_count=pseudo_count, name=self.name
        )

    def dist_to_encoder(self):
        return HyperedgeReplacementGrammarDataEncoder()


class HyperedgeReplacementGrammarSampler(DistributionSampler):
    """Sample graphs from a hyperedge-replacement grammar by derivation."""

    def __init__(self, grammar, start_symbol=None, orig_n=100, seed=None):
        self.grammar = grammar
        self.start_symbol = (
            start_symbol
            if start_symbol is not None
            else (
                max(grammar.rule_dict, key=lambda s: sum(r.frequency for r in grammar.rule_dict[s]))
                if grammar.rule_dict
                else None
            )
        )
        self.orig_n = orig_n
        self.rng = np.random.RandomState(seed)

    def _one(self):
        return generate_graph(self.grammar, self.start_symbol, target_n=self.orig_n, rng=self.rng)

    def sample(self, size=None, *, batched=True):
        if size is None:
            return self._one()
        return [self._one() for _ in range(int(size))]


class HyperedgeReplacementGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate Viterbi rule-firing counts: parse each graph and tally how often each rule fires."""

    def __init__(self, grammar=None, start_symbol=None, keys=None):
        self.structure = grammar
        self.start_symbol = start_symbol
        self.keys = keys
        self.counts = _zeroed_counts(grammar) if grammar is not None else None

    def _parse_model(self, estimate):
        if estimate is not None:
            return estimate.grammar, estimate._resolve_start()
        if self.structure is None or not self.structure.rule_dict:
            return None, None
        start = self.start_symbol
        if start is None:
            start = max(self.structure.rule_dict, key=lambda s: sum(r.frequency for r in self.structure.rule_dict[s]))
        return self.structure, start

    def update(self, x, weight, estimate):
        model_grammar, start = self._parse_model(estimate)
        if model_grammar is None or start is None:
            return
        if self.counts is None:
            self.counts = _zeroed_counts(model_grammar)
        _, derivation = best_derivation(x, model_grammar, start)
        if derivation is None:
            return
        position = {id(r): (s, i) for s, rules in model_grammar.rule_dict.items() for i, r in enumerate(rules)}
        for rule in derivation:
            symbol, index = position[id(rule)]
            self.counts.rule_dict[symbol][index].frequency += weight
        self.counts.refresh_rules()

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        for i in range(len(x)):
            self.initialize(x[i], weights[i], rng)

    def seq_update(self, x, weights, estimate):
        for i in range(len(x)):
            self.update(x[i], weights[i], estimate)

    def combine(self, suff_stat):
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
        return self.counts

    def from_value(self, x):
        self.counts = x
        return self

    def key_merge(self, stats_dict):
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict):
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self):
        return HyperedgeReplacementGrammarDataEncoder()


class HyperedgeReplacementGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Creates accumulators carrying the rule structure whose frequencies are estimated."""

    def __init__(self, grammar=None, start_symbol=None, keys=None):
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.keys = keys

    def make(self):
        return HyperedgeReplacementGrammarAccumulator(
            grammar=self.grammar, start_symbol=self.start_symbol, keys=self.keys
        )


class HyperedgeReplacementGrammarEstimator(ParameterEstimator):
    """Estimate rule FREQUENCIES from graphs by Viterbi parse-counting (the structure is given)."""

    def __init__(self, grammar=None, start_symbol=None, pseudo_count=None, name=None, keys=None):
        _require_networkx()
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self):
        return HyperedgeReplacementGrammarAccumulatorFactory(
            grammar=self.grammar, start_symbol=self.start_symbol, keys=self.keys
        )

    def estimate(self, nobs, suff_stat):
        grammar = suff_stat if suff_stat is not None else self.grammar
        if grammar is None:
            raise ValueError("HyperedgeReplacementGrammarEstimator needs a rule structure (grammar=...).")
        if self.pseudo_count is not None:
            for rules in grammar.rule_dict.values():
                for rule in rules:
                    rule.frequency += self.pseudo_count
        return HyperedgeReplacementGrammarDistribution(grammar, start_symbol=self.start_symbol, name=self.name)


class HyperedgeReplacementGrammarDataEncoder(DataSequenceEncoder):
    """Identity encoder for sequences of observed graphs."""

    def __str__(self):
        return "HyperedgeReplacementGrammarDataEncoder"

    def __eq__(self, other):
        return isinstance(other, HyperedgeReplacementGrammarDataEncoder)

    def seq_encode(self, x):
        return x
