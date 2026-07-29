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


def _validate_simple_graph(graph, *, context):
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError(f"{context} must be an undirected simple networkx Graph.")
    for _, _, attrs in graph.edges(data=True):
        try:
            finite_weight = np.isfinite(float(attrs.get("weight", 1.0)))
        except (TypeError, ValueError):
            finite_weight = False
        if not finite_weight:
            raise ValueError(f"{context} edge weights must be finite numeric values.")
    return graph


class Hypergraph:
    """A hypergraph: a networkx graph of terminal (rank-2) edges plus a list of nonterminal hyperedges.

    ``graph`` holds the nodes and terminal edges (with ``label`` / ``node_color`` / ``weight`` /
    ``edge_color`` attributes, as for vertex replacement). ``hyperedges`` is a list of
    ``(label, tuple_of_attachment_nodes)`` -- the nonterminal hyperedges still to be rewritten.
    """

    def __init__(self, graph=None, hyperedges=()):
        _require_networkx()
        source = nx.Graph() if graph is None else _validate_simple_graph(graph, context="hypergraph terminal graph")
        self.graph = source.copy()
        checked_hyperedges = []
        for label, attachments in hyperedges:
            if label is None:
                raise ValueError("hyperedge labels cannot be None.")
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError("hyperedge labels must be hashable.") from exc
            attachment_tuple = tuple(attachments)
            if len(set(attachment_tuple)) != len(attachment_tuple):
                raise ValueError("hyperedge attachment nodes must be distinct.")
            if any(node not in self.graph for node in attachment_tuple):
                raise ValueError("hyperedge attachments must name nodes in the terminal graph.")
            checked_hyperedges.append((label, attachment_tuple))
        if len(set(checked_hyperedges)) != len(checked_hyperedges):
            raise ValueError("duplicate nonterminal hyperedges are not supported.")
        self.hyperedges = checked_hyperedges

    def copy(self):
        """Return a structural copy of the terminal graph and nonterminal hyperedges."""
        return Hypergraph(self.graph.copy(), list(self.hyperedges))


class HyperedgeReplacementRule:
    """A production ``lhs -> rhs``: replace a rank-k nonterminal hyperedge by ``rhs``, fusing externals.

    ``external`` is the ordered tuple of ``rhs`` nodes (length = rank of ``lhs``) fused, in order, with
    the rewritten hyperedge's tentacles. ``frequency`` weights the production within its left-hand side.
    """

    __pysp_serializable__ = True

    def __init__(self, lhs, rhs, external, frequency=1.0) -> None:
        _require_networkx()
        if lhs is None:
            raise ValueError("HRG rule lhs cannot be None.")
        try:
            hash(lhs)
        except TypeError as exc:
            raise ValueError("HRG rule lhs must be hashable.") from exc
        checked_frequency = float(frequency)
        if not np.isfinite(checked_frequency) or checked_frequency < 0.0:
            raise ValueError("HRG rule frequency must be finite and non-negative.")
        owned_rhs = rhs.copy() if isinstance(rhs, Hypergraph) else Hypergraph(rhs, ())
        external_tuple = tuple(external)
        if len(set(external_tuple)) != len(external_tuple):
            raise ValueError("HRG external nodes must be distinct.")
        if any(node not in owned_rhs.graph for node in external_tuple):
            raise ValueError("HRG external nodes must name RHS graph nodes.")
        external_set = set(external_tuple)
        for node, attrs in owned_rhs.graph.nodes(data=True):
            if node in external_set:
                if attrs:
                    raise ValueError("HRG external RHS nodes cannot carry terminal attributes.")
            elif "label" not in attrs:
                raise ValueError("every internal HRG RHS node must define a label.")
        if owned_rhs.graph.number_of_nodes() == 0 and not owned_rhs.hyperedges:
            raise ValueError("empty HRG right-hand sides are not supported.")
        self.lhs = lhs
        self.rhs = owned_rhs
        self.external = external_tuple
        self.frequency = checked_frequency

    @property
    def rank(self) -> int:
        """Return the arity of the left-hand-side nonterminal hyperedge."""
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
        graph = json_graph.node_link_graph(state["graph"], edges="edges")
        restored = HyperedgeReplacementRule(
            state["lhs"],
            Hypergraph(graph, [(label, tuple(att)) for label, att in state["hyperedges"]]),
            state["external"],
            state["frequency"],
        )
        self.__dict__.update(restored.__dict__)

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
        if not isinstance(name, str):
            raise ValueError("HRG name must be a string.")
        self.name = name
        self.rule_dict = {}
        self.rule_list = []

    def add_rule(self, rule: HyperedgeReplacementRule) -> None:
        """Add a production rule and refresh the flattened rule list."""
        if not isinstance(rule, HyperedgeReplacementRule):
            raise ValueError("HyperedgeReplacementGrammar accepts only HRG rules.")
        self.rule_dict.setdefault(rule.lhs, []).append(rule)
        self.refresh_rules()

    def refresh_rules(self) -> None:
        """Rebuild the flattened rule list and cached rule count."""
        self.rule_list = [rule for rules in self.rule_dict.values() for rule in rules]
        self.num_rules = len(self.rule_list)

    def __pysp_getstate__(self):
        return {"name": self.name, "rule_dict": self.rule_dict}

    def __pysp_setstate__(self, state):
        restored = HyperedgeReplacementGrammar(state["name"])
        for symbol, rules in state["rule_dict"].items():
            for rule in rules:
                if rule.lhs != symbol:
                    raise ValueError("serialized HRG rule_dict key does not match rule lhs.")
                restored.add_rule(_copy_rule(rule))
        self.__dict__.update(restored.__dict__)

    def __str__(self) -> str:
        return "HyperedgeReplacementGrammar(name=%s, num_rules=%s)" % (repr(self.name), len(self.rule_list))


def _copy_rule(rule):
    return HyperedgeReplacementRule(rule.lhs, rule.rhs.copy(), rule.external, rule.frequency)


def _copy_grammar(grammar):
    if not isinstance(grammar, HyperedgeReplacementGrammar):
        raise ValueError("grammar must be a HyperedgeReplacementGrammar.")
    copied = HyperedgeReplacementGrammar(grammar.name)
    for symbol, rules in grammar.rule_dict.items():
        for rule in rules:
            if rule.lhs != symbol:
                raise ValueError("grammar rule_dict keys must agree with rule lhs.")
            copied.add_rule(_copy_rule(rule))
    return copied


def _validate_grammar(grammar, *, start_symbol=None):
    copied = _copy_grammar(grammar)
    if not copied.rule_dict:
        raise ValueError("HRG must contain at least one rule.")
    ranks = {}
    for symbol, rules in copied.rule_dict.items():
        symbol_ranks = {rule.rank for rule in rules}
        if len(symbol_ranks) != 1:
            raise ValueError(f"all HRG rules for {symbol!r} must have the same rank.")
        ranks[symbol] = next(iter(symbol_ranks))
        total = float(sum(rule.frequency for rule in rules))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"rules for HRG symbol {symbol!r} must have positive finite total frequency.")
    for rules in copied.rule_dict.values():
        for rule in rules:
            for label, attachments in rule.rhs.hyperedges:
                if label not in ranks:
                    raise ValueError(f"RHS references HRG symbol {label!r} with no rules.")
                if len(attachments) != ranks[label]:
                    raise ValueError(f"RHS hyperedge {label!r} has the wrong attachment rank.")
    resolved = (
        start_symbol
        if start_symbol is not None
        else max(copied.rule_dict, key=lambda symbol: sum(r.frequency for r in copied.rule_dict[symbol]))
    )
    if resolved not in ranks:
        raise ValueError("start_symbol must identify an HRG left-hand side.")
    if ranks[resolved] != 0:
        raise ValueError("graph-valued HRG distributions require a rank-zero start symbol.")
    return copied


@dataclass(frozen=True)
class HRGSamplingReceipt:
    completed: bool
    steps: int
    max_steps: int
    active_hyperedges: int
    node_count: int


class HRGSamplingTruncated(RuntimeError):
    def __init__(self, receipt: HRGSamplingReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            f"HRG sample truncated after {receipt.steps}/{receipt.max_steps} steps "
            f"with {receipt.active_hyperedges} active hyperedges"
        )


@dataclass(frozen=True)
class HRGParseReceipt:
    exact: bool
    derivable: bool
    expansions: int
    budget: int


class HRGParseTruncated(RuntimeError):
    def __init__(self, receipt: HRGParseReceipt) -> None:
        self.receipt = receipt
        super().__init__(f"HRG parse exhausted its {receipt.budget}-expansion budget")


# --- derivation (sampling) -------------------------------------------------------------------------
def _rhs_has_nonterminal(rule, rule_dict):
    return any(label in rule_dict for label, _ in rule.rhs.hyperedges)


def _choose_rule(rules, rng):
    candidates = [r for r in rules if r.frequency > 0.0]
    if not candidates:
        return None
    weights = np.asarray([r.frequency for r in candidates], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def generate_graph(grammar, start_symbol, target_n=100, rng=None, start_rank=0, *, with_receipt=False):
    """Generate a graph by a hyperedge-replacement derivation.

    Begins with a single nonterminal hyperedge ``start_symbol`` on ``start_rank`` fresh boundary nodes
    (default 0 -> no boundary). Repeatedly rewrites a nonterminal hyperedge by one of its symbol's rules
    (probability proportional to frequency), fusing the rule's external nodes onto the hyperedge's
    tentacles. ``target_n`` controls only the rewrite work budget and never
    changes the production law. Budget exhaustion raises
    :class:`HRGSamplingTruncated` with a receipt. Graph-valued sampling supports
    only a rank-zero start. Returns a networkx graph, or ``(graph, receipt)``.
    """
    rng = np.random.RandomState() if rng is None else rng
    grammar = _validate_grammar(grammar, start_symbol=start_symbol)
    checked_start_rank = _exact_nonnegative_int(start_rank, name="start_rank")
    if checked_start_rank != 0:
        raise ValueError("graph-valued HRG sampling requires start_rank=0.")
    target_n = _exact_positive_int(target_n, name="target_n")
    g = nx.Graph()
    counter = [0]

    def fresh():
        counter[0] += 1
        return counter[0] - 1

    boundary = ()
    g.add_nodes_from(boundary)
    hyperedges = [(start_symbol, boundary)]
    max_steps = 10 * target_n + 100

    steps = 0
    for _ in range(max_steps):
        active = [he for he in hyperedges if he[0] in grammar.rule_dict]
        if not active:
            break
        label, tentacles = active[rng.randint(len(active))]
        rule = _choose_rule(grammar.rule_dict[label], rng)
        hyperedges.remove((label, tentacles))
        if rule is None:
            raise ValueError(f"HRG symbol {label!r} has no positive-frequency production.")
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
        steps += 1

    active = [he for he in hyperedges if he[0] in grammar.rule_dict]
    receipt = HRGSamplingReceipt(
        completed=not active,
        steps=steps,
        max_steps=max_steps,
        active_hyperedges=len(active),
        node_count=g.number_of_nodes(),
    )
    if active:
        raise HRGSamplingTruncated(receipt)
    return (g, receipt) if with_receipt else g


# --- parsing (reduction) ---------------------------------------------------------------------------
def _hr_node_match(host_attrs, pat_attrs):
    # an external right-hand-side node matches any host node (it is just an attachment point); an
    # internal node must match the host terminal node's label/color.
    if pat_attrs.get("_external"):
        return True
    expected = dict(pat_attrs)
    expected.pop("_external", None)
    return dict(host_attrs) == expected


def _hr_edge_match(host_attrs, pat_attrs):
    return dict(host_attrs) == dict(pat_attrs)


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
            for mapping in matcher.subgraph_isomorphisms_iter():
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


def _active_hyperedge_count(hg, grammar):
    return sum(label in grammar.rule_dict for label, _ in hg.hyperedges)


def _freeze_state_value(value):
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((_freeze_state_value(key), _freeze_state_value(item)) for key, item in value.items()),
                key=repr,
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_state_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_state_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _hypergraph_state_key(hg, depth):
    nodes = frozenset((node, _freeze_state_value(dict(attrs))) for node, attrs in hg.graph.nodes(data=True))
    edges = frozenset(
        (frozenset((left, right)), _freeze_state_value(dict(attrs))) for left, right, attrs in hg.graph.edges(data=True)
    )
    return nodes, edges, frozenset(hg.hyperedges), depth


def best_derivation(graph, grammar, start_symbol, budget=_PARSE_BUDGET, with_status=False):
    """Best (Viterbi) hyperedge-replacement derivation of a graph: (log_prob, [rules]) or (-inf, None)."""
    checked_budget = _exact_positive_int(budget, name="budget")
    remaining = [checked_budget]
    truncated = [False]
    memo = {}

    def solve(hg, depth):
        key = _hypergraph_state_key(hg, depth)
        if key in memo:
            return memo[key]
        if _is_start(hg, start_symbol):
            return 0.0, []
        if depth <= 0 or remaining[0] <= 0:
            truncated[0] = True
            return float("-inf"), None
        best_lp, best_seq = float("-inf"), None
        for reduced, rule, total in _reductions(hg, grammar):
            remaining[0] -= 1
            if remaining[0] <= 0:
                truncated[0] = True
                break
            sub_lp, sub_seq = solve(reduced, depth - 1)
            if sub_seq is not None:
                active_sites = _active_hyperedge_count(reduced, grammar)
                if active_sites < 1:
                    continue
                lp = float(np.log(rule.frequency / total)) - float(np.log(active_sites)) + sub_lp
                if lp > best_lp:
                    best_lp, best_seq = lp, [rule, *sub_seq]
        memo[key] = (best_lp, best_seq)
        return memo[key]

    if graph.number_of_nodes() == 0:
        result = (float("-inf"), None)
    else:
        result = solve(Hypergraph(graph.copy(), []), 3 * graph.number_of_nodes() + 10)
    receipt = HRGParseReceipt(
        exact=not truncated[0],
        derivable=result[1] is not None,
        expansions=checked_budget - remaining[0],
        budget=checked_budget,
    )
    return (*result, receipt) if with_status else result


def marginal_log_prob(graph, grammar, start_symbol, budget=_PARSE_BUDGET, with_status=False):
    """Marginal log-likelihood: log-sum over ALL hyperedge-replacement derivations that yield the graph.

    Exact when the parse forest is fully explored; a certified partial-mass lower bound if the budget/depth
    cap truncates it. ``with_status`` returns ``(value, exact)`` with ``exact`` False iff a cap was hit.
    """
    checked_budget = _exact_positive_int(budget, name="budget")
    remaining = [checked_budget]
    truncated = [False]
    memo = {}

    def inside(hg, depth):
        key = _hypergraph_state_key(hg, depth)
        if key in memo:
            return memo[key]
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
                active_sites = _active_hyperedge_count(reduced, grammar)
                if active_sites > 0:
                    terms.append(float(np.log(rule.frequency / total)) - float(np.log(active_sites)) + sub)
        if not terms:
            memo[key] = float("-inf")
            return memo[key]
        high = max(terms)
        memo[key] = high + float(np.log(sum(np.exp(t - high) for t in terms)))
        return memo[key]

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


def _validate_terminal_graph(graph):
    _validate_simple_graph(graph, context="HRG observation")
    for _, attrs in graph.nodes(data=True):
        if "label" not in attrs:
            raise ValueError("every HRG observation node must define a label.")
    return graph


class HyperedgeReplacementGrammarStatistics(NamedTuple):
    schema_version: int
    counts: HyperedgeReplacementGrammar
    accepted_weight: float
    rejected_weight: float
    truncated_weight: float


@dataclass(frozen=True)
class HRGFitReceipt:
    accepted_weight: float
    rejected_weight: float
    truncated_weight: float


def _rules_share_structure(left, right):
    if left.lhs != right.lhs or left.rank != right.rank:
        return False
    matcher = iso.GraphMatcher(
        left.rhs.graph,
        right.rhs.graph,
        node_match=lambda a, b: dict(a) == dict(b),
        edge_match=lambda a, b: dict(a) == dict(b),
    )
    right_hyperedges = set(right.rhs.hyperedges)
    for mapping in matcher.isomorphisms_iter():
        if tuple(mapping[node] for node in left.external) != right.external:
            continue
        mapped_hyperedges = {
            (label, tuple(mapping[node] for node in attachments)) for label, attachments in left.rhs.hyperedges
        }
        if mapped_hyperedges == right_hyperedges:
            return True
    return False


def _grammars_share_structure(left, right):
    if (
        not isinstance(left, HyperedgeReplacementGrammar)
        or not isinstance(right, HyperedgeReplacementGrammar)
        or list(left.rule_dict) != list(right.rule_dict)
    ):
        return False
    return all(
        len(left.rule_dict[symbol]) == len(right.rule_dict[symbol])
        and all(
            _rules_share_structure(left_rule, right_rule)
            for left_rule, right_rule in zip(left.rule_dict[symbol], right.rule_dict[symbol])
        )
        for symbol in left.rule_dict
    )


def _validate_statistics(value, structure):
    if not isinstance(value, HyperedgeReplacementGrammarStatistics) or value.schema_version != 1:
        raise ValueError("HRG statistics must use schema version 1.")
    if not _grammars_share_structure(value.counts, structure):
        raise ValueError("HRG statistics do not match the estimator's rule structure.")
    weights = tuple(float(v) for v in (value.accepted_weight, value.rejected_weight, value.truncated_weight))
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("HRG receipt weights must be finite and non-negative.")
    counts = _copy_grammar(value.counts)
    for rules in counts.rule_dict.values():
        for rule in rules:
            if not np.isfinite(rule.frequency) or rule.frequency < 0.0:
                raise ValueError("HRG rule counts must be finite and non-negative.")
    return HyperedgeReplacementGrammarStatistics(1, counts, *weights)


# --- distribution / sampler / estimator (mirrors vertex_replacement_grammar) -----------------------
class HyperedgeReplacementGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A distribution over GRAPHS parameterised by a hyperedge-replacement grammar.

    ``log_density(graph)`` is the marginal likelihood (sum over derivations, by parsing); ``sample()``
    emits graphs by derivation; the estimator learns rule frequencies by Viterbi parse-counting.
    """

    def __init__(self, grammar, start_symbol=None, orig_n=100, name=None, keys=None):
        _require_networkx()
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        if keys is not None and not isinstance(keys, str):
            raise ValueError("keys must be a string or None.")
        self._grammar = _validate_grammar(grammar, start_symbol=start_symbol)
        self.start_symbol = start_symbol
        self.orig_n = _exact_positive_int(orig_n, name="orig_n")
        self.name = name
        self.keys = keys

    @property
    def grammar(self):
        return _copy_grammar(self._grammar)

    def __pysp_getstate__(self):
        return {
            "grammar": self._grammar,
            "start_symbol": self.start_symbol,
            "orig_n": self.orig_n,
            "name": self.name,
            "keys": self.keys,
        }

    def __pysp_setstate__(self, state):
        restored = HyperedgeReplacementGrammarDistribution(**state)
        self.__dict__.update(restored.__dict__)

    def __str__(self):
        return (
            "HyperedgeReplacementGrammarDistribution("
            f"grammar={self._grammar}, start_symbol={self.start_symbol!r}, orig_n={self.orig_n!r}, "
            f"name={self.name!r}, keys={self.keys!r})"
        )

    def _resolve_start(self):
        if self.start_symbol is not None:
            return self.start_symbol
        if not self._grammar.rule_dict:
            return None
        return max(self._grammar.rule_dict, key=lambda s: sum(r.frequency for r in self._grammar.rule_dict[s]))

    def density_semantics(self):
        """Return that graph densities are lower bounds when parsing is budget-truncated."""
        return DensitySemantics.LOWER_BOUND  # exact unless the parse budget truncates; see log_density(with_status)

    def density(self, x):
        """Return the marginal probability of a graph under the grammar."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x, with_status=False):
        """Marginal log-likelihood of graph ``x`` (see ``marginal_log_prob``). ``with_status`` -> (value, exact)."""
        start = self._resolve_start()
        if start is None:
            return (float("-inf"), True) if with_status else float("-inf")
        _validate_terminal_graph(x)
        return marginal_log_prob(x, self._grammar, start, with_status=with_status)

    def seq_encode(self, x):
        """Return graph observations unchanged for sequence scoring."""
        return tuple(_validate_terminal_graph(graph) for graph in x)

    def seq_log_density(self, x, with_status=False):
        """Return vectorized graph log-likelihoods, optionally with exactness flags."""
        if not with_status:
            return np.asarray([self.log_density(xx) for xx in x])
        pairs = [self.log_density(xx, with_status=True) for xx in x]
        return np.asarray([v for v, _ in pairs]), np.asarray([e for _, e in pairs], dtype=bool)

    def sampler(self, seed=None):
        """Return a derivation sampler for this grammar distribution."""
        return HyperedgeReplacementGrammarSampler(self._grammar, self.start_symbol, self.orig_n, seed)

    def estimator(self, pseudo_count=None):
        """Return a Viterbi parse-count estimator for this grammar's rule frequencies."""
        return HyperedgeReplacementGrammarEstimator(
            grammar=self._grammar,
            start_symbol=self.start_symbol,
            pseudo_count=pseudo_count,
            name=self.name,
            orig_n=self.orig_n,
            keys=self.keys,
        )

    def dist_to_encoder(self):
        """Return the identity graph encoder used by vectorized methods."""
        return HyperedgeReplacementGrammarDataEncoder()


class HyperedgeReplacementGrammarSampler(DistributionSampler):
    """Sample graphs from a hyperedge-replacement grammar by derivation."""

    def __init__(self, grammar, start_symbol=None, orig_n=100, seed=None):
        self._grammar = _validate_grammar(grammar, start_symbol=start_symbol)
        self.start_symbol = (
            start_symbol
            if start_symbol is not None
            else (
                max(self._grammar.rule_dict, key=lambda s: sum(r.frequency for r in self._grammar.rule_dict[s]))
                if self._grammar.rule_dict
                else None
            )
        )
        self.orig_n = _exact_positive_int(orig_n, name="orig_n")
        self.rng = np.random.RandomState(seed)

    @property
    def grammar(self):
        return _copy_grammar(self._grammar)

    def _one(self):
        return generate_graph(self._grammar, self.start_symbol, target_n=self.orig_n, rng=self.rng)

    def sample_with_receipt(self):
        """Draw one exact graph together with its completion receipt."""
        return generate_graph(
            self._grammar,
            self.start_symbol,
            target_n=self.orig_n,
            rng=self.rng,
            with_receipt=True,
        )

    def sample(self, size=None, *, batched=True):
        """Draw one graph or a list of graphs by HRG derivation."""
        if size is None:
            return self._one()
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self._one() for _ in range(sample_size)]


class HyperedgeReplacementGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate Viterbi rule-firing counts: parse each graph and tally how often each rule fires."""

    def __init__(self, grammar, start_symbol=None, keys=None):
        self.structure = _validate_grammar(grammar, start_symbol=start_symbol)
        self.start_symbol = start_symbol
        self.keys = keys
        self.counts = _zeroed_counts(self.structure)
        self.accepted_weight = 0.0
        self.rejected_weight = 0.0
        self.truncated_weight = 0.0

    def _parse_model(self, estimate):
        if estimate is not None:
            return estimate._grammar, estimate._resolve_start()
        start = self.start_symbol
        if start is None:
            start = max(self.structure.rule_dict, key=lambda s: sum(r.frequency for r in self.structure.rule_dict[s]))
        return self.structure, start

    def update(self, x, weight, estimate):
        """Update Viterbi rule counts from one observed graph."""
        model_grammar, start = self._parse_model(estimate)
        _validate_terminal_graph(x)
        checked_weight = float(weight)
        if not np.isfinite(checked_weight) or checked_weight < 0.0:
            raise ValueError("HRG observation weight must be finite and non-negative.")
        if checked_weight == 0.0:
            return
        _, derivation, receipt = best_derivation(x, model_grammar, start, with_status=True)
        if not receipt.exact:
            self.truncated_weight += checked_weight
            raise HRGParseTruncated(receipt)
        if derivation is None:
            self.rejected_weight += checked_weight
            raise ValueError("observed graph is outside the HRG's support.")
        position = {id(r): (s, i) for s, rules in model_grammar.rule_dict.items() for i, r in enumerate(rules)}
        for rule in derivation:
            symbol, index = position[id(rule)]
            self.counts.rule_dict[symbol][index].frequency += checked_weight
        self.accepted_weight += checked_weight
        self.counts.refresh_rules()

    def initialize(self, x, weight, rng):
        """Initialize rule counts from one observed graph."""
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Initialize rule counts from a batch of observed graphs."""
        self.seq_update(x, weights, None)

    def seq_update(self, x, weights, estimate):
        """Update Viterbi rule counts from a batch of observed graphs."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the graph batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = HyperedgeReplacementGrammarAccumulator(self.structure, self.start_symbol)
        for graph, weight in zip(x, checked_weights):
            pending.update(graph, float(weight), estimate)
        self.combine(pending.value())

    def combine(self, suff_stat):
        """Merge rule-frequency counts from another grammar accumulator value."""
        checked = _validate_statistics(suff_stat, self.structure)
        for symbol, rules in checked.counts.rule_dict.items():
            for index, rule in enumerate(rules):
                self.counts.rule_dict[symbol][index].frequency += rule.frequency
        self.accepted_weight += checked.accepted_weight
        self.rejected_weight += checked.rejected_weight
        self.truncated_weight += checked.truncated_weight
        self.counts.refresh_rules()
        return self

    def value(self):
        """Return the grammar-shaped rule-count accumulator."""
        return HyperedgeReplacementGrammarStatistics(
            1,
            _copy_grammar(self.counts),
            self.accepted_weight,
            self.rejected_weight,
            self.truncated_weight,
        )

    def from_value(self, x):
        """Restore grammar-shaped rule counts from ``value`` output."""
        checked = _validate_statistics(x, self.structure)
        self.counts = checked.counts
        self.accepted_weight = checked.accepted_weight
        self.rejected_weight = checked.rejected_weight
        self.truncated_weight = checked.truncated_weight
        return self

    def receipt(self):
        """Return accepted/rejected/truncated fit-weight accounting."""
        return HRGFitReceipt(self.accepted_weight, self.rejected_weight, self.truncated_weight)

    def key_merge(self, stats_dict):
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict):
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self):
        """Return the graph encoder compatible with this accumulator."""
        return HyperedgeReplacementGrammarDataEncoder()


class HyperedgeReplacementGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Creates accumulators carrying the rule structure whose frequencies are estimated."""

    def __init__(self, grammar, start_symbol=None, keys=None):
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.keys = keys

    def make(self):
        """Create an empty HRG rule-count accumulator."""
        return HyperedgeReplacementGrammarAccumulator(
            grammar=self.grammar, start_symbol=self.start_symbol, keys=self.keys
        )


class HyperedgeReplacementGrammarEstimator(ParameterEstimator):
    """Estimate rule FREQUENCIES from graphs by Viterbi parse-counting (the structure is given)."""

    def __init__(self, grammar, start_symbol=None, pseudo_count=None, name=None, keys=None, orig_n=100):
        _require_networkx()
        self._grammar = _validate_grammar(grammar, start_symbol=start_symbol)
        self.start_symbol = start_symbol
        self.pseudo_count = None if pseudo_count is None else float(pseudo_count)
        if self.pseudo_count is not None and (not np.isfinite(self.pseudo_count) or self.pseudo_count < 0.0):
            raise ValueError("pseudo_count must be finite and non-negative.")
        self.orig_n = _exact_positive_int(orig_n, name="orig_n")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        if keys is not None and not isinstance(keys, str):
            raise ValueError("keys must be a string or None.")
        self.name = name
        self.keys = keys

    @property
    def grammar(self):
        return _copy_grammar(self._grammar)

    def __pysp_getstate__(self):
        return {
            "grammar": self._grammar,
            "start_symbol": self.start_symbol,
            "pseudo_count": self.pseudo_count,
            "name": self.name,
            "keys": self.keys,
            "orig_n": self.orig_n,
        }

    def __pysp_setstate__(self, state):
        restored = HyperedgeReplacementGrammarEstimator(**state)
        self.__dict__.update(restored.__dict__)

    def accumulator_factory(self):
        """Return a factory for HRG Viterbi rule-count accumulators."""
        return HyperedgeReplacementGrammarAccumulatorFactory(
            grammar=self._grammar, start_symbol=self.start_symbol, keys=self.keys
        )

    def estimate(self, nobs, suff_stat):
        """Estimate rule frequencies from accumulated Viterbi parse counts."""
        checked = _validate_statistics(suff_stat, self._grammar)
        if checked.rejected_weight > 0.0 or checked.truncated_weight > 0.0:
            raise ValueError("exact HRG estimation rejects statistics with failed or truncated parses.")
        if checked.accepted_weight <= 0.0 and not (self.pseudo_count is not None and self.pseudo_count > 0.0):
            raise ValueError("cannot estimate HRG frequencies without accepted evidence or pseudo-count.")
        grammar = checked.counts
        if self.pseudo_count is not None:
            for rules in grammar.rule_dict.values():
                for rule in rules:
                    rule.frequency += self.pseudo_count
        for symbol, rules in grammar.rule_dict.items():
            if sum(rule.frequency for rule in rules) <= 0.0:
                source = self._grammar.rule_dict[symbol]
                for rule, original in zip(rules, source):
                    rule.frequency = original.frequency
        return HyperedgeReplacementGrammarDistribution(
            grammar,
            start_symbol=self.start_symbol,
            orig_n=self.orig_n,
            name=self.name,
            keys=self.keys,
        )


class HyperedgeReplacementGrammarDataEncoder(DataSequenceEncoder):
    """Identity encoder for sequences of observed graphs."""

    def __str__(self):
        return "HyperedgeReplacementGrammarDataEncoder"

    def __eq__(self, other):
        return isinstance(other, HyperedgeReplacementGrammarDataEncoder)

    def seq_encode(self, x):
        """Return graph observations unchanged."""
        return tuple(_validate_terminal_graph(graph) for graph in x)

    def row_count(self, x):
        """Return the number of validated graphs in an encoded payload."""
        if not isinstance(x, tuple):
            raise ValueError("encoded HRG payload must be a tuple.")
        for graph in x:
            _validate_terminal_graph(graph)
        return len(x)
