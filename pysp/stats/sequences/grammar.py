"""Evaluate, estimate, and sample from a graph-grammar distribution over networks.

This is a GRAPH grammar (a vertex/node-replacement grammar over networkx graphs), not a string/text
grammar: each rule is ``(lhs, right-hand-side graph, frequency)`` and rules are matched by graph
isomorphism (on node labels/colors and edge colors/weights). Defines the GrammarDistribution,
GrammarSampler, GrammarAccumulatorFactory, GrammarEstimatorAccumulator, GrammarEstimator, and the
GrammarDataEncoder classes for use with pysparkplug.

Data type: a graph-grammar object (VertexReplacementGrammar) with ``rule_list`` and ``rule_dict``
attributes. The model defines a probability over rules; the log-density of an observed grammar is the
sum over its rules of ``log p(rule)``, where ``p(rule)`` mixes a frequency-based match probability
(isomorphic model rule, left-hand side within ``lhs_delta``, optionally via connected-component
decomposition up to ``decomp_level``) with a node-degree background model weighted by ``mix_p``. Both
mixture components are valid probabilities, so the log-density is <= 0.

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
        raise ImportError(
            "The graph-grammar models require networkx. Install it with `pip install pysp-learn[grammar]`."
        )


from pysp.engines.arithmetic import *
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class GrammarRule:
    """Lightweight graph-grammar rule."""

    __pysp_serializable__ = True

    def __init__(self, lhs, graph, frequency=1.0) -> None:
        _require_networkx()
        self.lhs = lhs
        self.graph = graph.copy()
        self.frequency = float(frequency)

    def __pysp_getstate__(self):
        return {
            "lhs": self.lhs,
            "graph": json_graph.node_link_data(self.graph, edges="edges"),
            "frequency": self.frequency,
        }

    def __pysp_setstate__(self, state):
        self.lhs = state["lhs"]
        self.graph = json_graph.node_link_graph(state["graph"], edges="edges")
        self.frequency = float(state["frequency"])

    def __str__(self) -> str:
        return "GrammarRule(lhs=%s, frequency=%s, nodes=%s, edges=%s)" % (
            repr(self.lhs),
            repr(self.frequency),
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
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

    def add_rule(self, rule: GrammarRule) -> None:
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
    return GrammarRule(rule.lhs, rule.graph, rule.frequency)


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


def generate_graph(rule_dict, target_n=100, rng=None):
    """Generate a connected graph by repeated vertex-replacement-style rule application.

    Starts from a single sampled rule right-hand side, then grows the graph by sampling further
    right-hand sides (with probability proportional to rule frequency) and gluing each new block to
    the current graph with one edge between a random existing node and a random new node, until the
    graph reaches ``target_n`` nodes. A fully embedding-aware vertex-replacement derivation would
    reconnect a replaced node's neighbours according to per-rule connection instructions; the rule
    format here does not carry those, so this connected approximation is used (it keeps every
    generated graph in a single component rather than a disjoint union of rule graphs).

    Returns:
        Tuple of (networkx graph, list of the left-hand sides applied in order).
    """
    rng = np.random.RandomState() if rng is None else rng
    rules = [rule for rlist in rule_dict.values() for rule in rlist if rule.frequency > 0.0]
    if len(rules) == 0:
        return nx.Graph(), []

    weights = np.asarray([rule.frequency for rule in rules], dtype=float)
    weights /= weights.sum()
    out = nx.Graph()
    rule_ordering = []
    offset = 0
    target_n = max(0, int(target_n))
    max_steps = max(1, target_n + len(rules))

    for _ in range(max_steps):
        if out.number_of_nodes() >= target_n:
            break
        rule = rules[int(rng.choice(len(rules), p=weights))]
        graph = rule.graph.copy()
        if graph.number_of_nodes() == 0:
            break
        relabel = {node: offset + i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, relabel, copy=True)
        new_nodes = list(graph.nodes())
        existing_nodes = list(out.nodes())
        out = nx.compose(out, graph)
        if existing_nodes:  # glue the new block on so the result stays connected
            a = existing_nodes[rng.randint(len(existing_nodes))]
            b = new_nodes[rng.randint(len(new_nodes))]
            out.add_edge(a, b)
        offset += graph.number_of_nodes()
        rule_ordering.append(rule.lhs)

    return out, rule_ordering


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


def _background_log_prob(graph, degree_counts):
    """Mean log-probability of a graph's node degrees under the model's degree distribution.

    Each node contributes the (Laplace-smoothed) model probability of its degree; degrees unseen in
    the model fall back to the ``'inf'`` smoothing bucket. The returned value is in ``(-inf, 0]`` so
    its exponential is a proper probability in ``(0, 1]`` suitable for the background mixture term.
    """
    total = float(sum(degree_counts.values()))
    if graph.number_of_nodes() == 0:
        return float(np.log(degree_counts["inf"] / total))
    log_ps = [np.log(degree_counts.get(degree, degree_counts["inf"]) / total) for _, degree in graph.degree()]
    return float(np.mean(log_ps))


class GrammarDistribution(SequenceEncodableProbabilityDistribution):
    """GrammarDistribution object for evaluating the likelihood of node-replacement grammars (VertexReplacementGrammar objects)."""

    def __init__(self, grammar, mix_p, decomp_level=0, lhs_delta=0, name=None, orig_n=100):
        """GrammarDistribution object defined by a model grammar and mixing parameters.

        Args:
            grammar: VertexReplacementGrammar object serving as the model grammar.
            mix_p (float): Weight in [0, 1] on the degree-distribution background model.
            decomp_level (int): Maximum recursion depth for decomposing unmatched rules.
            lhs_delta (int): Allowed slack when matching rule left-hand sides.
            name (Optional[str]): String name of object instance.
            orig_n (int): Target number of nodes used when sampling graphs.

        Attributes:
            grammar: VertexReplacementGrammar object serving as the model grammar.
            mix_p (float): Weight in [0, 1] on the degree-distribution background model.
            decomp_level (int): Maximum recursion depth for decomposing unmatched rules.
            lhs_delta (int): Allowed slack when matching rule left-hand sides.
            name (Optional[str]): String name of object instance.
            orig_n (int): Target number of nodes used when sampling graphs.

        """
        _require_networkx()
        self.name = name
        self.grammar = grammar
        self.mix_p = mix_p
        self.decomp_level = decomp_level
        self.lhs_delta = lhs_delta
        self.orig_n = orig_n

    def __str__(self):

        return (
            "GrammarDistribution("
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
            x: Observed VertexReplacementGrammar object.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def _match_prob(self, lhs, graph, total_freq, depth):
        """Model match probability of a single rule (lhs, graph), in [0, 1].

        The direct term is the total frequency of model rules isomorphic to ``graph`` whose left-hand
        side is within ``lhs_delta`` of ``lhs``, divided by the model's total rule frequency. If nothing
        matches and ``depth`` remains, the graph is split into connected components and the match
        probabilities of the parts are multiplied (a conservative decomposition fallback).
        """
        matched = 0.0
        for cand_lhs in range(lhs - self.lhs_delta, lhs + self.lhs_delta + 1):
            for m_rule in self.grammar.rule_dict.get(cand_lhs, ()):
                if _isomorphic_rule_graph(m_rule.graph, graph):
                    matched += m_rule.frequency
        if matched > 0.0 or depth <= 0:
            return matched / total_freq
        decomposition = decomp_pair((lhs, graph))
        if not decomposition:
            return 0.0
        prob = 1.0
        for sub_lhs, sub_graph in decomposition:
            prob *= self._match_prob(sub_lhs, sub_graph, total_freq, depth - 1)
        return prob

    def log_density(self, x):
        """Log-density of the grammar distribution at an observed grammar x.

        The observed grammar is treated as a bag of rules drawn i.i.d. from the model, so the
        log-density is the sum over its rules of ``log p(rule)``. Each rule's probability is a
        two-component mixture::

            p(rule) = (1 - mix_p) * p_match(rule) + mix_p * p_background(rule)

        where ``p_match`` is the frequency of isomorphic model rules (left-hand side within
        ``lhs_delta``, optionally via connected-component decomposition up to ``decomp_level``) over the
        model's total rule frequency, and ``p_background`` is the rule's mean node-degree probability
        under the model degree distribution. Both components lie in [0, 1], so each ``p(rule)`` is a
        valid probability and the log-density is <= 0. An empty grammar has log-density 0 (the empty
        product over rules).

        Args:
            x: Observed VertexReplacementGrammar object.

        Returns:
            Log-density at observation x (a float <= 0, or -inf if some rule has zero probability).

        """
        if x is None or len(x.rule_list) == 0:
            return 0.0

        total_freq = sum(r.frequency for r in self.grammar.rule_list)
        if total_freq <= 0.0:
            total_freq = 1.0  # degenerate/empty model -> rely entirely on the background term
        degree_counts = get_degree_dist(self.grammar.rule_list) if self.grammar.rule_list else {"inf": 1}

        log_p = 0.0
        for t_rule in x.rule_list:
            p_match = self._match_prob(t_rule.lhs, t_rule.graph, total_freq, self.decomp_level)
            p_background = np.exp(_background_log_prob(t_rule.graph, degree_counts))
            p = (1.0 - self.mix_p) * p_match + self.mix_p * p_background
            if p <= 0.0:
                return float("-inf")
            log_p += float(np.log(p))
        return log_p

    # combine list of grammars into singular grammar? need to take multiple sample outputs as input
    def seq_encode(self, x):
        """Encode a sequence of grammar observations for vectorized calls (identity encoding).

        Args:
            x: Sequence of VertexReplacementGrammar objects.

        Returns:
            The input sequence unchanged.

        """
        return x

    def seq_log_density(self, x):
        """Evaluate log_density() at each encoded observation.

        Args:
            x: Sequence of VertexReplacementGrammar objects (from seq_encode).

        Returns:
            Numpy array of log-densities, one per observation.

        """
        return np.asarray([self.log_density(xx) for xx in x])

    def sampler(self, seed=None):
        """Create a GrammarSampler object from the model grammar of this instance.

        Args:
            seed (Optional[int]): Seed for the sampler random generator.

        Returns:
            GrammarSampler object.

        """
        return GrammarSampler(self.grammar, orig_n=self.orig_n, seed=seed)

    def estimator(self, pseudo_count=None):
        """Create a GrammarEstimator object.

        Args:
            pseudo_count (Optional[float]): Added to rule frequencies when estimating.

        Returns:
            GrammarEstimator object.

        """
        return GrammarEstimator(pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self):
        """Returns a GrammarDataEncoder object for encoding sequences of data."""
        return GrammarDataEncoder()


class GrammarSampler(DistributionSampler):
    """GrammarSampler object for sampling graphs generated from a node-replacement grammar."""

    def __init__(self, grammar, orig_n=100, seed=None):
        """GrammarSampler object.

        Args:
            grammar: VertexReplacementGrammar object to generate graphs from.
            orig_n (int): Default target number of nodes for generated graphs.
            seed (Optional[int]): Seed for the local random generator.

        Attributes:
            grammar: VertexReplacementGrammar object to generate graphs from.
            orig_n (int): Default target number of nodes for generated graphs.

        """
        self.grammar = grammar
        self.orig_n = orig_n
        self.rng = np.random.RandomState(seed)

    def sample(self):
        """Generate a single graph from the grammar with roughly orig_n nodes.

        Returns:
            A networkx graph generated from the grammar.

        """
        g, rule_ordering = generate_graph(rule_dict=self.grammar.rule_dict, target_n=self.orig_n, rng=self.rng)

        return g

    def sample_seq(self, size_arr):
        """Generate one graph per entry of size_arr, each targeting that many nodes.

        Args:
            size_arr: Sequence of target node counts.

        Returns:
            List of networkx graphs, one per requested size.

        """
        rv = []
        for size in size_arr:
            g, rule_ordering = generate_graph(rule_dict=self.grammar.rule_dict, target_n=size, rng=self.rng)
            rv.append(g)
        return rv


class GrammarEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """GrammarEstimatorAccumulator object for merging observed grammars into a frequency-weighted grammar."""

    def __init__(self, keys=None):
        """GrammarEstimatorAccumulator object.

        Args:
            keys (Optional[str]): Key for merging sufficient statistics with matching key'd objects.

        Attributes:
            grammar: VertexReplacementGrammar object accumulating frequency-weighted rules from observations.
            key (Optional[str]): Key for merging sufficient statistics with matching key'd objects.

        """
        #             self.rule_list = []
        #             self.rule_dict = {}
        self.grammar = VertexReplacementGrammar("mu_level_dl", "leiden", "", 4)
        self.keys = keys

    def update(self, grammar, weight, estimate):
        """Merge an observed grammar into the accumulated grammar with the given weight.

        Rules isomorphic to an already-accumulated rule (matching left-hand side, edge colors, weights, and
        node labels/colors) have their frequency incremented by weight times the observed frequency; new rules
        are copied in with weight-scaled frequency.

        Args:
            grammar: Observed grammar object.
            weight (float): Weight of the observation.
            estimate (Optional[GrammarDistribution]): Previous estimate (unused).

        Returns:
            The accumulated VertexReplacementGrammar object.

        """
        #   change to check for node color as well
        #            em = iso.numerical_edge_match('weight',1)
        #            rgrammar = estimate.grammar
        rgrammar = self.grammar
        #            for grammar in x:
        #                rgrammar.rule_list += grammar.rule_list
        rgrammar.cost += grammar.cost
        for lhs in grammar.rule_dict:
            if lhs not in rgrammar.rule_dict:
                #                        rgrammar.rule_dict[lhs] = []
                rgrammar.rule_dict[lhs] = [_copy_rule(rule) for rule in grammar.rule_dict[lhs]]
                for rule in rgrammar.rule_dict[lhs]:
                    rule.frequency *= weight
            #                    rgrammar.rule_dict[lhs] += grammar.rule_dict[lhs]
            else:
                for rule in grammar.rule_dict[lhs]:
                    found_rule = False
                    for r_rule in rgrammar.rule_dict[lhs]:
                        if _isomorphic_rule_graph(r_rule.graph, rule.graph):
                            found_rule = True
                            r_rule.frequency += weight * rule.frequency
                            break
                    if not found_rule:
                        crule = _copy_rule(rule)
                        crule.frequency *= weight
                        rgrammar.rule_dict[lhs].append(crule)

        rgrammar.refresh_rules()  # keep rule_list and num_rules consistent with rule_dict
        return rgrammar

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with a single weighted observation.

        Args:
            x: Observed VertexReplacementGrammar object.
            weight (float): Weight of the observation.
            rng: RandomState (unused).

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Initialize the accumulator with a sequence of weighted observations.

        Args:
            x: Sequence of VertexReplacementGrammar objects (from seq_encode).
            weights: Sequence of observation weights.
            rng: RandomState (unused).

        Returns:
            None.

        """
        for i in range(len(x)):
            self.initialize(x[i], weights[i], rng)

    def seq_update(self, x, weights, estimate):
        """Merge a sequence of weighted observed grammars into the accumulated grammar.

        Args:
            x: Sequence of VertexReplacementGrammar objects (from seq_encode).
            weights: Sequence of observation weights.
            estimate (Optional[GrammarDistribution]): Previous estimate (unused).

        Returns:
            None.

        """
        #            for grammar in x:
        for i in range(len(x)):
            grammar = x[i]
            weight = weights[i]
            self.update(grammar, weight, estimate)

    def combine(self, suff_stat):
        """Merge the sufficient statistic of another accumulator (a VertexReplacementGrammar object) into this one.

        Args:
            suff_stat: VertexReplacementGrammar object from another accumulator's value().

        Returns:
            This GrammarEstimatorAccumulator object.

        """
        self.update(suff_stat, 1.0, None)
        return self

    def value(self):
        """Returns the accumulated VertexReplacementGrammar object sufficient statistic."""
        return self.grammar

    def from_value(self, x):
        """Set the accumulated sufficient statistic from a VertexReplacementGrammar object.

        Args:
            x: VertexReplacementGrammar object.

        Returns:
            This GrammarEstimatorAccumulator object.

        """
        self.grammar = x
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
        """Returns a GrammarDataEncoder object for encoding sequences of data."""
        return GrammarDataEncoder()


class GrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """GrammarAccumulatorFactory object for creating GrammarEstimatorAccumulator objects."""

    def __init__(self, keys=None):
        """GrammarAccumulatorFactory object.

        Args:
            keys (Optional[str]): Key for merging sufficient statistics with matching key'd objects.

        """
        self.keys = keys

    def make(self):
        """Returns a new GrammarEstimatorAccumulator object."""
        return GrammarEstimatorAccumulator(keys=self.keys)


class GrammarEstimator(ParameterEstimator):
    """GrammarEstimator object for estimating GrammarDistribution objects from aggregated grammars."""

    def __init__(self, pseudo_count=None, name=None, keys=None):
        """GrammarEstimator object.

        Args:
            pseudo_count (Optional[float]): Added to each accumulated rule frequency when estimating.
            name (Optional[str]): String name of object instance.
            keys (Optional[str]): Key for merging sufficient statistics with matching key'd objects.

        Attributes:
            pseudo_count (Optional[float]): Added to each accumulated rule frequency when estimating.
            name (Optional[str]): String name of object instance.
            keys (Optional[str]): Key for merging sufficient statistics with matching key'd objects.

        """
        _require_networkx()
        self.name = name
        self.pseudo_count = pseudo_count
        self.keys = keys

    #       self.levels = levels

    #                self.grammar = VertexReplacementGrammar('mu_level_dl','leiden','',4)

    def accumulator_factory(self):
        """Returns a GrammarAccumulatorFactory object."""
        return GrammarAccumulatorFactory(keys=self.keys)

    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        """Estimate a GrammarDistribution from an accumulated grammar sufficient statistic.

        Args:
            nobs (Optional[float]): Weighted number of observations (unused).
            suff_stat: grammar object of accumulated rule frequencies.

        Returns:
            GrammarDistribution object.

        """
        grammar = suff_stat
        if self.pseudo_count is not None:
            for rlist in grammar.rule_dict.values():
                for rule in rlist:
                    rule.frequency += self.pseudo_count

        return GrammarDistribution(grammar, 0.01)


class GrammarDataEncoder(DataSequenceEncoder):
    """GrammarDataEncoder object for encoding sequences of grammar observations (identity encoding)."""

    def __str__(self):
        """Returns string representation of GrammarDataEncoder object."""
        return "GrammarDataEncoder"

    def __eq__(self, other):
        """Encoders are interchangeable iff other is also a GrammarDataEncoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a GrammarDataEncoder instance.

        """
        return isinstance(other, GrammarDataEncoder)

    def seq_encode(self, x):
        """Encode a sequence of grammar observations for vectorized calls (identity encoding).

        Args:
            x: Sequence of VertexReplacementGrammar objects.

        Returns:
            The input sequence unchanged.

        """
        return x


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
GrammarAccumulator = GrammarEstimatorAccumulator


# Backward-compatible alias for the former VRG (vertex replacement grammar) name.
VRG = VertexReplacementGrammar
