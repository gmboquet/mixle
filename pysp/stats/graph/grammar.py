"""Evaluate, estimate, and sample from a graph-grammar distribution over networks.

Defines the GrammarDistribution, GrammarSampler, GrammarAccumulatorFactory, GrammarEstimatorAccumulator,
GrammarEstimator, and the GrammarDataEncoder classes for use with pysparkplug.

Data type: A graph grammar object with ``rule_list`` and ``rule_dict`` attributes. The likelihood of an observed
grammar is computed rule-by-rule against the model grammar: a rule contributes the (frequency-weighted) probability of
an isomorphic model rule with a matching (or nearby, controlled by lhs_delta) left-hand side, mixed with a
degree-distribution background model weighted by mix_p.

"""

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from networkx.readwrite import json_graph

from pysp.arithmetic import *
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


def _edge_weights(graph):
    for a in graph:
        for b in graph[a]:
            edge_data = graph[a][b]
            if "weight" in edge_data:
                yield edge_data.get("weight", 1.0)
            else:
                for value in edge_data.values():
                    if isinstance(value, dict):
                        yield value.get("weight", 1.0)


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
    """Generate a graph by sampling rule right-hand-side graphs and taking their disjoint union.

    It preserves the sampler API but does not implement full node-replacement derivation machinery.
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
        idx = int(rng.choice(len(rules), p=weights))
        rule = rules[idx]
        graph = rule.graph.copy()
        if graph.number_of_nodes() == 0:
            break
        relabel = {node: offset + i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, relabel, copy=True)
        out = nx.compose(out, graph)
        offset += graph.number_of_nodes()
        rule_ordering.append(rule.lhs)

    return out, rule_ordering


def get_degree_dist(rule_list):
    """Compute the edge-weight histogram over the graphs of a list of grammar rules.

    Args:
        rule_list: List of rule objects, each with a networkx graph attribute.

    Returns:
        Dict mapping observed edge weight to its count, with an extra 'inf' bucket of count 1 for unseen weights.

    """
    dist = {}
    for rule in rule_list:
        for d in _edge_weights(rule.graph):
            if d not in dist:
                dist[d] = 0
            dist[d] += 1
    dist["inf"] = 1
    return dist


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

    def log_density(self, x):
        """Log-density of the grammar distribution at observation x.

        Each rule of the observed grammar is matched (up to isomorphism, with left-hand side slack lhs_delta)
        against the model grammar; matched rules contribute their frequency-weighted probability mixed with a
        degree-distribution background term weighted by mix_p. Unmatched rules can be recursively decomposed up
        to decomp_level times. The per-rule probabilities are averaged and logged.

        Args:
            x: Observed VertexReplacementGrammar object.

        Returns:
            Log-density at observation x.

        """
        total_p = 0.0
        # change to check for colors as well
        #                em = iso.numerical_edge_match('weight',1)
        model_grammar = self.grammar
        model_dd = get_degree_dist(model_grammar.rule_list)

        if len(x.rule_list) == 0:
            return 0.0

        else:
            total = 0.0
            for t_rule in x.rule_list:
                p = 0.0
                found_rule = False
                for i in np.append(
                    np.arange(t_rule.lhs, t_rule.lhs + self.lhs_delta + 1),
                    np.arange(t_rule.lhs - self.lhs_delta, t_rule.lhs),
                ):
                    if i in model_grammar.rule_dict:
                        found_rule = True
                        f_sum = sum([r.frequency for r in model_grammar.rule_dict[i]])
                        for m_rule in model_grammar.rule_dict[i]:
                            if _isomorphic_rule_graph(m_rule.graph, t_rule.graph):
                                p += (1.0 - self.mix_p) * (1.0 * m_rule.frequency) / f_sum

                if self.mix_p > 0.0:
                    rule_dd = get_degree_dist([t_rule])
                    for d, freq in rule_dd.items():
                        if d in model_dd:
                            dp = (self.mix_p * 1.0 * model_dd[d] / sum(model_dd.values())) ** freq
                            p += dp
                        else:
                            dp = (self.mix_p * 1.0 * model_dd["inf"] / sum(model_dd.values())) ** freq
                            p += dp

                # recursive decomp: only do if not found and has a decomp level set
                if not found_rule and self.decomp_level > 0:
                    recurs = 0
                    sub_rules = [(t_rule.lhs, t_rule.graph)]
                    while len(sub_rules) > 0 and recurs < self.decomp_level:
                        recurs += 1
                        new_sub_rules = []
                        for sub_rule in sub_rules:
                            found_rule = False
                            if sub_rule[0] in model_grammar.rule_dict:
                                f_sum = sum([r.frequency for r in model_grammar.rule_dict[sub_rule[0]]])
                                for m_rule in model_grammar.rule_dict[sub_rule[0]]:
                                    if _isomorphic_rule_graph(m_rule.graph, sub_rule[1]):
                                        found_rule = True
                                        p += (1.0 - self.mix_p) * (1.0 * m_rule.frequency) / f_sum
                            if not found_rule:
                                decomp = decomp_pair(sub_rule, "leiden")
                                for d in decomp:
                                    new_sub_rules.append(d)

                        sub_rules = new_sub_rules

                total_p += p
                total += 1.0
            if total > 0:
                total_p /= total
            rv = np.log(total_p)
            return rv

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
        rgrammar.num_rules += grammar.num_rules
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

        rgrammar.rule_list = []
        for rlist in rgrammar.rule_dict.values():
            rgrammar.rule_list += rlist
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
