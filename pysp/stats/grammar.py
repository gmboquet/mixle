"""Evaluate, estimate, and sample from a graph-grammar distribution over networks.

Defines the GrammarDistribution, GrammarSampler, GrammarAccumulatorFactory, GrammarEstimatorAccumulator,
GrammarEstimator, and the GrammarDataEncoder classes for use with pysparkplug.

Data type: A clustering-based node replacement grammar (a ``cnrg.VRG.VRG`` object) extracted from an observed
graph. The likelihood of an observed grammar is computed rule-by-rule against the model grammar: a rule
contributes the (frequency-weighted) probability of an isomorphic model rule with a matching (or nearby,
controlled by lhs_delta) left-hand side, mixed with a degree-distribution background model weighted by mix_p.

This module depends on the optional third-party package 'cnrg' (Clustering-based Node Replacement Grammars).
The module stays importable when 'cnrg' is missing; an informative ImportError is raised at the first use of
functionality that requires it (sampling and accumulation).

"""
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    ParameterEstimator,
    StatisticAccumulatorFactory,
    DataSequenceEncoder,
    DistributionSampler,
)
import numpy as np
from scipy.sparse import dok_matrix
import collections
from pysp.arithmetic import maxrandint

import glob
import logging
import math
import networkx as nx
from tqdm import tqdm
import copy

try:
    from cnrg.VRG import VRG
    from cnrg.extract import MuExtractor, LocalExtractor, GlobalExtractor
    from cnrg.Tree import create_tree
    import cnrg.partitions as partitions
    from cnrg.LightMultiGraph import LightMultiGraph
    from cnrg.MDL import graph_dl
    from cnrg.generate import generate_graph

    _CNRG_IMPORT_ERROR = None
except ImportError as _e:
    VRG = None
    MuExtractor = None
    LocalExtractor = None
    GlobalExtractor = None
    create_tree = None
    partitions = None
    LightMultiGraph = None
    graph_dl = None
    generate_graph = None
    _CNRG_IMPORT_ERROR = _e

from pprint import pprint
import networkx.algorithms.isomorphism as iso
import numpy as np
import random


def _require_cnrg():
    """Raise an informative ImportError if the optional 'cnrg' package is not installed.

    Returns:
        None. Raises ImportError (chained to the original import failure) when 'cnrg' is unavailable.

    """
    if _CNRG_IMPORT_ERROR is not None:
        raise ImportError(
            "pysp.stats.grammar requires the optional third-party package 'cnrg' "
            "(Clustering-based Node Replacement Grammars) for sampling and estimation. "
            "Install 'cnrg' to use this functionality."
        ) from _CNRG_IMPORT_ERROR


def get_degree_dist(rule_list):
    """Compute the edge-weight (degree) histogram over the graphs of a list of grammar rules.

    Args:
        rule_list: List of cnrg rule objects, each with a networkx graph attribute.

    Returns:
        Dict mapping observed edge weight to its count, with an extra 'inf' bucket of count 1 for unseen weights.

    """
    dist = {}
    for rule in rule_list:
        for a in rule.graph:
            for b in rule.graph[a]:
                d = rule.graph[a][b]["weight"]
                if d not in dist:
                    dist[d] = 0
                dist[d] += 1
    dist["inf"] = 1
    return dist


class GrammarDistribution(SequenceEncodableProbabilityDistribution):
    """GrammarDistribution object for evaluating the likelihood of node-replacement grammars (VRG objects)."""

    def __init__(
        self, grammar, mix_p, decomp_level=0, lhs_delta=0, name=None, orig_n=100
    ):
        """GrammarDistribution object defined by a model grammar and mixing parameters.

        Args:
            grammar: cnrg VRG object serving as the model grammar.
            mix_p (float): Weight in [0, 1] on the degree-distribution background model.
            decomp_level (int): Maximum recursion depth for decomposing unmatched rules.
            lhs_delta (int): Allowed slack when matching rule left-hand sides.
            name (Optional[str]): String name of object instance.
            orig_n (int): Target number of nodes used when sampling graphs.

        Attributes:
            grammar: cnrg VRG object serving as the model grammar.
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
            x: Observed cnrg VRG object.

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
            x: Observed cnrg VRG object.

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
                            g1 = nx.convert_node_labels_to_integers(m_rule.graph)
                            g2 = nx.convert_node_labels_to_integers(t_rule.graph)
                            #                                    if nx.is_isomorphic(g1,g2,edge_match=iso.numerical_edge_match('weight', 1.0),node_match=iso.categorical_node_match('label', '')):
                            #                                    if nx.is_isomorphic(g1,g2,edge_match=iso.categorical_edge_match(['weight','edge_color'], [1.0,'']),node_match=iso.categorical_node_match(['label','node_color'], ['',''])):
                            if nx.is_isomorphic(
                                g1,
                                g2,
                                edge_match=iso.categorical_edge_match("edge_color", ""),
                                node_match=iso.categorical_node_match(
                                    ["label", "node_color"], ["", ""]
                                ),
                            ) and nx.is_isomorphic(
                                g1,
                                g2,
                                edge_match=iso.numerical_edge_match("weight", 1.0),
                                node_match=iso.categorical_node_match(
                                    ["label", "node_color"], ["", ""]
                                ),
                            ):
                                p += (
                                    (1.0 - self.mix_p)
                                    * (1.0 * m_rule.frequency)
                                    / f_sum
                                )

                if self.mix_p > 0.0:
                    rule_dd = get_degree_dist([t_rule])
                    for d, freq in rule_dd.items():
                        if d in model_dd:
                            dp = (
                                self.mix_p * 1.0 * model_dd[d] / sum(model_dd.values())
                            ) ** freq
                            p += dp
                        else:
                            dp = (
                                self.mix_p
                                * 1.0
                                * model_dd["inf"]
                                / sum(model_dd.values())
                            ) ** freq
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
                                f_sum = sum(
                                    [
                                        r.frequency
                                        for r in model_grammar.rule_dict[sub_rule[0]]
                                    ]
                                )
                                for m_rule in model_grammar.rule_dict[sub_rule[0]]:
                                    g1 = nx.convert_node_labels_to_integers(
                                        m_rule.graph
                                    )
                                    g2 = nx.convert_node_labels_to_integers(sub_rule[1])
                                    #                                            if nx.is_isomorphic(g1,g2,edge_match=iso.numerical_edge_match('weight', 1.0),node_match=iso.categorical_node_match('label', '')):
                                    if nx.is_isomorphic(
                                        g1,
                                        g2,
                                        edge_match=iso.categorical_edge_match(
                                            "edge_color", ""
                                        ),
                                        node_match=iso.categorical_node_match(
                                            ["label", "node_color"], ["", ""]
                                        ),
                                    ) and nx.is_isomorphic(
                                        g1,
                                        g2,
                                        edge_match=iso.numerical_edge_match(
                                            "weight", 1.0
                                        ),
                                        node_match=iso.categorical_node_match(
                                            ["label", "node_color"], ["", ""]
                                        ),
                                    ):
                                        found_rule = True
                                        p += (
                                            (1.0 - self.mix_p)
                                            * (1.0 * m_rule.frequency)
                                            / f_sum
                                        )
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
            x: Sequence of cnrg VRG objects.

        Returns:
            The input sequence unchanged.

        """
        return x

    def seq_log_density(self, x):
        """Evaluate log_density() at each encoded observation.

        Args:
            x: Sequence of cnrg VRG objects (from seq_encode).

        Returns:
            Numpy array of log-densities, one per observation.

        """
        return np.asarray([self.log_density(xx) for xx in x])

    def sampler(self, seed=None):
        """Create a GrammarSampler object from the model grammar of this instance.

        Args:
            seed (Optional[int]): Unused; kept for protocol compatibility.

        Returns:
            GrammarSampler object.

        """
        return GrammarSampler(self.grammar, orig_n=self.orig_n)

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

    def __init__(self, grammar, orig_n=100):
        """GrammarSampler object.

        Args:
            grammar: cnrg VRG object to generate graphs from.
            orig_n (int): Default target number of nodes for generated graphs.

        Attributes:
            grammar: cnrg VRG object to generate graphs from.
            orig_n (int): Default target number of nodes for generated graphs.

        """
        self.grammar = grammar
        self.orig_n = orig_n

    def sample(self):
        """Generate a single graph from the grammar with roughly orig_n nodes.

        Returns:
            A networkx graph generated from the grammar.

        """
        _require_cnrg()

        g, rule_ordering = generate_graph(
            rule_dict=self.grammar.rule_dict, target_n=self.orig_n
        )

        return g

    def sample_seq(self, size_arr):
        """Generate one graph per entry of size_arr, each targeting that many nodes.

        Args:
            size_arr: Sequence of target node counts.

        Returns:
            List of networkx graphs, one per requested size.

        """
        _require_cnrg()

        rv = []
        for size in size_arr:
            g, rule_ordering = generate_graph(
                rule_dict=self.grammar.rule_dict, target_n=size
            )
            rv.append(g)
        return rv


class GrammarEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """GrammarEstimatorAccumulator object for merging observed grammars into a frequency-weighted grammar."""

    def __init__(self):
        """GrammarEstimatorAccumulator object.

        Attributes:
            grammar: cnrg VRG object accumulating frequency-weighted rules from observations.

        """
        _require_cnrg()
        #             self.rule_list = []
        #             self.rule_dict = {}
        self.grammar = VRG("mu_level_dl", "leiden", "", 4)

    def update(self, grammar, weight, estimate):
        """Merge an observed grammar into the accumulated grammar with the given weight.

        Rules isomorphic to an already-accumulated rule (matching left-hand side, edge colors, weights, and
        node labels/colors) have their frequency incremented by weight times the observed frequency; new rules
        are copied in with weight-scaled frequency.

        Args:
            grammar: Observed cnrg VRG object.
            weight (float): Weight of the observation.
            estimate (Optional[GrammarDistribution]): Previous estimate (unused).

        Returns:
            The accumulated cnrg VRG object.

        """
        #   change to check for node color as well
        #            em = iso.numerical_edge_match('weight',1)
        #            rgrammar = cnrg.VRG(x[0].type,x[0].clustering,x[0].name,x[0].mu)
        #            rgrammar = estimate.grammar
        rgrammar = self.grammar
        #            for grammar in x:
        #                rgrammar.rule_list += grammar.rule_list
        rgrammar.cost += grammar.cost
        rgrammar.num_rules += grammar.num_rules
        for lhs in grammar.rule_dict:
            if lhs not in rgrammar.rule_dict:
                #                        rgrammar.rule_dict[lhs] = []
                rgrammar.rule_dict[lhs] = grammar.rule_dict[lhs]
                for rule in rgrammar.rule_dict[lhs]:
                    rule.frequency *= weight
            #                    rgrammar.rule_dict[lhs] += grammar.rule_dict[lhs]
            else:
                for rule in grammar.rule_dict[lhs]:
                    found_rule = False
                    for r_rule in rgrammar.rule_dict[lhs]:
                        g1 = nx.convert_node_labels_to_integers(r_rule.graph)
                        g2 = nx.convert_node_labels_to_integers(rule.graph)
                        #                            if nx.is_isomorphic(g1,g2,edge_match=iso.numerical_edge_match('weight', 1.0),node_match=iso.categorical_node_match('label', '')):
                        if nx.is_isomorphic(
                            g1,
                            g2,
                            edge_match=iso.categorical_edge_match("edge_color", ""),
                            node_match=iso.categorical_node_match(
                                ["label", "node_color"], ["", ""]
                            ),
                        ) and nx.is_isomorphic(
                            g1,
                            g2,
                            edge_match=iso.numerical_edge_match("weight", 1.0),
                            node_match=iso.categorical_node_match(
                                ["label", "node_color"], ["", ""]
                            ),
                        ):
                            found_rule = True
                            r_rule.frequency += weight * rule.frequency
                            break
                    if not found_rule:
                        crule = copy.copy(rule)
                        crule.frequency *= weight
                        rgrammar.rule_dict[lhs].append(crule)

        rgrammar.rule_list = []
        for rlist in rgrammar.rule_dict.values():
            rgrammar.rule_list += rlist
        return rgrammar

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with a single weighted observation.

        Args:
            x: Observed cnrg VRG object.
            weight (float): Weight of the observation.
            rng: RandomState (unused).

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Initialize the accumulator with a sequence of weighted observations.

        Args:
            x: Sequence of cnrg VRG objects (from seq_encode).
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
            x: Sequence of cnrg VRG objects (from seq_encode).
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
        """Merge the sufficient statistic of another accumulator (a cnrg VRG object) into this one.

        Args:
            suff_stat: cnrg VRG object from another accumulator's value().

        Returns:
            This GrammarEstimatorAccumulator object.

        """
        self.update(suff_stat, 1.0, None)
        return self

    def value(self):
        """Returns the accumulated cnrg VRG object sufficient statistic."""
        return self.grammar

    def from_value(self, x):
        """Set the accumulated sufficient statistic from a cnrg VRG object.

        Args:
            x: cnrg VRG object.

        Returns:
            This GrammarEstimatorAccumulator object.

        """
        self.grammar = x
        return self

    def acc_to_encoder(self):
        """Returns a GrammarDataEncoder object for encoding sequences of data."""
        return GrammarDataEncoder()


class GrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """GrammarAccumulatorFactory object for creating GrammarEstimatorAccumulator objects."""

    def make(self):
        """Returns a new GrammarEstimatorAccumulator object."""
        return GrammarEstimatorAccumulator()


class GrammarEstimator(ParameterEstimator):
    """GrammarEstimator object for estimating GrammarDistribution objects from aggregated grammars."""

    def __init__(self, pseudo_count=None, name=None):
        """GrammarEstimator object.

        Args:
            pseudo_count (Optional[float]): Added to each accumulated rule frequency when estimating.
            name (Optional[str]): String name of object instance.

        Attributes:
            pseudo_count (Optional[float]): Added to each accumulated rule frequency when estimating.
            name (Optional[str]): String name of object instance.

        """
        self.name = name
        self.pseudo_count = pseudo_count

    #       self.levels = levels

    #                self.grammar = VRG('mu_level_dl','leiden','',4)

    def accumulator_factory(self):
        """Returns a GrammarAccumulatorFactory object."""
        return GrammarAccumulatorFactory()

    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        """Estimate a GrammarDistribution from an accumulated grammar sufficient statistic.

        Args:
            nobs (Optional[float]): Weighted number of observations (unused).
            suff_stat: cnrg VRG object of accumulated rule frequencies.

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
        return 'GrammarDataEncoder'

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
            x: Sequence of cnrg VRG objects.

        Returns:
            The input sequence unchanged.

        """
        return x
