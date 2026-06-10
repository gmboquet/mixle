from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import (
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    ParameterEstimator,
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

from cnrg.VRG import VRG
from cnrg.extract import MuExtractor, LocalExtractor, GlobalExtractor
from cnrg.Tree import create_tree
import cnrg.partitions as partitions
from cnrg.LightMultiGraph import LightMultiGraph
from cnrg.MDL import graph_dl
from cnrg.generate import generate_graph

from pprint import pprint
import networkx.algorithms.isomorphism as iso
import numpy as np
import random


def get_degree_dist(rule_list):
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
    def __init__(
        self, grammar, mix_p, decomp_level=0, lhs_delta=0, name=None, orig_n=100
    ):

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
        return np.exp(self.log_density(x))

    def log_density(self, x):
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
        return x

    def seq_log_density(self, x):
        return np.asarray([self.log_density(xx) for xx in x])

    def sampler(self, seed=None):
        return GrammarSampler(self.grammar, orig_n=self.orig_n)

    def estimator(self):
        return GrammarEstimator(self)


class GrammarSampler(object):
    def __init__(self, grammar, orig_n=100):

        self.grammar = grammar
        self.orig_n = orig_n

    def sample(self):

        g, rule_ordering = generate_graph(
            rule_dict=self.grammar.rule_dict, target_n=self.orig_n
        )

        return g

    def sample_seq(self, size_arr):
        rv = []
        for size in size_arr:
            g, rule_ordering = generate_graph(
                rule_dict=self.grammar.rule_dict, target_n=size
            )
            rv.append(g)
        return rv


class GrammarEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self):
        #             self.rule_list = []
        #             self.rule_dict = {}
        self.grammar = VRG("mu_level_dl", "leiden", "", 4)

    def update(self, grammar, weight, estimate):
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
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):
        #            for grammar in x:
        for i in range(len(x)):
            grammar = x[i]
            weight = weights[i]
            self.update(grammar, weight, estimate)

    def combine(self, suff_stat):
        self.update(suff_stat, 1.0, None)
        return self

    def value(self):
        return self.grammar

    def from_value(self, x):
        self.grammar = x
        return self


class GrammarEstimator(ParameterEstimator):
    def __init__(self, pseudo_count=None, name=None):

        self.name = name
        self.pseudo_count = pseudo_count

    #       self.levels = levels

    #                self.grammar = VRG('mu_level_dl','leiden','',4)

    def accumulatorFactory(self):

        obj = type(
            "", (object,), {"make": lambda self: GrammarEstimatorAccumulator()}
        )()

        return obj

    def estimate(self, nobs, suff_stat):
        grammar = suff_stat
        if self.pseudo_count is not None:
            for rlist in grammar.rule_dict.values():
                for rule in rlist:
                    rule.frequency += self.pseudo_count

        return GrammarDistribution(grammar, 0.01)
