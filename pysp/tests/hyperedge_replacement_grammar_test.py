"""Tests for the hyperedge-replacement graph grammar (HRG): derivation, parsing, estimation."""

import importlib.util
import unittest

import numpy as np

HAS_NETWORKX = importlib.util.find_spec("networkx") is not None
if HAS_NETWORKX:
    import networkx as nx


@unittest.skipUnless(HAS_NETWORKX, "networkx is not installed")
class HyperedgeReplacementGrammarTestCase(unittest.TestCase):
    @staticmethod
    def _node(g, n, label):
        g.add_node(n, label=label, node_color="")

    @staticmethod
    def _edge(g, a, b):
        g.add_edge(a, b, weight=1.0, edge_color="")

    def _path_grammar(self):
        # S -> [X . A(X)] ; A(u) -> [u - Y . A(Y)] (grow, freq 5) ; A(u) -> [u] (stop, freq 1)
        from pysp.stats.graphs.hyperedge_replacement_grammar import (
            HyperedgeReplacementGrammar,
            HyperedgeReplacementRule,
            Hypergraph,
        )

        s = nx.Graph()
        self._node(s, 0, "X")
        grow = nx.Graph()
        grow.add_node(0)
        self._node(grow, 1, "Y")
        self._edge(grow, 0, 1)
        stop = nx.Graph()
        stop.add_node(0)
        g = HyperedgeReplacementGrammar()
        g.add_rule(HyperedgeReplacementRule("S", Hypergraph(s, [("A", (0,))]), (), 1.0))
        g.add_rule(HyperedgeReplacementRule("A", Hypergraph(grow, [("A", (1,))]), (0,), 5.0))
        g.add_rule(HyperedgeReplacementRule("A", Hypergraph(stop, []), (0,), 1.0))
        return g

    def _path(self, n):
        g = nx.Graph()
        self._node(g, 0, "X")
        for i in range(1, n):
            self._node(g, i, "Y")
            self._edge(g, i - 1, i)
        return g

    def test_derivation_sampler_generates_connected_graphs(self):
        from pysp.stats.graphs.hyperedge_replacement_grammar import generate_graph

        for sd in range(5):
            g = generate_graph(self._path_grammar(), "S", target_n=6, rng=np.random.RandomState(sd))
            self.assertGreaterEqual(g.number_of_nodes(), 1)
            self.assertTrue(g.number_of_nodes() < 2 or nx.is_connected(g))
            self.assertTrue(all(d <= 2 for _, d in g.degree()))  # the path grammar yields paths

    def test_log_density_is_the_parse_based_marginal(self):
        from pysp.stats.graphs.hyperedge_replacement_grammar import best_derivation, marginal_log_prob

        g = self._path_grammar()
        # X-Y derives one way: S(1) * grow(5/6) * stop(1/6)
        self.assertAlmostEqual(marginal_log_prob(self._path(2), g, "S"), float(np.log(5 / 6 * 1 / 6)), places=9)
        self.assertAlmostEqual(marginal_log_prob(self._path(3), g, "S"), float(np.log(5 / 6 * 5 / 6 * 1 / 6)), places=9)
        # single parse here -> marginal == Viterbi
        self.assertAlmostEqual(
            marginal_log_prob(self._path(3), g, "S"), best_derivation(self._path(3), g, "S")[0], places=9
        )

    def test_generated_graphs_parse_and_foreign_is_neg_inf(self):
        from pysp.stats.graphs.hyperedge_replacement_grammar import generate_graph, marginal_log_prob

        g = self._path_grammar()
        for sd in range(8):
            graph = generate_graph(g, "S", target_n=6, rng=np.random.RandomState(sd))
            self.assertTrue(np.isfinite(marginal_log_prob(graph, g, "S")))
        triangle = nx.Graph()
        for i in range(3):
            self._node(triangle, i, "X")
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            self._edge(triangle, a, b)
        self.assertEqual(marginal_log_prob(triangle, g, "S"), float("-inf"))

    def test_distribution_is_a_lower_bound_density_over_graphs(self):
        import pysp.capability as cap
        from pysp.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(self._path_grammar(), start_symbol="S", orig_n=6)
        g = dist.sampler(seed=1).sample()
        self.assertIsInstance(g, nx.Graph)
        self.assertTrue(np.isfinite(dist.log_density(g)))
        self.assertFalse(cap.supports(dist, cap.ExactDensity))  # parse can truncate -> lower bound
        value, exact = dist.log_density(g, with_status=True)  # small graph: exact
        self.assertTrue(exact)
        self.assertEqual(len(dist.sampler(seed=2).sample(3)), 3)  # sample(size) honors the contract

    def test_estimator_recovers_frequencies_by_parsing(self):
        from pysp.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(self._path_grammar(), start_symbol="S", orig_n=4)
        acc = dist.estimator().accumulator_factory().make()
        for sd in range(40):
            acc.update(dist.sampler(seed=sd).sample(), 1.0, None)
        fit = dist.estimator().estimate(None, acc.value())
        grow = sum(r.frequency for r in fit.grammar.rule_dict["A"] if r.rhs.graph.number_of_nodes() == 2)
        stop = sum(r.frequency for r in fit.grammar.rule_dict["A"] if r.rhs.graph.number_of_nodes() == 1)
        self.assertGreater(grow, stop)  # grow fires many times per graph, stop once

    def test_serialization_round_trip(self):
        from pysp.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(self._path_grammar(), start_symbol="S", orig_n=6)
        loaded = HyperedgeReplacementGrammarDistribution.from_json(dist.to_json())
        self.assertAlmostEqual(loaded.log_density(self._path(3)), dist.log_density(self._path(3)), places=12)


if __name__ == "__main__":
    unittest.main()
