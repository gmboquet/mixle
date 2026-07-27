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
        from mixle.stats.graphs.hyperedge_replacement_grammar import (
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
        from mixle.stats.graphs.hyperedge_replacement_grammar import generate_graph

        for sd in range(5):
            g = generate_graph(self._path_grammar(), "S", target_n=6, rng=np.random.RandomState(sd))
            self.assertGreaterEqual(g.number_of_nodes(), 1)
            self.assertTrue(g.number_of_nodes() < 2 or nx.is_connected(g))
            self.assertTrue(all(d <= 2 for _, d in g.degree()))  # the path grammar yields paths

    def test_log_density_is_the_parse_based_marginal(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import best_derivation, marginal_log_prob

        g = self._path_grammar()
        # X-Y derives one way: S(1) * grow(5/6) * stop(1/6)
        self.assertAlmostEqual(marginal_log_prob(self._path(2), g, "S"), float(np.log(5 / 6 * 1 / 6)), places=9)
        self.assertAlmostEqual(marginal_log_prob(self._path(3), g, "S"), float(np.log(5 / 6 * 5 / 6 * 1 / 6)), places=9)
        # single parse here -> marginal == Viterbi
        self.assertAlmostEqual(
            marginal_log_prob(self._path(3), g, "S"), best_derivation(self._path(3), g, "S")[0], places=9
        )

    def test_generated_graphs_parse_and_foreign_is_neg_inf(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import generate_graph, marginal_log_prob

        g = self._path_grammar()
        for sd in (3, 4, 5):
            graph = generate_graph(g, "S", target_n=6, rng=np.random.RandomState(sd))
            self.assertTrue(np.isfinite(marginal_log_prob(graph, g, "S")))
        triangle = nx.Graph()
        for i in range(3):
            self._node(triangle, i, "X")
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            self._edge(triangle, a, b)
        self.assertEqual(marginal_log_prob(triangle, g, "S"), float("-inf"))

    def test_distribution_is_a_lower_bound_density_over_graphs(self):
        import mixle.capability as cap
        from mixle.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(self._path_grammar(), start_symbol="S", orig_n=6)
        g = dist.sampler(seed=4).sample()
        self.assertIsInstance(g, nx.Graph)
        self.assertTrue(np.isfinite(dist.log_density(g)))
        self.assertFalse(cap.supports(dist, cap.ExactDensity))  # parse can truncate -> lower bound
        value, exact = dist.log_density(g, with_status=True)  # small graph: exact
        self.assertTrue(exact)
        self.assertEqual(len(dist.sampler(seed=2).sample(3)), 3)  # sample(size) honors the contract

    def test_estimator_recovers_frequencies_by_parsing(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(self._path_grammar(), start_symbol="S", orig_n=4)
        acc = dist.estimator().accumulator_factory().make()
        for _ in range(5):
            for nodes in (1, 2, 3, 4):
                acc.update(self._path(nodes), 1.0, None)
        fit = dist.estimator().estimate(None, acc.value())
        grow = sum(r.frequency for r in fit.grammar.rule_dict["A"] if r.rhs.graph.number_of_nodes() == 2)
        stop = sum(r.frequency for r in fit.grammar.rule_dict["A"] if r.rhs.graph.number_of_nodes() == 1)
        self.assertGreater(grow, stop)  # grow fires many times per graph, stop once

    def test_serialization_round_trip(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(self._path_grammar(), start_symbol="S", orig_n=6)
        loaded = HyperedgeReplacementGrammarDistribution.from_json(dist.to_json())
        self.assertAlmostEqual(loaded.log_density(self._path(3)), dist.log_density(self._path(3)), places=12)

    def test_active_hyperedge_selection_keeps_marginal_normalized(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import (
            HyperedgeReplacementGrammar,
            HyperedgeReplacementGrammarDistribution,
            HyperedgeReplacementRule,
            Hypergraph,
        )

        start = nx.Graph()
        self._node(start, 0, "X")
        self._node(start, 1, "Y")
        stop_a = nx.Graph()
        stop_a.add_node(0)
        stop_b = nx.Graph()
        stop_b.add_node(0)
        grammar = HyperedgeReplacementGrammar()
        grammar.add_rule(HyperedgeReplacementRule("S", Hypergraph(start, [("A", (0,)), ("B", (1,))]), ()))
        grammar.add_rule(HyperedgeReplacementRule("A", Hypergraph(stop_a), (0,)))
        grammar.add_rule(HyperedgeReplacementRule("B", Hypergraph(stop_b), (0,)))
        observed = nx.Graph()
        self._node(observed, 0, "X")
        self._node(observed, 1, "Y")
        dist = HyperedgeReplacementGrammarDistribution(grammar, start_symbol="S")
        self.assertAlmostEqual(dist.log_density(observed), 0.0, places=12)

    def test_sampling_budget_never_changes_the_production_law(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import (
            HRGSamplingTruncated,
            HyperedgeReplacementGrammar,
            HyperedgeReplacementRule,
            Hypergraph,
            generate_graph,
        )

        start = nx.Graph()
        self._node(start, 0, "X")
        grow = nx.Graph()
        grow.add_node(0)
        self._node(grow, 1, "Y")
        self._edge(grow, 0, 1)
        stop = nx.Graph()
        stop.add_node(0)
        grammar = HyperedgeReplacementGrammar()
        grammar.add_rule(HyperedgeReplacementRule("S", Hypergraph(start, [("A", (0,))]), ()))
        grammar.add_rule(HyperedgeReplacementRule("A", Hypergraph(grow, [("A", (1,))]), (0,), 1.0))
        grammar.add_rule(HyperedgeReplacementRule("A", Hypergraph(stop), (0,), 1.0))
        steps = [
            generate_graph(
                grammar,
                "S",
                target_n=1,
                rng=np.random.RandomState(seed),
                with_receipt=True,
            )[1].steps
            for seed in range(20)
        ]
        self.assertIn(2, steps)
        self.assertTrue(any(step > 2 for step in steps))

        nonterminating = HyperedgeReplacementGrammar()
        nonterminating.add_rule(HyperedgeReplacementRule("S", Hypergraph(start, [("A", (0,))]), ()))
        nonterminating.add_rule(HyperedgeReplacementRule("A", Hypergraph(grow, [("A", (1,))]), (0,)))
        with self.assertRaises(HRGSamplingTruncated) as caught:
            generate_graph(nonterminating, "S", target_n=1, rng=np.random.RandomState(0))
        self.assertGreater(caught.exception.receipt.active_hyperedges, 0)

    def test_induced_matching_rejects_extra_terminal_edges(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import (
            HyperedgeReplacementGrammar,
            HyperedgeReplacementGrammarDistribution,
            HyperedgeReplacementRule,
            Hypergraph,
        )

        rhs = nx.Graph()
        self._node(rhs, 0, "X")
        self._node(rhs, 1, "Y")
        grammar = HyperedgeReplacementGrammar()
        grammar.add_rule(HyperedgeReplacementRule("S", Hypergraph(rhs), ()))
        dist = HyperedgeReplacementGrammarDistribution(grammar, start_symbol="S")
        self.assertEqual(dist.log_density(rhs), 0.0)
        extra = rhs.copy()
        self._edge(extra, 0, 1)
        self.assertEqual(dist.log_density(extra), float("-inf"))

    def test_rank_attachment_and_empty_rhs_contracts(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import (
            HyperedgeReplacementGrammar,
            HyperedgeReplacementGrammarDistribution,
            HyperedgeReplacementRule,
            Hypergraph,
        )

        with self.assertRaises(ValueError):
            Hypergraph(nx.Graph(), [("A", (0,))])
        with self.assertRaises(ValueError):
            HyperedgeReplacementRule("S", Hypergraph(), ())

        boundary = nx.Graph()
        boundary.add_node(0)
        grammar = HyperedgeReplacementGrammar()
        grammar.add_rule(HyperedgeReplacementRule("S", Hypergraph(boundary), (0,)))
        with self.assertRaises(ValueError):
            HyperedgeReplacementGrammarDistribution(grammar, start_symbol="S")

        malformed_start = nx.Graph()
        self._node(malformed_start, 0, "X")
        self._node(malformed_start, 1, "Y")
        malformed = HyperedgeReplacementGrammar()
        malformed.add_rule(
            HyperedgeReplacementRule("S", Hypergraph(malformed_start, [("A", (0, 1))]), ())
        )
        malformed.add_rule(HyperedgeReplacementRule("A", Hypergraph(boundary), (0,)))
        with self.assertRaises(ValueError):
            HyperedgeReplacementGrammarDistribution(malformed, start_symbol="S")

    def test_estimation_is_owned_idempotent_and_fail_closed(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        dist = HyperedgeReplacementGrammarDistribution(
            self._path_grammar(),
            start_symbol="S",
            orig_n=7,
            name="owned",
            keys="shared",
        )
        estimator = dist.estimator(pseudo_count=1.0)
        acc = estimator.accumulator_factory().make()
        with self.assertRaises(ValueError):
            acc.update(self._path(2), -1.0, None)
        with self.assertRaises(ValueError):
            acc.seq_update([self._path(2)], np.asarray([1.0, 2.0]), None)
        acc.update(self._path(2), 1.0, None)
        statistics = acc.value()
        owned = acc.value()
        owned.counts.rule_dict["A"][0].frequency = 99.0
        self.assertEqual(acc.value().counts.rule_dict["A"][0].frequency, 1.0)
        first = estimator.estimate(None, statistics)
        second = estimator.estimate(None, statistics)
        first_freqs = [rule.frequency for rule in first.grammar.rule_dict["A"]]
        second_freqs = [rule.frequency for rule in second.grammar.rule_dict["A"]]
        self.assertEqual(first_freqs, second_freqs)
        self.assertEqual([rule.frequency for rule in statistics.counts.rule_dict["A"]], [1.0, 1.0])
        self.assertEqual(first.orig_n, 7)
        self.assertEqual(first.keys, "shared")

        foreign = nx.Graph()
        self._node(foreign, 0, "Z")
        with self.assertRaises(ValueError):
            acc.update(foreign, 1.0, None)
        self.assertEqual(acc.receipt().rejected_weight, 1.0)
        with self.assertRaises(ValueError):
            estimator.estimate(None, acc.value())

    def test_distribution_owns_grammar_state(self):
        from mixle.stats.graphs.hyperedge_replacement_grammar import HyperedgeReplacementGrammarDistribution

        grammar = self._path_grammar()
        dist = HyperedgeReplacementGrammarDistribution(grammar, start_symbol="S")
        before = dist.log_density(self._path(2))
        grammar.rule_dict["A"][0].frequency = 0.0
        grammar.rule_dict["A"][0].rhs.graph.clear()
        self.assertEqual(dist.log_density(self._path(2)), before)
        exposed = dist.grammar
        exposed.rule_dict["A"][0].frequency = 0.0
        self.assertEqual(dist.log_density(self._path(2)), before)


if __name__ == "__main__":
    unittest.main()
