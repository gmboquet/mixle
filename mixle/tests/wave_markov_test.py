"""Tests for mixle.stats.graphs.vertex_replacement_grammar, mixle.stats.sequences.markov_transform, and mixle.stats.sequences.sparse_markov_transform.

Covers: clean grammar module imports without cnrg, the in-tree grammar accumulator/sampler path, the
VertexReplacementGrammarDistribution.estimator() fix, markov_transform sample/estimate smoke on tiny data, and DataSequenceEncoder
equality / encode round-trip consistency.
"""

import importlib
import importlib.util
import unittest
import warnings

import numpy as np

HAS_NETWORKX = importlib.util.find_spec("networkx") is not None
if HAS_NETWORKX:
    import networkx as nx


def _make_markov_transform_dist(alpha=0.05, with_len=True):
    from mixle.stats.combinator.composite import CompositeDistribution
    from mixle.stats.sequences.markov_transform import MarkovTransformDistribution
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    nw = 3
    init_prob = np.asarray([0.5, 0.3, 0.2])
    rng = np.random.RandomState(7)
    cond_prob = rng.rand(nw * nw, nw) + 0.1
    cond_prob /= cond_prob.sum(axis=1, keepdims=True)

    if with_len:
        len_dist = CompositeDistribution(
            (
                CategoricalDistribution({2: 0.5, 3: 0.5}),
                CategoricalDistribution({2: 0.5, 3: 0.5}),
                CategoricalDistribution({3: 0.6, 4: 0.4}),
            )
        )
    else:
        len_dist = None

    return MarkovTransformDistribution(init_prob, cond_prob, alpha=alpha, len_dist=len_dist)


def _make_sparse_assoc_dist(low_memory=False):
    from mixle.stats.combinator.composite import CompositeDistribution
    from mixle.stats.sequences.sparse_markov_transform import SparseMarkovAssociationDistribution
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    nw = 3
    init_prob = np.asarray([0.5, 0.3, 0.2])
    rng = np.random.RandomState(11)
    cond_prob = rng.rand(nw, nw) + 0.1
    cond_prob /= cond_prob.sum(axis=1, keepdims=True)

    len_dist = CompositeDistribution(
        (CategoricalDistribution({2: 0.5, 3: 0.5}), CategoricalDistribution({3: 0.6, 4: 0.4}))
    )

    return SparseMarkovAssociationDistribution(
        init_prob, cond_prob, alpha=0.1, len_dist=len_dist, low_memory=low_memory
    )


class ImportTestCase(unittest.TestCase):
    def test_markov_transform_imports(self):
        mod = importlib.import_module("mixle.stats.sequences.markov_transform")
        for name in (
            "MarkovTransformDistribution",
            "MarkovTransformSampler",
            "MarkovTransformAccumulator",
            "MarkovTransformAccumulatorFactory",
            "MarkovTransformEstimator",
            "MarkovTransformDataEncoder",
        ):
            self.assertTrue(hasattr(mod, name), name)

    def test_sparse_markov_transform_imports(self):
        mod = importlib.import_module("mixle.stats.sequences.sparse_markov_transform")
        for name in (
            "SparseMarkovAssociationDistribution",
            "SparseMarkovAssociationSampler",
            "SparseMarkovAssociationAccumulator",
            "SparseMarkovAssociationAccumulatorFactory",
            "SparseMarkovAssociationEstimator",
            "SparseMarkovAssociationDataEncoder",
        ):
            self.assertTrue(hasattr(mod, name), name)

    def test_grammar_imports_without_cnrg(self):
        # The module must import cleanly even when the optional 'cnrg' package is absent.
        mod = importlib.import_module("mixle.stats.graphs.vertex_replacement_grammar")
        for name in (
            "VertexReplacementGrammarDistribution",
            "VertexReplacementGrammarSampler",
            "VertexReplacementGrammarAccumulator",
            "VertexReplacementGrammarAccumulatorFactory",
            "VertexReplacementGrammarEstimator",
            "VertexReplacementGrammarDataEncoder",
        ):
            self.assertTrue(hasattr(mod, name), name)


@unittest.skipUnless(HAS_NETWORKX, "networkx is not installed")
class GrammarTestCase(unittest.TestCase):
    @staticmethod
    def _grammar():
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammar, VertexReplacementRule

        graph = nx.Graph()
        graph.add_node(0, label="A", node_color="")
        graph.add_node(1, label="B", node_color="")
        graph.add_edge(0, 1, weight=1.0, edge_color="")
        grammar = VertexReplacementGrammar(name="tiny")
        grammar.add_rule(VertexReplacementRule(2, graph, frequency=3.0))
        return grammar

    def test_estimator_does_not_pass_distribution_as_pseudo_count(self):
        from mixle.stats.graphs.vertex_replacement_grammar import (
            VertexReplacementGrammarDistribution,
            VertexReplacementGrammarEstimator,
        )

        dist = VertexReplacementGrammarDistribution(None, 0.01, name="g")
        est = dist.estimator()
        self.assertIsInstance(est, VertexReplacementGrammarEstimator)
        self.assertIsNone(est.pseudo_count)
        self.assertEqual(est.name, "g")

        est2 = dist.estimator(pseudo_count=2.0)
        self.assertEqual(est2.pseudo_count, 2.0)

    def test_accumulator_factory(self):
        from mixle.stats.graphs.vertex_replacement_grammar import (
            VertexReplacementGrammarAccumulatorFactory,
            VertexReplacementGrammarEstimator,
        )

        est = VertexReplacementGrammarEstimator()
        self.assertIsInstance(est.accumulator_factory(), VertexReplacementGrammarAccumulatorFactory)
        with self.assertWarns(DeprecationWarning):  # camelCase alias is deprecated
            self.assertIsInstance(est.accumulatorFactory(), VertexReplacementGrammarAccumulatorFactory)

    def test_encoder_equality(self):
        from mixle.stats.graphs.vertex_replacement_grammar import (
            VertexReplacementGrammarDataEncoder,
            VertexReplacementGrammarDistribution,
        )

        dist = VertexReplacementGrammarDistribution(None, 0.01)
        enc = dist.dist_to_encoder()
        self.assertEqual(enc, VertexReplacementGrammarDataEncoder())
        self.assertNotEqual(enc, object())
        data = ["a", "b"]
        self.assertEqual(enc.seq_encode(data), data)
        self.assertEqual(str(enc), "VertexReplacementGrammarDataEncoder")

    @staticmethod
    def _recursive_grammar(embedding=None):
        # symbol 1 -> [C - T - nonterminal(1)] (grow) | [E] (terminate); optional embedding relation.
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammar, VertexReplacementRule

        grow = nx.Graph()
        grow.add_node(0, label="C", node_color="")
        grow.add_node(1, label="T", node_color="")
        grow.add_node(2, nonterminal=1)
        grow.add_edge(0, 1, weight=1.0, edge_color="")
        grow.add_edge(1, 2, weight=1.0, edge_color="")
        stop = nx.Graph()
        stop.add_node(0, label="E", node_color="")
        g = VertexReplacementGrammar()
        g.add_rule(VertexReplacementRule(1, grow, frequency=5.0, embedding=embedding))
        g.add_rule(VertexReplacementRule(1, stop, frequency=1.0))
        return g

    @staticmethod
    def _labeled(builder):
        # attach the label/color attributes the isomorphism matcher expects to a plain networkx graph
        for v in builder.nodes:
            builder.nodes[v].setdefault("label", "X")
            builder.nodes[v].setdefault("node_color", "")
        for a, b in builder.edges:
            builder.edges[a, b].setdefault("weight", 1.0)
            builder.edges[a, b].setdefault("edge_color", "")
        return builder

    @staticmethod
    def _edge(la, lb):
        g = nx.Graph()
        g.add_node(0, label=la, node_color="")
        g.add_node(1, label=lb, node_color="")
        g.add_edge(0, 1, weight=1.0, edge_color="")
        return g

    def _freq_dist(self):
        # an unambiguous grammar: S -> A-B | A-C | A-D with true frequencies 5 : 3 : 2
        from mixle.stats.graphs.vertex_replacement_grammar import (
            VertexReplacementGrammar,
            VertexReplacementGrammarDistribution,
            VertexReplacementRule,
        )

        g = VertexReplacementGrammar()
        for lb, freq in (("B", 5.0), ("C", 3.0), ("D", 2.0)):
            g.add_rule(VertexReplacementRule("S", self._edge("A", lb), freq))
        return VertexReplacementGrammarDistribution(g, 0.0, start_symbol="S")

    # --- generation: the sampler runs a real derivation and emits graphs ---
    def test_non_recursive_grammar_derives_its_right_hand_side(self):
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammarSampler

        sampler = VertexReplacementGrammarSampler(self._grammar(), orig_n=4, seed=1)  # non-recursive -> exactly the RHS
        self.assertEqual(sampler.sample().number_of_nodes(), 2)
        self.assertTrue(all(g.number_of_nodes() == 2 for g in sampler.sample_seq([2, 3])))

    def test_recursive_derivation_grows_and_stays_connected(self):
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammarSampler

        g = VertexReplacementGrammarSampler(self._recursive_grammar(), orig_n=8, seed=0, start_symbol=1).sample()
        self.assertGreaterEqual(g.number_of_nodes(), 4)  # grew via recursion toward the budget
        self.assertTrue(nx.is_connected(g))
        self.assertTrue(all("nonterminal" not in g.nodes[n] for n in g.nodes))  # fully derived

    def test_embedding_controls_reconnection(self):
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammarSampler

        # the embedding connects a replaced node's 'T' neighbour to right-hand-side 'T' nodes; the
        # default would instead attach to the connector 'C', so 'T'-'T' edges prove the relation fired.
        g = VertexReplacementGrammarSampler(
            self._recursive_grammar(embedding=[("T", "T")]), orig_n=8, seed=2, start_symbol=1
        ).sample()
        tt_edges = [(a, b) for a, b in g.edges if g.nodes[a].get("label") == "T" and g.nodes[b].get("label") == "T"]
        self.assertTrue(tt_edges)
        self.assertTrue(nx.is_connected(g))

    def test_sample_honors_size_argument(self):
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammarSampler

        sampler = VertexReplacementGrammarSampler(self._grammar(), orig_n=2, seed=1)
        self.assertIsInstance(sampler.sample(), nx.Graph)
        batch = sampler.sample(5)
        self.assertEqual(len(batch), 5)
        self.assertTrue(all(isinstance(g, nx.Graph) for g in batch))

    # --- density and estimation: the grammar's own likelihood, computed by PARSING the graph ---
    def test_log_density_is_the_parse_based_likelihood(self):
        dist = self._freq_dist()  # each A-x has a single parse, so marginal == freq(x)/total
        for lb, freq in (("B", 5.0), ("C", 3.0), ("D", 2.0)):
            self.assertAlmostEqual(dist.log_density(self._edge("A", lb)), float(np.log(freq / 10.0)), places=9)
        self.assertEqual(dist.log_density(self._edge("A", "Z")), float("-inf"))  # grammar cannot derive it

    def test_marginal_likelihood_sums_over_derivations(self):
        # ambiguous grammar: A-B derives two ways (directly, or A-nt(T) then T->B), each probability 1/2,
        # so the marginal log-density is log(1/2 + 1/2) = 0 -- strictly above the Viterbi lower bound.
        from mixle.stats.graphs.vertex_replacement_grammar import (
            VertexReplacementGrammar,
            VertexReplacementGrammarDistribution,
            VertexReplacementRule,
            best_derivation,
        )

        a_t = nx.Graph()
        a_t.add_node(0, label="A", node_color="")
        a_t.add_node(1, nonterminal="T")
        a_t.add_edge(0, 1, weight=1.0, edge_color="")
        b = nx.Graph()
        b.add_node(0, label="B", node_color="")
        g = VertexReplacementGrammar()
        g.add_rule(VertexReplacementRule("S", self._edge("A", "B"), 1.0))
        g.add_rule(VertexReplacementRule("S", a_t, 1.0))
        g.add_rule(VertexReplacementRule("T", b, 1.0))
        dist = VertexReplacementGrammarDistribution(g, 0.0, start_symbol="S")
        marginal = dist.log_density(self._edge("A", "B"))
        viterbi = best_derivation(self._edge("A", "B"), g, "S")[0]
        self.assertAlmostEqual(marginal, 0.0, places=9)  # log(1/2 + 1/2)
        self.assertAlmostEqual(viterbi, float(np.log(0.5)), places=9)
        self.assertGreater(marginal, viterbi)

    def test_generated_graphs_parse_and_foreign_graphs_are_neg_inf(self):
        from mixle.stats.graphs.vertex_replacement_grammar import (
            VertexReplacementGrammarDistribution,
            VertexReplacementGrammarSampler,
        )

        recursive = self._recursive_grammar()
        dist = VertexReplacementGrammarDistribution(recursive, 0.0, start_symbol=1)
        for s in range(8):
            g = VertexReplacementGrammarSampler(recursive, orig_n=8, seed=s, start_symbol=1).sample()
            lp = dist.log_density(g)  # a graph the grammar generated must be derivable -> finite
            self.assertTrue(np.isfinite(lp))
            self.assertLessEqual(lp, 1e-9)
        self.assertEqual(dist.log_density(self._labeled(nx.complete_graph(4))), float("-inf"))

    def test_estimator_recovers_frequencies_by_parsing(self):
        # Viterbi parse-counting recovers the rule frequencies the data was generated with.
        dist = self._freq_dist()
        acc = dist.estimator().accumulator_factory().make()
        for s in range(3000):
            acc.update(dist.sampler(seed=s).sample(), 1.0, None)  # parse each graph, count the rule it fires
        fit = dist.estimator().estimate(None, acc.value())
        freqs = np.array([fit.grammar.rule_dict["S"][i].frequency for i in range(3)])
        self.assertLess(np.abs(freqs / freqs.sum() - np.array([0.5, 0.3, 0.2])).sum(), 0.06)

    def test_estimate_driver_integration(self):
        from mixle.inference import estimate
        from mixle.stats.graphs.vertex_replacement_grammar import VertexReplacementGrammarDistribution

        dist = self._freq_dist()
        fit = estimate([dist.sampler(seed=s).sample() for s in range(60)], dist.estimator())
        self.assertIsInstance(fit, VertexReplacementGrammarDistribution)
        self.assertTrue(np.isfinite(fit.log_density(self._edge("A", "B"))))

    def test_seq_log_density_reports_per_row_exactness(self):
        from mixle.stats.compute.pdist import DensitySemantics

        dist = self._freq_dist()
        self.assertIs(dist.density_semantics(), DensitySemantics.LOWER_BOUND)  # static: can truncate
        values, exact = dist.seq_log_density([self._edge("A", "B"), self._edge("A", "Z")], with_status=True)
        self.assertAlmostEqual(values[0], float(np.log(0.5)), places=9)
        self.assertEqual(values[1], float("-inf"))  # not derivable
        self.assertTrue(bool(exact[0]) and bool(exact[1]))  # small graphs: parsed exactly, not truncated
        val, ok = dist.log_density(self._edge("A", "B"), with_status=True)
        self.assertTrue(ok)

    def test_log_density_of_a_sample_is_finite(self):
        # sample() emits a graph and log_density scores a graph -- one sample space.
        dist = self._freq_dist()
        g = dist.sampler(seed=3).sample()
        self.assertIsInstance(g, nx.Graph)
        self.assertTrue(np.isfinite(dist.log_density(g)))

    def test_accumulator_counts_align_with_structure(self):
        dist = self._freq_dist()
        acc = dist.estimator().accumulator_factory().make()
        acc.update(self._edge("A", "B"), 2.0, None)
        acc.update(self._edge("A", "C"), 1.0, None)
        counts = acc.value()
        self.assertEqual(counts.num_rules, len(counts.rule_list))
        self.assertEqual(counts.rule_dict["S"][0].frequency, 2.0)  # the A-B rule fired with weight 2
        self.assertEqual(counts.rule_dict["S"][1].frequency, 1.0)  # the A-C rule fired with weight 1


class MarkovTransformTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.dist = _make_markov_transform_dist(with_len=True)
        self.data = self.dist.sampler(seed=11).sample(size=25)

    def test_sample_structure(self):
        single = self.dist.sampler(seed=3).sample()
        self.assertEqual(len(single), 3)
        for part in single:
            for v, c in part:
                self.assertTrue(0 <= int(v) < self.dist.num_vals)
                self.assertGreater(c, 0)
        self.assertEqual(len(self.data), 25)

    def test_seq_log_density_matches_log_density(self):
        enc = self.dist.dist_to_encoder()
        ex = enc.seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(ex)
        single_ll = np.asarray([self.dist.log_density(u) for u in self.data])
        self.assertTrue(np.all(np.isfinite(seq_ll)))
        self.assertTrue(np.allclose(seq_ll, single_ll))

    def test_legacy_seq_encode_matches_encoder(self):
        # Without a length distribution the legacy distribution method and the encoder must agree.
        dist = _make_markov_transform_dist(with_len=False)
        legacy = dist.seq_encode(self.data)
        modern = dist.dist_to_encoder().seq_encode(self.data)
        self.assertIsNone(legacy[1])
        self.assertIsNone(modern[1])
        self.assertTrue(np.allclose(dist.seq_log_density(legacy), dist.seq_log_density(modern)))

    def test_encoder_equality(self):
        from mixle.stats.sequences.markov_transform import MarkovTransformDataEncoder

        enc1 = self.dist.dist_to_encoder()
        enc2 = self.dist.dist_to_encoder()
        self.assertEqual(enc1, enc2)
        self.assertNotEqual(enc1, MarkovTransformDataEncoder(len_encoder=None))
        self.assertNotEqual(enc1, object())
        self.assertIn("MarkovTransformDataEncoder", str(enc1))

    def test_estimate_smoke(self):
        from mixle.stats.sequences.markov_transform import (
            MarkovTransformAccumulatorFactory,
            MarkovTransformDistribution,
            MarkovTransformEstimator,
        )

        dist = _make_markov_transform_dist(with_len=False)
        enc = dist.dist_to_encoder()
        ex = enc.seq_encode(self.data)
        weights = np.ones(len(self.data))

        est = MarkovTransformEstimator(dist.num_vals, alpha=0.05)
        self.assertIsInstance(est.accumulator_factory(), MarkovTransformAccumulatorFactory)
        with self.assertWarns(DeprecationWarning):  # camelCase alias is deprecated
            self.assertIsInstance(est.accumulatorFactory(), MarkovTransformAccumulatorFactory)

        acc = est.accumulator_factory().make()
        acc.seq_initialize(ex, weights, np.random.RandomState(5))
        model0 = est.estimate(None, acc.value())
        self.assertIsInstance(model0, MarkovTransformDistribution)
        self.assertAlmostEqual(float(np.sum(model0.init_prob_vec)), 1.0, places=8)

        acc2 = est.accumulator_factory().make()
        acc2.seq_update(ex, weights, model0)
        model1 = est.estimate(None, acc2.value())
        self.assertTrue(np.all(np.isfinite(model1.seq_log_density(ex))))

        # accumulator encoders must match the distribution encoder
        self.assertEqual(acc.acc_to_encoder(), enc)

    def test_update_and_initialize_single_obs(self):
        dist = _make_markov_transform_dist(with_len=False)
        from mixle.stats.sequences.markov_transform import MarkovTransformEstimator

        est = MarkovTransformEstimator(dist.num_vals, alpha=0.05)
        acc = est.accumulator_factory().make()
        acc.initialize(self.data[0], 1.0, np.random.RandomState(2))
        acc.update(self.data[1], 1.0, dist)
        init_count, trans_count, size_val = acc.value()
        self.assertGreater(float(np.sum(init_count)), 0.0)
        self.assertGreater(trans_count.sum(), 0.0)
        self.assertIsNone(size_val)

    def test_str_uses_class_name(self):
        self.assertTrue(str(self.dist).startswith("MarkovTransformDistribution("))


class SparseMarkovAssociationTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.dist = _make_sparse_assoc_dist(low_memory=False)
        self.data = self.dist.sampler(seed=21).sample(size=15)

    def test_seq_log_density_matches_log_density(self):
        enc = self.dist.dist_to_encoder()
        ex = enc.seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(ex)
        single_ll = np.asarray([self.dist.log_density(u) for u in self.data])
        self.assertTrue(np.all(np.isfinite(seq_ll)))
        self.assertTrue(np.allclose(seq_ll, single_ll))

    def test_low_memory_encoding_agrees(self):
        dist_lm = _make_sparse_assoc_dist(low_memory=True)
        enc_lm = dist_lm.dist_to_encoder()
        ex_lm = enc_lm.seq_encode(self.data)
        self.assertIsNone(ex_lm[3])
        seq_ll_lm = dist_lm.seq_log_density(ex_lm)

        ex = self.dist.dist_to_encoder().seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(ex)
        self.assertTrue(np.allclose(seq_ll, seq_ll_lm))

    def test_encoder_equality(self):
        enc1 = self.dist.dist_to_encoder()
        enc2 = self.dist.dist_to_encoder()
        self.assertEqual(enc1, enc2)
        enc_lm = _make_sparse_assoc_dist(low_memory=True).dist_to_encoder()
        self.assertNotEqual(enc1, enc_lm)
        self.assertNotEqual(enc1, object())
        self.assertIn("SparseMarkovAssociationDataEncoder", str(enc1))


if __name__ == "__main__":
    unittest.main()
