"""Tests for pysp.stats.sequences.grammar, pysp.stats.sequences.markov_transform, and pysp.stats.sequences.sparse_markov_transform.

Covers: clean grammar module imports without cnrg, the in-tree grammar accumulator/sampler path, the
GrammarDistribution.estimator() fix, markov_transform sample/estimate smoke on tiny data, and DataSequenceEncoder
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
    from pysp.stats.combinator.composite import CompositeDistribution
    from pysp.stats.sequences.markov_transform import MarkovTransformDistribution
    from pysp.stats.univariate.discrete.categorical import CategoricalDistribution

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
    from pysp.stats.combinator.composite import CompositeDistribution
    from pysp.stats.sequences.sparse_markov_transform import SparseMarkovAssociationDistribution
    from pysp.stats.univariate.discrete.categorical import CategoricalDistribution

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
        mod = importlib.import_module("pysp.stats.sequences.markov_transform")
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
        mod = importlib.import_module("pysp.stats.sequences.sparse_markov_transform")
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
        mod = importlib.import_module("pysp.stats.sequences.grammar")
        for name in (
            "GrammarDistribution",
            "GrammarSampler",
            "GrammarEstimatorAccumulator",
            "GrammarAccumulatorFactory",
            "GrammarEstimator",
            "GrammarDataEncoder",
        ):
            self.assertTrue(hasattr(mod, name), name)


@unittest.skipUnless(HAS_NETWORKX, "networkx is not installed")
class GrammarTestCase(unittest.TestCase):
    @staticmethod
    def _grammar():
        from pysp.stats.sequences.grammar import GrammarRule, VertexReplacementGrammar

        graph = nx.Graph()
        graph.add_node(0, label="A", node_color="")
        graph.add_node(1, label="B", node_color="")
        graph.add_edge(0, 1, weight=1.0, edge_color="")
        grammar = VertexReplacementGrammar(name="tiny")
        grammar.add_rule(GrammarRule(2, graph, frequency=3.0))
        return grammar

    def test_estimator_does_not_pass_distribution_as_pseudo_count(self):
        from pysp.stats.sequences.grammar import GrammarDistribution, GrammarEstimator

        dist = GrammarDistribution(None, 0.01, name="g")
        est = dist.estimator()
        self.assertIsInstance(est, GrammarEstimator)
        self.assertIsNone(est.pseudo_count)
        self.assertEqual(est.name, "g")

        est2 = dist.estimator(pseudo_count=2.0)
        self.assertEqual(est2.pseudo_count, 2.0)

    def test_accumulator_factory_and_alias(self):
        from pysp.stats.sequences.grammar import GrammarAccumulatorFactory, GrammarEstimator

        est = GrammarEstimator()
        self.assertIsInstance(est.accumulator_factory(), GrammarAccumulatorFactory)
        self.assertIsInstance(est.accumulatorFactory(), GrammarAccumulatorFactory)

    def test_encoder_equality(self):
        from pysp.stats.sequences.grammar import GrammarDataEncoder, GrammarDistribution

        dist = GrammarDistribution(None, 0.01)
        enc = dist.dist_to_encoder()
        self.assertEqual(enc, GrammarDataEncoder())
        self.assertNotEqual(enc, object())
        data = ["a", "b"]
        self.assertEqual(enc.seq_encode(data), data)
        self.assertEqual(str(enc), "GrammarDataEncoder")

    def test_in_tree_grammar_accumulates_scores_and_samples_without_cnrg(self):
        from pysp.stats.sequences.grammar import (
            GrammarDistribution,
            GrammarEstimator,
            GrammarEstimatorAccumulator,
            GrammarSampler,
        )

        grammar = self._grammar()
        dist = GrammarDistribution(grammar, 0.01)
        self.assertTrue(np.isfinite(dist.log_density(grammar)))

        acc = GrammarEstimatorAccumulator()
        acc.update(grammar, 2.0, None)
        fitted = GrammarEstimator().estimate(None, acc.value())
        self.assertIsInstance(fitted, GrammarDistribution)
        self.assertEqual(fitted.grammar.rule_dict[2][0].frequency, 6.0)

        sampler = GrammarSampler(grammar, orig_n=4, seed=1)
        graph = sampler.sample()
        self.assertGreaterEqual(graph.number_of_nodes(), 4)
        graphs = sampler.sample_seq([2, 3])
        self.assertEqual(len(graphs), 2)
        self.assertTrue(all(g.number_of_nodes() >= 2 for g in graphs))

    @staticmethod
    def _labeled_edge():
        g = nx.Graph()
        g.add_node(0, label="A", node_color="")
        g.add_node(1, label="B", node_color="")
        g.add_edge(0, 1, weight=1.0, edge_color="")
        return g

    def test_log_density_is_a_valid_log_probability(self):
        from pysp.stats.sequences.grammar import GrammarDistribution, VertexReplacementGrammar

        dist = GrammarDistribution(self._grammar(), 0.05)
        self.assertLessEqual(dist.log_density(self._grammar()), 1e-9)  # a real probability: log <= 0
        self.assertEqual(dist.log_density(VertexReplacementGrammar()), 0.0)  # empty product over rules

    def test_matching_rule_scores_higher_than_non_matching(self):
        from pysp.stats.sequences.grammar import GrammarDistribution, GrammarRule, VertexReplacementGrammar

        dist = GrammarDistribution(self._grammar(), 0.05)
        match = VertexReplacementGrammar()
        match.add_rule(GrammarRule(2, self._labeled_edge(), 1.0))
        star = nx.Graph()
        for i in range(4):
            star.add_node(i, label="Z", node_color="")
        for i in (1, 2, 3):
            star.add_edge(0, i, weight=1.0, edge_color="")
        miss = VertexReplacementGrammar()
        miss.add_rule(GrammarRule(2, star, 1.0))
        self.assertGreater(dist.log_density(match), dist.log_density(miss))

    def test_decomposition_matches_a_disconnected_rule(self):
        from pysp.stats.sequences.grammar import GrammarDistribution, GrammarRule, VertexReplacementGrammar

        two_edges = nx.disjoint_union(self._labeled_edge(), self._labeled_edge())
        obs = VertexReplacementGrammar()
        obs.add_rule(GrammarRule(4, two_edges, 1.0))
        without = GrammarDistribution(self._grammar(), mix_p=0.0, decomp_level=0)
        with_decomp = GrammarDistribution(self._grammar(), mix_p=0.0, decomp_level=2, lhs_delta=2)
        self.assertEqual(without.log_density(obs), float("-inf"))  # no direct match, no background
        self.assertTrue(np.isfinite(with_decomp.log_density(obs)))  # matched component-by-component

    def test_sampler_generates_a_connected_graph(self):
        from pysp.stats.sequences.grammar import GrammarSampler

        g = GrammarSampler(self._grammar(), orig_n=6, seed=1).sample()
        self.assertGreaterEqual(g.number_of_nodes(), 6)
        self.assertTrue(nx.is_connected(g))

    def test_accumulator_keeps_num_rules_consistent(self):
        from pysp.stats.sequences.grammar import GrammarEstimatorAccumulator, GrammarRule, VertexReplacementGrammar

        other = VertexReplacementGrammar()
        other.add_rule(GrammarRule(5, self._labeled_edge(), 2.0))
        acc = GrammarEstimatorAccumulator()
        acc.update(self._grammar(), 2.0, None)
        acc.update(other, 1.0, None)
        g = acc.value()
        self.assertEqual(g.num_rules, len(g.rule_list))
        self.assertEqual(g.num_rules, 2)


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
        from pysp.stats.sequences.markov_transform import MarkovTransformDataEncoder

        enc1 = self.dist.dist_to_encoder()
        enc2 = self.dist.dist_to_encoder()
        self.assertEqual(enc1, enc2)
        self.assertNotEqual(enc1, MarkovTransformDataEncoder(len_encoder=None))
        self.assertNotEqual(enc1, object())
        self.assertIn("MarkovTransformDataEncoder", str(enc1))

    def test_estimate_smoke(self):
        from pysp.stats.sequences.markov_transform import (
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
        from pysp.stats.sequences.markov_transform import MarkovTransformEstimator

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
