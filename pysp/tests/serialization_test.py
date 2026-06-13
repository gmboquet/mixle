import json
import unittest

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

import pysp.bstats as bstats
import pysp.stats as stats
from pysp.bstats.pdist import ParameterEstimator as BStatsParameterEstimator
from pysp.bstats.pdist import ProbabilityDistribution as BStatsProbabilityDistribution
from pysp.stats.select import SelectDistribution
from pysp.utils.serialization import (
    SerializationError,
    from_serializable,
    register_serializable_callable,
    serializable_class_ids,
    to_json,
)


def _select_by_type(x):
    return 0 if isinstance(x, str) else 1


class DistributionSerializationTestCase(unittest.TestCase):

    def assert_stats_roundtrip(self, dist, probes):
        loaded = type(dist).from_json(dist.to_json())
        self.assertIsInstance(loaded, type(dist))
        for probe in probes:
            self.assertAlmostEqual(loaded.log_density(probe), dist.log_density(probe), places=12)
        return loaded

    def assert_bstats_roundtrip(self, dist, probes):
        loaded = type(dist).from_json(dist.to_json())
        self.assertIsInstance(loaded, type(dist))
        for probe in probes:
            self.assertAlmostEqual(loaded.log_density(probe), dist.log_density(probe), places=12)
            self.assertAlmostEqual(loaded.expected_log_density(probe), dist.expected_log_density(probe), places=12)
        return loaded

    def assert_estimator_roundtrip(self, estimator):
        loaded = type(estimator).from_json(estimator.to_json())
        self.assertIsInstance(loaded, type(estimator))
        self.assertEqual(loaded.__dict__.keys(), estimator.__dict__.keys())
        return loaded

    def test_stats_json_round_trip_representative_models(self):
        cat = stats.CategoricalDistribution({('a', 1): 0.7, 'b': 0.3})
        cat_loaded = self.assert_stats_roundtrip(cat, [('a', 1), 'b', 'missing'])
        self.assertEqual(cat_loaded.pmap[('a', 1)], 0.7)

        mix = stats.MixtureDistribution([
            stats.GaussianDistribution(0.0, 1.0),
            stats.GaussianDistribution(2.0, 3.0),
        ], [0.4, 0.6], name='m')
        self.assert_stats_roundtrip(mix, [-1.0, 0.5, 3.0])

        comp = stats.CompositeDistribution((
            stats.CategoricalDistribution({'x': 0.25, 'y': 0.75}),
            stats.GaussianDistribution(1.0, 2.0),
        ))
        self.assert_stats_roundtrip(comp, [('x', 0.25), ('y', 2.0)])

        seq = stats.SequenceDistribution(
            stats.CategoricalDistribution({'a': 0.6, 'b': 0.4}),
            len_dist=stats.CategoricalDistribution({0: 0.1, 2: 0.9}),
        )
        self.assert_stats_roundtrip(seq, [[], ['a', 'b']])

        transform = stats.TransformDistribution(
            stats.GaussianDistribution(0.0, 1.0),
            transform=stats.AffineTransform(loc=2.0, scale=3.0),
            name='affine',
            keys='k',
        )
        loaded = self.assert_stats_roundtrip(transform, [2.0, 5.0])
        self.assertIsInstance(loaded.transform, stats.AffineTransform)
        self.assertEqual(loaded.transform.loc, 2.0)
        self.assertEqual(loaded.transform.scale, 3.0)

    def test_stats_json_round_trip_cached_structures(self):
        markov = stats.MarkovChainDistribution(
            {'a': 1.0},
            {'a': {'b': 1.0}},
            len_dist=stats.CategoricalDistribution({2: 1.0}),
        )
        markov_loaded = self.assert_stats_roundtrip(markov, [['a', 'b']])
        self.assertEqual(markov_loaded.all_vals, {'a', 'b'})
        self.assertEqual(markov_loaded.trans_log_pvec.getformat(), markov.trans_log_pvec.getformat())

        tree = stats.ICLTreeDistribution(
            [None, 0],
            [np.log(np.asarray([0.4, 0.6])), np.log(np.asarray([[0.7, 0.3], [0.2, 0.8]]))],
        )
        tree_loaded = self.assert_stats_roundtrip(tree, [[0, 1], [1, 1]])
        self.assertIsInstance(tree_loaded.feature_order, range)

        sparse = stats.SparseMarkovAssociationDistribution(
            [0.4, 0.6],
            csr_matrix(np.asarray([[0.9, 0.1], [0.2, 0.8]])),
            alpha=0.1,
        )
        sparse_loaded = self.assert_stats_roundtrip(sparse, [([(0, 1.0)], [(1, 1.0)])])
        self.assertEqual(sparse_loaded.cond_prob_mat.getformat(), 'csr')

    def test_grammar_distribution_json_round_trip(self):
        from pysp.stats.grammar import GrammarDistribution, GrammarRule, VRG

        graph = nx.Graph()
        graph.add_node(0, label='A', node_color='')
        graph.add_node(1, label='B', node_color='')
        graph.add_edge(0, 1, weight=1.0, edge_color='')
        grammar = VRG(name='json')
        grammar.add_rule(GrammarRule(2, graph, frequency=3.0))

        dist = GrammarDistribution(grammar, 0.01, orig_n=4)
        loaded = GrammarDistribution.from_json(dist.to_json())

        self.assertIsInstance(loaded.grammar, VRG)
        self.assertEqual(loaded.grammar.name, 'json')
        self.assertEqual(len(loaded.grammar.rule_list), 1)
        self.assertAlmostEqual(loaded.log_density(grammar), dist.log_density(grammar), places=12)

    def test_bstats_json_round_trip_representative_models(self):
        cat = bstats.CategoricalDistribution({'x': 0.7, 'y': 0.3}, name='c')
        self.assert_bstats_roundtrip(cat, ['x', 'y'])

        mix = bstats.MixtureDistribution([
            bstats.GaussianDistribution(0.0, 1.0),
            bstats.GaussianDistribution(2.0, 3.0),
        ], [0.25, 0.75], name='bm')
        self.assert_bstats_roundtrip(mix, [-0.5, 1.25])

        dpm = bstats.DirichletProcessMixtureDistribution(
            [
                bstats.GaussianDistribution(0.0, 1.0),
                bstats.GaussianDistribution(3.0, 2.0),
            ],
            np.asarray([0.55, 0.45]),
            1.5,
            np.asarray([[2.0, 3.0], [1.0, 1.0]]),
            [
                bstats.GaussianDistribution(0.0, 1.0).get_prior(),
                bstats.GaussianDistribution(3.0, 2.0).get_prior(),
            ],
            name='dpm',
        )
        loaded = self.assert_bstats_roundtrip(dpm, [0.0, 2.0])
        np.testing.assert_allclose(loaded.g, dpm.g)
        self.assertEqual(len(loaded.component_priors), 2)

    def test_stats_estimator_json_round_trip_representative_models(self):
        mix = stats.MixtureEstimator([
            stats.GaussianEstimator(name='g0', keys='gk0'),
            stats.GaussianEstimator(name='g1', keys='gk1'),
        ], fixed_weights=np.asarray([0.25, 0.75]), name='mix', keys=('wk', 'ck'))
        mix_loaded = self.assert_estimator_roundtrip(mix)
        np.testing.assert_allclose(mix_loaded.fixed_weights, mix.fixed_weights)
        self.assertEqual([type(u) for u in mix_loaded.estimators], [type(u) for u in mix.estimators])
        self.assertIsNotNone(mix_loaded.accumulator_factory().make())

        comp = stats.CompositeEstimator((
            stats.CategoricalEstimator(pseudo_count=1.0, suff_stat={'x': 2.0}, name='cat'),
            stats.GaussianEstimator(name='g'),
        ))
        comp_loaded = self.assert_estimator_roundtrip(comp)
        self.assertEqual(len(comp_loaded.estimators), 2)
        self.assertIsNotNone(comp_loaded.accumulator_factory().make())

        seq = stats.SequenceEstimator(
            stats.CategoricalEstimator(pseudo_count=0.5, suff_stat={'a': 1.0}),
            len_estimator=stats.CategoricalEstimator(pseudo_count=0.5, suff_stat={0: 1.0, 2: 3.0}),
            len_normalized=True,
            name='seq',
        )
        seq_loaded = self.assert_estimator_roundtrip(seq)
        self.assertTrue(seq_loaded.len_normalized)
        self.assertIsNotNone(seq_loaded.accumulator_factory().make())

        transform = stats.TransformEstimator(
            stats.GaussianEstimator(),
            transform=stats.AffineTransform(loc=2.0, scale=3.0),
            density_correction=True,
            name='affine',
            keys='k',
        )
        transform_loaded = self.assert_estimator_roundtrip(transform)
        self.assertIsInstance(transform_loaded.transform, stats.AffineTransform)

    def test_bstats_estimator_json_round_trip_representative_models(self):
        mix = bstats.MixtureEstimator([
            bstats.GaussianEstimator(name='g0'),
            bstats.GaussianEstimator(name='g1'),
        ], fixed_w=np.asarray([0.4, 0.6]), name='bm')
        mix_loaded = self.assert_estimator_roundtrip(mix)
        np.testing.assert_allclose(mix_loaded.fixed_w, mix.fixed_w)
        self.assertIsNotNone(mix_loaded.accumulator_factory().make())

        dpm = bstats.DirichletProcessMixtureEstimator([
            bstats.GaussianEstimator(name='g0'),
            bstats.GaussianEstimator(name='g1'),
        ], name='dpm')
        dpm_loaded = self.assert_estimator_roundtrip(dpm)
        self.assertEqual(dpm_loaded.num_components, 2)
        self.assertIsNotNone(dpm_loaded.accumulator_factory().make())

        intrange = bstats.IntegerCategoricalEstimator(min_index=2, max_index=5, default_value=0.1, name='ir')
        intrange_loaded = self.assert_estimator_roundtrip(intrange)
        self.assertEqual(intrange_loaded.minVal, 2)
        self.assertEqual(intrange_loaded.maxVal, 5)
        self.assertIsNotNone(intrange_loaded.accumulator_factory().make())

    def test_select_estimator_requires_registered_callable(self):
        unsafe = stats.SelectEstimator([
            stats.CategoricalEstimator(suff_stat={'a': 1.0}),
            stats.CategoricalEstimator(suff_stat={1: 1.0}),
        ], lambda x: 0)
        with self.assertRaises(SerializationError):
            unsafe.to_json()

        register_serializable_callable(_select_by_type, 'pysp.tests.serialization_test.select_by_type')
        safe = stats.SelectEstimator([
            stats.CategoricalEstimator(suff_stat={'a': 1.0}),
            stats.CategoricalEstimator(suff_stat={1: 1.0}),
        ], _select_by_type)
        loaded = stats.SelectEstimator.from_json(safe.to_json())
        self.assertIs(loaded.choice_function, _select_by_type)
        self.assertEqual(len(loaded.estimators), 2)

    def test_live_samplers_and_enumerators_are_not_json_serialized(self):
        with self.assertRaises(SerializationError):
            to_json(stats.GaussianDistribution(0.0, 1.0).sampler(seed=1))

        with self.assertRaises(SerializationError):
            to_json(stats.CategoricalDistribution({'a': 1.0}).enumerator())

    def test_package_helpers_use_json_and_support_collections(self):
        models = [
            stats.GaussianDistribution(0.0, 1.0),
            stats.CategoricalDistribution({'a': 1.0}),
        ]
        text = stats.dump_models(models)
        self.assertEqual(json.loads(text)[0]['__pysp_type__'], 'object')
        loaded = stats.load_models(text)
        self.assertEqual([type(u) for u in loaded], [type(u) for u in models])

        bmodel = bstats.GaussianDistribution(0.0, 1.0)
        bloaded = bstats.load_models(bstats.dump_models(bmodel))
        self.assertIsInstance(bloaded, bstats.GaussianDistribution)

        estimator = stats.MixtureEstimator([stats.GaussianEstimator(), stats.GaussianEstimator()])
        eloaded = stats.load_models(stats.dump_models(estimator))
        self.assertIsInstance(eloaded, stats.MixtureEstimator)

    def test_json_is_strict_and_not_eval_based(self):
        text = stats.CategoricalDistribution({'a': 1.0}).to_json()
        self.assertNotIn('Infinity', text)
        self.assertNotIn('NaN', text)
        self.assertEqual(stats.CategoricalDistribution.from_json(text).log_density('missing'), -np.inf)

        with self.assertRaises(Exception):
            stats.load_models("__import__('os').system('echo unsafe')")

        payload = {'__pysp_type__': 'object', 'type': 'os.system',
                   'state': {'__pysp_type__': 'dict', 'items': []}}
        with self.assertRaises(SerializationError):
            from_serializable(payload)

    def test_select_distribution_requires_registered_callable(self):
        dist = SelectDistribution([
            stats.CategoricalDistribution({'a': 1.0}),
            stats.CategoricalDistribution({1: 1.0}),
        ], lambda x: 0)
        with self.assertRaises(SerializationError):
            dist.to_json()

        register_serializable_callable(_select_by_type, 'pysp.tests.serialization_test.select_by_type')
        safe = SelectDistribution([
            stats.CategoricalDistribution({'a': 1.0}),
            stats.CategoricalDistribution({1: 1.0}),
        ], _select_by_type)
        loaded = SelectDistribution.from_json(safe.to_json())
        self.assertEqual(loaded.log_density('a'), 0.0)
        self.assertEqual(loaded.log_density(1), 0.0)
        self.assertIs(loaded.choice_function, _select_by_type)

    def test_registry_covers_pysp_distribution_and_estimator_subclasses(self):
        ids = serializable_class_ids()

        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        checked = 0
        for base in (stats.SequenceEncodableProbabilityDistribution, BStatsProbabilityDistribution,
                     stats.ParameterEstimator, BStatsParameterEstimator):
            for cls in subclasses(base):
                if cls.__module__.startswith(('pysp.stats.', 'pysp.bstats.', 'pysp.utils.automatic')):
                    self.assertIn('%s.%s' % (cls.__module__, cls.__name__), ids)
                    checked += 1
        self.assertGreater(checked, 50)


if __name__ == '__main__':
    unittest.main()
