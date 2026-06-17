"""Coverage tests for public smart enumerators.

These tests cover finite discrete distributions whose enumerators are easy to
cross-check exhaustively, plus package-level exports for public enumerator classes.
"""

import itertools
import unittest

import numpy as np

from pysp.engines import NumpyEngine, TorchEngine, torch
from pysp.stats import *
from pysp.utils.enumeration import freeze

TOL = 1e-9


def assert_matches_brute(test, dist, support, name):
    with np.errstate(divide="ignore"):
        brute = [(v, dist.log_density(v)) for v in support]
    brute = [(v, lp) for v, lp in brute if lp > -np.inf]
    brute.sort(key=lambda u: -u[1])

    items = list(dist.enumerator())
    test.assertEqual(len(items), len(brute), "%s: support size mismatch" % name)

    lps = [lp for _, lp in items]
    for i in range(len(lps) - 1):
        test.assertGreaterEqual(lps[i], lps[i + 1] - TOL, "%s: order violation at %d" % (name, i))

    keys = [freeze(v) for v, _ in items]
    test.assertEqual(len(keys), len(set(keys)), "%s: duplicate values yielded" % name)

    np.testing.assert_allclose(lps, [lp for _, lp in brute], atol=TOL, err_msg="%s: score sequence mismatch" % name)

    brute_by_key = {freeze(v): lp for v, lp in brute}
    for v, lp in items:
        test.assertAlmostEqual(lp, dist.log_density(v), delta=TOL, msg="%s: lp mismatch at %r" % (name, v))
        test.assertAlmostEqual(lp, brute_by_key[freeze(v)], delta=TOL, msg="%s: brute mismatch at %r" % (name, v))


class FiniteEnumeratorCoverageTestCase(unittest.TestCase):
    def test_spearman_ranking_enumerator(self):
        dist = SpearmanRankingDistribution([0.0, 1.0, 2.0], rho=0.7)
        support = [list(p) for p in itertools.permutations(range(3))]
        assert_matches_brute(self, dist, support, "spearman")
        total = np.logaddexp.reduce([lp for _, lp in dist.enumerator()])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_mallows_enumerator(self):
        dist = MallowsDistribution([2, 0, 1], theta=0.8)
        support = [list(p) for p in itertools.permutations(range(3))]
        assert_matches_brute(self, dist, support, "mallows")
        total = np.logaddexp.reduce([lp for _, lp in dist.enumerator()])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_icltree_enumerator(self):
        dist = ICLTreeDistribution([None, 0], [np.log([0.6, 0.4]), np.log([[0.7, 0.3], [0.2, 0.8]])])
        support = [list(v) for v in itertools.product(range(2), repeat=2)]
        assert_matches_brute(self, dist, support, "icltree")
        total = np.logaddexp.reduce([lp for _, lp in dist.enumerator()])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_icltree_string_representation_keeps_name_intact(self):
        dist = ICLTreeDistribution([None, 0], [np.log([0.6, 0.4]), np.log([[0.7, 0.3], [0.2, 0.8]])], name="tree")
        dist_str = str(dist)

        self.assertIn("name='tree'", dist_str)
        self.assertNotIn("name=',t,r,e,e,'", dist_str)

    def test_icltree_backend_metadata_and_scale(self):
        dist = ICLTreeDistribution([None, 0], [np.log([0.6, 0.4]), np.log([[0.7, 0.3], [0.2, 0.8]])])
        data = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0], [1, 1]])
        enc = dist.dist_to_encoder().seq_encode(data)
        expected_scores = dist.seq_log_density(enc)

        np.testing.assert_allclose(
            backend_seq_log_density(dist, enc, NumpyEngine()), expected_scores, rtol=1.0e-12, atol=1.0e-12
        )
        if torch is not None:
            scores = backend_seq_log_density(dist, enc, TorchEngine())
            np.testing.assert_allclose(TorchEngine().to_numpy(scores), expected_scores, rtol=1.0e-12, atol=1.0e-12)

        capabilities = capabilities_for(dist)
        self.assertEqual(capabilities.engine_ready, ("numpy", "torch"))
        self.assertEqual(capabilities.kernel_status, "generic_table")

        declaration = declaration_for(dist)
        self.assertEqual(declaration.name, "integer_chow_liu_tree")
        self.assertEqual(declaration.statistic_names, ("num_features", "num_states", "counts", "marginal_counts"))
        self.assertFalse(declaration.differentiable)

        weights = np.linspace(0.5, 1.5, len(data))
        c = 0.37
        estimator = ICLTreeEstimator()
        acc = estimator.accumulator_factory().make()
        acc.seq_update(enc, weights, None)
        self.assertIs(acc.scale(c), acc)

        expected = estimator.accumulator_factory().make()
        expected.seq_update(enc, weights * c, None)
        actual_value = acc.value()
        expected_value = expected.value()
        self.assertEqual(actual_value[0], expected_value[0])
        self.assertEqual(actual_value[1], expected_value[1])
        np.testing.assert_allclose(actual_value[2], expected_value[2], rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(actual_value[3], expected_value[3], rtol=1.0e-12, atol=1.0e-12)

        scaled_model = estimator.estimate(float(weights.sum() * c), actual_value)
        expected_model = estimator.estimate(float(weights.sum() * c), expected_value)
        np.testing.assert_allclose(
            scaled_model.seq_log_density(scaled_model.dist_to_encoder().seq_encode(data)),
            expected_model.seq_log_density(expected_model.dist_to_encoder().seq_encode(data)),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_integer_markov_chain_enumerator(self):
        init_dist = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.55, 0.45]), len_dist=IntegerCategoricalDistribution(1, [1.0])
        )
        len_dist = IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3])
        dist = IntegerMarkovChainDistribution(
            num_values=2, cond_dist=[[0.8, 0.2], [0.3, 0.7]], lag=1, init_dist=init_dist, len_dist=len_dist
        )
        support = [[]]
        support.extend([list(v) for n in (1, 2, 3) for v in itertools.product(range(2), repeat=n)])
        assert_matches_brute(self, dist, support, "integer_markov_chain")
        total = np.logaddexp.reduce([lp for _, lp in dist.enumerator()])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_integer_bernoulli_edit_enumerator(self):
        init_dist = IntegerBernoulliSetDistribution(np.log([0.6, 0.3]))
        edit = np.log([[0.7, 0.2, 0.3, 0.8], [0.4, 0.6, 0.6, 0.4]])
        dist = IntegerBernoulliEditDistribution(edit, init_dist=init_dist)
        subsets = [list(s) for n in range(3) for s in itertools.combinations(range(2), n)]
        support = [(x0, x1) for x0 in subsets for x1 in subsets]
        assert_matches_brute(self, dist, support, "integer_bernoulli_edit")
        total = np.logaddexp.reduce([lp for _, lp in dist.enumerator()])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_integer_step_bernoulli_edit_enumerator(self):
        init_dist = IntegerBernoulliSetDistribution(np.log([0.6, 0.3]))
        edit = np.log([[0.7, 0.2, 0.3, 0.8], [0.4, 0.6, 0.6, 0.4]])
        dist = IntegerStepBernoulliEditDistribution(edit, init_dist=init_dist)
        subsets = [list(s) for n in range(3) for s in itertools.combinations(range(2), n)]
        support = [(x0, x1) for x0 in subsets for x1 in subsets]
        assert_matches_brute(self, dist, support, "integer_step_bernoulli_edit")
        total = np.logaddexp.reduce([lp for _, lp in dist.enumerator()])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)


class EnumeratorExportCoverageTestCase(unittest.TestCase):
    def test_public_enumerators_are_exported(self):
        import pysp.stats as stats

        public_enumerators = [
            "BernoulliSetEnumerator",
            "ConditionalDistributionEnumerator",
            "DiracLengthMixtureEnumerator",
            "HierarchicalMixtureEnumerator",
            "ICLTreeEnumerator",
            "IntegerBernoulliEditEnumerator",
            "IntegerMarkovChainEnumerator",
            "IntegerMultinomialEnumerator",
            "IntegerStepBernoulliEditEnumerator",
            "IntegerUniformSpikeEnumerator",
            "JointMixtureEnumerator",
            "MallowsEnumerator",
            "MultinomialEnumerator",
            "SelectEnumerator",
            "SpearmanRankingEnumerator",
        ]
        for name in public_enumerators:
            self.assertTrue(hasattr(stats, name), name)
            self.assertIn(name, stats.__all__)


if __name__ == "__main__":
    unittest.main()
