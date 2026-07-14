"""Tests for the distribution API naming aliases.

Covers backward-compatible class-name aliases, constructor keyword-argument aliases, and the
``mixle.utils.aliasing`` helper (the module docstring there states the alias policy). Alias
constructors must produce models identical to the legacy spelling, and passing both spellings
(or neither, for required arguments) must raise.
"""

import unittest

import numpy as np

from mixle.utils.aliasing import MISSING, coalesce_alias, require


class CoalesceAliasHelperTestCase(unittest.TestCase):
    def test_prefers_either_supplied(self):
        self.assertEqual(coalesce_alias("w", 5, "weights", MISSING, default=MISSING), 5)
        self.assertEqual(coalesce_alias("w", MISSING, "weights", 9, default=MISSING), 9)

    def test_both_supplied_raises(self):
        with self.assertRaises(TypeError):
            coalesce_alias("w", 5, "weights", 9, default=MISSING)

    def test_neither_supplied(self):
        with self.assertRaises(TypeError):
            coalesce_alias("w", MISSING, "weights", MISSING, default=MISSING)
        self.assertIsNone(coalesce_alias("p_vec", None, "prob_vec", None, required=False, default=None))

    def test_none_is_a_valid_value_with_missing_sentinel(self):
        # None is a legitimate explicit value when MISSING marks "not supplied"
        self.assertIsNone(coalesce_alias("x", None, "y", MISSING, default=MISSING))

    def test_require(self):
        self.assertEqual(require("transitions", 3, default=MISSING), 3)
        with self.assertRaises(TypeError):
            require("transitions", MISSING, default=MISSING)


class ClassNameAliasTestCase(unittest.TestCase):
    def test_estimator_accumulator_aliases(self):
        from mixle.stats.combinator import optional, select
        from mixle.stats.latent import hierarchical_mixture, joint_mixture, lda, semi_supervised_mixture

        self.assertIs(lda.LDAAccumulator, lda.LDAEstimatorAccumulator)
        self.assertIs(lda.LDAAccumulatorFactory, lda.LDAEstimatorAccumulatorFactory)
        self.assertIs(
            hierarchical_mixture.HierarchicalMixtureAccumulator,
            hierarchical_mixture.HierarchicalMixtureEstimatorAccumulator,
        )
        self.assertIs(
            joint_mixture.JointMixtureAccumulatorFactory, joint_mixture.JointMixtureEstimatorAccumulatorFactory
        )
        self.assertIs(select.SelectAccumulator, select.SelectEstimatorAccumulator)
        self.assertIs(
            semi_supervised_mixture.SemiSupervisedMixtureAccumulator,
            semi_supervised_mixture.SemiSupervisedMixtureEstimatorAccumulator,
        )
        self.assertIs(optional.OptionalAccumulatorFactory, optional.OptionalEstimatorAccumulatorFactory)

    def test_family_stem_aliases(self):
        from mixle.stats.combinator import conditional as cond
        from mixle.stats.latent import gaussian_mixture
        from mixle.stats.latent import hidden_markov as hm
        from mixle.stats.latent import quantized_hidden_markov_model as qhmm
        from mixle.stats.latent import segmental_hidden_markov_model as seg
        from mixle.stats.latent import tree_hidden_markov_model as tree

        self.assertIs(hm.HiddenMarkovModelEstimator, hm.HiddenMarkovEstimator)
        self.assertIs(hm.HiddenMarkovModelSampler, hm.HiddenMarkovSampler)
        self.assertIs(hm.HiddenMarkovModelAccumulatorFactory, hm.HiddenMarkovAccumulatorFactory)
        self.assertIs(cond.ConditionalEstimator, cond.ConditionalDistributionEstimator)
        self.assertIs(cond.ConditionalEnumerator, cond.ConditionalDistributionEnumerator)
        self.assertIs(qhmm.QuantizedHiddenMarkovModelEstimator, qhmm.QuantizedHiddenMarkovEstimator)
        self.assertIs(
            gaussian_mixture.GaussianMixtureAccumulatorFactory,
            gaussian_mixture.GaussianMixtureEstimatorAccumulatorFactory,
        )
        self.assertIs(seg.SegmentalHiddenMarkovModelEstimator, seg.SegmentalHiddenMarkovEstimator)
        self.assertIs(tree.TreeHiddenMarkovModelEstimator, tree.TreeHiddenMarkovEstimator)

    def test_aliases_exported_from_package(self):
        from mixle.stats import (
            ConditionalEnumerator,
            ConditionalEstimator,
            HiddenMarkovModelEstimator,
            HiddenMarkovModelSampler,
            QuantizedHiddenMarkovModelEstimator,
            SegmentalHiddenMarkovModelEstimator,
            TreeHiddenMarkovModelEstimator,
        )

        self.assertTrue(
            all(
                callable(c)
                for c in [
                    HiddenMarkovModelEstimator,
                    HiddenMarkovModelSampler,
                    QuantizedHiddenMarkovModelEstimator,
                    ConditionalEstimator,
                    ConditionalEnumerator,
                    SegmentalHiddenMarkovModelEstimator,
                    TreeHiddenMarkovModelEstimator,
                ]
            )
        )


class WeightsAliasTestCase(unittest.TestCase):
    def _components(self):
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        return [CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"a": 0.5, "b": 0.5})]

    def test_mixture_family_weights(self):
        from mixle.stats.latent.heterogeneous_mixture import HeterogeneousMixtureDistribution
        from mixle.stats.latent.mixture import MixtureDistribution
        from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureDistribution

        for cls in (MixtureDistribution, HeterogeneousMixtureDistribution, SemiSupervisedMixtureDistribution):
            comps = self._components()
            a = cls(comps, [0.3, 0.7])
            b = cls(comps, weights=[0.3, 0.7])
            self.assertTrue(np.allclose(a.w, b.w), cls.__name__)
            with self.assertRaises(TypeError):
                cls(comps, [0.3, 0.7], weights=[0.3, 0.7])
            with self.assertRaises(TypeError):
                cls(comps)

    def test_gaussian_mixture_weights(self):
        from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution

        mu = [[0.0, 0.0], [3.0, 3.0]]
        sig2 = [[1.0, 1.0], [1.0, 1.0]]
        a = GaussianMixtureDistribution(mu, sig2, [0.4, 0.6])
        b = GaussianMixtureDistribution(mu, sig2, weights=[0.4, 0.6])
        self.assertTrue(np.allclose(a.w, b.w))

    def test_hmm_family_weights_and_transitions(self):
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        comps = self._components()
        trans = [[0.5, 0.5], [0.2, 0.8]]
        a = HiddenMarkovModelDistribution(comps, [0.5, 0.5], trans)
        b = HiddenMarkovModelDistribution(comps, weights=[0.5, 0.5], transitions=trans)
        self.assertTrue(np.allclose(a.w, b.w))
        self.assertTrue(np.allclose(a.transitions, b.transitions))
        with self.assertRaises(TypeError):
            HiddenMarkovModelDistribution(comps, [0.5, 0.5])  # transitions required
        with self.assertRaises(TypeError):
            HiddenMarkovModelDistribution(comps, [0.5, 0.5], trans, weights=[0.5, 0.5])


class ProbMapVecCovarAliasTestCase(unittest.TestCase):
    def test_prob_map(self):
        from mixle.stats.sets.bernoulli_set import BernoulliSetDistribution
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        self.assertEqual(
            CategoricalDistribution(prob_map={"a": 0.6, "b": 0.4}).pmap,
            CategoricalDistribution({"a": 0.6, "b": 0.4}).pmap,
        )
        self.assertEqual(BernoulliSetDistribution(prob_map={"a": 0.5}).pmap, BernoulliSetDistribution({"a": 0.5}).pmap)
        with self.assertRaises(TypeError):
            CategoricalDistribution({"a": 1.0}, prob_map={"a": 1.0})

    def test_prob_vec(self):
        from mixle.stats.multivariate.integer_multinomial import IntegerMultinomialDistribution
        from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution

        self.assertTrue(
            np.allclose(
                IntegerCategoricalDistribution(0, prob_vec=[0.5, 0.5]).p_vec,
                IntegerCategoricalDistribution(0, [0.5, 0.5]).p_vec,
            )
        )
        self.assertTrue(
            np.allclose(
                IntegerMultinomialDistribution(0, prob_vec=[0.3, 0.7]).p_vec,
                IntegerMultinomialDistribution(0, [0.3, 0.7]).p_vec,
            )
        )

    def test_covariance(self):
        from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        cov = [[2.0, 0.0], [0.0, 3.0]]
        self.assertTrue(
            np.allclose(
                MultivariateGaussianDistribution([0, 0], covariance=cov).covar,
                MultivariateGaussianDistribution([0, 0], cov).covar,
            )
        )
        self.assertTrue(
            np.allclose(
                DiagonalGaussianDistribution([0, 0], covariance=[2.0, 3.0]).covar,
                DiagonalGaussianDistribution([0, 0], [2.0, 3.0]).covar,
            )
        )
        with self.assertRaises(TypeError):
            MultivariateGaussianDistribution([0, 0], cov, covariance=cov)


class NumValuesMaxIterAliasTestCase(unittest.TestCase):
    def test_num_values(self):
        from mixle.stats.sequences.markov_transform import MarkovTransformEstimator
        from mixle.stats.sequences.sparse_markov_transform import SparseMarkovAssociationEstimator
        from mixle.stats.sets.integer_bernoulli_edit import IntegerBernoulliEditEstimator
        from mixle.stats.sets.integer_bernoulli_set import IntegerBernoulliSetEstimator
        from mixle.stats.sets.integer_step_bernoulli_edit import IntegerStepBernoulliEditEstimator

        for cls in (
            MarkovTransformEstimator,
            SparseMarkovAssociationEstimator,
            IntegerBernoulliSetEstimator,
            IntegerBernoulliEditEstimator,
            IntegerStepBernoulliEditEstimator,
        ):
            self.assertEqual(cls(num_values=6).num_vals, 6, cls.__name__)
            self.assertEqual(cls(6).num_vals, 6, cls.__name__)
            with self.assertRaises(TypeError):
                cls(6, num_values=6)

    def test_max_iter(self):
        from mixle.inference.em import RestartEM
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        model = CategoricalDistribution({"a": 1.0})
        self.assertEqual(RestartEM([model], max_iter=13).max_its, 13)
        self.assertEqual(RestartEM([model], max_its=4).max_its, 4)


if __name__ == "__main__":
    unittest.main()
