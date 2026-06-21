import math
import unittest

import numpy as np
from scipy.sparse import csr_matrix

import pysp.stats as stats
from pysp.stats.combinator.transform import AffineTransform


def _canonical(x):
    """Turn nested numpy-heavy samples into deterministic Python values."""
    if isinstance(x, np.ndarray):
        return _canonical(x.tolist())
    if isinstance(x, np.generic):
        return _canonical(x.item())
    if isinstance(x, float):
        return "NaN" if math.isnan(x) else round(x, 14)
    if isinstance(x, (list, tuple)):
        return [_canonical(v) for v in x]
    if isinstance(x, set):
        return sorted(_canonical(v) for v in x)
    if isinstance(x, dict):
        return sorted((_canonical(k), _canonical(v)) for k, v in x.items())
    return x


def _normalize_rows(x):
    x = np.asarray(x, dtype=float)
    return x / x.sum(axis=1, keepdims=True)


def _stats_public_distribution_catalog():
    cat_ab = stats.CategoricalDistribution({"a": 0.4, "b": 0.6})
    multinom_ab = stats.MultinomialDistribution(
        stats.CategoricalDistribution({"a": 0.6, "b": 0.4}), stats.CategoricalDistribution({3: 1.0})
    )
    int_multinom_2 = stats.IntegerMultinomialDistribution(
        0, [0.6, 0.4], len_dist=stats.CategoricalDistribution({3: 1.0})
    )
    int_set = stats.IntegerBernoulliSetDistribution(np.log([0.6, 0.3, 0.8]))
    log_edit = np.log(np.asarray([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]], dtype=float))

    cond_cat = stats.ConditionalDistribution(
        {
            "a": stats.CategoricalDistribution({"x": 0.7, "y": 0.3}),
            "b": stats.CategoricalDistribution({"x": 0.2, "z": 0.8}),
        },
        given_dist=stats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
    )

    hidden_assoc = stats.HiddenAssociationDistribution(
        cond_dist=stats.ConditionalDistribution(
            {
                "a": stats.CategoricalDistribution({"x": 0.8, "y": 0.2}),
                "b": stats.CategoricalDistribution({"x": 0.3, "y": 0.7}),
            }
        ),
        given_dist=multinom_ab,
        len_dist=stats.CategoricalDistribution({2: 1.0}),
    )

    hmm = stats.HiddenMarkovModelDistribution(
        [
            stats.CategoricalDistribution({"a": 0.8, "b": 0.2}),
            stats.CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        [0.6, 0.4],
        [[0.7, 0.3], [0.2, 0.8]],
        len_dist=stats.CategoricalDistribution({4: 1.0}),
        use_numba=False,
    )

    quantized_hmm = stats.QuantizedHiddenMarkovModelDistribution(
        0.5,
        ["a", "b", "c"],
        [[0, 1], [2, 0]],
        [[0, 1, 2], [2, 1, 0]],
        initial_exponents=[0, 1],
        len_dist=stats.CategoricalDistribution({3: 0.5, 4: 0.5}),
        use_numba=False,
    )

    hmix = stats.HierarchicalMixtureDistribution(
        topics=[
            stats.CategoricalDistribution({"a": 0.7, "b": 0.3}),
            stats.CategoricalDistribution({"b": 0.4, "c": 0.6}),
        ],
        mixture_weights=[0.6, 0.4],
        topic_weights=[[0.8, 0.2], [0.3, 0.7]],
        len_dist=stats.IntegerCategoricalDistribution(1, [0.6, 0.4]),
    )

    int_hidden_assoc = stats.IntegerHiddenAssociationDistribution(
        state_prob_mat=[[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]],
        cond_weights=[[0.8, 0.2], [0.3, 0.7]],
        alpha=0.05,
        prev_dist=int_multinom_2,
        len_dist=stats.CategoricalDistribution({3: 1.0}),
        use_numba=False,
    )

    imc_init = stats.SequenceDistribution(
        stats.IntegerCategoricalDistribution(0, [0.5, 0.5]), len_dist=stats.CategoricalDistribution({2: 1.0})
    )
    int_markov = stats.IntegerMarkovChainDistribution(
        num_values=2,
        cond_dist=[[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]],
        lag=2,
        init_dist=imc_init,
        len_dist=stats.CategoricalDistribution({5: 1.0}),
    )

    int_plsi = stats.IntegerProbabilisticLatentSemanticIndexingDistribution(
        state_word_mat=[[0.7, 0.2], [0.2, 0.5], [0.1, 0.3]],
        doc_state_mat=[[0.8, 0.2], [0.3, 0.7]],
        doc_vec=[0.55, 0.45],
        len_dist=stats.CategoricalDistribution({4: 1.0}),
    )

    joint_mix = stats.JointMixtureDistribution(
        components1=[stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(3.0, 2.0)],
        components2=[stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(2.0, 0.5)],
        w1=[0.6, 0.4],
        w2=[0.5, 0.5],
        taus12=[[0.7, 0.3], [0.2, 0.8]],
        taus21=[[0.7, 0.2], [0.3, 0.8]],
    )

    segmental = stats.SegmentalHiddenMarkovModelDistribution(
        [
            stats.GaussianDistribution(-2.0, 1.0),
            stats.StudentTDistribution(5.0, loc=2.0, scale=1.5),
        ],
        [0.6, 0.4],
        [[0.7, 0.3], [0.2, 0.8]],
        len_dist=stats.IntegerCategoricalDistribution(2, [1.0]),
    )

    sparse_assoc = stats.SparseMarkovAssociationDistribution(
        [0.5, 0.3, 0.2],
        csr_matrix(_normalize_rows([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]])),
        alpha=0.1,
        len_dist=stats.CompositeDistribution(
            (
                stats.CategoricalDistribution({2: 1.0}),
                stats.CategoricalDistribution({3: 1.0}),
            )
        ),
    )

    tree_hmm = stats.TreeHiddenMarkovModelDistribution(
        topics=[
            stats.CategoricalDistribution({"a": 0.8, "b": 0.2}),
            stats.CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        w=[0.6, 0.4],
        transitions=[[0.7, 0.3], [0.2, 0.8]],
        len_dist=stats.IntegerCategoricalDistribution(0, [0.0, 1.0, 0.0]),
        terminal_level=2,
        use_numba=False,
    )

    pcfg = stats.HeterogeneousPCFGDistribution(
        binary_rules={"S": [("A", "B", 1.0)]},
        terminal_rules={
            "A": [(stats.CategoricalDistribution({"a": 0.7, "b": 0.3}), 1.0)],
            "B": [(stats.CategoricalDistribution({"x": 0.4, "y": 0.6}), 1.0)],
        },
        start="S",
    )

    return {
        "BernoulliDistribution": stats.BernoulliDistribution(0.3),
        "BetaDistribution": stats.BetaDistribution(2.0, 5.0),
        "LaplaceDistribution": stats.LaplaceDistribution(0.0, 1.5),
        "LogisticDistribution": stats.LogisticDistribution(0.0, 1.0),
        "BinomialDistribution": stats.BinomialDistribution(0.4, 10, min_val=1),
        "CategoricalDistribution": cat_ab,
        "MultinomialDistribution": stats.MultinomialDistribution(
            stats.CategoricalDistribution({"a": 0.7, "b": 0.3}), stats.CategoricalDistribution({3: 1.0})
        ),
        "CompositeDistribution": stats.CompositeDistribution(
            (
                stats.CategoricalDistribution({"x": 0.5, "y": 0.5}),
                stats.GaussianDistribution(0.0, 1.0),
            )
        ),
        "RecordDistribution": stats.RecordDistribution(
            {
                "x": stats.GaussianDistribution(0.0, 1.0),
                "label": stats.CategoricalDistribution({"a": 0.4, "b": 0.6}),
            }
        ),
        "DictRecordDistribution": stats.DictRecordDistribution(
            {
                "x": stats.GaussianDistribution(1.0, 2.0),
                "label": stats.CategoricalDistribution({"left": 0.3, "right": 0.7}),
            }
        ),
        "ConditionalDistribution": cond_cat,
        "ChowLiuTreeDistribution": stats.ChowLiuTreeDistribution(
            [None, 0],
            [
                stats.CategoricalDistribution({"a": 0.6, "b": 0.4}),
                stats.CategoricalDistribution({0: 0.5, 1: 0.5}),
            ],
            [
                None,
                {
                    "a": stats.CategoricalDistribution({0: 0.7, 1: 0.3}),
                    "b": stats.CategoricalDistribution({0: 0.2, 1: 0.8}),
                },
            ],
        ),
        "DiracLengthMixtureDistribution": stats.DiracLengthMixtureDistribution(
            stats.IntegerCategoricalDistribution(1, [0.4, 0.6]), p=0.7, v=0
        ),
        "DirichletDistribution": stats.DirichletDistribution([1.0, 2.0, 3.0]),
        "DiagonalGaussianDistribution": stats.DiagonalGaussianDistribution([0.5, -1.0], [1.0, 2.0]),
        "ExponentialDistribution": stats.ExponentialDistribution(2.0),
        "ExponentiallyModifiedGaussianDistribution": stats.ExponentiallyModifiedGaussianDistribution(0.0, 1.0, 1.0),
        "GammaDistribution": stats.GammaDistribution(2.0, 3.0),
        "InverseGammaDistribution": stats.InverseGammaDistribution(3.0, 2.0),
        "GaussianDistribution": stats.GaussianDistribution(1.0, 2.0),
        "InverseGaussianDistribution": stats.InverseGaussianDistribution(2.0, 3.0),
        "GumbelDistribution": stats.GumbelDistribution(2.0, 1.5),
        "VonMisesDistribution": stats.VonMisesDistribution(0.7, 2.5),
        "HalfNormalDistribution": stats.HalfNormalDistribution(1.5),
        "GeometricDistribution": stats.GeometricDistribution(0.25),
        "LogSeriesDistribution": stats.LogSeriesDistribution(0.6),
        "NegativeBinomialDistribution": stats.NegativeBinomialDistribution(3.0, 0.45),
        "ParetoDistribution": stats.ParetoDistribution(2.0, 3.0),
        "RayleighDistribution": stats.RayleighDistribution(2.0),
        "SkellamDistribution": stats.SkellamDistribution(2.0, 1.0),
        "StudentTDistribution": stats.StudentTDistribution(5.0, loc=1.0, scale=2.0),
        "TweedieDistribution": stats.TweedieDistribution(2.0, 1.0, 1.5),
        "BirthDeathSamplingDistribution": stats.BirthDeathSamplingDistribution(
            0.6, 0.3, 0.2, initial_population=2, horizon=5.0
        ),
        "InhomogeneousPoissonProcessDistribution": stats.InhomogeneousPoissonProcessDistribution(
            [1.0, 3.0, 0.5], t_max=3.0
        ),
        "HawkesProcessDistribution": stats.HawkesProcessDistribution(0.5, 0.6, 1.2, 50.0),
        "ZeroInflatedDistribution": stats.ZeroInflatedDistribution(stats.PoissonDistribution(2.0), 0.3),
        "HurdleDistribution": stats.HurdleDistribution(stats.PoissonDistribution(2.0), 0.3),
        "GeneralizedParetoDistribution": stats.GeneralizedParetoDistribution(2.0, 0.3),
        "GeneralizedExtremeValueDistribution": stats.GeneralizedExtremeValueDistribution(0.0, 2.0, 0.2),
        "GaussianCopulaDistribution": stats.GaussianCopulaDistribution([[1.0, 0.5], [0.5, 1.0]]),
        "MatrixNormalDistribution": stats.MatrixNormalDistribution(
            [[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]], [[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.5]], [[1.4, 0.4], [0.4, 0.9]]
        ),
        "WrappedCauchyDistribution": stats.WrappedCauchyDistribution(0.7, 0.6),
        "BetaBinomialDistribution": stats.BetaBinomialDistribution(10, 2.0, 3.0),
        "DirichletMultinomialDistribution": stats.DirichletMultinomialDistribution([1.0, 2.0, 1.5], 8),
        "WatsonDistribution": stats.WatsonDistribution([0.0, 0.0, 1.0], 5.0),
        "WishartDistribution": stats.WishartDistribution(6, [[2.0, 0.3], [0.3, 1.0]]),
        "SkewNormalDistribution": stats.SkewNormalDistribution(0.5, 2.0, 4.0),
        "InverseWishartDistribution": stats.InverseWishartDistribution(8, [[2.0, 0.3], [0.3, 1.0]]),
        "SurvivalDistribution": stats.SurvivalDistribution(stats.WeibullDistribution(1.5, 2.0)),
        "UniformDistribution": stats.UniformDistribution(-1.0, 3.0),
        "WeibullDistribution": stats.WeibullDistribution(1.5, 2.0),
        "HeterogeneousMixtureDistribution": stats.HeterogeneousMixtureDistribution(
            [stats.GaussianDistribution(0.0, 1.0), stats.CategoricalDistribution({"a": 0.8, "b": 0.2})], [0.5, 0.5]
        ),
        "HeterogeneousPCFGDistribution": pcfg,
        "HiddenAssociationDistribution": hidden_assoc,
        "HiddenMarkovModelDistribution": hmm,
        "QuantizedHiddenMarkovModelDistribution": quantized_hmm,
        "HierarchicalMixtureDistribution": hmix,
        "IndianBuffetProcessDistribution": stats.IndianBuffetProcessDistribution(
            5, alpha=2.0, feature_probs=[0.9, 0.1, 0.5, 0.2, 0.8], data_format="sparse"
        ),
        "IntegerChowLiuTreeDistribution": stats.IntegerChowLiuTreeDistribution(
            [None, 0], [np.log([0.6, 0.4]), np.log([[0.8, 0.2], [0.1, 0.9]])]
        ),
        "IgnoredDistribution": stats.IgnoredDistribution(stats.GaussianDistribution(0.0, 1.0)),
        "IntegerBernoulliEditDistribution": stats.IntegerBernoulliEditDistribution(log_edit, init_dist=int_set),
        "IntegerStepBernoulliEditDistribution": stats.IntegerStepBernoulliEditDistribution(log_edit, init_dist=int_set),
        "IntegerHiddenAssociationDistribution": int_hidden_assoc,
        "IntegerMarkovChainDistribution": int_markov,
        "IntegerProbabilisticLatentSemanticIndexingDistribution": int_plsi,
        "IntegerUniformSpikeDistribution": stats.IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.5, min_val=0),
        "IntegerMultinomialDistribution": stats.IntegerMultinomialDistribution(
            0, [0.2, 0.5, 0.3], len_dist=stats.CategoricalDistribution({4: 1.0})
        ),
        "IntegerCategoricalDistribution": stats.IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
        "IntegerBernoulliSetDistribution": int_set,
        "JointMixtureDistribution": joint_mix,
        "LogGaussianDistribution": stats.LogGaussianDistribution(0.0, 1.0),
        "MarkovChainDistribution": stats.MarkovChainDistribution(
            {"a": 0.6, "b": 0.4},
            {"a": {"a": 0.7, "b": 0.3}, "b": {"a": 0.2, "b": 0.8}},
            len_dist=stats.CategoricalDistribution({4: 1.0}),
        ),
        "MixtureDistribution": stats.MixtureDistribution(
            [stats.GaussianDistribution(-2.0, 1.0), stats.GaussianDistribution(2.0, 1.0)], [0.4, 0.6]
        ),
        "MultivariateGaussianDistribution": stats.MultivariateGaussianDistribution(
            [0.5, -1.0], [[1.0, 0.2], [0.2, 2.0]]
        ),
        "MultivariateStudentTDistribution": stats.MultivariateStudentTDistribution(
            6.0, [0.5, -1.0], [[1.0, 0.2], [0.2, 2.0]]
        ),
        "ProbabilisticPCADistribution": stats.ProbabilisticPCADistribution(
            [[1.0, 0.2], [0.3, 0.8], [0.5, 0.1], [-0.2, 0.6]], [0.0, 1.0, -1.0, 0.5], 0.5
        ),
        "PlackettLuceDistribution": stats.PlackettLuceDistribution([1.5, 0.5, -0.5, -1.5]),
        "KnowledgeGraphDistribution": stats.KnowledgeGraphDistribution(
            np.linspace(-1, 1, 18).reshape(6, 3), np.linspace(-0.5, 0.5, 6).reshape(2, 3)
        ),
        "MallowsDistribution": stats.MallowsDistribution([0, 2, 1, 3], theta=1.0),
        "SpanningTreeDistribution": stats.SpanningTreeDistribution(
            [[0.0, 2.0, 1.0, 3.0], [2.0, 0.0, 4.0, 1.0], [1.0, 4.0, 0.0, 2.0], [3.0, 1.0, 2.0, 0.0]]
        ),
        "PitmanYorProcessDistribution": stats.PitmanYorProcessDistribution(1.5, 0.3, num_elements=8),
        "MatchingDistribution": stats.MatchingDistribution([[2.0, 1.0, 3.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]]),
        "NullDistribution": stats.NullDistribution(),
        "OptionalDistribution": stats.OptionalDistribution(stats.PoissonDistribution(2.0), p=0.25),
        "PoissonDistribution": stats.PoissonDistribution(3.0),
        "PointMassDistribution": stats.PointMassDistribution("fixed"),
        "SelectDistribution": stats.SelectDistribution(
            [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(100.0, 1.0)],
            lambda x: 0 if x < 50.0 else 1,
        ),
        "SequenceDistribution": stats.SequenceDistribution(
            stats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
            len_dist=stats.CategoricalDistribution({2: 0.4, 3: 0.6}),
        ),
        "SegmentalHiddenMarkovModelDistribution": segmental,
        "SegmentalHiddenMarkovDistribution": stats.SegmentalHiddenMarkovDistribution(
            [
                stats.GaussianDistribution(-2.0, 1.0),
                stats.StudentTDistribution(5.0, loc=2.0, scale=1.5),
            ],
            [0.6, 0.4],
            [[0.7, 0.3], [0.2, 0.8]],
            len_dist=stats.IntegerCategoricalDistribution(2, [1.0]),
        ),
        "BernoulliSetDistribution": stats.BernoulliSetDistribution({"a": 0.7, "b": 0.2, "c": 0.9}, min_prob=0.0),
        "SparseMarkovAssociationDistribution": sparse_assoc,
        "SpearmanRankingDistribution": stats.SpearmanRankingDistribution([0, 1, 2], rho=0.5),
        "SemiSupervisedMixtureDistribution": stats.SemiSupervisedMixtureDistribution(
            [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(3.0, 1.0)], [0.5, 0.5]
        ),
        "TreeHiddenMarkovModelDistribution": tree_hmm,
        "TransformDistribution": stats.TransformDistribution(
            stats.GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=1.0, scale=2.0)
        ),
        "FiniteStochasticTransformDistribution": stats.FiniteStochasticTransformDistribution(
            stats.IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]),
            [[0.7, 0.2, 0.1, 0.0], [0.1, 0.6, 0.2, 0.1], [0.0, 0.1, 0.3, 0.6]],
        ),
        "TruncatedDistribution": stats.TruncatedDistribution(
            stats.IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]), allowed=[0, 1]
        ),
        "CensoredDistribution": stats.CensoredDistribution(stats.GaussianDistribution(0.0, 1.0)),
        "ExponentialTiltedDistribution": stats.ExponentialTiltedDistribution(stats.PoissonDistribution(3.0), theta=0.4),
        "LDADistribution": stats.LDADistribution(
            [
                stats.IntegerCategoricalDistribution(0, [0.7, 0.3]),
                stats.IntegerCategoricalDistribution(0, [0.2, 0.8]),
            ],
            [0.5, 0.5],
            len_dist=stats.IntegerCategoricalDistribution(2, [1.0]),
        ),
        "VonMisesFisherDistribution": stats.VonMisesFisherDistribution([1.0, 0.0, 0.0], 2.0),
        "WeightedDistribution": stats.WeightedDistribution(stats.GaussianDistribution(0.0, 1.0)),
        "ErdosRenyiGraphDistribution": stats.ErdosRenyiGraphDistribution(0.4, num_nodes=6),
        "StochasticBlockGraphDistribution": stats.StochasticBlockGraphDistribution(
            [[0.8, 0.2], [0.2, 0.7]], [0, 0, 1, 1, 0, 1]
        ),
        "RandomDotProductGraphDistribution": stats.RandomDotProductGraphDistribution(
            [[0.7, 0.1], [0.6, 0.2], [0.1, 0.7], [0.2, 0.6], [0.5, 0.5], [0.3, 0.3]]
        ),
    }


def _bayes_only_distribution_catalog():
    """Conjugate-prior / variational families folded in from the former pysp.bstats.

    These are now exported from ``pysp.stats.__all__`` alongside the frequentist
    families; this catalog is merged into the public seed-repeatability sweep.
    """
    from pysp.stats.bayes.dict_dirichlet import DictDirichletDistribution
    from pysp.stats.bayes.dirichlet_process_mixture import DirichletProcessMixtureDistribution
    from pysp.stats.bayes.hierarchical_dirichlet_process_mixture import HierarchicalDirichletProcessMixtureDistribution
    from pysp.stats.bayes.multivariate_normal_gamma import MultivariateNormalGammaDistribution
    from pysp.stats.bayes.normal_gamma import NormalGammaDistribution
    from pysp.stats.bayes.normal_wishart import NormalWishartDistribution
    from pysp.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

    comps = [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(3.0, 2.0)]
    dpm = DirichletProcessMixtureDistribution(
        comps,
        np.asarray([0.55, 0.45]),
        1.5,
        np.asarray([[2.0, 3.0], [1.0, 1.0]]),
        [NormalGammaDistribution(0.0, 1.0, 1.0, 1.0), NormalGammaDistribution(3.0, 1.0, 1.0, 1.0)],
        name="dpm",
    )

    return {
        "DictDirichletDistribution": DictDirichletDistribution({"a": 1.0, "b": 2.0}),
        "DirichletProcessMixtureDistribution": dpm,
        "HierarchicalDirichletProcessMixtureDistribution": HierarchicalDirichletProcessMixtureDistribution(
            [stats.GaussianDistribution(-2.0, 1.0), stats.GaussianDistribution(2.0, 1.0)],
            beta=[0.6, 0.4],
            alpha=3.0,
            gamma=2.0,
            len_dist=stats.CategoricalDistribution({5: 1.0}),
        ),
        "MultivariateNormalGammaDistribution": MultivariateNormalGammaDistribution(
            np.array([0.0, 1.0]), np.array([1.0, 1.5]), np.array([2.0, 3.0]), np.array([4.0, 5.0])
        ),
        "NormalGammaDistribution": NormalGammaDistribution(0.0, 1.0, 2.0, 3.0),
        "NormalWishartDistribution": NormalWishartDistribution([0.0, 1.0], 2.0, [[2.0, 0.0], [0.0, 2.0]], 5.0),
        "SymmetricDirichletDistribution": SymmetricDirichletDistribution(2.0, dim=3),
    }


class SamplerSeedTestCase(unittest.TestCase):
    def assert_catalog_matches_exports(self, module, catalog):
        expected = {name for name in module.__all__ if name.endswith("Distribution")}
        self.assertEqual(expected, set(catalog))

    def assert_repeatable_sampler(self, name, dist):
        with self.subTest(name=name, mode="bulk"):
            first = _canonical(dist.sampler(seed=314159).sample(size=6))
            second = _canonical(dist.sampler(seed=314159).sample(size=6))
            self.assertEqual(first, second)

        with self.subTest(name=name, mode="scalar_stream"):
            first_sampler = dist.sampler(seed=271828)
            second_sampler = dist.sampler(seed=271828)
            first = [_canonical(first_sampler.sample()) for _ in range(6)]
            second = [_canonical(second_sampler.sample()) for _ in range(6)]
            self.assertEqual(first, second)

    def assert_sized_sample_contract(self, name, dist, null_is_sentinel=False):
        with self.subTest(name=name, mode="size_contract"):
            sample = dist.sampler(seed=123).sample(size=4)
            if null_is_sentinel and isinstance(dist, stats.NullDistribution):
                self.assertIsNone(sample)
            else:
                self.assertEqual(len(sample), 4)

    def test_all_public_stats_samplers_are_seed_repeatable(self):
        catalog = {**_stats_public_distribution_catalog(), **_bayes_only_distribution_catalog()}
        self.assert_catalog_matches_exports(stats, catalog)
        for name, dist in sorted(catalog.items()):
            self.assert_repeatable_sampler(name, dist)
            self.assert_sized_sample_contract(name, dist, null_is_sentinel=True)

    def test_bayes_only_samplers_are_seed_repeatable(self):
        catalog = _bayes_only_distribution_catalog()
        for name, dist in sorted(catalog.items()):
            self.assert_repeatable_sampler(name, dist)
            self.assert_sized_sample_contract(name, dist)

    def test_hmm_sampler_uses_one_transition_per_observation(self):
        dist = stats.HiddenMarkovModelDistribution(
            [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})],
            [1.0, 0.0],
            [[0.0, 1.0], [1.0, 0.0]],
            len_dist=stats.CategoricalDistribution({6: 1.0}),
        )

        self.assertEqual(_canonical(dist.sampler(seed=11).sample()), [0, 1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
