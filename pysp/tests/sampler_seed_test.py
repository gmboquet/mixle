import math
import unittest

import numpy as np
from scipy.sparse import csr_matrix

import pysp.bstats as bstats
import pysp.stats as stats
from pysp.stats.transform import AffineTransform


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

    int_plsi = stats.IntegerPLSIDistribution(
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
        "GammaDistribution": stats.GammaDistribution(2.0, 3.0),
        "GaussianDistribution": stats.GaussianDistribution(1.0, 2.0),
        "GeometricDistribution": stats.GeometricDistribution(0.25),
        "NegativeBinomialDistribution": stats.NegativeBinomialDistribution(3.0, 0.45),
        "ParetoDistribution": stats.ParetoDistribution(2.0, 3.0),
        "RayleighDistribution": stats.RayleighDistribution(2.0),
        "StudentTDistribution": stats.StudentTDistribution(5.0, loc=1.0, scale=2.0),
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
        "ICLTreeDistribution": stats.ICLTreeDistribution(
            [None, 0], [np.log([0.6, 0.4]), np.log([[0.8, 0.2], [0.1, 0.9]])]
        ),
        "IgnoredDistribution": stats.IgnoredDistribution(stats.GaussianDistribution(0.0, 1.0)),
        "IntegerBernoulliEditDistribution": stats.IntegerBernoulliEditDistribution(log_edit, init_dist=int_set),
        "IntegerStepBernoulliEditDistribution": stats.IntegerStepBernoulliEditDistribution(log_edit, init_dist=int_set),
        "IntegerHiddenAssociationDistribution": int_hidden_assoc,
        "IntegerMarkovChainDistribution": int_markov,
        "IntegerPLSIDistribution": int_plsi,
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
    }


def _bstats_public_distribution_catalog():
    comps = [bstats.GaussianDistribution(0.0, 1.0), bstats.GaussianDistribution(3.0, 2.0)]
    dpm = bstats.DirichletProcessMixtureDistribution(
        comps,
        np.asarray([0.55, 0.45]),
        1.5,
        np.asarray([[2.0, 3.0], [1.0, 1.0]]),
        [c.get_prior() for c in comps],
        name="dpm",
    )

    return {
        "BernoulliDistribution": bstats.BernoulliDistribution(0.3),
        "BernoulliSetDistribution": bstats.BernoulliSetDistribution({"a": 0.7, "b": 0.2, "c": 0.9}),
        "BetaDistribution": bstats.BetaDistribution(2.0, 5.0),
        "BinomialDistribution": bstats.BinomialDistribution(10, 0.4),
        "CategoricalDistribution": bstats.CategoricalDistribution({"a": 0.4, "b": 0.6}),
        "CompositeDistribution": bstats.CompositeDistribution(
            [
                bstats.CategoricalDistribution({"x": 0.5, "y": 0.5}),
                bstats.GaussianDistribution(0.0, 1.0),
            ]
        ),
        "DiagonalGaussianDistribution": bstats.DiagonalGaussianDistribution([0.5, -1.0], [1.0, 2.0]),
        "DictDirichletDistribution": bstats.DictDirichletDistribution({"a": 1.0, "b": 2.0}),
        "DirichletDistribution": bstats.DirichletDistribution([1.0, 2.0, 3.0]),
        "DirichletProcessMixtureDistribution": dpm,
        "ExponentialDistribution": bstats.ExponentialDistribution(2.0),
        "GaussianDistribution": bstats.GaussianDistribution(1.0, 2.0),
        "GammaDistribution": bstats.GammaDistribution(2.0, 3.0),
        "GeometricDistribution": bstats.GeometricDistribution(0.25),
        "HiddenMarkovModelDistribution": bstats.HiddenMarkovModelDistribution(
            [bstats.GaussianDistribution(-5.0, 1.0), bstats.GaussianDistribution(5.0, 1.0)],
            [0.7, 0.3],
            [[0.9, 0.1], [0.2, 0.8]],
            len_dist=bstats.CategoricalDistribution({4: 1.0}),
        ),
        "HierarchicalDirichletProcessMixtureDistribution": bstats.HierarchicalDirichletProcessMixtureDistribution(
            [bstats.GaussianDistribution(-2.0, 1.0), bstats.GaussianDistribution(2.0, 1.0)],
            beta=[0.6, 0.4],
            alpha=3.0,
            gamma=2.0,
            len_dist=bstats.CategoricalDistribution({5: 1.0}),
        ),
        "IgnoredDistribution": bstats.IgnoredDistribution(bstats.GaussianDistribution(0.0, 1.0)),
        "IntegerCategoricalDistribution": bstats.IntegerCategoricalDistribution([0.2, 0.3, 0.5], min_index=0),
        "LogGaussianDistribution": bstats.LogGaussianDistribution(0.0, 1.0),
        "MarkovChainDistribution": bstats.MarkovChainDistribution(
            [0.6, 0.4], [[0.7, 0.3], [0.2, 0.8]], len_dist=bstats.CategoricalDistribution({4: 1.0})
        ),
        "MixtureDistribution": bstats.MixtureDistribution(comps, [0.4, 0.6]),
        "MultivariateGaussianDistribution": bstats.MultivariateGaussianDistribution(
            [0.5, -1.0], [[1.0, 0.2], [0.2, 2.0]]
        ),
        "MultivariateNormalGammaDistribution": bstats.MultivariateNormalGammaDistribution(
            np.array([0.0, 1.0]), np.array([1.0, 1.5]), np.array([2.0, 3.0]), np.array([4.0, 5.0])
        ),
        "NormalGammaDistribution": bstats.NormalGammaDistribution(0.0, 1.0, 2.0, 3.0),
        "NormalWishartDistribution": bstats.NormalWishartDistribution([0.0, 1.0], 2.0, [[2.0, 0.0], [0.0, 2.0]], 5.0),
        "NullDistribution": bstats.NullDistribution(),
        "OptionalDistribution": bstats.OptionalDistribution(bstats.PoissonDistribution(2.0), p=0.25),
        "PoissonDistribution": bstats.PoissonDistribution(3.0),
        "SequenceDistribution": bstats.SequenceDistribution(
            bstats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
            len_dist=bstats.CategoricalDistribution({2: 0.4, 3: 0.6}),
        ),
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
        catalog = _stats_public_distribution_catalog()
        self.assert_catalog_matches_exports(stats, catalog)
        for name, dist in sorted(catalog.items()):
            self.assert_repeatable_sampler(name, dist)
            self.assert_sized_sample_contract(name, dist, null_is_sentinel=True)

    def test_all_public_bstats_samplers_are_seed_repeatable(self):
        catalog = _bstats_public_distribution_catalog()
        self.assert_catalog_matches_exports(bstats, catalog)
        for name, dist in sorted(catalog.items()):
            self.assert_repeatable_sampler(name, dist)
            self.assert_sized_sample_contract(name, dist)

    def test_bstats_hmm_sampler_uses_one_transition_per_observation(self):
        dist = bstats.HiddenMarkovModelDistribution(
            [bstats.CategoricalDistribution({0: 1.0}), bstats.CategoricalDistribution({1: 1.0})],
            [1.0, 0.0],
            [[0.0, 1.0], [1.0, 0.0]],
            len_dist=bstats.CategoricalDistribution({6: 1.0}),
        )

        self.assertEqual(_canonical(dist.sampler(seed=11).sample()), [0, 1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
