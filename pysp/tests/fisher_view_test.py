"""Tests for the generic Fisher-geometry view protocol."""

import unittest

import numpy as np

from pysp.stats import (
    BernoulliDistribution,
    BetaDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    DiagonalGaussianDistribution,
    DirichletDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    HeterogeneousPCFGDistribution,
    HiddenMarkovModelDistribution,
    IndianBuffetProcessDistribution,
    IntegerCategoricalDistribution,
    JointMixtureDistribution,
    LogGaussianDistribution,
    MixtureDistribution,
    MultinomialDistribution,
    MultivariateGaussianDistribution,
    NegativeBinomialDistribution,
    OptionalDistribution,
    PoissonDistribution,
    SelectDistribution,
    SequenceDistribution,
    WeightedDistribution,
)
from pysp.utils.fisher import FisherView, SufficientStatisticVectorizer
from pysp.utils.special import digamma, trigamma


class FisherViewTestCase(unittest.TestCase):
    @staticmethod
    def weighted_moments(mat, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum()
        mean = np.dot(probs, mat)
        second = np.dot((probs[:, None] * mat).T, mat)
        return mean, second - np.outer(mean, mean)

    def assert_data_and_encoded_match(self, dist, data):
        view = dist.to_fisher()
        raw = view.expected_statistics_matrix(data=data)
        if hasattr(dist, "dist_to_encoder"):
            enc = dist.dist_to_encoder().seq_encode(data)
        else:
            enc = dist.seq_encode(data)
        encoded = view.seq_expected_statistics(enc, vectorizer=view.vectorizer, fit=False)

        self.assertEqual(raw.shape, encoded.shape)
        self.assertTrue(np.all(np.isfinite(raw)))
        self.assertTrue(np.all(np.isfinite(encoded)))
        np.testing.assert_allclose(encoded, raw, atol=1.0e-10)

    def test_stats_distribution_inherits_to_fisher(self):
        dist = GaussianDistribution(0.0, 1.0)
        view = dist.to_fisher()

        self.assertIsInstance(view, FisherView)

        data = [-1.0, 2.0]
        mat = view.statistics_matrix(data=data)
        expected = np.asarray(
            [
                [-1.0, 1.0, 1.0, 1.0],
                [2.0, 4.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_allclose(mat, expected)

    def test_encoded_statistics_match_raw_statistics(self):
        dist = GaussianDistribution(0.0, 1.0)
        view = dist.to_fisher()
        data = [-2.0, 0.5, 3.0]

        raw = view.statistics_matrix(data=data)
        enc = dist.dist_to_encoder().seq_encode(data)
        encoded = view.seq_expected_statistics(enc, vectorizer=view.vectorizer, fit=False)

        np.testing.assert_allclose(encoded, raw)

    def test_generic_raw_statistics_do_not_replay_encoded_batch(self):
        view = FisherView(GaussianDistribution(0.0, 1.0))

        def fail_encode(data, estimate):
            raise AssertionError("generic raw data path should not encode the batch")

        view._encode_data = fail_encode
        mat = view.statistics_matrix(data=[-1.0, 2.0])

        np.testing.assert_allclose(mat, [[-1.0, 1.0, 1.0, 1.0], [2.0, 4.0, 1.0, 1.0]])

    def test_batch_statistics_use_encoder_for_dynamic_schemas(self):
        dist = SequenceDistribution(PoissonDistribution(2.0), IntegerCategoricalDistribution(0, [0.1, 0.2, 0.4, 0.3]))
        data = [[1, 2], [0, 3, 2], [1]]

        self.assert_data_and_encoded_match(dist, data)

    def test_latent_sequence_statistics_have_encoded_parity(self):
        dist = HiddenMarkovModelDistribution(
            [
                CategoricalDistribution({"a": 0.8, "b": 0.2}),
                CategoricalDistribution({"a": 0.2, "b": 0.8}),
            ],
            [0.5, 0.5],
            [[0.9, 0.1], [0.1, 0.9]],
            len_dist=IntegerCategoricalDistribution(1, [0.2, 0.4, 0.4]),
        )
        data = [["a", "a"], ["b", "b", "a"], ["a"]]

        self.assert_data_and_encoded_match(dist, data)

    def test_hmm_fast_statistics_match_accumulator_replay(self):
        dist = HiddenMarkovModelDistribution(
            [
                GaussianDistribution(-1.0, 1.0),
                GaussianDistribution(2.0, 1.5),
                GaussianDistribution(5.0, 2.0),
            ],
            [0.4, 0.35, 0.25],
            [[0.75, 0.2, 0.05], [0.15, 0.7, 0.15], [0.1, 0.2, 0.7]],
            len_dist=IntegerCategoricalDistribution(5, [0.1, 0.2, 0.25, 0.25, 0.2]),
        )
        data = [
            [-1.2, -0.8, 0.1, 1.0, 2.2],
            [2.4, 1.8, 3.1, 4.4, 4.9, 5.3, 5.1],
            [0.0, -0.5, 1.5, 2.5, 3.5, 4.0, 4.6, 5.0],
        ]
        enc = dist.dist_to_encoder().seq_encode(data)
        view = dist.to_fisher()

        fast = view.seq_expected_statistics(enc)
        replay = view._matrix_from_values(view._accumulator_value_rows(enc, dist))

        np.testing.assert_allclose(fast, replay, atol=1.0e-10)

    def test_pcfg_fast_statistics_match_accumulator_replay(self):
        dist = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "B", 0.4), ("B", "A", 0.6)]},
            terminal_rules={
                "A": [(CategoricalDistribution({"a": 0.7, "b": 0.3}), 1.0)],
                "B": [(CategoricalDistribution({"a": 0.2, "b": 0.8}), 1.0)],
            },
            start="S",
        )
        data = [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]]
        enc = dist.dist_to_encoder().seq_encode(data)
        view = dist.to_fisher()

        fast = view.seq_expected_statistics(enc)
        replay_values = []
        estimator = dist.estimator()
        for i in range(len(data)):
            weights = np.zeros(len(data), dtype=np.float64)
            weights[i] = 1.0
            acc = estimator.accumulator_factory().make()
            acc.seq_update(enc, weights, dist)
            replay_values.append(acc.value())
        replay = view._matrix_from_values(replay_values)
        raw = view.expected_statistics_matrix(data=data)

        np.testing.assert_allclose(fast, replay, atol=1.0e-10)
        np.testing.assert_allclose(raw, fast, atol=1.0e-10)

    def test_stats_combinators_have_encoded_parity(self):
        cases = [
            (
                OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25, missing_value=None),
                [None, -1.0, 0.5, None, 2.0],
            ),
            (
                WeightedDistribution(GaussianDistribution(0.0, 1.0)),
                [(-1.0, 0.5), (0.0, 1.0), (2.0, 2.0)],
            ),
            (
                CompositeDistribution([GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0)]),
                [(-1.0, 0), (0.5, 3), (2.0, 1)],
            ),
            (
                SelectDistribution(
                    [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.0)], lambda x: 0 if x < 0.0 else 1
                ),
                [-2.0, -0.5, 1.0, 3.0],
            ),
            (
                BinomialDistribution(0.4, 5),
                [0, 2, 5, 1],
            ),
        ]

        for dist, data in cases:
            with self.subTest(dist=type(dist).__name__):
                self.assert_data_and_encoded_match(dist, data)

    def test_important_stats_views_have_encoded_parity(self):
        cases = [
            (
                DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]),
                [[-1.0, 0.5], [0.0, 1.0], [2.0, 3.0]],
            ),
            (
                MultivariateGaussianDistribution([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]]),
                [[-1.0, 0.5], [0.0, 1.0], [2.0, 3.0]],
            ),
            (
                LogGaussianDistribution(0.0, 1.5),
                [0.25, 1.0, 3.0],
            ),
            (
                ExponentialDistribution(2.0),
                [0.1, 1.0, 3.0],
            ),
            (
                GammaDistribution(2.0, 3.0),
                [0.2, 1.5, 4.0],
            ),
            (
                NegativeBinomialDistribution(3.0, 0.4),
                [0, 2, 5],
            ),
            (
                BetaDistribution(2.0, 3.0),
                [0.2, 0.5, 0.8],
            ),
            (
                DirichletDistribution([2.0, 3.0, 4.0]),
                [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.2, 0.4]],
            ),
            (
                IndianBuffetProcessDistribution(3, feature_probs=[0.2, 0.5, 0.8], data_format="dense"),
                [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
            ),
            (
                MultinomialDistribution(
                    CategoricalDistribution({"a": 0.25, "b": 0.75}),
                    len_dist=IntegerCategoricalDistribution(1, [0.2, 0.4, 0.4]),
                ),
                [[("a", 1), ("b", 2)], [("b", 1)], [("a", 2)]],
            ),
            (
                MultinomialDistribution(
                    CategoricalDistribution({"a": 0.25, "b": 0.75}),
                    len_dist=IntegerCategoricalDistribution(1, [0.2, 0.4, 0.4]),
                    len_normalized=True,
                ),
                [[("a", 1), ("b", 2)], [("b", 1)], [("a", 2)]],
            ),
        ]

        for dist, data in cases:
            with self.subTest(dist=type(dist).__name__):
                self.assert_data_and_encoded_match(dist, data)

    def test_sequence_and_hmm_model_fisher_are_finite(self):
        seq = SequenceDistribution(GaussianDistribution(0.0, 1.0), IntegerCategoricalDistribution(1, [0.2, 0.5, 0.3]))
        hmm = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.8, 0.2], [0.3, 0.7]],
            len_dist=IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3]),
        )

        for dist in (seq, hmm):
            view = dist.to_fisher()
            info = view.fisher_information(diagonal=True, ridge=0.0)
            self.assertEqual(info.shape, (len(view.vectorizer.labels),))
            self.assertTrue(np.all(np.isfinite(info)))
            self.assertTrue(np.all(info >= -1.0e-12))

    def test_hmm_full_model_fisher_is_explicitly_diagonal_only(self):
        dist = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.5)],
            [0.6, 0.4],
            [[0.8, 0.2], [0.3, 0.7]],
            len_dist=IntegerCategoricalDistribution(1, [0.3, 0.4, 0.3]),
        )
        view = dist.to_fisher()
        data = [[-1.0, -0.5], [2.0, 2.5, -1.0], [0.0]]
        stats = view.expected_statistics_matrix(data=data)

        self.assertTrue(np.all(np.isfinite(view.fisher_information(diagonal=True, ridge=0.0))))
        with self.assertRaises(NotImplementedError):
            view.fisher_information(diagonal=False, ridge=0.0)
        with self.assertRaises(NotImplementedError):
            view.fisher_vectors(stats=stats, metric="full")

        info = view.observed_fisher_information(stats=stats, diagonal=False, ridge=0.0)
        self.assertEqual(info.shape, (stats.shape[1], stats.shape[1]))
        self.assertTrue(np.all(np.isfinite(info)))
        fv = view.observed_fisher_vectors(stats=stats, metric="full")
        self.assertEqual(fv.shape, stats.shape)
        self.assertTrue(np.all(np.isfinite(fv)))

    def test_bayes_combinators_have_encoded_parity(self):
        from pysp.stats.bayes.dirichlet import DirichletDistribution as BayesianDirichletDistribution
        from pysp.stats.combinator.optional import OptionalDistribution as BayesianOptionalDistribution
        from pysp.stats.combinator.sequence import SequenceDistribution as BayesianSequenceDistribution
        from pysp.stats.latent.hidden_markov import (
            HiddenMarkovModelDistribution as BayesianHiddenMarkovModelDistribution,
        )
        from pysp.stats.leaf.binomial import BinomialDistribution as BayesianBinomialDistribution
        from pysp.stats.leaf.categorical import CategoricalDistribution as BayesianCategoricalDistribution
        from pysp.stats.leaf.exponential import ExponentialDistribution as BayesianExponentialDistribution
        from pysp.stats.leaf.gamma import GammaDistribution as BayesianGammaDistribution
        from pysp.stats.leaf.gaussian import GaussianDistribution as BayesianGaussianDistribution
        from pysp.stats.leaf.integer_categorical import (
            IntegerCategoricalDistribution as BayesianIntegerCategoricalDistribution,
        )
        from pysp.stats.leaf.log_gaussian import LogGaussianDistribution as BayesianLogGaussianDistribution
        from pysp.stats.multivariate.diagonal_gaussian import (
            DiagonalGaussianDistribution as BayesianDiagonalGaussianDistribution,
        )
        from pysp.stats.multivariate.multivariate_gaussian import (
            MultivariateGaussianDistribution as BayesianMultivariateGaussianDistribution,
        )

        cases = [
            (
                BayesianSequenceDistribution(
                    BayesianGaussianDistribution(0.0, 1.0),
                    len_dist=BayesianIntegerCategoricalDistribution(1, [0.2, 0.5, 0.3]),
                ),
                [[-1.0, 0.0], [1.0], [2.0, 2.5, 3.0]],
            ),
            (
                BayesianOptionalDistribution(BayesianGaussianDistribution(0.0, 1.0), p=0.25, missing_value=None),
                [None, -1.0, 0.5, None, 2.0],
            ),
            (
                BayesianBinomialDistribution(0.4, 5),
                [0, 2, 5, 1],
            ),
            (
                BayesianCategoricalDistribution({"x": 0.2, "y": 0.8}),
                ["x", "y", "y", "x"],
            ),
            (
                BayesianHiddenMarkovModelDistribution(
                    [BayesianGaussianDistribution(-1.0, 1.0), BayesianGaussianDistribution(2.0, 1.5)],
                    [0.6, 0.4],
                    [[0.8, 0.2], [0.3, 0.7]],
                    len_dist=BayesianIntegerCategoricalDistribution(1, [0.3, 0.4, 0.3]),
                ),
                [[-1.0, -0.5], [2.0, 2.5, -1.0], [0.0]],
            ),
            (
                BayesianDiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]),
                [[-1.0, 0.5], [0.0, 1.0], [2.0, 3.0]],
            ),
            (
                BayesianMultivariateGaussianDistribution([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]]),
                [[-1.0, 0.5], [0.0, 1.0], [2.0, 3.0]],
            ),
            (
                BayesianLogGaussianDistribution(0.0, 1.5),
                [0.25, 1.0, 3.0],
            ),
            (
                BayesianExponentialDistribution(2.0),
                [0.1, 1.0, 3.0],
            ),
            (
                BayesianGammaDistribution(2.0, 3.0),
                [0.2, 1.5, 4.0],
            ),
            (
                BayesianDirichletDistribution([2.0, 3.0, 4.0]),
                [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.2, 0.4]],
            ),
        ]

        for dist, data in cases:
            with self.subTest(dist=type(dist).__name__):
                self.assert_data_and_encoded_match(dist, data)

        for dist in (cases[0][0], cases[4][0]):
            with self.subTest(model_fisher=type(dist).__name__):
                view = dist.to_fisher()
                info = view.fisher_information(diagonal=True, ridge=0.0)
                self.assertEqual(info.shape, (len(view.vectorizer.labels),))
                self.assertTrue(np.all(np.isfinite(info)))
                self.assertTrue(np.all(info >= -1.0e-12))

    def test_vectorizer_handles_dynamic_dictionary_support(self):
        dist = CategoricalDistribution({"a": 0.5, "b": 0.5})
        view = dist.to_fisher()

        mat = view.statistics_matrix(data=["a", "b", "a"])

        self.assertIsInstance(view.vectorizer, SufficientStatisticVectorizer)
        self.assertEqual(set(view.vectorizer.label_strings()), {"'a'", "'b'"})
        np.testing.assert_allclose(mat, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    def test_latent_model_uses_expected_complete_data_statistics(self):
        dist = MixtureDistribution([GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)], [0.25, 0.75])
        view = dist.to_fisher()

        ss = view.structured_statistics(0.0)

        np.testing.assert_allclose(ss[0], dist.posterior(0.0))

    def test_mixture_fisher_matrix_is_posterior_gated(self):
        dist = MixtureDistribution([GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)], [0.25, 0.75])
        view = dist.to_fisher()
        x = 0.3

        mat = view.expected_statistics_matrix(data=[x])
        z = dist.posterior(x)
        child_stats = GaussianDistribution(0.0, 1.0).to_fisher().expected_statistics_matrix(data=[x])[0]
        expected = np.concatenate((z, z[0] * child_stats, z[1] * child_stats)).reshape(1, -1)

        np.testing.assert_allclose(mat, expected)

    def test_mixture_view_has_complete_data_fisher_information(self):
        dist = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 3.0)], [0.4, 0.6])
        view = dist.to_fisher()

        stats = view.expected_statistics_matrix(data=[-2.0, 0.0, 3.0])
        info = view.fisher_information(ridge=0.0)

        self.assertEqual(stats.shape, (3, 10))
        self.assertEqual(info.shape, (10, 10))
        np.testing.assert_allclose(info, info.T, atol=1.0e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(info).min(), -1.0e-10)

    def test_joint_mixture_fisher_uses_equivalent_pair_mixture(self):
        dist = JointMixtureDistribution(
            components1=[GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.5)],
            components2=[GaussianDistribution(0.0, 2.0), GaussianDistribution(3.0, 0.75)],
            w1=[0.4, 0.6],
            w2=[0.5, 0.5],
            taus12=[[0.8, 0.2], [0.25, 0.75]],
            taus21=[[0.64, 0.18181818181818182], [0.36, 0.8181818181818182]],
        )
        data = [(-1.2, -0.3), (0.0, 2.1), (2.6, 3.5)]
        view = dist.to_fisher()

        self.assertEqual(type(view).__name__, "JointMixtureFisherView")
        self.assertEqual(view.num_pairs, 4)
        for x in data:
            self.assertAlmostEqual(view.log_density(x), dist.log_density(x), places=12)
        self.assert_data_and_encoded_match(dist, data)

    def test_observed_fisher_information_uses_score_covariance(self):
        dist = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 3.0)], [0.4, 0.6])
        view = dist.to_fisher()
        stats = view.expected_statistics_matrix(data=[-2.0, -0.5, 0.0, 1.0, 3.0])
        centered = stats - view._model_mean().reshape((1, -1))

        expected_diag = np.mean(centered * centered, axis=0)
        expected_full = np.dot(centered.T, centered) / float(centered.shape[0])

        np.testing.assert_allclose(
            view.observed_fisher_information(stats=stats, diagonal=True, ridge=0.0), expected_diag
        )
        np.testing.assert_allclose(
            view.observed_fisher_information(stats=stats, diagonal=False, ridge=0.0), expected_full
        )
        np.testing.assert_allclose(view.observed_fisher_vectors(stats=stats, metric="identity"), centered)

    def test_weighted_observed_fisher_uses_empirical_center(self):
        dist = WeightedDistribution(GaussianDistribution(0.0, 1.0))
        view = dist.to_fisher()
        stats = view.expected_statistics_matrix(data=[(-1.0, 0.5), (0.0, 1.0), (2.0, 2.0)])
        centered = stats - stats.mean(axis=0, keepdims=True)

        np.testing.assert_allclose(view.score_center(stats=stats), stats.mean(axis=0))
        np.testing.assert_allclose(view.observed_fisher_vectors(stats=stats, metric="identity"), centered)

    def test_fisher_vectors_use_model_center_for_exact_views(self):
        dist = GaussianDistribution(0.0, 1.0)
        view = dist.to_fisher()

        stats = view.statistics_matrix(data=[-1.0, 0.0, 1.0])
        fv = view.fisher_vectors(stats=stats, metric="identity")

        self.assertEqual(fv.shape, stats.shape)
        self.assertTrue(np.all(np.isfinite(fv)))
        np.testing.assert_allclose(fv, stats - view.mean_statistics())

    def test_leaf_views_have_exact_fisher_information(self):
        g_view = GaussianDistribution(0.0, 2.0).to_fisher()
        np.testing.assert_allclose(g_view.fisher_information(ridge=0.0), np.diag([2.0, 8.0, 0.0, 0.0]))

        count_cases = [
            (BernoulliDistribution(0.35), 0.35, 0.35 * 0.65),
            (PoissonDistribution(3.0), 3.0, 3.0),
            (GeometricDistribution(0.25), 4.0, 12.0),
            (BinomialDistribution(0.4, 5), 2.0, 1.2),
        ]
        for dist, mean, var in count_cases:
            with self.subTest(dist=type(dist).__name__):
                view = dist.to_fisher()
                np.testing.assert_allclose(view.mean_statistics(), [1.0, mean])
                np.testing.assert_allclose(view.fisher_information(ridge=0.0), np.diag([0.0, var]))

        c_view = CategoricalDistribution({"a": 0.25, "b": 0.75}).to_fisher()
        np.testing.assert_allclose(c_view.fisher_information(ridge=0.0), [[0.1875, -0.1875], [-0.1875, 0.1875]])

        ic_view = IntegerCategoricalDistribution(2, [0.2, 0.3, 0.5]).to_fisher()
        p = np.asarray([0.2, 0.3, 0.5])
        np.testing.assert_allclose(ic_view.mean_statistics(), p)
        np.testing.assert_allclose(ic_view.fisher_information(ridge=0.0), np.diag(p) - np.outer(p, p))

        dg_view = DiagonalGaussianDistribution([1.0, -0.5], [2.0, 3.0]).to_fisher()
        np.testing.assert_allclose(dg_view.mean_statistics(), [1.0, -0.5, 3.0, 3.25, 1.0])
        dg_info = np.zeros((5, 5), dtype=np.float64)
        dg_info[0, 0] = 2.0
        dg_info[1, 1] = 3.0
        dg_info[0, 2] = dg_info[2, 0] = 4.0
        dg_info[1, 3] = dg_info[3, 1] = -3.0
        dg_info[2, 2] = 16.0
        dg_info[3, 3] = 21.0
        np.testing.assert_allclose(dg_view.fisher_information(ridge=0.0), dg_info)

        mvn_view = MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).to_fisher()
        np.testing.assert_allclose(mvn_view.mean_statistics(), [0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(mvn_view.fisher_information(ridge=0.0), np.diag([1.0, 1.0, 2.0, 1.0, 2.0, 0.0]))

        lg_view = LogGaussianDistribution(0.0, 2.0).to_fisher()
        np.testing.assert_allclose(lg_view.mean_statistics(), [0.0, 2.0, 1.0, 1.0])
        np.testing.assert_allclose(lg_view.fisher_information(ridge=0.0), np.diag([2.0, 8.0, 0.0, 0.0]))

        exp_view = ExponentialDistribution(2.0).to_fisher()
        np.testing.assert_allclose(exp_view.mean_statistics(), [1.0, 2.0])
        np.testing.assert_allclose(exp_view.fisher_information(ridge=0.0), np.diag([0.0, 4.0]))

        gamma_view = GammaDistribution(3.0, 2.0).to_fisher()
        np.testing.assert_allclose(gamma_view.mean_statistics(), [1.0, digamma(3.0) + np.log(2.0), 6.0])
        gamma_info = np.zeros((3, 3), dtype=np.float64)
        gamma_info[1, 1] = trigamma(3.0)
        gamma_info[1, 2] = gamma_info[2, 1] = 2.0
        gamma_info[2, 2] = 12.0
        np.testing.assert_allclose(gamma_view.fisher_information(ridge=0.0), gamma_info)

        nb_view = NegativeBinomialDistribution(5.0, 0.4).to_fisher()
        np.testing.assert_allclose(nb_view.mean_statistics(), [1.0, 7.5])
        np.testing.assert_allclose(nb_view.fisher_information(ridge=0.0), np.diag([0.0, 18.75]))

        beta_view = BetaDistribution(2.0, 3.0).to_fisher()
        beta_info = np.zeros((3, 3), dtype=np.float64)
        beta_info[1, 1] = trigamma(2.0) - trigamma(5.0)
        beta_info[1, 2] = beta_info[2, 1] = -trigamma(5.0)
        beta_info[2, 2] = trigamma(3.0) - trigamma(5.0)
        np.testing.assert_allclose(beta_view.fisher_information(ridge=0.0), beta_info)

        dir_view = DirichletDistribution([2.0, 3.0, 4.0]).to_fisher()
        alpha = np.asarray([2.0, 3.0, 4.0])
        dir_info = np.zeros((4, 4), dtype=np.float64)
        dir_info[:3, :3] = np.diag(trigamma(alpha)) - trigamma(alpha.sum())
        np.testing.assert_allclose(
            dir_view.mean_statistics(), np.concatenate((digamma(alpha) - digamma(alpha.sum()), [1.0]))
        )
        np.testing.assert_allclose(dir_view.fisher_information(ridge=0.0), dir_info)

        ibp_view = IndianBuffetProcessDistribution(3, feature_probs=[0.2, 0.5, 0.8], data_format="dense").to_fisher()
        p = np.asarray([0.2, 0.5, 0.8])
        np.testing.assert_allclose(ibp_view.mean_statistics(), p)
        np.testing.assert_allclose(ibp_view.fisher_information(ridge=0.0), np.diag(p * (1.0 - p)))

    def test_generic_fisher_vectors_add_ridge_once(self):
        view = FisherView(GaussianDistribution(0.0, 1.0))
        stats = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 1.0],
                [4.0, 3.0],
            ]
        )
        ridge = 0.25
        centered = stats - stats.mean(axis=0, keepdims=True)
        diag = np.mean(centered * centered, axis=0)

        np.testing.assert_allclose(
            view.fisher_vectors(stats=stats, metric="diagonal", ridge=ridge),
            centered / np.sqrt(diag.reshape((1, -1)) + ridge),
        )

    def test_composite_model_fisher_is_child_block_diagonal(self):
        dist = CompositeDistribution((GaussianDistribution(1.0, 2.0), PoissonDistribution(3.0)))
        view = dist.to_fisher()
        blocks = [child.fisher_information(ridge=0.0) for child in view.child_views]
        expected = np.zeros_like(view.fisher_information(ridge=0.0))
        pos = 0
        for block in blocks:
            n = block.shape[0]
            expected[pos : pos + n, pos : pos + n] = block
            pos += n

        np.testing.assert_allclose(view.fisher_information(ridge=0.0), expected)

    def test_optional_model_fisher_matches_exact_enumeration(self):
        dist = OptionalDistribution(CategoricalDistribution({"a": 0.2, "b": 0.8}), p=0.3, missing_value=None)
        data = [None, "a", "b"]
        probs = np.asarray([0.3, 0.7 * 0.2, 0.7 * 0.8])
        view = dist.to_fisher()
        stats = view.expected_statistics_matrix(data=data)
        mean, cov = self.weighted_moments(stats, probs)

        np.testing.assert_allclose(view.mean_statistics(), mean, atol=1.0e-12)
        np.testing.assert_allclose(view.fisher_information(ridge=0.0), cov, atol=1.0e-12)

    def test_sequence_model_fisher_matches_exact_enumeration(self):
        child = IntegerCategoricalDistribution(0, [0.3, 0.7])
        length = IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        child_probs = dict(enumerate(child.p_vec))

        for len_normalized in (False, True):
            with self.subTest(len_normalized=len_normalized):
                dist = SequenceDistribution(child, length, len_normalized=len_normalized)
                data = [[]]
                probs = [length.p_vec[0]]
                for a in (0, 1):
                    data.append([a])
                    probs.append(length.p_vec[1] * child_probs[a])
                for a in (0, 1):
                    for b in (0, 1):
                        data.append([a, b])
                        probs.append(length.p_vec[2] * child_probs[a] * child_probs[b])

                view = dist.to_fisher()
                stats = view.expected_statistics_matrix(data=data)
                mean, cov = self.weighted_moments(stats, probs)

                np.testing.assert_allclose(view.mean_statistics(), mean, atol=1.0e-12)
                np.testing.assert_allclose(view.fisher_information(ridge=0.0), cov, atol=1.0e-12)

    def test_hmm_model_fisher_matches_exact_finite_enumeration(self):
        dist = HiddenMarkovModelDistribution(
            [
                CategoricalDistribution({"a": 0.8, "b": 0.2}),
                CategoricalDistribution({"a": 0.3, "b": 0.7}),
            ],
            [0.6, 0.4],
            [[0.7, 0.3], [0.2, 0.8]],
            len_dist=IntegerCategoricalDistribution(1, [0.55, 0.45]),
        )
        enumerated = list(dist.enumerator())
        data = [value for value, _ in enumerated]
        probs = np.asarray([np.exp(log_prob) for _, log_prob in enumerated], dtype=np.float64)
        view = dist.to_fisher()
        stats = view.expected_statistics_matrix(data=data)
        mean, cov = self.weighted_moments(stats, probs)

        np.testing.assert_allclose(probs.sum(), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(view.mean_statistics(), mean, atol=1.0e-12)
        np.testing.assert_allclose(view.fisher_information(ridge=0.0), cov, atol=1.0e-12)
        np.testing.assert_allclose(view.fisher_information(diagonal=True, ridge=0.0), np.diag(cov), atol=1.0e-12)

        fv = view.fisher_vectors(stats=stats, metric="full", ridge=1.0e-8)
        self.assertEqual(fv.shape, stats.shape)
        self.assertTrue(np.all(np.isfinite(fv)))

    def test_pcfg_model_fisher_matches_exact_finite_enumeration(self):
        dist = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "B", 0.4), ("B", "A", 0.6)]},
            terminal_rules={
                "A": [(CategoricalDistribution({"a": 0.7, "b": 0.3}), 1.0)],
                "B": [(CategoricalDistribution({"a": 0.2, "b": 0.8}), 1.0)],
            },
            start="S",
        )
        enumerated = list(dist.enumerator())
        data = [value for value, _ in enumerated]
        probs = np.asarray([np.exp(log_prob) for _, log_prob in enumerated], dtype=np.float64)
        view = dist.to_fisher()
        stats = view.expected_statistics_matrix(data=data)
        mean, cov = self.weighted_moments(stats, probs)

        np.testing.assert_allclose(probs.sum(), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(view.mean_statistics(), mean, atol=1.0e-12)
        np.testing.assert_allclose(view.fisher_information(ridge=0.0), cov, atol=1.0e-12)

    def test_recursive_pcfg_uses_observed_metric_on_data(self):
        dist = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("S", "A", 0.5)]},
            terminal_rules={
                "S": [(CategoricalDistribution({"z": 1.0}), 0.5)],
                "A": [(CategoricalDistribution({"a": 1.0}), 1.0)],
            },
            start="S",
        )
        view = dist.to_fisher()
        stats = view.expected_statistics_matrix(data=[["z"], ["z", "a"], ["z", "a", "a"]])

        with self.assertRaises(NotImplementedError):
            view.fisher_information(ridge=0.0)

        info = view.observed_fisher_information(stats=stats, diagonal=False, ridge=0.0)
        self.assertEqual(info.shape, (stats.shape[1], stats.shape[1]))
        self.assertTrue(np.all(np.isfinite(info)))

    def test_fresh_single_fisher_vector_has_schema(self):
        view = GaussianDistribution(0.0, 1.0).to_fisher()
        fv = view.fisher_vector(1.25)

        self.assertEqual(fv.shape, (4,))
        self.assertTrue(np.all(np.isfinite(fv)))

    def test_bayes_distribution_inherits_to_fisher(self):
        from pysp.stats.leaf.gaussian import GaussianDistribution as BayesianGaussianDistribution

        dist = BayesianGaussianDistribution(0.0, 1.0)
        view = dist.to_fisher()
        mat = view.statistics_matrix(data=[0.5, 1.5])

        self.assertIsInstance(view, FisherView)
        self.assertEqual(mat.shape[0], 2)
        self.assertGreater(mat.shape[1], 0)
        self.assertTrue(np.all(np.isfinite(mat)))

    def test_bayes_categorical_default_mass_uses_generic_view(self):
        from pysp.stats.leaf.categorical import CategoricalDistribution as BayesianCategoricalDistribution

        dist = BayesianCategoricalDistribution({"a": 0.5}, default_value=0.5)
        view = dist.to_fisher()
        mat = view.expected_statistics_matrix(data=["a", "unmapped"])

        self.assertEqual(type(view), FisherView)
        self.assertEqual(mat.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(mat)))


if __name__ == "__main__":
    unittest.main()
