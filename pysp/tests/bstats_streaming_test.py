import unittest

import numpy as np

from pysp.bstats import (
    BayesianStreamingEstimator,
    BetaDistribution,
    BinomialDistribution,
    BinomialEstimator,
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeEstimator,
    DictDirichletDistribution,
    DirichletDistribution,
    ExponentialDistribution,
    ExponentialEstimator,
    GammaDistribution,
    GaussianDistribution,
    GaussianEstimator,
    GeometricDistribution,
    GeometricEstimator,
    HiddenMarkovModelEstimator,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    MixtureDistribution,
    MixtureEstimator,
    OptionalDistribution,
    OptionalEstimator,
    PoissonDistribution,
    PoissonEstimator,
    SequenceEstimator,
    forgetting,
    mixture_prior,
)
from pysp.bstats.normgamma import NormalGammaDistribution


def _accumulate(estimator, model, data):
    enc = model.seq_encode(data)
    acc = estimator.accumulator_factory().make()
    acc.seq_update(enc, np.ones(len(data)), model)
    return acc.value()


def _scale_tuple(x, c):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _scale_tuple(v, c) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_scale_tuple(v, c) for v in x)
    if isinstance(x, list):
        return [_scale_tuple(v, c) for v in x]
    if isinstance(x, np.ndarray):
        return x * c
    return x * c


def _scale_for_estimator(estimator, suff_stat, c):
    hook = getattr(estimator, "scale_suff_stat", None)
    if callable(hook):
        return hook(suff_stat, c)
    return _scale_tuple(suff_stat, c)


def _posterior_stream_cases():
    def poisson_case():
        prior = GammaDistribution(2.0, 0.5)
        return (PoissonDistribution(2.0, prior=prior), PoissonEstimator(prior=prior), [0, 1, 3, 2], [4, 1, 0])

    def exponential_case():
        prior = GammaDistribution(2.0, 0.25)
        return (
            ExponentialDistribution(1.2, prior=prior),
            ExponentialEstimator(prior=prior),
            [0.5, 1.5, 1.0],
            [2.0, 0.25],
        )

    def binomial_case():
        prior = BetaDistribution(2.0, 3.0)
        return (BinomialDistribution(10, 0.4, prior=prior), BinomialEstimator(10, prior=prior), [3, 7, 5, 2], [8, 4, 1])

    def geometric_case():
        prior = BetaDistribution(2.0, 3.0)
        return (GeometricDistribution(0.4, prior=prior), GeometricEstimator(prior=prior), [1, 2, 3, 1], [4, 2])

    def categorical_case():
        prior = DictDirichletDistribution({"a": 2.0, "b": 1.5, "c": 1.0})
        return (
            CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}, prior=prior),
            CategoricalEstimator(prior=prior),
            ["a", "b", "a", "c"],
            ["c", "b", "b"],
        )

    def integer_categorical_case():
        prior = DirichletDistribution(np.asarray([2.0, 3.0, 4.0]))
        return (
            IntegerCategoricalDistribution([0.2, 0.5, 0.3], min_index=2, prior=prior),
            IntegerCategoricalEstimator(min_index=2, max_index=4, prior=prior),
            [2, 3, 4, 3],
            [4, 2, 3],
        )

    def optional_case():
        child_prior = NormalGammaDistribution(0.0, 1.0, 2.0, 3.0)
        prior = BetaDistribution(2.0, 5.0)
        return (
            OptionalDistribution(GaussianDistribution(0.0, 1.0, prior=child_prior), p=0.25, prior=prior),
            OptionalEstimator(GaussianEstimator(prior=child_prior), prior=prior),
            [None, -1.0, 0.5, None, 2.0],
            [1.5, None, 0.0],
        )

    return [
        ("poisson", poisson_case),
        ("exponential", exponential_case),
        ("binomial", binomial_case),
        ("geometric", geometric_case),
        ("categorical", categorical_case),
        ("integer_categorical", integer_categorical_case),
        ("optional", optional_case),
    ]


class BayesianStreamingEstimatorTestCase(unittest.TestCase):
    def assertPriorClose(self, actual, expected):
        self.assertEqual(type(actual), type(expected))
        if hasattr(actual, "dists"):
            self.assertEqual(len(actual.dists), len(expected.dists))
            for a, e in zip(actual.dists, expected.dists):
                self.assertPriorClose(a, e)
            return

        actual_params = actual.get_parameters()
        expected_params = expected.get_parameters()
        if isinstance(actual_params, dict):
            self.assertEqual(set(actual_params.keys()), set(expected_params.keys()))
            for key in actual_params:
                np.testing.assert_allclose(
                    np.asarray(actual_params[key], dtype=float),
                    np.asarray(expected_params[key], dtype=float),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            return

        np.testing.assert_allclose(
            np.asarray(actual_params, dtype=float), np.asarray(expected_params, dtype=float), rtol=1.0e-12, atol=1.0e-12
        )

    def test_posterior_carry_uses_previous_posterior_as_next_prior(self):
        prior = NormalGammaDistribution(0.0, 1.0, 2.0, 3.0)
        start = GaussianDistribution(0.0, 1.0, prior=prior)
        stream = BayesianStreamingEstimator(GaussianEstimator(prior=prior), model=start)

        batch1 = [-1.0, 0.0, 2.0]
        model1 = stream.update(batch1)
        expected1 = GaussianEstimator(prior=prior).estimate(_accumulate(GaussianEstimator(prior=prior), start, batch1))
        self.assertPriorClose(model1.get_prior(), expected1.get_prior())

        batch2 = [3.0, 4.0]
        model2 = stream.update(batch2)
        expected2_est = GaussianEstimator(prior=expected1.get_prior())
        expected2 = expected2_est.estimate(_accumulate(expected2_est, model1, batch2))
        self.assertPriorClose(model2.get_prior(), expected2.get_prior())
        self.assertPriorClose(stream.estimator.get_prior(), model2.get_prior())

    def test_forgetting_scales_batch_sufficient_statistics(self):
        prior = NormalGammaDistribution(1.0, 2.0, 3.0, 4.0)
        start = GaussianDistribution(1.0, 2.0, prior=prior)
        stream = BayesianStreamingEstimator(
            GaussianEstimator(prior=prior),
            mode="forgetting",
            schedule=forgetting(0.5),
            model=start,
        )

        batch = [0.0, 2.0, 4.0]
        model = stream.update(batch)
        raw_stats = _accumulate(GaussianEstimator(prior=prior), start, batch)
        expected = GaussianEstimator(prior=prior).estimate(_scale_tuple(raw_stats, 0.5))

        self.assertPriorClose(model.get_prior(), expected.get_prior())
        self.assertAlmostEqual(stream.nobs, 1.5)

    def test_mixture_posterior_carry_updates_weight_and_component_priors(self):
        weight_prior = DirichletDistribution(np.asarray([2.0, 3.0]))
        component_priors = [
            NormalGammaDistribution(-2.0, 1.0, 2.0, 3.0),
            NormalGammaDistribution(2.0, 1.0, 2.0, 3.0),
        ]
        prior = mixture_prior(weight_prior, component_priors)
        start = MixtureDistribution(
            [GaussianDistribution(-1.5, 1.0), GaussianDistribution(1.5, 1.0)],
            [0.45, 0.55],
            prior=prior,
        )
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()], prior=prior)
        stream = BayesianStreamingEstimator(estimator, model=start)

        data = [-2.0, -1.0, 1.5, 2.5]
        model = stream.update(data)

        expected_est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()], prior=prior)
        expected = expected_est.estimate(_accumulate(expected_est, start, data))

        np.testing.assert_allclose(
            model.get_prior().dists[0].get_parameters(),
            expected.get_prior().dists[0].get_parameters(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        for actual_prior, expected_prior in zip(model.get_prior().dists[1].dists, expected.get_prior().dists[1].dists):
            self.assertPriorClose(actual_prior, expected_prior)

    def test_mixture_forgetting_delegates_nested_support_metadata(self):
        rho = 0.35
        weight_prior = DirichletDistribution(np.asarray([2.0, 2.0]))
        child_priors = [
            DirichletDistribution(np.asarray([2.0, 3.0, 4.0])),
            DirichletDistribution(np.asarray([3.0, 2.0, 5.0])),
        ]
        missing_priors = [BetaDistribution(2.0, 5.0), BetaDistribution(3.0, 4.0)]
        components = [
            OptionalDistribution(
                IntegerCategoricalDistribution([0.70, 0.20, 0.10], min_index=2, prior=child_priors[0]),
                p=0.20,
                prior=missing_priors[0],
            ),
            OptionalDistribution(
                IntegerCategoricalDistribution([0.15, 0.25, 0.60], min_index=2, prior=child_priors[1]),
                p=0.35,
                prior=missing_priors[1],
            ),
        ]
        component_estimators = [
            OptionalEstimator(
                IntegerCategoricalEstimator(min_index=2, max_index=4, prior=child_priors[i]),
                prior=missing_priors[i],
            )
            for i in range(2)
        ]
        prior = mixture_prior(weight_prior, [est.get_prior() for est in component_estimators])
        start = MixtureDistribution(components, [0.45, 0.55], prior=prior)
        estimator = MixtureEstimator(component_estimators, prior=prior)
        stream = BayesianStreamingEstimator(estimator, mode="forgetting", schedule=forgetting(rho), model=start)

        data = [None, 2, 3, 4, None, 4, 2, 3]
        model = stream.update(data)

        expected_estimator = MixtureEstimator(
            [
                OptionalEstimator(
                    IntegerCategoricalEstimator(min_index=2, max_index=4, prior=child_priors[i]),
                    prior=missing_priors[i],
                )
                for i in range(2)
            ],
            prior=prior,
        )
        raw_stats = _accumulate(expected_estimator, start, data)
        expected = expected_estimator.estimate(expected_estimator.scale_suff_stat(raw_stats, rho))

        self.assertPriorClose(model.get_prior(), expected.get_prior())
        for component in model.components:
            self.assertEqual(component.dist.min_index, 2)
        self.assertAlmostEqual(stream.nobs, len(data) * rho)

    def test_nested_estimator_scalers_preserve_child_support_metadata(self):
        entry_prior = DirichletDistribution(np.asarray([2.0, 3.0, 4.0]))
        entry_stat = (2, np.asarray([5.0, 7.0, 11.0]))
        len_prior = DirichletDistribution(np.asarray([1.5, 2.5, 3.5]))
        len_stat = (0, np.asarray([3.0, 4.0, 5.0]))
        rho = 0.25

        opt_est = OptionalEstimator(
            IntegerCategoricalEstimator(min_index=2, max_index=4, prior=entry_prior),
            prior=BetaDistribution(2.0, 5.0),
        )
        opt_scaled = opt_est.scale_suff_stat((2.0, 6.0, entry_stat), rho)
        self.assertEqual(opt_scaled[2][0], 2)
        np.testing.assert_allclose(opt_scaled[2][1], entry_stat[1] * rho)

        comp_est = CompositeEstimator(
            [
                IntegerCategoricalEstimator(min_index=2, max_index=4, prior=entry_prior),
                IntegerCategoricalEstimator(min_index=2, max_index=4, prior=entry_prior),
            ]
        )
        comp_scaled = comp_est.scale_suff_stat((entry_stat, entry_stat), rho)
        self.assertEqual(comp_scaled[0][0], 2)
        self.assertEqual(comp_scaled[1][0], 2)

        seq_est = SequenceEstimator(
            IntegerCategoricalEstimator(min_index=2, max_index=4, prior=entry_prior),
            len_estimator=IntegerCategoricalEstimator(min_index=0, max_index=2, prior=len_prior),
        )
        seq_scaled = seq_est.scale_suff_stat((entry_stat, len_stat), rho)
        self.assertEqual(seq_scaled[0][0], 2)
        self.assertEqual(seq_scaled[1][0], 0)

        hmm_est = HiddenMarkovModelEstimator(
            [
                IntegerCategoricalEstimator(min_index=2, max_index=4, prior=entry_prior),
                IntegerCategoricalEstimator(min_index=2, max_index=4, prior=entry_prior),
            ],
            prior=(
                DirichletDistribution(np.ones(2)),
                [DirichletDistribution(np.ones(2)), DirichletDistribution(np.ones(2))],
            ),
            len_estimator=IntegerCategoricalEstimator(min_index=0, max_index=2, prior=len_prior),
        )
        hmm_stat = (
            np.asarray([3.0, 5.0]),
            np.asarray([[4.0, 1.0], [2.0, 6.0]]),
            (entry_stat, entry_stat),
            len_stat,
        )
        hmm_scaled = hmm_est.scale_suff_stat(hmm_stat, rho)
        np.testing.assert_allclose(hmm_scaled[0], hmm_stat[0] * rho)
        np.testing.assert_allclose(hmm_scaled[1], hmm_stat[1] * rho)
        self.assertEqual(hmm_scaled[2][0][0], 2)
        self.assertEqual(hmm_scaled[2][1][0], 2)
        self.assertEqual(hmm_scaled[3][0], 0)

    def test_posterior_carry_across_conjugate_families(self):
        for name, factory in _posterior_stream_cases():
            with self.subTest(family=name):
                start, estimator, batch1, batch2 = factory()
                stream = BayesianStreamingEstimator(estimator, model=start)

                model1 = stream.update(batch1)
                expected_start, expected_estimator, _, _ = factory()
                expected1 = expected_estimator.estimate(_accumulate(expected_estimator, expected_start, batch1))
                self.assertPriorClose(model1.get_prior(), expected1.get_prior())

                model2 = stream.update(batch2)
                expected2_estimator = expected1.estimator()
                expected2 = expected2_estimator.estimate(_accumulate(expected2_estimator, expected1, batch2))
                self.assertPriorClose(model2.get_prior(), expected2.get_prior())
                self.assertPriorClose(stream.estimator.get_prior(), model2.get_prior())

    def test_forgetting_across_conjugate_families(self):
        rho = 0.4
        for name, factory in _posterior_stream_cases():
            with self.subTest(family=name):
                start, estimator, batch1, _ = factory()
                stream = BayesianStreamingEstimator(
                    estimator,
                    mode="forgetting",
                    schedule=forgetting(rho),
                    model=start,
                )

                model = stream.update(batch1)
                expected_start, expected_estimator, _, _ = factory()
                raw_stats = _accumulate(expected_estimator, expected_start, batch1)
                expected = expected_estimator.estimate(_scale_for_estimator(expected_estimator, raw_stats, rho))

                self.assertPriorClose(model.get_prior(), expected.get_prior())
                self.assertAlmostEqual(stream.nobs, len(batch1) * rho)


if __name__ == "__main__":
    unittest.main()
