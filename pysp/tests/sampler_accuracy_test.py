import math
import unittest

import numpy as np

import pysp.bstats as bstats
import pysp.stats as stats


class SamplerAccuracyTestCase(unittest.TestCase):

    def assert_mean_var_close(self, name, samples, expected_mean, expected_var,
                              mean_z=7.0, var_rel_tol=0.15):
        samples = np.asarray(samples, dtype=float)
        n = float(len(samples))
        obs_mean = float(np.mean(samples))
        obs_var = float(np.var(samples))
        mean_tol = max(1.0e-12, mean_z * math.sqrt(expected_var / n))

        self.assertLessEqual(
            abs(obs_mean - expected_mean), mean_tol,
            '%s sample mean %r differs from expected %r by more than %r' %
            (name, obs_mean, expected_mean, mean_tol))
        self.assertLessEqual(
            abs(obs_var - expected_var) / max(abs(expected_var), 1.0e-12), var_rel_tol,
            '%s sample variance %r differs from expected %r by more than %.1f%%' %
            (name, obs_var, expected_var, 100.0 * var_rel_tol))

    def test_core_scalar_sampler_moments_match_parameterization(self):
        n = 50_000
        scalar_cases = [
            ('bernoulli', stats.BernoulliDistribution(0.3), 0.3, 0.3 * 0.7),
            ('binomial', stats.BinomialDistribution(0.4, 10, min_val=1), 1.0 + 10.0 * 0.4, 10.0 * 0.4 * 0.6),
            ('poisson', stats.PoissonDistribution(3.0), 3.0, 3.0),
            ('negative_binomial', stats.NegativeBinomialDistribution(3.0, 0.45),
             3.0 * 0.55 / 0.45, 3.0 * 0.55 / (0.45 ** 2)),
            ('gaussian', stats.GaussianDistribution(1.0, 2.0), 1.0, 2.0),
            ('gamma', stats.GammaDistribution(2.0, 3.0), 6.0, 18.0),
            ('exponential', stats.ExponentialDistribution(2.0), 2.0, 4.0),
            ('beta', stats.BetaDistribution(2.0, 5.0), 2.0 / 7.0, (2.0 * 5.0) / ((7.0 ** 2) * 8.0)),
            ('laplace', stats.LaplaceDistribution(0.0, 1.5), 0.0, 2.0 * 1.5 ** 2),
            ('logistic', stats.LogisticDistribution(0.0, 1.0), 0.0, math.pi ** 2 / 3.0),
            ('uniform', stats.UniformDistribution(-1.0, 3.0), 1.0, (4.0 ** 2) / 12.0),
            ('student_t', stats.StudentTDistribution(8.0, loc=1.0, scale=1.5),
             1.0, (1.5 ** 2) * 8.0 / 6.0),
            ('rayleigh', stats.RayleighDistribution(2.0),
             2.0 * math.sqrt(math.pi / 2.0), ((4.0 - math.pi) / 2.0) * 2.0 ** 2),
            ('weibull', stats.WeibullDistribution(1.5, 2.0),
             2.0 * math.gamma(1.0 + 1.0 / 1.5),
             4.0 * (math.gamma(1.0 + 2.0 / 1.5) - math.gamma(1.0 + 1.0 / 1.5) ** 2)),
        ]

        for name, dist, expected_mean, expected_var in scalar_cases:
            with self.subTest(name=name):
                samples = dist.sampler(seed=911).sample(size=n)
                self.assert_mean_var_close(name, samples, expected_mean, expected_var)

    def test_discrete_sampler_frequencies_match_probabilities(self):
        n = 60_000

        cat = stats.CategoricalDistribution({'a': 0.2, 'b': 0.3, 'c': 0.5})
        cat_samples = cat.sampler(seed=17).sample(size=n)
        for value, expected in cat.pmap.items():
            observed = cat_samples.count(value) / float(n)
            self.assertLessEqual(abs(observed - expected), 0.012)

        int_cat = stats.IntegerCategoricalDistribution(2, [0.2, 0.3, 0.5])
        int_samples = np.asarray(int_cat.sampler(seed=19).sample(size=n), dtype=int)
        for value, expected in zip([2, 3, 4], [0.2, 0.3, 0.5]):
            observed = float(np.mean(int_samples == value))
            self.assertLessEqual(abs(observed - expected), 0.012)

        bern_set = stats.BernoulliSetDistribution({'a': 0.15, 'b': 0.55, 'c': 0.85}, min_prob=0.0)
        set_samples = bern_set.sampler(seed=23).sample(size=n)
        for value, expected in bern_set.pmap.items():
            observed = sum(value in sample for sample in set_samples) / float(n)
            self.assertLessEqual(abs(observed - expected), 0.012)

        ibp = stats.IndianBuffetProcessDistribution(
            4, feature_probs=[0.1, 0.4, 0.7, 0.9], data_format='dense')
        ibp_samples = np.asarray(ibp.sampler(seed=29).sample(size=n), dtype=float)
        np.testing.assert_allclose(ibp_samples.mean(axis=0), ibp.feature_probs, atol=0.012, rtol=0.0)

    def test_vector_sampler_moments_match_parameters(self):
        n = 40_000

        dirichlet = stats.DirichletDistribution([1.0, 2.0, 3.0])
        dir_samples = np.asarray(dirichlet.sampler(seed=31).sample(size=n), dtype=float)
        np.testing.assert_allclose(dir_samples.mean(axis=0), np.asarray([1.0, 2.0, 3.0]) / 6.0,
                                   atol=0.01, rtol=0.0)

        dmvn = stats.DiagonalGaussianDistribution([0.5, -1.0], [1.0, 2.0])
        dmvn_samples = np.asarray(dmvn.sampler(seed=37).sample(size=n), dtype=float)
        np.testing.assert_allclose(dmvn_samples.mean(axis=0), [0.5, -1.0], atol=0.035, rtol=0.0)
        np.testing.assert_allclose(dmvn_samples.var(axis=0), [1.0, 2.0], atol=0.08, rtol=0.0)

        mvn = stats.MultivariateGaussianDistribution([0.5, -1.0], [[1.0, 0.25], [0.25, 2.0]])
        mvn_samples = np.asarray(mvn.sampler(seed=41).sample(size=n), dtype=float)
        np.testing.assert_allclose(mvn_samples.mean(axis=0), [0.5, -1.0], atol=0.04, rtol=0.0)
        np.testing.assert_allclose(np.cov(mvn_samples, rowvar=False), [[1.0, 0.25], [0.25, 2.0]],
                                   atol=0.08, rtol=0.0)

    def assert_markov_transition_rates(self, samples, init_prob, trans_prob, tol=0.025):
        init_counts = np.zeros(len(init_prob), dtype=float)
        trans_counts = np.zeros_like(np.asarray(trans_prob, dtype=float))
        for seq in samples:
            init_counts[int(seq[0])] += 1.0
            for a, b in zip(seq[:-1], seq[1:]):
                trans_counts[int(a), int(b)] += 1.0

        init_obs = init_counts / init_counts.sum()
        trans_obs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(init_obs, init_prob, atol=tol, rtol=0.0)
        np.testing.assert_allclose(trans_obs, trans_prob, atol=tol, rtol=0.0)

    def test_markov_and_hmm_transition_accuracy(self):
        n = 8_000
        init = np.asarray([0.65, 0.35])
        trans = np.asarray([[0.75, 0.25], [0.15, 0.85]])

        markov = stats.MarkovChainDistribution(
            {0: init[0], 1: init[1]},
            {0: {0: trans[0, 0], 1: trans[0, 1]}, 1: {0: trans[1, 0], 1: trans[1, 1]}},
            len_dist=stats.CategoricalDistribution({30: 1.0}))
        self.assert_markov_transition_rates(markov.sampler(seed=43).sample(size=n), init, trans)

        hmm = stats.HiddenMarkovModelDistribution(
            [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})],
            init, trans, len_dist=stats.CategoricalDistribution({30: 1.0}), use_numba=False)
        self.assert_markov_transition_rates(hmm.sampler(seed=47).sample(size=n), init, trans)

        bhmm = bstats.HiddenMarkovModelDistribution(
            [bstats.CategoricalDistribution({0: 1.0}), bstats.CategoricalDistribution({1: 1.0})],
            init, trans, len_dist=bstats.CategoricalDistribution({30: 1.0}))
        self.assert_markov_transition_rates(bhmm.sampler(seed=53).sample(size=n), init, trans)

    def test_nested_child_samplers_are_not_reusing_identical_streams(self):
        n = 30_000

        comp = stats.CompositeDistribution((
            stats.GaussianDistribution(0.0, 1.0),
            stats.GaussianDistribution(0.0, 1.0),
        ))
        comp_samples = np.asarray(comp.sampler(seed=59).sample(size=n), dtype=float)
        self.assertLess(abs(float(np.corrcoef(comp_samples[:, 0], comp_samples[:, 1])[0, 1])), 0.05)

        mix = stats.MixtureDistribution([
            stats.CompositeDistribution((stats.PointMassDistribution(0), stats.GaussianDistribution(0.0, 1.0))),
            stats.CompositeDistribution((stats.PointMassDistribution(1), stats.GaussianDistribution(0.0, 1.0))),
        ], [0.5, 0.5])
        mix_samples = mix.sampler(seed=61).sample(size=n)
        labels = np.asarray([x[0] for x in mix_samples], dtype=float)
        values = np.asarray([x[1] for x in mix_samples], dtype=float)
        self.assertLess(abs(float(np.corrcoef(labels, values)[0, 1])), 0.05)
        self.assertLess(abs(float(values[labels == 0.0].mean() - values[labels == 1.0].mean())), 0.05)

        cond = stats.ConditionalDistribution(
            {
                0: stats.BernoulliDistribution(0.5),
                1: stats.BernoulliDistribution(0.5),
            },
            given_dist=stats.CategoricalDistribution({0: 0.5, 1: 0.5}))
        cond_samples = cond.sampler(seed=67).sample(size=n)
        given = np.asarray([x[0] for x in cond_samples], dtype=int)
        emitted = np.asarray([x[1] for x in cond_samples], dtype=float)
        self.assertLess(abs(float(np.mean(emitted[given == 0]) - np.mean(emitted[given == 1]))), 0.025)
        self.assertTrue(cond.sampler(seed=67).has_given_sampler)

        seq = stats.SequenceDistribution(
            stats.BernoulliDistribution(0.5),
            len_dist=stats.IntegerCategoricalDistribution(1, [0.5, 0.5]))
        seq_samples = seq.sampler(seed=71).sample(size=n)
        lengths = np.asarray([len(x) for x in seq_samples], dtype=float)
        first_values = np.asarray([x[0] for x in seq_samples], dtype=float)
        self.assertLess(abs(float(np.corrcoef(lengths, first_values)[0, 1])), 0.05)


if __name__ == '__main__':
    unittest.main()
