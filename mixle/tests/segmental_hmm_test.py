import itertools
import unittest

import numpy as np
from scipy.special import logsumexp

from mixle.inference import seq_estimate
from mixle.stats import (
    CategoricalDistribution,
    GaussianDistribution,
    IntegerCategoricalDistribution,
    SegmentalHiddenMarkovEstimator,
    SegmentalHiddenMarkovModelDistribution,
    SequenceDistribution,
    SequenceEstimator,
    StudentTDistribution,
    seq_encode,
)


class SegmentalHiddenMarkovTestCase(unittest.TestCase):
    def make_scalar_model(self):
        return SegmentalHiddenMarkovModelDistribution(
            [GaussianDistribution(-2.0, 1.0), StudentTDistribution(5.0, loc=2.0, scale=1.5)],
            [0.6, 0.4],
            [[0.7, 0.3], [0.2, 0.8]],
            len_dist=IntegerCategoricalDistribution(0, [0.0, 0.0, 1.0, 0.0]),
            name="seg",
        )

    def brute_log_density(self, model, x):
        vals = []
        for states in itertools.product(range(model.n_states), repeat=len(x)):
            lp = model.log_w[states[0]]
            lp += model.emissions[states[0]].log_density(x[0])
            for t in range(1, len(x)):
                lp += model.log_transitions[states[t - 1], states[t]]
                lp += model.emissions[states[t]].log_density(x[t])
            vals.append(lp)
        return logsumexp(vals) + model.len_dist.log_density(len(x))

    def test_log_density_matches_brute_force(self):
        model = self.make_scalar_model()
        x = [-1.5, 2.25]
        self.assertAlmostEqual(model.log_density(x), self.brute_log_density(model, x), places=10)

    def test_seq_log_density_matches_scalar(self):
        model = self.make_scalar_model()
        data = [[-2.0, 1.0], [2.5, 2.0], [-1.0, -2.5]]
        enc = model.dist_to_encoder().seq_encode(data)
        seq_ll = model.seq_log_density(enc)
        scalar_ll = np.asarray([model.log_density(x) for x in data])
        self.assertTrue(np.allclose(seq_ll, scalar_ll, rtol=1.0e-12, atol=1.0e-12))

    def test_variable_length_segment_emissions_sample_and_score(self):
        short_len = CategoricalDistribution({1: 0.8, 2: 0.2})
        long_len = CategoricalDistribution({2: 0.3, 3: 0.7})
        model = SegmentalHiddenMarkovModelDistribution(
            [
                SequenceDistribution(GaussianDistribution(-3.0, 0.5), len_dist=short_len),
                SequenceDistribution(GaussianDistribution(3.0, 0.5), len_dist=long_len),
            ],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
            len_dist=CategoricalDistribution({2: 1.0}),
        )
        data = model.sampler(3).sample(10)
        for obs in data:
            self.assertEqual(len(obs), 2)
            self.assertTrue(all(isinstance(seg, list) for seg in obs))
            self.assertTrue(np.isfinite(model.log_density(obs)))
        enc = seq_encode(data, model=model)
        self.assertTrue(np.all(np.isfinite(model.seq_log_density(enc[0][1]))))

    def test_em_step_with_heterogeneous_emission_classes(self):
        model = self.make_scalar_model()
        data = model.sampler(4).sample(100)
        est = SegmentalHiddenMarkovEstimator(
            [GaussianDistribution(0.0, 4.0).estimator(), StudentTDistribution(5.0).estimator()],
            len_estimator=IntegerCategoricalDistribution(0, [0.1, 0.1, 0.8, 0.0]).estimator(),
            pseudo_count=(1.0, 1.0),
        )
        enc = seq_encode(data, model=model)
        fitted = seq_estimate(enc, est, model)
        self.assertIsInstance(fitted, SegmentalHiddenMarkovModelDistribution)
        self.assertAlmostEqual(fitted.w.sum(), 1.0)
        self.assertTrue(np.allclose(fitted.transitions.sum(axis=1), np.ones(fitted.n_states)))
        self.assertTrue(np.all(np.isfinite(fitted.seq_log_density(enc[0][1]))))

    def test_sequence_emission_estimator_runs(self):
        model = SegmentalHiddenMarkovModelDistribution(
            [
                SequenceDistribution(
                    GaussianDistribution(-2.0, 1.0), len_dist=CategoricalDistribution({1: 0.4, 2: 0.6})
                ),
                SequenceDistribution(
                    GaussianDistribution(2.0, 1.0), len_dist=CategoricalDistribution({2: 0.6, 3: 0.4})
                ),
            ],
            [0.5, 0.5],
            [[0.6, 0.4], [0.3, 0.7]],
            len_dist=CategoricalDistribution({2: 1.0}),
        )
        data = model.sampler(5).sample(80)
        est = SegmentalHiddenMarkovEstimator(
            [
                SequenceEstimator(
                    GaussianDistribution(-1.0, 2.0).estimator(),
                    len_estimator=CategoricalDistribution({1: 0.5, 2: 0.5}).estimator(),
                ),
                SequenceEstimator(
                    GaussianDistribution(1.0, 2.0).estimator(),
                    len_estimator=CategoricalDistribution({2: 0.5, 3: 0.5}).estimator(),
                ),
            ],
            len_estimator=CategoricalDistribution({2: 1.0}).estimator(),
            pseudo_count=(1.0, 1.0),
        )
        enc = seq_encode(data, model=model)
        fitted = seq_estimate(enc, est, model)
        self.assertIsInstance(fitted, SegmentalHiddenMarkovModelDistribution)
        self.assertTrue(np.all(np.isfinite(fitted.seq_log_density(enc[0][1]))))


if __name__ == "__main__":
    unittest.main()
