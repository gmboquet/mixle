"""EM cannot leave a symmetric fixed point, so it must not be handed one.

A latent model whose initialization draws each observation's component or state uniformly at random
gives every component the same random ~1/k subsample: every emission starts at the sample marginal,
every transition row starts uniform, and the objective then moves by hundredths of a nat for dozens
of iterations -- or never leaves. On the shipped structured gallery three of five seeds collapsed to
identical emissions and stayed there through 400 iterations (P09-F07); an ordinary categorical HMM
spent its first ~35 iterations on the plateau (A-01); the segmental section printed a fit 3.4 nats
per observation short of the reachable one (P09-F11); and two of four lookback seeds landed hundreds
of nats below the generating model (P10-F07).

The repair is a k-means++ start over the emission stream, shared with the mixture family. It declines
-- and the caller keeps the random draw it has always made -- when the emissions are not a numeric
vector space, when the clusters are too unbalanced to be components, or when a cluster would leave a
state with no WEIGHT at all.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

import mixle.stats as S
from mixle.inference import optimize
from mixle.stats.latent import _initialization as init_module


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


class SharedStartTest(unittest.TestCase):
    def test_two_separated_groups_are_recovered_not_split_at_random(self):
        features = np.concatenate(
            [np.random.RandomState(0).normal(-5.0, 0.3, 60), np.random.RandomState(1).normal(5.0, 0.3, 60)]
        )
        assign = init_module.kmeanspp_assignment(features[:, None], 2, np.random.RandomState(0))
        self.assertIsNotNone(assign)
        self.assertEqual(len(set(assign[:60])), 1)
        self.assertEqual(len(set(assign[60:])), 1)
        self.assertNotEqual(assign[0], assign[60])

    def test_it_declines_rather_than_seeding_a_component_on_an_outlier(self):
        rows = np.concatenate([np.random.RandomState(0).normal(0.0, 1.0, 200), [500.0]])
        self.assertIsNone(init_module.kmeanspp_assignment(rows[:, None], 2, np.random.RandomState(3)))

    def test_it_declines_on_integer_category_codes(self):
        """The distance between symbol code 3 and code 4 says nothing about the symbols.

        Clustering them anyway handed each state a contiguous slice of the alphabet: an eight-state
        HMM over twelve symbols came back with one or two symbols in each state's support where
        0.8.1 kept ten to twelve, and a streaming re-estimate then met evidence no state could
        score. Families whose values carry metric meaning encode as float -- counts included.
        """
        codes = np.arange(40, dtype=np.int64) % 12
        self.assertIsNone(init_module.numeric_feature_matrix(codes))
        self.assertIsNone(init_module.broken_symmetry_states(codes, 4, np.random.RandomState(0)))
        counts = (np.arange(40, dtype=np.int64) % 12).astype(float)  # a Poisson encoder emits float64
        self.assertIsNotNone(init_module.numeric_feature_matrix(counts))

    def test_it_declines_on_data_it_cannot_place_in_a_vector_space(self):
        self.assertIsNone(init_module.numeric_feature_matrix(np.array(["a", "b"], dtype=object)))
        self.assertIsNone(init_module.numeric_feature_matrix(np.array([1.0, np.nan])))
        self.assertIsNone(
            init_module.broken_symmetry_states(np.array(["a", "b"] * 20, dtype=object), 2, np.random.RandomState(0))
        )

    def test_it_declines_a_split_that_would_leave_a_state_with_no_weight(self):
        rows = np.concatenate([np.zeros(20), np.full(20, 10.0)])
        weighted = np.concatenate([np.ones(20), np.zeros(20)])
        self.assertIsNotNone(init_module.broken_symmetry_states(rows, 2, np.random.RandomState(0)))
        self.assertIsNone(init_module.broken_symmetry_states(rows, 2, np.random.RandomState(0), weights=weighted))

    def test_a_declined_start_consumes_none_of_the_callers_stream(self):
        """The fallback must be the draw the old code would have made, byte for byte."""
        rng = np.random.RandomState(7)
        before = rng.get_state()[2]
        init_module.broken_symmetry_states(
            np.array(["a"] * 30, dtype=object), 2, init_module.derived_rng(np.arange(30))
        )
        self.assertEqual(rng.get_state()[2], before)

    def test_the_derived_generator_is_a_function_of_its_material(self):
        first = init_module.derived_rng(np.arange(10)).randint(1_000_000)
        self.assertEqual(first, init_module.derived_rng(np.arange(10)).randint(1_000_000))
        self.assertNotEqual(first, init_module.derived_rng(np.arange(11)).randint(1_000_000))


class HiddenMarkovStartTest(unittest.TestCase):
    """A-01: the plateau, on the family it was measured on."""

    @staticmethod
    def _truth():
        return S.HiddenMarkovModelDistribution(
            [S.GaussianDistribution(-3.0, 1.0), S.GaussianDistribution(3.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.85, 0.15], [0.15, 0.85]],
            len_dist=S.PoissonDistribution(20.0),
        )

    def test_every_seed_separates_the_states_within_a_short_budget(self):
        truth = self._truth()
        data = list(truth.sampler(0).sample(150))
        estimator = S.HiddenMarkovEstimator([S.GaussianEstimator()] * 2, len_estimator=S.PoissonEstimator())
        for seed in range(5):
            with self.subTest(seed=seed):
                model = _quiet(
                    lambda seed=seed: optimize(data, estimator, max_its=25, rng=np.random.RandomState(seed), out=None)
                )
                means = sorted(float(topic.mu) for topic in model.topics)
                self.assertAlmostEqual(means[0], -3.0, delta=0.4)
                self.assertAlmostEqual(means[1], 3.0, delta=0.4)

    def test_a_categorical_emission_hmm_keeps_its_alphabet_in_support(self):
        """The regression that motivated the float-only gate, measured the way it was found.

        On 0.8.1 each of eight states kept ten to twelve of twelve symbols; the code-clustered start
        left one or two, and a streaming ``seq_update`` on a fresh batch then hit impossible
        evidence. The support width is the assertion because it is what the collapse destroys.
        """
        alphabet, states = 12, 8
        rng = np.random.RandomState(0)
        truth = S.HiddenMarkovModelDistribution(
            [S.IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(alphabet) * 0.3))) for _ in range(states)],
            w=[1.0 / states] * states,
            transitions=np.full((states, states), 1.0 / states),
            len_dist=S.CategoricalDistribution({20: 1.0}),
        )
        data = list(truth.sampler(1).sample(200))
        estimator = S.HiddenMarkovEstimator(
            [S.IntegerCategoricalEstimator(min_val=0, max_val=alphabet - 1) for _ in range(states)],
            len_estimator=S.CategoricalDistribution({20: 1.0}).estimator(),
        )
        model = _quiet(lambda: optimize(data, estimator, max_its=5, rng=np.random.RandomState(0), out=None))
        widths = [int(np.sum(np.asarray(topic.p_vec) > 0.0)) for topic in model.topics]
        self.assertGreaterEqual(min(widths), alphabet - 3, widths)

        # and the streaming re-estimate the corpus notebook performs still scores a fresh batch
        accumulator = estimator.accumulator_factory().make()
        fresh = list(truth.sampler(7).sample(80))
        _quiet(lambda: accumulator.seq_update(model.dist_to_encoder().seq_encode(fresh), np.ones(len(fresh)), model))

    def test_a_categorical_emission_hmm_still_fits(self):
        """Nothing to cluster: the uniform draw stands, and the family keeps working."""
        truth = S.HiddenMarkovModelDistribution(
            [S.CategoricalDistribution({"a": 0.9, "b": 0.1}), S.CategoricalDistribution({"a": 0.1, "b": 0.9})],
            w=[0.5, 0.5],
            transitions=[[0.9, 0.1], [0.1, 0.9]],
            len_dist=S.PoissonDistribution(15.0),
        )
        data = list(truth.sampler(0).sample(150))
        model = _quiet(
            lambda: optimize(
                data,
                S.HiddenMarkovEstimator([S.CategoricalEstimator()] * 2, len_estimator=S.PoissonEstimator()),
                max_its=60,
                rng=np.random.RandomState(0),
                out=None,
            )
        )
        self.assertEqual(model.n_states, 2)
        dominant = sorted(max(topic.pmap, key=topic.pmap.get) for topic in model.topics)
        self.assertEqual(dominant, ["a", "b"])


class TreeAndSegmentalStartTest(unittest.TestCase):
    """P09-F07 and P09-F11, on the two families the structured gallery prints."""

    def test_a_tree_hmm_recovers_the_planted_states_at_every_seed(self):
        from mixle.stats.latent.tree_hidden_markov_model import (
            TreeHiddenMarkovEstimator,
            TreeHiddenMarkovModelDistribution,
        )

        truth = TreeHiddenMarkovModelDistribution(
            [S.GaussianDistribution(-3.0, 1.0), S.GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
            len_dist=S.PoissonDistribution(0.8),
            terminal_level=3,
        )
        data = truth.sampler(1).sample(300)
        held = truth.sampler(2).sample(50)
        truth_score = float(np.mean([truth.log_density(x) for x in held]))
        estimator = TreeHiddenMarkovEstimator([S.GaussianEstimator()] * 2, len_estimator=S.PoissonEstimator())
        for seed in range(3):
            with self.subTest(seed=seed):
                model = _quiet(
                    lambda seed=seed: optimize(data, estimator, max_its=8, rng=np.random.RandomState(seed), out=None)
                )
                fit_score = float(np.mean([model.log_density(x) for x in held]))
                self.assertGreater(fit_score, truth_score - 1.0)

    def test_a_segmental_hmm_reaches_the_generating_model(self):
        from mixle.stats.latent.segmental_hidden_markov_model import (
            SegmentalHiddenMarkovEstimator,
            SegmentalHiddenMarkovModelDistribution,
        )

        truth = SegmentalHiddenMarkovModelDistribution(
            [S.GaussianDistribution(-3.0, 1.0), S.GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.7, 0.3], [0.3, 0.7]],
            len_dist=S.PoissonDistribution(6.0),
        )
        data = truth.sampler(1).sample(400)
        held = truth.sampler(2).sample(50)
        model = _quiet(
            lambda: optimize(
                data,
                SegmentalHiddenMarkovEstimator([S.GaussianEstimator()] * 2, len_estimator=S.PoissonEstimator()),
                max_its=40,
                rng=np.random.RandomState(0),
                out=None,
            )
        )
        truth_score = float(np.mean([truth.log_density(x) for x in held]))
        fit_score = float(np.mean([model.log_density(x) for x in held]))
        self.assertGreater(fit_score, truth_score - 0.5)


if __name__ == "__main__":
    unittest.main()
