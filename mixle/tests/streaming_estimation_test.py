import unittest

import numpy as np

from mixle.inference.estimation import constant, harmonic
from mixle.inference.streaming import IncrementalEstimator, StreamingEstimator
from mixle.stats import (
    BinomialDistribution,
    BinomialEstimator,
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    UniformDistribution,
    UniformEstimator,
)
from mixle.utils.parallel.planner import LocalEncodedData, Resources


def _assert_suff_close(test_case, actual, expected):
    if isinstance(actual, dict):
        test_case.assertEqual(set(actual.keys()), set(expected.keys()))
        for key in actual:
            _assert_suff_close(test_case, actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)):
        test_case.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_suff_close(test_case, a, e)
        return
    if actual is None or expected is None:
        test_case.assertEqual(actual, expected)
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1.0e-12, atol=1.0e-12
        )
        return
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1.0e-12, atol=1.0e-12
    )


def _batch_accumulator(estimator, model, data):
    enc = model.dist_to_encoder().seq_encode(data)
    acc = estimator.accumulator_factory().make()
    acc.seq_update(enc, np.ones(len(data)), model)
    return acc


def _assert_scaled_accumulator_matches(test_case, estimator, model, data, c=0.37):
    enc = model.dist_to_encoder().seq_encode(data)
    weights = np.linspace(0.5, 1.5, len(data))
    scaled = estimator.accumulator_factory().make()
    scaled.seq_update(enc, weights, model)
    test_case.assertIs(scaled.scale(c), scaled)

    expected = estimator.accumulator_factory().make()
    expected.seq_update(enc, weights * c, model)
    _assert_suff_close(test_case, scaled.value(), expected.value())


class StreamingEstimatorTestCase(unittest.TestCase):
    def test_schedule_helpers_validate_and_evaluate(self):
        self.assertEqual(constant(0.25)(10), 0.25)
        self.assertAlmostEqual(harmonic(0.75)(1), 1.0)
        self.assertLess(harmonic(0.75)(3), harmonic(0.75)(2))
        with self.assertRaises(ValueError):
            constant(0.0)
        with self.assertRaises(ValueError):
            harmonic(0.5)

    def test_streaming_update_matches_manual_decayed_accumulator(self):
        estimator = GaussianEstimator()
        start = GaussianDistribution(0.0, 1.0)
        stream = StreamingEstimator(estimator, schedule=constant(0.25), model=start)

        batch1 = np.asarray([-1.0, 0.0, 1.0])
        model1 = stream.update(batch1)
        expected = _batch_accumulator(estimator, start, batch1)
        _assert_suff_close(self, stream.value(), expected.value())

        batch2 = np.asarray([2.0, 3.0])
        stream.update(batch2)
        expected.scale(0.75)
        batch2_acc = _batch_accumulator(estimator, model1, batch2)
        batch2_acc.scale(0.25)
        expected.combine(batch2_acc.value())

        _assert_suff_close(self, stream.value(), expected.value())

    def test_dataset_size_scales_minibatch_statistics_to_a_stable_full_data_objective(self):
        estimator = GaussianEstimator()
        start = GaussianDistribution(0.0, 1.0)
        stream = StreamingEstimator(estimator, schedule=constant(0.5), model=start, dataset_size=100)

        stream.update(np.asarray([1.0, 1.0]))
        self.assertEqual(stream.nobs, 100.0)
        self.assertEqual(stream.last_batch_scale, 50.0)
        first = stream.value()
        self.assertEqual(first[2], 100.0)

        stream.update(np.asarray([3.0] * 10))
        self.assertEqual(stream.nobs, 100.0)
        self.assertEqual(stream.last_batch_scale, 10.0)
        self.assertEqual(stream.value()[2], 100.0)

    def test_streaming_update_accepts_local_encoded_data_handle(self):
        estimator = GaussianEstimator()
        start = GaussianDistribution(0.0, 1.0)
        stream = StreamingEstimator(estimator, schedule=constant(0.25), model=start)

        batch1 = list(np.asarray([-1.0, 0.0, 1.0]))
        with LocalEncodedData(batch1, model=start, estimator=estimator, resources=Resources.local(num_cpus=2)) as enc:
            model1 = stream.update(enc_data=enc)
        expected = _batch_accumulator(estimator, start, batch1)
        _assert_suff_close(self, stream.value(), expected.value())

        batch2 = list(np.asarray([2.0, 3.0]))
        with LocalEncodedData(
            batch2, model=model1, estimator=estimator, resources=Resources.local(num_cpus=2), sub_chunks=2
        ) as enc:
            stream.update(enc_data=enc)
        expected.scale(0.75)
        batch2_acc = _batch_accumulator(estimator, model1, batch2)
        batch2_acc.scale(0.25)
        expected.combine(batch2_acc.value())

        _assert_suff_close(self, stream.value(), expected.value())

    def test_streaming_composite_preserves_nested_support_bounds(self):
        dist = CompositeDistribution((UniformDistribution(0.0, 4.0), BinomialDistribution(0.5, 5)))
        estimator = CompositeEstimator((UniformEstimator(), BinomialEstimator()))
        stream = StreamingEstimator(estimator, schedule=constant(0.4), model=dist)

        batch1 = [(0.5, 0), (3.5, 5)]
        model1 = stream.update(batch1)
        expected = _batch_accumulator(estimator, dist, batch1)

        batch2 = [(1.0, 2), (2.0, 4)]
        stream.update(batch2)
        expected.scale(0.6)
        batch2_acc = _batch_accumulator(estimator, model1, batch2)
        batch2_acc.scale(0.4)
        expected.combine(batch2_acc.value())

        _assert_suff_close(self, stream.value(), expected.value())
        uniform_stats, binomial_stats = stream.value()
        self.assertEqual(uniform_stats[1:], (0.5, 3.5))
        self.assertEqual(binomial_stats[2:], (0, 5))

    def test_hmm_family_scaling_preserves_metadata(self):
        from mixle.stats.combinator.sequence import SequenceDistribution, SequenceEstimator
        from mixle.stats.latent.lookback_hidden_markov_model import (
            LookbackHiddenMarkovModelDistribution,
            LookbackHiddenMarkovModelEstimator,
        )
        from mixle.stats.latent.semi_supervised_hidden_markov_model import (
            SemiSupervisedHiddenMarkovEstimator,
            SemiSupervisedHiddenMarkovModelDistribution,
        )
        from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovModelDistribution
        from mixle.stats.sequences.integer_markov_chain import (
            IntegerMarkovChainDistribution,
            IntegerMarkovChainEstimator,
        )
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
        from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution

        init_dist = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]), CategoricalDistribution({1: 1.0})
        )
        lookback_topics = [
            IntegerMarkovChainDistribution(3, [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
            IntegerMarkovChainDistribution(3, [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]),
        ]
        lookback_est = [
            IntegerMarkovChainEstimator(3, lag=1, pseudo_count=0.1),
            IntegerMarkovChainEstimator(3, lag=1, pseudo_count=0.1),
        ]
        init_est = SequenceEstimator(
            IntegerCategoricalDistribution(0, [1.0, 0.0, 0.0]).estimator(pseudo_count=0.1),
            len_estimator=CategoricalEstimator(pseudo_count=0.1),
        )
        lookback_data = [[0, 1, 1, 2, 2, 0], [2, 2, 1, 1, 0], [0, 0, 1, 2, 1, 0]]
        for dist_cls, est_cls in ((LookbackHiddenMarkovModelDistribution, LookbackHiddenMarkovModelEstimator),):
            with self.subTest(dist=dist_cls.__module__):
                dist = dist_cls(
                    lookback_topics,
                    w=[0.6, 0.4],
                    transitions=[[0.8, 0.2], [0.3, 0.7]],
                    lag=1,
                    init_dist=[init_dist, init_dist],
                    len_dist=CategoricalDistribution({5: 0.5, 6: 0.5}),
                )
                estimator = est_cls(
                    lookback_est,
                    lag=1,
                    init_estimators=[init_est, init_est],
                    len_estimator=CategoricalEstimator(pseudo_count=0.1),
                    pseudo_count=(1.0, 1.0),
                )
                _assert_scaled_accumulator_matches(self, estimator, dist, lookback_data)

        semi_sup = SemiSupervisedHiddenMarkovModelDistribution(
            [
                CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
                CategoricalDistribution({"a": 0.1, "b": 0.2, "c": 0.7}),
            ],
            [[0.8, 0.2], [0.3, 0.7]],
            len_dist=CategoricalDistribution({3: 0.5, 4: 0.5}),
        )
        semi_sup_est = SemiSupervisedHiddenMarkovEstimator(
            [CategoricalEstimator(), CategoricalEstimator()],
            len_estimator=CategoricalEstimator(),
            pseudo_count=1.0,
        )
        # observations are (emission_seq, per-position state prior); None prior leaves the states free
        semi_sup_data = [(["a", "b", "a"], None), (["c", "b", "c", "a"], None), (["a", "a", "b"], None)]
        _assert_scaled_accumulator_matches(self, semi_sup_est, semi_sup, semi_sup_data)

        tree = TreeHiddenMarkovModelDistribution(
            topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(10.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.7, 0.3], [0.3, 0.7]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.6]),
            terminal_level=4,
            use_numba=True,
        )
        tree_data = [
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9)],
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 0), 9.9)],
        ]
        _assert_scaled_accumulator_matches(self, tree.estimator(), tree, tree_data)

    def test_incremental_update_replaces_chunk_contribution(self):
        estimator = GaussianEstimator()
        start = GaussianDistribution(0.0, 1.0)
        inc = IncrementalEstimator(estimator, model=start)

        chunk_a = np.asarray([-2.0, -1.0, 0.0])
        chunk_b = np.asarray([1.0, 2.0])
        chunk_a_new = np.asarray([-3.0, -2.0])

        model_a = inc.update(chunk_a, chunk_id="a")
        expected = _batch_accumulator(estimator, start, chunk_a)
        _assert_suff_close(self, inc.value(), expected.value())

        inc.update(chunk_b, chunk_id="b")
        expected.combine(_batch_accumulator(estimator, model_a, chunk_b).value())
        _assert_suff_close(self, inc.value(), expected.value())

        inc.update(chunk_a_new, chunk_id="a")
        expected = _batch_accumulator(estimator, inc.model, chunk_b)
        expected.combine(_batch_accumulator(estimator, inc.model, chunk_a_new).value())
        _assert_suff_close(self, inc.value(), expected.value())
        self.assertEqual(set(inc.chunk_values.keys()), {"a", "b"})
        self.assertEqual(inc.nobs, float(len(chunk_a_new) + len(chunk_b)))

        pooled = np.concatenate([chunk_a_new, chunk_b])
        pooled_acc = _batch_accumulator(estimator, inc.model, pooled)
        pooled_model = estimator.estimate(float(len(pooled)), pooled_acc.value())
        self.assertAlmostEqual(inc.model.mu, pooled_model.mu, places=12)
        self.assertAlmostEqual(inc.model.sigma2, pooled_model.sigma2, places=12)

        with self.assertRaises(KeyError):
            inc.chunk_value("missing")
        with self.assertRaises(ValueError):
            inc.update(chunk_a, chunk_id=None)

    def test_incremental_replacement_rebuilds_noninvertible_support_statistics(self):
        """Replacing an extrema-owning chunk must remove its obsolete min/max."""
        inc = IncrementalEstimator(UniformEstimator())
        inc.update([0.0, 10.0], chunk_id="a")
        inc.update([2.0, 8.0], chunk_id="b")

        model = inc.update([3.0, 7.0], chunk_id="a")

        self.assertEqual((model.low, model.high), (2.0, 8.0))
        self.assertEqual(inc.value(), (4.0, 2.0, 8.0))

    def test_incremental_update_accepts_local_encoded_data_handle(self):
        estimator = GaussianEstimator()
        start = GaussianDistribution(0.0, 1.0)
        inc = IncrementalEstimator(estimator, model=start)

        chunk_a = [-2.0, -1.0, 0.0]
        chunk_b = [1.0, 2.0]
        with LocalEncodedData(chunk_a, model=start, estimator=estimator, resources=Resources.local(num_cpus=2)) as enc:
            model_a = inc.update(enc_data=enc, chunk_id="a")
        with LocalEncodedData(
            chunk_b, model=model_a, estimator=estimator, resources=Resources.local(num_cpus=2)
        ) as enc:
            inc.update(enc_data=enc, chunk_id="b")

        expected = _batch_accumulator(estimator, start, chunk_a)
        expected.combine(_batch_accumulator(estimator, model_a, chunk_b).value())
        _assert_suff_close(self, inc.value(), expected.value())


if __name__ == "__main__":
    unittest.main()
