import copy
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


def _assert_suff_equal(test_case, actual, expected):
    """Bitwise structural equality between payload trees (no tolerance)."""
    if isinstance(actual, dict):
        test_case.assertEqual(set(actual.keys()), set(expected.keys()))
        for key in actual:
            _assert_suff_equal(test_case, actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)):
        test_case.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_suff_equal(test_case, a, e)
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        test_case.assertTrue(np.array_equal(np.asarray(actual), np.asarray(expected)))
        return
    test_case.assertEqual(actual, expected)


class IncrementalRebaseTestCase(unittest.TestCase):
    """rebase() and its triggers: the exactness escape hatch for the subtract fast path."""

    @staticmethod
    def _canonical_reduce(inc):
        """From-scratch reduction of the stored payloads in sorted-chunk_id order."""
        ids = sorted(inc.chunk_values)
        acc = inc.estimator.accumulator_factory().make()
        acc.from_value(copy.deepcopy(inc.chunk_values[ids[0]]))
        for cid in ids[1:]:
            acc.combine(copy.deepcopy(inc.chunk_values[cid]))
        nobs = 0.0
        for cid in ids:
            nobs += inc.nobs_by_chunk[cid]
        return nobs, acc

    @staticmethod
    def _adversarial_chunks():
        """One chunk at magnitude 1e9 among unit-scale chunks, all deterministic.

        The big chunk's sum-of-squares payload (~1e21) absorbs the small chunks'
        (~1e3) entirely at add time -- float64 spacing at 1e21 is ~1.3e5 -- so
        revisiting the big chunk reveals the loss: the pooled second moment of
        what remains goes negative.
        """
        jitter = (np.arange(1000) % 7 - 3.0) / 10.0
        return {
            "big": 1.0e9 + jitter,
            "b": 1.0 + jitter,
            "c": 1.0 - jitter,
            "replacement": 1.0 + jitter / 2.0,
        }

    def _run_adversarial_revisit(self, inc):
        chunks = self._adversarial_chunks()
        inc.update(chunks["big"], chunk_id="a")
        inc.update(chunks["b"], chunk_id="b")
        inc.update(chunks["c"], chunk_id="c")
        return inc.update(chunks["replacement"], chunk_id="a")

    def test_revisit_cancellation_corrupts_stats_and_rebase_repairs(self):
        inc = IncrementalEstimator(GaussianEstimator(), model=GaussianDistribution(0.0, 1.0))
        subtract_model = self._run_adversarial_revisit(inc)

        # the pathology, at the statistic level: the pooled second moment implied by
        # the running (sum_wx, sum_wxx, sum_w) payload is NEGATIVE after the revisit
        sum_x, sum_xx, sum_w, _ = inc.value()
        self.assertLess(sum_xx / sum_w - (sum_x / sum_w) ** 2, 0.0)

        nobs, expected_acc = self._canonical_reduce(inc)
        expected_model = inc.estimator.estimate(nobs, expected_acc.value())
        # at the model level the variance floor caught the negative moment, collapsing
        # sigma2 orders of magnitude below the true pooled variance (~0.03)
        self.assertLess(subtract_model.sigma2, expected_model.sigma2 / 100.0)

        rebased = inc.rebase()
        _assert_suff_equal(self, inc.value(), expected_acc.value())
        self.assertEqual(inc.nobs, nobs)
        self.assertEqual(rebased.mu, expected_model.mu)
        self.assertEqual(rebased.sigma2, expected_model.sigma2)
        self.assertGreater(rebased.sigma2, 0.01)

    def test_cancellation_guard_auto_rebases(self):
        inc = IncrementalEstimator(
            GaussianEstimator(),
            model=GaussianDistribution(0.0, 1.0),
            cancellation_bits=20.0,
        )
        guarded_model = self._run_adversarial_revisit(inc)

        self.assertEqual(inc.cancellation_rebases, 1)
        nobs, expected_acc = self._canonical_reduce(inc)
        expected_model = inc.estimator.estimate(nobs, expected_acc.value())
        _assert_suff_equal(self, inc.value(), expected_acc.value())
        self.assertEqual(guarded_model.sigma2, expected_model.sigma2)
        self.assertGreater(guarded_model.sigma2, 0.01)

    def test_benign_revisit_does_not_trigger_guard(self):
        inc = IncrementalEstimator(
            GaussianEstimator(),
            model=GaussianDistribution(0.0, 1.0),
            cancellation_bits=20.0,
        )
        inc.update(np.asarray([-2.0, -1.0, 0.0]), chunk_id="a")
        inc.update(np.asarray([1.0, 2.0]), chunk_id="b")
        inc.update(np.asarray([-3.0, -2.0]), chunk_id="a")
        self.assertEqual(inc.cancellation_rebases, 0)

    def test_rebase_every_matches_canonical_reduce(self):
        inc = IncrementalEstimator(
            GaussianEstimator(),
            model=GaussianDistribution(0.0, 1.0),
            rebase_every=2,
        )
        inc.update(np.asarray([-2.0, -1.0, 0.0]), chunk_id="a")
        inc.update(np.asarray([1.0, 2.0]), chunk_id="b")
        inc.update(np.asarray([-3.0, -2.0]), chunk_id="a")
        inc.update(np.asarray([0.5, 1.5]), chunk_id="c")

        nobs, expected_acc = self._canonical_reduce(inc)
        _assert_suff_equal(self, inc.value(), expected_acc.value())
        self.assertEqual(inc.nobs, nobs)

    def test_constructor_validation_and_noop_rebase(self):
        with self.assertRaises(ValueError):
            IncrementalEstimator(GaussianEstimator(), rebase_every=0)
        with self.assertRaises(ValueError):
            IncrementalEstimator(GaussianEstimator(), cancellation_bits=0.0)

        start = GaussianDistribution(0.0, 1.0)
        inc = IncrementalEstimator(GaussianEstimator(), model=start)
        self.assertIs(inc.rebase(), start)
        self.assertIsNone(inc.running_accumulator)


if __name__ == "__main__":
    unittest.main()
