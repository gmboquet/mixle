"""Edge contracts for sequence encoding and convenience scoring."""

import pickle
import unittest
from unittest.mock import patch

import numpy as np

from mixle.inference import estimate, initialize, seq_estimate, seq_initialize
from mixle.stats import GaussianDistribution, GaussianEstimator, log_density, seq_encode
from mixle.stats.compute import sequence as sequence_module
from mixle.stats.compute.pdist import DataSequenceEncoder
from mixle.stats.compute.sequence import _partition_random_states, seq_log_density, seq_log_density_sum


class _DroppingEncoder(DataSequenceEncoder):
    def seq_encode(self, x):
        return np.asarray(list(x)[:-1])

    def __eq__(self, other):
        return isinstance(other, _DroppingEncoder)


class _FakeBroadcast:
    def __init__(self, value):
        self.value = value
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class _FakeSparkContext:
    def __init__(self):
        self.broadcasts = []

    def broadcast(self, value):
        broadcast = _FakeBroadcast(value)
        self.broadcasts.append(broadcast)
        return broadcast


class _FakeRDD:
    def __init__(self, partitions, context=None):
        self.partitions = [list(partition) for partition in partitions]
        self.context = context or _FakeSparkContext()
        self.checkpointed = False

    def _with(self, partitions):
        return type(self)(partitions, self.context)

    def getNumPartitions(self):
        return len(self.partitions)

    def glom(self):
        return self._with([[list(partition)] for partition in self.partitions])

    def map(self, function):
        return self._with([[function(value) for value in partition] for partition in self.partitions])

    def mapPartitions(self, function):
        return self._with([list(function(iter(partition))) for partition in self.partitions])

    def mapPartitionsWithIndex(self, function, _preserves_partitioning=False):
        return self._with(
            [list(function(index, iter(partition))) for index, partition in enumerate(self.partitions)]
        )

    def collect(self):
        return [value for partition in self.partitions for value in partition]

    def reduce(self, function):
        values = self.collect()
        result = values[0]
        for value in values[1:]:
            result = function(result, value)
        return result

    def treeReduce(self, function):
        return self.reduce(function)

    def localCheckpoint(self):
        self.checkpointed = True


class SequenceContractTest(unittest.TestCase):
    def setUp(self):
        self.model = GaussianDistribution(0.0, 1.0)

    def test_partition_controls_are_exact_positive_integers(self):
        for kwargs in (
            {"num_chunks": 0},
            {"num_chunks": -1},
            {"num_chunks": 1.5},
            {"num_chunks": True},
            {"chunk_size": 0},
            {"chunk_size": -1},
            {"chunk_size": 1.5},
            {"chunk_size": True},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    seq_encode([1.0, 2.0], model=self.model, **kwargs)
        with self.assertRaisesRegex(ValueError, "both given explicitly"):
            seq_encode([1.0, 2.0], model=self.model, num_chunks=2, chunk_size=1)

    def test_encoded_rows_must_conserve_input_rows(self):
        with self.assertRaisesRegex(ValueError, "encoded-row conservation failed"):
            seq_encode([1.0, 2.0, 3.0], encoder=_DroppingEncoder())

    def test_empty_log_density_is_a_defined_empty_vector(self):
        result = log_density([], self.model)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.float64)

    def test_valid_chunking_conserves_all_rows(self):
        chunks = seq_encode(np.arange(7.0), model=self.model, num_chunks=3)
        self.assertEqual([count for count, _ in chunks], [3, 2, 2])
        self.assertEqual(sum(count for count, _ in chunks), 7)

    def test_local_estimation_passes_the_validated_observation_count(self):
        class RecordingEstimator(GaussianEstimator):
            seen_nobs = None

            def estimate(self, nobs, suff_stat):
                self.seen_nobs = nobs
                return super().estimate(nobs, suff_stat)

        estimator = RecordingEstimator()
        encoded = seq_encode([1.0, 2.0, 3.0], model=self.model, num_chunks=2)
        seq_estimate(encoded, estimator, self.model)
        self.assertEqual(estimator.seen_nobs, 3)

    def test_estimation_rejects_false_chunk_metadata(self):
        encoded = self.model.dist_to_encoder().seq_encode([1.0, 2.0])
        for declared in (3, -1):
            with self.subTest(declared=declared):
                with self.assertRaises(ValueError):
                    seq_estimate([(declared, encoded)], GaussianEstimator(), self.model)
        for declared in (True, 2.0):
            with self.subTest(declared=declared):
                with self.assertRaises(TypeError):
                    seq_estimate([(declared, encoded)], GaussianEstimator(), self.model)

    def test_partition_seed_streams_are_deterministic_and_distinct(self):
        seeds = np.random.RandomState(7).randint(2**31, size=3)
        first = [_partition_random_states(seeds, index) for index in range(3)]
        replay = [_partition_random_states(seeds.copy(), index) for index in range(3)]

        first_draws = [(rng.randint(2**31), weights.randint(2**31)) for rng, weights in first]
        replay_draws = [(rng.randint(2**31), weights.randint(2**31)) for rng, weights in replay]
        self.assertEqual(first_draws, replay_draws)
        self.assertEqual(len(set(first_draws)), 3)

        with self.assertRaises(IndexError):
            _partition_random_states(seeds, 3)
        with self.assertRaises(TypeError):
            _partition_random_states(pickle.dumps(seeds), 0)

    def test_spark_drivers_release_broadcasts_without_mutating_input_lineage(self):
        raw = _FakeRDD([[1.0, 2.0], [3.0]])
        estimator = GaussianEstimator()
        with patch.object(sequence_module, "RDD_TYPES", (_FakeRDD,)):
            encoded = seq_encode(raw, model=self.model)
            self.assertEqual(raw.context.broadcasts, [])

            operations = (
                lambda: seq_log_density(encoded, self.model),
                lambda: seq_log_density_sum(encoded, self.model),
                lambda: seq_estimate(encoded, estimator, self.model),
                lambda: seq_initialize(encoded, estimator, np.random.RandomState(3), p=1.0),
                lambda: initialize(raw, estimator, np.random.RandomState(3), p=1.0),
                lambda: estimate(raw, estimator, self.model),
            )
            for operation in operations:
                before = len(raw.context.broadcasts)
                operation()
                owned = raw.context.broadcasts[before:]
                self.assertTrue(owned)
                self.assertTrue(all(broadcast.destroyed for broadcast in owned))

        self.assertFalse(raw.checkpointed)
        self.assertFalse(encoded.checkpointed)

    def test_broadcast_cleanup_does_not_mask_the_primary_failure(self):
        class BrokenBroadcast(_FakeBroadcast):
            def destroy(self):
                raise RuntimeError("cleanup failed")

        class BrokenContext(_FakeSparkContext):
            def broadcast(self, value):
                broadcast = BrokenBroadcast(value)
                self.broadcasts.append(broadcast)
                return broadcast

        class BrokenRDD(_FakeRDD):
            def mapPartitions(self, _function):
                return self

            def collect(self):
                raise ValueError("primary failure")

        raw = BrokenRDD([[1.0]], BrokenContext())
        with patch.object(sequence_module, "RDD_TYPES", (BrokenRDD,)):
            with self.assertRaisesRegex(ValueError, "primary failure"):
                seq_log_density(raw, self.model)


if __name__ == "__main__":
    unittest.main()
