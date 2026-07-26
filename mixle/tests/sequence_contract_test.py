"""Edge contracts for sequence encoding and convenience scoring."""

import unittest

import numpy as np

from mixle.inference import seq_estimate
from mixle.stats import GaussianDistribution, GaussianEstimator, log_density, seq_encode
from mixle.stats.compute.pdist import DataSequenceEncoder


class _DroppingEncoder(DataSequenceEncoder):
    def seq_encode(self, x):
        return np.asarray(list(x)[:-1])

    def __eq__(self, other):
        return isinstance(other, _DroppingEncoder)


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


if __name__ == "__main__":
    unittest.main()
