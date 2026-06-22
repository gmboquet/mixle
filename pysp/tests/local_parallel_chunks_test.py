"""Tests for opt-in parallel chunk folding in LocalEncodedData (WS-C2 estimation speed).

The local encoded-data backend folds resident chunks serially. With ``parallel_chunks=True``
the per-chunk numpy accumulation/scoring runs on a thread pool, and because results are folded
in chunk order the outcome must be *bit-identical* to the serial path.
"""

import unittest

from pysp.stats.base.gaussian import GaussianDistribution, GaussianEstimator
from pysp.utils.parallel.planner import LocalEncodedData, encoded_data


class LocalParallelChunksTest(unittest.TestCase):
    def setUp(self):
        dist = GaussianDistribution(1.5, 4.0)
        self.data = list(dist.sampler(0).sample(400))
        self.encoder = dist.dist_to_encoder()
        self.estimator = GaussianEstimator()
        self.model = dist

    def _handle(self, parallel_chunks):
        # Force several chunks so threading actually has work to fan out.
        return encoded_data(
            self.data,
            estimator=self.estimator,
            encoder=self.encoder,
            backend="local",
            sub_chunks=8,
            parallel_chunks=parallel_chunks,
        )

    def test_multiple_chunks_present(self):
        handle = self._handle(False)
        self.assertIsInstance(handle, LocalEncodedData)
        self.assertGreater(handle.num_chunks, 1)  # otherwise the parallel path is never exercised

    def test_seq_estimate_bit_identical(self):
        serial = self._handle(False).pysp_seq_estimate(self.estimator, self.model)
        parallel = self._handle(True).pysp_seq_estimate(self.estimator, self.model)
        # The fold is order-preserving, so the M-step output must match exactly.
        self.assertEqual(serial.mu, parallel.mu)
        self.assertEqual(serial.sigma2, parallel.sigma2)

    def test_log_density_sum_bit_identical(self):
        s_count, s_total = self._handle(False).pysp_seq_log_density_sum(self.model)
        p_count, p_total = self._handle(True).pysp_seq_log_density_sum(self.model)
        self.assertEqual(s_count, p_count)
        self.assertEqual(s_total, p_total)

    def test_stream_accumulate_bit_identical(self):
        s_nobs, s_val = self._handle(False).pysp_stream_accumulate(self.estimator, self.model)
        p_nobs, p_val = self._handle(True).pysp_stream_accumulate(self.estimator, self.model)
        self.assertEqual(s_nobs, p_nobs)
        # Resulting estimates from the tied statistics must agree exactly.
        s_est = self.estimator.estimate(s_nobs, s_val)
        p_est = self.estimator.estimate(p_nobs, p_val)
        self.assertEqual(s_est.mu, p_est.mu)
        self.assertEqual(s_est.sigma2, p_est.sigma2)

    def test_explicit_worker_count_is_honored(self):
        handle = encoded_data(
            self.data,
            estimator=self.estimator,
            encoder=self.encoder,
            backend="local",
            sub_chunks=8,
            parallel_chunks=True,
            chunk_workers=2,
        )
        self.assertEqual(handle._chunk_workers, 2)
        est = handle.pysp_seq_estimate(self.estimator, self.model)
        serial = self._handle(False).pysp_seq_estimate(self.estimator, self.model)
        self.assertEqual(est.mu, serial.mu)
        self.assertEqual(est.sigma2, serial.sigma2)


if __name__ == "__main__":
    unittest.main()
