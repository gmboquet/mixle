import io
import os
import sys
import unittest

import numpy as np

from pysp.planner import SparkEncodedData, encoded_data, is_encoded_data_handle
from pysp.stats import GaussianDistribution, GaussianEstimator, seq_encode, seq_estimate, seq_log_density_sum
from pysp.utils.estimation import StreamingEstimator, constant, optimize, streaming_accumulate


def _spark_context():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    try:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("pysparkplug-spark-encoded-data-test")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
    except Exception:
        # no usable JVM (e.g. Java not installed); treat as unavailable
        return None
    return spark.sparkContext


class SparkEncodedDataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = _spark_context()
        if cls.sc is None:
            raise unittest.SkipTest("pyspark or a usable JVM is not available")
        cls.sc.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "sc", None) is not None:
            cls.sc.stop()

    def test_spark_handle_matches_local_scoring_and_estimate(self):
        data = list(np.linspace(-2.0, 2.0, 40))
        rdd = self.sc.parallelize(data, 2)
        model = GaussianDistribution(0.25, 1.5)
        estimator = GaussianEstimator()
        enc_local = seq_encode(data, model=model)

        with SparkEncodedData(rdd, model=model, estimator=estimator) as enc:
            self.assertTrue(is_encoded_data_handle(enc))
            self.assertEqual(len(enc), len(data))
            cnt_h, ll_h = seq_log_density_sum(enc, model)
            fitted_h = seq_estimate(enc, estimator, model)

        cnt_l, ll_l = seq_log_density_sum(enc_local, model)
        fitted_l = seq_estimate(enc_local, estimator, model)

        self.assertEqual(cnt_h, cnt_l)
        self.assertAlmostEqual(ll_h, ll_l, places=10)
        self.assertAlmostEqual(fitted_h.mu, fitted_l.mu, places=10)
        self.assertAlmostEqual(fitted_h.sigma2, fitted_l.sigma2, places=10)

    def test_factory_and_optimize_can_use_spark_backend(self):
        data = list(np.linspace(-2.0, 2.0, 60))
        rdd = self.sc.parallelize(data, 3)
        start = GaussianDistribution(1.0, 4.0)
        estimator = GaussianEstimator()

        with encoded_data(rdd, model=start, estimator=estimator, backend="spark") as enc:
            self.assertIsInstance(enc, SparkEncodedData)

        fitted = optimize(
            rdd, estimator, prev_estimate=start, backend="spark", max_its=2, delta=None, out=io.StringIO()
        )
        local = optimize(data, estimator, prev_estimate=start, max_its=2, delta=None, out=io.StringIO())

        self.assertAlmostEqual(fitted.mu, local.mu, places=10)
        self.assertAlmostEqual(fitted.sigma2, local.sigma2, places=10)

    def test_spark_handle_streaming_accumulate_matches_local(self):
        data = list(np.linspace(-1.0, 3.0, 25))
        rdd = self.sc.parallelize(data, 2)
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        enc_local = seq_encode(data, model=model)

        with SparkEncodedData(rdd, model=model, estimator=estimator) as enc:
            n_h, acc_h = streaming_accumulate(enc, estimator, model)
            stream = StreamingEstimator(estimator, schedule=constant(0.5), model=model)
            fitted_h = stream.update(enc_data=enc)

        n_l, acc_l = streaming_accumulate(enc_local, estimator, model)
        stream_l = StreamingEstimator(estimator, schedule=constant(0.5), model=model)
        fitted_l = stream_l.update(enc_data=enc_local)

        self.assertEqual(n_h, n_l)
        np.testing.assert_allclose(acc_h.value(), acc_l.value(), rtol=1.0e-12, atol=1.0e-12)
        self.assertAlmostEqual(fitted_h.mu, fitted_l.mu, places=10)
        self.assertAlmostEqual(fitted_h.sigma2, fitted_l.sigma2, places=10)


if __name__ == "__main__":
    unittest.main()
