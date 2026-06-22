import io
import unittest

import numpy as np

from pysp.inference.estimation import constant, optimize
from pysp.planner import DaskEncodedData, encoded_data, is_encoded_data_handle
from pysp.stats import GaussianDistribution, GaussianEstimator, seq_encode, seq_estimate, seq_log_density_sum
from pysp.utils.streaming import StreamingEstimator, streaming_accumulate


class _Future:
    def __init__(self, value):
        self.value = value


class _SynchronousClient:
    def submit(self, fn, *args, **kwargs):
        kwargs.pop("pure", None)
        values = [arg.value if isinstance(arg, _Future) else arg for arg in args]
        return _Future(fn(*values, **kwargs))

    def gather(self, futures):
        if isinstance(futures, list):
            return [future.value if isinstance(future, _Future) else future for future in futures]
        return futures.value if isinstance(futures, _Future) else futures

    def cancel(self, futures, force=False):
        return None

    def scheduler_info(self):
        return {"workers": {"worker-0": {}, "worker-1": {}}}


def _dask_client():
    try:
        from distributed import Client
    except ImportError:
        return None
    return Client(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None)


class DaskEncodedDataSynchronousClientTestCase(unittest.TestCase):
    def test_dask_handle_protocol_with_synchronous_client(self):
        client = _SynchronousClient()
        data = list(np.linspace(-2.0, 2.0, 40))
        model = GaussianDistribution(0.25, 1.5)
        estimator = GaussianEstimator()
        enc_local = seq_encode(data, model=model)

        with DaskEncodedData(
            data, model=model, estimator=estimator, client=client, num_partitions=3, sub_chunks=2
        ) as enc:
            cnt_h, ll_h = seq_log_density_sum(enc, model)
            fitted_h = seq_estimate(enc, estimator, model)
            n_h, acc_h = streaming_accumulate(enc, estimator, model)

        cnt_l, ll_l = seq_log_density_sum(enc_local, model)
        fitted_l = seq_estimate(enc_local, estimator, model)
        n_l, acc_l = streaming_accumulate(enc_local, estimator, model)

        self.assertEqual(cnt_h, cnt_l)
        self.assertEqual(n_h, n_l)
        self.assertAlmostEqual(ll_h, ll_l, places=10)
        self.assertAlmostEqual(fitted_h.mu, fitted_l.mu, places=10)
        self.assertAlmostEqual(fitted_h.sigma2, fitted_l.sigma2, places=10)
        np.testing.assert_allclose(acc_h.value(), acc_l.value(), rtol=1.0e-12, atol=1.0e-12)

    def test_optimize_can_use_dask_backend_with_explicit_client(self):
        client = _SynchronousClient()
        data = list(np.linspace(-2.0, 2.0, 60))
        start = GaussianDistribution(1.0, 4.0)
        estimator = GaussianEstimator()

        fitted = optimize(
            data,
            estimator,
            prev_estimate=start,
            backend="dask",
            client=client,
            num_chunks=3,
            max_its=2,
            delta=None,
            out=io.StringIO(),
        )
        local = optimize(data, estimator, prev_estimate=start, max_its=2, delta=None, out=io.StringIO())

        self.assertAlmostEqual(fitted.mu, local.mu, places=10)
        self.assertAlmostEqual(fitted.sigma2, local.sigma2, places=10)


class DaskEncodedDataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _dask_client()
        if cls.client is None:
            raise unittest.SkipTest("dask.distributed is not installed")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "client", None) is not None:
            cls.client.close()

    def test_dask_handle_matches_local_scoring_and_estimate(self):
        data = list(np.linspace(-2.0, 2.0, 40))
        model = GaussianDistribution(0.25, 1.5)
        estimator = GaussianEstimator()
        enc_local = seq_encode(data, model=model)

        with DaskEncodedData(
            data, model=model, estimator=estimator, client=self.client, num_partitions=3, sub_chunks=2
        ) as enc:
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

    def test_factory_and_optimize_can_use_dask_backend(self):
        data = list(np.linspace(-2.0, 2.0, 60))
        start = GaussianDistribution(1.0, 4.0)
        estimator = GaussianEstimator()

        with encoded_data(
            data, model=start, estimator=estimator, backend="dask", client=self.client, num_chunks=3
        ) as enc:
            self.assertIsInstance(enc, DaskEncodedData)

        fitted = optimize(
            data,
            estimator,
            prev_estimate=start,
            backend="dask",
            client=self.client,
            num_chunks=3,
            max_its=2,
            delta=None,
            out=io.StringIO(),
        )
        local = optimize(data, estimator, prev_estimate=start, max_its=2, delta=None, out=io.StringIO())

        self.assertAlmostEqual(fitted.mu, local.mu, places=10)
        self.assertAlmostEqual(fitted.sigma2, local.sigma2, places=10)

    def test_dask_handle_streaming_accumulate_matches_local(self):
        data = list(np.linspace(-1.0, 3.0, 25))
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        enc_local = seq_encode(data, model=model)

        with DaskEncodedData(data, model=model, estimator=estimator, client=self.client, num_partitions=2) as enc:
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
