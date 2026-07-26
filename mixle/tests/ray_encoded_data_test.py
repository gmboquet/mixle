"""Tests for the Ray encoded-data backend / distributed sufficient-statistic folding (WS-C2).

Ray is optional; the distributed tests skip when it is not installed. The backend is always
*registered* (the factory lazy-imports Ray only when used), which is checked unconditionally.
"""

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from numpy.random import RandomState

from mixle.stats import GaussianDistribution, GaussianEstimator
from mixle.utils.parallel.planner import available_encoded_data_backends, encoded_data
from mixle.utils.parallel.ray_data import RayEncodedData

HAS_RAY = importlib.util.find_spec("ray") is not None


class RayBackendRegistrationTest(unittest.TestCase):
    def test_backend_is_registered(self):
        self.assertIn("ray", available_encoded_data_backends())


class RayLifecycleOwnershipTest(unittest.TestCase):
    class _Encoder:
        def seq_encode(self, rows):
            return tuple(rows)

    def _fake_ray(self, *, initialized: bool):
        state = {"initialized": initialized, "init": 0, "shutdown": 0, "puts": []}

        def init(**kwargs):
            state["initialized"] = True
            state["init"] += 1

        def shutdown():
            state["initialized"] = False
            state["shutdown"] += 1

        def put(value):
            reference = SimpleNamespace(value=value)
            state["puts"].append(reference)
            return reference

        fake = SimpleNamespace(
            is_initialized=lambda: state["initialized"],
            init=init,
            shutdown=shutdown,
            put=put,
        )
        return fake, state

    def test_owned_runtime_is_released_idempotently(self):
        ray, state = self._fake_ray(initialized=False)
        with mock.patch.dict(sys.modules, {"ray": ray}):
            handle = RayEncodedData([1, 2, 3], encoder=self._Encoder(), num_partitions=2)
            self.assertEqual(state["init"], 1)
            self.assertEqual(handle.num_chunks, 2)
            handle.close()
            handle.close()
        self.assertEqual(state["shutdown"], 1)
        self.assertEqual(handle.num_chunks, 0)

    def test_shared_runtime_is_never_shutdown(self):
        ray, state = self._fake_ray(initialized=True)
        with mock.patch.dict(sys.modules, {"ray": ray}):
            handle = RayEncodedData([1, 2], encoder=self._Encoder(), num_partitions=1)
            handle.close()
        self.assertEqual(state["init"], 0)
        self.assertEqual(state["shutdown"], 0)


@unittest.skipUnless(HAS_RAY, "ray is not installed")
class RayEncodedDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray

        ray.init(
            ignore_reinit_error=True, include_dashboard=False, configure_logging=False, log_to_driver=False, num_cpus=2
        )

    @classmethod
    def tearDownClass(cls):
        import ray

        if ray.is_initialized():
            ray.shutdown()

    def setUp(self):
        self.data = list(RandomState(0).normal(3.0, 2.0, size=600))
        self.estimator = GaussianEstimator()

    def test_estimate_matches_local_backend(self):
        prev = GaussianDistribution(0.0, 1.0)
        local = encoded_data(self.data, estimator=self.estimator, backend="local")
        ray_h = encoded_data(self.data, estimator=self.estimator, backend="ray", num_chunks=4)
        m_local = local.pysp_seq_estimate(self.estimator, prev)
        m_ray = ray_h.pysp_seq_estimate(self.estimator, prev)
        self.assertAlmostEqual(m_local.mu, m_ray.mu, places=8)
        self.assertAlmostEqual(m_local.sigma2, m_ray.sigma2, places=8)

    def test_log_density_sum_matches_local(self):
        est = GaussianDistribution(3.0, 4.0)
        local = encoded_data(self.data, estimator=self.estimator, backend="local")
        ray_h = encoded_data(self.data, estimator=self.estimator, backend="ray", num_chunks=4)
        c_local, t_local = local.pysp_seq_log_density_sum(est)
        c_ray, t_ray = ray_h.pysp_seq_log_density_sum(est)
        self.assertEqual(c_local, c_ray)
        self.assertAlmostEqual(t_local, t_ray, places=6)


if __name__ == "__main__":
    unittest.main()
