"""Model-parallel executor (C3.P0): factor-parallel EM is bit-identical to the replicated path."""

import unittest

import numpy as np

import pysp.stats as stats
from pysp.inference import optimize
from pysp.utils.parallel.model_parallel import ModelParallelEncodedData
from pysp.utils.parallel.planner import available_encoded_data_backends, encoded_data


def _composite():
    est = stats.CompositeEstimator((stats.GaussianEstimator(), stats.PoissonEstimator(), stats.CategoricalEstimator()))
    init = stats.CompositeDistribution(
        (
            stats.GaussianDistribution(0.0, 1.0),
            stats.PoissonDistribution(1.0),
            stats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
        )
    )
    rng = np.random.RandomState(0)
    data = [(float(rng.randn()), int(rng.poisson(3)), "a" if rng.rand() < 0.5 else "b") for _ in range(400)]
    return est, init, data


class RegistrationTest(unittest.TestCase):
    def test_backend_is_registered(self):
        self.assertIn("model_parallel", available_encoded_data_backends())
        self.assertIsInstance(
            encoded_data(_composite()[2], model=_composite()[1], backend="model_parallel"), ModelParallelEncodedData
        )


class FactorParallelExactnessTest(unittest.TestCase):
    def test_estep_value_is_bit_identical(self):
        # the factor-parallel fold runs each child's seq_update with the IDENTICAL call as the serial
        # path, so the M-step output is exactly equal (not merely close).
        est, init, data = _composite()
        enc = init.dist_to_encoder().seq_encode(data)
        local = est.accumulator_factory().make()
        local.seq_update(enc, np.ones(len(data)), init)
        d = {}
        local.key_merge(d)
        local.key_replace(d)
        m_local = est.estimate(float(len(data)), local.value())

        mp = ModelParallelEncodedData(data, estimator=est, model=init, num_workers=3)
        m_mp = mp.pysp_seq_estimate(est, init)
        self.assertEqual(str(m_local), str(m_mp))

    def test_optimize_end_to_end_bit_identical(self):
        est, init, data = _composite()
        local = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))  # same init + bit-identical folds => identical EM trajectory

    def test_log_density_sum_matches(self):
        est, init, data = _composite()
        mp = ModelParallelEncodedData(data, estimator=est, model=init)
        n, ll = mp.pysp_seq_log_density_sum(init)
        self.assertEqual(n, float(len(data)))
        self.assertAlmostEqual(
            ll, float(np.sum(init.seq_log_density(init.dist_to_encoder().seq_encode(data)))), places=9
        )


class ComponentParallelTest(unittest.TestCase):
    """Mixtures are component-parallel: scoring + accumulation distributed, normalization central, exact."""

    def _mixture(self):
        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(4)])
        init = stats.MixtureDistribution(
            [stats.GaussianDistribution(float(i) - 1.5, 1.0) for i in range(4)], [0.25] * 4
        )
        rng = np.random.RandomState(1)
        data = [float(rng.randn() + 3 * (rng.randint(4) - 1.5)) for _ in range(400)]
        return est, init, data

    def test_component_estep_bit_identical(self):
        est, init, data = self._mixture()
        enc = init.dist_to_encoder().seq_encode(data)
        local = est.accumulator_factory().make()
        local.seq_update(enc, np.ones(len(data)), init)
        d = {}
        local.key_merge(d)
        local.key_replace(d)
        m_local = est.estimate(float(len(data)), local.value())
        mp = ModelParallelEncodedData(data, estimator=est, model=init, num_workers=3)
        m_mp = mp.pysp_seq_estimate(est, init)
        self.assertEqual(str(m_local), str(m_mp))

    def test_optimize_end_to_end_bit_identical(self):
        est, init, data = self._mixture()
        local = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))


class FallbackTest(unittest.TestCase):
    def test_leaf_model_falls_back_and_is_identical(self):
        # a plain Gaussian is atomic -> replicated accumulation, still exact via the same handle.
        est = stats.GaussianEstimator()
        init = stats.GaussianDistribution(0.0, 1.0)
        rng = np.random.RandomState(2)
        data = [float(rng.randn() * 2 + 1) for _ in range(200)]
        local = optimize(data, est, prev_estimate=init, max_its=5, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=5, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))


if __name__ == "__main__":
    unittest.main()
