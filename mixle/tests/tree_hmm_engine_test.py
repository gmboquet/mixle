"""Engine-resident E-step parity for the tree HMM (numpy + torch), pure (non-numba) encoding."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovEstimator, TreeHiddenMarkovModelDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class TreeHmmEngineTestCase(unittest.TestCase):
    def setUp(self):
        topics = [GaussianDistribution(mu=0.0, sigma2=1.0), GaussianDistribution(mu=10.0, sigma2=1.0)]
        w = np.array([0.6, 0.4])
        trans = np.array([[0.7, 0.3], [0.2, 0.8]])
        self.dist = TreeHiddenMarkovModelDistribution(
            topics=topics, w=w, transitions=trans, terminal_level=6, use_numba=False
        )
        self.trees = [
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9)],
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9), ((3, 2), 0.3)],
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 0), 9.9)],
            [((0, -1), 9.5), ((1, 0), 9.7), ((2, 0), 0.1), ((3, 1), 9.9), ((4, 1), 0.2)],
            [((0, -1), 0.4)],
        ]
        self.weights = np.array([1.0, 0.7, 1.3, 0.9, 1.1])
        self.est = TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], use_numba=False)
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.trees)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        # value(): (num_states, init_counts, state_counts, trans_counts, emission_ss, len_ss)
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "TreeHiddenMarkovKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertTrue(
                    np.allclose(np.asarray(hv[1]), np.asarray(value[1]), atol=1.0e-8), "%s init_counts differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[2]), np.asarray(value[2]), atol=1.0e-8), "%s state_counts differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[3]), np.asarray(value[3]), atol=1.0e-8), "%s trans_counts differ" % name
                )
                for ha, ea in zip(hv[4], value[4]):
                    self.assertTrue(
                        np.allclose(np.asarray(ha[1]), np.asarray(ea[1]), atol=1.0e-7),
                        "%s emission suff-stats differ" % name,
                    )


if __name__ == "__main__":
    unittest.main()
