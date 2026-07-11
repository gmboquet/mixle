"""Tests for the Random Dot Product Graph distribution (density, sampling, torch parity, ASE)."""

import unittest

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats import RandomDotProductGraphDistribution

_X = np.array([[0.7, 0.1], [0.6, 0.2], [0.1, 0.7], [0.2, 0.6], [0.5, 0.5], [0.3, 0.3]])
_MASK = np.triu(np.ones((6, 6), dtype=bool), 1)


class RandomDotProductGraphTestCase(unittest.TestCase):
    def test_edge_probabilities_are_dot_products(self):
        dist = RandomDotProductGraphDistribution(_X)
        expected = np.clip(_X @ _X.T, 0.0, 1.0)
        np.testing.assert_allclose(dist.probs[_MASK], expected[_MASK], atol=1.0e-9)

    def test_seq_matches_scalar(self):
        dist = RandomDotProductGraphDistribution(_X)
        graphs = dist.sampler(seed=2).sample(5)
        enc = dist.dist_to_encoder().seq_encode(graphs)
        np.testing.assert_allclose(dist.seq_log_density(enc), [dist.log_density(g) for g in graphs])

    def test_torch_backend_matches_numpy(self):
        try:
            import torch

            from mixle.engines import TorchEngine
        except Exception as exc:  # pragma: no cover - torch optional  # noqa: BLE001
            self.skipTest("torch unavailable: %s" % exc)
        dist = RandomDotProductGraphDistribution(_X)
        enc = dist.dist_to_encoder().seq_encode(dist.sampler(seed=2).sample(5))
        engine = TorchEngine(dtype=torch.float64)
        backend = np.asarray(engine.to_numpy(dist.backend_seq_log_density(enc, engine)))
        np.testing.assert_allclose(backend, dist.seq_log_density(enc), rtol=1.0e-9, atol=1.0e-9)

    def test_string_round_trip(self):
        dist = RandomDotProductGraphDistribution(_X, name="r", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_sampler_edge_frequencies_match_probabilities(self):
        dist = RandomDotProductGraphDistribution(_X)
        samples = dist.sampler(seed=0).sample(4000)
        mean_adj = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean_adj[_MASK], dist.probs[_MASK], atol=0.04)

    def test_ase_recovers_edge_probability_matrix(self):
        true = RandomDotProductGraphDistribution(_X)
        data = true.sampler(seed=3).sample(2000)
        fitted = fit(data, true.estimator(), max_its=1, rng=np.random.RandomState(0), print_iter=0)
        # latent positions are identifiable only up to rotation; the edge probabilities X X^T are not.
        np.testing.assert_allclose(fitted.probs[_MASK], true.probs[_MASK], atol=0.1)

    def test_invalid_positions_raise(self):
        with self.assertRaises(ValueError):
            RandomDotProductGraphDistribution(np.array([1.0, 2.0]))  # not 2-D


if __name__ == "__main__":
    unittest.main()
