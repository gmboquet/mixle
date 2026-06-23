"""WS-11: knowledge-gradient acquisition (Frazier 2009), checked vs Monte Carlo."""

import importlib.util
import unittest

import numpy as np

from pysp.doe import knowledge_gradient, propose_knowledge_gradient

HAS_TORCH = importlib.util.find_spec("torch") is not None  # propose_knowledge_gradient uses the torch GP surrogate


def _kg_mc(mean, cov, noise=1e-6, m=60000, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros(mean.size)
    best = mean.max()
    for x in range(mean.size):
        b = cov[:, x] / np.sqrt(cov[x, x] + noise)
        z = rng.standard_normal(m)
        out[x] = (mean[:, None] + b[:, None] * z[None, :]).max(0).mean() - best
    return out


class KnowledgeGradientTest(unittest.TestCase):
    def test_matches_monte_carlo(self):
        for seed in range(15):
            rng = np.random.RandomState(seed)
            n = rng.randint(4, 10)
            a = rng.randn(n, n)
            cov = a @ a.T / n + 0.1 * np.eye(n)
            mean = rng.randn(n)
            kg = knowledge_gradient(mean, cov)
            with self.subTest(seed=seed):
                self.assertTrue(np.all(kg >= -1e-9))  # KG is non-negative
                self.assertTrue(np.allclose(kg, _kg_mc(mean, cov), atol=0.02))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_propose_finds_optimum_region(self):
        # minimize (x-0.7)^2; KG should propose a point in [0,1]
        rng = np.random.RandomState(0)
        x = rng.rand(8, 1)
        y = (x[:, 0] - 0.7) ** 2
        nxt = propose_knowledge_gradient(x, y, [(0.0, 1.0)], seed=1)
        self.assertEqual(nxt.shape, (1,))
        self.assertTrue(0.0 <= nxt[0] <= 1.0)


if __name__ == "__main__":
    unittest.main()
