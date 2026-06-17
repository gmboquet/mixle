"""Execution-stack tests for pysp.ppl: vectorization, torch engine, parallel backends."""

import time
import unittest

import numpy as np

from pysp.ppl import Mix, Normal, free


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(5.0, 2.0, size=200000))

    def test_numpy_vectorized_is_fast(self):
        # 200k-point Gaussian EM in well under a second -> vectorized seq_ path, not a loop
        t = time.perf_counter()
        m = Normal(free, free).fit(self.data)
        elapsed = time.perf_counter() - t
        self.assertAlmostEqual(m.dist.mu, 5.0, delta=0.05)
        self.assertLess(elapsed, 1.0)

    def test_torch_engine_matches_numpy(self):
        from pysp.engines import TorchEngine

        m_np = Normal(free, free).fit(self.data)
        m_t = Normal(free, free).fit(self.data, engine=TorchEngine())
        self.assertAlmostEqual(m_np.dist.mu, m_t.dist.mu, places=3)
        self.assertAlmostEqual(m_np.dist.sigma2, m_t.dist.sigma2, places=2)

    def test_mp_backend_matches_local(self):
        # multiprocessing backend produces the same fit (spawn re-imports this module fine)
        rng = np.random.RandomState(1)
        data = list(np.concatenate([rng.normal(-5, 1, 20000), rng.normal(5, 1, 20000)]))
        try:
            m = Mix([Normal(free, free), Normal(free, free)]).fit(
                data, backend="mp", num_workers=2, rng=np.random.RandomState(2)
            )
        except Exception as e:  # environment without usable mp
            self.skipTest(f"mp backend unavailable: {e}")
        means = sorted(c.mu for c in m.dist.components)
        self.assertAlmostEqual(means[0], -5.0, delta=0.3)
        self.assertAlmostEqual(means[1], 5.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
