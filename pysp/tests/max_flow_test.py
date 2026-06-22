"""WS-1: max-flow / min-cut (Edmonds-Karp), checked vs scipy + the max-flow min-cut theorem."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from pysp.relations import max_flow, min_cut


class MaxFlowTest(unittest.TestCase):
    def test_documented_example(self):
        cap = np.array([[0, 3, 2, 0, 0], [0, 0, 1, 3, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 4], [0, 0, 0, 0, 0]])
        value, _ = max_flow(cap, 0, 4)
        self.assertEqual(value, 5.0)
        cut_cap, side, _ = min_cut(cap, 0, 4)
        self.assertEqual(cut_cap, 5.0)
        self.assertIn(0, side)
        self.assertNotIn(4, side)

    def test_matches_scipy_and_theorem_on_random_graphs(self):
        for seed in range(60):
            r = np.random.RandomState(seed)
            n = r.randint(4, 10)
            cap = r.randint(0, 7, size=(n, n))
            np.fill_diagonal(cap, 0)
            src, snk = 0, n - 1
            value, flow = max_flow(cap, src, snk)
            scipy_val = maximum_flow(csr_matrix(cap.astype(np.int64)), src, snk).flow_value
            cut_cap, _, _ = min_cut(cap, src, snk)
            with self.subTest(seed=seed):
                self.assertAlmostEqual(value, float(scipy_val), places=9)   # vs scipy
                self.assertAlmostEqual(cut_cap, value, places=9)            # max-flow min-cut theorem
                for k in range(n):                                         # flow conservation
                    if k not in (src, snk):
                        self.assertAlmostEqual(flow[k].sum(), flow[:, k].sum(), places=9)
                self.assertTrue(np.all(flow <= cap + 1e-9))                # capacity respected

    def test_disconnected_is_zero(self):
        cap = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])  # no path 0->2
        self.assertEqual(max_flow(cap, 0, 2)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
