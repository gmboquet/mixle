"""WS-11: Thompson-sampling acquisition for Bayesian optimization."""

import unittest

import numpy as np

from pysp.doe import available_acquisitions, propose_next, thompson_sampling


class ThompsonAcquisitionTest(unittest.TestCase):
    def test_registered_under_name_and_aliases(self):
        for name in ("thompson_sampling", "thompson", "ts"):
            self.assertIn(name, available_acquisitions())

    def test_merit_is_negated_draw_for_minimization(self):
        mean = np.array([0.0, 1.0, 2.0])
        std = np.array([1.0, 1.0, 1.0])
        # with std=0 the draw is exactly the mean, so min-merit == -mean and max-merit == +mean
        zmin = thompson_sampling(mean, np.zeros(3), 0.0, maximize=False, rng=np.random.RandomState(0))
        zmax = thompson_sampling(mean, np.zeros(3), 0.0, maximize=True, rng=np.random.RandomState(0))
        self.assertTrue(np.allclose(zmin, -mean))
        self.assertTrue(np.allclose(zmax, mean))
        # non-zero std injects randomness
        self.assertFalse(np.allclose(thompson_sampling(mean, std, 0.0, rng=np.random.RandomState(0)), -mean))

    def test_proposes_near_the_optimum(self):
        # minimize f(x) = (x - 0.7)^2 on [0, 1]; Thompson should steer toward the basin
        rng = np.random.RandomState(0)
        x = rng.rand(12, 1)
        y = (x[:, 0] - 0.7) ** 2
        props = [
            propose_next(x, y, bounds=[(0.0, 1.0)], acq="thompson",
                         acq_kwargs={"rng": np.random.RandomState(s)}, seed=s)[0]
            for s in range(8)
        ]
        # the median proposal lands in the low-objective region around 0.7
        self.assertLess(abs(float(np.median(props)) - 0.7), 0.3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in props))


if __name__ == "__main__":
    unittest.main()
