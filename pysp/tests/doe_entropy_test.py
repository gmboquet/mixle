"""Max-value Entropy Search information-theoretic BO (pysp.doe.entropy)."""

import importlib.util
import unittest
import warnings

import numpy as np

from pysp.doe.entropy import max_value_entropy_search, sample_max_values

HAS_TORCH = importlib.util.find_spec("torch") is not None


class MaxValueEntropyTest(unittest.TestCase):
    def test_y_star_samples_above_best_mean(self):
        mu = np.array([0.0, 0.5, 0.9, 1.0])
        sd = np.array([0.5, 0.5, 0.5, 0.01])
        ystar = sample_max_values(mu, sd, 500, seed=0)
        self.assertGreaterEqual(ystar.min(), mu.max() - 1e-9)  # the max is never below the best mean

    def test_information_is_nonnegative_and_favors_uncertainty(self):
        mu = np.array([0.0, 0.5, 0.9, 1.0])
        sd = np.array([0.5, 0.5, 0.5, 0.01])
        ystar = sample_max_values(mu, sd, 500, seed=0)
        mes = max_value_entropy_search(mu, sd, ystar, maximize=True)
        self.assertTrue(np.all(mes >= -1e-9))
        # an uncertain near-optimal candidate beats a near-certain one
        self.assertGreater(mes[1], mes[3])


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class MesDriverTest(unittest.TestCase):
    def test_bo_loop_converges(self):
        from pysp.doe import propose_mes

        def f(x):
            return -(np.sin(3 * x) + 0.3 * x**2)  # maximize

        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, (6, 1))
        y = f(x[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(8):
                xn = propose_mes(x, y, [(-3.0, 3.0)], n_candidates=200, max_samples=48, maximize=True, seed=i)
                x = np.vstack([x, xn])
                y = np.append(y, f(xn[0]))
        true_max = f(np.linspace(-3, 3, 4000)).max()
        self.assertGreater(y.max(), true_max - 0.1)  # reaches near the global optimum


if __name__ == "__main__":
    unittest.main()
