"""Cost-aware multi-fidelity Bayesian optimization (mixle.doe.multifidelity)."""

import importlib.util
import unittest
import warnings

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class MultiFidelityTest(unittest.TestCase):
    def test_uses_both_fidelities_and_finds_target_optimum(self):
        from mixle.doe import multi_fidelity_minimize

        opt = np.array([0.3, -0.4, 0.1])

        def obj(x, s):
            base = float(np.sum((x - opt) ** 2))
            return base if s == 1.0 else base + 0.05 * np.sin(8 * x[0])  # cheap, slightly biased low fidelity

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = multi_fidelity_minimize(
                obj,
                [(-1.0, 1.0)] * 3,
                fidelities=(0.5, 1.0),
                costs=(1.0, 5.0),
                n_init=5,
                max_cost=80.0,
                n_candidates=200,
                seed=0,
            )
        n_low = int(np.sum(res["X"][:, -1] == 0.5))
        n_high = int(np.sum(res["X"][:, -1] == 1.0))
        self.assertGreater(n_low, 5)  # spent cheap evaluations exploring
        self.assertGreater(n_high, 5)  # and expensive ones refining
        self.assertLess(np.linalg.norm(res["x"] - opt), 0.25)  # reached the target optimum
        self.assertLessEqual(res["cost"], 90.0)


if __name__ == "__main__":
    unittest.main()
