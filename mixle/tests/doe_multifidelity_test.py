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
        self.assertTrue(res["target_evaluated"])  # a genuine target-fidelity result, not a fallback


# --------------------------------------------------------------------------- MXR-080-0181
# `target` need not have been a member of `fidelities`. Such a target was never evaluated (the
# fidelity-selection loop only ever picks from `fidelities`), and the final fallback silently returned
# the best observation at ANY fidelity, mislabeled as the target-fidelity result: with fidelities
# (0.2, 0.5) and target=1, the function returned the 0.2-fidelity value as if it answered the
# target(=1.0)-fidelity question. `target` must now be validated as a member of `fidelities` up front.
class TargetFidelityValidationTest(unittest.TestCase):
    def test_target_outside_fidelities_is_rejected(self):
        from mixle.doe import multi_fidelity_minimize

        def obj(x, s):
            return float(np.sum(x))

        with self.assertRaisesRegex(ValueError, "target fidelity"):
            multi_fidelity_minimize(
                obj, [(0.0, 1.0)], fidelities=(0.2, 0.5), target=1, n_init=1, max_cost=3.0, n_candidates=8, seed=0
            )

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_target_in_fidelities_still_works_and_is_genuinely_evaluated(self):
        from mixle.doe import multi_fidelity_minimize

        opt = np.array([0.3])

        def obj(x, s):
            base = float(np.sum((x - opt) ** 2))
            return base if s == 1.0 else base + 0.05

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = multi_fidelity_minimize(
                obj,
                [(-1.0, 1.0)],
                fidelities=(0.5, 1.0),
                target=1.0,
                n_init=3,
                max_cost=12.0,
                n_candidates=32,
                seed=0,
            )
        self.assertTrue(res["target_evaluated"])
        self.assertIsNotNone(res["x"])
        self.assertIsNotNone(res["y"])
        self.assertIn(1.0, res["X"][:, -1].tolist())  # the target fidelity was genuinely queried


if __name__ == "__main__":
    unittest.main()
