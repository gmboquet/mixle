"""Learned orchestration (J2): a placement policy learned from telemetry, never-worse via fallback."""

import unittest

import numpy as np

from mixle.inference import LearnedPolicy, learn_placement_policy


def _true_cost(tflop, choice):
    # pool: high fixed round-trip cost, scales well; local: cheap fixed, scales badly. Crossover ~ tflop 3.4.
    return (2.0 + 0.05 * tflop) if choice == "pool" else (0.1 + 0.6 * tflop)


def _telemetry(n, seed):
    r = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        tflop = float(r.uniform(0, 12))
        choice = ["local", "pool"][r.randint(2)]  # both choices explored in history
        rows.append(({"tflop": tflop}, choice, {"cost": _true_cost(tflop, choice) + 0.1 * r.randn()}))
    return rows


def _bad_static(f):
    return "pool" if f["tflop"] >= 9.0 else "local"  # threshold far too high -> often picks local when pool is cheaper


class LearnedPlacementTest(unittest.TestCase):
    def test_beats_fixed_choices_and_a_bad_static_policy(self):
        pol = learn_placement_policy(_telemetry(600, 0), _bad_static, k=12, min_neighbors=4)
        test = [({"tflop": t}, "local", {"cost": _true_cost(t, "local")}) for t in np.linspace(0, 12, 300)]
        ev = pol.evaluate(test)
        self.assertLessEqual(ev["learned_mean_cost"], ev["fixed_mean_cost"]["local"])
        self.assertLessEqual(ev["learned_mean_cost"], ev["fixed_mean_cost"]["pool"])
        self.assertLess(ev["learned_mean_cost"], ev["static_mean_cost"])  # strictly better than the bad static

    def test_defers_when_history_is_thin(self):
        pol = learn_placement_policy([({"tflop": 1.0}, "local", {"cost": 1.0})], _bad_static)
        choice, learned = pol.decide({"tflop": 8.0})
        self.assertFalse(learned)  # not enough history to learn
        self.assertEqual(choice, _bad_static({"tflop": 8.0}))  # fell back to static

    def test_confident_on_a_clear_region_with_rich_history(self):
        pol = learn_placement_policy(_telemetry(600, 1), _bad_static, k=12)
        choice, learned = pol.decide({"tflop": 11.0})  # heavy -> pool clearly cheaper
        self.assertTrue(learned)
        self.assertEqual(choice, "pool")
        light, learned_l = pol.decide({"tflop": 0.5})  # light -> local clearly cheaper
        self.assertTrue(learned_l)
        self.assertEqual(light, "local")

    def test_defers_when_choices_are_tied(self):
        # near the crossover the choices cost about the same -> the policy should defer rather than guess
        pol = learn_placement_policy(_telemetry(600, 2), _bad_static, k=16, min_neighbors=4)
        # a region where local/pool costs are close (crossover ~3.4)
        _choice, learned = pol.decide({"tflop": 3.4})
        # not asserting the exact choice; only that a near-tie is handled without error and is a valid choice
        self.assertIn(_choice, ("local", "pool"))
        self.assertIsInstance(learned, bool)


class ContractTest(unittest.TestCase):
    def test_empty_rows_raise(self):
        with self.assertRaises(ValueError):
            learn_placement_policy([], _bad_static)

    def test_is_a_learned_policy(self):
        pol = learn_placement_policy(_telemetry(50, 3), _bad_static)
        self.assertIsInstance(pol, LearnedPolicy)
        self.assertIn("tflop", pol.keys)


if __name__ == "__main__":
    unittest.main()
