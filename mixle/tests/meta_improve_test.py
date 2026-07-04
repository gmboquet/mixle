"""J4 learned scheduling + J5 meta-improve loop: promote only on a never-worse holdout receipt."""

import unittest

import numpy as np

from mixle.inference.orchestration import learn_schedule_policy, meta_improve


def _sched_rows(n=60, seed=0):
    """Small jobs run faster locally; big jobs faster on the pool (crossover ~2.7)."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        size = float(rng.uniform(0, 10))
        for choice in ["run_local", "queue_pool"]:
            lat = size * 2.0 + 1.0 if choice == "run_local" else size * 0.5 + 5.0
            rows.append(({"size": size}, choice, {"latency": lat + 0.1 * rng.randn()}))
    return rows


def _always_local(f):
    return "run_local"


class SchedulePolicyTest(unittest.TestCase):
    def test_learns_the_size_crossover(self):
        pol = learn_schedule_policy(_sched_rows(), _always_local)
        self.assertEqual(pol.decide({"size": 1.0})[0], "run_local")  # small stays local
        self.assertEqual(pol.decide({"size": 9.0})[0], "queue_pool")  # big goes to the pool

    def test_thin_history_defers_to_the_static_scheduler(self):
        pol = learn_schedule_policy(_sched_rows(n=2), _always_local, min_neighbors=8)
        choice, learned = pol.decide({"size": 9.0})
        self.assertEqual(choice, "run_local")  # deferred
        self.assertFalse(learned)


class MetaImproveTest(unittest.TestCase):
    def test_promotes_over_a_bad_static_with_a_receipt(self):
        out = meta_improve(_sched_rows(), _always_local, cost_key="latency", seed=1)
        self.assertTrue(out["promoted"])
        r = out["receipt"]
        self.assertLess(r["learned_mean_cost"], r["static_mean_cost"])
        self.assertIn("matched held-out", r["reason"])
        self.assertEqual(out["policy"]({"size": 9.0}), "queue_pool")  # the promoted policy is usable

    def test_no_matched_support_means_no_promotion(self):
        def teleport(f):
            return "teleport"  # a choice never logged -> static cannot be scored

        out = meta_improve(_sched_rows(), teleport, cost_key="latency", seed=1)
        self.assertFalse(out["promoted"])
        self.assertIs(out["policy"], teleport)  # the teacher is kept
        self.assertIn("insufficient matched", out["receipt"]["reason"])

    def test_receipt_is_returned_either_way(self):
        out = meta_improve(_sched_rows(), _always_local, cost_key="latency", seed=1)
        for key in ("learned_mean_cost", "static_mean_cost", "n_holdout", "reason"):
            self.assertIn(key, out["receipt"])  # a non-promotion is auditable, never silent

    def test_too_few_rows_raise(self):
        with self.assertRaises(ValueError):
            meta_improve(_sched_rows(n=1)[:2], _always_local)


if __name__ == "__main__":
    unittest.main()
