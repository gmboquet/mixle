"""VerifiableOracle + optimize_under_oracle: the honesty boundary for de novo optimization (workstream
I1-I3). Proven here against a cheap closed-form oracle (a scored parameter vector), per the plan's own
build order, before any domain oracle exists.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.doe.oracle import DesignRun, OracleResult, VerifiableOracle, optimize_under_oracle


def _quadratic_bowl_oracle(target, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)

    def score_fn(x):
        d2 = float(np.sum((np.asarray(x, dtype=float) - target) ** 2))
        noisy = -d2 + (rng.normal(0, noise) if noise else 0.0)
        return OracleResult(score=noisy, receipt={"target_dist2": d2}, cost=1.0)

    return VerifiableOracle(name="quadratic_bowl", tier="executable", score_fn=score_fn, fidelity="exact, noiseless")


class VerifiableOracleConstructionTest(unittest.TestCase):
    def test_valid_tiers_construct(self):
        for tier in ("executable", "simulation", "held_out_truth", "real_measurement"):
            VerifiableOracle(name="ok", tier=tier, score_fn=lambda x: OracleResult(0.0))  # must not raise

    def test_self_graded_tier_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as ctx:
            VerifiableOracle(name="bad", tier="self_graded", score_fn=lambda x: OracleResult(0.0))
        self.assertIn("self_graded", str(ctx.exception))

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            VerifiableOracle(name="bad", tier="vibes", score_fn=lambda x: OracleResult(0.0))

    def test_oracle_is_callable(self):
        oracle = VerifiableOracle(name="ok", tier="executable", score_fn=lambda x: OracleResult(score=float(x[0])))
        result = oracle(np.array([3.0]))
        self.assertIsInstance(result, OracleResult)
        self.assertEqual(result.score, 3.0)


class NoOracleGuardTest(unittest.TestCase):
    def test_no_oracle_refuses_rather_than_fabricates(self):
        with self.assertRaises(ValueError) as ctx:
            optimize_under_oracle(None, [(-1.0, 1.0)])
        self.assertIn("no verifiable objective", str(ctx.exception))


@unittest.skipUnless(_HAS_TORCH, "the BO proposal model needs torch")
class DesignLoopTest(unittest.TestCase):
    def test_finds_the_target_and_returns_a_receipted_run(self):
        target = np.array([0.5, -1.0])
        oracle = _quadratic_bowl_oracle(target)
        run = optimize_under_oracle(
            oracle,
            [(-3.0, 3.0), (-3.0, 3.0)],
            n_init=6,
            n_iter=20,
            seed=0,
            n_candidates=256,
            fit_kwargs={"max_its": 60},
        )
        self.assertIsInstance(run, DesignRun)
        self.assertEqual(run.oracle_calls, 26)
        self.assertLess(run.best.result.receipt["target_dist2"], 0.5)  # genuinely converged, not a lucky init draw

    def test_report_names_the_oracle_identity_and_fidelity(self):
        oracle = _quadratic_bowl_oracle(np.array([1.0, 1.0]))
        run = optimize_under_oracle(
            oracle, [(-3.0, 3.0)] * 2, n_init=6, n_iter=10, seed=1, n_candidates=256, fit_kwargs={"max_its": 60}
        )
        rep = run.report()
        self.assertEqual(rep["oracle"], "quadratic_bowl")
        self.assertEqual(rep["tier"], "executable")
        self.assertEqual(rep["fidelity"], "exact, noiseless")
        self.assertEqual(rep["oracle_calls"], 16)
        self.assertIn("target_dist2", rep["best_receipt"])

    def test_beats_random_search_at_matched_oracle_call_budget(self):
        """The I acceptance: on a known closed-form oracle, the loop beats random search at matched budget."""
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        oracle = _quadratic_bowl_oracle(target)
        budget = 26  # n_init=6 + n_iter=20, matching the loop's own call count

        run = optimize_under_oracle(
            oracle, bounds, n_init=6, n_iter=20, seed=0, n_candidates=256, fit_kwargs={"max_its": 60}
        )
        bo_dist2 = run.best.result.receipt["target_dist2"]

        rng = np.random.RandomState(0)
        random_pts = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(budget, 2))
        random_best = min(float(np.sum((p - target) ** 2)) for p in random_pts)

        self.assertLess(bo_dist2, random_best)  # acquisition-driven search beats matched-budget random search

    def test_negative_result_is_kept_in_the_history_not_hidden(self):
        """A candidate the surrogate visited but that scored poorly stays in the run log."""
        oracle = _quadratic_bowl_oracle(np.array([0.0, 0.0]))
        run = optimize_under_oracle(
            oracle, [(-3.0, 3.0)] * 2, n_init=6, n_iter=10, seed=2, n_candidates=256, fit_kwargs={"max_its": 60}
        )
        scores = run.scores()
        self.assertEqual(len(scores), run.oracle_calls)
        self.assertGreater(
            scores.max() - scores.min(), 0.0
        )  # some candidates were verifiably worse, and are still there


if __name__ == "__main__":
    unittest.main()
