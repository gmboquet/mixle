"""FAULT-a's last two modes wired into live consumers: model_error (Router routes past a raising tier)
and oracle_timeout (VerifiableOracle abstains rather than block/guess on a slow score_fn).
"""

import time
import unittest

from mixle.doe.oracle import OracleResult, VerifiableOracle
from mixle.task.router import Router


class _RaisingTier:
    def decide(self, x):
        raise RuntimeError("tier is on fire")


class _WorkingTier:
    def __init__(self, label):
        self.label = label

    def decide(self, x):
        return self.label


class RouterModelErrorTest(unittest.TestCase):
    def test_a_raising_tier_is_routed_past_and_flagged_not_crashed(self):
        router = Router(tiers=[("broken", _RaisingTier(), 0.0), ("frontier", lambda x: "teacher-answer", 1.0)])
        out = router(1)

        self.assertEqual(out, "teacher-answer")
        self.assertEqual(router.stats.tiers[-1].answered, 1)
        self.assertEqual(len(router.stats.degraded), 1)
        event = router.stats.degraded[0]
        self.assertEqual(event.mode, "model_error")
        self.assertIn("broken", event.reason)
        self.assertIn("tier is on fire", event.reason)

    def test_a_working_tier_after_a_broken_one_still_answers_locally(self):
        router = Router(
            tiers=[
                ("broken", _RaisingTier(), 0.0),
                ("cheap", _WorkingTier("cheap-answer"), 0.5),
                ("frontier", lambda x: "teacher-answer", 1.0),
            ]
        )
        out = router(1)

        self.assertEqual(out, "cheap-answer")
        self.assertEqual(router.stats.tiers[1].answered, 1)
        self.assertEqual(router.stats.tiers[-1].answered, 0)  # never reached the frontier
        self.assertEqual(len(router.stats.degraded), 1)

    def test_no_degradation_recorded_when_every_tier_behaves(self):
        router = Router(tiers=[("cheap", _WorkingTier("cheap-answer"), 0.0), ("frontier", lambda x: "x", 1.0)])
        router(1)
        self.assertEqual(router.stats.degraded, [])


class OracleTimeoutTest(unittest.TestCase):
    def test_a_slow_score_fn_abstains_instead_of_blocking(self):
        def _slow_score(candidate):
            time.sleep(5)
            return OracleResult(score=1.0)

        oracle = VerifiableOracle(name="slow", tier="executable", score_fn=_slow_score, timeout=0.2)

        started = time.monotonic()
        result = oracle(candidate=[0.0])
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 4.0)  # aborted well before the score_fn's own 5s sleep would return
        self.assertEqual(result.score, float("-inf"))
        self.assertEqual(result.cost, 0.0)
        self.assertEqual(result.receipt["degraded_mode"], "oracle_timeout")
        self.assertIn("oracle_id", result.receipt)

    def test_a_fast_score_fn_is_unaffected_by_a_timeout_budget(self):
        oracle = VerifiableOracle(
            name="fast", tier="executable", score_fn=lambda c: OracleResult(score=3.0), timeout=5.0
        )
        result = oracle(candidate=[0.0])
        self.assertEqual(result.score, 3.0)

    def test_no_timeout_configured_behaves_exactly_as_before(self):
        oracle = VerifiableOracle(name="plain", tier="executable", score_fn=lambda c: OracleResult(score=7.0))
        self.assertEqual(oracle(candidate=[0.0]).score, 7.0)


if __name__ == "__main__":
    unittest.main()
