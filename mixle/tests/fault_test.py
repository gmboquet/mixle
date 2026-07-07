"""Degradation policy primitives (mixle.fault), CARD FAULT-a: named modes, each flagged, never silent."""

import unittest

from mixle.fault import DegradedResult, abstain_on_timeout, route_past, with_fallback


class WithFallbackTest(unittest.TestCase):
    def test_primary_success_is_not_degraded(self):
        result = with_fallback(lambda: 42, lambda exc: 0, mode="unused")
        self.assertEqual(result, DegradedResult(value=42, degraded=False))

    def test_primary_failure_flags_the_named_mode_and_reason(self):
        def boom():
            raise ValueError("teacher endpoint unreachable")

        result = with_fallback(boom, lambda exc: "fallback answer", mode="teacher_down")
        self.assertTrue(result.degraded)
        self.assertEqual(result.mode, "teacher_down")
        self.assertEqual(result.value, "fallback answer")
        self.assertIn("teacher endpoint unreachable", result.reason)

    def test_fallback_that_also_fails_propagates(self):
        def boom():
            raise ValueError("primary down")

        def fallback_boom(exc):
            raise RuntimeError("fallback has nothing either")

        with self.assertRaises(RuntimeError):
            with_fallback(boom, fallback_boom, mode="teacher_down")

    def test_to_receipt_fields_shape(self):
        result = DegradedResult(value=1, degraded=True, mode="store_down", reason="disk full")
        self.assertEqual(result.to_receipt_fields(), {"degraded_mode": "store_down", "degraded_reason": "disk full"})


class AbstainOnTimeoutTest(unittest.TestCase):
    def test_timeout_abstains_with_none_value(self):
        def slow():
            raise TimeoutError("oracle call exceeded budget")

        result = abstain_on_timeout(slow)
        self.assertTrue(result.degraded)
        self.assertEqual(result.mode, "oracle_timeout")
        self.assertIsNone(result.value)

    def test_non_timeout_failure_propagates(self):
        def broken():
            raise ValueError("not a timeout")

        with self.assertRaises(ValueError):
            abstain_on_timeout(broken)

    def test_success_is_not_degraded(self):
        result = abstain_on_timeout(lambda: "score")
        self.assertFalse(result.degraded)
        self.assertEqual(result.value, "score")


class RoutePastTest(unittest.TestCase):
    def test_first_tier_success_is_not_degraded(self):
        result = route_past([lambda: "cheap tier answer", lambda: "frontier answer"], names=["local", "frontier"])
        self.assertFalse(result.degraded)
        self.assertEqual(result.value, "cheap tier answer")

    def test_failing_tier_is_routed_past_and_flagged(self):
        def broken():
            raise RuntimeError("local model errored")

        result = route_past([broken, lambda: "frontier answer"], names=["local", "frontier"])
        self.assertTrue(result.degraded)
        self.assertEqual(result.mode, "model_error")
        self.assertEqual(result.value, "frontier answer")
        self.assertIn("local", result.reason)

    def test_every_tier_failing_raises_the_last_exception(self):
        def boom_a():
            raise RuntimeError("tier a down")

        def boom_b():
            raise ValueError("tier b down")

        with self.assertRaises(ValueError):
            route_past([boom_a, boom_b], names=["a", "b"])


if __name__ == "__main__":
    unittest.main()
