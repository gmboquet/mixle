"""Degradation policy primitives (mixle.system.fault), CARD FAULT-a: named modes, each flagged, never silent."""

import unittest

from mixle.system import DegradedResult, abstain_on_timeout, route_past, with_fallback
from mixle.system.fault import NON_RECOVERABLE_FAULTS


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


class RecoverabilityPolicyTest(unittest.TestCase):
    """MXR-080-1693: both boundaries caught EVERY exception with no declared recoverability policy, so
    an authorization denial and an internal programming bug took the same route as a provider outage --
    a PermissionError came back as somebody else's successful answer, and a TypeError became an
    ordinary fallback value with the defect never surfacing."""

    def test_an_authorization_denial_is_not_a_degraded_answer(self):
        def denied():
            raise PermissionError("denied")

        with self.assertRaises(PermissionError):
            with_fallback(denied, lambda exc: "fallback answer", mode="teacher_down")
        with self.assertRaises(PermissionError):
            route_past([denied, lambda: "next tier answer"], names=["restricted", "next"])

    def test_a_programming_error_is_not_a_degraded_answer(self):
        def bug():
            raise TypeError("unsupported operand type(s)")

        with self.assertRaises(TypeError):
            with_fallback(bug, lambda exc: "fallback answer", mode="teacher_down")
        with self.assertRaises(TypeError):
            route_past([bug, lambda: "next tier answer"], names=["local", "frontier"])

    def test_genuine_provider_outages_still_degrade(self):
        def down():
            raise ConnectionError("endpoint unreachable")

        result = with_fallback(down, lambda exc: "store-only answer", mode="teacher_down")
        self.assertTrue(result.degraded)
        self.assertEqual(result.value, "store-only answer")
        routed = route_past([down, lambda: "frontier answer"], names=["local", "frontier"])
        self.assertTrue(routed.degraded)
        self.assertEqual(routed.value, "frontier answer")

    def test_an_explicit_allowlist_narrows_a_route(self):
        def down():
            raise ConnectionError("endpoint unreachable")

        with self.assertRaises(ConnectionError):
            with_fallback(down, lambda exc: "x", mode="teacher_down", recoverable=(TimeoutError,))
        allowed = with_fallback(down, lambda exc: "x", mode="teacher_down", recoverable=(ConnectionError,))
        self.assertTrue(allowed.degraded)
        with self.assertRaises(ConnectionError):
            route_past([down, lambda: "next"], names=["a", "b"], recoverable=(TimeoutError,))

    def test_every_attempted_tier_keeps_its_causal_evidence(self):
        def down_a():
            raise ConnectionError("a unreachable")

        def down_b():
            raise TimeoutError("b timed out")

        result = route_past([down_a, down_b, lambda: "ok"], names=["a", "b", "c"])
        self.assertEqual([name for name, _ in result.attempts], ["a", "b"])
        self.assertIn("a unreachable", result.attempts[0][1])
        self.assertIn("b timed out", result.attempts[1][1])

    def test_the_non_recoverable_set_covers_programmer_and_policy_failures(self):
        for cls in (TypeError, AttributeError, NameError, AssertionError, PermissionError):
            self.assertIn(cls, NON_RECOVERABLE_FAULTS)


if __name__ == "__main__":
    unittest.main()
