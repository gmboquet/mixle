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


class ReceiptConsistencyTest(unittest.TestCase):
    """MXR-080-1694: the public frozen receipt validated nothing, so a self-contradictory record --
    a *successful* result carrying a degradation mode and failure reason, or a degradation nobody can
    attribute -- constructed and serialized as an ordinary audit row."""

    def test_a_success_cannot_carry_a_degradation_mode_or_reason(self):
        with self.assertRaises(ValueError):
            DegradedResult("unsafe", False, "teacher_down", "failure")
        with self.assertRaises(ValueError):
            DegradedResult(value=1, degraded=False, mode="store_down")
        with self.assertRaises(ValueError):
            DegradedResult(value=1, degraded=False, reason="disk full")

    def test_a_degradation_must_name_its_mode_and_reason(self):
        with self.assertRaises(ValueError):
            DegradedResult(value=1, degraded=True)
        with self.assertRaises(ValueError):
            DegradedResult(value=1, degraded=True, mode="  ", reason="x")
        with self.assertRaises(ValueError):
            DegradedResult(value=1, degraded=True, mode="store_down")

    def test_an_unnamed_mode_is_refused_before_the_primary_path_runs(self):
        calls = []

        with self.assertRaises(ValueError):
            with_fallback(lambda: calls.append("primary"), lambda exc: None, mode="")
        self.assertEqual(calls, [])

    def test_caller_named_modes_outside_this_module_still_degrade(self):
        # mixle.inference.production.serving degrades under `model_unavailable`; a closed vocabulary
        # would turn its recoverable outage into a hard constructor failure.
        def down():
            raise ConnectionError("endpoint unreachable")

        result = with_fallback(down, lambda exc: "fallback", mode="model_unavailable")
        self.assertTrue(result.degraded)
        self.assertEqual(result.mode, "model_unavailable")

    def test_an_exception_with_no_message_still_produces_a_receipt(self):
        def down():
            raise ConnectionError

        result = with_fallback(down, lambda exc: "fallback", mode="teacher_down")
        self.assertTrue(result.degraded)
        self.assertEqual(result.reason, "")

    def test_the_helpers_all_build_consistent_receipts(self):
        self.assertTrue(with_fallback(lambda: 1, lambda exc: 0, mode="teacher_down").degraded is False)
        self.assertTrue(abstain_on_timeout(lambda: 1).degraded is False)
        timed_out = abstain_on_timeout(lambda: (_ for _ in ()).throw(TimeoutError("slow")))
        self.assertEqual(timed_out.mode, "oracle_timeout")


class DegradedAttemptsAreConsistentTest(unittest.TestCase):
    """MXR-080-1902 (High): ``DegradedResult.attempts`` is documented as "the structured causal
    evidence" for a degradation, and ``with_fallback``/``route_past`` both record it -- but an
    ``oracle_timeout`` abstention returned ``attempts=()``. Audit code walking ``attempts`` could not
    tell a timed-out oracle from a route that never degraded at all; ``reason`` is a human-readable
    string, not the same field."""

    def test_a_timeout_abstention_records_its_causal_evidence(self):
        def slow():
            raise TimeoutError("oracle call exceeded budget")

        result = abstain_on_timeout(slow)
        self.assertTrue(result.degraded)
        self.assertEqual(len(result.attempts), 1, "a degraded result carried no causal evidence")
        name, exc_repr = result.attempts[0]
        self.assertEqual(name, "oracle_timeout")
        self.assertIn("oracle call exceeded budget", exc_repr)
        self.assertIn("TimeoutError", exc_repr)

    def test_every_degraded_helper_now_names_at_least_one_failed_attempt(self):
        def boom():
            raise ConnectionError("provider outage")

        degraded = [
            with_fallback(boom, lambda exc: "fallback", mode="teacher_down"),
            route_past([boom, lambda: "second tier"], names=["first", "second"]),
            abstain_on_timeout(lambda: (_ for _ in ()).throw(TimeoutError("slow"))),
        ]
        for result in degraded:
            with self.subTest(mode=result.mode):
                self.assertTrue(result.degraded)
                self.assertTrue(result.attempts, f"{result.mode} degraded with no attempts recorded")
                self.assertTrue(all(isinstance(n, str) and isinstance(r, str) for n, r in result.attempts))

    def test_a_non_degraded_result_still_records_no_attempts(self):
        # Negative control: attempts are evidence of a FAILED attempt, so a clean run has none.
        for result in (
            with_fallback(lambda: 1, lambda exc: 0, mode="teacher_down"),
            route_past([lambda: 1]),
            abstain_on_timeout(lambda: 1),
        ):
            self.assertFalse(result.degraded)
            self.assertEqual(result.attempts, ())


if __name__ == "__main__":
    unittest.main()
