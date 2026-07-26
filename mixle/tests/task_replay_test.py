"""ExecutionTrace + replay: bit-identical re-execution given the same recorded args/seed (workstream H2)."""

import unittest

import numpy as np

from mixle.task.replay import ExecutionTrace, TraceStep, diff, is_bit_identical_replay, record_step, replay


def _draw_normal(n: int, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    return rng.normal(size=n).tolist()


def _uppercase(text: str) -> str:
    return text.upper()


_TOOLS = {"draw_normal": _draw_normal, "uppercase": _uppercase}


class ReplayTest(unittest.TestCase):
    def _record(self) -> ExecutionTrace:
        step1 = record_step(_TOOLS, "uppercase", {"text": "hello"})
        step2 = record_step(_TOOLS, "draw_normal", {"n": 5}, seed=42)
        return ExecutionTrace(request="demo", steps=[step1, step2])

    def test_replay_is_bit_identical_given_the_same_seed(self):
        trace = self._record()
        self.assertTrue(is_bit_identical_replay(trace, _TOOLS))

        replayed = replay(trace, _TOOLS)
        self.assertEqual(trace.steps[0].result, replayed.steps[0].result)
        self.assertEqual(trace.steps[1].result, replayed.steps[1].result)  # stochastic step, same seed

    def test_diff_detects_a_changed_seed(self):
        trace = self._record()
        original = trace.steps[1]
        tampered = ExecutionTrace(
            request=trace.request,
            steps=[
                trace.steps[0],
                TraceStep(
                    tool="draw_normal",
                    args={"n": 5},
                    seed=99,
                    result=original.result,
                    rng_state_before=original.rng_state_before,
                    rng_state_after=original.rng_state_after,
                ),
            ],
        )
        replayed = replay(tampered, _TOOLS)
        mismatches = diff(tampered, replayed)
        self.assertEqual(mismatches, [(1, "draw_normal")])

    def test_trace_round_trips_through_json(self):
        trace = self._record()
        restored = ExecutionTrace.from_json(trace.to_json())
        self.assertEqual(trace.dumps(), restored.dumps())
        self.assertTrue(is_bit_identical_replay(restored, _TOOLS))

    def test_length_mismatch_is_reported(self):
        trace = self._record()
        shorter = ExecutionTrace(request=trace.request, steps=trace.steps[:1])
        mismatches = diff(trace, shorter)
        self.assertIn((1, "length_mismatch"), mismatches)

    def test_request_arguments_and_seed_are_part_of_replay_identity(self):
        trace = self._record()
        changed_request = ExecutionTrace(request="different", steps=trace.steps)
        self.assertIn((-1, "request_mismatch"), diff(trace, changed_request))

        first = trace.steps[0]
        changed_args = TraceStep(
            tool=first.tool,
            args={"text": "different"},
            result=first.result,
            rng_state_before=first.rng_state_before,
            rng_state_after=first.rng_state_after,
        )
        self.assertEqual(diff(ExecutionTrace("x", [first]), ExecutionTrace("x", [changed_args])), [(0, "uppercase")])

    def test_empty_trace_is_not_replay_evidence_and_seed_must_be_supported(self):
        self.assertEqual(diff(ExecutionTrace("x"), ExecutionTrace("x")), [(0, "empty_trace")])
        with self.assertRaisesRegex(ValueError, "does not accept"):
            record_step(_TOOLS, "uppercase", {"text": "hello"}, seed=1)

    def test_global_rng_state_is_captured_and_restored_for_replay(self):
        tools = {"global": lambda: float(np.random.random())}
        np.random.seed(123)
        trace = ExecutionTrace("rng", [record_step(tools, "global", {})])
        np.random.seed(999)
        replayed = replay(trace, tools)
        self.assertEqual(diff(trace, replayed), [])


if __name__ == "__main__":
    unittest.main()
