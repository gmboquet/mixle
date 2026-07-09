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
        tampered = ExecutionTrace(
            request=trace.request,
            steps=[trace.steps[0], TraceStep(tool="draw_normal", args={"n": 5}, seed=99, result=trace.steps[1].result)],
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


if __name__ == "__main__":
    unittest.main()
