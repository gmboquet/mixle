"""Parallel teacher labeling (mixle.task.distill._teacher_labels and the n_jobs knob).

The claims: n_jobs=1 is byte-for-byte the sequential batched call; n_jobs>1 preserves input order
for BOTH per-item and batched teachers; requests genuinely overlap in flight (measured, not
assumed); and the knob reaches the torch-free structured entry point end to end.
"""

import threading
import time
import unittest

from mixle.task.distill import _as_batched, _teacher_labels


def _label(text: str) -> str:
    return "long" if len(str(text)) > 5 else "short"


class TeacherLabelsTest(unittest.TestCase):
    TEXTS = [f"item-{i}" * (1 + i % 3) for i in range(23)]

    def test_sequential_path_is_exactly_as_batched(self):
        want = _as_batched(_label)(list(self.TEXTS))
        self.assertEqual(_teacher_labels(_label, list(self.TEXTS), n_jobs=1), want)

    def test_per_item_teacher_keeps_order_across_threads(self):
        want = [_label(t) for t in self.TEXTS]
        self.assertEqual(_teacher_labels(_label, list(self.TEXTS), n_jobs=4), want)

    def test_batched_teacher_keeps_order_across_threads(self):
        def batched_teacher(texts):
            return [_label(t) for t in texts]

        want = [_label(t) for t in self.TEXTS]
        self.assertEqual(_teacher_labels(batched_teacher, list(self.TEXTS), n_jobs=3), want)

    def test_requests_actually_overlap(self):
        lock = threading.Lock()
        state = {"in_flight": 0, "max_in_flight": 0}

        def slow_teacher(text):
            with lock:
                state["in_flight"] += 1
                state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            time.sleep(0.02)
            with lock:
                state["in_flight"] -= 1
            return _label(text)

        want = [_label(t) for t in self.TEXTS]
        self.assertEqual(_teacher_labels(slow_teacher, list(self.TEXTS), n_jobs=4), want)
        self.assertGreaterEqual(state["max_in_flight"], 2)  # concurrency observed, not assumed

    def test_single_item_short_circuits(self):
        self.assertEqual(_teacher_labels(_label, ["ab"], n_jobs=8), ["short"])


class EndToEndTest(unittest.TestCase):
    def test_structured_distill_accepts_n_jobs(self):
        # distill_structured is torch-free: the full path from parallel teacher labels to a fitted
        # student, with the parallel and sequential labelings producing the same student decisions.
        from mixle.task.distill import distill_structured

        records = [(float(i % 7), ("a", "b", "c")[i % 3]) for i in range(60)]

        def teacher(rec):
            if isinstance(rec, list):  # the batched-probe protocol hands the whole list first
                return [teacher(r) for r in rec]
            return "big" if rec[0] >= 3.0 else "small"

        seq = distill_structured(teacher, records, seed=0)
        par = distill_structured(teacher, records, seed=0, n_jobs=4)
        self.assertEqual([seq(r) for r in records], [par(r) for r in records])


if __name__ == "__main__":
    unittest.main()
