"""Telemetry: typed events, local JSONL buffer, training-row extraction, global recorder."""

import os
import tempfile
import unittest

from mixle.telemetry import Event, Telemetry, get_default_recorder, record, set_default_recorder


class EventTest(unittest.TestCase):
    def test_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            Event(kind="telepathy")

    def test_as_row_round_trips(self):
        ev = Event(kind="fit", features={"n": 10}, choice="em", outcome={"ll": -3.2}, tags={"task": "x"})
        back = Event(**ev.as_row())
        self.assertEqual((back.kind, back.choice, back.features, back.outcome), ("fit", "em", {"n": 10}, {"ll": -3.2}))


class RecorderTest(unittest.TestCase):
    def test_record_buffer_and_filter(self):
        t = Telemetry()
        t.record("fit", features={"n": 1}, choice="closed_form")
        t.record("placement", features={"tflop": 8}, choice="pool")
        t.record("fit", features={"n": 2}, choice="em")
        self.assertEqual(len(t), 3)
        self.assertEqual(len(list(t.events(kind="fit"))), 2)

    def test_training_rows_are_feature_choice_outcome(self):
        t = Telemetry()
        t.record("placement", features={"tflop": 8.2, "has_pool": True}, choice="pool", outcome={"cost": 0.41})
        rows = t.training_rows("placement")
        self.assertEqual(rows, [({"tflop": 8.2, "has_pool": True}, "pool", {"cost": 0.41})])

    def test_outcome_can_be_closed_later(self):
        t = Telemetry()
        ev = t.record("escalation", features={"conf": 0.6}, choice="escalate")
        ev.outcome["correct"] = True
        self.assertEqual(t.training_rows("escalation")[0][2], {"correct": True})

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            t = Telemetry(path)
            for i in range(5):
                t.record("route", features={"i": i}, choice="tier0")
            t.flush()
            t2 = Telemetry(path)
            self.assertEqual(len(t2), 5)
            self.assertEqual([e.features["i"] for e in t2.events()], [0, 1, 2, 3, 4])

    def test_deterministic_monotonic_clock(self):
        t = Telemetry()
        a = t.record("fit", choice="x")
        b = t.record("fit", choice="y")
        self.assertLess(a.ts, b.ts)  # strictly increasing without a wall clock (deterministic)
        c = t.record("fit", choice="z", when=1000.0)
        self.assertEqual(c.ts, 1000.0)  # explicit time honored


class GlobalRecorderTest(unittest.TestCase):
    def setUp(self):
        self._saved = get_default_recorder()

    def tearDown(self):
        set_default_recorder(self._saved)

    def test_global_one_liner(self):
        set_default_recorder(Telemetry())
        record("reason", features={"budget": 5}, choice="retrieve", outcome={"gain": 1.2})
        self.assertEqual(len(get_default_recorder()), 1)


if __name__ == "__main__":
    unittest.main()
