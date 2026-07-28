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

    def test_mutating_a_recorded_event_into_an_invalid_state_is_refused(self):
        # MXR-080-1733: record() hands back the live Event and every flush rewrites the whole log
        # from the current field values, so assigning a bogus kind (or a NaN timestamp) rewrote
        # already-persisted history as a row no reader could load back. The mutable-outcome
        # "close the loop later" pattern stays supported; the event's identity does not.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            t = Telemetry(path)
            ev = t.record("fit", features={"n": 1})
            with self.assertRaises(ValueError):
                ev.kind = "invented"
            with self.assertRaises(ValueError):
                ev.ts = float("nan")
            ev.outcome["ll"] = -1.0  # still the documented pattern
            t.flush()
            reloaded = list(Telemetry(path).events())
            self.assertEqual([(e.kind, e.outcome) for e in reloaded], [("fit", {"ll": -1.0})])


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

    def test_outcome_mutation_after_record_is_persisted_on_explicit_flush(self):
        # Reproduces the reported bug: a record initially logged with a placeholder outcome (e.g.
        # "pending") is mutated in place afterward -- the documented "close the loop later"
        # pattern exercised by test_outcome_can_be_closed_later above. Before the fix, `record()`
        # flushed (and forgot) each event immediately (default flush_every=1), so a later
        # `flush()` was a no-op for it and a fresh reader of the JSONL log saw the stale
        # pre-mutation value forever. Flushing now rewrites the whole buffer, so the mutation
        # reaches disk.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            t = Telemetry(path)
            ev = t.record("escalation", features={"conf": 0.6}, choice="escalate", outcome={"status": "pending"})
            ev.outcome["status"] = "success"
            ev.outcome["correct"] = True
            t.flush()

            reloaded = list(Telemetry(path).events(kind="escalation"))
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded[0].outcome, {"status": "success", "correct": True})

    def test_outcome_mutation_is_swept_up_by_a_later_auto_flush_too(self):
        # No explicit flush() needed: recording a subsequent event triggers the default
        # flush_every=1 auto-flush, which (since it rewrites the full buffer) also persists the
        # earlier event's mutated outcome.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            t = Telemetry(path)
            ev = t.record("escalation", features={"conf": 0.6}, choice="escalate")
            ev.outcome["correct"] = True
            t.record("escalation", features={"conf": 0.9}, choice="answer")

            reloaded = list(Telemetry(path).events(kind="escalation"))
            self.assertEqual(reloaded[0].outcome, {"correct": True})

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

    def test_flush_never_writes_through_a_preplaced_temp_symlink(self):
        # MXR-080-1735: the flush always opened the fixed path "<log>.tmp" with mode "w", so a
        # symlink pre-placed there (trivial in a shared or world-writable log directory) got the
        # telemetry row written through it into an arbitrary external file -- and the following
        # rename moved the symlink itself onto the log path, redirecting every later flush too.
        with tempfile.TemporaryDirectory() as d:
            outside = os.path.join(d, "outside.txt")
            with open(outside, "w") as f:
                f.write("precious\n")
            logdir = os.path.join(d, "logs")
            os.mkdir(logdir)
            path = os.path.join(logdir, "events.jsonl")
            os.symlink(outside, path + ".tmp")

            t = Telemetry(path)
            t.record("fit", features={"n": 1})

            with open(outside) as f:
                self.assertEqual(f.read(), "precious\n")  # untouched
            self.assertFalse(os.path.islink(path))
            self.assertEqual(len(list(Telemetry(path).events())), 1)

    def test_independent_writers_do_not_erase_each_others_events(self):
        # MXR-080-1734: each recorder loaded one snapshot and later replaced the whole log with its
        # own buffer, so with two recorders open on the same path the second flush destroyed the
        # first's events outright.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            a = Telemetry(path)
            b = Telemetry(path)
            a.record("fit", features={"who": "A"})
            b.record("fit", features={"who": "B"})
            a.record("fit", features={"who": "A2"})

            who = sorted(ev.features["who"] for ev in Telemetry(path).events())
            self.assertEqual(who, ["A", "A2", "B"])

    def test_adopted_external_rows_are_not_duplicated_by_later_flushes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            a = Telemetry(path)
            b = Telemetry(path)
            b.record("fit", features={"who": "B"})
            ev = a.record("fit", features={"who": "A"}, outcome={"status": "pending"})
            ev.outcome["status"] = "done"
            a.flush()
            a.flush()

            rows = [(e.features["who"], e.outcome) for e in Telemetry(path).events()]
            self.assertEqual(sorted(rows), [("A", {"status": "done"}), ("B", {})])

    def test_deterministic_monotonic_clock(self):
        t = Telemetry()
        a = t.record("fit", choice="x")
        b = t.record("fit", choice="y")
        self.assertLess(a.ts, b.ts)  # strictly increasing without a wall clock (deterministic)
        c = t.record("fit", choice="z", when=1000.0)
        self.assertEqual(c.ts, 1000.0)  # explicit time honored

    def test_explicit_when_does_not_leave_clock_behind(self):
        # An explicitly-supplied `when=` must fold into the fallback clock's high-water mark,
        # not just override the one Event's `.ts`. Otherwise the clock stays wherever the
        # auto-increment counter left it, and the very next default-timestamped record() could
        # hand out a ts smaller than the one the caller just set explicitly -- the same class of
        # ordering corruption as the reload bug covered below, triggered without ever touching
        # disk.
        t = Telemetry()
        explicit = t.record("fit", choice="x", when=1000.0)
        self.assertEqual(explicit.ts, 1000.0)
        auto = t.record("fit", choice="y")
        self.assertGreater(auto.ts, explicit.ts)

    def test_reload_advances_clock_past_loaded_timestamps(self):
        # Reproduces the reported bug: a log containing an event at ts=100.0 (auto-assigned by
        # a recorder whose fallback clock had already advanced that far) is reopened in a fresh
        # Telemetry instance, which triggers _load(). Before the fix, _load() restored the
        # buffered events but never advanced the new instance's own _clock past what it just
        # read, so the freshly-loaded process's clock started over from 0.0 -- the very next
        # default-timestamped record() got ts=1.0, and the on-disk event order became
        # [100.0, 1.0]: a later real-world event with a *smaller* logical timestamp than an
        # earlier one, corrupting any ordering/causality assumption downstream code makes about
        # this log. _load() now scans the loaded events for the max timestamp and advances
        # _clock to at least that value.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            t1 = Telemetry(path)
            t1._clock = 99.0  # simulate a recorder that already auto-assigned up through ts=99.0
            first = t1.record("fit", choice="closed_form")
            self.assertEqual(first.ts, 100.0)

            t2 = Telemetry(path)  # reopen -- triggers _load()
            second = t2.record("fit", choice="em")

            self.assertGreater(second.ts, first.ts)  # monotone, not the broken [100.0, 1.0]

            reloaded = [ev.ts for ev in Telemetry(path).events()]
            self.assertEqual(reloaded, sorted(reloaded))


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
