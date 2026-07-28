"""mixle.epistemic.journal: append-only, replayable decision log (Card E5)."""

import unittest
from dataclasses import replace

import numpy as np

from mixle.epistemic.journal import EpistemicJournal
from mixle.epistemic.loop import step
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio


def _gaussian_likelihood(hypothesis, observation):
    return float(np.exp(-0.5 * (observation - hypothesis.payload) ** 2))


def _five_steps():
    hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 2.0), Hypothesis("h2", 5.0)]
    portfolio = HypothesisPortfolio(hyps, np.array([1 / 3, 1 / 3, 1 / 3]), w_open=0.0)
    rng = np.random.RandomState(0)
    journal = EpistemicJournal()
    for i in range(5):
        observation = rng.normal(loc=2.0, scale=1.0)
        outcome = step(portfolio, observation, _gaussian_likelihood)
        journal.append(outcome, rationale=f"step {i}", timestamp=float(i))
        portfolio = outcome.portfolio_after
    return journal


def _single_hypothesis_journal(payload):
    """A minimal one-hypothesis, one-record journal -- for payload type-fidelity tests where the
    belief update itself is a no-op (single hypothesis, constant likelihood)."""
    hyps = [Hypothesis("h0", payload)]
    portfolio = HypothesisPortfolio(hyps, np.array([1.0]), w_open=0.0)
    outcome = step(portfolio, "obs", lambda h, y: 1.0)
    journal = EpistemicJournal()
    journal.append(outcome, rationale="payload fidelity check")
    return journal


class JournalRoundTripTest(unittest.TestCase):
    def test_to_json_from_json_round_trips_exactly(self):
        journal = _five_steps()
        restored = EpistemicJournal.from_json(journal.to_json())
        self.assertEqual(len(restored), len(journal))
        for original, back in zip(journal.records, restored.records):
            self.assertEqual(original, back)


class ContentAddressTest(unittest.TestCase):
    def test_hash_is_stable_for_identical_content_and_changes_when_weights_differ(self):
        journal = _five_steps()
        first, second = journal.records[0], journal.records[1]
        self.assertEqual(first.belief_snapshot_hash, first.belief_snapshot_hash)
        self.assertNotEqual(first.belief_snapshot_hash, second.belief_snapshot_hash)

    def test_verify_detects_a_corrupted_snapshot(self):
        journal = _five_steps()
        self.assertTrue(journal.verify())
        corrupted = journal.records[2].portfolio_snapshot
        corrupted["w_open"] = corrupted["w_open"] + 0.5  # mutate in place, hash now stale
        self.assertFalse(journal.verify())


class ReplayTest(unittest.TestCase):
    def test_replay_reconstructs_the_belief_trajectory(self):
        journal = _five_steps()
        trajectory = journal.replay()
        self.assertEqual(len(trajectory), len(journal))
        last_record_weights = journal.records[-1].portfolio_snapshot["weights"]
        self.assertTrue(np.allclose(trajectory[-1].weights, last_record_weights))


class PayloadTypeFidelityTest(unittest.TestCase):
    """A hypothesis payload's TYPE, not just its JSON-rendered value, must survive to_json/from_json.

    JSON has no tuple type: a naive round trip silently turns ``(1, 2)`` into ``[1, 2]`` -- a
    different Python value (``(1, 2) != [1, 2]``) that happens to serialize to identical JSON text, so
    a hash computed over that text can't tell the two apart either. These cover the codec
    (``mixle.epistemic.journal._tag_encode``/``_tag_decode``) that closes this for the small set of
    non-JSON-native types it supports exactly (tuple, numpy scalar, numpy array), and the honest,
    non-crashing, warned fallback for everything else.
    """

    def test_tuple_payload_round_trips_as_a_tuple_not_a_list(self):
        journal = _single_hypothesis_journal((1, 2))
        restored = EpistemicJournal.from_json(journal.to_json())
        payload = restored.replay()[0].hypotheses[0].payload
        self.assertIsInstance(payload, tuple)
        self.assertEqual(payload, (1, 2))
        self.assertTrue(restored.verify())

    def test_verify_detects_a_tuple_silently_replaced_by_an_equal_valued_list(self):
        # Negative control for the above: verify() must be sensitive to the tuple-vs-list distinction
        # itself, not just to content -- otherwise the round-trip test would pass for the wrong reason
        # (this is exactly the pre-fix bug: a tuple corrupted into an equal-valued list used to hash
        # identically and verify() returned True over it).
        journal = _single_hypothesis_journal((1, 2))
        self.assertTrue(journal.verify())
        journal.records[0].portfolio_snapshot["hypotheses"][0]["payload"] = [1, 2]  # same values, wrong type
        self.assertFalse(journal.verify())

    def test_numpy_scalar_and_array_payloads_round_trip_with_dtype(self):
        hyps = [Hypothesis("scalar", np.float32(3.5)), Hypothesis("array", np.arange(4, dtype=np.int16))]
        portfolio = HypothesisPortfolio(hyps, np.array([0.5, 0.5]), w_open=0.0)
        outcome = step(portfolio, "obs", lambda h, y: 1.0)
        journal = EpistemicJournal()
        journal.append(outcome, rationale="numpy payload fidelity")

        restored = EpistemicJournal.from_json(journal.to_json())
        restored_hyps = restored.replay()[0].hypotheses
        scalar_payload, array_payload = restored_hyps[0].payload, restored_hyps[1].payload

        self.assertIsInstance(scalar_payload, np.float32)
        self.assertEqual(scalar_payload, np.float32(3.5))
        self.assertIsInstance(array_payload, np.ndarray)
        self.assertEqual(array_payload.dtype, np.int16)
        np.testing.assert_array_equal(array_payload, np.arange(4, dtype=np.int16))
        self.assertTrue(restored.verify())

    def test_unsupported_payload_type_falls_back_to_a_warned_string_snapshot(self):
        # No general codec for arbitrary custom classes (out of scope -- see the module docstring):
        # real callers (e.g. mixle.task.discrepancy_invention_loop) journal fitted-model objects as
        # hypothesis payloads today, so this must degrade gracefully (str() + warn), not raise.
        class Opaque:
            def __repr__(self):
                return "Opaque()"

        with self.assertWarns(UserWarning):
            journal = _single_hypothesis_journal(Opaque())
        self.assertTrue(journal.verify())  # the stored string form is self-consistent, unmodified

        with self.assertWarns(UserWarning):
            restored = EpistemicJournal.from_json(journal.to_json())
        payload = restored.replay()[0].hypotheses[0].payload
        self.assertEqual(payload, "Opaque()")  # str(), not the original object -- documented limitation
        self.assertTrue(restored.verify())  # verify() now only attests the STRING is unchanged, and it is

    def test_dict_payload_with_reserved_type_key_falls_back_gracefully(self):
        # A dict payload that happens to use "__type__" as a key would be ambiguous with the codec's
        # own tagging scheme on decode, so it is routed through the same opaque str() fallback instead
        # of being silently misread as (or crashing on) one of the codec's own tags.
        with self.assertWarns(UserWarning):
            journal = _single_hypothesis_journal({"__type__": "not-ours", "value": 1})
        restored = EpistemicJournal.from_json(journal.to_json())
        payload = restored.replay()[0].hypotheses[0].payload
        self.assertIsInstance(payload, str)
        self.assertTrue(restored.verify())

    def test_from_json_rejects_an_unrecognized_codec_tag(self):
        journal = _single_hypothesis_journal((1, 2))
        corrupted = journal.to_json().replace('"tuple"', '"totally_unknown_tag"')
        with self.assertRaises(ValueError):
            EpistemicJournal.from_json(corrupted)


class WholeRecordIntegrityTest(unittest.TestCase):
    """MXR-080-1746: verify() covered only the snapshot, so every decision field was unprotected."""

    def _mutated(self, **changes):
        journal = _five_steps()
        self.assertTrue(journal.verify())
        return EpistemicJournal([replace(journal.records[0], **changes), *journal.records[1:]])

    def test_every_decision_field_is_covered_by_the_record_hash(self):
        for field_name, value in (
            ("surprise", 0.123),
            ("action_chosen", "a-different-action"),
            ("action_considered", ["fabricated"]),
            ("action_eig", 99.0),
            ("timestamp", 12345.0),
            ("rationale", "a rewritten justification"),
            ("step_index", 3),
        ):
            with self.subTest(field=field_name):
                self.assertFalse(self._mutated(**{field_name: value}).verify())

    def test_deleting_a_record_breaks_the_chain(self):
        journal = _five_steps()
        self.assertFalse(EpistemicJournal(list(journal.records[:2]) + list(journal.records[3:])).verify())

    def test_reordering_records_breaks_the_chain(self):
        records = list(_five_steps().records)
        records[1], records[2] = records[2], records[1]
        self.assertFalse(EpistemicJournal(records).verify())

    def test_truncating_from_the_front_breaks_the_chain(self):
        self.assertFalse(EpistemicJournal(list(_five_steps().records[1:])).verify())

    def test_appending_a_forged_record_breaks_the_chain(self):
        journal = _five_steps()
        forged = replace(journal.records[-1], step_index=len(journal), rationale="forged")
        self.assertFalse(EpistemicJournal([*journal.records, forged]).verify())

    def test_an_intact_journal_still_verifies_across_a_json_round_trip(self):
        journal = _five_steps()
        self.assertTrue(journal.verify())
        self.assertTrue(EpistemicJournal.from_json(journal.to_json()).verify())

    def test_the_backing_record_list_is_not_publicly_mutable(self):
        journal = _five_steps()
        journal.records[:1]  # indexing/slicing still works
        with self.assertRaises(AttributeError):
            journal.records.clear()
        self.assertEqual(len(journal), 5)


if __name__ == "__main__":
    unittest.main()
