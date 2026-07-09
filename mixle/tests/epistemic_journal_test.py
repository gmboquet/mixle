"""mixle.epistemic.journal: append-only, replayable decision log (Card E5)."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
