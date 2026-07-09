"""CARD I3-a: propose_verify_retrain vs random search and a fixed greedy heuristic, matched budget.

A synthetic sequence-scoring oracle whose optimum is known by construction (a fixed target
sequence; score = clean position-match count + noise) lets us check that the reweighted-MLE
population loop in :mod:`mixle.task.propose` recovers more of the true optimum than random search
or single-candidate greedy coordinate ascent, at an EXACTLY matched oracle-call budget -- the noise
makes any one candidate's score unreliable, which is exactly the regime a population-based refit
should be more robust in than picking one noisy "best so far" sample.
"""

import unittest

import numpy as np

from mixle.doe.oracle import OracleResult, VerifiableOracle
from mixle.task.propose import ProposeVerifyResult, RoundLog, SequenceProposal, propose_verify_retrain

ALPHABET = ("A", "B", "C", "D")
LENGTH = 6
TARGET = ("A", "B", "C", "D", "A", "B")


def _matches(seq) -> int:
    return sum(1 for a, b in zip(seq, TARGET) if a == b)


def _noisy_oracle(seed: int, sigma: float = 1.0) -> VerifiableOracle:
    noise_rng = np.random.default_rng(seed)

    def score_fn(seq):
        return OracleResult(score=_matches(seq) + noise_rng.normal(0.0, sigma), receipt={"matches": _matches(seq)})

    return VerifiableOracle(name="toy-sequence", tier="simulation", score_fn=score_fn)


def _random_search_best_matches(oracle: VerifiableOracle, budget: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    best_score, best_matches = -np.inf, 0
    for _ in range(budget):
        seq = tuple(str(rng.choice(ALPHABET)) for _ in range(LENGTH))
        result = oracle(seq)
        if result.score > best_score:
            best_score, best_matches = result.score, _matches(seq)
    return best_matches


def _greedy_coordinate_ascent_best_matches(oracle: VerifiableOracle, budget: int, seed: int) -> int:
    """Fixed heuristic: single-candidate coordinate ascent, one position at a time, restarted with
    fresh random starts until the same oracle-call budget is spent."""
    rng = np.random.default_rng(seed)

    def one_pass():
        seq = [str(rng.choice(ALPHABET)) for _ in range(LENGTH)]
        calls = 0
        for i in range(LENGTH):
            best_sym, best_score = seq[i], oracle(tuple(seq)).score
            calls += 1
            for sym in ALPHABET:
                if sym == seq[i]:
                    continue
                trial = list(seq)
                trial[i] = sym
                score = oracle(tuple(trial)).score
                calls += 1
                if score > best_score:
                    best_score, best_sym = score, sym
            seq[i] = best_sym
        return tuple(seq), calls

    best_score, best_matches, calls_used = -np.inf, 0, 0
    while calls_used < budget:
        seq, calls = one_pass()
        calls_used += calls
        result = oracle(seq)
        if result.score > best_score:
            best_score, best_matches = result.score, _matches(seq)
    return best_matches


class SequenceProposalTestCase(unittest.TestCase):
    def test_sample_shape_and_alphabet(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=LENGTH)
        rng = np.random.default_rng(0)
        sequences = proposal.sample(25, rng)
        self.assertEqual(len(sequences), 25)
        for seq in sequences:
            self.assertEqual(len(seq), LENGTH)
            self.assertTrue(all(sym in ALPHABET for sym in seq))

    def test_refit_concentrates_toward_winners(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=3)
        winners = [("A", "A", "A")] * 5 + [("B", "B", "B")]
        weights = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 1.0])
        refit = proposal.refit(winners, weights)
        for model in refit.position_models:
            self.assertGreater(model.pmap["A"], model.pmap["B"])
            self.assertGreater(model.pmap["A"], 0.5)

    def test_refit_rejects_mismatched_weights(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=3)
        with self.assertRaises(ValueError):
            proposal.refit([("A", "A", "A")], np.array([1.0, 2.0]))

    def test_refit_rejects_all_nonpositive_weights(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=3)
        with self.assertRaises(ValueError):
            proposal.refit([("A", "A", "A")], np.array([0.0]))


class ProposeVerifyRetrainTestCase(unittest.TestCase):
    def test_no_oracle_refuses(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=LENGTH)
        with self.assertRaises(ValueError):
            propose_verify_retrain(proposal, None, k_per_round=4, rounds=1)

    def test_bad_keep_frac_raises(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=LENGTH)
        oracle = _noisy_oracle(seed=0)
        with self.assertRaises(ValueError):
            propose_verify_retrain(proposal, oracle, k_per_round=4, rounds=1, keep_frac=0.0)

    def test_budget_and_logging_accounting(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=LENGTH)
        oracle = _noisy_oracle(seed=0)
        k_per_round, rounds = 10, 3
        result = propose_verify_retrain(proposal, oracle, k_per_round=k_per_round, rounds=rounds, seed=0)
        self.assertIsInstance(result, ProposeVerifyResult)
        self.assertEqual(result.oracle_calls, k_per_round * rounds)
        self.assertEqual(len(result.rounds), rounds)
        for round_log in result.rounds:
            self.assertIsInstance(round_log, RoundLog)
            self.assertEqual(len(round_log.candidates), k_per_round)
            self.assertEqual(len(round_log.results), k_per_round)
            self.assertGreaterEqual(len(round_log.kept_indices), 1)
        # dead ends retained: every candidate tried appears in all_candidates(), not just winners
        self.assertEqual(len(result.all_candidates()), k_per_round * rounds)

    def test_deterministic_given_seed(self):
        proposal = SequenceProposal(alphabet=ALPHABET, length=LENGTH)
        oracle_a = _noisy_oracle(seed=7)
        oracle_b = _noisy_oracle(seed=7)
        result_a = propose_verify_retrain(proposal, oracle_a, k_per_round=8, rounds=2, seed=42)
        result_b = propose_verify_retrain(proposal, oracle_b, k_per_round=8, rounds=2, seed=42)
        for round_a, round_b in zip(result_a.rounds, result_b.rounds):
            self.assertEqual(round_a.candidates, round_b.candidates)

    def test_beats_random_search_and_fixed_heuristic_at_matched_budget(self):
        # A noisy oracle score makes any single observed candidate's score unreliable -- exactly the
        # regime a population-based refit should be more robust in than one noisy "best so far"
        # sample. Average over several independent seeds so the comparison is not one noisy draw.
        k_per_round, rounds = 40, 6
        budget = k_per_round * rounds
        propose_matches, random_matches, heuristic_matches = [], [], []
        for seed in range(8):
            proposal = SequenceProposal(alphabet=ALPHABET, length=LENGTH)
            result = propose_verify_retrain(
                proposal, _noisy_oracle(seed=seed), k_per_round=k_per_round, rounds=rounds, keep_frac=0.25, seed=seed
            )
            self.assertEqual(result.oracle_calls, budget)
            propose_matches.append(_matches(result.best_candidate))
            random_matches.append(
                _random_search_best_matches(_noisy_oracle(seed=seed), budget=budget, seed=seed + 1000)
            )
            heuristic_matches.append(
                _greedy_coordinate_ascent_best_matches(_noisy_oracle(seed=seed), budget=budget, seed=seed + 2000)
            )

        self.assertGreater(np.mean(propose_matches), np.mean(random_matches))
        self.assertGreater(np.mean(propose_matches), np.mean(heuristic_matches))


if __name__ == "__main__":
    unittest.main()
