"""Speculative enumeration (RescoredIndex) + the teacher-forcing scoring contracts.

The economics under test: a cheap DRAFT model pays for the index build; the expensive TARGET is touched
only for returned sequences, via ONE batched teacher-forcing call (``score_sequences``). Verified on small
enumerable models where the target-exact truth is computable: top_k/slice return target-exact scores in
target order and match the target's own exact index when the window covers the draft/target disagreement;
the ``assumed_gap`` certificate is sound (certified results ARE the true top-k); ``log_density`` uses one
scorer call instead of a per-token walk when prefixes are uncached; ``harvest`` fills L cache entries from
one all-position call; corpus-calibrated envelopes equal ancestrally-calibrated ones given the same
contexts.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import AREnvelopeIndex, AutoregressiveEnumerable, RescoredIndex, SeekIndex


def _logits_model(W):
    L = W.shape[0]

    def nlp(prefix):
        d = len(prefix)
        last = prefix[-1] if prefix else 0
        lg = W[d, last]
        m = np.max(lg)
        return np.arange(W.shape[1]), lg - (m + math.log(np.sum(np.exp(lg - m))))

    return nlp, L


def _pair(V=6, L=4, seed=0, noise=0.3):
    """A target model and a draft = the target with perturbed logits (bounded score gap)."""
    rng = np.random.RandomState(seed)
    Wt = rng.randn(L, V, V)
    Wd = Wt + rng.randn(L, V, V) * noise
    target_nlp, _ = _logits_model(Wt)
    draft_nlp, _ = _logits_model(Wd)
    return target_nlp, draft_nlp


class RescoredIndexTest(unittest.TestCase):
    def setUp(self):
        self.target_nlp, self.draft_nlp = _pair(seed=1)
        self.target = AutoregressiveEnumerable(self.target_nlp, max_len=4)
        self.draft = AutoregressiveEnumerable(self.draft_nlp, max_len=4)
        self.exact = SeekIndex(self.target)
        self.exact.ensure_bits(40.0)

    def test_top_k_matches_brute_target_truth(self):
        # brute truth over the whole 6**4 support: rescoring by CONTINUOUS target scores is finer than
        # any bucketed index, so the returned set must equal the true top-10 exactly (window covers the
        # draft/target disagreement at this seed)
        import itertools

        brute = sorted(
            ((s, self.target.log_density(s)) for s in itertools.product(range(6), repeat=4)),
            key=lambda u: -u[1],
        )
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=500)
        out = ri.top_k(10)
        lps = [lp for _s, lp in out["items"]]
        self.assertEqual(lps, sorted(lps, reverse=True))
        for seq, lp in out["items"]:
            self.assertAlmostEqual(lp, self.target.log_density(seq), places=12)
        self.assertEqual({s for s, _ in out["items"]}, {s for s, _ in brute[:10]})
        self.assertAlmostEqual(lps[-1], brute[9][1], places=12)

    def test_slice_is_target_ordered_and_scored(self):
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=300)
        out = ri.slice(5, 7)
        self.assertEqual(len(out["items"]), 7)
        lps = [lp for _s, lp in out["items"]]
        self.assertEqual(lps, sorted(lps, reverse=True))
        for seq, lp in out["items"]:
            self.assertAlmostEqual(lp, self.target.log_density(seq), places=12)

    def test_certificate_soundness_under_assumed_gap(self):
        # the draft's perturbation gives a TRUE global gap bound; compute it and certify against it
        true_gap = 0.0
        import itertools

        for seq in itertools.product(range(6), repeat=4):
            true_gap = max(true_gap, abs(self.target.log_density(seq) - self.draft.log_density(seq)))
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=250, assumed_gap=true_gap)
        out = ri.top_k(5)
        self.assertIsNotNone(out["certified"])
        if out["certified"]:  # a certified result must BE the true top-5 (as a set)
            true5 = {self.exact.unrank(i)[0] for i in range(5)}
            self.assertEqual({s for s, _ in out["items"]}, true5)
        self.assertLessEqual(out["gap"], true_gap + 1e-9)  # observed gap never exceeds the true bound

    def test_certified_true_when_draft_exhausted(self):
        # window larger than the whole support: nothing unpulled exists, so the result is provably complete
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=3000, assumed_gap=1e6)
        out = ri.top_k(5)
        self.assertTrue(out["certified"])
        true5 = {self.exact.unrank(i)[0] for i in range(5)}
        self.assertEqual({s for s, _ in out["items"]}, true5)

    def test_unrank_returns_target_exact_scores(self):
        ri = RescoredIndex(SeekIndex(self.draft), self.target)
        seq, lp = ri.unrank(17)
        self.assertAlmostEqual(lp, self.target.log_density(seq), places=12)

    def test_composes_with_envelope_draft_for_deep_ranks(self):
        # draft = envelope index over the CHEAP model: deep rank access + target-exact scores
        env = AREnvelopeIndex(self.draft, n_paths=32, seed=0)
        ri = RescoredIndex(env, self.target)
        seq, lp = ri.unrank(700)
        self.assertEqual(len(seq), 4)
        self.assertAlmostEqual(lp, self.target.log_density(seq), places=12)

    def test_one_batched_target_call_per_query(self):
        calls = []

        def counting_scorer(seqs):
            calls.append(len(seqs))
            return np.array([self.target.log_density(s) for s in seqs])

        ri = RescoredIndex(SeekIndex(self.draft), counting_scorer, rerank_window=50)
        ri.top_k(10)
        self.assertEqual(len(calls), 1)  # one batched forward for the whole query
        self.assertEqual(calls[0], 60)  # k + window


class ScoringContractsTest(unittest.TestCase):
    def test_log_density_uses_batch_scorer_when_uncached(self):
        target_nlp, _ = _pair(seed=2)
        scored = []

        def scorer(seqs):
            scored.append(list(seqs))
            plain = AutoregressiveEnumerable(target_nlp, max_len=3)
            return np.array([plain.log_density(s) for s in seqs])

        ar = AutoregressiveEnumerable(target_nlp, max_len=3, batch_score_sequences=scorer)
        lp = ar.log_density((1, 2, 0))
        self.assertEqual(len(scored), 1)  # one teacher-forcing call, not a per-token walk
        plain = AutoregressiveEnumerable(target_nlp, max_len=3)
        self.assertAlmostEqual(lp, plain.log_density((1, 2, 0)), places=12)

    def test_score_sequences_batches_or_falls_back(self):
        target_nlp, _ = _pair(seed=3)
        ar_plain = AutoregressiveEnumerable(target_nlp, max_len=3)
        seqs = [(0, 1, 2), (3, 3, 3), (5, 0, 1)]
        fallback = ar_plain.score_sequences(seqs)  # cached-walk fallback
        expected = np.array([ar_plain.log_density(s) for s in seqs])
        np.testing.assert_allclose(fallback, expected, rtol=0, atol=1e-12)

    def test_harvest_fills_cache_from_one_call(self):
        target_nlp, _ = _pair(seed=4)
        calls = []

        def all_positions(seq):
            calls.append(tuple(seq))
            return [target_nlp(tuple(seq[:d])) for d in range(len(seq))]

        ar = AutoregressiveEnumerable(target_nlp, max_len=3, all_position_logprobs=all_positions)
        ar.harvest((2, 4, 1))
        self.assertEqual(len(calls), 1)
        for d in range(3):
            self.assertIn((2, 4, 1)[:d], ar._cache)  # every prefix cached from the single call
        # harvested cache serves log_density without further model calls
        plain = AutoregressiveEnumerable(target_nlp, max_len=3)
        self.assertAlmostEqual(ar.log_density((2, 4, 1)), plain.log_density((2, 4, 1)), places=12)

    def test_corpus_calibrated_envelope_matches_ancestral_given_same_contexts(self):
        # an iid model: ANY calibration contexts give the exact envelope, so corpus == ancestral
        rng = np.random.RandomState(5)
        logits = rng.randn(5)
        lp0 = logits - (np.max(logits) + math.log(np.sum(np.exp(logits - np.max(logits)))))

        def nlp(prefix):
            return np.arange(5), lp0

        ar = AutoregressiveEnumerable(nlp, max_len=3)
        corpus = [(0, 1, 2), (4, 4, 4), (2, 0, 3)]
        env_corpus = AREnvelopeIndex(ar, calibration_sequences=corpus)
        env_ancestral = AREnvelopeIndex(ar, n_paths=3, seed=0)
        self.assertEqual(env_corpus.total(), env_ancestral.total())
        self.assertEqual(int(env_corpus.total()), 125)  # exact for iid: 5**3

    def test_corpus_calibration_validates_length(self):
        target_nlp, _ = _pair(seed=6)
        ar = AutoregressiveEnumerable(target_nlp, max_len=4)
        with self.assertRaises(ValueError):
            AREnvelopeIndex(ar, calibration_sequences=[(1, 2)])  # too short for the model length
        with self.assertRaises(ValueError):
            AREnvelopeIndex(ar, calibration_sequences=[])


if __name__ == "__main__":
    unittest.main()
