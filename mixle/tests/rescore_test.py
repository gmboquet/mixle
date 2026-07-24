"""Speculative enumeration (RescoredIndex) + the teacher-forcing scoring contracts.

The economics under test: a cheap DRAFT model pays for the index build; the expensive TARGET is touched
only for returned sequences, via ONE batched teacher-forcing call (``score_sequences``). Verified on small
enumerable models where the target-exact truth is computable: top_k/slice return target-exact scores in
target order and match the target's own exact index when the window covers the draft/target disagreement;
the ``assumed_gap`` certificate is sound (certified results ARE the true top-k); ``log_density`` uses one
scorer call instead of a per-token walk when prefixes are uncached; ``harvest`` fills L cache entries from
one all-position call; corpus-calibrated envelopes equal ancestrally-calibrated ones given the same
contexts.

``RescoreCertificationSoundnessTest`` covers MXR-080-0233: the OLD certificate trusted the last pulled
candidate's own exact draft score as a bound on every unpulled candidate, which the draft/envelope
indexes' quantized (bucket-approximate, not exact) ordering never actually guaranteed.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import AREnvelopeIndex, AutoregressiveEnumerable, RescoredIndex, RescoreResult, SeekIndex
from mixle.enumeration.quantization.core import Quantizer


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
        lps = [lp for _s, lp in out.items]
        self.assertEqual(lps, sorted(lps, reverse=True))
        for seq, lp in out.items:
            self.assertAlmostEqual(lp, self.target.log_density(seq), places=12)
        self.assertEqual({s for s, _ in out.items}, {s for s, _ in brute[:10]})
        self.assertAlmostEqual(lps[-1], brute[9][1], places=12)

    def test_slice_is_target_ordered_and_scored(self):
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=300)
        out = ri.slice(5, 7)
        self.assertEqual(len(out.items), 7)
        lps = [lp for _s, lp in out.items]
        self.assertEqual(lps, sorted(lps, reverse=True))
        for seq, lp in out.items:
            self.assertAlmostEqual(lp, self.target.log_density(seq), places=12)

    def test_certificate_soundness_under_assumed_gap(self):
        # the draft's perturbation gives a TRUE global gap bound; compute it and certify against it
        true_gap = 0.0
        import itertools

        for seq in itertools.product(range(6), repeat=4):
            true_gap = max(true_gap, abs(self.target.log_density(seq) - self.draft.log_density(seq)))
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=250, assumed_gap=true_gap)
        out = ri.top_k(5)
        self.assertIsInstance(out, RescoreResult)
        self.assertIsNotNone(out.certified)
        if out.certified:  # a certified result must BE the true top-5 (as a set)
            true5 = {self.exact.unrank(i)[0] for i in range(5)}
            self.assertEqual({s for s, _ in out.items}, true5)
            self.assertIsNotNone(out.bound)  # a real proof always cites the bound it rests on
        self.assertLessEqual(out.gap, true_gap + 1e-9)  # observed gap never exceeds the true bound

    def test_certified_true_when_draft_exhausted(self):
        # window larger than the whole support: nothing unpulled exists, so the result is provably complete
        ri = RescoredIndex(SeekIndex(self.draft), self.target, rerank_window=3000, assumed_gap=1e6)
        out = ri.top_k(5)
        self.assertTrue(out.certified)
        self.assertIsNone(out.bound)  # exhaustion needs no bound: vacuously nothing is unpulled
        true5 = {self.exact.unrank(i)[0] for i in range(5)}
        self.assertEqual({s for s, _ in out.items}, true5)

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


class _FixedDraftIndex:
    """A minimal draft index over an explicit, hand-picked (seq, draft_log_prob) rank order.

    Deliberately NOT monotonic in exact draft score -- this is the point: SeekIndex/AREnvelopeIndex only
    guarantee ordering up to their quantized fine-bucket width (MXR-080-0233), so a real draft index can
    return a later-ranked item with a HIGHER exact draft score than an earlier one, as long as both share
    (or the later one falls in) a bucket at or after the earlier item's. Exposes ``.quantizer`` like the
    two accepted index types so :meth:`RescoredIndex._sound_draft_bound` can compute a real bound.
    """

    def __init__(self, ranked_items, quantizer):
        self._items = list(ranked_items)
        self.quantizer = quantizer

    def unrank(self, i):
        if i < 0 or i >= len(self._items):
            raise IndexError("rank %d beyond support" % i)
        return self._items[i]


class _NoQuantizerDraftIndex:
    """A conforming ``unrank``-only draft index that exposes no ``quantizer`` -- e.g. a bare user index."""

    def __init__(self, ranked_items):
        self._items = list(ranked_items)

    def unrank(self, i):
        if i < 0 or i >= len(self._items):
            raise IndexError("rank %d beyond support" % i)
        return self._items[i]


class RescoreCertificationSoundnessTest(unittest.TestCase):
    """Regression coverage for MXR-080-0233: draft rescoring must never issue a false top-k certificate.

    The audit's own reproduction: a pulled draft score of -2.01 must not certify top-1 when an unpulled
    candidate scores -2.0 under both draft AND target -- the two scores share one quantized fine bucket
    (verified below against the real Quantizer), so the draft index's ordering guarantee does not
    actually rule the unpulled candidate out, and a sound certificate must say so.
    """

    def setUp(self):
        self.quantizer = Quantizer(bin_width_bits=1.0, oversample=8)
        self.seq_pulled = (0,)
        self.seq_unpulled = (1,)
        # the OLD code's premise, made concrete: the item it pulls and certifies against has draft
        # score -2.01; the item it never looks at (rerank_window=0 pulls only rank 0) has draft AND
        # target score -2.0 -- strictly BETTER under both scorers, yet ranked after the pulled one.
        self.draft_scores = {self.seq_pulled: -2.01, self.seq_unpulled: -2.0}
        self.target_scores = {self.seq_pulled: -2.01, self.seq_unpulled: -2.0}
        # confirm the fixture actually reproduces the audit's premise before trusting any assertion
        # built on top of it: both scores must land in the SAME quantized fine bucket.
        self.assertEqual(self.quantizer.fine_bucket(-2.01), self.quantizer.fine_bucket(-2.0))

    def _target_scorer(self, seqs):
        return np.array([self.target_scores[tuple(s)] for s in seqs])

    def test_MXR_080_0233_quantized_bucket_tie_is_not_falsely_certified(self):
        draft = _FixedDraftIndex([(self.seq_pulled, self.draft_scores[self.seq_pulled])], self.quantizer)
        ri = RescoredIndex(draft, self._target_scorer, rerank_window=0, assumed_gap=0.0)
        out = ri.top_k(1)

        # what the OLD code computed and trusted as sound: edge_draft_lp was the pulled item's own
        # exact draft score (-2.01), and certified = (kth_target_lp >= edge_draft_lp + assumed_gap).
        old_edge_draft_lp = -2.01
        old_kth_target_lp = -2.01
        self.assertTrue(old_kth_target_lp >= old_edge_draft_lp + 0.0)  # the false certificate the bug issued

        # the NEW code must not reproduce it: it returns exactly what was pulled (window=0 pulls one
        # item) but must NOT certify that as a proven top-1, because -2.0 (same bucket, unpulled) could
        # legitimately outrank it.
        self.assertEqual(out.items, [(self.seq_pulled, -2.01)])
        self.assertIsNotNone(out.certified)
        self.assertFalse(out.certified)
        self.assertIsNotNone(out.bound)
        # the honest bound is the bucket's own best-case edge -- strictly looser than (never equal to)
        # the exact pulled score the old code used, and it is what correctly fails to clear -2.01.
        expected_bound = -float(self.quantizer.fine_bucket(-2.01)) * (1.0 / 8.0) * math.log(2.0)
        self.assertAlmostEqual(out.bound, expected_bound, places=12)
        self.assertGreater(out.bound, old_edge_draft_lp)
        self.assertFalse(old_kth_target_lp >= out.bound + 0.0)  # the bound the new code actually enforces

    def test_MXR_080_0233_same_scenario_via_slice(self):
        # the identical premise through slice(0, 1) instead of top_k(1) -- both entry points share the
        # same _certify path, so both must be immune to the false-certificate premise.
        draft = _FixedDraftIndex([(self.seq_pulled, self.draft_scores[self.seq_pulled])], self.quantizer)
        ri = RescoredIndex(draft, self._target_scorer, rerank_window=0, assumed_gap=0.0)
        out = ri.slice(0, 1)
        self.assertFalse(out.certified)
        self.assertIsNotNone(out.bound)

    def test_target_score_clearing_the_real_bound_still_certifies(self):
        # positive control: the fix is not simply "never certify", it is "certify only from a real
        # bound". The bound depends only on the PULLED boundary item's own bucket (it can never peek
        # at an unpulled item's actual score -- that is the entire point of a sound worst-case bound),
        # so to make certification succeed honestly, the boundary item's own TARGET score must clear
        # that bound directly: give it a target score set (deliberately) just above the same bucket
        # edge the adversarial test proved does NOT certify a plain draft-score tie.
        expected_bound = -float(self.quantizer.fine_bucket(-2.01)) * (1.0 / 8.0) * math.log(2.0)
        target_boundary = expected_bound + 1e-3  # comfortably clears the bound with assumed_gap=0
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01), (self.seq_unpulled, -2.0)], self.quantizer)
        scores = {self.seq_pulled: target_boundary, self.seq_unpulled: -2.0}
        ri = RescoredIndex(
            draft, lambda seqs: np.array([scores[tuple(s)] for s in seqs]), rerank_window=0, assumed_gap=0.0
        )
        out = ri.top_k(1)
        self.assertAlmostEqual(out.bound, expected_bound, places=12)
        self.assertTrue(out.certified)

    def test_uncertifiable_without_quantizer_returns_false_not_true(self):
        # a conforming (unrank-only) draft index that exposes no bound-supplying quantizer must never
        # be silently trusted as if it had -- certified must fail closed to False, with no bound cited.
        draft = _NoQuantizerDraftIndex([(self.seq_pulled, -2.01), (self.seq_unpulled, -2.0)])
        ri = RescoredIndex(draft, self._target_scorer, rerank_window=0, assumed_gap=0.0)
        out = ri.top_k(1)
        self.assertFalse(out.certified)
        self.assertIsNone(out.bound)

    def test_uncertified_without_assumed_gap_stays_none(self):
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01), (self.seq_unpulled, -2.0)], self.quantizer)
        ri = RescoredIndex(draft, self._target_scorer, rerank_window=0)  # no assumed_gap
        out = ri.top_k(1)
        self.assertIsNone(out.certified)
        self.assertIsNone(out.bound)

    def test_negative_rerank_window_rejected(self):
        # MXR-080-0233: a negative window used to make n = k + rerank_window collapse to <= 0, so the
        # pull loop touched nothing and top_k/slice still reported certified=True on zero items.
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01)], self.quantizer)
        with self.assertRaises(ValueError):
            RescoredIndex(draft, self._target_scorer, rerank_window=-1)
        with self.assertRaises(ValueError):
            RescoredIndex(draft, self._target_scorer, rerank_window=-64)

    def test_negative_assumed_gap_rejected(self):
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01)], self.quantizer)
        with self.assertRaises(ValueError):
            RescoredIndex(draft, self._target_scorer, assumed_gap=-1e-9)

    def test_nan_assumed_gap_rejected(self):
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01)], self.quantizer)
        with self.assertRaises(ValueError):
            RescoredIndex(draft, self._target_scorer, assumed_gap=float("nan"))

    def test_infinite_assumed_gap_rejected(self):
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01)], self.quantizer)
        with self.assertRaises(ValueError):
            RescoredIndex(draft, self._target_scorer, assumed_gap=float("inf"))

    def test_zero_assumed_gap_is_allowed(self):
        # 0.0 (a literal claim that draft == target exactly) is a valid, if aggressive, boundary value.
        draft = _FixedDraftIndex([(self.seq_pulled, -2.01)], self.quantizer)
        ri = RescoredIndex(draft, self._target_scorer, assumed_gap=0.0)
        self.assertEqual(ri.assumed_gap, 0.0)

    def test_empty_draft_index_is_vacuously_certified_only_with_assumed_gap(self):
        draft = _FixedDraftIndex([], self.quantizer)
        uncertified = RescoredIndex(draft, self._target_scorer, rerank_window=0)
        out = uncertified.top_k(1)
        self.assertEqual(out.items, [])
        self.assertIsNone(out.certified)  # no assumed_gap: never claim a verdict that was not requested

        certified = RescoredIndex(draft, self._target_scorer, rerank_window=0, assumed_gap=0.0)
        out2 = certified.top_k(1)
        self.assertEqual(out2.items, [])
        self.assertTrue(out2.certified)  # genuinely nothing exists to have missed
        self.assertIsNone(out2.bound)


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
