"""Count / threshold / unrank for arbitrary autoregressive (next_logprobs) models.

Verifies :class:`mixle.enumeration.AutoregressiveEnumerable` against brute-force enumeration of the whole
support: that ``unrank`` reproduces the exact descending-probability order (up to the documented intra-bucket
permutation, which vanishes as ``oversample`` grows), that ``count`` / ``threshold`` / ``mass_above`` match
the brute truth, that the shared ``seek``/``rank``/``cumulative``/``nucleus_size`` density-rank machinery
works on the adapter unchanged, that ``eos`` gives correct variable-length enumeration, and -- the whole point
-- that the number of distinct model queries is bounded by the distinct prefixes (<= V^(L-1)), not by the rank.
"""

import itertools
import math
import unittest

import numpy as np

from mixle.enumeration import AutoregressiveEnumerable, autoregressive_count_index
from mixle.enumeration.autoregressive import _ar_count_index_fast
from mixle.enumeration.quantization.core import Quantizer, count_budget_index


def _model(V, L, seed=0, scale=1.5):
    """A genuinely prefix-dependent autoregressive model: logits[depth, last_token] -> next-token logits."""
    rng = np.random.RandomState(seed)
    W = rng.randn(L, V, V) * scale

    def next_logprobs(prefix):
        d = len(prefix)
        last = prefix[-1] if prefix else 0
        logits = W[d, last]
        return list(enumerate(logits - _logsumexp(logits)))

    return next_logprobs


def _logsumexp(a):
    m = np.max(a)
    return m + math.log(np.sum(np.exp(a - m)))


def _brute(next_logprobs, L):
    out = []
    for seq in itertools.product(range(_VOCAB), repeat=L):
        lp, prefix = 0.0, ()
        for t in seq:
            lp += dict(next_logprobs(prefix))[t]
            prefix += (t,)
        out.append((tuple(seq), lp))
    return sorted(out, key=lambda u: -u[1])


_VOCAB, _LEN = 4, 3

_EOS = 2
_TERM_P = {None: [0.4, 0.3, 0.3], 0: [0.3, 0.3, 0.4], 1: [0.4, 0.2, 0.4], 2: [0.0, 0.0, 1.0]}


def _terminating_model():
    """3-symbol chain (0, 1, eos=_EOS); same shape as the P/nlp used inline by the existing terminating
    tests below. Eos is absorbing but -- like a real softmax -- assigns a nonzero (tiny, smoothed)
    probability to "continuing" past it, so a naive per-step-only check would find a token placed after
    eos locally available. Every state has a strictly positive direct transition to eos, so termination
    happens with probability 1 -- the total mass over all eos-terminated sequences is exactly 1.
    """
    logt = {k: np.log(np.array(v) + 1e-30) for k, v in _TERM_P.items()}

    def nlp(prefix):
        last = prefix[-1] if prefix else None
        return [(t, float(lp)) for t, lp in enumerate(logt[last])]

    return nlp


def _walk_lp(next_logprobs, seq):
    """Independent reference log-density: sum the per-step log-probs by directly walking ``next_logprobs``,
    the same accumulation :func:`_brute` uses -- so comparisons against it don't just check ``log_density``
    against itself.
    """
    lp, prefix = 0.0, ()
    for t in seq:
        lp += dict(next_logprobs(prefix))[t]
        prefix += (t,)
    return lp


class AutoregressiveEnumerableTest(unittest.TestCase):
    def setUp(self):
        self.next_logprobs = _model(_VOCAB, _LEN)
        self.brute = _brute(self.next_logprobs, _LEN)
        self.N = len(self.brute)

    def test_unrank_is_exact_descending_order(self):
        # With a fine bucket the intra-bucket ambiguity vanishes and unrank == brute order at every rank.
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=4096)
        got = [ar.unrank(i) for i in range(self.N)]
        self.assertEqual([s for s, _ in got], [s for s, _ in self.brute])
        for (s, lp), (bs, blp) in zip(got, self.brute):
            self.assertAlmostEqual(lp, blp, places=9)

    def test_unrank_is_always_a_permutation_with_exact_logprobs(self):
        # Even at a coarse bucket: every sequence appears exactly once, each with its EXACT log-prob, and the
        # ordering is non-increasing to within one bucket width (the count-DP contract).
        truth = dict(self.brute)
        for ov in (8, 64):
            ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=ov)
            got = [ar.unrank(i) for i in range(self.N)]
            self.assertEqual(sorted(s for s, _ in got), sorted(s for s, _ in self.brute))
            for s, lp in got:
                self.assertAlmostEqual(lp, truth[s], places=9)
            # A joint bucket sums L floor-quantized step buckets, so adjacent ranks may invert by up to the
            # accumulated rounding L/oversample bits (in nats); that's the count-DP contract, not an error.
            tol_nats = _LEN * (1.0 / ov) * math.log(2) + 1e-9
            self.assertTrue(all(got[i][1] >= got[i + 1][1] - tol_nats for i in range(self.N - 1)))

    def test_count_matches_brute(self):
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=64)
        for rank in (1, 10, 20, 40, self.N):
            tau = self.brute[rank - 1][1]
            brute_count = sum(1 for _, lp in self.brute if lp >= tau - 1e-12)
            self.assertEqual(ar.count(tau), brute_count)

    def test_threshold_is_the_kth_log_prob(self):
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=4096)
        for rank in (1, 5, 21, self.N):
            self.assertAlmostEqual(ar.threshold(rank), self.brute[rank - 1][1], places=9)

    def test_top_k_matches_brute(self):
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN)
        self.assertEqual([s for s, _ in ar.top_k(8)], [s for s, _ in self.brute[:8]])

    def test_mass_above_brackets_true_head_mass(self):
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=64)
        for rank in (1, 10, 30):
            tau = self.brute[rank - 1][1]
            true_mass = sum(math.exp(lp) for _, lp in self.brute if lp >= tau - 1e-12)
            lo, hi = ar.mass_above(tau)
            self.assertLessEqual(lo - 1e-9, true_mass)
            self.assertGreaterEqual(hi + 1e-9, true_mass)

    def test_density_rank_delegations_agree_with_brute(self):
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=64)
        # seek(i).value is a within-bucket member at rank i; its log-prob equals the brute i-th log-prob.
        for i in (0, 3, 5):
            r = ar.seek(i)
            self.assertAlmostEqual(ar.log_density(r.value), self.brute[i][1], places=9)
        # rank(value) of a known sequence is consistent (0-based count of strictly-more-probable outcomes).
        self.assertEqual(ar.rank(self.brute[0][0]).rank, 0)
        # nucleus_size returns a size bracket around the true minimal >=p set.
        res = ar.nucleus_size(0.5)
        cum, k = 0.0, 0
        for _, lp in self.brute:
            cum += math.exp(lp)
            k += 1
            if cum >= 0.5:
                break
        self.assertLessEqual(res.size_lower, k)
        self.assertGreaterEqual(res.size_upper, k)

    def test_nucleus_size_matches_brute_across_targets(self):
        # nucleus_size delegated to the generic density_rank.count_dp_top_p, which assumes its exact
        # per-item mass histogram and the count index share one bucket numbering -- true for
        # Composite/Record/Sequence, false here (quantized_count_index buckets a sequence by the SUM of
        # per-step floors, not the floor of the exact total; floor(a)+floor(b) <= floor(a+b), so the two
        # numbering schemes diverge). This model/seed/scale (unlike this file's default _VOCAB=4/_LEN=3
        # fixture, whose particular rounding happens not to expose the gap at p=0.5) reproducibly hit it:
        # nucleus_size(0.5) returned size_lower=size_upper=0 against a brute-force truth of 5.
        vocab, length = 5, 3
        rng = np.random.RandomState(0)
        table = rng.randn(length, vocab, vocab) * 1.5

        def next_logprobs(prefix):
            depth = len(prefix)
            last = prefix[-1] if prefix else 0
            logits = table[depth, last]
            return list(enumerate(logits - _logsumexp(logits)))

        brute = []
        for seq in itertools.product(range(vocab), repeat=length):
            lp, prefix = 0.0, ()
            for t in seq:
                lp += dict(next_logprobs(prefix))[t]
                prefix += (t,)
            brute.append((tuple(seq), lp))
        brute.sort(key=lambda u: -u[1])

        ar = AutoregressiveEnumerable(next_logprobs, max_len=length, oversample=64)
        for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0):
            cum, k = 0.0, 0
            for _, lp in brute:
                cum += math.exp(lp)
                k += 1
                if cum >= p - 1e-9:
                    break
            res = ar.nucleus_size(p)
            self.assertLessEqual(res.size_lower, k, msg=f"p={p}")
            if not res.truncated:
                self.assertGreaterEqual(res.size_upper, k, msg=f"p={p}")

    def test_nucleus_size_forwards_instance_oversample(self):
        # nucleus_size used to call density_rank.count_dp_top_p(self, p) with no oversample/bin_width_bits
        # forwarded, so the instance's own oversample was silently ignored -- the returned
        # CountDPTopPResult.oversample field (and the bracket's tightness) never reflected it.
        ar_coarse = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=8)
        ar_fine = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=4096)
        res_coarse = ar_coarse.nucleus_size(0.5)
        res_fine = ar_fine.nucleus_size(0.5)
        self.assertEqual(res_coarse.oversample, 8)
        self.assertEqual(res_fine.oversample, 4096)
        self.assertGreaterEqual(
            res_coarse.size_upper - res_coarse.size_lower, res_fine.size_upper - res_fine.size_lower
        )

    def test_plugs_into_core_count_budget_driver(self):
        # The adapter is a drop-in for the existing count-budget driver (not just its own convenience methods).
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN)
        index = count_budget_index(ar, budget_bits=4.0, oversample=4096)  # fine bucket -> exact order
        self.assertGreaterEqual(len(index), 16)
        for i in range(16):
            value, lp = index.get(i)
            self.assertEqual(value, self.brute[i][0])
            self.assertAlmostEqual(lp, self.brute[i][1], places=9)

    def test_forward_passes_bounded_by_distinct_prefixes_not_rank(self):
        # The point of count+unrank: distinct model queries <= distinct prefixes (1 + V + ... + V^(L-1)),
        # far fewer than the N sequences, and reused across deepening / log-density calls (memoized).
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=64)
        q = Quantizer(oversample=64)
        ar.quantized_count_index(q, max_fine_bucket=10**6)  # full support
        distinct_prefixes = sum(_VOCAB**d for d in range(_LEN))  # 1 + 4 + 16 = 21
        self.assertLessEqual(len(ar._cache), distinct_prefixes)
        self.assertLess(len(ar._cache), self.N)  # fewer queries than sequences

    def test_terminating_model_enumerates_only_its_support(self):
        # A terminating model's support is ONLY eos-terminated sequences (any length, bounded by probability,
        # NOT by a length cap). Verify count / unrank / top_k over exactly that support vs brute force.
        eos = 2
        P = {None: [0.4, 0.3, 0.3], 0: [0.3, 0.3, 0.4], 1: [0.4, 0.2, 0.4], 2: [0.0, 0.0, 1.0]}
        logt = {k: np.log(np.array(v) + 1e-30) for k, v in P.items()}

        def nlp(prefix):
            last = prefix[-1] if prefix else None
            return [(t, float(lp)) for t, lp in enumerate(logt[last])]

        def brute_terminating(max_body=14):
            out = {}  # all (body of 0/1s) + eos -- the entire terminating support up to max_body
            for body_len in range(max_body + 1):
                for body in itertools.product((0, 1), repeat=body_len):
                    seq, lp, prefix = body + (eos,), 0.0, ()
                    for t in seq:
                        lp += dict(nlp(prefix))[t]
                        prefix += (t,)
                    out[seq] = lp
            return sorted(out.items(), key=lambda u: -u[1])

        ar = AutoregressiveEnumerable(nlp, eos=eos, oversample=4096, max_depth=40)
        bt = brute_terminating()
        # top-k are the most probable COMPLETE sentences, in order (each ends in eos, varying length)
        for (s, lp), (bs, blp) in zip(ar.top_k(10), bt[:10]):
            self.assertEqual(s, bs)
            self.assertAlmostEqual(lp, blp, places=8)
        # every unranked sequence is in the support (ends in eos) and matches the brute order -- no truncations
        for i in range(40):
            seq, lp = ar.unrank(i)
            self.assertEqual(seq[-1], eos)
            self.assertEqual(seq, bt[i][0])
            self.assertAlmostEqual(lp, bt[i][1], places=8)
        # count above a threshold matches the brute terminating-support count (at the same quantization)
        q = ar._quantizer()
        fb_tau = q.fine_bucket(bt[20][1])
        self.assertEqual(ar.count(bt[20][1]), sum(1 for _, lp in bt if q.fine_bucket(lp) <= fb_tau))

    def test_terminating_support_excludes_truncations(self):
        # The index over a terminating model must NEVER contain a non-eos (truncated) sequence, at any rank.
        eos = 2
        P = {None: [0.4, 0.3, 0.3], 0: [0.3, 0.3, 0.4], 1: [0.4, 0.2, 0.4], 2: [0.0, 0.0, 1.0]}
        logt = {k: np.log(np.array(v) + 1e-30) for k, v in P.items()}

        def nlp(prefix):
            last = prefix[-1] if prefix else None
            return [(t, float(lp)) for t, lp in enumerate(logt[last])]

        idx = AutoregressiveEnumerable(nlp, eos=eos, oversample=64, max_depth=20).budget_index(budget_bits=10.0)
        self.assertGreater(len(idx), 100)
        for i in range(len(idx)):
            seq, _ = idx.get(i)
            self.assertEqual(seq[-1], eos)  # in the support: terminated, never a length-capped truncation

    def test_constructor_requires_max_len_or_eos(self):
        with self.assertRaises(ValueError):
            AutoregressiveEnumerable(self.next_logprobs)  # neither a length nor a terminator -> no support

    def test_raw_count_index_function(self):
        # The underlying tree-recursive builder returns a count index whose total == full support size.
        q = Quantizer(oversample=64)
        index, truncated = autoregressive_count_index(
            lambda p: sorted(self.next_logprobs(p), key=lambda u: -u[1]), (), _LEN, q, 10**6
        )
        self.assertFalse(truncated)
        self.assertEqual(index.total(), self.N)

    def test_fast_path_matches_reference_bit_for_bit(self):
        # The numpy/int64 fast path is identical to the arbitrary-precision Python reference: same histogram
        # counts and the same unranked value at every (fine bucket, offset).
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=16)
        self.assertTrue(ar._use_fast())
        q = Quantizer(oversample=16)
        ref, rt = autoregressive_count_index(
            lambda p: sorted(self.next_logprobs(p), key=lambda u: -u[1]), (), _LEN, q, 10**9
        )
        fast, ft = _ar_count_index_fast(ar._steps_np, (), _LEN, q, 10**9)
        self.assertEqual((ref.total(), rt), (fast.total(), ft))
        for fb in range(fast.hist.base, fast.hist.base + len(fast.hist.data)):
            self.assertEqual(ref.hist.count_at(fb), fast.hist.count_at(fb))
            for off in range(fast.hist.count_at(fb)):
                self.assertEqual(ref.get_in_bucket(fb, off), fast.get_in_bucket(fb, off))

    def test_batched_prefetch_matches_unbatched(self):
        # Batched forward prefetch yields an identical index (same forwards, same unrank) to one-at-a-time.
        def batch(prefixes):
            return [self.next_logprobs(p) for p in prefixes]

        plain = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, oversample=64)
        batched = AutoregressiveEnumerable(
            self.next_logprobs, max_len=_LEN, oversample=64, batch_next_logprobs=batch, batch_size=4
        )
        self.assertEqual([plain.unrank(i) for i in range(self.N)], [batched.unrank(i) for i in range(self.N)])
        self.assertEqual(len(plain._cache), len(batched._cache))

    # -- MXR-080-0220: log_density/score_sequences used to check only per-step local availability, never
    # the sequence's GLOBAL structure -- so sequences outside the declared support scored as finite. -----

    def test_log_density_rejects_wrong_length_for_fixed_model(self):
        # Reproduce the audit's exact fixed-length case: a length-2 model used to score (), (0,), and
        # (0, 0, 0) all as finite. max_len=2 is deliberately shorter than self.next_logprobs's own depth
        # capacity (built for _LEN=3), so the old code could walk right past the declared length without
        # erroring -- exactly the shape that let the bug hide.
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=2)
        self.assertEqual(ar.log_density(()), float("-inf"))
        self.assertEqual(ar.log_density((0,)), float("-inf"))
        self.assertEqual(ar.log_density((0, 0, 0)), float("-inf"))
        # Negative control: a genuine length-2 sequence is unaffected -- same finite value as a direct walk.
        valid = (1, 2)
        expected = _walk_lp(self.next_logprobs, valid)
        self.assertTrue(math.isfinite(expected))
        self.assertAlmostEqual(ar.log_density(valid), expected, places=12)

    def test_log_density_rejects_unterminated_and_misplaced_eos(self):
        # Reproduce the audit's exact terminating case: the empty sequence, an unterminated prefix, and a
        # real token placed AFTER eos all used to score as finite.
        nlp = _terminating_model()
        ar = AutoregressiveEnumerable(nlp, eos=_EOS, max_depth=40)
        self.assertEqual(ar.log_density(()), float("-inf"))  # never reaches eos at all
        self.assertEqual(ar.log_density((0, 1, 0)), float("-inf"))  # unterminated prefix
        self.assertEqual(ar.log_density((_EOS, 0)), float("-inf"))  # a real token after eos
        # Negative control: a genuine eos-terminated sequence is unaffected.
        valid = (0, 1, _EOS)
        expected = _walk_lp(nlp, valid)
        self.assertTrue(math.isfinite(expected))
        self.assertAlmostEqual(ar.log_density(valid), expected, places=12)

    def test_log_density_rejects_tokens_after_eos_even_when_locally_available(self):
        # "No tokens after eos" must hold structurally, not just per-step: this fixture's eos state
        # assigns a tiny but nonzero (smoothed) probability to 0/1 "continuing" past it (see
        # _terminating_model's docstring), so the OLD per-step-only check would find such a continuation
        # locally available and happily sum it into a finite score.
        nlp = _terminating_model()
        ar = AutoregressiveEnumerable(nlp, eos=_EOS, max_depth=40)
        self.assertEqual(ar.log_density((0, _EOS, 0)), float("-inf"))  # eos mid-sequence, real token after
        self.assertEqual(ar.log_density((_EOS, _EOS)), float("-inf"))  # eos immediately repeated
        self.assertEqual(ar.log_density((0, 1, _EOS, 1)), float("-inf"))  # longer prefix before misplaced eos

    def test_log_density_enforces_terminating_depth_cap(self):
        # eos + an explicit max_len is a terminating model with a HARD depth cap ("the model doesn't
        # define further generation" beyond it, per the class docstring) -- a correctly eos-terminated
        # sequence that exceeds the cap must still score -inf, even though it is otherwise well-formed.
        nlp = _terminating_model()
        ar = AutoregressiveEnumerable(nlp, eos=_EOS, max_len=2)
        too_long = (0, 1, _EOS)  # properly terminated, but length 3 > cap 2
        self.assertEqual(ar.log_density(too_long), float("-inf"))
        # Negative control: within the cap, genuine sequences are unaffected.
        valid = (0, _EOS)
        self.assertAlmostEqual(ar.log_density(valid), _walk_lp(nlp, valid), places=12)
        immediate = (_EOS,)
        self.assertAlmostEqual(ar.log_density(immediate), _walk_lp(nlp, immediate), places=12)

    def test_score_sequences_batch_scorer_rejects_invalid_before_calling_scorer(self):
        # Batch scoring used to bypass validation entirely, trusting the external scorer with anything.
        # Structurally invalid sequences must now score -inf WITHOUT ever reaching that scorer.
        calls = []

        def scorer(seqs):
            calls.append([tuple(s) for s in seqs])
            return np.full(len(seqs), -777.0)  # a fake sentinel: proves whether it round-tripped or not

        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, batch_score_sequences=scorer)
        valid = self.brute[0][0]
        invalid_short = valid[:-1]
        invalid_long = valid + (0,)
        out = ar.score_sequences([invalid_short, valid, invalid_long])
        self.assertEqual(out[0], float("-inf"))
        self.assertEqual(out[1], -777.0)
        self.assertEqual(out[2], float("-inf"))
        self.assertEqual(calls, [[valid]])  # the scorer only ever saw the structurally valid entry

    def test_log_density_batch_scorer_shortcut_rejects_invalid_before_calling_scorer(self):
        # log_density has its OWN batch_score_sequences shortcut (for uncached prefixes) that used to
        # bypass validation the same way -- check it directly, not just through score_sequences.
        calls = []

        def scorer(seqs):
            calls.append([tuple(s) for s in seqs])
            return np.full(len(seqs), -777.0)

        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN, batch_score_sequences=scorer)
        # No prefixes are cached yet, so the un-fixed code would have taken the batch-scorer shortcut.
        self.assertEqual(ar.log_density((0, 0)), float("-inf"))  # wrong length: short-circuits first
        self.assertEqual(calls, [])  # the external scorer was never reached

    def test_total_probability_mass_sums_to_one_fixed_length(self):
        # The ultimate soundness check: log_density integrates to (approximately) 1 over the TRUE support,
        # and structurally-invalid (wrong-length) sequences contribute exactly zero -- no leaked mass.
        ar = AutoregressiveEnumerable(self.next_logprobs, max_len=_LEN)
        valid_mass = sum(math.exp(ar.log_density(s)) for s, _ in self.brute)
        self.assertAlmostEqual(valid_mass, 1.0, places=6)
        garbage = [(), (0,), (0, 0), (0, 0, 0, 0), (1, 2, 3, 0)]
        for seq in garbage:
            self.assertEqual(ar.log_density(seq), float("-inf"))
        leaked = sum(math.exp(ar.log_density(s)) for s in garbage)
        self.assertEqual(leaked, 0.0)

    def test_total_probability_mass_sums_to_one_terminating(self):
        # Same soundness check for a terminating model: eos is absorbing and reached with probability 1
        # from every state, so summing exp(log_density) over the (deep-enough) enumerated support
        # approaches 1 from below -- a partial sum over a subset of the support can only fall short (the
        # unenumerated deep tail), never exceed the true total.
        nlp = _terminating_model()
        ar = AutoregressiveEnumerable(nlp, eos=_EOS, max_depth=40)
        sequences = [body + (_EOS,) for body_len in range(15) for body in itertools.product((0, 1), repeat=body_len)]
        total = sum(math.exp(ar.log_density(s)) for s in sequences)
        self.assertLessEqual(total, 1.0 + 1e-9)
        self.assertGreaterEqual(total, 1.0 - 0.01)
        self.assertEqual(ar.log_density(()), float("-inf"))
        self.assertEqual(ar.log_density((0, 1)), float("-inf"))  # unterminated, must not count toward mass


if __name__ == "__main__":
    unittest.main()
