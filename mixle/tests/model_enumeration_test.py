"""Generic model-output enumeration (neural nets / transformers / any scoring callable).

Verifies that best_first_decode enumerates an autoregressive model's sequences in EXACT descending
total-log-probability order (against brute-force enumeration of the whole support), that beam_search returns
valid descending-scored sequences, that top_k_scored picks the true top-k of a finite candidate set, and that
an admissible heuristic does not change the exact ordering.
"""

import math
import unittest

import numpy as np

from mixle.enumeration.model_enumeration import (
    beam_search,
    best_first,
    best_first_decode,
    quantized_best_first_decode,
    top_k_scored,
)

# a tiny autoregressive model over vocab {0,1,2} with EOS=2: next-token log-probs depend on the last token.
_EOS = 2
_RNG = np.random.RandomState(0)
_TRANS = {None: np.array([0.5, 0.4, 0.1]), 0: np.array([0.3, 0.5, 0.2]), 1: np.array([0.6, 0.2, 0.2])}
_LOGT = {k: np.log(v) for k, v in _TRANS.items()}


def _next_logprobs(prefix):
    last = prefix[-1] if prefix else None
    return list(enumerate(_LOGT[last]))


def _brute_force(max_len):
    """All complete sequences (ending in EOS, or reaching max_len) with total log-prob."""
    out = []

    def rec(prefix, lp):
        if (prefix and prefix[-1] == _EOS) or len(prefix) >= max_len:
            out.append((tuple(prefix), lp))
            return
        for tok, tlp in _next_logprobs(tuple(prefix)):
            rec(prefix + [tok], lp + tlp)

    rec([], 0.0)
    return sorted(out, key=lambda u: -u[1])


class ModelEnumerationTestCase(unittest.TestCase):
    def test_best_first_decode_is_exact(self):
        max_len = 5
        brute = _brute_force(max_len)
        mine = list(best_first_decode(_next_logprobs, eos=_EOS, max_len=max_len))
        self.assertEqual(len(mine), len(brute))
        # exact descending order and same (sequence, logprob) multiset
        np.testing.assert_allclose([lp for _, lp in mine], [lp for _, lp in brute], atol=1e-9)
        self.assertEqual({s for s, _ in mine}, {s for s, _ in brute})
        lps = [lp for _, lp in mine]
        self.assertTrue(all(lps[i] >= lps[i + 1] - 1e-12 for i in range(len(lps) - 1)))
        # probabilities of all complete sequences sum to 1 (proper model)
        self.assertAlmostEqual(sum(math.exp(lp) for _, lp in mine), 1.0, places=9)

    def test_lazy_top_k_matches_prefix(self):
        max_len = 8  # larger support; only pull the top 5 lazily
        top5 = list(best_first_decode(_next_logprobs, eos=_EOS, max_len=max_len, max_results=5))
        self.assertEqual(len(top5), 5)
        # compare by log-prob (sequences can tie -- e.g. (2,) and (0,2) both have probability 0.1)
        brute_top5 = _brute_force(max_len)[:5]
        np.testing.assert_allclose([lp for _, lp in top5], [lp for _, lp in brute_top5], atol=1e-9)

    def test_admissible_heuristic_preserves_order(self):
        max_len = 5
        # h=0 is the trivially-admissible bound (remaining log-prob is always <= 0); passing it explicitly
        # must match the default. (A remaining_steps*best_step bound would be INADMISSIBLE here because an
        # early EOS completes with fewer, less-negative steps.)
        with_h = list(best_first_decode(_next_logprobs, eos=_EOS, max_len=max_len, heuristic=lambda prefix: 0.0))
        without_h = list(best_first_decode(_next_logprobs, eos=_EOS, max_len=max_len))
        np.testing.assert_allclose([lp for _, lp in with_h], [lp for _, lp in without_h], atol=1e-9)

    def test_beam_search_returns_valid_descending(self):
        res = beam_search(_next_logprobs, beam_width=4, eos=_EOS, max_len=6, num_results=3)
        self.assertEqual(len(res), 3)
        lps = [lp for _, lp in res]
        self.assertTrue(all(lps[i] >= lps[i + 1] for i in range(len(lps) - 1)))
        # the top beam result should match the exact best
        exact_best = next(best_first_decode(_next_logprobs, eos=_EOS, max_len=6))
        self.assertEqual(res[0][0], exact_best[0])

    def test_top_k_scored(self):
        rng = np.random.RandomState(1)
        logits = rng.randn(20)
        labels = list(range(20))
        got = top_k_scored(labels, score=lambda c: logits[c], k=5)
        want = sorted(labels, key=lambda c: -logits[c])[:5]
        self.assertEqual([c for c, _ in got], want)
        # k=None returns all sorted
        self.assertEqual(len(top_k_scored(labels, score=lambda c: logits[c])), 20)

    def test_generic_best_first_over_a_grid(self):
        # a non-sequence example: enumerate (i, j) cells in descending score
        scores = {(i, j): -(i * i + j * j) for i in range(3) for j in range(3)}
        results = list(
            best_first(
                start=(0, 0),
                successors=lambda s: [(s[0] + 1, s[1]), (s[0], s[1] + 1)] if s[0] < 2 and s[1] < 2 else [],
                is_goal=lambda s: s[0] == 2 or s[1] == 2,
                score=lambda s: scores[s],
            )
        )
        self.assertTrue(results)
        sc = [v for _, v in results]
        self.assertTrue(all(sc[i] >= sc[i + 1] for i in range(len(sc) - 1)))


class _PeakedModel:
    """A peaked autoregressive model over vocab 0..V-1 (EOS=V-1): most mass on a few tokens per step.

    Counts forward calls so the tests can show the nucleus-pruning / batching speedup.
    """

    def __init__(self, vocab=10, seed=0):
        self.vocab = vocab
        self.eos = vocab - 1
        rng = np.random.RandomState(seed)
        # sharp logits per last-token context (peaked), normalized to log-probs
        self.logp = {}
        for ctx in [None] + list(range(vocab - 1)):
            z = rng.randn(vocab) * 4.0
            self.logp[ctx] = z - np.log(np.exp(z).sum())
        self.calls = 0
        self.batch_calls = 0

    def next_logprobs(self, prefix):
        self.calls += 1
        return list(enumerate(self.logp[prefix[-1] if prefix else None]))

    def batch_next_logprobs(self, prefixes):
        self.batch_calls += 1
        self.calls += len(prefixes)
        return [list(enumerate(self.logp[p[-1] if p else None])) for p in prefixes]


class QuantizedDecodeTestCase(unittest.TestCase):
    def test_matches_exact_without_pruning(self):
        m = _PeakedModel(vocab=4, seed=1)
        exact = list(best_first_decode(m.next_logprobs, eos=m.eos, max_len=5))
        # no pruning + fine buckets -> same descending log-probs (tie-robust)
        quant = list(quantized_best_first_decode(m.next_logprobs, eos=m.eos, max_len=5, bucket_bits=20, batch_size=1))
        self.assertEqual(len(quant), len(exact))
        np.testing.assert_allclose([lp for _, lp in quant], [lp for _, lp in exact], atol=1e-9)

    def test_nucleus_pruning_covers_mass_and_cuts_work(self):
        m = _PeakedModel(vocab=12, seed=2)
        full = _PeakedModel(vocab=12, seed=2)
        # exact top-20
        exact_top = list(best_first_decode(full.next_logprobs, eos=full.eos, max_len=8, max_results=20))
        # top_p nucleus pruning
        pruned = list(quantized_best_first_decode(m.next_logprobs, eos=m.eos, max_len=8, top_p=0.95, max_results=20))
        self.assertEqual(len(pruned), 20)
        # the pruned top sequence equals the exact best; pruning used far fewer model calls
        self.assertEqual(pruned[0][0], exact_top[0][0])
        self.assertLess(m.calls, full.calls)
        # pruned results are valid descending probabilities
        lps = [lp for _, lp in pruned]
        self.assertTrue(all(lps[i] >= lps[i + 1] - 1e-9 for i in range(len(lps) - 1)))

    def test_batched_scoring_matches_and_batches(self):
        per = _PeakedModel(vocab=10, seed=3)
        bat = _PeakedModel(vocab=10, seed=3)
        # coarse buckets group near-equal-score prefixes so the batched path expands several per forward call
        a = list(
            quantized_best_first_decode(
                per.next_logprobs, eos=per.eos, max_len=6, top_k=4, bucket_bits=2, max_results=10
            )
        )
        b = list(
            quantized_best_first_decode(
                batch_next_logprobs=bat.batch_next_logprobs,
                eos=bat.eos,
                max_len=6,
                top_k=4,
                bucket_bits=2,
                batch_size=32,
                max_results=10,
            )
        )
        # same set of results (within a bucket the yield order may differ by batch size, so compare sorted)
        np.testing.assert_allclose(sorted(lp for _, lp in a), sorted(lp for _, lp in b), atol=1e-9)
        # the batched path scored more prefixes than it made forward calls -> it really batched
        self.assertGreater(bat.calls, bat.batch_calls)

    def test_min_mass_early_stop(self):
        m = _PeakedModel(vocab=8, seed=4)
        got = list(quantized_best_first_decode(m.next_logprobs, eos=m.eos, max_len=6, top_p=0.99, min_mass=0.5))
        self.assertGreaterEqual(sum(math.exp(lp) for _, lp in got), 0.5)
        # stopped early: covered ~0.5, not the whole (near-1) support
        self.assertLess(sum(math.exp(lp) for _, lp in got), 0.95)


if __name__ == "__main__":
    unittest.main()
