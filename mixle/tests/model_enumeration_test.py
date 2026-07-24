"""Generic model-output enumeration (neural nets / transformers / any scoring callable).

Verifies that best_first_decode enumerates an autoregressive model's sequences in EXACT descending
total-log-probability order (against brute-force enumeration of the whole support), that beam_search returns
valid descending-scored sequences, that top_k_scored picks the true top-k of a finite candidate set, and that
an admissible heuristic does not change the exact ordering.
"""

import math
import threading
import unittest

import numpy as np

from mixle.enumeration.model_enumeration import (
    _prune_step,
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


class ScoreContractValidationTestCase(unittest.TestCase):
    """MXR-080-0226: best_first_decode's exactness proof holds only because every continuation's
    log-probability is <= 0 (so a prefix's score never increases as it is extended -- the property that
    makes "already popped" provably outrank "not yet generated"). A positive or NaN score from a
    caller's next_logprobs silently breaks that proof while the API keeps calling its output exact, so
    both must be rejected loudly instead of propagated into the search.
    """

    def test_positive_continuation_score_is_rejected(self):
        def bad_next_logprobs(prefix):
            if not prefix:
                return [(0, -0.1), (1, 0.2)]  # token 1's log-prob is invalid: a probability > 1
            return [(2, 0.0)]

        with self.assertRaises(ValueError):
            list(best_first_decode(bad_next_logprobs, eos=2, max_len=4))

    def test_nan_continuation_score_is_rejected(self):
        def bad_next_logprobs(prefix):
            if not prefix:
                return [(0, -0.1), (1, float("nan"))]
            return [(2, 0.0)]

        with self.assertRaises(ValueError):
            list(best_first_decode(bad_next_logprobs, eos=2, max_len=4))

    def test_negative_infinity_continuation_score_is_accepted(self):
        # -inf is the standard "impossible token" sentinel used elsewhere in this codebase (e.g.
        # quantization.core.Quantizer.bits) and cannot break best-first's monotonicity -- it must NOT be
        # rejected the way NaN/positive scores are.
        def next_logprobs(prefix):
            if not prefix:
                return [(0, math.log(0.5)), (1, -math.inf), (2, math.log(0.5))]
            return [(2, 0.0)]

        results = list(best_first_decode(next_logprobs, eos=2, max_len=4))
        self.assertIn((2,), [seq for seq, _ in results])

    def test_valid_finite_nonpositive_scores_are_unaffected(self):
        # Negative control: the existing hand-verified fixture (all step log-probs <= 0) is unaffected
        # by score validation -- still the exact descending-probability enumeration.
        max_len = 3
        brute = _brute_force(max_len)
        mine = list(best_first_decode(_next_logprobs, eos=_EOS, max_len=max_len))
        self.assertEqual(len(mine), len(brute))
        np.testing.assert_allclose([lp for _, lp in mine], [lp for _, lp in brute], atol=1e-9)

    def test_top_k_scored_rejects_invalid_k(self):
        labels = list(range(5))
        for bad_k in (0, -1, -5):
            with self.assertRaises(ValueError):
                top_k_scored(labels, score=lambda c: float(c), k=bad_k)

    def test_top_k_scored_valid_k_is_unaffected(self):
        rng = np.random.RandomState(2)
        logits = rng.randn(10)
        labels = list(range(10))
        got = top_k_scored(labels, score=lambda c: logits[c], k=3)
        want = sorted(labels, key=lambda c: -logits[c])[:3]
        self.assertEqual([c for c, _ in got], want)


class PruningContractValidationTestCase(unittest.TestCase):
    """MXR-080-0226: quantized_best_first_decode's top-k / top-p (nucleus) pruning (_prune_step) must
    reject invalid k/p bounds and malformed probability mass instead of silently mis-pruning a step --
    a NaN log-prob compares False against everything, so an unvalidated nucleus loop would never stop
    early and just keep whatever is left, and an unvalidated positive log-prob breaks the same
    monotonicity best_first_decode relies on, since pruned values are exactly what gets pushed back onto
    the quantized frontier.
    """

    def test_prune_step_rejects_invalid_top_k(self):
        items = [(0, math.log(0.5)), (1, math.log(0.5))]
        for bad_k in (0, -1, -100):
            with self.assertRaises(ValueError):
                _prune_step(items, top_k=bad_k, top_p=None)

    def test_prune_step_rejects_invalid_top_p(self):
        items = [(0, math.log(0.5)), (1, math.log(0.5))]
        for bad_p in (0.0, -0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                _prune_step(items, top_k=None, top_p=bad_p)

    def test_prune_step_rejects_nan_probability_mass(self):
        items = [(0, math.log(0.5)), (1, float("nan"))]
        with self.assertRaises(ValueError):
            _prune_step(items, top_k=None, top_p=0.9)

    def test_prune_step_rejects_positive_probability_mass(self):
        items = [(0, math.log(0.5)), (1, 0.3)]  # 0.3 is an invalid log-prob (implies probability > 1)
        with self.assertRaises(ValueError):
            _prune_step(items, top_k=None, top_p=0.9)

    def test_prune_step_valid_bounds_prune_correctly(self):
        # Negative control, hand-computable: three tokens with probabilities 0.5, 0.3, 0.2.
        items = [(0, math.log(0.5)), (1, math.log(0.3)), (2, math.log(0.2))]
        top2 = _prune_step(items, top_k=2, top_p=None)
        self.assertEqual([t for t, _ in top2], [0, 1])
        # top_p=0.7: the smallest prefix mass reaching >= 0.7 is 0.5+0.3=0.8 (0.5 alone falls short).
        nucleus = _prune_step(items, top_k=None, top_p=0.7)
        self.assertEqual([t for t, _ in nucleus], [0, 1])
        # top_p=1.0 is a valid, non-restrictive bound: everything is kept.
        everything = _prune_step(items, top_k=None, top_p=1.0)
        self.assertEqual([t for t, _ in everything], [0, 1, 2])

    def test_quantized_decode_with_valid_bounds_matches_brute_force(self):
        # Negative control, end-to-end through the public entry point: non-restrictive but VALID top_k /
        # top_p bounds must not change the exact result on the existing hand-verified fixture.
        max_len = 3
        brute = _brute_force(max_len)
        mine = list(
            quantized_best_first_decode(
                _next_logprobs, eos=_EOS, max_len=max_len, top_k=3, top_p=1.0, bucket_bits=20, batch_size=1
            )
        )
        self.assertEqual(len(mine), len(brute))
        np.testing.assert_allclose(sorted(lp for _, lp in mine), sorted(lp for _, lp in brute), atol=1e-9)


class QuantizedDecodeControlValidationTestCase(unittest.TestCase):
    """MXR-080-0227: quantized_best_first_decode's controls must be validated before the search runs.

    A negative batch_size previously took an empty slice of a non-empty live bucket without consuming
    it, so the frontier never shrank and the search never terminated. A batched callback that returned
    fewer step tables than prefixes requested was paired up with zip(), which silently stops at the
    shorter list -- the reproduced case (one live prefix, an under-returning callback) came back as an
    empty enumeration with no error at all, even though a real prefix was live and dropped.
    """

    def test_negative_batch_size_is_rejected_not_hung(self):
        # Regression watchdog, mirroring quantization_test.py's
        # test_pool_starts_cleanly_under_background_thread_lock_contention: a daemon-thread join
        # timeout turns a REINTRODUCED hang into an ordinary, fast test failure instead of blocking the
        # whole suite forever. With the fix in place this raises immediately (control validation runs
        # before the search loop even starts), so the thread should finish almost instantly.
        outcome = {}

        def _run():
            try:
                next(iter(quantized_best_first_decode(_next_logprobs, eos=_EOS, max_len=5, batch_size=-1)))
            except BaseException as exc:  # noqa: BLE001 -- captured across a thread boundary, re-raised below
                outcome["error"] = exc

        runner = threading.Thread(target=_run, daemon=True)
        runner.start()
        runner.join(timeout=5)
        self.assertFalse(runner.is_alive(), "negative batch_size hung instead of being rejected (MXR-080-0227)")
        self.assertIsInstance(outcome.get("error"), ValueError)
        self.assertIn("batch_size", str(outcome["error"]))

    def test_zero_and_noninteger_batch_size_are_rejected(self):
        for bad in (0, -5, 2.5, float("nan")):
            with self.assertRaises(ValueError):
                next(iter(quantized_best_first_decode(_next_logprobs, eos=_EOS, max_len=5, batch_size=bad)))

    def test_batched_callback_under_return_raises_instead_of_dropping_the_frontier(self):
        # Reproduces the audit's exact case: a single live prefix (the empty start prefix), a batched
        # callback that returns fewer step tables than requested (zero, for the one asked for).
        def under_returning(prefixes):
            return []

        with self.assertRaises(ValueError) as ctx:
            list(quantized_best_first_decode(batch_next_logprobs=under_returning, eos=_EOS, max_len=3))
        self.assertIn("expected 1, got 0", str(ctx.exception))

    def test_batched_callback_over_return_also_raises(self):
        def over_returning(prefixes):
            return [list(_next_logprobs(pf)) for pf in prefixes] + [[]]

        with self.assertRaises(ValueError) as ctx:
            list(quantized_best_first_decode(batch_next_logprobs=over_returning, eos=_EOS, max_len=3))
        self.assertIn("expected 1, got 2", str(ctx.exception))

    def test_bucket_bits_out_of_range_is_rejected(self):
        for bad in (-1, 1024, 2.5, float("nan")):
            with self.assertRaises(ValueError):
                next(iter(quantized_best_first_decode(_next_logprobs, eos=_EOS, max_len=3, bucket_bits=bad)))

    def test_max_results_is_validated(self):
        for bad in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                next(iter(quantized_best_first_decode(_next_logprobs, eos=_EOS, max_len=3, max_results=bad)))

    def test_min_mass_is_validated(self):
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                next(iter(quantized_best_first_decode(_next_logprobs, eos=_EOS, max_len=3, min_mass=bad)))

    def test_valid_batch_size_and_well_behaved_callback_process_full_frontier(self):
        # Negative control: a normal, well-formed run (valid positive batch_size forcing several
        # take/commit iterations, a callback that returns exactly one step table per requested prefix)
        # still processes the WHOLE frontier correctly and matches brute force exactly -- the fix does
        # not regress the ordinary case.
        max_len = 5
        brute = _brute_force(max_len)

        def batch_next_logprobs(prefixes):
            return [list(_next_logprobs(pf)) for pf in prefixes]

        mine = list(
            quantized_best_first_decode(
                batch_next_logprobs=batch_next_logprobs, eos=_EOS, max_len=max_len, batch_size=2, bucket_bits=20
            )
        )
        self.assertEqual(len(mine), len(brute))
        np.testing.assert_allclose(sorted(lp for _, lp in mine), sorted(lp for _, lp in brute), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
