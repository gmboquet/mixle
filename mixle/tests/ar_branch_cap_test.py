"""Certified branch-cap: approximate autoregressive enumeration over the per-context top-m sub-support.

``branch_cap=m`` makes the count-index tree recurse into only the top-``m`` in-budget tokens per prefix --
the ~V/m-per-level shrink that makes exact-head enumeration affordable on LLM-sized vocabularies. The
contract verified here: the kept index is EXACT on its sub-support (every enumerated sequence's tokens are
each within their context's top-``m``, log-probs exact), the skipped remainder carries a SOUND certificate
(``true_total <= kept_total + dropped_upper``, checked against the uncapped truth on both the fast int64
path and the reference bigint path), capped drops do not masquerade as budget truncation (no runaway
deepening), and the tree actually shrinks (forward count bounded by the capped fan-out).
"""

import math
import unittest

import numpy as np

from mixle.enumeration import AutoregressiveEnumerable
from mixle.enumeration.quantization.core import Quantizer


def _model(V, L, seed=0, scale=1.0):
    W = np.random.RandomState(seed).randn(L, V, V) * scale

    def nlp(prefix):
        d = len(prefix)
        last = prefix[-1] if prefix else 0
        lg = W[d, last]
        m = np.max(lg)
        return np.arange(V), lg - (m + math.log(np.sum(np.exp(lg - m))))

    return nlp


def _top_m_tokens(nlp, prefix, m):
    tokens, lps = nlp(prefix)
    order = np.argsort(-lps, kind="stable")
    return set(tokens[order][:m].tolist())


class BranchCapSoundnessTest(unittest.TestCase):
    def setUp(self):
        self.V, self.L, self.m = 12, 3, 4
        self.nlp = _model(self.V, self.L, seed=1)
        self.q = Quantizer(oversample=8)
        self.fb = int(math.ceil(30.0 * self.q.fine_per_bit()))  # deep enough to cover all 12**3 sequences

    def _indices(self, count_mode):
        full = AutoregressiveEnumerable(self.nlp, max_len=self.L, count_mode=count_mode)
        capped = AutoregressiveEnumerable(self.nlp, max_len=self.L, count_mode=count_mode, branch_cap=self.m)
        fi, _ = full.quantized_count_index(self.q, self.fb)
        ci, _ = capped.quantized_count_index(self.q, self.fb)
        return fi, ci

    def test_bracket_contains_truth_fast_and_reference_paths(self):
        for count_mode in ("auto", "exact"):  # exercises the numpy fast path AND the bigint reference
            fi, ci = self._indices(count_mode)
            true_total = float(fi.total())
            kept_total = float(ci.total())
            self.assertLessEqual(kept_total, true_total, count_mode)
            self.assertGreaterEqual(kept_total + ci.dropped_upper, true_total, count_mode)
            self.assertGreater(ci.dropped_upper, 0.0, count_mode)  # 12 > 4: something was certifiably skipped
            self.assertEqual(float(fi.dropped_upper), 0.0, count_mode)  # exhaustive index carries no drop

    def test_kept_subsupport_is_exact_top_m(self):
        _fi, ci = self._indices("auto")
        capped = AutoregressiveEnumerable(self.nlp, max_len=self.L, branch_cap=self.m)
        hist = ci.hist
        seen = 0
        for j, c in enumerate(hist.data):
            fb = hist.base + j
            for off in range(int(c)):
                seq, lp = ci.get_in_bucket(fb, off)
                self.assertAlmostEqual(lp, capped.log_density(seq), places=12)
                prefix = ()
                for tok in seq:  # every token within its context's top-m
                    self.assertIn(tok, _top_m_tokens(self.nlp, prefix, self.m))
                    prefix = prefix + (tok,)
                seen += 1
        self.assertEqual(seen, self.m**self.L)  # top-m at every level of a fixed-length model

    def test_leaf_drop_bound_is_exact_count(self):
        # depth-1 model: each skipped in-budget token is exactly one completion, so the bracket is TIGHT
        nlp = _model(10, 1, seed=2)
        full = AutoregressiveEnumerable(nlp, max_len=1)
        capped = AutoregressiveEnumerable(nlp, max_len=1, branch_cap=3)
        q = Quantizer(oversample=8)
        fb = int(math.ceil(30.0 * q.fine_per_bit()))
        fi, _ = full.quantized_count_index(q, fb)
        ci, _ = capped.quantized_count_index(q, fb)
        self.assertEqual(int(ci.total()), 3)
        self.assertEqual(ci.dropped_upper, float(fi.total() - 3))


class BranchCapBehaviorTest(unittest.TestCase):
    def test_no_runaway_deepening(self):
        # capped drops must NOT set truncated -- SeekIndex would deepen to max_depth chasing them
        nlp = _model(10, 2, seed=3)
        ar = AutoregressiveEnumerable(nlp, max_len=2, branch_cap=3)
        si = ar.seek_index()
        si.ensure_bits(40.0)  # deep enough that the budget excludes nothing
        self.assertFalse(si.truncated)  # cap-drops are not budget truncation
        self.assertGreater(si.dropped_upper, 0.0)
        builds = si.builds
        with self.assertRaises(IndexError):
            si.unrank(10**6)  # beyond the capped sub-support: raises instead of deepening forever
        self.assertLessEqual(si.builds - builds, 1)

    def test_count_bracket_convenience(self):
        nlp = _model(12, 3, seed=4)
        full = AutoregressiveEnumerable(nlp, max_len=3)
        capped = AutoregressiveEnumerable(nlp, max_len=3, branch_cap=4)
        thr = full.unrank(200)[1]
        lo, hi = capped.count_bracket(thr)
        true_count = float(full.count(thr))
        self.assertLessEqual(lo, true_count)
        self.assertGreaterEqual(hi, true_count)

    def test_tree_shrinks_to_capped_fanout(self):
        V, L, m = 200, 3, 5
        nlp = _model(V, L, seed=5, scale=0.3)  # flat-ish: uncapped would go wide
        ar = AutoregressiveEnumerable(nlp, max_len=L, branch_cap=m)
        ar.seek_index().ensure_bits(30.0)
        # forwards = expanded prefixes <= 1 + m + m^2 (root + two capped levels)
        self.assertLessEqual(len(ar._cache), 1 + m + m * m)

    def test_validates_cap(self):
        with self.assertRaises(ValueError):
            AutoregressiveEnumerable(_model(4, 2), max_len=2, branch_cap=0)


if __name__ == "__main__":
    unittest.main()
