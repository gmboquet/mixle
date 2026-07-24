"""Tests for the quantized seek index's direct/low-level construction and rank validation.

MXR-080-0206 (Critical): ``LazyQuantizedEnumerationIndex`` (and :func:`build_budget_index`,
which builds one) must certify each bucket count is an exact integer instead of silently
truncating an approximate (float-count-mode) count with ``int()``. A count of ``2.9`` becoming
``int(2.9) == 2`` addressable values is not a small error localized to one bucket: it shifts every
later cumulative rank boundary, so values can be omitted, duplicated, or unranked through the
wrong structural offset. This module chose the certification/reject-ambiguous-input approach
(see :func:`mixle.enumeration.quantization.seek._certified_integer`) over full certified-interval
propagation through the rank-query API: exact-mode counts (arbitrary-precision Python ints) and
float-mode counts that land exactly on an integer -- which is what this package's own float64
count arithmetic (pure addition/multiplication, never division) always produces -- are accepted
unchanged; a count that is not EXACTLY integer-valued is refused at construction with a typed
``AmbiguousCountError`` instead of being silently truncated into a structurally corrupt index.

Since ``LazyQuantizedEnumerationIndex`` has no separate checked factory method (this constructor
is the only way to build one -- used directly by ``build_budget_index``, ``composite.py``,
``heterogeneous_pcfg.py``), its constructor is also hardened here against the MXR-080-0207
invariants (finite bounds, certified/unique bin ids, callable getter) that a checked factory
would otherwise enforce -- there is nowhere else for that validation to live.

MXR-080-0207 (High): the direct constructors of ``QuantizedEnumerationIndex`` and
``QuantizedCrossIndex`` (which DO have separate checked factory methods -- ``from_enumerator``/
``from_items``) must enforce the same invariants those factories already enforce: finite bounds,
ordered/unique bins (duplicate or unsorted bins corrupt the bisect rank table), component arity,
and per-row score length (``QuantizedCrossIndex`` could otherwise build joint bin-id keys whose
dimensionality silently differs from ``max_bits``).
"""

import math
import unittest

from mixle.enumeration.quantization.core import Quantizer, build_budget_index
from mixle.enumeration.quantization.seek import (
    AmbiguousCountError,
    LazyQuantizedEnumerationIndex,
    QuantizedCrossIndex,
    QuantizedEnumerationIndex,
)


class _FakeHist:
    """Minimal duck-typed stand-in for CountHistogram -- just ``.base``/``.data`` -- so these
    tests exercise build_budget_index's own count-certification logic in isolation, independent
    of CountHistogram's own (separately validated, MXR-080-0204) constructor."""

    def __init__(self, base, data):
        self.base = base
        self.data = list(data)


class _FakeCountIndex:
    """Minimal duck-typed stand-in for CountIndex -- just ``.hist``/``.get_in_bucket``."""

    def __init__(self, hist, getter):
        self.hist = hist
        self._getter = getter

    def get_in_bucket(self, fine_bucket, offset):
        return self._getter(fine_bucket, offset)


def _labeled_getter(prefix="v"):
    def getter(fb, off):
        return (f"{prefix}{fb}-{off}", -float(fb))

    return getter


class BuildBudgetIndexCountCertificationTestCase(unittest.TestCase):
    """MXR-080-0206: build_budget_index must certify fine-bucket counts, not truncate them."""

    def test_fractional_count_is_rejected_not_silently_truncated(self):
        # The audit's exact example: a count of 2.9 must not silently become int(2.9) == 2
        # addressable values.
        hist = _FakeHist(base=0, data=[2.9])
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        with self.assertRaises(AmbiguousCountError):
            build_budget_index(index, q, budget_bits=10.0)

    def test_fractional_count_in_a_later_bucket_is_also_rejected(self):
        # Regression proof for "every later cumulative boundary changes": a bad count deep in
        # the histogram (not just the first bucket) must still be caught at construction, not
        # silently baked into a corrupted cumulative offset for every bucket beyond it.
        hist = _FakeHist(base=0, data=[2, 3, 1.5, 4])
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        with self.assertRaises(AmbiguousCountError):
            build_budget_index(index, q, budget_bits=10.0)

    def test_negative_count_is_rejected(self):
        hist = _FakeHist(base=0, data=[2, -1, 3])
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        with self.assertRaises(ValueError):
            build_budget_index(index, q, budget_bits=10.0)

    def test_non_finite_count_is_rejected(self):
        hist = _FakeHist(base=0, data=[2, float("nan"), 3])
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        with self.assertRaises(ValueError):
            build_budget_index(index, q, budget_bits=10.0)
        hist_inf = _FakeHist(base=0, data=[2, float("inf")])
        index_inf = _FakeCountIndex(hist_inf, _labeled_getter())
        with self.assertRaises(ValueError):
            build_budget_index(index_inf, q, budget_bits=10.0)

    def test_bool_count_is_rejected(self):
        # Pre-fix, `bin_total += True` silently launders the bool into a plain int (0 + True ==
        # 1) BEFORE it ever reaches a certification point -- so this specifically proves
        # build_budget_index catches it at the earliest point, not just LazyQuantizedEnumerationIndex.
        hist = _FakeHist(base=0, data=[2, True])
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        with self.assertRaises(TypeError):
            build_budget_index(index, q, budget_bits=10.0)

    def test_exact_integer_counts_still_support_full_random_access(self):
        # Negative control: exact-integer-mode counts (no approximation involved) are unaffected
        # -- every rank still resolves to the right bucket with correct cumulative offsets. This
        # is exactly the scenario that WOULD have been corrupted pre-fix if any upstream bucket's
        # count had been mis-truncated: bucket boundaries here are proven exactly right.
        hist = _FakeHist(base=0, data=[2, 3, 1, 4])  # 4 fine buckets, all integral
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        idx = build_budget_index(index, q, budget_bits=10.0)
        self.assertEqual(idx.total_count, 10)
        seen = [idx.get(i)[0] for i in range(10)]
        self.assertEqual(len(set(seen)), 10)  # every rank maps to a distinct value
        self.assertEqual([idx.get(i)[0] for i in range(0, 2)], ["v0-0", "v0-1"])
        self.assertEqual([idx.get(i)[0] for i in range(2, 5)], ["v1-0", "v1-1", "v1-2"])
        self.assertEqual([idx.get(i)[0] for i in range(5, 6)], ["v2-0"])
        self.assertEqual([idx.get(i)[0] for i in range(6, 10)], ["v3-0", "v3-1", "v3-2", "v3-3"])

    def test_float_valued_but_exactly_integral_counts_are_accepted(self):
        # Negative control: count_mode='float' in practice always lands exactly on an integer --
        # this package's float64 count arithmetic is pure addition/multiplication on
        # integer-sourced data, which never produces a genuine fraction (only bounded rounding
        # that still lands on SOME integer). An integer-valued float like 3.0 must be accepted,
        # certified down to a genuine Python int, not rejected outright.
        hist = _FakeHist(base=0, data=[2.0, 3.0])
        index = _FakeCountIndex(hist, _labeled_getter())
        q = Quantizer(bin_width_bits=1.0, oversample=1)
        idx = build_budget_index(index, q, budget_bits=10.0)
        self.assertEqual(idx.total_count, 5)
        self.assertIsInstance(idx.counts[0], int)
        self.assertIsInstance(idx.counts[1], int)
        self.assertNotIsInstance(idx.counts[0], bool)


class LazyQuantizedEnumerationIndexDirectConstructorTestCase(unittest.TestCase):
    """MXR-080-0206 + MXR-080-0207: there is no separate checked factory for this class -- the
    constructor IS the only way to build one (used directly by build_budget_index, composite.py,
    heterogeneous_pcfg.py), so it alone must enforce every invariant."""

    def test_rejects_fractional_count(self):
        with self.assertRaises(AmbiguousCountError):
            LazyQuantizedEnumerationIndex(
                {0: 2.9, 1: 3}, bin_width_bits=1.0, max_bits=4.0, truncated=False, getter=_labeled_getter()
            )

    def test_rejects_negative_count(self):
        with self.assertRaises(ValueError):
            LazyQuantizedEnumerationIndex(
                {0: -1, 1: 3}, bin_width_bits=1.0, max_bits=4.0, truncated=False, getter=_labeled_getter()
            )

    def test_rejects_bool_count(self):
        with self.assertRaises(TypeError):
            LazyQuantizedEnumerationIndex(
                {0: True, 1: 3}, bin_width_bits=1.0, max_bits=4.0, truncated=False, getter=_labeled_getter()
            )

    def test_rejects_non_finite_max_bits(self):
        with self.assertRaises(ValueError):
            LazyQuantizedEnumerationIndex(
                {0: 2}, bin_width_bits=1.0, max_bits=float("inf"), truncated=False, getter=_labeled_getter()
            )
        with self.assertRaises(ValueError):
            LazyQuantizedEnumerationIndex(
                {0: 2}, bin_width_bits=1.0, max_bits=float("nan"), truncated=False, getter=_labeled_getter()
            )

    def test_rejects_nonpositive_bin_width_bits(self):
        with self.assertRaises(ValueError):
            LazyQuantizedEnumerationIndex(
                {0: 2}, bin_width_bits=0.0, max_bits=4.0, truncated=False, getter=_labeled_getter()
            )

    def test_rejects_noncallable_getter(self):
        with self.assertRaises(TypeError):
            LazyQuantizedEnumerationIndex({0: 2}, bin_width_bits=1.0, max_bits=4.0, truncated=False, getter="nope")

    def test_well_formed_direct_construction_still_works(self):
        idx = LazyQuantizedEnumerationIndex(
            {0: 2, 1: 3}, bin_width_bits=1.0, max_bits=4.0, truncated=False, getter=_labeled_getter()
        )
        self.assertEqual(idx.total_count, 5)
        self.assertEqual(idx.get(0), ("v0-0", -0.0))
        self.assertEqual(idx.get(4), ("v1-2", -1.0))

    def test_integer_valued_float_count_accepted(self):
        idx = LazyQuantizedEnumerationIndex(
            {0: 2.0, 1: 3.0}, bin_width_bits=1.0, max_bits=4.0, truncated=False, getter=_labeled_getter()
        )
        self.assertEqual(idx.total_count, 5)
        self.assertIsInstance(idx.counts[0], int)


class QuantizedEnumerationIndexDirectConstructorTestCase(unittest.TestCase):
    """MXR-080-0207: the direct constructor must enforce the same invariants the checked
    factory methods (from_enumerator/from_items) already enforce."""

    def test_rejects_non_finite_max_bits(self):
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex([(0, [("a", 0.0)])], bin_width_bits=1.0, max_bits=float("nan"), truncated=False)
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex([(0, [("a", 0.0)])], bin_width_bits=1.0, max_bits=float("inf"), truncated=False)
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex([(0, [("a", 0.0)])], bin_width_bits=1.0, max_bits=-1.0, truncated=False)

    def test_rejects_non_finite_bin_width_bits(self):
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex([(0, [("a", 0.0)])], bin_width_bits=float("nan"), max_bits=4.0, truncated=False)

    def test_rejects_nonpositive_bin_width_bits(self):
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex([(0, [("a", 0.0)])], bin_width_bits=0.0, max_bits=4.0, truncated=False)

    def test_rejects_duplicate_bin_ids(self):
        # Pre-fix, this would silently corrupt _cum_starts/_cum_bins/_bin_lookup (the last
        # occurrence's items overwrite the first's in the lookup dict while the position table
        # still reserves ranks for both) -- exactly the "duplicate bins corrupt the bisect table"
        # failure mode the finding describes.
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex(
                [(0, [("a", 0.0)]), (1, [("b", -1.0)]), (0, [("c", -2.0)])],
                bin_width_bits=1.0,
                max_bits=4.0,
                truncated=False,
            )

    def test_rejects_unsorted_bins(self):
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex(
                [(1, [("a", -1.0)]), (0, [("b", 0.0)])], bin_width_bits=1.0, max_bits=4.0, truncated=False
            )

    def test_rejects_fractional_bin_id(self):
        with self.assertRaises(ValueError):
            QuantizedEnumerationIndex([(0.5, [("a", 0.0)])], bin_width_bits=1.0, max_bits=4.0, truncated=False)

    def test_rejects_bool_bin_id(self):
        with self.assertRaises(TypeError):
            QuantizedEnumerationIndex([(True, [("a", 0.0)])], bin_width_bits=1.0, max_bits=4.0, truncated=False)

    def test_well_formed_direct_construction_still_works(self):
        # Negative control: sorted, unique, finite-bounded input still builds correctly -- exactly
        # what the factory methods themselves always pass through.
        idx = QuantizedEnumerationIndex(
            [(0, [("a", 0.0)]), (1, [("b", -1.0), ("c", -1.0)])], bin_width_bits=1.0, max_bits=4.0, truncated=False
        )
        self.assertEqual(idx.total_count, 3)
        self.assertEqual(idx.counts, {0: 1, 1: 2})
        self.assertEqual(idx.get(0), ("a", 0.0))
        self.assertEqual(idx.get(1), ("b", -1.0))
        self.assertEqual(idx.get(2), ("c", -1.0))

    def test_factories_still_work(self):
        # Negative control: the checked factory methods (which always pass sorted/unique bins
        # derived from a dict) are unaffected by the new direct-constructor validation.
        items = [("a", math.log(0.5)), ("b", math.log(0.3)), ("c", math.log(0.2))]
        idx = QuantizedEnumerationIndex.from_items(items, max_bits=8.0)
        self.assertEqual(idx.total_count, 3)


class QuantizedCrossIndexDirectConstructorTestCase(unittest.TestCase):
    """MXR-080-0207: QuantizedCrossIndex's direct constructor must reject rows whose score-vector
    length does not match max_bits's declared dimensionality, instead of silently building joint
    bin-id keys whose arity differs from num_components."""

    def test_rejects_dimensionality_mismatch(self):
        # Pre-fix, this would build successfully: num_components == 2 (from max_bits) but the
        # joint bin-id key stored in self.counts would have length 3 (from log_probs) -- an
        # internal consistency violation between the declared bit-width and the actual key shape.
        with self.assertRaises(ValueError):
            QuantizedCrossIndex([("v", [0.0, -1.0, -2.0])], max_bits=[4.0, 4.0], bin_width_bits=1.0)

    def test_rejects_non_finite_max_bits(self):
        with self.assertRaises(ValueError):
            QuantizedCrossIndex([("v", [0.0, -1.0])], max_bits=[4.0, float("inf")], bin_width_bits=1.0)

    def test_rejects_non_finite_bin_width_bits(self):
        with self.assertRaises(ValueError):
            QuantizedCrossIndex([("v", [0.0, -1.0])], max_bits=[4.0, 4.0], bin_width_bits=float("nan"))

    def test_rejects_nonpositive_bin_width_bits(self):
        with self.assertRaises(ValueError):
            QuantizedCrossIndex([("v", [0.0, -1.0])], max_bits=[4.0, 4.0], bin_width_bits=0.0)

    def test_well_formed_direct_construction_still_works(self):
        idx = QuantizedCrossIndex([("v1", [0.0, -1.0]), ("v2", [-2.0, -3.0])], max_bits=[4.0, 4.0], bin_width_bits=1.0)
        self.assertEqual(idx.num_components, 2)
        self.assertEqual(idx.total_count, 2)
        self.assertTrue(all(len(key) == 2 for key in idx.counts))  # every joint bin key has arity 2

    def test_from_items_factory_still_works(self):
        idx = QuantizedCrossIndex.from_items(
            [("v1", [0.0, -1.0]), ("v2", [-2.0, -3.0])], max_bits=[4.0, 4.0], bin_width_bits=1.0
        )
        self.assertEqual(idx.num_components, 2)
        self.assertTrue(all(len(key) == 2 for key in idx.counts))


if __name__ == "__main__":
    unittest.main()
