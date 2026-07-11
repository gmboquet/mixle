"""One-shot iterators are materialized before profiling (worklist I6.2).

``mixle.utils.automatic`` profiles its input by iterating the record stream more than once (schema
detection, then empirical sufficient statistics / fitting). A one-shot iterator -- a generator,
``map``/``filter``/``zip``, or a file object -- returns itself from ``__iter__`` and is exhausted after
the first pass, so without materializing it the profiler would silently work on an empty stream on the
second pass and build a wrong model with no error. ``normalize_input`` must materialize such an iterator
to a reusable list while leaving a real sequence (list/tuple/ndarray) untouched.
"""

import unittest

import numpy as np

from mixle.utils.automatic.profiling import normalize_input


class OneShotIteratorTest(unittest.TestCase):
    def test_one_shot_iterators_are_materialized(self):
        data = [0.1, 0.5, 0.9, 1.2, 0.3]
        # a generator, a bare iterator, and a map object are all one-shot -> materialized to the list
        self.assertEqual(normalize_input(x for x in data), data)
        self.assertEqual(normalize_input(iter(data)), data)
        self.assertEqual(normalize_input(map(float, data)), data)
        # and the result is a reusable list, not another one-shot iterator
        materialized = normalize_input(x for x in data)
        self.assertEqual(list(materialized), list(materialized))  # a second pass still sees the records

    def test_reusable_sequences_are_left_unchanged(self):
        lst = [1.0, 2.0, 3.0]
        self.assertIs(normalize_input(lst), lst)
        tup = (1.0, 2.0, 3.0)
        self.assertIs(normalize_input(tup), tup)
        arr = np.array([1.0, 2.0, 3.0])
        self.assertIs(normalize_input(arr), arr)

    def test_get_estimator_on_a_generator_matches_the_list(self):
        from mixle.utils.automatic import get_estimator

        data = [0.1, 0.5, 0.9, 1.2, 0.3, 0.7, 1.1, 0.4]
        # profiling a generator detects the same family as profiling the equivalent list (before the fix
        # the generator was consumed mid-profiling and could silently select a different/degenerate model)
        self.assertEqual(
            type(get_estimator(x for x in data)).__name__,
            type(get_estimator(list(data))).__name__,
        )


if __name__ == "__main__":
    unittest.main()
