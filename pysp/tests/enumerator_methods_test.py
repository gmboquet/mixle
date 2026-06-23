"""WS-3: the enumerator object exposes rank / seek / cumulative / nucleus_size / from_index as methods.

These replace the free-function syntax (density_rank(dist, value), count_dp_seek(dist, index), …). The
enumerator is the one home for "where does a value sit / what is at this index / iterate from here"
over the weighted structure it enumerates.
"""

import unittest

from pysp.stats.combinator.composite import CompositeDistribution
from pysp.stats.univariate.discrete.categorical import CategoricalDistribution
from pysp.stats.univariate.discrete.poisson import PoissonDistribution


class EnumeratorMethodsTest(unittest.TestCase):
    def test_rank_and_cumulative(self):
        c = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        r = c.enumerator().rank("b")
        self.assertEqual(r.rank, 1)  # one outcome ('a') is strictly more probable
        self.assertAlmostEqual(r.cumulative_probability, 0.8, places=6)  # p(a)+p(b)
        self.assertAlmostEqual(c.enumerator().cumulative("c"), 1.0, places=6)

    def test_seek_is_inverse_of_rank(self):
        c = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        self.assertEqual(c.enumerator().seek(0).value, "a")
        self.assertEqual(c.enumerator().seek(1).value, "b")
        # deep structural seek on a decomposable (Composite) support — no prefix enumeration
        comp = CompositeDistribution((PoissonDistribution(3.0), CategoricalDistribution({0: 0.6, 1: 0.4})))
        seeked = comp.enumerator().seek(50).value
        self.assertIsInstance(seeked, tuple)
        self.assertEqual(len(seeked), 2)

    def test_nucleus_size_matches_top_p(self):
        c = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        materialized = len(c.enumerator().top_p(0.9))
        sized = c.enumerator().nucleus_size(0.9)
        self.assertLessEqual(sized.size_lower, materialized)
        self.assertLessEqual(materialized, sized.size_upper)

    def test_from_index_iterates_from_a_position(self):
        c = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        values = [v for v, _ in c.enumerator().from_index(1, 3)]
        self.assertEqual(values, ["b", "c"])
        # from_index uses a fresh enumeration — calling it does not consume an enumerator used elsewhere
        e = c.enumerator()
        _ = list(e.from_index(0, 2))
        self.assertEqual(e.top_k(1), [("a", e.dist.log_density("a"))])


if __name__ == "__main__":
    unittest.main()
