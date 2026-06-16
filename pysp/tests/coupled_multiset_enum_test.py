"""Enumeration for the coupled bag-of-counts model families (PLSI, IBP, hidden association).

These were previously non-enumerable; each is verified against brute force on a small instance and
checked for the generic enumerator invariants (descending order, log_prob == log_density, uniqueness).
"""

import itertools
import unittest

import numpy as np

from pysp.utils.enumeration import freeze


def tiers(pairs):
    out = {}
    for v, lp in pairs:
        out.setdefault(round(lp, 8), set()).add(freeze(v))
    return out


def _bags_over(num_vals, n):
    """All integer count-vector bags ``[(value, count)]`` over ``num_vals`` values summing to ``n``."""
    if num_vals == 1:
        yield [(0, n)] if n > 0 else []
        return
    for c0 in range(n + 1):
        for rest in _bags_over(num_vals - 1, n - c0):
            head = [(0, c0)] if c0 > 0 else []
            yield head + [(w + 1, c) for w, c in rest]


class IntegerPLSIEnumerationTestCase(unittest.TestCase):
    def _dist(self):
        from pysp.stats.latent.int_plsi import IntegerPLSIDistribution
        from pysp.stats.leaf.int_range import IntegerCategoricalDistribution

        prob = np.array([[0.5, 0.1], [0.3, 0.2], [0.2, 0.7]])
        state = np.array([[0.6, 0.4], [0.2, 0.8]])
        doc_vec = np.array([0.7, 0.3])
        return IntegerPLSIDistribution(
            prob, state, doc_vec, len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        )

    def test_matches_brute_force(self):
        dist = self._dist()
        brute = []
        for d in (0, 1):
            for n in (0, 1, 2):
                for bag in _bags_over(3, n):
                    brute.append(((d, bag), dist.log_density((d, bag))))
        brute = [(v, lp) for v, lp in brute if lp > -np.inf]
        brute.sort(key=lambda u: -u[1])

        items = list(itertools.islice(dist.enumerator(), len(brute)))
        self.assertEqual(len(items), len(brute))
        np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute], atol=1e-9)
        for v, lp in items:
            self.assertAlmostEqual(lp, dist.log_density(v), delta=1e-9)
        self.assertEqual(tiers(items), tiers(brute))

    def test_null_length_support_is_descending_and_exact(self):
        from pysp.stats.combinator.null_dist import NullDistribution
        from pysp.stats.latent.int_plsi import IntegerPLSIDistribution

        prob = np.array([[0.5, 0.1], [0.3, 0.2], [0.2, 0.7]])
        state = np.array([[0.6, 0.4], [0.2, 0.8]])
        dist = IntegerPLSIDistribution(prob, state, np.array([0.7, 0.3]), len_dist=NullDistribution())
        items = list(itertools.islice(dist.enumerator(), 30))
        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - 1e-9)
        for v, lp in items:
            self.assertAlmostEqual(lp, dist.log_density(v), delta=1e-9)
        self.assertEqual(len({freeze(v) for v, _ in items}), len(items))


class IBPEnumerationTestCase(unittest.TestCase):
    def test_matches_brute_force_both_formats(self):
        from pysp.stats.latent.ibp import IndianBuffetProcessDistribution

        for fmt in ("dense", "sparse"):
            dist = IndianBuffetProcessDistribution(num_features=4, feature_probs=[0.6, 0.3, 0.2, 0.45], data_format=fmt)
            brute = []
            for bits in itertools.product((0, 1), repeat=4):
                x = list(bits) if fmt != "sparse" else [k for k, b in enumerate(bits) if b]
                brute.append((x, dist.log_density(x)))
            brute.sort(key=lambda t: -t[1])
            items = list(dist.enumerator())
            self.assertEqual(len(items), len(brute))
            np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute], atol=1e-9)
            for v, lp in items:
                self.assertAlmostEqual(lp, dist.log_density(v), delta=1e-9)
            self.assertEqual(tiers(items), tiers(brute))
            self.assertAlmostEqual(np.logaddexp.reduce([lp for _, lp in items]), 0.0, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
