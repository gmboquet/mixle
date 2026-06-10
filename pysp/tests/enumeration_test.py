"""Tests for the smart enumeration API.

Every enumerable distribution is run through the generic invariants: descending order,
consistency with log_density, uniqueness, completeness (finite supports), and agreement
with brute-force enumeration for small models. Error cases verify the fail-fast
EnumerationError contract.
"""
import itertools
import unittest

import numpy as np

from pysp.stats import *
from pysp.stats.pdist import EnumerationError
from pysp.stats.hidden_markov_ind_pi import IndPiHiddenMarkovModelDistribution
from pysp.utils.enumeration import freeze, supports_enumeration

TOL = 1e-9


def make_cases():
    """Returns a list of (name, dist, n_to_take, expected_count_or_None) cases."""
    cat = CategoricalDistribution({'a': 0.7, 'b': 0.3})
    cat3 = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
    intcat = IntegerCategoricalDistribution(2, [0.1, 0.0, 0.6, 0.3])
    geo = GeometricDistribution(0.3)
    mix_geo = MixtureDistribution([GeometricDistribution(0.5), GeometricDistribution(0.1)], [0.6, 0.4])

    hmm = HiddenMarkovModelDistribution(
        topics=[CategoricalDistribution({'a': 0.8, 'b': 0.2}),
                CategoricalDistribution({'b': 0.6, 'c': 0.4})],
        w=[0.7, 0.3], transitions=[[0.9, 0.1], [0.4, 0.6]],
        len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))

    ind_pi_hmm = IndPiHiddenMarkovModelDistribution(
        topics=[CategoricalDistribution({'a': 0.8, 'b': 0.2}),
                CategoricalDistribution({'b': 0.6, 'c': 0.4})],
        w=[[0.7, 0.3], [0.5, 0.5]], transitions=[[0.9, 0.1], [0.4, 0.6]], taus=None,
        len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))

    mc = MarkovChainDistribution(
        {'x': 0.6, 'y': 0.4},
        {'x': {'x': 0.8, 'y': 0.2}, 'y': {'x': 0.5, 'y': 0.5}},
        len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))

    return [
        ('categorical', cat3, 10, 3),
        ('intcat_zero_prob', intcat, 10, 3),
        ('binomial', BinomialDistribution(p=0.4, n=6, min_val=3), 20, 7),
        ('geometric', geo, 40, None),
        ('poisson', PoissonDistribution(lam=4.7), 40, None),
        ('poisson_int_lam', PoissonDistribution(lam=3.0), 40, None),
        ('poisson_small_lam', PoissonDistribution(lam=0.2), 40, None),
        ('composite', CompositeDistribution((cat3, intcat)), 30, 9),
        ('composite_nested', CompositeDistribution((cat, CompositeDistribution((cat, intcat)))), 30, 12),
        ('composite_with_mixture', CompositeDistribution((cat, mix_geo)), 50, None),
        ('mixture_overlap', MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]),
             IntegerCategoricalDistribution(1, [0.5, 0.5])], [0.6, 0.4]), 10, 3),
        ('mixture_geometrics', mix_geo, 40, None),
        ('mixture_zero_weight_gaussian', MixtureDistribution(
            [intcat, GaussianDistribution(0.0, 1.0)], [1.0, 0.0]), 10, 3),
        ('het_mixture', HeterogeneousMixtureDistribution(
            [cat3, CategoricalDistribution({'b': 0.5, 'd': 0.5})], [0.5, 0.5]), 10, 4),
        ('optional', OptionalDistribution(cat3, p=0.2, missing_value='MISSING'), 10, 4),
        ('optional_collision', OptionalDistribution(cat3, p=0.2, missing_value='a'), 10, 3),
        ('weighted', WeightedDistribution(cat3), 10, 3),
        ('null', NullDistribution(), 5, 1),
        ('sequence', SequenceDistribution(cat, len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])), 30, 7),
        ('sequence_geo_len', SequenceDistribution(cat, len_dist=GeometricDistribution(0.5)), 25, None),
        ('intsetdist', IntegerBernoulliSetDistribution(np.log([0.8, 0.4, 0.01])), 10, 8),
        ('markov_chain', mc, 30, 15),
        ('hmm', hmm, 60, 40),
        ('ind_pi_hmm', ind_pi_hmm, 60, 40),
    ]


class EnumerationInvariantTestCase(unittest.TestCase):

    def setUp(self):
        self.cases = make_cases()

    def test_ordering_non_increasing(self):
        for name, dist, n, _ in self.cases:
            items = dist.enumerator().top_k(n)
            lps = [lp for _, lp in items]
            for i in range(len(lps) - 1):
                self.assertGreaterEqual(lps[i], lps[i + 1] - TOL, '%s: order violated at %d' % (name, i))

    def test_log_prob_matches_log_density(self):
        for name, dist, n, _ in self.cases:
            with np.errstate(divide='ignore'):
                for v, lp in dist.enumerator().top_k(n):
                    self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL,
                                           msg='%s: lp mismatch at %r' % (name, v))

    def test_values_unique(self):
        for name, dist, n, _ in self.cases:
            items = dist.enumerator().top_k(n)
            keys = [freeze(v) for v, _ in items]
            self.assertEqual(len(keys), len(set(keys)), '%s: duplicate values yielded' % name)

    def test_finite_completeness_and_total_mass(self):
        for name, dist, n, expected in self.cases:
            if expected is None:
                continue
            items = list(dist.enumerator())
            self.assertEqual(len(items), expected, '%s: wrong support size' % name)
            total = np.logaddexp.reduce([lp for _, lp in items])
            if name == 'optional_collision':
                # When missing_value collides with a base support value, log_density routes
                # it to the missing branch and the base mass of that value is unreachable —
                # the distribution itself sums to 0.2 + 0.8*(0.3+0.2) = 0.6, and the
                # enumeration faithfully reflects that.
                self.assertAlmostEqual(total, np.log(0.6), delta=1e-8, msg=name)
            else:
                self.assertAlmostEqual(total, 0.0, delta=1e-8, msg='%s: total mass != 1' % name)

    def test_top_k(self):
        for name, dist, n, expected in self.cases:
            top3 = dist.enumerator().top_k(3)
            fresh = dist.enumerator().top_k(max(3, n))[:3]
            self.assertEqual([freeze(v) for v, _ in top3], [freeze(v) for v, _ in fresh], name)
            if expected is not None:
                self.assertEqual(len(dist.enumerator().top_k(expected + 100)), expected, name)


class BruteForceCrossCheckTestCase(unittest.TestCase):
    """Independently constructs small supports, scores them with log_density, and compares."""

    def assert_matches_brute(self, dist, support, name):
        with np.errstate(divide='ignore'):
            brute = [(v, dist.log_density(v)) for v in support]
        brute = [(v, lp) for v, lp in brute if lp > -np.inf]
        brute.sort(key=lambda u: -u[1])
        items = list(dist.enumerator())
        self.assertEqual(len(items), len(brute), '%s: size mismatch' % name)
        np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute],
                                   atol=TOL, err_msg='%s: score sequence mismatch' % name)
        # Compare value sets per strictly-descending tier to avoid tie-order false failures.
        def tiers(pairs):
            out = {}
            for v, lp in pairs:
                out.setdefault(round(lp, 8), set()).add(freeze(v))
            return out
        self.assertEqual(tiers(items), tiers(brute), '%s: tier values mismatch' % name)

    def test_composite(self):
        cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
        intcat = IntegerCategoricalDistribution(0, [0.6, 0.4])
        dist = CompositeDistribution((cat, intcat))
        support = [(s, i) for s in 'abc' for i in (0, 1)]
        self.assert_matches_brute(dist, support, 'composite')

    def test_mixture_overlapping(self):
        dist = MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]),
             IntegerCategoricalDistribution(1, [0.5, 0.5])], [0.6, 0.4])
        self.assert_matches_brute(dist, list(range(-1, 5)), 'mixture')

    def test_sequence(self):
        cat = CategoricalDistribution({'a': 0.7, 'b': 0.3})
        dist = SequenceDistribution(cat, len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]))
        support = [[]] + [list(t) for L in (1, 2) for t in itertools.product('ab', repeat=L)]
        self.assert_matches_brute(dist, support, 'sequence')

    def test_markov_chain(self):
        dist = MarkovChainDistribution(
            {'x': 0.6, 'y': 0.4},
            {'x': {'x': 0.8, 'y': 0.2}, 'y': {'x': 0.5, 'y': 0.5}},
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))
        support = [[]] + [list(t) for L in (1, 2, 3) for t in itertools.product('xy', repeat=L)]
        self.assert_matches_brute(dist, support, 'markov_chain')

    def test_hmm(self):
        dist = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution({'a': 0.8, 'b': 0.2}),
                    CategoricalDistribution({'b': 0.6, 'c': 0.4})],
            w=[0.7, 0.3], transitions=[[0.9, 0.1], [0.4, 0.6]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))
        support = [[]] + [list(t) for L in (1, 2, 3) for t in itertools.product('abc', repeat=L)]
        self.assert_matches_brute(dist, support, 'hmm')

    def test_intsetdist(self):
        dist = IntegerBernoulliSetDistribution(np.log([0.8, 0.4, 0.01]))
        support = [sorted(c) for r in range(4) for c in itertools.combinations(range(3), r)]
        support = [list(c) for c in support]
        self.assert_matches_brute(dist, support, 'intsetdist')


class InfiniteSupportMassDominanceTestCase(unittest.TestCase):
    """The first N items of an infinite enumeration must contain every value that is
    strictly more probable than the N-th item."""

    def test_poisson_and_geometric(self):
        # Geometric support starts at 1 (its log_density applies the pmf formula outside
        # the support too, so scanning from 0 would test values the distribution excludes).
        for dist, lo in ((PoissonDistribution(4.7), 0), (PoissonDistribution(3.0), 0),
                         (GeometricDistribution(0.3), 1)):
            items = dist.enumerator().top_k(40)
            cutoff = items[-1][1]
            seen = set(v for v, _ in items)
            for x in range(lo, 200):
                if dist.log_density(x) > cutoff + TOL:
                    self.assertIn(x, seen, '%s missing %d' % (dist, x))

    def test_mixture_of_geometrics(self):
        dist = MixtureDistribution([GeometricDistribution(0.5), GeometricDistribution(0.1)], [0.6, 0.4])
        items = dist.enumerator().top_k(40)
        cutoff = items[-1][1]
        seen = set(v for v, _ in items)
        for x in range(1, 300):
            if dist.log_density(x) > cutoff + TOL:
                self.assertIn(x, seen, 'mixture missing %d' % x)


class EnumerationErrorTestCase(unittest.TestCase):

    def test_continuous_raises(self):
        with self.assertRaises(EnumerationError):
            GaussianDistribution(0.0, 1.0).enumerator()

    def test_composite_error_names_child(self):
        cat = CategoricalDistribution({'a': 1.0})
        with self.assertRaises(EnumerationError) as cm:
            CompositeDistribution((cat, GaussianDistribution(0.0, 1.0))).enumerator()
        self.assertIn('dists[1]', str(cm.exception))
        self.assertIn('GaussianDistribution', str(cm.exception))

    def test_nested_error_path(self):
        cat = CategoricalDistribution({'a': 1.0})
        inner = MixtureDistribution([GaussianDistribution(0.0, 1.0), cat], [0.5, 0.5])
        with self.assertRaises(EnumerationError) as cm:
            CompositeDistribution((cat, inner)).enumerator()
        msg = str(cm.exception)
        self.assertIn('dists[1]', msg)
        self.assertIn('components[0]', msg)
        self.assertIn('GaussianDistribution', msg)

    def test_categorical_default_value_raises(self):
        with self.assertRaises(EnumerationError):
            CategoricalDistribution({'a': 0.9}, default_value=0.1).enumerator()

    def test_sequence_without_len_dist_raises(self):
        with self.assertRaises(EnumerationError):
            SequenceDistribution(CategoricalDistribution({'a': 1.0})).enumerator()

    def test_sequence_len_normalized_raises(self):
        with self.assertRaises(EnumerationError):
            SequenceDistribution(CategoricalDistribution({'a': 1.0}),
                                 len_dist=GeometricDistribution(0.5), len_normalized=True).enumerator()

    def test_ignored_raises(self):
        with self.assertRaises(EnumerationError):
            IgnoredDistribution(CategoricalDistribution({'a': 1.0})).enumerator()

    def test_optional_without_p_raises(self):
        with self.assertRaises(EnumerationError):
            OptionalDistribution(CategoricalDistribution({'a': 1.0})).enumerator()

    def test_hmm_terminal_values_raises(self):
        with self.assertRaises(EnumerationError):
            HiddenMarkovModelDistribution(
                topics=[CategoricalDistribution({'a': 1.0})], w=[1.0], transitions=[[1.0]],
                len_dist=GeometricDistribution(0.5), terminal_values={'a'}).enumerator()

    def test_markov_chain_default_value_raises(self):
        with self.assertRaises(EnumerationError):
            MarkovChainDistribution({'x': 1.0}, {'x': {'x': 1.0}}, default_value=0.1,
                                    len_dist=GeometricDistribution(0.5)).enumerator()

    def test_zero_weight_type_incompatible_component(self):
        # A zero-weight Gaussian mixed with a string categorical must never be evaluated,
        # neither for stream generation nor for exact re-scoring.
        dist = MixtureDistribution(
            [CategoricalDistribution({'a': 0.6, 'b': 0.4}), GaussianDistribution(0.0, 1.0)], [1.0, 0.0])
        items = list(dist.enumerator())
        self.assertEqual([v for v, _ in items], ['a', 'b'])
        np.testing.assert_allclose([lp for _, lp in items], np.log([0.6, 0.4]), atol=TOL)

    def test_supports_enumeration_predicate(self):
        self.assertTrue(supports_enumeration(CategoricalDistribution({'a': 1.0})))
        self.assertFalse(supports_enumeration(GaussianDistribution(0.0, 1.0)))


class FlagshipCompositionTestCase(unittest.TestCase):
    """The motivating case: a composite of a categorical and a mixture of geometrics."""

    def test_composite_of_categorical_and_mixture(self):
        m = MixtureDistribution([GeometricDistribution(0.5), GeometricDistribution(0.1)], [0.6, 0.4])
        c = CompositeDistribution((CategoricalDistribution({'a': 0.7, 'b': 0.3}), m))
        items = c.enumerator().top_k(50)
        lps = [lp for _, lp in items]
        self.assertEqual(len(items), 50)
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)
        for v, lp in items:
            self.assertAlmostEqual(lp, c.log_density(v), delta=TOL)
        # mass dominance against a brute scan of the head of the support
        cutoff = lps[-1]
        seen = set(freeze(v) for v, _ in items)
        for s in 'ab':
            for x in range(1, 100):
                if c.log_density((s, x)) > cutoff + TOL:
                    self.assertIn(freeze((s, x)), seen)


if __name__ == '__main__':
    unittest.main()
