"""Regression test: two DataSequenceEncoder __eq__ implementations violated the basic equivalence-
relation axioms (symmetry, and no exceptions from a well-formed comparison).

CompositeDataEncoder.__eq__ iterated `self.encoders` and indexed `other.encoders[i]` without ever
checking the two have the same length: a shorter `other` raised IndexError, and -- since the loop
never even looks at `other`'s length -- a *longer* `other` with a matching prefix compared equal to
a shorter `self`, while the reverse comparison (longer self, shorter other) raised. Neither
direction was reliable.

DirichletProcessMixtureDataEncoder.__eq__ special-cased "other is not a DirichletProcessMixture-
DataEncoder" to fall back to `self.encoder == other`, i.e. comparing the wrapper directly against
its *unwrapped* child encoder. Since the child's own __eq__ checks `isinstance(other, type(self))`
and the wrapper is not an instance of the child's type, `child == wrapper` is False while
`wrapper == child` was True -- asymmetric.
"""

import unittest

from mixle.stats.bayes.dirichlet_process_mixture import DirichletProcessMixtureDataEncoder
from mixle.stats.combinator.composite import CompositeDataEncoder
from mixle.stats.univariate.continuous.gaussian import GaussianDataEncoder
from mixle.stats.univariate.discrete.poisson import PoissonDataEncoder


class CompositeDataEncoderEqualityTestCase(unittest.TestCase):
    def test_equal_same_length(self):
        a = CompositeDataEncoder([GaussianDataEncoder(), PoissonDataEncoder()])
        b = CompositeDataEncoder([GaussianDataEncoder(), PoissonDataEncoder()])
        self.assertEqual(a, b)
        self.assertEqual(b, a)

    def test_different_length_is_not_equal_in_either_direction(self):
        short = CompositeDataEncoder([GaussianDataEncoder()])
        long = CompositeDataEncoder([GaussianDataEncoder(), PoissonDataEncoder()])
        # neither direction may raise IndexError, and neither may claim equality
        self.assertFalse(short == long)
        self.assertFalse(long == short)
        self.assertNotEqual(short, long)
        self.assertNotEqual(long, short)

    def test_different_component_types_same_length_is_not_equal(self):
        a = CompositeDataEncoder([GaussianDataEncoder(), PoissonDataEncoder()])
        b = CompositeDataEncoder([GaussianDataEncoder(), GaussianDataEncoder()])
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, a)

    def test_not_equal_to_unrelated_type(self):
        a = CompositeDataEncoder([GaussianDataEncoder()])
        self.assertNotEqual(a, GaussianDataEncoder())
        self.assertNotEqual(GaussianDataEncoder(), a)


class DirichletProcessMixtureDataEncoderEqualityTestCase(unittest.TestCase):
    def test_equal_same_child(self):
        a = DirichletProcessMixtureDataEncoder(GaussianDataEncoder())
        b = DirichletProcessMixtureDataEncoder(GaussianDataEncoder())
        self.assertEqual(a, b)
        self.assertEqual(b, a)

    def test_different_child_is_not_equal(self):
        a = DirichletProcessMixtureDataEncoder(GaussianDataEncoder())
        b = DirichletProcessMixtureDataEncoder(PoissonDataEncoder())
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, a)

    def test_wrapper_is_not_equal_to_its_unwrapped_child_in_either_direction(self):
        child = GaussianDataEncoder()
        wrapper = DirichletProcessMixtureDataEncoder(child)
        self.assertFalse(wrapper == child)
        self.assertFalse(child == wrapper)
        self.assertNotEqual(wrapper, child)
        self.assertNotEqual(child, wrapper)


if __name__ == "__main__":
    unittest.main()
