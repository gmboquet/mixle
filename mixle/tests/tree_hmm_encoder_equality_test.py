"""TreeHiddenMarkovDataEncoder.__eq__ must consider emission_encoder and use_numba, not just
len_encoder, matching its sibling HiddenMarkovDataEncoder (mixle.stats.latent.hidden_markov).

Comparing len_encoder alone silently treated any two encoders with the same length model as equal
regardless of what they emit -- the heterogeneous grouping machinery (e.g. a mixture over tree HMMs
with unlike emission families) keys off encoder equality, so this made unlike encoders
indistinguishable. The old shape also returned an implicit None (not False) for a same-type encoder
with a different len_encoder, since the `if .. == ..: return True` had no else on that branch.
"""

import unittest

from mixle.stats import GaussianDistribution, PoissonDistribution
from mixle.stats.combinator.null_dist import NullDataEncoder
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovDataEncoder


class TreeHmmEncoderEqualityTest(unittest.TestCase):
    def setUp(self):
        self.gaussian_enc = GaussianDistribution(0.0, 1.0).dist_to_encoder()
        self.poisson_enc = PoissonDistribution(1.0).dist_to_encoder()

    def test_equal_when_every_field_matches(self):
        a = TreeHiddenMarkovDataEncoder(self.gaussian_enc, use_numba=False)
        b = TreeHiddenMarkovDataEncoder(self.gaussian_enc, use_numba=False)
        self.assertEqual(a, b)

    def test_unequal_emission_encoder_makes_them_unequal(self):
        a = TreeHiddenMarkovDataEncoder(self.gaussian_enc, use_numba=False)
        b = TreeHiddenMarkovDataEncoder(self.poisson_enc, use_numba=False)
        self.assertNotEqual(a, b)
        self.assertIs(a == b, False)  # a definite False, never an implicit None

    def test_unequal_use_numba_makes_them_unequal(self):
        a = TreeHiddenMarkovDataEncoder(self.gaussian_enc, use_numba=False)
        b = TreeHiddenMarkovDataEncoder(self.gaussian_enc, use_numba=True)
        self.assertNotEqual(a, b)

    def test_unequal_len_encoder_makes_them_unequal_and_returns_false_not_none(self):
        a = TreeHiddenMarkovDataEncoder(self.gaussian_enc, len_encoder=NullDataEncoder(), use_numba=False)
        b = TreeHiddenMarkovDataEncoder(
            self.gaussian_enc, len_encoder=PoissonDistribution(1.0).dist_to_encoder(), use_numba=False
        )
        result = a == b
        self.assertIs(result, False)  # the old code fell off the end of __eq__ here, returning None

    def test_not_equal_to_a_non_encoder(self):
        a = TreeHiddenMarkovDataEncoder(self.gaussian_enc, use_numba=False)
        self.assertIs(a == 5, False)
        self.assertIs(a == None, False)  # noqa: E711 - exercising __eq__, not identity


if __name__ == "__main__":
    unittest.main()
