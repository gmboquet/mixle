"""Tests for IntegerMultinomialDistribution's log_density/seq_log_density agreement, in particular
with a real len_dist (trial-count distribution) set. log_density previously omitted the len_dist term
entirely -- seq_log_density and the sibling MultinomialDistribution.log_density both include it -- so
the two disagreed by exactly the trial-count term whenever len_dist was not the default NullDistribution.
"""

import unittest

import numpy as np

from mixle.stats.multivariate.integer_multinomial import IntegerMultinomialDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

DATA = [
    [(0, 2.0), (2, 1.0)],
    [(1, 3.0)],
    [(0, 1.0), (1, 1.0), (2, 1.0)],
    [(0, 5.0)],
]


class IntegerMultinomialLenDistTestCase(unittest.TestCase):
    def test_log_density_includes_len_dist_term(self):
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5], len_dist=PoissonDistribution(3.0))
        for x in DATA:
            with self.subTest(x=x):
                total = sum(cnt for _, cnt in x)
                category_term = sum(d.log_p_vec[v] * cnt for v, cnt in x)
                expected = category_term + PoissonDistribution(3.0).log_density(total)
                self.assertAlmostEqual(d.log_density(x), expected, places=10)

    def test_log_density_matches_seq_log_density_with_len_dist(self):
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5], len_dist=PoissonDistribution(3.0))
        enc = d.dist_to_encoder().seq_encode(DATA)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(x) for x in DATA])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_log_density_matches_seq_log_density_without_len_dist(self):
        # sanity: the no-len_dist (default NullDistribution) path is unaffected by the fix.
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5])
        enc = d.dist_to_encoder().seq_encode(DATA)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(x) for x in DATA])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_len_dist_term_is_nonzero_when_set(self):
        # guards against a fix that adds the term but has it silently evaluate to 0 (e.g. wrong count).
        d_null = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5])
        d_poisson = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5], len_dist=PoissonDistribution(3.0))
        x = DATA[0]
        self.assertNotAlmostEqual(d_null.log_density(x), d_poisson.log_density(x), places=6)


if __name__ == "__main__":
    unittest.main()
