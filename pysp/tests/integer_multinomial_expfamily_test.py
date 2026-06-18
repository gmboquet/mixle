"""Exponential-family canonical-map test for IntegerMultinomial (WS-J).

IntegerMultinomial (with no separate length/trials distribution) is an exponential family with the
per-category count vector ``T(x)``, natural parameter ``eta = log(p_vec)``, ``A = 0`` and base
``h(x) = 0`` on the support ``[min_val, min_val+K)``. This density omits the multinomial coefficient,
so ``log p(x) = sum_k count_k * log p_k``. Because ``eta`` has ``-inf`` entries when a category has
``p = 0``, the generic ``<eta, T>`` dot form is NaN-prone (``0 * -inf``) for zero-count categories —
so the spec sets ``runtime_scoring=False``: scoring keeps its safe indexing backend path while
``to_exponential_family`` still exposes the canonical map. Standalone to avoid the shared
``exp_family_test`` catalog.
"""

import unittest

import numpy as np

from pysp.engines import NUMPY_ENGINE
from pysp.stats.exp_family import ExponentialFamilyForm, is_exponential_family, to_exponential_family
from pysp.stats.leaf.int_multinomial import IntegerMultinomialDistribution


class IntegerMultinomialExponentialFamilyTest(unittest.TestCase):
    def test_reconstruction_positive_probs(self):
        cases = [
            (0, [0.2, 0.3, 0.5], [[(0, 2.0), (2, 1.0)], [(1, 3.0)], [(0, 1.0), (1, 1.0), (2, 1.0)]]),
            (-2, [0.1, 0.4, 0.25, 0.25], [[(-2, 1.0), (1, 2.0)], [(0, 4.0)], [(-1, 2.0), (1, 1.0)]]),
            (3, [0.6, 0.4], [[(3, 5.0)], [(4, 2.0), (3, 1.0)]]),
        ]
        for min_val, p, x in cases:
            with self.subTest(min_val=min_val, p=p):
                d = IntegerMultinomialDistribution(min_val=min_val, p_vec=p)  # len_dist defaults to Null
                self.assertTrue(is_exponential_family(d))
                form = to_exponential_family(d)
                self.assertIsInstance(form, ExponentialFamilyForm)

                eta = form.natural_parameters()
                t = np.asarray(form.sufficient_statistics(x), dtype=np.float64)
                a = float(form.log_partition())
                h = np.asarray(form.log_base_measure(x), dtype=np.float64)
                self.assertEqual(t.shape, (len(x), len(p)))
                self.assertEqual(t.shape[1], eta.shape[0])
                self.assertEqual(form.dim, len(p))

                recon = h + t @ eta - a
                ref = np.asarray(d.seq_log_density(d.dist_to_encoder().seq_encode(x)), dtype=np.float64)
                np.testing.assert_allclose(recon, ref, atol=1e-9)

    def test_sufficient_statistic_is_the_count_vector(self):
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.5, 0.3, 0.2])
        form = to_exponential_family(d)
        # Two of category 0 and three of category 2 -> count vector [2, 0, 3].
        t = np.asarray(form.sufficient_statistics([[(0, 2.0), (2, 3.0)]]), dtype=np.float64)
        np.testing.assert_array_equal(t, [[2.0, 0.0, 3.0]])

    def test_out_of_support_is_neg_inf(self):
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.3, 0.7])
        form = to_exponential_family(d)
        x = [[(0, 1.0), (1, 2.0)], [(5, 1.0)]]  # value 5 is off support
        recon = np.asarray(form.log_density(x), dtype=np.float64)
        self.assertTrue(np.isfinite(recon[0]))
        self.assertTrue(np.isneginf(recon[1]))

    def test_not_exp_family_with_a_length_distribution(self):
        # A separate trials distribution means the canonical multinomial map is not the full density.
        from pysp.stats.leaf.poisson import PoissonDistribution

        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.5, 0.5], len_dist=PoissonDistribution(3.0))
        self.assertFalse(is_exponential_family(d))

    def test_runtime_scoring_safe_with_zero_prob_category(self):
        """The dist's own indexing scoring stays finite with a zero-prob category present.

        ``runtime_scoring=False`` keeps scoring on the indexing path rather than the canonical
        ``<eta, T>`` dot form, whose ``eta = log(p)`` has ``-inf`` for the ``p = 0`` category and
        would yield ``0 * -inf = NaN`` for the zero-count entry of other observations.
        """
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.5, 0.0, 0.5])  # category 1 has p=0
        enc = d.dist_to_encoder().seq_encode([[(0, 2.0), (2, 1.0)], [(2, 3.0)]])  # no category-1 counts
        scored = np.asarray(d.seq_log_density(enc), dtype=np.float64)
        backend = np.asarray(d.backend_seq_log_density(enc, NUMPY_ENGINE), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(scored)), "indexing scoring produced NaN/inf for p>0 observations")
        np.testing.assert_allclose(scored, backend, atol=1e-12)

        # The canonical-map dot form is NaN here for the zero-count p=0 category (0 * -inf), which is
        # exactly why runtime_scoring is False (the canonical map is only used where p > 0).
        with np.errstate(divide="ignore", invalid="ignore"):
            dot = np.asarray(to_exponential_family(d).log_density([[(0, 2.0), (2, 1.0)], [(2, 3.0)]]), dtype=np.float64)
        self.assertTrue(np.any(np.isnan(dot)))


if __name__ == "__main__":
    unittest.main()
