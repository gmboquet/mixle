"""Exponential-family canonical-map test for IntegerCategorical (WS-J).

IntegerCategorical is an exponential family with one-hot sufficient statistic ``T(x)``,
natural parameter ``eta = log(p_vec)``, ``A = 0`` and base ``h(x) = 1`` on the support. Because
``eta`` has ``-inf`` entries when a category has ``p = 0``, the generic ``<eta, T>`` dot form is
NaN-prone (``0 * -inf``) — so the spec sets ``runtime_scoring=False``: scoring keeps its safe
indexing backend path while ``to_exponential_family`` still exposes the canonical map. These tests
pin both the reconstruction (where ``p > 0``) and that runtime scoring is unaffected (no NaN even
with a zero-probability category). Standalone to avoid the shared ``exp_family_test`` catalog.
"""

import unittest

import numpy as np

from pysp.engines import NUMPY_ENGINE
from pysp.stats.exp_family import ExponentialFamilyForm, is_exponential_family, to_exponential_family
from pysp.stats.leaf.int_range import IntegerCategoricalDistribution


class IntegerCategoricalExponentialFamilyTest(unittest.TestCase):
    def test_reconstruction_positive_probs(self):
        for min_val, p in [(0, [0.2, 0.3, 0.5]), (-2, [0.1, 0.4, 0.25, 0.25]), (3, [0.6, 0.4])]:
            with self.subTest(min_val=min_val, p=p):
                d = IntegerCategoricalDistribution(p_vec=p, min_val=min_val)
                self.assertTrue(is_exponential_family(d))
                form = to_exponential_family(d)
                self.assertIsInstance(form, ExponentialFamilyForm)

                x = [min_val + i for i in range(len(p))] * 3
                eta = form.natural_parameters()
                t = np.asarray(form.sufficient_statistics(x), dtype=np.float64)
                a = float(form.log_partition())
                h = np.asarray(form.log_base_measure(x), dtype=np.float64)
                self.assertEqual(t.shape[1], eta.shape[0])
                self.assertEqual(form.dim, len(p))

                recon = h + t @ eta - a
                ref = np.asarray(d.seq_log_density(d.dist_to_encoder().seq_encode(x)), dtype=np.float64)
                np.testing.assert_allclose(recon, ref, atol=1e-9)

    def test_out_of_support_is_neg_inf(self):
        d = IntegerCategoricalDistribution(p_vec=[0.3, 0.7], min_val=0)
        form = to_exponential_family(d)
        x = [0, 1, 5, -1]  # 5 and -1 are off support
        recon = np.asarray(form.log_density(x), dtype=np.float64)
        self.assertTrue(np.isneginf(recon[2]) and np.isneginf(recon[3]))
        self.assertTrue(np.all(np.isfinite(recon[:2])))

    def test_runtime_scoring_safe_with_zero_prob_category(self):
        """The dist's own scoring (indexing) stays finite with a zero-prob category present.

        ``runtime_scoring=False`` keeps scoring on this indexing path rather than the canonical
        ``<eta, T>`` dot form, whose ``eta = log(p)`` has ``-inf`` for the ``p = 0`` category and
        would yield ``0 * -inf = NaN`` for observations of other categories.
        """
        d = IntegerCategoricalDistribution(p_vec=[0.5, 0.0, 0.5], min_val=0)  # category 1 has p=0
        enc = d.dist_to_encoder().seq_encode([0, 2, 0, 2])  # observations of other categories
        scored = np.asarray(d.seq_log_density(enc), dtype=np.float64)
        backend = np.asarray(d.backend_seq_log_density(enc, NUMPY_ENGINE), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(scored)), "indexing scoring produced NaN/inf for p>0 observations")
        np.testing.assert_allclose(scored, backend, atol=1e-12)

        # The canonical-map dot form, by contrast, is NaN here for the same observations — which is
        # exactly why runtime_scoring is False (the canonical map is only used where p > 0).
        dot = np.asarray(to_exponential_family(d).log_density([0, 2, 0, 2]), dtype=np.float64)
        self.assertTrue(np.any(np.isnan(dot)))


if __name__ == "__main__":
    unittest.main()
