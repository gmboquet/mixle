"""Exponential-family canonical-map test for Categorical (WS-J).

Categorical over hashable labels is an exponential family with one-hot sufficient statistic ``T(x)``
(categories in canonical ``sorted(pmap, key=repr)`` order), natural parameter ``eta = log(pmap)``,
``A = 0`` and base ``h(x) = 0`` on the support (the keys of ``pmap``). Because ``eta`` has ``-inf``
entries when a label has ``p = 0``, the generic ``<eta, T>`` dot form is NaN-prone (``0 * -inf``) — so
the spec sets ``runtime_scoring=False``: scoring keeps its safe dict-indexing backend path while
``to_exponential_family`` still exposes the canonical map. These tests pin both the reconstruction
(plain ``default_value=0`` categorical) and that runtime scoring is unaffected by a zero-probability
label. Standalone to avoid the shared ``exp_family_test`` catalog.
"""

import unittest

import numpy as np

from pysp.engines import NUMPY_ENGINE
from pysp.stats.base.categorical import CategoricalDistribution
from pysp.stats.exp_family import ExponentialFamilyForm, is_exponential_family, to_exponential_family


class CategoricalExponentialFamilyTest(unittest.TestCase):
    def test_reconstruction_positive_probs(self):
        for pmap in (
            {"a": 0.2, "b": 0.3, "c": 0.5},
            {0: 0.1, 1: 0.4, 2: 0.25, 3: 0.25},
            {"x": 0.6, "y": 0.4},
        ):
            with self.subTest(pmap=pmap):
                d = CategoricalDistribution(pmap)
                self.assertTrue(is_exponential_family(d))
                form = to_exponential_family(d)
                self.assertIsInstance(form, ExponentialFamilyForm)

                keys = sorted(pmap.keys(), key=repr)
                x = keys * 3
                eta = form.natural_parameters()
                t = np.asarray(form.sufficient_statistics(x), dtype=np.float64)
                a = float(form.log_partition())
                h = np.asarray(form.log_base_measure(x), dtype=np.float64)
                self.assertEqual(t.shape[1], eta.shape[0])
                self.assertEqual(form.dim, len(pmap))

                recon = h + t @ eta - a
                ref = np.asarray(d.seq_log_density(d.dist_to_encoder().seq_encode(x)), dtype=np.float64)
                np.testing.assert_allclose(recon, ref, atol=1e-9)

    def test_canonical_order_matches_natural_parameters(self):
        # T columns and eta must share the sorted-by-repr key order: eta_j == log p of the j-th key.
        pmap = {"banana": 0.5, "apple": 0.3, "cherry": 0.2}
        form = to_exponential_family(CategoricalDistribution(pmap))
        keys = sorted(pmap.keys(), key=repr)
        np.testing.assert_allclose(form.natural_parameters(), np.log([pmap[k] for k in keys]), atol=1e-12)

    def test_out_of_support_is_neg_inf(self):
        d = CategoricalDistribution({"a": 0.3, "b": 0.7})
        form = to_exponential_family(d)
        x = ["a", "b", "z", "q"]  # "z","q" are off support
        recon = np.asarray(form.log_density(x), dtype=np.float64)
        self.assertTrue(np.isneginf(recon[2]) and np.isneginf(recon[3]))
        self.assertTrue(np.all(np.isfinite(recon[:2])))

    def test_runtime_scoring_safe_with_zero_prob_label(self):
        """The dist's own scoring (dict indexing) stays finite with a zero-prob label present.

        ``runtime_scoring=False`` keeps scoring on the dict-indexing path rather than the canonical
        ``<eta, T>`` dot form, whose ``eta = log(p)`` has ``-inf`` for the ``p = 0`` label and would
        yield ``0 * -inf = NaN`` for observations of other labels.
        """
        d = CategoricalDistribution({"a": 0.5, "b": 0.0, "c": 0.5})  # label "b" has p=0
        enc = d.dist_to_encoder().seq_encode(["a", "c", "a", "c"])  # observations of other labels
        scored = np.asarray(d.seq_log_density(enc), dtype=np.float64)
        backend = np.asarray(d.backend_seq_log_density(enc, NUMPY_ENGINE), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(scored)), "indexing scoring produced NaN/inf for p>0 observations")
        np.testing.assert_allclose(scored, backend, atol=1e-12)

        # The canonical-map dot form, by contrast, is NaN here for the same observations — which is
        # exactly why runtime_scoring is False (the canonical map is only used where p > 0). The
        # log(0) and 0*-inf below are the very pathology being demonstrated, so silence the warnings.
        with np.errstate(divide="ignore", invalid="ignore"):
            dot = np.asarray(to_exponential_family(d).log_density(["a", "c", "a", "c"]), dtype=np.float64)
        self.assertTrue(np.any(np.isnan(dot)))


if __name__ == "__main__":
    unittest.main()
