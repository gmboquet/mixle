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

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.exp_family import ExponentialFamilyForm, is_exponential_family, to_exponential_family
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)


class IntegerCategoricalExponentialFamilyTest(unittest.TestCase):
    def test_reconstruction_positive_probs(self):
        for min_val, p in [(0, [0.2, 0.3, 0.5]), (-2, [0.1, 0.4, 0.25, 0.25]), (3, [0.6, 0.4])]:
            with self.subTest(min_val=repr(min_val), p=repr(p)):
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


class IntegerCategoricalDefaultValueTest(unittest.TestCase):
    """Out-of-support mass, so a support fitted from data can score an integer it never saw.

    The end-to-end case lives in flagship_heterogeneous_adult_smoke_test, but that one skips when the
    dataset host is unreachable, so the semantics are pinned here too.
    """

    def test_default_value_zero_is_exactly_the_historical_behaviour(self):
        d = IntegerCategoricalDistribution(2, [0.6, 0.0, 0.4])
        self.assertEqual(d.default_value, 0.0)
        for x in (9, -1, 3):  # outside the range, below it, and an interior zero
            self.assertEqual(d.log_density(x), -np.inf)
            self.assertEqual(d.density(x), 0.0)
        self.assertAlmostEqual(d.log_density(2), float(np.log(0.6)))
        np.testing.assert_allclose(d.seq_log_density(np.asarray([2, 3, 9])), [np.log(0.6), -np.inf, -np.inf])

    def test_unseen_integers_inside_and_outside_the_range_score_alike(self):
        d = IntegerCategoricalDistribution(2, [0.6, 0.0, 0.4], default_value=0.01)
        expected_out = float(np.log(0.01) - np.log1p(0.01))
        # An interior hole is stored as 0.0 only because the vector is dense; it is the same
        # "never observed" state CategoricalDistribution represents by an absent pmap key.
        self.assertAlmostEqual(d.log_density(3), expected_out)
        self.assertAlmostEqual(d.log_density(99), expected_out)
        self.assertAlmostEqual(d.log_density(2), float(np.log(0.6) - np.log1p(0.01)))
        self.assertEqual(d.log_density(2.5), -np.inf)  # a non-integer is still impossible

    def test_every_scoring_path_agrees(self):
        d = IntegerCategoricalDistribution(2, [0.6, 0.0, 0.4], default_value=0.05)
        xs = np.asarray([2, 3, 4, 9, -1])
        scalar = np.asarray([d.log_density(int(x)) for x in xs], dtype=np.float64)
        np.testing.assert_allclose(d.seq_log_density(xs), scalar)
        np.testing.assert_allclose(d.backend_seq_log_density(xs, NUMPY_ENGINE), scalar)
        stacked = IntegerCategoricalDistribution.backend_stacked_params([d, d], NUMPY_ENGINE)
        matrix = IntegerCategoricalDistribution.backend_stacked_log_density(xs, stacked, NUMPY_ENGINE)
        np.testing.assert_allclose(np.asarray(matrix)[:, 0], scalar)

    def test_stacking_rejects_mixed_default_values(self):
        a = IntegerCategoricalDistribution(2, [0.6, 0.4], default_value=0.01)
        b = IntegerCategoricalDistribution(2, [0.6, 0.4], default_value=0.20)
        with self.assertRaisesRegex(ValueError, "shared default_value"):
            IntegerCategoricalDistribution.backend_stacked_params([a, b], NUMPY_ENGINE)

    def test_invalid_default_value_is_rejected_not_clamped(self):
        for bad in (1.5, -0.1, float("nan")):
            with self.subTest(default_value=repr(bad)), self.assertRaisesRegex(ValueError, "default_value"):
                IntegerCategoricalDistribution(2, [0.5, 0.5], default_value=bad)

    def test_an_estimator_carries_default_value_onto_the_fit(self):
        est = IntegerCategoricalEstimator(min_val=0, max_val=3, default_value=0.02)
        fitted = est.estimate(None, (0, np.asarray([4.0, 0.0, 4.0, 2.0])))
        self.assertEqual(fitted.default_value, 0.02)
        self.assertGreater(fitted.log_density(1), -np.inf)  # the interior zero is now scorable
        self.assertGreater(fitted.log_density(7), -np.inf)  # and so is a value past max_val
