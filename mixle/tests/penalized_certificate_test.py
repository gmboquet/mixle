"""E2: certificates downgrade honestly under penalized objectives (soft constraints / residual factors)."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import Guarantee, certify, optimize
from mixle.ppl import Beta, Normal


class PenalizedCertifyTest(unittest.TestCase):
    def _gaussian(self):
        return optimize(
            [float(x) for x in np.random.RandomState(0).normal(5, 2, 100)], st.GaussianEstimator(), out=None
        )

    def test_penalized_caps_global_unique_at_stationary(self):
        g = self._gaussian()
        self.assertEqual(certify(g).guarantee, Guarantee.GLOBAL_UNIQUE)
        pen = certify(g, penalized="PINN residual")
        self.assertEqual(pen.guarantee, Guarantee.STATIONARY)  # no block may claim more than STATIONARY
        self.assertIn("PINN residual", pen.blocks[0].reason)  # the penalty is NAMED in the downgrade
        self.assertIn("surrogate, not the likelihood", pen.blocks[0].reason)

    def test_penalized_true_uses_the_generic_reason(self):
        pen = certify(self._gaussian(), penalized=True)
        self.assertEqual(pen.guarantee, Guarantee.STATIONARY)
        self.assertIn("soft-constraint / residual penalty", pen.blocks[0].reason)

    def test_unpenalized_is_unchanged(self):
        self.assertEqual(certify(self._gaussian()).guarantee, Guarantee.GLOBAL_UNIQUE)


class ConstrainedPplFitTest(unittest.TestCase):
    def _fit(self, **kw):
        a = Normal(2, 5, name="alpha")
        b = Normal(5, 5, name="beta")
        data = list(np.random.RandomState(0).beta(2, 5, 500))
        return Beta(a, b).fit(data, how="map", **({"constraints": a < b} | kw))

    def test_constrained_fit_attaches_a_downgraded_certificate(self):
        fit = self._fit()
        self.assertIsNotNone(fit.certificate)
        self.assertEqual(fit.certificate.guarantee, Guarantee.STATIONARY)
        self.assertIn("DOWNGRADED", fit.certificate.blocks[0].reason)
        self.assertIn("soft constraints", fit.certificate.blocks[0].reason)

    def test_unconstrained_fit_attaches_nothing(self):
        a = Normal(2, 5, name="alpha")
        b = Normal(5, 5, name="beta")
        data = list(np.random.RandomState(0).beta(2, 5, 500))
        fit = Beta(a, b).fit(data, how="map")
        self.assertIsNone(fit.certificate)  # no penalty -> no claim attached, nothing fabricated


if __name__ == "__main__":
    unittest.main()
