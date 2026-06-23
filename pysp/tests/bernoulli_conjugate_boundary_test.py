"""Regression test for the Bernoulli conjugate posterior-mode boundary.

A Beta(1, 1) prior with empty/zero sufficient statistics drove the unguarded
posterior-mode formula ``(psum + a - 1) / (psum + nsum + a + b - 2)`` to a 0/0
division. The conjugate path must fall back to the posterior mean on that
boundary, matching the Geometric/Binomial conjugate families.
"""

import unittest

from pysp.stats.univariate.continuous.beta import BetaDistribution
from pysp.stats.univariate.discrete.bernoulli import BernoulliEstimator


class TestBernoulliConjugateBoundary(unittest.TestCase):
    def test_empty_beta11_does_not_divide_by_zero(self) -> None:
        est = BernoulliEstimator(prior=BetaDistribution(1.0, 1.0))
        # Empty statistics: count = psum = 0 -> mode denominator is 0.
        model = est.estimate(None, (0.0, 0.0))
        # Falls back to the posterior mean a' / (a' + b') = 1 / (1 + 1) = 0.5.
        self.assertAlmostEqual(model.p, 0.5)
        new_a, new_b = model.prior.get_parameters()
        self.assertEqual((new_a, new_b), (1.0, 1.0))

    def test_interior_still_uses_posterior_mode(self) -> None:
        # With Beta(2, 2) and 4 successes / 1 failure, posterior is Beta(6, 3),
        # mode = (psum + a - 1) / (psum + nsum + a + b - 2) = (4 + 1) / (5 + 2) = 5/7.
        est = BernoulliEstimator(prior=BetaDistribution(2.0, 2.0))
        model = est.estimate(None, (5.0, 4.0))
        self.assertAlmostEqual(model.p, 5.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
