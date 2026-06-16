"""pysp.ppl: equality relations and soft-penalty constraints over random variables.

Extends the hard inequality/region constraints with:
  * equality / algebraic-equality relations built via ``eq(...)`` / ``rv.eq(...)`` (``==`` is not
    overloaded because RVs are identity-keyed), and
  * a soft-penalty inference mode ``fit(..., constraints=..., penalty=w)`` that adds a smooth
    ``-0.5*w*sum(residual^2)`` term to the joint log-target, so equality, convex, and algebraic
    relations are honored by gradient / MCMC inference (hard rejection cannot reach a measure-zero set).
"""

import unittest

import numpy as np

from pysp.ppl import Beta, Normal, constrain, eq, equal, ne


class ResidualMechanicsTestCase(unittest.TestCase):
    def test_equality_residual_is_signed_gap(self):
        a, b = Normal(0, 1, name="a"), Normal(0, 1, name="b")
        c = eq(a, b)
        self.assertAlmostEqual(float(np.asarray(c.residual({a: 2.0, b: 0.5}))), 1.5)
        self.assertAlmostEqual(float(np.asarray(c.residual({a: 1.0, b: 1.0}))), 0.0)

    def test_inequality_residual_is_a_hinge(self):
        a, b = Normal(0, 1, name="a"), Normal(0, 1, name="b")
        c = a < b
        self.assertAlmostEqual(float(np.atleast_1d(c.residual({a: 1.0, b: 0.0}))[0]), 1.0)  # violated by 1
        self.assertEqual(float(np.atleast_1d(c.residual({a: 0.0, b: 1.0}))[0]), 0.0)  # satisfied

    def test_and_stacks_residuals_or_takes_min(self):
        a, b, c = Normal(0, 1, name="a"), Normal(0, 1, name="b"), Normal(0, 1, name="c")
        both = eq(a, b) & eq(b, c)
        r = np.asarray(both.residual({a: 1.0, b: 0.0, c: -1.0}))
        self.assertEqual(r.size, 2)
        either = (a > 0) | (b > 0)
        # a satisfies (residual 0), so OR residual magnitude is 0 regardless of b.
        self.assertEqual(float(np.atleast_1d(either.residual({a: 1.0, b: -5.0}))[0]), 0.0)

    def test_negation_has_no_residual(self):
        a = Normal(0, 1, name="a")
        self.assertIsNone((~(a > 0)).residual)

    def test_rv_eq_and_ne_methods(self):
        a, b = Normal(0, 1, name="a"), Normal(0, 1, name="b")
        self.assertTrue(a.eq(b).pred({a: 1.0, b: 1.0}))
        self.assertTrue(a.ne(b).pred({a: 1.0, b: 2.0}))
        self.assertIs(equal, eq)


class SoftConstraintInferenceTestCase(unittest.TestCase):
    def setUp(self):
        self.data = list(np.random.RandomState(0).beta(3.0, 3.0, 2000))  # symmetric: MLE has a == b

    def _model(self):
        a = Normal(3, 5, name="alpha")
        b = Normal(3, 5, name="beta")
        return a, b

    def test_equality_map(self):
        a, b = self._model()
        fit = Beta(a, b).fit(self.data, how="map", constraints=eq(a, b), penalty=200.0)
        self.assertAlmostEqual(fit.params["a"], fit.params["b"], delta=0.1)

    def test_equality_ensemble(self):
        a, b = self._model()
        fit = Beta(a, b).fit(
            self.data,
            how="ensemble",
            constraints=eq(a, b),
            penalty=150.0,
            draws=600,
            burn=200,
            rng=np.random.RandomState(2),
        )
        self.assertAlmostEqual(fit.params["a"], fit.params["b"], delta=0.15)

    def test_equality_hmc(self):
        a, b = self._model()
        fit = Beta(a, b).fit(
            self.data, how="hmc", constraints=eq(a, b), penalty=150.0, draws=400, burn=200, rng=np.random.RandomState(4)
        )
        self.assertAlmostEqual(fit.params["a"], fit.params["b"], delta=0.2)

    def test_algebraic_equality_tightens_with_penalty(self):
        # alpha + beta == 8 fights the data (which prefers ~6); a larger penalty pulls the sum closer.
        a1, b1 = self._model()
        loose = Beta(a1, b1).fit(self.data, how="map", constraints=eq(a1 + b1, 8.0), penalty=20.0)
        a2, b2 = self._model()
        tight = Beta(a2, b2).fit(self.data, how="map", constraints=eq(a2 + b2, 8.0), penalty=5000.0)
        loose_sum = loose.params["a"] + loose.params["b"]
        tight_sum = tight.params["a"] + tight.params["b"]
        self.assertGreater(tight_sum, loose_sum)
        self.assertAlmostEqual(tight_sum, 8.0, delta=0.2)

    def test_soft_inequality_mcmc_runs(self):
        a, b = self._model()
        fit = Beta(a, b).fit(
            self.data, how="mcmc", constraints=(a < b), penalty=50.0, draws=800, burn=300, rng=np.random.RandomState(1)
        )
        self.assertTrue(np.isfinite(fit.params["a"]) and np.isfinite(fit.params["b"]))

    def test_negation_with_penalty_rejected(self):
        mu = Normal(0, 10, name="m")
        with self.assertRaises(ValueError):
            Normal(mu, 1).fit(self.data[:200], how="map", constraints=~(mu > 0), penalty=10.0)

    def test_penalty_on_non_prior_rejected(self):
        z = Normal(0, 1, name="z")
        a, b = self._model()
        with self.assertRaises(ValueError):
            Beta(a, b).fit(self.data, how="map", constraints=eq(z, 0.0), penalty=10.0)


class GenerativeEqualityTestCase(unittest.TestCase):
    def test_ne_constrains_generatively(self):
        # a discrete-ish use: ne is boolean-only but still composes as a Constraint
        a, b = Normal(0, 1, name="a"), Normal(0, 1, name="b")
        c = constrain(a < b)
        self.assertEqual(c.columns, ["a", "b"])
        self.assertIsNotNone(ne(a, b))


if __name__ == "__main__":
    unittest.main()
